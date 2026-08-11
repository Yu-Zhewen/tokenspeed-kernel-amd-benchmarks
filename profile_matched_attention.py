#!/usr/bin/env python3
"""Run identical production-shaped attention cases on gfx950 and gfx1250."""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
import math
import runpy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


PAGE_SIZE = 64
KIMI_HEADS = 12
KIMI_QK_DIM = 192
KIMI_VALUE_DIM = 128
KV_LORA_RANK = 512
ROPE_DIM = 64
DSA_INDEX_HEADS = 32
DSA_INDEX_HEAD_DIM = 128
DSA_ATTN_HEADS = 8
DSA_QK_NOPE_DIM = 192
TOPK = 2048


CASE_SPECS: dict[str, dict[str, Any]] = {
    "mla-decode": {
        "model": "Kimi-K3",
        "phase": "decode",
        "shape": "B1, context=4096, q_len=1, heads=12, rank=512, rope=64",
        "dtypes": "FP8 absorbed Q/KV; BF16 value projection, gate, and output",
        "layout": "dense paged KV, page_size=64, projected-value/gate epilogue",
        "long_running": False,
    },
    "mla-prefill": {
        "model": "Kimi-K3",
        "phase": "prefill",
        "shape": "B1, prefix=0, extend=4096, heads=12, QK=192, V=128",
        "dtypes": "FP8 Q/K/V",
        "layout": "packed varlen, causal",
        "long_running": False,
    },
    "kda-prefill": {
        "model": "Kimi-K3",
        "phase": "prefill",
        "shape": "B1, prefix=0, extend=4096, heads=12, K/V=128",
        "dtypes": "BF16 Q/K/V/gates, FP32 parameters and recurrent state",
        "layout": "packed sequence, 64-token implementation chunks",
        "long_running": False,
    },
    "dsa-decode-pipeline": {
        "model": "GLM-5.2",
        "phase": "decode top-k plus selected attention",
        "shape": (
            "B1, context=4096, q_len=1, index_heads=32, index_dim=128, "
            "attention_heads=8, topk=2048, rank=512, rope=64"
        ),
        "dtypes": "BF16 index Q, packed FP8 index-K, FP8 attention Q/KV",
        "layout": "production dense-KV selected-attention path",
        "long_running": False,
    },
    "dsa-prefill-pipeline-4k": {
        "model": "GLM-5.2",
        "phase": "pure prefill top-k plus selected attention",
        "shape": (
            "B1, prefix=0, extend=4096, index_heads=32, index_dim=128, "
            "attention_heads=8, causal topk<=2048, rank=512, rope=64"
        ),
        "dtypes": "BF16 index Q, packed FP8 index-K, FP8 attention Q/KV",
        "layout": "production dense-KV selected-attention path",
        "long_running": True,
    },
}


@dataclass(frozen=True)
class Workload:
    run: Callable[[], object]
    details: dict[str, Any]


def _load_kernel_apis() -> None:
    global dsa_decode
    global dsa_decode_topk
    global dsa_prefill
    global dsa_prefill_topk
    global kda_paged_prefill
    global mla_decode_with_kvcache
    global mla_prefill

    kernel_spec = importlib.util.find_spec("tokenspeed_kernel")
    if kernel_spec is None or kernel_spec.origin is None:
        raise ImportError(
            "tokenspeed_kernel is not importable; install it or add "
            "tokenspeed-kernel/python to PYTHONPATH"
        )
    triton_shim_path = Path(kernel_spec.origin).parent / "_triton.py"
    if triton_shim_path.is_file():
        triton_shim = runpy.run_path(str(triton_shim_path))
        triton_redirect = triton_shim["redirect_triton_to_tokenspeed_triton"]()
        triton_redirect.__enter__()
        atexit.register(triton_redirect.__exit__, None, None, None)

    from tokenspeed_kernel import (
        dsa_decode,
        dsa_decode_topk,
        dsa_prefill,
        dsa_prefill_topk,
        mla_decode_with_kvcache,
        mla_prefill,
    )
    from tokenspeed_kernel.ops.attention import kda_paged_prefill


def _arch() -> str:
    name = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if name not in ("gfx950", "gfx1250"):
        raise RuntimeError(f"expected gfx950 or gfx1250, got {name}")
    return name


def _gluon_dsa_override(operator: str, arch: str) -> str:
    return f"gluon_dsa_{operator}_{arch}"


def _time_us(fn: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) * 1000.0 / repeats


def _pack_index_k(index_k: torch.Tensor) -> torch.Tensor:
    scale = index_k.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 448.0
    fp8 = (index_k.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    num_slots = index_k.shape[0]
    num_pages = num_slots // PAGE_SIZE
    row_bytes = DSA_INDEX_HEAD_DIM + 4
    packed = torch.empty(
        (num_slots, row_bytes), dtype=torch.uint8, device=index_k.device
    )
    flat = packed.reshape(-1)
    page_bytes = PAGE_SIZE * row_bytes
    fp8_view = torch.as_strided(
        flat.view(torch.float8_e4m3fn),
        (num_pages, PAGE_SIZE, DSA_INDEX_HEAD_DIM),
        (page_bytes, DSA_INDEX_HEAD_DIM, 1),
    )
    scale_view = torch.as_strided(
        flat.view(torch.float32),
        (num_pages, PAGE_SIZE, 1),
        (page_bytes // 4, 1, 1),
        (PAGE_SIZE * DSA_INDEX_HEAD_DIM) // 4,
    )
    fp8_view.copy_(fp8.reshape(num_pages, PAGE_SIZE, DSA_INDEX_HEAD_DIM))
    scale_view.copy_(scale.reshape(num_pages, PAGE_SIZE, 1))
    return packed


def _mla_decode() -> Workload:
    batch, context = 1, 4096
    qk_dim = KV_LORA_RANK + ROPE_DIM
    pages = math.ceil(context / PAGE_SIZE)
    q = torch.full(
        (batch, 1, KIMI_HEADS, qk_dim),
        0.25,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    kv_cache = torch.zeros(
        (pages, PAGE_SIZE, 1, qk_dim),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    page_table = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, -1)
    cache_seqlens = torch.tensor([context], dtype=torch.int32, device="cuda")
    value_weight = torch.full(
        (KIMI_HEADS, KV_LORA_RANK, KIMI_VALUE_DIM),
        0.015625,
        dtype=torch.bfloat16,
        device="cuda",
    )
    gate = torch.zeros(
        (batch, KIMI_HEADS * KIMI_VALUE_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    out = torch.empty_like(gate)

    def run() -> object:
        return mla_decode_with_kvcache(
            q=q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=context,
            qk_nope_head_dim=128,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=ROPE_DIM,
            softmax_scale=1.0 / math.sqrt(KIMI_QK_DIM),
            out=out,
            value_weight=value_weight,
            gate=gate,
        )

    return Workload(
        run,
        {
            "max_seqlen_k": context,
            "projected_value": True,
            "gate": "sigmoid",
        },
    )


def _mla_prefill() -> Workload:
    context = 4096
    q = torch.full(
        (context, KIMI_HEADS, KIMI_QK_DIM),
        0.125,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    k = torch.full_like(q, 0.0625)
    v = torch.full(
        (context, KIMI_HEADS, KIMI_VALUE_DIM),
        0.25,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    cu_seqlens = torch.tensor([0, context], dtype=torch.int32, device="cuda")

    def run() -> object:
        return mla_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=context,
            max_seqlen_kv=context,
            softmax_scale=1.0 / math.sqrt(KIMI_QK_DIM),
            is_causal=True,
            solution="gluon",
        )

    return Workload(run, {})


def _kda_prefill() -> Workload:
    context = 4096
    shape = (1, context, KIMI_HEADS, KIMI_VALUE_DIM)
    q = torch.full(shape, 0.125, dtype=torch.bfloat16, device="cuda")
    k = torch.full_like(q, 0.0625)
    v = torch.full_like(q, 0.25)
    g_raw = torch.zeros_like(q)
    beta_logits = torch.zeros(
        (1, context, KIMI_HEADS), dtype=torch.bfloat16, device="cuda"
    )
    a_log = torch.zeros(KIMI_HEADS, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(
        (KIMI_HEADS, KIMI_VALUE_DIM), dtype=torch.float32, device="cuda"
    )
    initial_state = torch.zeros(
        (1, KIMI_HEADS, KIMI_VALUE_DIM, KIMI_VALUE_DIM),
        dtype=torch.float32,
        device="cuda",
    )
    cu_seqlens = torch.tensor([0, context], dtype=torch.int32, device="cuda")

    def run() -> object:
        return kda_paged_prefill(
            q=q,
            k=k,
            v=v,
            g_raw=g_raw,
            beta_logits=beta_logits,
            A_log=a_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            lower_bound=-5.0,
            solution="gluon",
        )

    return Workload(run, {})


def _dsa_common() -> dict[str, torch.Tensor]:
    context = 4096
    torch.manual_seed(42)
    index_k = torch.randn(
        context, DSA_INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    packed_index_k = _pack_index_k(index_k)
    dense_kv = torch.randn(
        context,
        KV_LORA_RANK + ROPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    decode_index_q = torch.randn(
        1,
        DSA_INDEX_HEADS,
        DSA_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    decode_weights = torch.randn(
        1, DSA_INDEX_HEADS, dtype=torch.float32, device="cuda"
    )
    decode_q = torch.randn(
        1,
        DSA_ATTN_HEADS,
        KV_LORA_RANK + ROPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    pages = context // PAGE_SIZE
    block_table = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, -1)
    seq_lens = torch.tensor([context], dtype=torch.int32, device="cuda")
    return {
        "packed_index_k": packed_index_k,
        "dense_kv": dense_kv,
        "decode_index_q": decode_index_q,
        "decode_weights": decode_weights,
        "decode_q": decode_q,
        "block_table": block_table,
        "seq_lens": seq_lens,
    }


def _dsa_decode_pipeline(arch: str) -> Workload:
    x = _dsa_common()

    def run() -> object:
        slots, lens = dsa_decode_topk(
            x["decode_index_q"],
            x["decode_weights"],
            x["seq_lens"],
            x["block_table"],
            page_size=PAGE_SIZE,
            topk=TOPK,
            softmax_scale=DSA_INDEX_HEAD_DIM**-0.5,
            q_len_per_req=1,
            index_k_cache=x["packed_index_k"],
            solution="gluon",
        )
        return dsa_decode(
            q=x["decode_q"],
            kv_cache=x["dense_kv"],
            sparse_kv_cache=None,
            topk_slots=slots,
            topk_lens=lens,
            max_seqlen_k=4096,
            qk_nope_head_dim=DSA_QK_NOPE_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=ROPE_DIM,
            softmax_scale=1.0 / math.sqrt(DSA_QK_NOPE_DIM + ROPE_DIM),
            page_size=PAGE_SIZE,
            q_len_per_req=1,
            override=_gluon_dsa_override("decode", arch),
        )

    return Workload(run, {"selection": "live output of dsa_decode_topk"})


def _dsa_prefill_common() -> dict[str, torch.Tensor]:
    context = 4096
    torch.manual_seed(42)
    index_k = torch.randn(
        context, DSA_INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    packed_index_k = _pack_index_k(index_k)
    dense_kv = torch.randn(
        context,
        KV_LORA_RANK + ROPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    index_q = torch.randn(
        context,
        DSA_INDEX_HEADS,
        DSA_INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    weights = torch.randn(
        context, DSA_INDEX_HEADS, dtype=torch.float32, device="cuda"
    )
    attention_q = torch.randn(
        context,
        DSA_ATTN_HEADS,
        KV_LORA_RANK + ROPE_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    workspace_slots = torch.arange(context, dtype=torch.int64, device="cuda")
    row_starts = torch.zeros(context, dtype=torch.int32, device="cuda")
    row_ends = torch.arange(1, context + 1, dtype=torch.int32, device="cuda")
    return {
        "packed_index_k": packed_index_k,
        "dense_kv": dense_kv,
        "index_q": index_q,
        "weights": weights,
        "attention_q": attention_q,
        "workspace_slots": workspace_slots,
        "row_starts": row_starts,
        "row_ends": row_ends,
    }


def _dsa_prefill_pipeline() -> Workload:
    x = _dsa_prefill_common()

    def run() -> object:
        slots, lens = dsa_prefill_topk(
            x["index_q"],
            x["weights"],
            x["workspace_slots"],
            x["row_starts"],
            x["row_ends"],
            page_size=PAGE_SIZE,
            topk=TOPK,
            softmax_scale=DSA_INDEX_HEAD_DIM**-0.5,
            index_k_cache=x["packed_index_k"],
            solution="gluon",
        )
        return dsa_prefill(
            q=x["attention_q"],
            kv_cache=x["dense_kv"],
            sparse_kv_cache=None,
            topk_slots=slots,
            topk_lens=lens,
            max_seqlen_k=4096,
            qk_nope_head_dim=DSA_QK_NOPE_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=ROPE_DIM,
            softmax_scale=1.0 / math.sqrt(DSA_QK_NOPE_DIM + ROPE_DIM),
            page_size=PAGE_SIZE,
        )

    return Workload(run, {"selection": "live output of dsa_prefill_topk"})


def _build(name: str, arch: str) -> Workload:
    builders: dict[str, Callable[[], Workload]] = {
        "mla-decode": _mla_decode,
        "mla-prefill": _mla_prefill,
        "kda-prefill": _kda_prefill,
        "dsa-decode-pipeline": lambda: _dsa_decode_pipeline(arch),
        "dsa-prefill-pipeline-4k": _dsa_prefill_pipeline,
    }
    return builders[name]()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched gfx950/gfx1250 attention benchmark cases"
    )
    parser.add_argument(
        "--case",
        nargs="+",
        choices=("all", *CASE_SPECS),
        default=["all"],
        help="Case(s) to run. 'all' runs the complete required matched set.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--expected-arch",
        choices=("gfx950", "gfx1250"),
        help="Fail instead of collecting data on an unexpected GPU architecture.",
    )
    parser.add_argument(
        "--environment",
        choices=("physical", "ffm", "am"),
        default="physical",
        help="Label the execution environment in JSON; event timing is physical only.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print case definitions without requiring a GPU.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be non-negative and repeats must be positive")

    if "all" in args.case and args.case != ["all"]:
        parser.error("'all' cannot be combined with explicit case names")
    names = list(CASE_SPECS) if args.case == ["all"] else args.case
    if args.describe:
        print(json.dumps({name: CASE_SPECS[name] for name in names}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")

    _load_kernel_apis()
    arch = _arch()
    if args.expected_arch is not None and arch != args.expected_arch:
        raise RuntimeError(f"expected {args.expected_arch}, got {arch}")
    results: dict[str, Any] = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(0),
        "arch": arch,
        "environment": args.environment,
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": {},
    }
    for name in names:
        workload = _build(name, arch)
        workload.run()
        torch.cuda.synchronize()
        results["cases"][name] = {
            "spec": CASE_SPECS[name],
            "details": workload.details,
            "latency_us": (
                _time_us(workload.run, args.warmup, args.repeats)
                if args.environment == "physical"
                else None
            ),
        }
        del workload
        torch.cuda.empty_cache()

    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
