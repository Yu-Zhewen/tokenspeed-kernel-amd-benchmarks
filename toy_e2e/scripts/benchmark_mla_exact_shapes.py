#!/usr/bin/env python3
"""Benchmark Kimi-K3 TP8 rank-0 MLA subpaths at whole-model shapes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from tokenspeed_kernel import (
    kimi3_mla_qkv_gate_projection,
    mla_extend_with_kvcache,
    mla_normalize_project_query,
    mla_prefill,
    mla_project_value,
)


def _measure(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    ordered = sorted(samples)
    return {
        "repeats": repeats,
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p90_ms": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "max_ms": ordered[-1],
    }


def _fp8_random(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )


def _bench_prefill(
    *,
    batch: int,
    tokens_per_request: int,
    waves_per_eu: int | None = None,
    warmup: int,
    repeats: int,
) -> dict:
    total = batch * tokens_per_request
    q = _fp8_random((total, 12, 192))
    k = _fp8_random((total, 12, 192))
    v = _fp8_random((total, 12, 128))
    cu = torch.arange(
        0,
        total + 1,
        tokens_per_request,
        device="cuda",
        dtype=torch.int32,
    )
    out = torch.empty((total, 12, 128), device="cuda", dtype=torch.bfloat16)

    if waves_per_eu is None:

        def run():
            return mla_prefill(
                q,
                k,
                v,
                cu,
                cu,
                tokens_per_request,
                tokens_per_request,
                192**-0.5,
                out=out,
                solution="gluon",
            )

        suffix = ""
    else:
        from tokenspeed_kernel_amd.ops.gfx1250.attention.mla import prefill

        config = prefill.get_config(
            q=q,
            k=k,
            cu_seqlens_q=cu,
            max_seqlen_q=tokens_per_request,
            softmax_scale=192**-0.5,
        )

        def run():
            prefill._mla_prefill_gfx1250_kernel[config.grid](
                q,
                k,
                v,
                out,
                out,
                cu,
                cu,
                q.stride(0),
                q.stride(1),
                k.stride(0),
                k.stride(1),
                v.stride(0),
                v.stride(1),
                out.stride(0),
                out.stride(1),
                out.stride(0),
                out.stride(1),
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                config.rope_dim,
                config.sm_scale,
                True,
                False,
                config.block_m,
                config.block_n,
                True,
                num_warps=config.num_warps,
                waves_per_eu=waves_per_eu,
            )
            return out

        suffix = f"_waves{waves_per_eu}"

    return {
        "name": f"mla_prefill_b{batch}_t{tokens_per_request}{suffix}",
        "shape": {
            "batch_size": batch,
            "total_tokens": total,
            "tokens_per_request": tokens_per_request,
            "num_heads": 12,
            "qk_head_dim": 192,
            "value_head_dim": 128,
            "dtype": "float8_e4m3fn",
            "waves_per_eu": 1 if waves_per_eu is None else waves_per_eu,
        },
        "timing": _measure(run, warmup=warmup, repeats=repeats),
    }


def _bench_normalize_project(
    *,
    tokens: int,
    warmup: int,
    repeats: int,
) -> dict:
    query = torch.randn((tokens, 1536), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((tokens, 512), device="cuda", dtype=torch.bfloat16)
    q_weight = torch.randn((1536,), device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn((512,), device="cuda", dtype=torch.bfloat16)
    projection = torch.randn((2304, 1536), device="cuda", dtype=torch.bfloat16)

    def run():
        return mla_normalize_project_query(
            query,
            kv,
            q_weight,
            kv_weight,
            projection,
            eps=1e-6,
        )

    return {
        "name": f"mla_normalize_project_query_t{tokens}",
        "shape": {
            "num_tokens": tokens,
            "query_width": 1536,
            "kv_width": 512,
            "output_width": 2304,
            "dtype": "bfloat16",
        },
        "timing": _measure(run, warmup=warmup, repeats=repeats),
    }


def _bench_qkv_gate_projection(
    *,
    tokens: int,
    solution: str,
    warmup: int,
    repeats: int,
) -> dict:
    hidden = torch.randn((tokens, 7168), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((3648, 7168), device="cuda", dtype=torch.bfloat16)

    def run():
        return kimi3_mla_qkv_gate_projection(
            hidden,
            weight,
            2112,
            solution=solution,
        )

    return {
        "name": f"mla_qkv_gate_projection_t{tokens}_{solution}",
        "shape": {
            "num_tokens": tokens,
            "input_width": 7168,
            "qkv_width": 2112,
            "gate_width": 1536,
            "dtype": "bfloat16",
            "solution": solution,
        },
        "timing": _measure(run, warmup=warmup, repeats=repeats),
    }


def _bench_project_value(
    *,
    batch: int,
    warmup: int,
    repeats: int,
) -> dict:
    attention = torch.randn((batch, 12, 512), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((12, 512, 128), device="cuda", dtype=torch.bfloat16)
    gate = torch.randn((batch, 1536), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(gate)

    def run():
        return mla_project_value(attention, weight, gate=gate, out=out)

    return {
        "name": f"mla_project_value_gate_b{batch}",
        "shape": {
            "batch_size": batch,
            "num_heads": 12,
            "latent_dim": 512,
            "value_dim": 128,
            "gate_kind": "sigmoid",
            "dtype": "bfloat16",
        },
        "timing": _measure(run, warmup=warmup, repeats=repeats),
    }


def _bench_extend(
    *,
    batch: int,
    query_tokens_per_request: int,
    prefix_tokens: int,
    warmup: int,
    repeats: int,
) -> dict:
    total_q = batch * query_tokens_per_request
    visible_tokens = prefix_tokens + query_tokens_per_request
    pages_per_request = math.ceil(visible_tokens / 64)
    q = _fp8_random((total_q, 12, 576))
    kv_cache = _fp8_random((batch * pages_per_request, 64, 1, 576))
    page_table = torch.arange(
        batch * pages_per_request,
        device="cuda",
        dtype=torch.int32,
    ).view(batch, pages_per_request)
    cache_seqlens = torch.full(
        (batch,),
        visible_tokens,
        device="cuda",
        dtype=torch.int32,
    )
    cu_q = torch.arange(
        0,
        total_q + 1,
        query_tokens_per_request,
        device="cuda",
        dtype=torch.int32,
    )
    cu_kv = torch.arange(
        0,
        batch * visible_tokens + 1,
        visible_tokens,
        device="cuda",
        dtype=torch.int32,
    )
    out = torch.empty((total_q, 12, 512), device="cuda", dtype=torch.bfloat16)

    def run():
        return mla_extend_with_kvcache(
            q,
            kv_cache,
            page_table,
            cache_seqlens,
            cu_q,
            cu_kv,
            query_tokens_per_request,
            visible_tokens,
            128,
            512,
            64,
            192**-0.5,
            out=out,
            solution="gluon",
        )

    return {
        "name": (f"mla_extend_b{batch}_q{query_tokens_per_request}_p{prefix_tokens}"),
        "shape": {
            "batch_size": batch,
            "total_q": total_q,
            "query_tokens_per_request": query_tokens_per_request,
            "prefix_tokens": prefix_tokens,
            "num_heads": 12,
            "latent_dim": 512,
            "rope_dim": 64,
            "dtype": "float8_e4m3fn",
        },
        "timing": _measure(run, warmup=warmup, repeats=repeats),
    }


def run(args: argparse.Namespace) -> dict:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exact-shape MLA benchmark requires exactly one GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith("gfx1250"):
        raise RuntimeError(f"expected gfx1250, detected {arch}")

    cases = []
    if args.prefill_only:
        builders = [
            lambda batch=batch, tokens=tokens, waves=waves: _bench_prefill(
                batch=batch,
                tokens_per_request=tokens,
                waves_per_eu=waves,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            for batch in args.prefill_batches
            for tokens in args.prefill_seq_lengths
            for waves in args.prefill_waves_per_eu
        ]
    else:
        builders = [
        lambda: _bench_prefill(
            batch=1,
            tokens_per_request=4096,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        lambda: _bench_prefill(
            batch=2,
            tokens_per_request=4096,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        *(
            lambda batch=batch, waves=waves: _bench_prefill(
                batch=batch,
                tokens_per_request=4096,
                waves_per_eu=waves,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            for batch in (1, 2)
            for waves in (0, 1, 2, 3)
        ),
        lambda: _bench_normalize_project(
            tokens=4096,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        lambda: _bench_normalize_project(
            tokens=8192,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        lambda: _bench_normalize_project(
            tokens=16,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        lambda: _bench_project_value(
            batch=1,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        lambda: _bench_project_value(
            batch=16,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        *(
            lambda tokens=tokens, solution=solution: _bench_qkv_gate_projection(
                tokens=tokens,
                solution=solution,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            for tokens in (4096, 8192)
            for solution in ("split", "fused")
        ),
        lambda: _bench_extend(
            batch=1,
            query_tokens_per_request=128,
            prefix_tokens=3968,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        lambda: _bench_extend(
            batch=16,
            query_tokens_per_request=128,
            prefix_tokens=3968,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        ]
    for build in builders:
        case = build()
        print(json.dumps(case, sort_keys=True), flush=True)
        cases.append(case)
        torch.cuda.empty_cache()
    return {
        "format": "tokenspeed_kimi_k3_mla_exact_shapes_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "architecture": arch,
        "tokenspeed_revision": args.tokenspeed_revision,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": cases,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenspeed-revision", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help="run only the parameterized MLA prefill occupancy sweep",
    )
    parser.add_argument(
        "--prefill-batches",
        type=int,
        nargs="+",
        default=[1, 2],
    )
    parser.add_argument(
        "--prefill-seq-lengths",
        type=int,
        nargs="+",
        default=[4096],
    )
    parser.add_argument(
        "--prefill-waves-per-eu",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
