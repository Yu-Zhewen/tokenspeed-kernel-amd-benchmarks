#!/usr/bin/env python3
"""Sweep gfx1250 MLA value-projection tile and warp configurations."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import torch
from tokenspeed_kernel_amd.ops.gfx1250.attention.mla import (
    decode,
    project_value,
    reduce_project_value,
)


def _measure_us(fn, warmup: int, repeats: int) -> dict:
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
        samples.append(float(start.elapsed_time(end) * 1_000.0))
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def _parse_configs(values: list[str]) -> list[tuple[int, int]]:
    configs = []
    for value in values:
        block_n, num_warps = value.lower().split("x", maxsplit=1)
        configs.append((int(block_n), int(num_warps)))
    return configs


def _run_batch(
    *,
    batch: int,
    seq_len: int,
    configs: list[tuple[int, int]],
    warmup: int,
    repeats: int,
) -> dict:
    torch.manual_seed(307 + batch)
    heads = 12
    latent = 512
    value = 128
    rope = 64
    page_size = 64
    pages_per_sequence = math.ceil(seq_len / page_size)
    num_pages = batch * pages_per_sequence
    q = torch.randn(
        batch,
        1,
        heads,
        latent + rope,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    kv_cache = torch.randn(
        num_pages,
        page_size,
        1,
        latent + rope,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    page_table = torch.arange(
        num_pages,
        device="cuda",
        dtype=torch.int32,
    ).reshape(batch, pages_per_sequence)
    cache_seqlens = torch.tensor(
        [seq_len - (index % 4) * page_size for index in range(batch)],
        device="cuda",
        dtype=torch.int32,
    )
    attention = torch.randn(
        batch,
        heads,
        latent,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        heads,
        latent,
        value,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate_storage = torch.randn(
        batch,
        3648,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gate = gate_storage[:, -(heads * value) :]
    project_outputs = {
        config: torch.empty(
            batch,
            heads * value,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for config in configs
    }
    decode_outputs = {
        config: torch.empty_like(project_outputs[config]) for config in configs
    }
    decode_args = {
        "q": q,
        "kv_cache": kv_cache,
        "page_table": page_table,
        "cache_seqlens": cache_seqlens,
        "max_seqlen_k": seq_len,
        "qk_nope_head_dim": 128,
        "kv_lora_rank": latent,
        "qk_rope_head_dim": rope,
        "softmax_scale": 1.0 / math.sqrt(192),
        "value_weight": weight,
        "gate": gate,
    }

    timings = {}
    project_reference = None
    decode_reference = None
    for block_n, num_warps in configs:
        key = f"{block_n}x{num_warps}"
        project_out = project_outputs[(block_n, num_warps)]
        decode_out = decode_outputs[(block_n, num_warps)]

        def run_project() -> torch.Tensor:
            return project_value.gluon_mla_project_value_gfx1250(
                attention,
                weight,
                gate=gate,
                out=project_out,
            )

        def run_decode() -> torch.Tensor:
            return decode.gluon_mla_decode_projected_value_gfx1250(
                **decode_args,
                out=decode_out,
            )

        with (
            mock.patch.object(project_value, "_BLOCK_N", block_n),
            mock.patch.object(project_value, "_NUM_WARPS", num_warps),
            mock.patch.object(reduce_project_value, "_BLOCK_N", block_n),
            mock.patch.object(reduce_project_value, "_NUM_WARPS", num_warps),
        ):
            run_project()
            run_decode()
            torch.cuda.synchronize()
            if project_reference is None:
                project_reference = project_out.clone()
                decode_reference = decode_out.clone()
            else:
                torch.testing.assert_close(
                    project_out,
                    project_reference,
                    atol=0.125,
                    rtol=0.05,
                )
                torch.testing.assert_close(
                    decode_out,
                    decode_reference,
                    atol=0.125,
                    rtol=0.05,
                )
            timings[key] = {
                "project_value": _measure_us(run_project, warmup, repeats),
                "decode_projected_value": _measure_us(run_decode, warmup, repeats),
            }

    best_project = min(
        timings,
        key=lambda key: timings[key]["project_value"]["median_us"],
    )
    best_decode = min(
        timings,
        key=lambda key: timings[key]["decode_projected_value"]["median_us"],
    )
    return {
        "batch_size": batch,
        "sequence_length": seq_len,
        "shape": {
            "q": list(q.shape),
            "kv_cache": list(kv_cache.shape),
            "gate": list(gate.shape),
            "gate_stride": list(gate.stride()),
        },
        "timings": timings,
        "best_project": best_project,
        "best_decode": best_decode,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--seq-len", type=int, default=4161)
    parser.add_argument(
        "--config",
        nargs="+",
        default=["8x8", "8x4", "16x4", "16x8", "32x4", "32x8"],
        help="BLOCK_N x num_warps",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--tokenspeed-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("MLA tile sweep requires exactly one GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith("gfx1250"):
        raise RuntimeError(f"expected gfx1250, detected {arch}")
    configs = _parse_configs(args.config)
    cases = [
        _run_batch(
            batch=batch,
            seq_len=args.seq_len,
            configs=configs,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for batch in args.batch
    ]
    result = {
        "format": "tokenspeed_kimi_k3_mla_project_tiles_gfx1250_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "architecture": arch,
        "tokenspeed_revision": args.tokenspeed_revision,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
