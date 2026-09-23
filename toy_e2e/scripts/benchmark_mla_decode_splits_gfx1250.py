#!/usr/bin/env python3
"""Sweep gfx1250 projected-MLA decode split counts at Kimi-K3 shapes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import torch
from tokenspeed_kernel_amd.ops.gfx1250.attention.mla import decode


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


def _run_shape(
    *,
    batch: int,
    seq_len: int,
    splits: list[int],
    warmup: int,
    repeats: int,
) -> dict:
    torch.manual_seed(211 + batch + seq_len)
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
    outputs = {
        split: torch.empty(
            batch,
            heads * value,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for split in splits
    }
    reference = torch.empty(
        batch,
        heads * value,
        device="cuda",
        dtype=torch.bfloat16,
    )

    common = {
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
    decode.gluon_mla_decode_projected_value_gfx1250(
        **common,
        out=reference,
    )
    torch.cuda.synchronize()

    timings = {}
    numerics = {}
    for split in splits:
        output = outputs[split]

        def run() -> torch.Tensor:
            return decode.gluon_mla_decode_projected_value_gfx1250(
                **common,
                out=output,
            )

        with mock.patch.object(
            decode,
            "_select_num_kv_splits",
            return_value=split,
        ):
            run()
            torch.cuda.synchronize()
            close = torch.isclose(output, reference, atol=0.125, rtol=0.05)
            difference = (output.float() - reference.float()).abs()
            numerics[str(split)] = {
                "within_established_tolerance": bool(close.all().item()),
                "mismatched_elements": int((~close).sum().item()),
                "max_abs_difference": float(difference.max().item()),
            }
            timings[str(split)] = _measure_us(run, warmup, repeats)

    accepted_splits = [
        split
        for split in splits
        if numerics[str(split)]["within_established_tolerance"]
    ]
    if not accepted_splits:
        raise RuntimeError("no split candidate met the established tolerance")
    best_split = min(
        accepted_splits,
        key=lambda split: timings[str(split)]["median_us"],
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
        "numerics": numerics,
        "best_split": best_split,
        "best_median_us": timings[str(best_split)]["median_us"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--seq-len", type=int, nargs="+", default=[4097, 5120])
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--tokenspeed-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("MLA split sweep requires exactly one GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith("gfx1250"):
        raise RuntimeError(f"expected gfx1250, detected {arch}")
    cases = [
        _run_shape(
            batch=batch,
            seq_len=seq_len,
            splits=args.splits,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for batch in args.batch
        for seq_len in args.seq_len
    ]
    result = {
        "format": "tokenspeed_kimi_k3_mla_decode_splits_gfx1250_v1",
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
