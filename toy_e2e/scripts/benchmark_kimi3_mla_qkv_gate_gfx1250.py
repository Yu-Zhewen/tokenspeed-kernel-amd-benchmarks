#!/usr/bin/env python3
"""Sweep fused versus split Kimi-K3 MLA QKV/gate projection on gfx1250."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenspeed_kernel.ops.gemm.kimi3 import (
    Kimi3MLAQKVGateProjection,
    kimi3_mla_qkv_gate_projection,
)
from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
    gluon_wmma_tdm_mla_qkv_gate_gfx1250,
)


QKV_WIDTH = 2112
GATE_WIDTH = 1536
INPUT_WIDTH = 7168
SOLUTIONS = ("auto", "fused", "split", "tdm_candidate")


def _launch(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    solution: str,
) -> Kimi3MLAQKVGateProjection | torch.Tensor:
    if solution == "tdm_candidate":
        return gluon_wmma_tdm_mla_qkv_gate_gfx1250(
            hidden_states,
            weight,
        )
    return kimi3_mla_qkv_gate_projection(
        hidden_states,
        weight,
        QKV_WIDTH,
        solution=solution,
    )


def _packed_for_check(
    projection: Kimi3MLAQKVGateProjection | torch.Tensor,
) -> torch.Tensor:
    if isinstance(projection, torch.Tensor):
        return projection
    if projection.packed is not None:
        return projection.packed
    return torch.cat((projection.qkv, projection.gate), dim=-1)


def _measure(
    activations: list[torch.Tensor],
    weights: list[torch.Tensor],
    solution: str,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    rotation = len(activations)
    for index in range(warmup):
        _launch(
            activations[index % rotation],
            weights[index % rotation],
            solution,
        )
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        slot = index % rotation
        start.record()
        _launch(activations[slot], weights[slot], solution)
        end.record()
    ends[-1].synchronize()
    samples = [
        float(start.elapsed_time(end) * 1.0e3)
        for start, end in zip(starts, ends, strict=True)
    ]
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument("--rotation", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one visible GPU")
    architecture = torch.cuda.get_device_properties(0).gcnArchName
    if not architecture.startswith("gfx1250"):
        raise RuntimeError(f"expected gfx1250, detected {architecture}")
    if args.rotation < 1 or args.warmup < 1 or args.repeats < 2:
        raise ValueError("rotation/warmup must be positive and repeats must be >= 2")

    results = []
    for m in args.m:
        if m < 1 or m > 128:
            raise ValueError("this decode sweep requires 1 <= M <= 128")
        generator = torch.Generator(device="cuda")
        generator.manual_seed(1_250_3648 + m)
        activations = [
            torch.randn(
                (m, INPUT_WIDTH),
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            for _ in range(args.rotation)
        ]
        weights = [
            torch.randn(
                (QKV_WIDTH + GATE_WIDTH, INPUT_WIDTH),
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            for _ in range(args.rotation)
        ]

        reference = torch.nn.functional.linear(activations[0], weights[0])
        correctness = {}
        timings = {}
        for solution in SOLUTIONS:
            output = _packed_for_check(
                _launch(activations[0], weights[0], solution)
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
            difference = (output.float() - reference.float()).abs()
            correctness[solution] = {
                "max_abs_error": float(difference.max().item()),
                "mean_abs_error": float(difference.mean().item()),
            }
            timings[solution] = _measure(
                activations,
                weights,
                solution,
                warmup=args.warmup,
                repeats=args.repeats,
            )

        fused_us = float(timings["fused"]["median_us"])
        split_us = float(timings["split"]["median_us"])
        candidate_us = (
            float(timings["tdm_candidate"]["median_us"])
            if "tdm_candidate" in timings
            else None
        )
        result = {
            "M": m,
            "shape": {
                "input": [m, INPUT_WIDTH],
                "weight": [QKV_WIDTH + GATE_WIDTH, INPUT_WIDTH],
                "qkv_width": QKV_WIDTH,
                "gate_width": GATE_WIDTH,
            },
            "correctness": correctness,
            "timings": timings,
            "split_speedup_pct": 100.0 * (fused_us - split_us) / fused_us,
            "candidate_speedup_pct": (
                100.0 * (fused_us - candidate_us) / fused_us
                if candidate_us is not None
                else None
            ),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "format": "kimi3_mla_qkv_gate_gfx1250_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "architecture": architecture,
        "rotation": args.rotation,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
