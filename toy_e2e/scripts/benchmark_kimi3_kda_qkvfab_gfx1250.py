#!/usr/bin/env python3
"""Sweep the CDNA5 Kimi-K3 KDA QKVFAB projection candidate."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenspeed_kernel.ops.gemm.kimi3 import kimi3_qkvfab_projection
from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
    gluon_wmma_tdm_kda_qkvfab_gfx1250,
)


def _measure(
    launch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    activations: list[torch.Tensor],
    weights: list[torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    rotation = len(activations)
    for index in range(warmup):
        launch(activations[index % rotation], weights[index % rotation])
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        slot = index % rotation
        start.record()
        launch(activations[slot], weights[slot])
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
        default=[1, 2, 4, 8, 16, 32],
    )
    parser.add_argument("--rotation", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
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

    launches: dict[
        str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = {
        "auto": lambda activation, weight: kimi3_qkvfab_projection(
            activation,
            weight,
        ),
        "torch": lambda activation, weight: torch.nn.functional.linear(
            activation,
            weight,
        ),
        "vendor": lambda activation, weight: kimi3_qkvfab_projection(
            activation,
            weight,
            solution="torch",
        ),
        "tdm_candidate": gluon_wmma_tdm_kda_qkvfab_gfx1250,
    }
    results = []
    for m in args.m:
        if m not in {1, 2, 4, 8, 16, 32}:
            raise ValueError("M must be one of 1, 2, 4, 8, 16, or 32")
        generator = torch.Generator(device="cuda")
        generator.manual_seed(6_288_1250 + m)
        activations = [
            torch.randn(
                (m, 7168),
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            for _ in range(args.rotation)
        ]
        weights = [
            torch.randn(
                (6288, 7168),
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            for _ in range(args.rotation)
        ]

        reference = torch.nn.functional.linear(activations[0], weights[0])
        correctness = {}
        timings = {}
        for name, launch in launches.items():
            output = launch(activations[0], weights[0])
            torch.cuda.synchronize()
            torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
            difference = (output.float() - reference.float()).abs()
            correctness[name] = {
                "max_abs_error": float(difference.max().item()),
                "mean_abs_error": float(difference.mean().item()),
            }
            timings[name] = _measure(
                launch,
                activations,
                weights,
                warmup=args.warmup,
                repeats=args.repeats,
            )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = gluon_wmma_tdm_kda_qkvfab_gfx1250(
                activations[0],
                weights[0],
            )
        activations[0].copy_(torch.randn_like(activations[0]))
        mutated_reference = torch.nn.functional.linear(
            activations[0],
            weights[0],
        )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            captured,
            mutated_reference,
            rtol=2e-2,
            atol=2e-2,
        )

        baseline_us = float(timings["vendor"]["median_us"])
        auto_us = float(timings["auto"]["median_us"])
        candidate_us = float(timings["tdm_candidate"]["median_us"])
        result = {
            "M": m,
            "shape": {"M": m, "N": 6288, "K": 7168},
            "correctness": correctness,
            "timings": timings,
            "candidate_speedup_pct": (
                100.0 * (baseline_us - candidate_us) / baseline_us
            ),
            "candidate_vs_auto_pct": (
                100.0 * (auto_us - candidate_us) / auto_us
            ),
            "graph_replay_with_mutation": "passed",
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "format": "kimi3_kda_qkvfab_gfx1250_v2",
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
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
