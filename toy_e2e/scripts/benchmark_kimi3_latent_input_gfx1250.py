#!/usr/bin/env python3
"""Compare one-launch gfx1250 M1 latent input against the portable path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenspeed_kernel.ops.moe.triton.latent_input import triton_latent_input_packed
from tokenspeed_kernel_amd.ops.gfx1250.moe.fp16.latent_input_decode import (
    gluon_latent_input_decode_gfx1250,
)


def _measure_us(fn, warmup: int, repeats: int) -> dict[str, object]:
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
        samples.append(float(start.elapsed_time(end) * 1000.0))
    ordered = sorted(samples)
    return {
        "samples_us": samples,
        "median_us": ordered[len(ordered) // 2],
        "mean_us": sum(samples) / len(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith("gfx1250"):
        raise RuntimeError(f"expected gfx1250, detected {arch}")

    generator = torch.Generator(device="cuda").manual_seed(7)
    hidden = torch.randn(
        (1, 7168), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    packed = torch.randn(
        (896 + 3584 + 1536, 7168),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    router = packed[:896]
    routed = packed[896 : 896 + 3584]
    shared = packed[896 + 3584 :]

    def baseline():
        return triton_latent_input_packed(
            hidden,
            router,
            routed,
            shared,
            gate_clamp=4.0,
            up_clamp=25.0,
        )

    def candidate():
        return gluon_latent_input_decode_gfx1250(
            hidden,
            router,
            routed,
            shared,
            beta=4.0,
            linear_beta=25.0,
        )

    expected = baseline()
    actual = candidate()
    for candidate_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            candidate_output, expected_output, rtol=2e-2, atol=2e-2
        )

    baseline_timing = _measure_us(baseline, args.warmup, args.repeats)
    candidate_timing = _measure_us(candidate, args.warmup, args.repeats)
    baseline_median = float(baseline_timing["median_us"])
    candidate_median = float(candidate_timing["median_us"])
    result = {
        "architecture": arch,
        "baseline": baseline_timing,
        "candidate": candidate_timing,
        "candidate_launches": 1,
        "baseline_launches": 2,
        "speedup": baseline_median / candidate_median,
        "speedup_pct": (baseline_median / candidate_median - 1.0) * 100.0,
        "correct": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
