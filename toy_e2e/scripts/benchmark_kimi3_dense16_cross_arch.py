#!/usr/bin/env python3
"""Compare exact Kimi-K3 dense BF16 projections across AMD architectures."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import statistics
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import torch
import tokenspeed_kernel


SHAPES = {
    "attn_output": (7168, 1536),
    "latent_up": (7168, 3584),
    "latent_down": (3584, 7168),
    "shared_down": (7168, 768),
    "qkv_gate": (8448, 7168),
    "state_projection": (3072, 512),
    "packed_output": (7168, 4224),
    "mla_qkv": (2112, 7168),
    "mla_gate": (1536, 7168),
    "mla_query": (2304, 1536),
    "kda_qkvfab": (6288, 7168),
    "kda_fb": (1536, 128),
}
LATENT_SHAPES = {(3584, 7168), (7168, 3584)}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize(samples_us: list[float], m: int, n: int, k: int) -> dict[str, object]:
    median_us = statistics.median(samples_us)
    return {
        "samples_us": samples_us,
        "median_us": median_us,
        "mean_us": statistics.fmean(samples_us),
        "min_us": min(samples_us),
        "p10_us": _percentile(samples_us, 0.10),
        "p90_us": _percentile(samples_us, 0.90),
        "max_us": max(samples_us),
        "tflops_at_median": 2.0 * m * n * k / median_us / 1.0e6,
    }


def _measure(
    launch: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    activations: list[torch.Tensor],
    weights: list[torch.Tensor],
    outputs: list[torch.Tensor],
    *,
    repeats: int,
    warmup: int,
    rotating: bool,
) -> list[float]:
    count = len(activations) if rotating else 1
    for index in range(warmup):
        slot = index % count
        launch(activations[slot], weights[slot], outputs[slot])
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        slot = index % count
        start.record()
        launch(activations[slot], weights[slot], outputs[slot])
        end.record()
    ends[-1].synchronize()
    return [
        float(start.elapsed_time(end) * 1.0e3)
        for start, end in zip(starts, ends, strict=True)
    ]


def _runtime_launch(
    k: int, n: int
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    if (k, n) in LATENT_SHAPES:
        return lambda activation, weight, output: (
            tokenspeed_kernel.kimi3_latent_projection(
                activation,
                weight,
                out=output,
            )
        )
    return lambda activation, weight, output: tokenspeed_kernel.mm(
        activation,
        weight,
        out=output,
    )


def _custom_launches(
    architecture: str,
    k: int,
    n: int,
) -> dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
    if architecture.startswith("gfx950"):
        if n % 256 != 0 or k < 256:
            return {}
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
            gluon_mm_a16w16_largem_gfx950,
        )

        return {
            "gluon": lambda activation, weight, output: gluon_mm_a16w16_largem_gfx950(
                activation,
                weight,
                activation.dtype,
                out=output,
            )
        }
    if architecture.startswith("gfx1250"):
        from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.fused import (
            PrecisionConfig,
            matmul,
        )

        def _launch(
            activation: torch.Tensor,
            weight: torch.Tensor,
            output: torch.Tensor,
            *,
            block_m: int,
            block_n: int,
        ) -> torch.Tensor:
            result, _ = matmul(
                activation,
                weight.T,
                None,
                precision_config=PrecisionConfig(out_dtype=activation.dtype),
                block_m=block_m,
                block_n=block_n,
                block_k=128,
                group_m=8,
                xcd_swizzle=1,
                w_transpose=True,
                num_buffers=2,
                num_warps=4,
                out=output,
            )
            return result

        launches = {}
        if n % 32 == 0:
            for block_m in (128, 256):
                block_n = next(
                    candidate
                    for candidate in (block_m, 128, 64, 32)
                    if n % candidate == 0
                )
                launches[f"gluon_{block_m}x{block_n}"] = (
                    lambda activation, weight, output, block_m=block_m, block_n=block_n: (
                        _launch(
                            activation,
                            weight,
                            output,
                            block_m=block_m,
                            block_n=block_n,
                        )
                    )
                )

        from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
            gluon_mm_a16w16_largem_gfx1250,
        )

        if (k, n) in {
            (512, 3072),
            (768, 7168),
            (1536, 2304),
            (1536, 7168),
            (3584, 7168),
            (4224, 7168),
            (7168, 1536),
            (7168, 2112),
            (7168, 3584),
            (7168, 6288),
            (7168, 8448),
        }:
            launches["gluon_generalized"] = lambda activation, weight, output: (
                gluon_mm_a16w16_largem_gfx1250(
                    activation,
                    weight,
                    out=output,
                )
            )
        return launches
    return {}


def _benchmark_shape(
    architecture: str,
    name: str,
    m: int,
    *,
    include_custom: bool,
    rotation: int,
    repeats: int,
    warmup: int,
) -> dict[str, object]:
    n, k = SHAPES[name]
    generator = torch.Generator(device="cuda")
    generator.manual_seed(450_950 + m + n + k)
    activations = [
        torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(rotation)
    ]
    weights = [
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(rotation)
    ]
    outputs = [
        torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        for _ in range(rotation)
    ]

    launches: dict[
        str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ] = {
        "torch": lambda activation, weight, output: torch.mm(
            activation, weight.T, out=output
        ),
        "runtime": _runtime_launch(k, n),
    }
    if include_custom:
        launches.update(_custom_launches(architecture, k, n))

    reference = torch.empty_like(outputs[0])
    torch.mm(activations[0], weights[0].T, out=reference)
    correctness: dict[str, object] = {}
    for backend, launch in launches.items():
        launch(activations[0], weights[0], outputs[0])
        torch.cuda.synchronize()
        torch.testing.assert_close(outputs[0], reference, rtol=2.0e-2, atol=2.0e-2)
        difference = (outputs[0].float() - reference.float()).abs()
        correctness[backend] = {
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
        }

    measurements: dict[str, object] = {}
    for cache_mode, rotating in (("hot", False), ("rotating", True)):
        measurements[cache_mode] = {}
        for backend, launch in launches.items():
            samples = _measure(
                launch,
                activations,
                weights,
                outputs,
                repeats=repeats,
                warmup=max(warmup, rotation if rotating else warmup),
                rotating=rotating,
            )
            measurements[cache_mode][backend] = _summarize(samples, m, n, k)

    del reference, activations, weights, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "name": name,
        "shape": {"M": m, "N": n, "K": k},
        "rotation": rotation,
        "correctness": correctness,
        "measurements": measurements,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--include-custom", action="store_true")
    parser.add_argument(
        "--shape",
        choices=tuple(SHAPES),
        nargs="+",
        default=list(SHAPES),
    )
    parser.add_argument("--rotation", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one visible GPU")
    if args.rotation < 2 or args.warmup < 1 or args.repeats < 2:
        raise ValueError("rotation >= 2, warmup >= 1, and repeats >= 2 are required")
    properties = torch.cuda.get_device_properties(0)
    architecture = properties.gcnArchName
    if not (architecture.startswith("gfx950") or architecture.startswith("gfx1250")):
        raise RuntimeError(f"expected gfx950 or gfx1250, detected {architecture}")

    results = []
    for m in args.m:
        if m < 1 or m > 8192:
            raise ValueError("exact Kimi-K3 benchmark supports 1 <= M <= 8192")
        for name in args.shape:
            result = _benchmark_shape(
                architecture,
                name,
                m,
                include_custom=args.include_custom,
                rotation=args.rotation,
                repeats=args.repeats,
                warmup=args.warmup,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "format": "kimi3_dense16_cross_arch_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": properties.name,
        "architecture": architecture,
        "torch_version": torch.__version__,
        "triton_version": importlib.metadata.version("tokenspeed-triton"),
        "benchmark": {
            "m": args.m,
            "shapes": args.shape,
            "include_custom": args.include_custom,
            "rotation": args.rotation,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
