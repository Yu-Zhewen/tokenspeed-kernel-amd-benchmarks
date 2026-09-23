#!/usr/bin/env python3
"""Benchmark large-M Kimi-K3 BF16 projection shapes on gfx1250."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.fused import (
    PrecisionConfig,
    matmul,
)

SHAPES = {
    "attn_output": (7168, 1536),
    "qkv_gate": (8448, 7168),
    "packed_output": (7168, 4224),
    "shared_down": (7168, 1536),
    "routed_down": (3584, 7168),
    "routed_up": (7168, 3584),
    "kda_qkvfab": (6288, 7168),
}

CONFIGS = (
    *(
        {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": 128,
            "num_buffers": 2,
            "num_warps": 4,
        }
        for block_m in (16, 32)
        for block_n in (64, 128, 256)
    ),
    {
        "block_m": 64,
        "block_n": 64,
        "block_k": 128,
        "num_buffers": 2,
        "num_warps": 4,
    },
    {
        "block_m": 64,
        "block_n": 128,
        "block_k": 128,
        "num_buffers": 2,
        "num_warps": 4,
    },
    {
        "block_m": 128,
        "block_n": 64,
        "block_k": 128,
        "num_buffers": 2,
        "num_warps": 4,
    },
    {
        "block_m": 128,
        "block_n": 128,
        "block_k": 128,
        "num_buffers": 2,
        "num_warps": 4,
    },
    {
        "block_m": 128,
        "block_n": 128,
        "block_k": 256,
        "num_buffers": 2,
        "num_warps": 4,
    },
    {
        "block_m": 256,
        "block_n": 64,
        "block_k": 128,
        "num_buffers": 2,
        "num_warps": 4,
    },
    {
        "block_m": 256,
        "block_n": 128,
        "block_k": 128,
        "num_buffers": 2,
        "num_warps": 4,
    },
    {
        "block_m": 256,
        "block_n": 128,
        "block_k": 256,
        "num_buffers": 2,
        "num_warps": 4,
    },
    {
        "block_m": 128,
        "block_n": 256,
        "block_k": 128,
        "num_buffers": 2,
        "num_warps": 4,
    },
    {
        "block_m": 256,
        "block_n": 256,
        "block_k": 128,
        "num_buffers": 2,
        "num_warps": 4,
    },
    {
        "block_m": 128,
        "block_n": 128,
        "block_k": 256,
        "num_buffers": 3,
        "num_warps": 4,
    },
    {
        "block_m": 128,
        "block_n": 128,
        "block_k": 256,
        "num_buffers": 3,
        "num_warps": 4,
        "schedule": "sliceK",
    },
    {
        "block_m": 128,
        "block_n": 128,
        "block_k": 256,
        "num_buffers": 3,
        "num_warps": 8,
        "pingpong": True,
    },
    {
        "block_m": 256,
        "block_n": 16,
        "block_k": 128,
        "num_buffers": 2,
        "num_warps": 4,
    },
    {
        "block_m": 256,
        "block_n": 32,
        "block_k": 128,
        "num_buffers": 2,
        "num_warps": 4,
    },
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
        samples.append(float(start.elapsed_time(end) * 1e3))
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def _gluon_projection(
    activation: torch.Tensor,
    weight: torch.Tensor,
    config: dict[str, object],
) -> torch.Tensor:
    output, _ = matmul(
        activation,
        weight.T,
        None,
        precision_config=PrecisionConfig(out_dtype=activation.dtype),
        group_m=8,
        xcd_swizzle=1,
        w_transpose=True,
        **config,
    )
    return output


def _benchmark_shape(
    name: str,
    m: int,
    *,
    configs: tuple[dict[str, object], ...],
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    n, k = SHAPES[name]
    print(f"allocating {name} M={m} N={n} K={k}", flush=True)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1250 + m + n + k)
    activation = torch.randn(
        m,
        k,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    weight = torch.randn(
        n,
        k,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    def baseline_fn() -> torch.Tensor:
        return torch.nn.functional.linear(activation, weight)

    print(f"measuring baseline {name} M={m}", flush=True)
    reference = baseline_fn()
    torch.cuda.synchronize()
    baseline = _measure_us(baseline_fn, warmup, repeats)

    candidates = []
    for config in configs:
        print(f"starting candidate {name} M={m} config={config}", flush=True)
        try:
            output = _gluon_projection(activation, weight, config)
            torch.cuda.synchronize()
            torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)

            def candidate_fn() -> torch.Tensor:
                return _gluon_projection(activation, weight, config)

            timing = _measure_us(candidate_fn, warmup, repeats)
            candidates.append({"config": config, "status": "passed", "timing": timing})
            print(
                f"completed candidate {name} M={m} median_us={timing['median_us']:.3f}",
                flush=True,
            )
        except Exception as error:
            candidates.append(
                {
                    "config": config,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                f"failed candidate {name} M={m}: {type(error).__name__}: {error}",
                flush=True,
            )

    passed = [candidate for candidate in candidates if candidate["status"] == "passed"]
    if not passed:
        raise RuntimeError(f"all candidates failed for {name} M={m}")
    best = min(passed, key=lambda candidate: candidate["timing"]["median_us"])
    baseline_us = float(baseline["median_us"])
    best_us = float(best["timing"]["median_us"])
    flops = 2 * m * n * k
    minimum_bytes = 2 * (m * k + n * k + m * n)
    return {
        "name": name,
        "shape": {"M": m, "N": n, "K": k},
        "baseline": baseline,
        "candidates": candidates,
        "best_config": best["config"],
        "best_timing": best["timing"],
        "speedup": baseline_us / best_us,
        "speedup_pct": 100.0 * (baseline_us - best_us) / baseline_us,
        "baseline_tflops": flops / baseline_us / 1e6,
        "best_tflops": flops / best_us / 1e6,
        "baseline_minimum_bandwidth_gbps": minimum_bytes / baseline_us / 1e3,
        "best_minimum_bandwidth_gbps": minimum_bytes / best_us / 1e3,
    }


def _compile_smoke(
    name: str,
    m: int,
    config: dict[str, object],
) -> None:
    n, k = SHAPES[name]
    print(
        f"compile smoke {name} M={m} N={n} K={k} config={config}",
        flush=True,
    )
    activation = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros(n, k, dtype=torch.bfloat16, device="cuda")
    output = _gluon_projection(activation, weight, config)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("compile-smoke output contains non-finite values")
    print("compile smoke passed", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument(
        "--shape",
        choices=tuple(SHAPES),
        nargs="+",
        default=list(SHAPES),
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--config-index",
        type=int,
        help="run exactly one zero-based CONFIGS entry",
    )
    parser.add_argument(
        "--compile-smoke",
        action="store_true",
        help="compile and launch one candidate without baseline timing",
    )
    parser.add_argument("--tokenspeed-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("prefill GEMM benchmark requires exactly one visible GPU")
    architecture = torch.cuda.get_device_properties(0).gcnArchName
    if not architecture.startswith("gfx1250"):
        raise RuntimeError(f"expected gfx1250, detected {architecture}")

    if args.config_index is None:
        configs = CONFIGS
    elif 0 <= args.config_index < len(CONFIGS):
        configs = (CONFIGS[args.config_index],)
    else:
        raise ValueError(
            f"config-index must be in [0, {len(CONFIGS) - 1}], got {args.config_index}"
        )
    if args.compile_smoke:
        if args.config_index is None or len(args.shape) != 1 or len(args.m) != 1:
            raise ValueError(
                "compile-smoke requires one --config-index, --shape, and --m"
            )
        _compile_smoke(args.shape[0], args.m[0], configs[0])
        return 0

    benchmarks = []
    for m in args.m:
        if m <= 0:
            raise ValueError(f"M must be positive, got {m}")
        for name in args.shape:
            benchmarks.append(
                _benchmark_shape(
                    name,
                    m,
                    configs=configs,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()

    result = {
        "format": "tokenspeed_kimi3_prefill_gemm_gfx1250_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokenspeed_revision": args.tokenspeed_revision,
        "architecture": architecture,
        "dtype": str(torch.bfloat16),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "benchmarks": benchmarks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
