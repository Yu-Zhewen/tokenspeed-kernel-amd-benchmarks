#!/usr/bin/env python3
"""Benchmark fused CDNA5 Kimi-K3 M16 projection-plus-add3 candidates."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenspeed_kernel.ops.gemm.kimi3 import kimi3_latent_projection_add3
from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
    gluon_wmma_tdm_add3_m16_gfx1250,
    triton_mm_a16w16_add3_m16_gfx1250,
)

PRODUCTION_TRITON_CONFIG = {
    "block_n": 32,
    "block_k": 64,
    "num_warps": 2,
    "waves_per_eu": 1,
}


TRITON_CONFIGS = (
    {"block_n": 16, "block_k": 16, "num_warps": 1, "waves_per_eu": 1},
    {"block_n": 16, "block_k": 32, "num_warps": 1, "waves_per_eu": 1},
    {"block_n": 16, "block_k": 32, "num_warps": 2, "waves_per_eu": 1},
    {"block_n": 32, "block_k": 16, "num_warps": 1, "waves_per_eu": 1},
    {"block_n": 32, "block_k": 16, "num_warps": 2, "waves_per_eu": 1},
    {"block_n": 32, "block_k": 32, "num_warps": 2, "waves_per_eu": 1},
    {"block_n": 32, "block_k": 32, "num_warps": 2, "waves_per_eu": 2},
    {"block_n": 32, "block_k": 64, "num_warps": 2, "waves_per_eu": 1},
    {"block_n": 64, "block_k": 16, "num_warps": 2, "waves_per_eu": 1},
    {"block_n": 64, "block_k": 16, "num_warps": 4, "waves_per_eu": 1},
    {"block_n": 64, "block_k": 32, "num_warps": 4, "waves_per_eu": 1},
    {"block_n": 64, "block_k": 64, "num_warps": 4, "waves_per_eu": 1},
    {"block_n": 128, "block_k": 32, "num_warps": 4, "waves_per_eu": 1},
    {
        "block_n": 32,
        "block_k": 64,
        "num_warps": 2,
        "num_stages": 1,
        "waves_per_eu": 1,
        "bf16_boundary": False,
    },
    {
        "block_n": 32,
        "block_k": 128,
        "num_warps": 2,
        "num_stages": 2,
        "waves_per_eu": 1,
        "bf16_boundary": False,
    },
    {
        "block_n": 32,
        "block_k": 256,
        "num_warps": 2,
        "num_stages": 2,
        "waves_per_eu": 1,
        "bf16_boundary": False,
    },
    {
        "block_n": 64,
        "block_k": 128,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 1,
        "bf16_boundary": False,
    },
    {
        "block_n": 64,
        "block_k": 256,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 1,
        "bf16_boundary": False,
    },
    {
        "block_n": 32,
        "block_k": 128,
        "num_warps": 4,
        "num_stages": 3,
        "waves_per_eu": 1,
        "bf16_boundary": False,
    },
)


def _run_candidate(
    backend: str,
    config: dict[str, object],
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    prefix: torch.Tensor,
    shared_output: torch.Tensor,
) -> torch.Tensor:
    if backend == "gluon_tdm":
        return gluon_wmma_tdm_add3_m16_gfx1250(
            hidden_states,
            weight,
            prefix,
            shared_output,
        )
    if backend in {"triton_production", "triton"}:
        return triton_mm_a16w16_add3_m16_gfx1250(
            hidden_states,
            weight,
            prefix,
            shared_output,
            **config,
        )
    raise ValueError(f"unknown backend {backend!r}")


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-backends",
        choices=("gluon_tdm", "triton_production", "triton"),
        nargs="+",
        default=("gluon_tdm", "triton"),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--tokenspeed-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("add3 benchmark requires exactly one visible GPU")
    architecture = torch.cuda.get_device_properties(0).gcnArchName
    if not architecture.startswith("gfx1250"):
        raise RuntimeError(f"expected gfx1250, detected {architecture}")

    torch.manual_seed(829)
    hidden_states = torch.randn(
        16, 3584, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(
        7168, 3584, device="cuda", dtype=torch.bfloat16
    )
    prefix = torch.randn(16, 7168, device="cuda", dtype=torch.bfloat16)
    shared_lane = torch.randn(16, 10752, device="cuda", dtype=torch.bfloat16)
    shared_output = shared_lane[:, 3584:]

    def composed() -> torch.Tensor:
        return kimi3_latent_projection_add3(
            hidden_states,
            weight,
            prefix,
            shared_output,
            solution="composed",
        )

    reference = composed()
    torch.cuda.synchronize()
    baseline = _measure_us(composed, args.warmup, args.repeats)
    candidates = []
    candidate_specs = []
    if "gluon_tdm" in args.candidate_backends:
        candidate_specs.append(("gluon_tdm", {}))
    if "triton_production" in args.candidate_backends:
        candidate_specs.append(("triton_production", PRODUCTION_TRITON_CONFIG))
    if "triton" in args.candidate_backends:
        candidate_specs.extend(("triton", config) for config in TRITON_CONFIGS)
    for backend, config in candidate_specs:
        try:
            output = _run_candidate(
                backend,
                config,
                hidden_states,
                weight,
                prefix,
                shared_output,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)

            def candidate() -> torch.Tensor:
                return _run_candidate(
                    backend,
                    config,
                    hidden_states,
                    weight,
                    prefix,
                    shared_output,
                )

            timing = _measure_us(candidate, args.warmup, args.repeats)
            semantic_match = backend == "gluon_tdm" or bool(
                config.get("bf16_boundary", True)
            )
            candidates.append(
                {
                    "backend": backend,
                    "config": config,
                    "semantic_match": semantic_match,
                    "timing": timing,
                    "status": "passed",
                }
            )
        except Exception as error:
            candidates.append(
                {
                    "backend": backend,
                    "config": config,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    passed = [
        candidate
        for candidate in candidates
        if candidate["status"] == "passed" and candidate["semantic_match"]
    ]
    if not passed:
        raise RuntimeError("all fused add3 candidates failed")
    best = min(passed, key=lambda candidate: candidate["timing"]["median_us"])
    best_backend = best["backend"]
    best_config = best["config"]
    baseline_median = float(baseline["median_us"])
    best_median = float(best["timing"]["median_us"])
    speedup_pct = 100.0 * (baseline_median - best_median) / baseline_median

    def best_candidate() -> torch.Tensor:
        return _run_candidate(
            best_backend,
            best_config,
            hidden_states,
            weight,
            prefix,
            shared_output,
        )

    best_candidate()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = best_candidate()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, reference, rtol=2e-2, atol=2e-2)

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    )
    profiler.start()
    best_candidate()
    torch.cuda.synchronize()
    profiler.stop()
    profiler.export_chrome_trace(str(args.trace))
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    kernel_names = [
        event["name"]
        for event in trace.get("traceEvents", [])
        if event.get("ph") == "X"
        and str(event.get("cat", "")).lower() in {"kernel", "gpu_kernel"}
    ]

    result = {
        "format": "tokenspeed_kimi3_add3_gfx1250_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokenspeed_revision": args.tokenspeed_revision,
        "architecture": architecture,
        "shape": {"M": 16, "N": 7168, "K": 3584},
        "dtype": str(hidden_states.dtype),
        "shared_output_stride": list(shared_output.stride()),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "baseline_composed": baseline,
        "candidates": candidates,
        "best_backend": best_backend,
        "best_config": best_config,
        "best_timing": best["timing"],
        "speedup_pct": speedup_pct,
        "acceptance_threshold_pct": 5.0,
        "accepted": speedup_pct >= 5.0 and len(kernel_names) == 1,
        "graph_replay": "passed",
        "profile_kernel_count": len(kernel_names),
        "profile_kernel_names": kernel_names,
        "trace": str(args.trace),
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
