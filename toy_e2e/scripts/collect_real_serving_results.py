#!/usr/bin/env python3
"""Normalize EvalScope outputs from real Kimi-K3 TP8/EP1 serving."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_unique(root: Path, filename: str) -> tuple[Path, Any]:
    paths = list(root.rglob(filename))
    if len(paths) != 1:
        raise ValueError(
            f"expected one {filename} below {root}, found {len(paths)}"
        )
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


def _run(root: Path, concurrency: int, gpu_count: int) -> dict[str, Any]:
    run_root = root / f"c{concurrency}"
    summary_path, summary = _read_unique(run_root, "benchmark_summary.json")
    percentile_path, percentiles = _read_unique(
        run_root, "benchmark_percentile.json"
    )
    throughput_path, throughput = _read_unique(
        run_root, "workload_throughput.json"
    )
    args_path, benchmark_args = _read_unique(run_root, "benchmark_args.json")
    p50 = next(row for row in percentiles if row["Percentiles"] == "50%")
    p90 = next(row for row in percentiles if row["Percentiles"] == "90%")
    completion = next(
        row for row in throughput["rows"] if row["metric"] == "Completion tok/s"
    )
    if int(summary["Concurrency"]) != concurrency:
        raise ValueError(f"unexpected concurrency in {summary_path}")
    if summary["Success Requests"] != summary["Total Requests"]:
        raise ValueError(f"failed requests recorded in {summary_path}")
    if summary["Avg Input Tokens"] != 4096 or summary["Avg Output Tokens"] != 1024:
        raise ValueError(f"unexpected token lengths in {summary_path}")
    mean_latency_s = float(summary["Avg Latency (s)"])
    mean_ttft_s = float(summary["TTFT (ms)"]) / 1e3
    mean_post_ttft_s = mean_latency_s - mean_ttft_s
    summary = {
        key: value
        for key, value in summary.items()
        if key not in {"Decoded Tok/Iter", "Spec. Accept Rate"}
    }
    return {
        "concurrency": concurrency,
        "request_count": int(summary["Total Requests"]),
        "warmup_requests": int(benchmark_args["warmup_num"]),
        "summary": summary,
        "p50": p50,
        "p90": p90,
        "workload_throughput": throughput,
        "derived": {
            "overall_output_tps_per_gpu": (
                float(summary["Output Throughput (tok/s)"]) / gpu_count
            ),
            "steady_output_tps": float(completion["steady_state"]),
            "steady_output_tps_per_gpu": (
                float(completion["steady_state"]) / gpu_count
            ),
            "mean_post_ttft_s": mean_post_ttft_s,
            "mean_ttft_share_pct": 100.0 * mean_ttft_s / mean_latency_s,
            "mean_post_ttft_share_pct": (
                100.0 * mean_post_ttft_s / mean_latency_s
            ),
        },
        "source_files": {
            "args": str(args_path.relative_to(root)),
            "summary": str(summary_path.relative_to(root)),
            "percentiles": str(percentile_path.relative_to(root)),
            "workload_throughput": str(throughput_path.relative_to(root)),
        },
    }


def collect(
    root: Path,
    *,
    gpu_count: int,
    device: str,
    architecture: str,
    tokenspeed_revision: str,
    model_revision: str,
) -> dict[str, Any]:
    runs = [_run(root, concurrency, gpu_count) for concurrency in (1, 16)]
    return {
        "format": "tokenspeed_real_serving_v1",
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "real Kimi-K3 TP8/EP1 HTTP serving",
        "hardware": {
            "device": device,
            "architecture": architecture,
            "gpu_count": gpu_count,
        },
        "software": {
            "tokenspeed_revision": tokenspeed_revision,
            "model_revision": model_revision,
            "evalscope": "1.9.1",
            "pytorch": "2.11.0+rocm7.2",
            "hip": "7.2.26015",
            "transformers": "5.12.0",
            "triton": "3.6.0",
        },
        "topology": {
            "attention_tp": 8,
            "dense_tp": 8,
            "moe_tp": 8,
            "expert_parallel": 1,
            "physical_ranks": 8,
            "collectives": "RCCL plus Iris all-reduce",
        },
        "server": {
            "checkpoint": "full source safetensors",
            "max_model_len": 8192,
            "max_num_seqs": 16,
            "max_prefill_tokens": 8192,
            "chunked_prefill_size": 8192,
            "decode_cuda_graphs": True,
            "decode_cuda_graph_sizes": [1, 2, 4, 8, 16],
            "kv_cache_dtype": "fp8_e4m3",
            "cache_token_capacity": 3197056,
            "prefix_caching": False,
            "kvstore": False,
            "torch_nccl_blocking_wait": True,
        },
        "workload": {
            "api": "OpenAI completions streaming",
            "dataset": "EvalScope random",
            "prompt_tokens": 4096,
            "output_tokens": 1024,
            "ignore_eos": True,
            "temperature": 0,
            "concurrencies": [1, 16],
            "measurement": "one full-concurrency warmup plus three saturated waves",
        },
        "metric_notes": {
            "output_throughput": (
                "Completed output tokens divided by end-to-end measured wall time; "
                "includes prefill and decode."
            ),
            "evalscope_no_spec_fields": (
                "EvalScope 1.9.1 emitted decoded-tokens-per-iteration and "
                "speculative-acceptance fields even though speculative decoding "
                "was disabled. They are semantically invalid and omitted here."
            ),
        },
        "runs": runs,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--device", default="AMD Instinct MI355X")
    parser.add_argument(
        "--architecture",
        default="gfx950:sramecc+:xnack-",
    )
    parser.add_argument("--tokenspeed-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = collect(
        args.input,
        gpu_count=args.gpu_count,
        device=args.device,
        architecture=args.architecture,
        tokenspeed_revision=args.tokenspeed_revision,
        model_revision=args.model_revision,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
