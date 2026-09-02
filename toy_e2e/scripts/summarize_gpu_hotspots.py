#!/usr/bin/env python3
"""Summarize per-rank TokenSpeed GPU traces into serving hotspots."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TRACE_NAMES = (
    (
        "torch",
        re.compile(
            r"^(?P<profile>.+)-TP(?P<rank>\d+)"
            r"(?:-(?P<stage>[A-Z][A-Z0-9_]*))?"
            r"\.trace\.json(?:\.gz)?$"
        ),
    ),
    (
        "proton",
        re.compile(
            r"^(?P<profile>.+)-TP(?P<rank>\d+)"
            r"(?:-(?P<stage>[A-Z][A-Z0-9_]*))?"
            r"\.proton\.chrome_trace(?:\.json)?(?:\.gz)?$"
        ),
    ),
)


def _read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_identity(path: Path) -> tuple[str, re.Match[str]]:
    for source, pattern in _TRACE_NAMES:
        match = pattern.match(path.name)
        if match is not None:
            return source, match
    raise ValueError(f"unrecognized GPU trace filename: {path.name}")


def _category(kernel_name: str) -> str:
    name = kernel_name.lower()
    if any(
        marker in name
        for marker in (
            "nccl",
            "rccl",
            "all_reduce",
            "allreduce",
            "all_gather",
            "allgather",
            "reduce_scatter",
            "reducescatter",
            "iris_",
            "msccl",
            "device_barrier",
        )
    ):
        return "communication"
    if any(
        marker in name
        for marker in (
            "moe",
            "expert",
            "router",
            "routing",
            "moe_softmax_topk",
            "grouped_a16w4",
            "masked_topk",
            "sigmoid_bias_topk_route",
            "scatter_kernel",
            "split_epilogue",
            "situ_",
        )
    ):
        return "moe"
    if any(
        marker in name
        for marker in (
            "kda",
            "gated_delta",
            "gdn_",
            "fla_kda",
            "linear_attention",
            "latent_input",
            "packed_projection",
            "packed_input_projection",
            "state_scan",
            "preprocess_intra",
        )
    ):
        return "kda_attention"
    if any(
        marker in name
        for marker in (
            "mla",
            "attention",
            "attn",
            "flash_fwd",
            "flash_bwd",
            "paged_decode",
            "paged_prefill",
        )
    ):
        return "mla_or_attention"
    if any(
        marker in name
        for marker in (
            "gemm",
            "matmul",
            "rocblas",
            "hipblas",
            "mxfp",
            "mfma",
        )
    ):
        return "gemm_or_quant"
    if "norm" in name:
        return "normalization"
    if "cache" in name:
        return "kv_cache"
    if any(marker in name for marker in ("sampling", "multinomial", "top_p", "top_k")):
        return "sampling"
    if any(
        marker in name
        for marker in (
            "elementwise",
            "pointwise",
            "index_",
            "copy",
            "reduce_kernel",
            "cast",
        )
    ):
        return "elementwise_or_reduction"
    return "other"


def _rank_trace(path: Path) -> dict[str, Any]:
    source, match = _trace_identity(path)
    setting = next(
        (
            part.lower()
            for part in reversed(path.parts)
            if re.fullmatch(r"c\d+", part, flags=re.IGNORECASE)
        ),
        path.parent.name,
    )
    stage = match.group("stage")
    if stage is None:
        normalized_parts = {part.lower() for part in path.parts}
        if normalized_parts & {"prefill", "extend"}:
            stage = "EXTEND"
        elif "decode" in normalized_parts:
            stage = "DECODE"
        else:
            stage = "ALL"
    report = _read_json(path)
    kernels: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"calls": 0, "total_ms": 0.0}
    )
    categories: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"calls": 0, "total_ms": 0.0}
    )
    for event in report.get("traceEvents", []):
        if (
            event.get("ph") != "X"
            or event.get("cat") != "kernel"
            or not isinstance(event.get("dur"), (int, float))
        ):
            continue
        name = str(event.get("name", "<unnamed>"))
        # Chrome trace timestamps and durations are expressed in microseconds.
        # displayTimeUnit controls only the viewer's presentation.
        elapsed_ms = float(event["dur"]) / 1e3
        kernels[name]["calls"] = int(kernels[name]["calls"]) + 1
        kernels[name]["total_ms"] = float(kernels[name]["total_ms"]) + elapsed_ms
        category = _category(name)
        categories[category]["calls"] = int(categories[category]["calls"]) + 1
        categories[category]["total_ms"] = (
            float(categories[category]["total_ms"]) + elapsed_ms
        )
    return {
        "source": source,
        "setting": setting,
        "profile_id": match.group("profile"),
        "rank": int(match.group("rank")),
        "stage": stage,
        "path": str(path),
        "kernel_calls": sum(int(item["calls"]) for item in kernels.values()),
        "kernel_ms": sum(float(item["total_ms"]) for item in kernels.values()),
        "kernels": dict(kernels),
        "categories": dict(categories),
    }


def _aggregate_items(
    ranks: list[dict[str, Any]],
    field: str,
    *,
    top_k: int | None,
) -> list[dict[str, Any]]:
    labels = sorted({label for rank in ranks for label in rank[field]})
    all_rank_kernel_ms = sum(float(rank["kernel_ms"]) for rank in ranks)
    result = []
    for label in labels:
        rank_ms = [
            float(rank[field].get(label, {}).get("total_ms", 0.0)) for rank in ranks
        ]
        rank_calls = [
            int(rank[field].get(label, {}).get("calls", 0)) for rank in ranks
        ]
        mean_ms = statistics.fmean(rank_ms)
        total_ms = sum(rank_ms)
        calls = sum(rank_calls)
        result.append(
            {
                "name": label,
                "calls": calls,
                "total_ms": total_ms,
                "gpu_time_pct": (
                    100.0 * total_ms / all_rank_kernel_ms
                    if all_rank_kernel_ms
                    else 0.0
                ),
                "avg_us": 1e3 * total_ms / calls if calls else 0.0,
                "rank_mean_ms": mean_ms,
                "rank_min_ms": min(rank_ms),
                "rank_max_ms": max(rank_ms),
                "rank_imbalance_pct": (
                    100.0 * (max(rank_ms) - min(rank_ms)) / mean_ms
                    if mean_ms
                    else 0.0
                ),
                "calls_per_rank_mean": statistics.fmean(rank_calls),
            }
        )
    result.sort(key=lambda item: (-item["total_ms"], item["name"]))
    return result if top_k is None else result[:top_k]


def summarize(trace_paths: list[Path], *, top_k: int) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not trace_paths:
        raise ValueError("at least one trace path is required")
    rank_traces = [_rank_trace(path) for path in trace_paths]
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in rank_traces:
        key = (
            trace["source"],
            trace["setting"],
            trace["profile_id"],
            trace["stage"],
        )
        groups[key].append(trace)

    profiles = []
    for (source, setting, profile_id, stage), ranks in sorted(groups.items()):
        ranks.sort(key=lambda item: item["rank"])
        rank_kernel_ms = [float(rank["kernel_ms"]) for rank in ranks]
        mean_kernel_ms = statistics.fmean(rank_kernel_ms)
        all_categories = _aggregate_items(ranks, "categories", top_k=None)
        all_kernels = _aggregate_items(ranks, "kernels", top_k=None)
        profiles.append(
            {
                "source": source,
                "setting": setting,
                "profile_id": profile_id,
                "stage": stage,
                "rank_count": len(ranks),
                "rank_kernel_ms": {
                    "total": sum(rank_kernel_ms),
                    "min": min(rank_kernel_ms),
                    "mean": mean_kernel_ms,
                    "max": max(rank_kernel_ms),
                    "imbalance_pct": (
                        100.0
                        * (max(rank_kernel_ms) - min(rank_kernel_ms))
                        / mean_kernel_ms
                        if mean_kernel_ms
                        else 0.0
                    ),
                },
                "kernel_calls_per_rank": {
                    "min": min(rank["kernel_calls"] for rank in ranks),
                    "mean": statistics.fmean(
                        rank["kernel_calls"] for rank in ranks
                    ),
                    "max": max(rank["kernel_calls"] for rank in ranks),
                },
                "top_categories": all_categories[:top_k],
                "top_kernels": all_kernels[:top_k],
                "all_kernels": all_kernels,
                "ranks": [
                    {
                        "rank": rank["rank"],
                        "kernel_calls": rank["kernel_calls"],
                        "kernel_ms": rank["kernel_ms"],
                        "trace": rank["path"],
                    }
                    for rank in ranks
                ],
            }
        )
    return {
        "format": "tokenspeed_gpu_hotspots_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trace_count": len(rank_traces),
        "profiles": profiles,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--csv-dir",
        type=Path,
        help="write one complete exact-name kernel CSV per setting and stage",
    )
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


def _write_csvs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["kernel_name", "calls", "total_ms", "gpu_time_pct", "avg_us"]
    for profile in result["profiles"]:
        path = output_dir / (
            f"{profile['setting']}_{str(profile['stage']).lower()}.csv"
        )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for kernel in profile["all_kernels"]:
                writer.writerow(
                    {
                        "kernel_name": kernel["name"],
                        "calls": kernel["calls"],
                        "total_ms": f"{kernel['total_ms']:.6f}",
                        "gpu_time_pct": f"{kernel['gpu_time_pct']:.6f}",
                        "avg_us": f"{kernel['avg_us']:.6f}",
                    }
                )


def main() -> int:
    args = _parse_args()
    candidates = {
        *args.input.rglob("*.trace.json"),
        *args.input.rglob("*.trace.json.gz"),
        *args.input.rglob("*.proton.chrome_trace"),
        *args.input.rglob("*.proton.chrome_trace.json"),
        *args.input.rglob("*.proton.chrome_trace.gz"),
    }
    trace_paths = sorted(candidates)
    if not trace_paths:
        raise FileNotFoundError(f"no GPU Chrome traces found below {args.input}")
    result = summarize(trace_paths, top_k=args.top_k)
    if args.csv_dir is not None:
        _write_csvs(result, args.csv_dir)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
