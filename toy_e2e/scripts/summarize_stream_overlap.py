#!/usr/bin/env python3
"""Separate cross-stream overlap from suspicious same-stream trace intervals."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from toy_e2e.scripts.summarize_gpu_hotspots import _category


def summarize_trace(path: Path, *, top_k: int) -> dict[str, Any]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    kernels = sorted(
        (
            event
            for event in trace.get("traceEvents", ())
            if event.get("ph") == "X"
            and str(event.get("cat", "")).lower() in {"kernel", "gpu_kernel"}
            and float(event.get("dur", 0.0)) > 0.0
        ),
        key=lambda event: float(event["ts"]),
    )
    cross_category_pairs: Counter[tuple[str, str]] = Counter()
    cross_kernel_pairs: Counter[tuple[str, str]] = Counter()
    same_category_pairs: Counter[tuple[str, str]] = Counter()
    same_kernel_pairs: Counter[tuple[str, str]] = Counter()
    stream_pairs: Counter[tuple[str, str]] = Counter()
    active: list[dict[str, Any]] = []
    for event in kernels:
        start = float(event["ts"])
        end = start + float(event["dur"])
        stream = str(event.get("args", {}).get("stream", "unknown"))
        active = [
            other
            for other in active
            if float(other["ts"]) + float(other["dur"]) > start
        ]
        for other in active:
            other_stream = str(other.get("args", {}).get("stream", "unknown"))
            overlap = min(
                end,
                float(other["ts"]) + float(other["dur"]),
            ) - start
            if overlap <= 0.0:
                continue
            name_pair = tuple(
                sorted((str(other.get("name", "")), str(event.get("name", ""))))
            )
            category_pair = tuple(
                sorted((_category(name_pair[0]), _category(name_pair[1])))
            )
            stream_pair = tuple(sorted((other_stream, stream)))
            if other_stream == stream:
                same_kernel_pairs[name_pair] += overlap
                same_category_pairs[category_pair] += overlap
            else:
                cross_kernel_pairs[name_pair] += overlap
                cross_category_pairs[category_pair] += overlap
                stream_pairs[stream_pair] += overlap
        active.append(event)

    def rows(counter: Counter[tuple[str, str]]) -> list[dict[str, Any]]:
        return [
            {"pair": list(pair), "overlap_ms": duration / 1e3}
            for pair, duration in counter.most_common(top_k)
        ]

    return {
        "trace": str(path),
        "kernel_count": len(kernels),
        "cross_stream_pairwise_overlap_ms": sum(cross_kernel_pairs.values()) / 1e3,
        "same_stream_pairwise_overlap_ms": sum(same_kernel_pairs.values()) / 1e3,
        "stream_pairs": rows(stream_pairs),
        "cross_stream_category_pairs": rows(cross_category_pairs),
        "cross_stream_kernel_pairs": rows(cross_kernel_pairs),
        "same_stream_category_pairs": rows(same_category_pairs),
        "same_stream_kernel_pairs": rows(same_kernel_pairs),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = {
        "format": "tokenspeed_stream_overlap_attribution_v2",
        "profiles": [
            summarize_trace(path, top_k=args.top_k)
            for path in (
                [args.input]
                if args.input.is_file()
                else sorted(
                    path
                    for path in args.input.rglob("*.trace.json")
                    if not path.name.endswith(".trace.shapes.json")
                )
            )
        ],
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
