#!/usr/bin/env python3
"""Aggregate exact GEMM shapes captured beside logical-rank traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize(paths: list[Path]) -> dict[str, Any]:
    counts: Counter[tuple[str, str, str, int, int, int]] = Counter()
    for path in paths:
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            if record.get("family") != "gemm":
                continue
            shape = record.get("shape_params", {})
            if not {"M", "N", "K"} <= shape.keys():
                continue
            key = (
                str(record.get("kernel_name", "unknown")),
                str(record.get("mode", "unknown")),
                str(record.get("dtype", "unknown")),
                int(shape["M"]),
                int(shape["N"]),
                int(shape["K"]),
            )
            counts[key] += 1

    shapes = []
    for (kernel, mode, dtype, m, n, k), calls in counts.items():
        shapes.append(
            {
                "kernel_name": kernel,
                "mode": mode,
                "dtype": dtype,
                "shape": {"M": m, "N": n, "K": k},
                "calls": calls,
                "total_flops": 2 * m * n * k * calls,
            }
        )
    shapes.sort(key=lambda row: (-row["total_flops"], -row["calls"]))
    return {
        "format": "tokenspeed_gemm_shape_summary_v1",
        "input_files": [str(path) for path in paths],
        "shapes": shapes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = (
        [args.input]
        if args.input.is_file()
        else sorted(args.input.rglob("*.trace.shapes.json"))
    )
    result = summarize(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
