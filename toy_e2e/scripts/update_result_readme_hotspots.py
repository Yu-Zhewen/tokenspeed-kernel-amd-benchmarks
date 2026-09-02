#!/usr/bin/env python3
"""Refresh hotspot tables in a collected result README."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SUMMARY_HEADER = (
    "| C | Stage | Ranks | Forwards | Summed GPU time | Communication | "
    "MoE | KDA | MLA / attention | GEMM / quant | Other |"
)
SUMMARY_SEPARATOR = (
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)
KERNEL_HEADER = (
    "| C | Stage | Order | Exact kernel name | Calls | Total across ranks | "
    "GPU share | Average call |"
)
KERNEL_SEPARATOR = "|---:|---|---:|---|---:|---:|---:|---:|"
CORE_CATEGORIES = {
    "communication",
    "moe",
    "kda_attention",
    "mla_or_attention",
    "gemm_or_quant",
}


def _replace_table(markdown: str, header: str, replacement: str) -> str:
    start = markdown.index(header)
    end = markdown.index("\n\n", start)
    return f"{markdown[:start]}{replacement}{markdown[end:]}"


def _profile_lookup(hotspots: dict) -> dict[tuple[int, str], dict]:
    return {
        (
            int(profile["setting"].removeprefix("c")),
            "prefill" if profile["stage"] == "EXTEND" else "decode",
        ): profile
        for profile in hotspots["profiles"]
    }


def _forward_lookup(manifest: dict) -> dict[tuple[int, str], int]:
    return {
        (int(run["concurrency"]), stage): int(run[stage]["forward_count"])
        for run in manifest["runs"]
        for stage in ("prefill", "decode")
    }


def _summary_table(hotspots: dict, manifest: dict) -> str:
    profiles = _profile_lookup(hotspots)
    forwards = _forward_lookup(manifest)
    rows = [SUMMARY_HEADER, SUMMARY_SEPARATOR]
    for concurrency in (1, 16):
        for stage in ("prefill", "decode"):
            profile = profiles[(concurrency, stage)]
            categories = {
                item["name"]: float(item["gpu_time_pct"])
                for item in profile["top_categories"]
            }
            other = sum(
                share
                for name, share in categories.items()
                if name not in CORE_CATEGORIES
            )
            rows.append(
                "| "
                f"{concurrency} | {stage} | {profile['rank_count']} | "
                f"{forwards[(concurrency, stage)]} | "
                f"{profile['rank_kernel_ms']['total']:,.2f} ms | "
                f"{categories.get('communication', 0.0):.2f}% | "
                f"{categories.get('moe', 0.0):.2f}% | "
                f"{categories.get('kda_attention', 0.0):.2f}% | "
                f"{categories.get('mla_or_attention', 0.0):.2f}% | "
                f"{categories.get('gemm_or_quant', 0.0):.2f}% | "
                f"{other:.2f}% |"
            )
    return "\n".join(rows)


def _kernel_table(csv_dir: Path) -> str:
    rows = [KERNEL_HEADER, KERNEL_SEPARATOR]
    for concurrency in (1, 16):
        for stage, csv_stage in (("prefill", "extend"), ("decode", "decode")):
            with (csv_dir / f"c{concurrency}_{csv_stage}.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                kernels = list(csv.DictReader(handle))[:10]
            if len(kernels) != 10:
                raise ValueError(
                    f"expected 10 kernels for C{concurrency} {stage}, "
                    f"found {len(kernels)}"
                )
            for order, kernel in enumerate(kernels, start=1):
                rows.append(
                    "| "
                    f"{concurrency} | {stage} | {order} | "
                    f"`{kernel['kernel_name']}` | "
                    f"{int(kernel['calls']):,} | "
                    f"{float(kernel['total_ms']):,.2f} ms | "
                    f"{float(kernel['gpu_time_pct']):.2f}% | "
                    f"{float(kernel['avg_us']):,.2f} µs |"
                )
    return "\n".join(rows)


def update_readme(
    *,
    readme: Path,
    hotspots_path: Path,
    csv_dir: Path,
    profile_manifest_path: Path,
    check: bool,
) -> None:
    hotspots = json.loads(hotspots_path.read_text(encoding="utf-8"))
    manifest = json.loads(profile_manifest_path.read_text(encoding="utf-8"))
    original = readme.read_text(encoding="utf-8")
    rendered = _replace_table(
        original,
        SUMMARY_HEADER,
        _summary_table(hotspots, manifest),
    )
    rendered = _replace_table(
        rendered,
        KERNEL_HEADER,
        _kernel_table(csv_dir),
    )
    if check:
        if rendered != original:
            raise SystemExit(f"{readme} hotspot tables are stale")
        return
    readme.write_text(rendered, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--hotspots", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    update_readme(
        readme=args.readme,
        hotspots_path=args.hotspots,
        csv_dir=args.csv_dir,
        profile_manifest_path=args.profile_manifest,
        check=args.check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
