#!/usr/bin/env python3
"""Aggregate exact MLA, KDA, and AttnRes shapes captured per model stage."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _component(family: str, mode: str) -> str | None:
    if family == "attn_res" and mode == "fwd":
        return "attn_res"
    if family != "attention":
        return None
    if mode.startswith("kda_"):
        return "kda"
    return {
        "mla_normalize_project_query": "mla.normalize_query",
        "mla_prefill": "mla.prefill",
        "mla_extend_with_kvcache": "mla.extend",
        "mla_decode_with_kvcache": "mla.decode",
        "mla_decode_projected_value": "mla.decode_projected_value",
        "mla_project_value": "mla.value_projection_gate",
    }.get(mode)


def _resolve_shapes(
    manifest_dir: Path,
    recorded_path: str,
    *,
    concurrency: int,
    phase: str,
) -> Path:
    path = Path(recorded_path)
    if path.is_file():
        return path
    relocated = manifest_dir / f"c{concurrency}" / phase / path.name
    if relocated.is_file():
        return relocated
    raise FileNotFoundError(f"shape capture not found: {path} or {relocated}")


def _shape_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        record["family"],
        record["mode"],
        record["kernel_name"],
        record["dtype"],
        json.dumps(record["shape_params"], sort_keys=True, separators=(",", ":")),
    )


def summarize(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profiles = []
    for run in manifest["runs"]:
        concurrency = int(run["concurrency"])
        for phase in ("prefill", "decode"):
            stage = run[phase]
            forward_count = int(stage["forward_count"])
            if forward_count == 0 or stage.get("shapes") is None:
                continue
            shapes_path = _resolve_shapes(
                manifest_path.parent,
                stage["shapes"],
                concurrency=concurrency,
                phase=phase,
            )
            records = json.loads(shapes_path.read_text(encoding="utf-8"))
            counts = Counter(
                _shape_key(record)
                for record in records
                if _component(record["family"], record["mode"]) is not None
            )
            component_calls: dict[str, int] = defaultdict(int)
            calls = []
            for key, count in sorted(counts.items()):
                family, mode, kernel_name, dtype, encoded_shape = key
                component = _component(family, mode)
                assert component is not None
                component_calls[component] += count
                calls.append(
                    {
                        "component": component,
                        "family": family,
                        "mode": mode,
                        "kernel_name": kernel_name,
                        "dtype": dtype,
                        "shape": json.loads(encoded_shape),
                        "calls": count,
                        "calls_per_forward": count / forward_count,
                    }
                )
            profiles.append(
                {
                    "name": f"c{concurrency}_{phase}",
                    "concurrency": concurrency,
                    "phase": phase,
                    "forward_count": forward_count,
                    "shapes": str(shapes_path),
                    "components": {
                        component: {
                            "calls": count,
                            "calls_per_forward": count / forward_count,
                        }
                        for component, count in sorted(component_calls.items())
                    },
                    "fallbacks": [
                        call
                        for call in calls
                        if "fallback" in call["kernel_name"].lower()
                    ],
                    "calls": calls,
                }
            )
    return {
        "format": "tokenspeed_attention_shape_attribution_v1",
        "profile_manifest": str(manifest_path),
        "profiles": profiles,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = summarize(args.profile_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
