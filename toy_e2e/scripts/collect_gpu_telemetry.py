#!/usr/bin/env python3
"""Sample AMD GPU clocks, power, temperature, utilization, and throttle data."""

from __future__ import annotations

import argparse
import json
import signal
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _integer(path: Path) -> int | None:
    value = _read(path)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _labeled_values(
    hwmon: Path,
    prefix: str,
    suffix: str,
    *,
    divisor: float,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for value_path in sorted(hwmon.glob(f"{prefix}*_{suffix}")):
        index = value_path.name[len(prefix) :].split("_", 1)[0]
        label = _read(hwmon / f"{prefix}{index}_label") or value_path.stem
        value = _integer(value_path)
        if value is not None:
            result[label] = value / divisor
    return result


def _summaries(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = sorted(
        {
            key
            for sample in samples
            for key, value in sample.items()
            if isinstance(value, (int, float)) and key != "monotonic_s"
        }
    )
    return {
        key: {
            "min": min(values),
            "mean": statistics.fmean(values),
            "p50": statistics.median(values),
            "max": max(values),
        }
        for key in keys
        if (values := [float(sample[key]) for sample in samples if key in sample])
    }


def _select_card(requested: str | None) -> Path:
    if requested is not None:
        card = Path("/sys/class/drm") / requested
        if _read(card / "device/uevent") is None:
            raise FileNotFoundError(f"DRM card does not exist: {card}")
        return card
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        if "DRIVER=amdgpu" in (_read(card / "device/uevent") or ""):
            return card
    raise FileNotFoundError("no AMD GPU DRM card found")


def collect(card: Path, interval_s: float, output: Path) -> None:
    device = card / "device"
    hwmons = sorted((device / "hwmon").glob("hwmon*"))
    if len(hwmons) != 1:
        raise RuntimeError(f"expected one hwmon under {device}, got {hwmons}")
    hwmon = hwmons[0]
    samples: list[dict[str, Any]] = []
    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    while running:
        sample: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "monotonic_s": time.monotonic() - started,
        }
        for key, path in (
            ("gpu_busy_pct", device / "gpu_busy_percent"),
            ("memory_busy_pct", device / "mem_busy_percent"),
        ):
            value = _integer(path)
            if value is not None:
                sample[key] = value
        for label, value in _labeled_values(
            hwmon, "freq", "input", divisor=1.0
        ).items():
            sample[f"{label}_hz"] = value
        for label, value in _labeled_values(
            hwmon, "power", "input", divisor=1e6
        ).items():
            sample[f"{label}_w"] = value
        for label, value in _labeled_values(
            hwmon, "power", "cap", divisor=1e6
        ).items():
            sample[f"{label}_cap_w"] = value
        for label, value in _labeled_values(
            hwmon, "temp", "input", divisor=1e3
        ).items():
            sample[f"{label}_c"] = value
        samples.append(sample)
        time.sleep(interval_s)

    result = {
        "format": "tokenspeed_gpu_telemetry_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "card": card.name,
        "pci_slot": next(
            (
                line.removeprefix("PCI_SLOT_NAME=")
                for line in (_read(device / "uevent") or "").splitlines()
                if line.startswith("PCI_SLOT_NAME=")
            ),
            "unavailable",
        ),
        "sample_interval_s": interval_s,
        "sample_count": len(samples),
        "power_dpm_force_performance_level": _read(
            device / "power_dpm_force_performance_level"
        )
        or "unavailable",
        "thermal_throttling_logging": _read(
            device / "thermal_throttling_logging"
        )
        or "unavailable",
        "direct_throttle_state": _read(device / "throttle_status")
        or "unavailable",
        "summaries": _summaries(samples),
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", help="DRM card name, for example card1")
    parser.add_argument("--interval-s", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.interval_s <= 0:
        raise ValueError("--interval-s must be positive")
    collect(_select_card(args.card), args.interval_s, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
