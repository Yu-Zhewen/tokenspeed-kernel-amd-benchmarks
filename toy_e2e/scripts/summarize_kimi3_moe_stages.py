#!/usr/bin/env python3
"""Attribute profiled GPU kernels to named Kimi-K3 MoE operation scopes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_SCOPE_PREFIX = "tokenspeed::kimi3_moe."


def _interval_union_us(events: list[dict[str, Any]]) -> float:
    intervals = sorted(
        (float(event["ts"]), float(event["ts"]) + float(event["dur"]))
        for event in events
        if float(event.get("dur", 0.0)) > 0.0
    )
    if not intervals:
        return 0.0
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def _scope_for_timed_events(
    timed_events: list[dict[str, Any]],
    scopes: list[dict[str, Any]],
    *,
    id_argument: str,
) -> dict[int, str]:
    """Map event IDs to their innermost named operation scope."""

    scopes_by_thread: dict[tuple[object, object], list[dict[str, Any]]] = defaultdict(
        list
    )
    events_by_thread: dict[tuple[object, object], list[dict[str, Any]]] = defaultdict(
        list
    )
    for scope in scopes:
        scopes_by_thread[(scope.get("pid"), scope.get("tid"))].append(scope)
    for event in timed_events:
        events_by_thread[(event.get("pid"), event.get("tid"))].append(event)

    external_id_to_scope: dict[int, str] = {}
    for thread, events in events_by_thread.items():
        thread_scopes = sorted(
            scopes_by_thread.get(thread, ()), key=lambda event: float(event["ts"])
        )
        events.sort(key=lambda event: float(event["ts"]))
        active: list[dict[str, Any]] = []
        next_scope = 0
        for event in events:
            timestamp = float(event["ts"])
            while (
                next_scope < len(thread_scopes)
                and float(thread_scopes[next_scope]["ts"]) <= timestamp
            ):
                active.append(thread_scopes[next_scope])
                next_scope += 1
            active = [
                scope
                for scope in active
                if float(scope["ts"]) + float(scope["dur"]) >= timestamp
            ]
            containing = [
                scope
                for scope in active
                if timestamp <= float(scope["ts"]) + float(scope["dur"])
            ]
            if not containing:
                continue
            innermost = min(containing, key=lambda scope: float(scope["dur"]))
            event_id = event.get("args", {}).get(id_argument)
            if event_id is not None:
                external_id_to_scope[int(event_id)] = str(innermost["name"])[
                    len(_SCOPE_PREFIX) :
                ]
    return external_id_to_scope


def summarize_trace(
    path: Path,
    *,
    moe_layers: int,
    forward_count: int | None = None,
) -> dict[str, Any]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    events = trace.get("traceEvents", ())
    cpu_events = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") == "cpu_op"
    ]
    scopes = [
        event
        for event in events
        if event.get("ph") == "X"
        and str(event.get("name", "")).startswith(_SCOPE_PREFIX)
    ]
    kernels = [
        event
        for event in events
        if event.get("ph") == "X"
        and str(event.get("cat", "")).lower() in {"kernel", "gpu_kernel"}
    ]
    runtime_launches = [
        event
        for event in events
        if event.get("ph") == "X"
        and str(event.get("cat", "")).lower()
        in {"cuda_runtime", "hip_runtime"}
        and event.get("args", {}).get("correlation") is not None
        and (
            event.get("args", {}).get("kernel") is not None
            or "launchkernel" in str(event.get("name", "")).lower()
        )
    ]
    external_id_to_scope = _scope_for_timed_events(
        cpu_events,
        scopes,
        id_argument="External id",
    )
    correlation_to_scope = _scope_for_timed_events(
        runtime_launches,
        scopes,
        id_argument="correlation",
    )

    stage_kernels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for kernel in kernels:
        args = kernel.get("args", {})
        correlation = args.get("correlation")
        stage = (
            correlation_to_scope.get(int(correlation))
            if correlation is not None
            else None
        )
        external_id = args.get("External id")
        if stage is None and external_id is not None:
            stage = external_id_to_scope.get(int(external_id))
        if stage is not None:
            stage_kernels[stage].append(kernel)

    scope_calls = Counter(
        str(scope["name"])[len(_SCOPE_PREFIX) :] for scope in scopes
    )
    complete_stage_calls = [
        count for stage, count in scope_calls.items() if stage == "router"
    ]
    forwards = forward_count
    if forwards is None:
        forwards = (
            complete_stage_calls[0] / moe_layers
            if complete_stage_calls and moe_layers > 0
            else None
        )
    stages = []
    for stage in sorted(scope_calls):
        assigned = stage_kernels.get(stage, [])
        kernel_sum_us = sum(float(event.get("dur", 0.0)) for event in assigned)
        kernel_union_us = _interval_union_us(assigned)
        calls = scope_calls[stage]
        name_counts = Counter(str(event.get("name", "")) for event in assigned)
        name_time = Counter()
        stream_time = Counter()
        for event in assigned:
            duration = float(event.get("dur", 0.0))
            name_time[str(event.get("name", ""))] += duration
            stream_time[str(event.get("args", {}).get("stream", "unknown"))] += duration
        stages.append(
            {
                "stage": stage,
                "scope_calls": calls,
                "kernel_calls": len(assigned),
                "kernel_sum_ms": kernel_sum_us / 1e3,
                "kernel_union_ms": kernel_union_us / 1e3,
                "kernel_overlap_ms": max(kernel_sum_us - kernel_union_us, 0.0) / 1e3,
                "kernel_ms_per_scope": (
                    kernel_sum_us / calls / 1e3 if calls else None
                ),
                "kernel_ms_per_forward": (
                    kernel_sum_us / forwards / 1e3 if forwards else None
                ),
                "streams_ms": {
                    stream: duration / 1e3
                    for stream, duration in sorted(stream_time.items())
                },
                "top_kernels": [
                    {
                        "name": name,
                        "calls": name_counts[name],
                        "total_ms": duration / 1e3,
                    }
                    for name, duration in name_time.most_common(10)
                ],
            }
        )

    scoped_kernels = [
        kernel for assigned in stage_kernels.values() for kernel in assigned
    ]
    scoped_sum_us = sum(float(event.get("dur", 0.0)) for event in scoped_kernels)
    scoped_union_us = _interval_union_us(scoped_kernels)
    return {
        "trace": str(path),
        "moe_layers_per_forward": moe_layers,
        "inferred_forwards": forwards,
        "scope_calls": dict(sorted(scope_calls.items())),
        "scoped_kernel_calls": len(scoped_kernels),
        "scoped_kernel_sum_ms": scoped_sum_us / 1e3,
        "scoped_kernel_union_ms": scoped_union_us / 1e3,
        "scoped_kernel_overlap_ms": max(scoped_sum_us - scoped_union_us, 0.0) / 1e3,
        "stages": stages,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--moe-layers", type=int, default=92)
    parser.add_argument(
        "--profile-manifest",
        type=Path,
        help="use exact per-stage forward counts from the profile manifest",
    )
    return parser.parse_args()


def _manifest_forward_counts(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    counts = {}
    for run in manifest["runs"]:
        for phase in ("prefill", "decode"):
            stage = run[phase]
            trace = stage.get("trace")
            if trace is not None:
                counts[Path(trace).name] = int(stage["forward_count"])
    return counts


def main() -> int:
    args = _parse_args()
    if args.moe_layers <= 0:
        raise ValueError("--moe-layers must be positive")
    paths = (
        [args.input]
        if args.input.is_file()
        else sorted(
            path
            for path in args.input.rglob("*.trace.json")
            if not path.name.endswith(".trace.shapes.json")
        )
    )
    forward_counts = _manifest_forward_counts(args.profile_manifest)
    result = {
        "format": "tokenspeed_kimi3_moe_stage_attribution_v2",
        "profiles": [
            summarize_trace(
                path,
                moe_layers=args.moe_layers,
                forward_count=forward_counts.get(path.name),
            )
            for path in paths
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
