#!/usr/bin/env python3
"""Split rocprofv3 kernel records into one Chrome trace per profile window.

Windows are the roctx ranges the profiler hook names after each window's
output directory (``.../c{C}/{prefill,decode}``).
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path


def rows(root, suffix):
    for path in sorted(root.rglob(f"*{suffix}")):
        # rocprofv3 can write the bytes behind a stale name pointer, NULs included.
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            yield from csv.DictReader(line.replace("\0", "") for line in f)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    windows = []
    for row in rows(src, "marker_api_trace.csv"):
        message = row.get("Function") or row.get("Message") or ""
        parts = Path(message).parts
        if len(parts) >= 2 and parts[-1] in ("prefill", "decode") and parts[-2].startswith("c"):
            windows.append((int(row["Start_Timestamp"]), int(row["End_Timestamp"]), parts[-2], parts[-1]))
    kernels, unreadable, skipped = [], Counter(), 0
    for r in rows(src, "kernel_trace.csv"):
        try:
            begin, finish = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        except (TypeError, ValueError):
            skipped += 1
            continue
        name = r["Kernel_Name"]
        if "\ufffd" in name:
            unreadable[r["Kernel_Id"]] += 1
            name = f"<unreadable name, kernel id {r['Kernel_Id']}>"
        kernels.append((begin, finish, name, r.get("Queue_Id", "0")))
    print(f"{len(kernels)} kernel records, {len(windows)} windows")
    for kernel_id, count in sorted(unreadable.items()):
        print(f"warning: {count} records of kernel id {kernel_id} have an unreadable name")
    if skipped:
        print(f"warning: skipped {skipped} kernel records without valid timestamps")
    for start, end, setting, stage_dir in windows:
        stage = "EXTEND" if stage_dir == "prefill" else "DECODE"
        events = [
            {
                "ph": "X",
                "cat": "kernel",
                "name": name,
                "ts": begin / 1e3,
                "dur": (finish - begin) / 1e3,
                "pid": 0,
                "tid": int(queue or 0),
            }
            for begin, finish, name, queue in kernels
            if start <= begin <= end
        ]
        out = dst / setting / stage_dir / f"serve-{setting}-TP0-{stage}.trace.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"traceEvents": events}))
        print(f"{setting} {stage}: {len(events)} kernels over {(end - start) / 1e6:.1f} ms")


if __name__ == "__main__":
    main()
