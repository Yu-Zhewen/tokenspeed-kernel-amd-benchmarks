#!/usr/bin/env python3
"""Collective and compute kernel time of every profiled rank, per window.

Reads a model's `rocprof/` and `profiled_ranks.txt` and prints one line per
rank and window, the format generate_hotspot_report.py reads from
`<model>/rank_balance.txt`:

  TP0 c1/decode comm_ms 244.9 calls 7308 compute_ms 420.7

A collective kernel ends on every rank together, so a rank that reaches it
late spends less time in it than the others. Equal times on every rank mean
nobody waits on a straggler among the profiled ranks.

Usage: rank_balance.py <model result dir>
"""

import csv
import re
import sys
from pathlib import Path

COMMUNICATION = re.compile(r"nccl|rccl|iris|^_device_barrier_kernel$")


def lines(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        yield from csv.DictReader(line.replace("\0", "") for line in f)


def main():
    root = Path(sys.argv[1])
    ranks = {}
    for line in (root / "profiled_ranks.txt").read_text().splitlines():
        _, tag, _, pid = line.split()
        ranks[pid] = tag
    for pid, tag in sorted(ranks.items(), key=lambda item: item[1]):
        windows = []
        for path in root.glob(f"rocprof/*/{pid}_marker_api_trace.csv"):
            for row in lines(path):
                parts = Path(row.get("Function") or row.get("Message") or "").parts
                if len(parts) >= 2 and parts[-1] in ("prefill", "decode"):
                    windows.append((int(row["Start_Timestamp"]), int(row["End_Timestamp"]),
                                    f"{parts[-2]}/{parts[-1]}"))
        totals = {name: [0.0, 0, 0.0] for _, _, name in windows}
        for path in root.glob(f"rocprof/*/{pid}_kernel_trace.csv"):
            for row in lines(path):
                try:
                    begin, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
                except (TypeError, ValueError):
                    continue
                for start, stop, name in windows:
                    if start <= begin <= stop:
                        total = totals[name]
                        if COMMUNICATION.search(row["Kernel_Name"]):
                            total[0] += (end - begin) / 1e6
                            total[1] += 1
                        else:
                            total[2] += (end - begin) / 1e6
                        break
        for name, (comm, calls, compute) in totals.items():
            print(f"{tag} {name} comm_ms {comm:.1f} calls {calls} compute_ms {compute:.1f}")


if __name__ == "__main__":
    main()
