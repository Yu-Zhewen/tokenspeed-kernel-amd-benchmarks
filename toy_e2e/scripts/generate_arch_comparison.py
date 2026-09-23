#!/usr/bin/env python3
"""Turn one commit's gfx950 and gfx1250 runs into a single comparison document.

Takes the `performance` result.json and the `hotspots` hotspots.json from each
architecture, all four collected at the same TokenSpeed commit, and writes a
result directory holding:

  README.md                     the comparison document
  kernel-tables/<arch>/*.csv    every kernel symbol per stage, not just the top

Everything in the document is derived from those four files, so regenerating it
from the same inputs reproduces it byte for byte.

Kernel names carry an architecture suffix and the two architectures split some
work differently, so the document compares functional groups rather than
individual symbols. `CATEGORIES` below is that mapping, and it is also the
`Category` column of the kernel tables, so the two always agree.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# (category, gfx950 pattern, gfx1250 pattern). First match wins, so order
# matters: narrower categories come before the buckets that would swallow them.
CATEGORIES = [
    ("KDA state scan", r"state_scan", r"state_scan"),
    (
        "MoE",
        r"mxfp4_moe|gather_package|warp_decode|moe_partial_reduce"
        r"|dynamic_fp8_single_pass|sigmoid_bias_topk|sigmoid_mul|_situ_kernel",
        r"^_matmul\.|^_matmul_decode\.|topk_route|weighted_topk_reduce"
        r"|sigmoid_bias_topk|sigmoid_mul|_situ_kernel",
    ),
    (
        "dense GEMM",
        r"mm_a16w16|^Cijk_|^Custom_Cijk_",
        r"wmma_tdm_dense|^Cijk_|^Custom_Cijk_|rowcta_gemv|smallm",
    ),
    (
        "input projections",
        r"packed_input_projections|latent_input",
        r"packed_input_projections|latent_input",
    ),
    ("MLA attention", r"mla_prefill|mla_decode", r"mla_prefill|mla_decode"),
    ("AttnRes", r"attn_res", r"attn_res"),
    (
        "KDA other",
        r"kda_paged_prefill_preprocess|kda_paged_prefill_wu_vector"
        r"|kda_paged_prefill_solve_merge|kda_paged_prefill_gfx|kda_fused"
        r"|causal_conv1d",
        r"kda_paged_prefill_preprocess|kda_paged_prefill_wu_vector"
        r"|kda_paged_prefill_solve_merge|kda_paged_prefill_gfx|kda_fused"
        r"|causal_conv1d",
    ),
    ("add3", r"_add3_kernel", r"_add3_kernel|wmma_tdm_add3"),
    ("rmsnorm", r"rmsnorm|layernorm", r"rmsnorm|layernorm"),
    ("elementwise", r"elementwise|CatArrayBatched|copyBuffer",
     r"elementwise|CatArrayBatched|copyBuffer"),
]

STAGES = [("EXTEND", "c16"), ("EXTEND", "c1"), ("DECODE", "c16"), ("DECODE", "c1")]
STAGE_LABEL = {
    ("EXTEND", "c16"): "prefill c16",
    ("EXTEND", "c1"): "prefill c1",
    ("DECODE", "c16"): "decode c16",
    ("DECODE", "c1"): "decode c1",
}
# The ratio the strongest categories already reach, used as the bar for
# reporting how much the weaker ones still have to give.
TARGET_RATIO = 1.5

ARCHES = [("gfx950", "MI355X"), ("gfx1250", "MI455X")]


def categorize(name: str, arch_index: int) -> str:
    for category, *patterns in CATEGORIES:
        if re.search(patterns[arch_index], name):
            return category
    return "other"


def read_hotspots(path: Path, arch_index: int):
    """Return {(stage, setting): [{name, ms, calls, category}, ...]}."""
    stages = {}
    for prof in json.loads(path.read_text())["profiles"]:
        key = (prof["stage"], str(prof.get("setting")))
        rows = [
            {
                "name": k.get("name", ""),
                "ms": k.get("total_ms", 0.0),
                "calls": k.get("calls", 0),
                "category": categorize(k.get("name", ""), arch_index),
            }
            for k in prof["all_kernels"]
        ]
        rows.sort(key=lambda r: -r["ms"])
        stages[key] = rows
    return stages


def read_performance(path: Path):
    """Return {concurrency: benchmark dict}."""
    doc = json.loads(path.read_text())
    return doc, {r["benchmark"]["concurrency"]: r["benchmark"] for r in doc["runs"]}


def fmt(value, digits=1):
    return f"{value:,.{digits}f}"


def collected_dates(paths):
    """Return the sorted set of UTC dates the inputs were generated on."""
    dates = set()
    for path in paths:
        stamp = json.loads(path.read_text()).get("generated_at")
        if stamp:
            dates.add(stamp[:10])
    return sorted(dates)


PERF_METRICS = [
    ("prefill TTFT p50 (ms)", lambda b: b["time_to_first_token_ms"]["p50"], "lower"),
    ("prefill step p50 (ms)", lambda b: b["step_wall_ms"]["prefill"]["p50"], "lower"),
    ("decode step p50 (ms)", lambda b: b["steady_decode_step_ms"]["p50"], "lower"),
    ("decode tok/s per user", lambda b: b["per_user_decode_tps"], "higher"),
    ("aggregate output tok/s", lambda b: b["aggregate_output_tps"], "higher"),
]


def perf_table(perf950, perf1250) -> list[str]:
    out = [
        "| Metric | Batch | MI355X (gfx950) | MI455X (gfx1250) | MI455X advantage |",
        "|---|---:|---:|---:|---:|",
    ]
    for conc in sorted(set(perf950) & set(perf1250)):
        for label, get, better in PERF_METRICS:
            a, b = get(perf950[conc]), get(perf1250[conc])
            ratio = a / b if better == "lower" else b / a
            out.append(
                f"| {label} | {conc} | {fmt(a)} | {fmt(b)} | {ratio:.2f}x |"
            )
    return out


def category_totals(stages, key):
    totals = {}
    for row in stages.get(key, []):
        entry = totals.setdefault(row["category"], [0.0, 0])
        entry[0] += row["ms"]
        entry[1] += row["calls"]
    return totals


def breakdown_tables(hs950, hs1250) -> tuple[list[str], list[str]]:
    per_stage = {
        key: (category_totals(hs950, key), category_totals(hs1250, key))
        for key in STAGES
    }
    categories = {c for a, b in per_stage.values() for c in set(a) | set(b)}

    def rank_key(category):
        total = 0.0
        for a, b in per_stage.values():
            ms_a = a.get(category, [0.0, 0])[0]
            ms_b = b.get(category, [0.0, 0])[0]
            if ms_a:
                total += ms_b - ms_a / TARGET_RATIO
        return total

    order = sorted(categories, key=rank_key, reverse=True)
    header = "| Category | " + " | ".join(
        STAGE_LABEL[s] for s in STAGES
    ) + " |"
    align = "|---|" + "---:|" * len(STAGES)

    ratios = [header, align]
    headroom = [header.replace("Category", "Category"), align]
    for category in order:
        rcells, hcells = [], []
        for key in STAGES:
            a, b = per_stage[key]
            ms_a = a.get(category, [0.0, 0])[0]
            ms_b = b.get(category, [0.0, 0])[0]
            rcells.append(f"{ms_a / ms_b:.2f}x" if ms_a and ms_b else "—")
            if ms_a and ms_b:
                h = ms_b - ms_a / TARGET_RATIO
                hcells.append(f"{h:,.0f}")
            else:
                hcells.append("—")
        ratios.append(f"| {category} | " + " | ".join(rcells) + " |")
        headroom.append(f"| {category} | " + " | ".join(hcells) + " |")
    return ratios, headroom


def kernel_table(rows, top) -> list[str]:
    total = sum(r["ms"] for r in rows) or 1.0
    out = [
        "| Rank | Category | Exact kernel name | GPU duration (ms) | Calls "
        "| Mean/call (us) | Stage GPU time |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows[:top], 1):
        per_call = r["ms"] / r["calls"] * 1000 if r["calls"] else 0.0
        out.append(
            f"| {i} | {r['category']} | `{r['name']}` | {fmt(r['ms'], 3)} "
            f"| {r['calls']:,} | {fmt(per_call, 1)} | {r['ms'] / total * 100:.2f}% |"
        )
    return out


def write_csvs(out_dir: Path, arch: str, stages) -> None:
    dest = out_dir / "kernel-tables" / arch
    dest.mkdir(parents=True, exist_ok=True)
    for (stage, setting), rows in stages.items():
        name = f"{setting}-{'prefill' if stage == 'EXTEND' else 'decode'}.csv"
        total = sum(r["ms"] for r in rows) or 1.0
        with (dest / name).open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["category", "kernel_name", "calls", "total_ms", "avg_us",
                 "stage_gpu_time_pct"]
            )
            for r in rows:
                avg = r["ms"] / r["calls"] * 1000 if r["calls"] else 0.0
                writer.writerow([
                    r["category"], r["name"], r["calls"], f"{r['ms']:.3f}",
                    f"{avg:.3f}", f"{r['ms'] / total * 100:.4f}",
                ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", required=True, help="TokenSpeed commit under test")
    ap.add_argument("--gfx950-performance", required=True, type=Path)
    ap.add_argument("--gfx950-hotspots", required=True, type=Path)
    ap.add_argument("--gfx1250-performance", required=True, type=Path)
    ap.add_argument("--gfx1250-hotspots", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--top-kernels", type=int, default=20)
    args = ap.parse_args()

    doc950, perf950 = read_performance(args.gfx950_performance)
    doc1250, perf1250 = read_performance(args.gfx1250_performance)
    hs950 = read_hotspots(args.gfx950_hotspots, 0)
    hs1250 = read_hotspots(args.gfx1250_hotspots, 1)

    work = doc950["workload"]
    ratios, headroom = breakdown_tables(hs950, hs1250)
    dates = collected_dates([
        args.gfx950_performance, args.gfx950_hotspots,
        args.gfx1250_performance, args.gfx1250_hotspots,
    ])
    if not dates:
        collected = "unknown"
        slug = args.revision[:8]
    else:
        collected = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"
        slug = f"{dates[-1].replace('-', '')}_{args.revision[:8]}"
    short = args.revision[:8]

    lines = [
        f"# Kimi-K3 TP8/EP1 logical rank 0: MI355X vs MI455X at `{short}`",
        "",
        "One physical GPU per architecture executes rank 0 of the TP8 model with",
        "local substitutes for rank-spanning collectives. Both architectures ran",
        "the same TokenSpeed commit, the same workload, and the same harness.",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| TokenSpeed revision | `{args.revision}` |",
        f"| Collected | {collected} |",
        f"| Model revision | `{doc950['software']['model_revision']}` |",
        f"| Model | {doc950['model']['model_type']}, "
        f"{doc950['model']['num_layers']} layers, "
        f"attn tp{doc950['model']['attn_tp_size']} "
        f"moe tp{doc950['model']['moe_tp_size']} "
        f"ep{doc950['model']['moe_ep_size']} |",
        f"| Prompt / output tokens | {work['prompt_tokens']:,} / "
        f"{work['output_tokens']:,} |",
        f"| Concurrencies | {', '.join(str(c) for c in work['concurrencies'])} |",
        f"| Prefill budget | {work['chunked_prefill_size']:,} tokens |",
        f"| Warmup / measured waves | {work['warmup_waves']} / "
        f"{work['measurement_waves']} |",
        f"| MI355X | {doc950['hardware']['device']}, "
        f"`{doc950['hardware']['architecture']}` |",
        f"| MI455X | {doc1250['hardware']['device']}, "
        f"`{doc1250['hardware']['architecture']}` |",
        f"| Container images | MI355X "
        f"`{doc950['software']['container_image'].split('@')[0]}`, MI455X "
        f"`{doc1250['software']['container_image'].split('@')[0]}` |",
        f"| ROCm / PyTorch / Triton | {doc950['software']['hip']} / "
        f"{doc950['software']['pytorch']} / {doc950['software']['triton']} |",
        "",
        "## End-to-end performance",
        "",
        "Latency rows are lower-is-better and throughput rows higher-is-better;",
        "the last column is always MI455X's advantage, so above 1.00x favours",
        "MI455X either way.",
        "",
        *perf_table(perf950, perf1250),
        "",
        "## Where the time goes",
        "",
        "Kernels are bucketed by function because the two architectures do not",
        "split the work into the same kernels. Ratios are accumulated GPU kernel",
        "duration, MI355X over MI455X, so above 1.00x means MI455X is ahead.",
        "",
        *ratios,
        "",
        "The same buckets, expressed as the milliseconds MI455X would save if a",
        f"bucket reached {TARGET_RATIO:g}x, the ratio its strongest buckets already",
        "reach. Negative means MI455X is already past that bar, so the bucket has",
        "nothing to give relative to MI355X whatever its absolute cost.",
        "",
        *headroom,
        "",
        "## Heaviest kernels",
        "",
        "Percent is the share of accumulated GPU kernel duration within the stage,",
        "not wall time, and kernels may overlap. The heaviest",
        f"{args.top_kernels} symbols appear below; the CSVs under",
        "`kernel-tables/` keep every symbol.",
        "",
    ]

    for arch, label in ARCHES:
        stages = hs950 if arch == "gfx950" else hs1250
        for key in STAGES:
            if key not in stages:
                continue
            lines += [
                f"### {label} ({arch}) {STAGE_LABEL[key]}",
                "",
                *kernel_table(stages[key], args.top_kernels),
                "",
            ]

    lines += [
        "## Method and limitations",
        "",
        "- Timing and profiling are separate runs. The performance tables come",
        "  from an unprofiled run; the kernel tables come from a profiled run at",
        "  the same commit and workload.",
        "- Rank-local execution omits physical collectives, HTTP serving, and",
        "  valid full-TP MoE routing, so absolute numbers are a compute estimate",
        "  rather than serving performance.",
        "- A `—` in the bucket tables means one architecture has no kernel in that",
        "  bucket for that stage, so no ratio exists. That is a difference in how",
        "  the work is split, not a measurement of zero.",
        "- Dtype columns are omitted. The profiler symbols alone do not prove",
        "  operand, accumulation, and output precision for most of these kernels.",
        "",
        "Regenerate this document with:",
        "",
        "```bash",
        "python3 toy_e2e/scripts/generate_arch_comparison.py \\",
        f"  --revision {args.revision} \\",
        "  --gfx950-performance <gfx950 performance result.json> \\",
        "  --gfx950-hotspots <gfx950 hotspots.json> \\",
        "  --gfx1250-performance <gfx1250 performance result.json> \\",
        "  --gfx1250-hotspots <gfx1250 hotspots.json> \\",
        f"  --output-dir toy_e2e/results/arch_compare_{slug}",
        "```",
        "",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "README.md").write_text("\n".join(lines))
    write_csvs(args.output_dir, "gfx950", hs950)
    write_csvs(args.output_dir, "gfx1250", hs1250)
    print(f"wrote {args.output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
