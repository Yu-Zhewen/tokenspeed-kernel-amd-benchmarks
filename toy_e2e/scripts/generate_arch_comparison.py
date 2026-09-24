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
# One pattern per category, matched against the kernel name with its arch
# suffix stripped, so a kernel present on both architectures cannot land in
# different buckets. Order matters: the first match wins, so a narrow pattern
# must precede any broader one that would swallow it. `_rowcta_gemv_add3` is
# the example that caught me out -- it is a GEMV with a fused add, so it
# belongs with the GEMMs and must be claimed before the add3 bucket sees it.
CATEGORIES = [
    ("KDA state scan", r"state_scan"),
    (
        "MoE",
        r"mxfp4_moe|gather_package|warp_decode|moe_partial_reduce"
        r"|dynamic_fp8_single_pass|sigmoid_bias_topk|sigmoid_mul|_situ_kernel"
        r"|^_matmul\.|^_matmul_decode\.|topk_route|weighted_topk_reduce",
    ),
    (
        "dense GEMM",
        r"rowcta_gemv|projection_gemv|wmma_tdm_dense|wmma_tdm_add3|mm_a16w16|^Cijk_"
        r"|^Custom_Cijk_|smallm|gluon_bmm|torch_mm|torch_bmm",
    ),
    ("input projections", r"packed_input_projections|latent_input"),
    ("MLA attention", r"mla_prefill|mla_decode|project_value|kv_pack"),
    ("AttnRes", r"attn_res|attnres"),
    (
        "KDA other",
        r"kda_paged_prefill_preprocess|kda_paged_prefill_wu_vector"
        r"|kda_paged_prefill_solve_merge|kda_paged_prefill_gfx|kda_fused"
        r"|causal_conv1d",
    ),
    # Only the elementwise add in ops/activation. The *_add3 GEMM epilogues
    # live in gemm/ and are claimed by dense GEMM above.
    ("add3", r"^_add3_kernel"),
    ("rmsnorm", r"rmsnorm|layernorm"),
    ("elementwise", r"elementwise|CatArrayBatched|copyBuffer"),
]

STAGES = [("EXTEND", "c16"), ("EXTEND", "c1"), ("DECODE", "c16"), ("DECODE", "c1")]
STAGE_LABEL = {
    ("EXTEND", "c16"): "prefill c16",
    ("EXTEND", "c1"): "prefill c1",
    ("DECODE", "c16"): "decode c16",
    ("DECODE", "c1"): "decode c1",
}
ARCHES = [("gfx950", "MI355X"), ("gfx1250", "MI455X")]


def strip_arch(name: str) -> str:
    """Drop the arch suffix so both architectures match the same pattern."""
    return re.sub(r"_gfx\d+", "", name)


def categorize(name: str, arch_index: int = 0) -> str:
    """Bucket a kernel by function. ``arch_index`` is accepted and ignored."""
    bare = strip_arch(name)
    for category, pattern in CATEGORIES:
        if re.search(pattern, bare):
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


def check_bucket_symmetry(stages_a, stages_b) -> list[str]:
    """Report kernels that both architectures run but that bucket differently.

    A mismatch silently ruins a ratio: the kernel counts towards one bucket on
    one side and another on the other, so neither total describes the same
    work. This has gone wrong three times, so it is checked rather than
    assumed.
    """
    seen = {}
    problems = []
    for stages, label in ((stages_a, "gfx950"), (stages_b, "gfx1250")):
        for rows in stages.values():
            for row in rows:
                bare = strip_arch(row["name"])
                prior = seen.setdefault(bare, (row["category"], label))
                if prior[0] != row["category"]:
                    problems.append(
                        f"{bare}: {prior[1]} -> {prior[0]}, {label} -> "
                        f"{row['category']}"
                    )
    return sorted(set(problems))


def check_bucket_coverage(stages_a, stages_b, floor_ms: float = 5.0):
    """Flag buckets that are substantial on one architecture and absent on the other.

    Symmetric names are not enough. If the two architectures fuse an operation
    differently, the kernels have unrelated names and land in different
    buckets, so the ratio compares unlike work even though no single kernel is
    miscategorised. gfx1250 fusing a matmul and its add into one kernel while
    gfx950 leaves the matmul in hipBLASLt is the case that motivated this.
    """
    warnings = []
    for stage in sorted(set(stages_a) | set(stages_b), key=str):
        totals = []
        for stages in (stages_a, stages_b):
            by_cat = {}
            for row in stages.get(stage, []):
                by_cat[row["category"]] = (
                    by_cat.get(row["category"], 0.0) + row["ms"]
                )
            totals.append(by_cat)
        for category in set(totals[0]) | set(totals[1]):
            a = totals[0].get(category, 0.0)
            b = totals[1].get(category, 0.0)
            if max(a, b) >= floor_ms and min(a, b) < floor_ms:
                warnings.append(
                    f"{stage} bucket {category!r} is {a:.1f}ms on gfx950 and "
                    f"{b:.1f}ms on gfx1250; the architectures likely split "
                    f"this work differently, so the ratio is not meaningful"
                )
    return warnings


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


# The measured step latency behind each stage column, so the breakdown can
# close with the number the machine actually reported rather than a total of
# its own rows.
STAGE_E2E = {
    "EXTEND": lambda b: b["step_wall_ms"]["prefill"]["p50"],
    "DECODE": lambda b: b["steady_decode_step_ms"]["p50"],
}


def breakdown_tables(hs950, hs1250, perf950, perf1250) -> list[str]:
    per_stage = {
        key: (category_totals(hs950, key), category_totals(hs1250, key))
        for key in STAGES
    }
    categories = {c for a, b in per_stage.values() for c in set(a) | set(b)}
    # Fixed order, not size. Sorting by time moved a row whenever that bucket
    # got faster, so the same category was not on the same line across commits.
    named = [name for name, _pattern in CATEGORIES]
    order = [name for name in named if name in categories]
    order += sorted(categories - set(named) - {"other"})
    if "other" in categories:
        order.append("other")
    header = "| Category | " + " | ".join(
        STAGE_LABEL[s] for s in STAGES
    ) + " |"
    align = "|---|" + "---:|" * len(STAGES)

    ratios = [header, align]
    for category in order:
        rcells = []
        for key in STAGES:
            a, b = per_stage[key]
            ms_a = a.get(category, [0.0, 0])[0]
            ms_b = b.get(category, [0.0, 0])[0]
            rcells.append(f"{ms_a / ms_b:.2f}x" if ms_a and ms_b else "—")
        ratios.append(f"| {category} | " + " | ".join(rcells) + " |")

    # Close with the measured step latency rather than a total of the rows
    # above, so the breakdown is anchored to what the machine reported.
    ocells = []
    for stage, conc in STAGES:
        get = STAGE_E2E[stage]
        batch = int(conc.lstrip("c"))
        try:
            a, b = get(perf950[batch]), get(perf1250[batch])
        except KeyError:
            ocells.append("—")
            continue
        ocells.append(f"**{a / b:.2f}x**")
    ratios.append("| **End-to-end (step p50)** | " + " | ".join(ocells) + " |")
    return ratios


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
    ap.add_argument(
        "--commit-date",
        required=True,
        help="UTC date of the commit as YYYY-MM-DD, from "
        "TZ=UTC git show -s --format=%cd --date=format-local:%Y-%m-%d <sha>",
    )
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

    lopsided = check_bucket_coverage(hs950, hs1250)
    for warning in lopsided:
        print(f"warning: {warning}")
    mismatches = check_bucket_symmetry(hs950, hs1250)
    if mismatches:
        raise SystemExit(
            "kernels bucket differently per architecture, so the ratios would "
            "compare unlike work:\n  " + "\n  ".join(mismatches)
        )
    work = doc950["workload"]
    ratios = breakdown_tables(hs950, hs1250, perf950, perf1250)
    dates = collected_dates([
        args.gfx950_performance, args.gfx950_hotspots,
        args.gfx1250_performance, args.gfx1250_hotspots,
    ])
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.commit_date):
        raise SystemExit("--commit-date must be YYYY-MM-DD")
    collected = "unknown" if not dates else (
        dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"
    )
    slug = f"{args.commit_date.replace('-', '')}_{args.revision[:8]}"
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
        f"| Commit date (UTC) | {args.commit_date} |",
        f"| Measured | {collected} |",
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
        "The final row is the measured step latency from the end-to-end "
        "table above, not a total of the rows, so the buckets can be read "
        "against what the machine actually reported.",
        "",
        "Kernels are bucketed by function because the two architectures do not",
        "split the work into the same kernels. Rows follow that category list,",
        "the same order in every entry. Ratios are accumulated GPU kernel",
        "duration, MI355X over MI455X, so above 1.00x means MI455X is ahead.",
        "",
        *ratios,
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
        f"  --commit-date {args.commit_date} \\",
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
