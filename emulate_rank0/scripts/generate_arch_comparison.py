#!/usr/bin/env python3
"""Turn one gfx950 and one gfx1250 run of scripts/run.sh into a comparison.

Each input is a result directory holding, per model, `serve_bench.json` and
`batches.log` from the perf phase and `hotspots/hotspots.json` from the
hotspots phase. Writes:

  README.md                             the comparison document
  kernel-tables/<model>/<arch>/*.csv    every kernel symbol per stage

Everything in the document is derived from the two directories, so
regenerating it from the same inputs reproduces it byte for byte.

The architectures split the same work into different kernels, so the
document compares functional groups rather than symbols. `CATEGORIES` is
that mapping and also the `Category` column of the kernel tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

# A model without a speculative algorithm is the baseline for the speculative
# model with the same label.
MODELS = [
    ("glm53flash", "GLM-5.3-Flash", "MTP"),
    ("dsv41", "DeepSeek-V4.1-Flash", "DSPARK"),
    ("kimik3", "Kimi-K3", "EAGLE3"),
    ("kimik3-nospec", "Kimi-K3", None),
]
ARCHES = [("gfx950", "MI355X"), ("gfx1250", "MI455X")]

# Matched against the kernel name with its arch and `.kd` suffixes stripped.
# The first match wins, so a narrow pattern must precede any broader one that
# would swallow it.
CATEGORIES = [
    ("KDA state scan", r"state_scan"),
    (
        "KDA other",
        r"kda_paged_prefill_preprocess|kda_paged_prefill_wu_vector"
        r"|kda_paged_prefill_solve_merge|kda_paged_prefill$|kda_fused"
        r"|causal_conv1d|recurrent_kda|prefill_recurrent|prefill_state_inputs"
        r"|checkpoint_output|recurrent_checkpoints",
    ),
    (
        "MoE",
        r"moe|^_stage[12]_kernel$|^_routing_kernel$|^_combine_kernel$"
        r"|gather_package|warp_decode|fp8_warp_gemv|sigmoid_bias_topk"
        r"|sigmoid_mul|_situ_kernel|^_matmul$|^_matmul_decode$|topk_route"
        r"|weighted_topk_reduce|softplus_topk",
    ),
    (
        "sparse attention",
        # Not torch's gpu_index_kernel, which is plain indexing.
        r"dsa_|indexer|index_k(?!ernel)|index_topk|sparse|flatkv|topk_to_global"
        r"|selected_attention|dsv4_prefill|dsv4_decode|kpool|hadamard",
    ),
    ("input projections", r"packed_input_projections|latent_input"),
    (
        "dense GEMM",
        r"rowcta_gemv|projection_gemv|wmma_tdm|wmma_dense|mm_a16w16|^Cijk_"
        r"|^Custom_Cijk_|smallm|gluon_bmm|w8a8_block_fp8_matmul"
        r"|mm_fp8_blockscale|mm_mxfp8|mxfp8_split_reduce|torch_mm|torch_bmm",
    ),
    (
        "attention",
        r"mla_|attn_merge|flash_attn|fmha|paged_attention|unified_attention",
    ),
    ("AttnRes", r"attn_res|attnres"),
    ("hyper-connections", r"^_mhc_|^gluon_mhc_"),
    # The elementwise add only; *_add3 GEMM epilogues are dense GEMM above.
    ("add3", r"^_add3_kernel$"),
    # Activation quantization stays here whichever op consumes it: gfx1250
    # quantizes the MoE input with the same generic kernel as everything else.
    (
        "quantize",
        r"per_token_group_quant|fp8_quantize|fp8_token_group|quant_8bit"
        r"|ue8m0_quantize|dynamic_fp8_|mxfp4_quantize",
    ),
    ("rmsnorm", r"rmsnorm|layer_norm|layernorm"),
    ("top-k and sort", r"warptopk|mbtopk|sbtopk|radixSort|sort"),
    ("sampling", r"argmax"),
    (
        "elementwise",
        r"elementwise|CatArrayBatched|copyBuffer|scatter_gather|reduce_kernel"
        r"|silu_and_mul|rope",
    ),
]

STAGES = [("EXTEND", "c1"), ("EXTEND", "c16"), ("DECODE", "c1"), ("DECODE", "c16")]
STAGE_LABEL = {
    ("EXTEND", "c1"): "prefill c1",
    ("EXTEND", "c16"): "prefill c16",
    ("DECODE", "c1"): "decode c1",
    ("DECODE", "c16"): "decode c16",
}


def bare_name(name: str) -> str:
    name = re.sub(r"\s*\[clone \.kd\]$", "", name)
    name = re.sub(r"\.kd$", "", name)
    return re.sub(r"_gfx\d+", "", name)


def categorize(name: str) -> str:
    bare = bare_name(name)
    for category, pattern in CATEGORIES:
        if re.search(pattern, bare):
            return category
    return "other"


def read_hotspots(path: Path):
    """Return {(stage, setting): [{name, ms, calls, category}, ...]}."""
    stages = {}
    for prof in json.loads(path.read_text())["profiles"]:
        rows = [
            {
                "name": k.get("name", ""),
                "ms": k.get("total_ms", 0.0),
                "calls": k.get("calls", 0),
                "category": categorize(k.get("name", "")),
            }
            for k in prof["all_kernels"]
        ]
        rows.sort(key=lambda r: -r["ms"])
        stages[(prof["stage"], str(prof.get("setting")))] = rows
    return stages


def read_bench(path: Path):
    doc = json.loads(path.read_text())
    return doc, {p["concurrency"]: p for p in doc["phases"]}


def read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


DECODE_LOG = re.compile(
    r"Decode batch\..*#running-req: (\d+),.*gen throughput \(token/s\): ([0-9.]+)"
)


def read_steady_decode(path: Path) -> dict[int, float]:
    """Median decode tok/s per batch size over log windows with no prefill.

    The server logs throughput every 40 decode steps. A window counts only
    when the line before it is a decode line at the same batch size, so no
    prefill chunk ran and no request joined or left during it.
    """
    rates, previous = {}, None
    for line in read_lines(path):
        match = DECODE_LOG.search(line)
        if match and previous == int(match[1]):
            rates.setdefault(int(match[1]), []).append(float(match[2]))
        previous = int(match[1]) if match else None
    return {batch: statistics.median(values) for batch, values in rates.items()}


def read_run(root: Path) -> dict:
    rev = read_lines(root / "tokenspeed.rev")
    env = {}
    ran = [m for m, _, _ in MODELS if (root / m / "env.txt").exists()]
    for line in read_lines(root / (ran[0] if ran else MODELS[0][0]) / "env.txt"):
        key, _, value = line.partition(" ")
        # "arch gfx1250 memory_gib 432.0" carries the memory size too.
        env[key] = value.split()[0] if key == "arch" and value else value
    image = read_lines(root / "image.id")
    return {
        "root": root,
        "revision": next(
            (line.split()[1] for line in rev if line.startswith("tokenspeed ")), ""
        ),
        "status": [line for line in rev if not line.startswith(("arch ", "tokenspeed "))],
        "diff": (root / "tokenspeed.diff").read_bytes()
        if (root / "tokenspeed.diff").exists()
        else b"",
        "image": image[1] if len(image) > 1 else (image[0] if image else ""),
        "env": env,
    }


def read_layout(path: Path) -> str:
    """The parallel layout a serve command emulates, such as `TP4`."""
    command = " ".join(read_lines(path))
    tp = re.search(r"--(?:tensor-parallel-size|attn-tp-size) (\d+)", command)
    ep = re.search(r"--ep-size (\d+)", command)
    if not tp:
        return "unknown"
    return f"TP{tp[1]}" + (f" EP{ep[1]}" if ep and int(ep[1]) > 1 else "")


def fmt(value, digits=1):
    return f"{value:,.{digits}f}"


def steady_tpot(rate, conc):
    return 1e3 * conc / rate if rate else None


def perf_rows(phase, rate):
    conc = phase["concurrency"]
    return [
        ("TTFT p50 (ms)", phase["time_to_first_token_ms"]["median"], "lower"),
        ("TPOT p50 (ms)", phase["time_per_output_token_ms"]["median"], "lower"),
        ("steady decode TPOT (ms)", steady_tpot(rate, conc), "lower"),
        ("decode tok/s per user", rate / conc if rate else None, "higher"),
        ("aggregate output tok/s", phase["output_tokens_per_s"]["median"], "higher"),
    ]


def perf_table(benches, steady) -> list[str]:
    out = [
        "| Metric | Batch | MI355X (gfx950) | MI455X (gfx1250) | MI455X advantage |",
        "|---|---:|---:|---:|---:|",
    ]
    a_phases, b_phases = benches
    for conc in sorted(set(a_phases) & set(b_phases)):
        rows_a = perf_rows(a_phases[conc], steady[0].get(conc))
        rows_b = perf_rows(b_phases[conc], steady[1].get(conc))
        for (label, a, better), (_, b, _) in zip(rows_a, rows_b):
            if a is None or b is None:
                out.append(f"| {label} | {conc} | — | — | — |")
                continue
            ratio = a / b if better == "lower" else b / a
            out.append(f"| {label} | {conc} | {fmt(a)} | {fmt(b)} | {ratio:.2f}x |")
    return out


def spec_gain_table(baseline, speculative, algo) -> list[str]:
    """Each architecture's speculative run against its run without speculation.

    Both arguments are the (benches, steady) pairs of the two models' sections.
    """
    labels = [label for _, label in ARCHES]
    out = [
        "| Metric | Batch | "
        + " | ".join(f"{a} off | {a} {algo} | {a} gain" for a in labels)
        + " |",
        "|---|---:|" + "---:|" * (3 * len(labels)),
    ]
    (off_benches, off_steady), (on_benches, on_steady) = baseline, speculative
    shared = set.intersection(*(set(b) for b in (*off_benches, *on_benches)))
    for conc in sorted(shared):
        per_arch = [
            (
                perf_rows(off_benches[i][conc], off_steady[i].get(conc)),
                perf_rows(on_benches[i][conc], on_steady[i].get(conc)),
            )
            for i in range(len(ARCHES))
        ]
        for row, (label, _, better) in enumerate(per_arch[0][0]):
            cells = []
            for off_rows, on_rows in per_arch:
                off, on = off_rows[row][1], on_rows[row][1]
                if off is None or on is None:
                    cells += ["—"] * 3
                    continue
                gain = off / on if better == "lower" else on / off
                cells += [fmt(off), fmt(on), f"{gain:.2f}x"]
            out.append(f"| {label} | {conc} | " + " | ".join(cells) + " |")
    return out


def category_totals(rows):
    totals = {}
    for row in rows:
        totals[row["category"]] = totals.get(row["category"], 0.0) + row["ms"]
    return totals


def stage_latency(stage, conc, phases, steady):
    """The measured latency a stage column is anchored to, or None."""
    if stage == "EXTEND":
        return phases[conc]["time_to_first_token_ms"]["median"] if conc in phases else None
    return steady_tpot(steady.get(conc), conc)


def breakdown_table(hotspots, benches, steady) -> tuple[list[str], list[str]]:
    per_stage = {
        key: tuple(category_totals(h.get(key, [])) for h in hotspots)
        for key in STAGES
    }
    present = {c for a, b in per_stage.values() for c in set(a) | set(b)}
    named = [name for name, _ in CATEGORIES]
    order = [name for name in named if name in present]
    if "other" in present:
        order.append("other")

    lines = [
        "| Category | " + " | ".join(STAGE_LABEL[s] for s in STAGES) + " |",
        "|---|" + "---:|" * len(STAGES),
    ]
    for category in order:
        cells = []
        for key in STAGES:
            a, b = (t.get(category, 0.0) for t in per_stage[key])
            cells.append(f"{a / b:.2f}x" if a and b else "—")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    cells = []
    for key in STAGES:
        a, b = (sum(t.values()) for t in per_stage[key])
        cells.append(f"{a / b:.2f}x" if a and b else "—")
    lines.append("| all kernels | " + " | ".join(cells) + " |")

    cells = []
    for stage, setting in STAGES:
        conc = int(setting.lstrip("c"))
        a, b = (
            stage_latency(stage, conc, phases, rates)
            for phases, rates in zip(benches, steady)
        )
        label = "TTFT" if stage == "EXTEND" else "steady TPOT"
        cells.append(f"**{a / b:.2f}x** ({label})" if a and b else "—")
    lines.append("| **End-to-end** | " + " | ".join(cells) + " |")

    warnings = []
    for key in STAGES:
        a, b = per_stage[key]
        for category in set(a) | set(b):
            ms_a, ms_b = a.get(category, 0.0), b.get(category, 0.0)
            if max(ms_a, ms_b) >= 5.0 and min(ms_a, ms_b) < 5.0:
                warnings.append(
                    f"{STAGE_LABEL[key]} {category!r}: {ms_a:.1f} ms on gfx950, "
                    f"{ms_b:.1f} ms on gfx1250"
                )
    return lines, sorted(warnings)


def kernel_table(rows, top) -> list[str]:
    total = sum(r["ms"] for r in rows) or 1.0
    out = [
        "| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows[:top], 1):
        per_call = r["ms"] / r["calls"] * 1000 if r["calls"] else 0.0
        name = r["name"] if len(r["name"]) <= 120 else r["name"][:117] + "..."
        out.append(
            f"| {i} | {r['category']} | `{name}` | {fmt(r['ms'], 3)} "
            f"| {r['calls']:,} | {fmt(per_call, 1)} | {r['ms'] / total * 100:.2f}% |"
        )
    return out


def write_csvs(dest: Path, stages) -> None:
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
    ap.add_argument("--gfx950", required=True, type=Path, help="MI355X result dir")
    ap.add_argument("--gfx1250", required=True, type=Path, help="MI455X result dir")
    ap.add_argument(
        "--commit-date",
        required=True,
        help="UTC date of the commit as YYYY-MM-DD, from "
        "TZ=UTC git show -s --format=%%cd --date=format-local:%%Y-%%m-%%d <sha>",
    )
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--top-kernels", type=int, default=10)
    args = ap.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.commit_date):
        raise SystemExit("--commit-date must be YYYY-MM-DD")

    runs = [read_run(args.gfx950), read_run(args.gfx1250)]
    if runs[0]["revision"] != runs[1]["revision"]:
        raise SystemExit(
            f"revisions differ: {runs[0]['revision']} vs {runs[1]['revision']}"
        )
    if runs[0]["diff"] != runs[1]["diff"] or runs[0]["status"] != runs[1]["status"]:
        raise SystemExit("the two runs applied different changes on top of the commit")
    revision = runs[0]["revision"]
    short = revision[:8]

    sections, dates, accepts, all_warnings, perf_by_model = [], set(), [], [], {}
    for model, label, algo in MODELS:
        dirs = [run["root"] / model for run in runs]
        if not all((d / "serve_bench.json").exists() for d in dirs):
            print(f"skipping {model}: missing serve_bench.json")
            continue
        docs, benches = zip(*(read_bench(d / "serve_bench.json") for d in dirs))
        dates.update(d["generated_at"][:10] for d in docs if d.get("generated_at"))
        hotspots = [
            read_hotspots(d / "hotspots" / "hotspots.json")
            if (d / "hotspots" / "hotspots.json").exists()
            else {}
            for d in dirs
        ]
        logged = []
        for d in dirs:
            text = " ".join(read_lines(d / "accept.txt"))
            match = re.search(r"logged median ([0-9.]+) over (\d+) windows", text)
            logged.append(
                f"{match[1]} ({match[2]} windows)" if match else "not logged"
            )
        layouts = {read_layout(d / "serve_cmd.txt") for d in dirs}
        if len(layouts) != 1:
            raise SystemExit(f"{model}: the two runs emulated different layouts {layouts}")
        layout = layouts.pop()
        if algo:
            simulated = read_lines(dirs[0] / "simulation.txt")
            accept = next(
                (s.split("=", 1)[1] for s in simulated if s.startswith("TOKENSPEED_SPEC")),
                "",
            )
            accepts.append((label, algo, layout, accept, logged))
        elif any(entry != "not logged" for entry in logged):
            all_warnings.append(f"{model} logged an accept length without speculation")

        steady = [read_steady_decode(d / "batches.log") for d in dirs]
        perf_by_model[model] = (benches, steady)
        title = f"{label} {algo}" if algo else f"{label} without speculation"
        lines = [f"## {title}", "", *perf_table(benches, steady), ""]
        if all(hotspots):
            table, warnings = breakdown_table(hotspots, benches, steady)
            all_warnings += [f"{label}: {w}" for w in warnings]
            lines += [
                "Accumulated GPU kernel time per bucket, MI355X over MI455X, so",
                "above 1.00x means MI455X is ahead. The last row compares the",
                "measured TTFT p50 and steady decode TPOT from the table above; it",
                "is not a total of the rows.",
                "",
                *table,
                "",
            ]
            for (arch, arch_label), stages in zip(ARCHES, hotspots):
                write_csvs(args.output_dir / "kernel-tables" / model / arch, stages)
                for key in STAGES:
                    if key in stages:
                        lines += [
                            f"<details><summary>{arch_label} ({arch}) "
                            f"{STAGE_LABEL[key]}: heaviest {args.top_kernels} "
                            "kernels</summary>",
                            "",
                            *kernel_table(stages[key], args.top_kernels),
                            "",
                            "</details>",
                            "",
                        ]
        sections += lines

    baselines = {label: m for m, label, algo in MODELS if not algo and m in perf_by_model}
    for model, label, algo in MODELS:
        if algo and model in perf_by_model and label in baselines:
            sections += [
                f"## {label}: {algo} against no speculation",
                "",
                f"Each architecture's {algo} run over its own run without",
                f"speculation, from the two sections above. Above 1.00x means {algo}",
                "is ahead.",
                "",
                *spec_gain_table(perf_by_model[baselines[label]], perf_by_model[model], algo),
                "",
            ]

    for warning in all_warnings:
        print(f"warning: {warning}")
    dates = sorted(dates)
    measured = "unknown" if not dates else (
        dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"
    )
    work = docs[0]["config"]
    env950, env1250 = runs[0]["env"], runs[1]["env"]
    slug = f"{args.commit_date.replace('-', '')}_{short}"

    if baselines:
        intro = [
            "# Emulated rank 0 with and without speculative decoding: MI355X vs "
            f"MI455X at `{short}`",
            "",
            "One GPU per architecture serves global rank 0 of each model's CI layout",
            "with `tokenspeed serve --emulate-rank-zero`, dummy weights and uniform MoE",
            "routing. Speculative runs fix a simulated accept length. Collectives",
            "are local substitutes, so the numbers are a compute estimate, not",
            "serving performance. Both architectures ran the same tree and workload.",
        ]
    else:
        intro = [
            f"# Emulated rank 0 with speculative decoding: MI355X vs MI455X at `{short}`",
            "",
            "One GPU per architecture serves global rank 0 of each model's CI layout",
            "with `tokenspeed serve --emulate-rank-zero`, dummy weights, uniform MoE",
            "routing and a fixed simulated accept length. Collectives are local",
            "substitutes, so the numbers are a compute estimate, not serving",
            "performance. Both architectures ran the same tree and workload.",
        ]
    lines = [
        *intro,
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| TokenSpeed revision | `{revision}` |",
        f"| Commit date (UTC) | {args.commit_date} |",
        f"| Changes on top | {len(runs[0]['status'])} files, identical on both hosts |",
        f"| Measured | {measured} |",
        f"| Prompt / output tokens | {work['prompt_tokens']:,} / {work['output_tokens']:,} |",
        f"| Concurrencies | {', '.join(str(c) for c in work['concurrency'])} |",
        f"| Warmup / measured waves | {work['warmup_waves']} / {work['measurement_waves']} |",
        f"| MI355X | `{env950.get('arch', '?')}`, {env950.get('torch', '?')}, "
        f"`{runs[0]['image']}` |",
        f"| MI455X | `{env1250.get('arch', '?')}`, {env1250.get('torch', '?')}, "
        f"`{runs[1]['image']}` |",
        f"| tokenspeed-triton | {env950.get('tokenspeed-triton', '?')} / "
        f"{env1250.get('tokenspeed-triton', '?')} |",
        "",
        "## Simulated acceptance",
        "",
        "Logged values are medians over the server's 40-step decode windows;",
        "windows that straddle a prefill keep fewer tokens. See",
        "`emulate_rank0/README.md` for where each simulated value comes from.",
        "",
        "| Model | Layout emulated | Simulated | MI355X logged | MI455X logged |",
        "|---|---|---:|---|---|",
        *(
            f"| {label} {algo} | {layout} | {accept} | {logged[0]} | {logged[1]} |"
            for label, algo, layout, accept, logged in accepts
        ),
        "",
        "Latency rows are lower-is-better and throughput rows higher-is-better;",
        "the last column is always MI455X's advantage. TPOT p50 is per request",
        "over its whole decode, as CI reports it. At batch 16 every request",
        "arrives at once, so a request whose prompt finishes early decodes",
        "between other requests' prefill chunks, and its TPOT mostly measures",
        "prefill. Steady decode TPOT is the median over the server's 40-step",
        "decode windows in which every request was decoding and no prefill ran;",
        "decode tok/s per user and the decode columns below use it. Both include",
        "the speculative speedup when speculation is on." if baselines
        else "the speculative speedup.",
        "",
        *sections,
        "## Method and limitations",
        "",
        "- Timing and profiling are separate server runs. Performance comes from",
        "  a run with decode CUDA graphs; kernels come from an eager run without",
        "  overlap scheduling, under rocprofv3.",
        "- Emulation omits communication time, so multi-GPU collectives and",
        "  their overlap with compute are not measured.",
        "- Dummy weights with uniform routing give every expert equal load, which",
        "  real routing does not.",
        "- Accept lengths are fixed, so these runs compare the cost of a verify",
        "  step at a given acceptance, not drafter quality.",
        "- A `—` means one architecture has no kernel in that bucket for that",
        "  stage, a difference in how the work is split rather than zero time.",
        "",
        "Regenerate this document with:",
        "",
        "```bash",
        "python3 emulate_rank0/scripts/generate_arch_comparison.py \\",
        "  --gfx950 <MI355X result dir> --gfx1250 <MI455X result dir> \\",
        f"  --commit-date {args.commit_date} \\",
        f"  --output-dir emulate_rank0/results/arch_compare_{slug}",
        "```",
        "",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "README.md").write_text("\n".join(lines))
    print(f"wrote {args.output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
