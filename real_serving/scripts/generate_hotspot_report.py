#!/usr/bin/env python3
"""Turn one MI455X run of real_serving/scripts/run.sh into a hotspot document.

The input holds, per model, `serve_bench.json` and `batches.log` from the
perf phase and `hotspots/hotspots.json` (rank 0's kernels) from the hotspots
phase. Writes:

  README.md                       the document
  kernel-tables/<model>/*.csv     every kernel symbol per stage

Kernels are grouped with emulate_rank0's `CATEGORIES`, behind a
`communication` bucket that emulation never ran.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "arch_compare",
    Path(__file__).resolve().parents[2]
    / "emulate_rank0/scripts/generate_arch_comparison.py",
)
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)

COMMUNICATION = "communication"
# RCCL kernels, TokenSpeed's own all-reduce, all-gather and reduce-scatter
# kernels, and Iris symmetric-memory collectives with their barrier.
ac.CATEGORIES = [
    (
        COMMUNICATION,
        r"nccl|rccl|all_?reduce|all_?gather|reduce_?scatter|cross_device_reduce"
        r"|one_shot|two_shot|iris|all_to_all|alltoall|^_device_barrier_kernel$",
    ),
    # The paged KDA prefill's prepare, summary and walk kernels postdate the
    # emulate_rank0 pattern.
    *(
        (name, pattern + r"|kda_paged_prefill") if name == "KDA other" else (name, pattern)
        for name, pattern in ac.CATEGORIES
    ),
]

MODELS = [
    ("glm53flash", "GLM-5.3-Flash", "MTP"),
    ("dsv41", "DeepSeek-V4.1-Flash", "DSPARK"),
    ("kimik3", "Kimi-K3", "EAGLE3"),
]
STAGES = ac.STAGES
STAGE_LABEL = ac.STAGE_LABEL
fmt = ac.fmt


def logged_accept(model_dir: Path) -> float | None:
    text = " ".join(ac.read_lines(model_dir / "accept.txt"))
    match = re.search(r"logged median ([0-9.]+)", text)
    return float(match[1]) if match else None


def read_env(model_dir: Path) -> dict:
    env = {}
    for line in ac.read_lines(model_dir / "env.txt"):
        key, _, value = line.partition(" ")
        env[key] = value
    return env


def layout(model_dir: Path, env: dict) -> str:
    base = ac.read_layout(model_dir / "serve_cmd.txt")
    nodes = int(env.get("nnodes", "1") or 1)
    gpus = int(env.get("gpus_per_node", "0") or 0)
    where = f"{nodes} nodes x {gpus} GPUs" if nodes > 1 else f"1 node, {gpus} GPUs"
    return f"{base} ({where})"


def perf_table(phases, rate_by_batch, accept) -> list[str]:
    batches = sorted(phases)
    out = [
        "| Metric | " + " | ".join(f"Batch {b}" for b in batches) + " |",
        "|---|" + "---:|" * len(batches),
    ]
    rows = {}
    for b in batches:
        for label, value, _ in ac.perf_rows(phases[b], rate_by_batch.get(b)):
            rows.setdefault(label, []).append(value)
        step = verify_step_ms(rate_by_batch.get(b), b, accept)
        rows.setdefault("steady verify step (ms)", []).append(step)
    for label, values in rows.items():
        cells = ["—" if v is None else fmt(v, 2 if "step" in label else 1) for v in values]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return out


def verify_step_ms(rate, batch, accept):
    """Wall time of one decode forward: `accept` tokens per request each."""
    if not rate or not accept:
        return None
    return 1e3 * batch * accept / rate


def stage_scale(stage, decode_steps):
    """Kernel totals per prefill wave, and per decode step."""
    return 1.0 if stage == "EXTEND" else 1.0 / decode_steps


def breakdown_table(hotspots, decode_steps, phases, rates, accept):
    per_stage = {key: ac.category_totals(hotspots.get(key, [])) for key in STAGES}
    present = {c for totals in per_stage.values() for c in totals}
    order = [name for name, _ in ac.CATEGORIES if name in present]
    if "other" in present:
        order.append("other")
    lines = [
        "| Category | " + " | ".join(STAGE_LABEL[s] for s in STAGES) + " |",
        "|---|" + "---:|" * len(STAGES),
    ]

    def cell(ms, total, key):
        if not ms:
            return "—"
        scaled = ms * stage_scale(key[0], decode_steps)
        return f"{fmt(scaled, 2)} ({ms / total * 100:.0f}%)"

    totals = {key: sum(t.values()) or 1.0 for key, t in per_stage.items()}
    for category in order:
        cells = [cell(per_stage[k].get(category, 0.0), totals[k], k) for k in STAGES]
        name = f"**{category}**" if category == COMMUNICATION else category
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    cells = []
    for key in STAGES:
        total = sum(per_stage[key].values())
        cells.append(fmt(total * stage_scale(key[0], decode_steps), 2) if total else "—")
    lines.append("| all kernels | " + " | ".join(cells) + " |")
    cells = []
    for stage, setting in STAGES:
        batch = int(setting.lstrip("c"))
        if stage == "EXTEND":
            # The window holds the whole wave, which ends with the last TTFT.
            value = phases[batch]["time_to_first_token_ms"]["max"] if batch in phases else None
            label = "TTFT max"
        else:
            value = verify_step_ms(rates.get(batch), batch, accept)
            label = "verify step"
        cells.append(f"**{fmt(value, 2)}** ({label})" if value else "—")
    lines.append("| **Measured, CUDA graphs** | " + " | ".join(cells) + " |")
    return lines, per_stage


def read_rank_balance(path: Path) -> dict:
    """{(rank, window): (comm_ms, calls, compute_ms)} from rank_balance.py."""
    out = {}
    for line in ac.read_lines(path):
        tag, window, _, comm, _, calls, _, compute = line.split()
        out[(tag, window)] = (float(comm), int(calls), float(compute))
    return out


def read_probe(path: Path) -> dict:
    """{(mode, tokens): (bytes, us)} from allreduce_probe.py."""
    out = {}
    for line in ac.read_lines(path):
        parts = line.split()
        if len(parts) == 7 and parts[1] == "tokens":
            out[(parts[0], int(parts[2]))] = (int(parts[4]), float(parts[6]))
    return out


DECODE_WINDOWS = ("c1/decode", "c16/decode")


def rank_balance_table(balance, steps) -> list[str]:
    ranks = sorted({tag for tag, _ in balance}, key=lambda t: int(re.sub(r"\D", "", t) or 0))
    out = [
        "| Rank | " + " | ".join(
            f"{w.replace('/', ' ')}: per call (us) | {w.replace('/', ' ')}: comm / compute per step (ms)"
            for w in DECODE_WINDOWS
        ) + " |",
        "|---|" + "---:|---:|" * len(DECODE_WINDOWS),
    ]
    for tag in ranks:
        cells = []
        for window in DECODE_WINDOWS:
            if (tag, window) not in balance:
                cells += ["—", "—"]
                continue
            comm, calls, compute = balance[(tag, window)]
            cells += [fmt(comm / calls * 1e3 if calls else 0.0),
                      f"{fmt(comm / steps, 2)} / {fmt(compute / steps, 2)}"]
        out.append(f"| {tag} | " + " | ".join(cells) + " |")
    return out


def probe_table(probe) -> list[str]:
    sizes = sorted({tokens for _, tokens in probe})
    out = [
        "| Tokens | Bytes | Back-to-back (us) | CUDA-graph replay (us) |",
        "|---:|---:|---:|---:|",
    ]
    for tokens in sizes:
        size = (probe.get(("eager", tokens)) or probe.get(("graph", tokens)))[0]
        eager, graph = (probe.get((m, tokens)) for m in ("eager", "graph"))
        out.append(
            f"| {tokens:,} | {size:,} | {fmt(eager[1]) if eager else '—'} "
            f"| {fmt(graph[1]) if graph else '—'} |"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="MI455X result dir")
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

    sections, summary, accepts, dates, revisions, modes = [], [], [], set(), set(), set()
    collectives = []
    image, env0, work = "", {}, None
    for model, label, algo in MODELS:
        d = args.run / model
        if not (d / "serve_bench.json").exists():
            print(f"skipping {model}: missing serve_bench.json")
            continue
        doc, phases = ac.read_bench(d / "serve_bench.json")
        work = work or doc["config"]
        if doc.get("generated_at"):
            dates.add(doc["generated_at"][:10])
        env = read_env(d)
        env0 = env0 or env
        revisions.add(env.get("tokenspeed", "unknown"))
        ids = ac.read_lines(d / "image.id")
        image = image or (ids[1] if len(ids) > 1 else "")
        accept = logged_accept(d)
        simulated = next(
            (s.split("=", 1)[1] for s in ac.read_lines(d / "simulation.txt")
             if s.startswith("TOKENSPEED_SPEC")),
            "",
        )
        where = layout(d, env)
        accepts.append((label, algo, where, simulated or "off", accept))
        rates = ac.read_steady_decode(d / "batches.log")
        lines = [f"## {label} {algo}, {where}", "", *perf_table(phases, rates, accept), ""]

        hot_path = d / "hotspots" / "hotspots.json"
        eager = "--enforce-eager" in " ".join(ac.read_lines(d / "serve_hotspots_cmd.txt"))
        mode = "eager" if eager else "CUDA-graph"
        modes.add(mode)
        if hot_path.exists():
            hotspots = ac.read_hotspots(hot_path)
            steps = doc["config"].get("decode_steps", 64)
            table, per_stage = breakdown_table(hotspots, steps, phases, rates, accept)
            lines += [
                f"Rank 0's accumulated GPU kernel time per bucket from the {mode}",
                "profiling run, with each bucket's share of the stage. Prefill",
                "columns cover one wave's whole prefill (every chunk of every",
                f"request); decode columns are per verify step, averaged over {steps}.",
                "The last row is the perf run's slowest TTFT in the wave, when its",
                "prefill ends, and its steady verify step; it is not a total of",
                "the rows. Communication kernels include the time they wait for",
                "the other ranks.",
                "",
                *table,
                "",
            ]
            shares = []
            for key in STAGES:
                total = sum(per_stage[key].values())
                comm = per_stage[key].get(COMMUNICATION, 0.0)
                shares.append(f"{comm / total * 100:.0f}%" if total else "—")
            summary.append(f"| {label} {algo} | {where} | " + " | ".join(shares) + " |")
            balance = read_rank_balance(d / "rank_balance.txt")
            probe = read_probe(d / "allreduce_probe.txt")
            if balance or probe:
                lines += ["### Collectives", ""]
            if balance:
                profiled = len({tag for tag, _ in balance})
                world = int(env.get("nnodes", "1") or 1) * int(env.get("gpus_per_node", "0") or 0)
                which = (
                    f"all {profiled} ranks" if profiled == world
                    else f"the {profiled} ranks on node 0, of {world}"
                )
                lines += [
                    f"Every profiled rank ({which}), from `rank_balance.txt`. A",
                    "collective ends on every rank together, so a rank that reached",
                    "it late would spend less time in it than the others.",
                    "",
                    *rank_balance_table(balance, steps),
                    "",
                ]
            if probe:
                calls = {w: balance.get(("TP0", w)) for w in DECODE_WINDOWS}
                per_call = [
                    f"{fmt(c[0] / c[1] * 1e3)} us at {w.split('/')[0]}"
                    for w, c in calls.items() if c and c[1]
                ]
                lines += [
                    "A bf16 all-reduce of hidden-size rows over the same GPUs and",
                    "network setup outside the server (`allreduce_probe.txt`). Rank 0's",
                    "decode collectives inside the server average "
                    f"{' and '.join(per_call) or 'an unknown time'} per call. Some",
                    "server collectives carry fused norms or wider rows than the",
                    "hidden size.",
                    "",
                    *probe_table(probe),
                    "",
                ]
                graph4 = probe.get(("graph", 4))
                if graph4 and calls.get("c1/decode") and calls["c1/decode"][1]:
                    comm, n, _ = calls["c1/decode"]
                    collectives.append(
                        f"| {label} {algo} | {where} | {n / steps:.0f} "
                        f"| {fmt(comm / n * 1e3)} | {fmt(graph4[1])} "
                        f"| {fmt(comm / steps, 2)} |"
                    )
            ac.write_csvs(args.output_dir / "kernel-tables" / model, hotspots)
            for key in STAGES:
                if key in hotspots:
                    lines += [
                        f"<details><summary>{STAGE_LABEL[key]}: heaviest "
                        f"{args.top_kernels} kernels</summary>",
                        "",
                        *ac.kernel_table(hotspots[key], args.top_kernels),
                        "",
                        "</details>",
                        "",
                    ]
        sections += lines

    if work is None:
        raise SystemExit(f"no model results under {args.run}")
    if len(revisions) != 1:
        raise SystemExit(f"models ran different trees: {sorted(revisions)}")
    revision = revisions.pop()
    short = revision[:8]
    dates = sorted(dates)
    measured = "unknown" if not dates else (
        dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"
    )
    slug = f"{args.commit_date.replace('-', '')}_{short}"
    lines = [
        f"# Real-weight serving hotspots on MI455X at `{short}`",
        "",
        "Each model serves its real checkpoint over its CI job's parallel layout",
        "on MI455X (gfx1250) GPUs, with real MoE routing and real collectives",
        "(RCCL and Iris over the GPUs' scale-up fabric; `real_serving/README.md`",
        "describes the nodes). Speculative runs fix the accept length",
        "emulate_rank0 simulates, so a verify step does the same work as there.",
        "Kernel tables are rank 0's.",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| TokenSpeed revision | `{revision}` |",
        f"| Commit date (UTC) | {args.commit_date} |",
        f"| Measured | {measured} |",
        f"| Prompt / output tokens | {work['prompt_tokens']:,} / {work['output_tokens']:,} |",
        f"| Concurrencies | {', '.join(str(c) for c in work['concurrency'])} |",
        f"| Warmup / measured waves | {work['warmup_waves']} / {work['measurement_waves']} |",
        f"| GPU | `{env0.get('arch', '?').split()[0]}`, "
        f"{env0.get('arch', '').split()[-1] if 'memory_gib' in env0.get('arch', '') else '?'} GiB |",
        f"| torch / tokenspeed-triton | {env0.get('torch', '?')} / "
        f"{env0.get('tokenspeed-triton', '?')} |",
        f"| Image | `{image}` |",
        "",
        "## Communication share of rank 0's kernel time",
        "",
        "| Model | Layout | " + " | ".join(STAGE_LABEL[s] for s in STAGES) + " |",
        "|---|---|" + "---:|" * len(STAGES),
        *summary,
        "",
        *(
            [
                "Batch 1 decode collectives on rank 0 against the same GPUs'",
                "all-reduce of four hidden-size rows (one verify step's tokens)",
                "replayed from a CUDA graph outside the server; see each model's",
                "Collectives section.",
                "",
                "| Model | Layout | Calls per step | In server (us per call) "
                "| Isolated, 4 rows (us) | Collectives per step (ms) |",
                "|---|---|---:|---:|---:|---:|",
                *collectives,
                "",
            ]
            if collectives else []
        ),
        "## Acceptance",
        "",
        "Logged values are medians over the server's 40-step decode windows;",
        "windows that straddle a prefill keep fewer tokens.",
        "",
        "| Model | Layout | Simulated | Logged |",
        "|---|---|---:|---:|",
        *(
            f"| {label} {algo} | {where} | {sim} | {'—' if acc is None else acc} |"
            for label, algo, where, sim, acc in accepts
        ),
        "",
        "TPOT p50 is per request over its whole decode, as CI reports it. At",
        "batch 16 every request arrives at once, so a request whose prompt",
        "finishes early decodes between other requests' prefill chunks, and its",
        "TPOT mostly measures prefill. Steady decode TPOT is the median over the",
        "server's 40-step decode windows in which every request was decoding and",
        "no prefill ran. A steady verify step is that window's wall time per",
        "forward: the logged accept length times the steady TPOT.",
        "",
        *sections,
        "## Method and limitations",
        "",
        *(
            [
                "- Timing and profiling are separate server runs with the same flags.",
                "  rocprofv3 traces the kernels inside the decode CUDA graphs too, so",
                "  the profile sees the collectives as the perf run issues them.",
            ]
            if modes == {"CUDA-graph"}
            else [
                "- Timing and profiling are separate server runs. Performance comes",
                "  from a run with decode CUDA graphs; kernels come from an eager run",
                "  without overlap scheduling, under rocprofv3. Eager collectives also",
                "  time how far the other ranks' kernel launches lag rank 0's.",
            ]
        ),
        "- Kernel tables are rank 0's; the Collectives sections cover every",
        "  rank on node 0. A rank's communication kernels include the time it",
        "  waits at each collective for the slowest rank.",
        "- The isolated all-reduce sends hidden-size bf16 rows; the server's",
        "  collectives can be wider or fused with a norm, so it shows what that",
        "  many hidden-size rows cost on these GPUs, not what each call sends.",
        "- Prompts are random token ids, so MoE routing is real but not that of",
        "  natural text.",
        "- Accept lengths are fixed, so these runs measure the cost of a verify",
        "  step at a given acceptance, not drafter quality.",
        "",
        "Regenerate this document with:",
        "",
        "```bash",
        "python3 real_serving/scripts/generate_hotspot_report.py \\",
        "  --run <MI455X result dir> \\",
        f"  --commit-date {args.commit_date} \\",
        f"  --output-dir real_serving/results/hotspots_{slug}",
        "```",
        "",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "README.md").write_text("\n".join(lines))
    print(f"wrote {args.output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
