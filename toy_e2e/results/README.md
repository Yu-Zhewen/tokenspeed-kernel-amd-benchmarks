# Kimi-K3 TP8/EP1 result index

Result directories use `<architecture>_<TokenSpeed short SHA>`. Do not compare
files from different directories as if hardware were the only changed
variable; the full TokenSpeed SHA, model revision, runtime, and command are
recorded in each result and report.

## Current collection

- [`gfx950` report at `0b1061eb`](gfx950_0b1061eb/README.md)
- [`gfx950_0b1061eb/`](gfx950_0b1061eb/): outputs for:
  - real Kimi-K3 HTTP serving on eight physical MI355X GPUs;
  - the one-GPU rank-local estimator loaded from the portable artifact;
  - the same one-GPU estimator loaded from the full source checkpoint;
  - C1 and C16 eager `EXTEND`/`DECODE` hotspot summaries;
  - complete exact-name kernel CSVs using the same columns as the TokenSpeed
    performance issues: calls, total time, GPU-time share, and average time.

The two one-GPU JSON files are load-path checks for the same estimator. The
file named `one_gpu_full_source_4k_1k.json` does **not** represent eight-GPU
serving. Only `real_8gpu_tp8ep1_4k_1k.json` contains measurements from real
RCCL-connected TP8 serving.

## Adding gfx1250

Run the physical MI450 procedure in
[`../docs/gfx1250-validation.md`](../docs/gfx1250-validation.md), then create:

```text
gfx1250_<TokenSpeed-short-SHA>/
  one_gpu_rank_local_4k_1k.json
  README.md
```

Preserve the complete benchmark JSON and console log. Add a revision-specific
report beside the gfx950 report; if the TokenSpeed revision differs, discuss
software and hardware changes separately rather than presenting a direct
architecture-only ratio.
