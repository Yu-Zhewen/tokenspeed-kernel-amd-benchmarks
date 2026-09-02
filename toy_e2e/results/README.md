# Kimi-K3 TP8/EP1 result index

Result directories use `<architecture>_<TokenSpeed short SHA>`. Do not compare
files from different directories as if hardware were the only changed
variable; the full TokenSpeed SHA, model revision, runtime, and command are
recorded in each result and report.

## Current collection

- [`gfx950` revision index at `0b1061eb`](gfx950_0b1061eb/README.md)
  - [`one_gpu/`](gfx950_0b1061eb/one_gpu/): logical TP8 rank-0 estimate,
    portable-artifact/full-source comparison, and model-component hotspots
  - [`real_8gpu/`](gfx950_0b1061eb/real_8gpu/): physical TP8/EP1 HTTP serving,
    all-rank kernel hotspots, and complete exact-name CSVs

The two folders deliberately keep estimator and serving artifacts separate.
Their reports use matching section headings and C1/C16 analysis tables, while
retaining the measurement unit appropriate to each run: model components for
the logical rank and GPU kernels for physical serving.

## Adding gfx1250

Run the physical MI450 procedure in
[`../docs/gfx1250-validation.md`](../docs/gfx1250-validation.md), then create:

```text
gfx1250_<TokenSpeed-short-SHA>/
  README.md
  one_gpu/
    README.md
    one_gpu_rank_local_4k_1k.json
    one_gpu_rank_local_4k_1k.log
```

Preserve the complete benchmark JSON and console log. Add a revision-specific
index and a scoped report matching the gfx950 layout. If the TokenSpeed
revision differs, discuss software and hardware changes separately rather
than presenting a direct architecture-only ratio.
