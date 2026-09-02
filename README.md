# TokenSpeed AMD kernel benchmarks

Reproducible AMD kernel benchmarks and early model-integration prototypes.

- [`attention/`](attention/): matched gfx950/gfx1250 attention benchmarks,
  plans, templates, and collected results.
- [`toy_e2e/`](toy_e2e/): matched Kimi-K3 TP8/EP1 toy 1-GPU and real 8-GPU
  benchmarks, with one test plan, runbook, result template, uniform kernel
  hotspots, collected gfx950 results, and a pending physical gfx1250 run.

The repository preserves the revision history imported from the original
benchmark gist. The toy E2E harness depends on a compatible TokenSpeed checkout;
its README pins the tested revision and execution environment.
