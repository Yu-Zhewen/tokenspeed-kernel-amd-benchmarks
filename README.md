# TokenSpeed AMD kernel benchmarks

Reproducible AMD kernel benchmarks and early model-integration prototypes.

- [`attention/`](attention/): matched gfx950/gfx1250 attention benchmarks,
  plans, templates, and collected results.
- [`toy_e2e/`](toy_e2e/): full-depth Kimi-K3 TP8/EP1 rank benchmark on one
  physical gfx950 or gfx1250 GPU, with a portable raw rank-local checkpoint,
  scheduler-driven 4K/1K workloads, tests, results, and rerun guide.

The repository preserves the revision history imported from the original
benchmark gist. The toy E2E harness depends on a compatible TokenSpeed checkout;
its README pins the tested revision and execution environment.
