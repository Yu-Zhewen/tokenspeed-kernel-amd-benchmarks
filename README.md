# TokenSpeed AMD kernel benchmarks

Reproducible AMD kernel benchmarks and early model-integration prototypes.

- [`attention/`](attention/): matched gfx950/gfx1250 attention benchmarks,
  plans, templates, and collected results.
- [`toy_e2e/`](toy_e2e/): single-MI450 logical TP8/EP8 Kimi-K3 integration
  test, reduced-checkpoint preparation tools, tests, findings, and rerun guide.

The repository preserves the revision history imported from the original
benchmark gist. The toy E2E harness depends on a compatible TokenSpeed checkout;
its README pins the tested revision and execution environment.
