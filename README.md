# TokenSpeed AMD kernel benchmarks

Reproducible AMD kernel benchmarks and early model-integration prototypes.

- [`attention/`](attention/): matched gfx950/gfx1250 attention benchmarks,
  plans, templates, and collected results.
- [`toy_e2e/`](toy_e2e/): Kimi-K3 TP8/EP1 logical rank 0 on one GPU per
  architecture, producing a per-commit MI355X versus MI455X comparison.
- [`frontier_e2e/`](frontier_e2e/): shared logical-rank harness and portable
  checkpoint format for DeepSeek-V4.1-Flash and GLM-5.3-Flash on gfx950/gfx1250.

The repository preserves the revision history imported from the original
benchmark gist. The toy E2E harness runs against a TokenSpeed worktree at the
commit under test; see its README for the procedure and the harness
compatibility notes.
