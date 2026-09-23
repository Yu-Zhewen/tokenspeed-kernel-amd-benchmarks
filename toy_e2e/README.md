# Kimi-K3 toy logical-rank benchmark

This package measures Kimi-K3 TP8/EP1 on one AMD GPU per architecture and
produces a per-commit comparison of MI355X (`gfx950`) against MI455X
(`gfx1250`).

## What "toy logical rank" means

One physical GPU executes **rank 0 of a TP8 model**, with local substitutes
standing in for every rank-spanning collective. It is not TP1: the model is
configured exactly as rank 0 of an eight-way tensor-parallel deployment, so
every weight shard, expert assignment, and kernel shape matches what rank 0
would see in a real TP8 run.

What that buys, and what it costs:

| Property | Status |
|---|---|
| Per-rank kernel shapes and dtypes | real |
| Weight shards, MoE expert assignment, layer count | real |
| Scheduler, CUDA graph capture, rolling KV metadata | real, production `ModelExecutor` |
| Physical RCCL/Iris collectives | substituted locally |
| HTTP serving and tokenizer text | absent |
| MoE routing semantics | rank-local, so not equivalent to full TP8 |

So the numbers are a **compute estimate for one rank**, useful for comparing
architectures or commits against each other, and not a serving figure. Two
consequences worth keeping in mind: decoded output is not semantically valid
text because seven TP contributions are missing, and any change that shifts
communication cost is invisible here.

Prompts are deterministic varied synthetic token IDs (seed 7, vocabulary
160,000), not a repeated token, so prefill does real work rather than hitting
degenerate cache behaviour.

## Producing a per-commit comparison

Read [`docs/arch-comparison.md`](docs/arch-comparison.md). The short version is
four runs and one command:

1. Create a detached worktree at the commit on each host.
2. Smoke-test the harness against it with `--load-format dummy`, which catches
   API drift in about forty seconds instead of after a ten-minute checkpoint
   load.
3. Run `performance` then `hotspots` on each architecture, same commit, same
   workload.
4. Run `scripts/generate_arch_comparison.py` over the four resulting JSON files
   to write `results/arch_compare_<short-sha>/`.

Results live in [`results/`](results/README.md), one directory per commit.

## Package files

- `benchmark_logical_rank.py`: the rolling-graph performance run.
- `logical_rank.py`: TP8/EP1 rank-0 model configuration and the local
  collective substitutes.
- `workload.py`: deterministic synthetic-token generator.
- `rank_checkpoint.py`: portable raw rank-state writer and loader.
- `scripts/export_rank_local_checkpoint.py`: one-time rank-0 checkpoint export;
  see [`docs/checkpoint-preparation.md`](docs/checkpoint-preparation.md).
- `scripts/profile_logical_rank_stages.py`: per-stage GPU traces.
- `scripts/summarize_gpu_hotspots.py`: kernel aggregation shared by every
  hotspot result.
- `scripts/generate_arch_comparison.py`: builds a result entry from four JSON
  files.
- `scripts/collect_gpu_telemetry.py`, `scripts/source_tree_snapshot.py`:
  called by the host runner scripts to record telemetry and provenance.
- `scripts/summarize_attention_shapes.py`, `summarize_gemm_shapes.py`,
  `summarize_kimi3_moe_stages.py`, `summarize_stream_overlap.py`: optional
  per-area breakdowns from the same traces.

## Validate the package

`generate_arch_comparison.py` and its test need no GPU:

```bash
python3 -m pytest -q -p no:cacheprovider toy_e2e/tests/test_generate_arch_comparison.py
python3 -m ruff check toy_e2e
```

The remaining tests import `tokenspeed_kernel`, which requires a GPU device:

```bash
export PYTHONPATH="/path/to/this/repo:/path/to/tokenspeed/python:/path/to/tokenspeed/tokenspeed-kernel/python:/path/to/tokenspeed/tokenspeed-kernel-amd/python"
python3 -m pytest -q -p no:cacheprovider toy_e2e/tests
```

Do not run them while a measurement holds the GPU; they will perturb it.
