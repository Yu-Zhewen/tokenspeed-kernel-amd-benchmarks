# Kimi-K3 toy-rank and real-TP8 benchmarks

This package maintains three Kimi-K3 TP8/EP1 measurement targets:

| Target | Physical GPUs | Purpose | Status |
|---|---:|---|---|
| gfx950 toy 1-GPU | 1 | logical TP8 rank-0 compute estimate | [complete](results/gfx950_toy_1gpu_0b1061eb/) |
| gfx950 real 8-GPU | 8 | physical TP8/EP1 serving | [complete](results/gfx950_real_8gpu_0b1061eb/) |
| gfx1250 toy 1-GPU | 1 | logical TP8 rank-0 compute estimate | [pending](results/gfx1250_toy_1gpu_pending/) |

“Toy 1-GPU” means one physical GPU executes rank 0 of a TP8 model with local
substitutes for rank-spanning collectives. It is not TP1. “Real 8-GPU”
executes ranks 0–7 with physical RCCL/Iris collectives and HTTP serving.

## Documentation

- [`TEST_PLAN.md`](TEST_PLAN.md): exact workload, required metrics, profile
  coverage, naming, and completion rules.
- [`RUNBOOK.md`](RUNBOOK.md): step-by-step setup, performance collection, and
  hotspot collection for all three targets.
- [`RESULT_TEMPLATE.md`](RESULT_TEMPLATE.md): required uniform result README.
- [`results/README.md`](results/README.md): the three approved result entries.
- [`docs/checkpoint-preparation.md`](docs/checkpoint-preparation.md): portable
  raw TP8 rank-0 checkpoint contract.

The organization follows the repository's `attention/` benchmark pattern:
one explicit test contract, one reusable result template, revision-scoped raw
artifacts, complete/incomplete status, exact commands, and no silent omission
of unavailable data.

## Matched contract

All targets use:

- Kimi-K3 revision
  `eaf5a944bfc8c57438bbce226feef9f6bdbdaae1`;
- TokenSpeed revision
  `0b1061eb9fe1df36a4e48e5c9c291cd753af9e89` for the current result set;
- attention TP8, dense TP8, MoE TP8, EP1;
- exact 4096-token input and 1024-token output;
- concurrency 1 and 16;
- 8192-token prefill budget;
- FP8 E4M3 KV cache;
- no prefix cache or host KV store;
- unprofiled performance plus separate eager C1/C16 prefill/decode profiles.

Every hotspot result uses
[`scripts/summarize_gpu_hotspots.py`](scripts/summarize_gpu_hotspots.py).
It reports the same semantic categories and exact-name CSV columns for one
rank or eight ranks:

```text
kernel_name,calls,total_ms,gpu_time_pct,avg_us
```

This removes the prior mismatch where the toy report used model-component
hooks and the real report used GPU kernels.

## Current gfx950 results

| Target | C1 primary decode | C16 primary decode | C1 overall output | C16 overall output |
|---|---:|---:|---:|---:|
| toy 1-GPU logical rank | 10.711 ms graph | 18.553 ms graph | 15.20 tok/s eager | 196.90 tok/s eager |
| real 8-GPU serving | 12.36 ms TPOT | 24.28 ms TPOT | 78.22 tok/s | 556.13 tok/s |

The units share a table shape but not an execution scope. Toy graph latency
excludes physical communication and serving; real TPOT includes the physical
TP8 path. Each result README labels its scope.

The unified eager hotspot results show:

- toy prefill is MoE-heavy: 45.45% at C1 and 36.79% at C16;
- toy decode is led by GEMM/quant: 30.47% at C1 and 37.57% at C16;
- real prefill is split between MoE and communication;
- real decode is communication-dominated: 88.55% at C1 and 62.52% at C16.

## Package files

- `benchmark_logical_rank.py`: one-GPU scheduler, eager, graph, and logical
  collective benchmark.
- `logical_rank.py`: TP8/EP1 rank-0 model configuration and local collective
  substitutes.
- `rank_checkpoint.py`: portable raw rank-state writer and loader.
- `scripts/export_rank_local_checkpoint.py`: one-time rank-0 checkpoint export.
- `scripts/profile_logical_rank_stages.py`: toy prefill/decode GPU traces.
- `scripts/run_evalscope_4k1k.sh`: exact real-serving C1/C16 load contract.
- `scripts/collect_real_serving_results.py`: normalize EvalScope outputs.
- `scripts/profile_serving_stages.py`: real all-rank prefill/decode traces.
- `scripts/summarize_gpu_hotspots.py`: shared category and exact-kernel
  aggregation.

## Validate the package

Run in a matching TokenSpeed environment:

```bash
export PYTHONPATH="/path/to/this/repo:/path/to/tokenspeed/python:/path/to/tokenspeed/tokenspeed-kernel/python:/path/to/tokenspeed/tokenspeed-kernel-amd/python"
python3 -m pytest -q -p no:cacheprovider toy_e2e/tests
python3 -m ruff check toy_e2e
```

The tests cover checkpoint integrity, architecture-neutral load order,
logical collective accounting, result normalization, and hotspot aggregation.
