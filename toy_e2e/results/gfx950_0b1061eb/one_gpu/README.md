# One-GPU logical-rank result

## Overview

This result runs one physical MI355X as logical TP8 rank 0 for Kimi-K3
TP8/EP1. It executes the complete 93-layer rank-local model and real weights,
while rank-spanning collectives are replaced by local traffic-recording
substitutes.

Status: **PASS** for the portable rank-local load path, full-checkpoint
reference, 4K/1K scheduler progression, component profiling, and CUDA-graph
decode.

## Environment

- Physical hardware: 1× AMD Instinct MI355X, `gfx950:sramecc+:xnack-`
- TokenSpeed:
  `0b1061eb9fe1df36a4e48e5c9c291cd753af9e89`
- Kimi-K3:
  `eaf5a944bfc8c57438bbce226feef9f6bdbdaae1`
- Runtime: PyTorch `2.11.0+rocm7.2`, HIP `7.2.26015`,
  Transformers `5.12.0`, Triton `3.6.0`
- Logical topology: attention TP8, dense TP8, MoE TP8, EP1, rank 0
- Model allocation after preprocessing: 209.95 GiB
- KV cache: 32 GiB, 2,334,336 scheduler-visible tokens

## Measurement contract

- Prompt/output: exactly 4096/1024 tokens
- Concurrency: C1 and C16
- Prefill budget: 8192 tokens
- Eager workload: all 1024 output tokens
- Component profile: one C1 or eight C16 prefill steps, then seven decode steps
- Graph profile: first full decode batch, 20 replays
- Communication: shapes and bytes recorded, but no RCCL execution

The graph result is the best rank-compute estimate. The eager run validates
scheduler and cache progression; its Python and launch overhead should not be
used as the production decode estimate.

## Performance results

| C | First-token p50 | Graph decode p50 | Per-user graph decode | Aggregate graph decode | Full eager-workload output |
|---:|---:|---:|---:|---:|---:|
| 1 | 277.93 ms | 10.736 ms | 93.13 tok/s | 93.13 tok/s | 15.01 tok/s |
| 16 | 1,951.35 ms | 18.470 ms | 54.13 tok/s | 866.11 tok/s | 198.66 tok/s |

Portable rank-local versus full-source validation:

| C | Rank-local graph p50 | Full-source graph p50 | Difference |
|---:|---:|---:|---:|
| 1 | 10.736 ms | 10.691 ms | +0.42% |
| 16 | 18.470 ms | 18.546 ms | -0.41% |

The portable artifact loaded in 46.70 seconds versus 138.00 seconds for the
full checkpoint, a 2.96× speedup. Both paths selected the same gfx950 kernels
and retained the same model allocation.

## Hotspot breakdown

Component values aggregate each category across all 93 layers. Percentages
use the instrumented model-forward p50 as denominator.

| Setting and phase | Profile scope | Dominant category | Category p50 / model share | Hottest module | Module p50 / model share |
|---|---|---|---:|---|---:|
| C1 prefill | 1 forward | MoE | 179.221 ms / 66.24% | layer 39 MoE | 2.087 ms / 0.77% |
| C1 decode | 7 forwards | MoE | 29.225 ms / 42.72% | layer 67 MLA | 0.485 ms / 0.71% |
| C16 prefill | 8 forwards | MoE | 264.214 ms / 61.79% | layer 63 MoE | 2.947 ms / 0.69% |
| C16 decode | 7 forwards | MoE | 36.496 ms / 44.71% | layer 39 MLA | 0.648 ms / 0.79% |

Top three timed modules per setting:

| Setting and phase | Rank | Category | Layer | Module p50 |
|---|---:|---|---:|---:|
| C1 prefill | 1 | MoE | 39 | 2.087 ms |
| C1 prefill | 2 | MoE | 55 | 2.084 ms |
| C1 prefill | 3 | MoE | 34 | 2.063 ms |
| C1 decode | 1 | MLA attention | 67 | 0.485 ms |
| C1 decode | 2 | MLA attention | 7 | 0.470 ms |
| C1 decode | 3 | MLA attention | 3 | 0.470 ms |
| C16 prefill | 1 | MoE | 63 | 2.947 ms |
| C16 prefill | 2 | MoE | 53 | 2.944 ms |
| C16 prefill | 3 | MoE | 92 | 2.942 ms |
| C16 decode | 1 | MLA attention | 39 | 0.648 ms |
| C16 decode | 2 | MLA attention | 67 | 0.625 ms |
| C16 decode | 3 | MoE | 81 | 0.600 ms |

The JSON retains the top 15 layer/module records for every setting and phase.

## Interpretation

- Rank-local and full-source graph medians agree within 0.5%, so the portable
  artifact introduces no measurable steady-state regression.
- MoE is the largest sampled component in every phase.
- Individual MLA layers are the hottest decode modules even though summed MoE
  remains the largest full-model category.
- C16 increases graph latency from 10.736 to 18.470 ms but raises aggregate
  rank-compute capacity from 93.13 to 866.11 tok/s.

## Artifacts

- [`one_gpu_rank_local_4k_1k.json`](one_gpu_rank_local_4k_1k.json):
  primary portable-artifact result
- [`one_gpu_rank_local_4k_1k.log`](one_gpu_rank_local_4k_1k.log):
  complete primary console log
- [`one_gpu_full_source_4k_1k.json`](one_gpu_full_source_4k_1k.json):
  full-checkpoint reference
- [`one_gpu_full_source_4k_1k.log`](one_gpu_full_source_4k_1k.log):
  complete reference console log

## Validation and limitations

```text
python3 -m pytest -q -p no:cacheprovider toy_e2e/tests
8 passed

python3 -m ruff check toy_e2e
All checks passed!
```

This is not real TP8 serving throughput. It excludes RCCL latency,
cross-rank synchronization, communication/compute overlap, HTTP, tokenization,
and sampling.
