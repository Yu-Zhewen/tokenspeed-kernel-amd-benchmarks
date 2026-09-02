# Kimi-K3 gfx1250 toy 1-GPU result (pending)

## Status

- Collection date: unavailable
- Target: gfx1250 toy 1-GPU logical TP8 rank 0
- Overall status: **pending physical run**
- Performance cases: C1 unavailable, C16 unavailable
- Stage profiles: C1 prefill unavailable, C1 decode unavailable, C16 prefill
  unavailable, C16 decode unavailable
- Missing or invalid data: all physical gfx1250 measurements

## Software and hardware setup

| Field | Value |
|---|---|
| Device | unavailable; physical gfx1250 required |
| Architecture | `gfx1250` expected |
| Physical GPUs / ranks | 1 / 1 expected |
| Measurement environment | physical required |
| Host / container | unavailable |
| OS | unavailable |
| ROCm / HIP | unavailable |
| PyTorch | unavailable |
| Transformers | unavailable |
| Triton package / module | prebuilt TokenSpeed Triton required; version unavailable |
| EvalScope | unavailable; direct logical-rank harness |
| TokenSpeed commit | `0b1061eb9fe1df36a4e48e5c9c291cd753af9e89` required for the matched set |
| Kimi-K3 revision | `eaf5a944bfc8c57438bbce226feef9f6bdbdaae1` required |

## Workload and topology

| Field | Value |
|---|---|
| Checkpoint | portable raw-rank-state; path unavailable |
| Prompt / output | 4096 / 1024 tokens required |
| Concurrency | 1 and 16 required |
| Prefill budget | 8192 tokens required |
| Attention / dense / MoE / EP | TP8 / TP8 / TP8 / EP1 required |
| KV cache | FP8 E4M3, capacity unavailable |
| Prefix cache / host KV | disabled / disabled required |
| Sampling | rank-local greedy, `ignore_eos=true` |
| Prompt source | deterministic varied IDs, seed 7, vocabulary range 160,000 |
| Warmup / measured requests | C1: 1 / 3; C16: 16 / 48 |
| Decode graphs / scheduling | buckets 1/2/4/8/16; depth-1 dispatch/commit overlap |
| Performance measurement | complete rolling `ModelExecutor` CUDA-graph workload |
| Hotspot measurement | separate eager PyTorch/roctracer run; varied prompts and deterministic decode input ID 1 |

One physical gfx1250 GPU must execute logical TP8 rank 0. Rank-spanning
collectives remain local substitutes. FFM or AM output must not be entered as
physical performance.

## Correctness

| C | Requests or sequences completed | Exact input length | Exact output length | Failures | Status |
|---:|---:|---:|---:|---:|---|
| 1 | unavailable | 4096 required | 1024 required | unavailable | pending |
| 16 | unavailable | 4096 required | 1024 required | unavailable | pending |

## Unprofiled performance

| C | First-token p50 / p90 | Primary decode p50 / p90 | Per-user decode | Overall output | Steady decode capacity | Scope |
|---:|---:|---:|---:|---:|---:|---|
| 1 | unavailable | unavailable | unavailable | unavailable | unavailable | physical gfx1250 pending |
| 16 | unavailable | unavailable | unavailable | unavailable | unavailable | physical gfx1250 pending |

Primary decode will be request TPOT from the complete rolling graph workload.
Steady capacity will be `concurrency / mean rolling decode-step latency` after
the first decode transition.

| C | Decode input context | Resulting context | Step p50 / p90 | Samples |
|---:|---:|---:|---:|---:|
| 1 | 4097–5119 required | 4098–5120 required | unavailable | unavailable |
| 16 | 4097–5119 required | 4098–5120 required | unavailable | unavailable |

## Stage hotspot summary

| C | Stage | Ranks | Forwards | Summed GPU time | Communication | MoE | KDA | MLA / attention | GEMM / quant | Other |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | prefill | 1 required | 1 required | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 1 | decode | 1 required | 64 required | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 16 | prefill | 1 required | 8 required | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| 16 | decode | 1 required | 64 required | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

## Top exact kernels

When collected, this table must show the top 10 exact GPU-profiler kernel
names for each C1/C16 prefill/decode setting, not model-component or layer
names.

| C | Stage | Order | Exact kernel name | Calls | Total across ranks | GPU share | Average call |
|---:|---|---:|---|---:|---:|---:|---:|
| unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

## Incomplete or failed work

| Case | Stage | Status | Error or reason | Raw artifact |
|---|---|---|---|---|
| C1 and C16 | performance | pending | no physical gfx1250 collection yet | unavailable |
| C1 and C16 | prefill and decode profiles | pending | no physical gfx1250 collection yet | unavailable |

## Exact commands

### Performance

```bash
python3 toy_e2e/benchmark_logical_rank.py \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --expected-arch gfx1250 \
  --tokenspeed-revision 0b1061eb9fe1df36a4e48e5c9c291cd753af9e89 \
  --model-revision eaf5a944bfc8c57438bbce226feef9f6bdbdaae1 \
  --container-image "<physical-gfx1250-image-and-ID>" \
  --prompt-tokens 4096 --output-tokens 1024 \
  --concurrency 1 16 --chunked-prefill-size 8192 --cache-gib 32 \
  --warmup-waves 1 --measurement-waves 3 \
  --prompt-seed 7 --synthetic-vocabulary-size 160000 \
  --output result.json
```

### Stage profiles

```bash
TOKENSPEED_KERNEL_PROFILE_BACKEND=roctracer \
python3 toy_e2e/scripts/profile_logical_rank_stages.py \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --expected-arch gfx1250 \
  --output-dir /data/results/kimi-k3-toy-1gpu-gfx1250-0b1061eb-rolling/eager-profile \
  --tokenspeed-revision 0b1061eb9fe1df36a4e48e5c9c291cd753af9e89 \
  --model-revision eaf5a944bfc8c57438bbce226feef9f6bdbdaae1 \
  --container-image "<physical-gfx1250-image-and-ID>" \
  --prompt-tokens 4096 --concurrency 1 16 \
  --chunked-prefill-size 8192 --cache-gib 32 \
  --prompt-seed 7 --synthetic-vocabulary-size 160000 \
  --decode-steps 64
```

### Hotspot aggregation

```bash
python3 toy_e2e/scripts/summarize_gpu_hotspots.py \
  --input /data/results/kimi-k3-toy-1gpu-gfx1250-0b1061eb-rolling/eager-profile \
  --top-k 15 \
  --csv-dir toy_e2e/results/gfx1250_toy_1gpu_pending/hotspots/csv \
  --output toy_e2e/results/gfx1250_toy_1gpu_pending/hotspots/hotspots.json

python3 toy_e2e/scripts/update_result_readme_hotspots.py \
  --readme toy_e2e/results/gfx1250_toy_1gpu_pending/README.md \
  --hotspots toy_e2e/results/gfx1250_toy_1gpu_pending/hotspots/hotspots.json \
  --csv-dir toy_e2e/results/gfx1250_toy_1gpu_pending/hotspots/csv \
  --profile-manifest /data/results/kimi-k3-toy-1gpu-gfx1250-0b1061eb-rolling/eager-profile/profile_manifest.json
```

See [`../../RUNBOOK.md`](../../RUNBOOK.md), section 4, for physical setup,
artifact transfer, and validation.

## Raw artifacts

- Primary result JSON: unavailable
- Complete run log: unavailable
- Hotspot summary: unavailable
- Exact-name CSVs: unavailable
- Raw traces and manifests: unavailable
- Service logs and EvalScope outputs: unavailable for a direct logical-rank run

## Conclusions and limitations

- No physical gfx1250 performance has been collected.
- No hotspot claim can be made before all four required profiles exist.
- The portable checkpoint is architecture-capable, but that is not a
  substitute for a physical benchmark.
- Rename this directory to `gfx1250_toy_1gpu_<TokenSpeed-short-SHA>` only after
  the result is complete.

Profiled GPU-duration sums will be hotspot weights, not critical-path latency.
Do not report FFM or AM model time as hardware latency.
