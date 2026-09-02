# Real eight-GPU TP8/EP1 result

## Overview

This result runs Kimi-K3 TP8/EP1 serving on eight physical MI355X GPUs. All
896 experts are tensor-parallel across all eight ranks; expert parallelism is
one.

Status: **PASS** for exact 4096-input/1024-output service at C1 and C16,
unprofiled graph-mode performance measurement, and stage-separated eager
kernel attribution across all ranks. All 51 measured requests completed with
the requested token counts and no failures.

## Environment

- Physical hardware: 8× AMD Instinct MI355X, `gfx950:sramecc+:xnack-`
- TokenSpeed and kernel source:
  `0b1061eb9fe1df36a4e48e5c9c291cd753af9e89`
- Kimi-K3:
  `eaf5a944bfc8c57438bbce226feef9f6bdbdaae1`
- Runtime: PyTorch `2.11.0+rocm7.2`, HIP `7.2.26015`,
  Transformers `5.12.0`, Triton `3.6.0`
- Topology: attention TP8, dense TP8, MoE TP8, EP1
- Collectives: RCCL plus fused Iris all-reduce kernels
- KV cache: FP8 E4M3; prefix cache and host KV store disabled
- Prefill: eager, 8192-token chunk and scheduling budget
- Decode: CUDA graphs at batch sizes 1, 2, 4, 8, and 16

`TORCH_NCCL_BLOCKING_WAIT=1` was required to avoid the ProcessGroupNCCL
watchdog polling an event while HIP graph capture was active.

## Measurement contract

- Prompt/output: exactly 4096/1024 tokens
- Concurrency: C1 and C16
- Client: streaming raw `/v1/completions`, EvalScope random prompts
- Sampling: greedy with `ignore_eos=true`
- Load: closed loop, one full-concurrency warmup, three measured waves
- Service metrics: unprofiled graph-mode server
- Hotspots: separate otherwise-identical `--enforce-eager` server
- Prefill profile: armed before request submission
- Decode profile: first 64 forward batches
- Kernel aggregation: summed GPU duration across all eight rank traces

Graph replay does not expose individual kernels to the available
PyTorch/roctracer profiler, so hotspot data is deliberately separated from
the graph-mode latency measurement.

## Performance results

| C | Success | Output tok/s | Total tok/s | Mean latency | TTFT p50 / p90 | TPOT p50 / p90 | Steady output tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3/3 | 78.22 | 391.08 | 13.092 s | 405.61 / 405.81 ms | 12.36 / 12.49 ms | 78.50 |
| 16 | 48/48 | 556.13 | 2,780.65 | 29.457 s | 4,596.48 / 4,727.20 ms | 24.28 / 27.60 ms | 816.84 |

Mean request-time split:

| C | Mean TTFT | Mean post-TTFT | TTFT share | Post-TTFT share |
|---:|---:|---:|---:|---:|
| 1 | 0.405 s | 12.686 s | 3.10% | 96.90% |
| 16 | 4.030 s | 25.427 s | 13.68% | 86.32% |

Output throughput includes prefill and decode over measured wall time. The
steady value is EvalScope's final 30-second-window estimate.

## Hotspot breakdown

Percentages divide each category or exact kernel's summed duration by the
stage's summed GPU kernel duration across all eight rank traces.

| Setting and phase | Profile scope | Dominant category | Category total / GPU share | Hottest exact kernel | Kernel total / GPU share |
|---|---|---|---:|---|---:|
| C1 prefill | full prefill | MoE | 1,018.53 ms / 35.16% | `ncclDevKernel_Generic_1(...)` | 601.47 ms / 20.77% |
| C1 decode | 64 forwards | communication | 46,859.31 ms / 88.55% | `iris_stage_one_shot_allreduce_residual_attnres_gluon_kernel` | 18,026.47 ms / 34.06% |
| C16 prefill | full prefill | MoE | 11,232.10 ms / 24.90% | `ncclDevKernel_Generic_1(...)` | 10,251.01 ms / 22.73% |
| C16 decode | 64 forwards | communication | 70,896.98 ms / 62.52% | `iris_stage_one_shot_allreduce_kernel` | 50,520.44 ms / 44.55% |

Top three exact kernels per setting:

| Setting and phase | Rank | Exact kernel | Calls | Total across ranks | GPU share |
|---|---:|---|---:|---:|---:|
| C1 prefill | 1 | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1,528 | 601.47 ms | 20.77% |
| C1 prefill | 2 | `gluon_mxfp4_moe_stage1_kernel` | 736 | 496.32 ms | 17.14% |
| C1 prefill | 3 | `gluon_mxfp4_moe_stage2_1x2_kernel` | 736 | 375.09 ms | 12.95% |
| C1 decode | 1 | `iris_stage_one_shot_allreduce_residual_attnres_gluon_kernel` | 43,345 | 18,026.47 ms | 34.06% |
| C1 decode | 2 | `iris_stage_one_shot_allreduce_kernel` | 5,099 | 16,293.80 ms | 30.79% |
| C1 decode | 3 | `iris_reduce_symmetric_gluon_kernel` | 46,914 | 12,412.30 ms | 23.45% |
| C16 prefill | 1 | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 13,640 | 10,251.01 ms | 22.73% |
| C16 prefill | 2 | `gluon_mxfp4_moe_stage1_kernel` | 6,624 | 5,592.55 ms | 12.40% |
| C16 prefill | 3 | `_attn_res_rmsnorm_kernel` | 13,480 | 4,627.23 ms | 10.26% |
| C16 decode | 1 | `iris_stage_one_shot_allreduce_kernel` | 48,351 | 50,520.44 ms | 44.55% |
| C16 decode | 2 | `iris_reduce_symmetric_two_stage_gluon_kernel` | 46,814 | 11,846.94 ms | 10.45% |
| C16 decode | 3 | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 13,126 | 8,526.76 ms | 7.52% |

The complete exact-name CSVs retain calls, total milliseconds, GPU-time
share, and average microseconds per call. This follows the reporting contract
in [TokenSpeed issue #78](https://github.com/raikonenfnu/tokenspeed/issues/78);
the graph-observability limitation is documented in
[issue #68](https://github.com/raikonenfnu/tokenspeed/issues/68).

## Interpretation

- C16 raises overall output throughput from 78.22 to 556.13 tok/s and reaches
  816.84 tok/s in the final steady window.
- Communication is 88.55% of summed C1 decode GPU residency and remains the
  largest C16 decode category at 62.52%.
- MoE is the largest prefill category at both concurrency settings.
- C16 increases compute work per decode batch, reducing communication's share
  while increasing total serving throughput.

## Artifacts

- [`real_8gpu_tp8ep1_4k_1k.json`](real_8gpu_tp8ep1_4k_1k.json):
  normalized unprofiled service data
- [`hotspots/eager_kernel_hotspots.json`](hotspots/eager_kernel_hotspots.json):
  category, exact-name, and per-rank aggregates
- [`hotspots/csv/`](hotspots/csv/): complete C1/C16 prefill/decode tables
- [`real_8gpu_graph_server.log`](real_8gpu_graph_server.log):
  unprofiled service log
- [`real_8gpu_eager_profile_server.log`](real_8gpu_eager_profile_server.log):
  eager attribution server log

The complete EvalScope directory, 32 Chrome traces, per-profile manifests, and
failed-profiler diagnostics remain at:

```text
/data/results/kimi-k3-real-tp8ep1-gfx950-0b1061eb/
```

## Validation and limitations

```text
python3 -m pytest -q -p no:cacheprovider toy_e2e/tests
8 passed

python3 -m ruff check toy_e2e
All checks passed!
```

Summed kernel durations are hotspot weights, not critical-path wall time.
Collective residency includes time waiting for peers, eager profiling heavily
perturbs timings, and name-based categories are secondary to the exact-name
CSV data.
