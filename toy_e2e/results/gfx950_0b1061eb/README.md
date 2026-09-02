# Kimi-K3 TP8/EP1 on gfx950 at `0b1061eb`

## Outcome

**PASS** for real Kimi-K3 TP8/EP1 serving on eight physical MI355X GPUs,
rank-local/full-source one-GPU validation, and per-setting hotspot capture.

The real service reached 78.22 output tok/s at C1 and 556.13 output tok/s at
C16 for exact 4096-input/1024-output requests. C16's final steady window
reached 816.84 output tok/s. All 51 measured requests completed with the
requested token counts and no failures.

## Revisions and topology

- Hardware: 8× AMD Instinct MI355X, `gfx950:sramecc+:xnack-`
- TokenSpeed and kernel source:
  `0b1061eb9fe1df36a4e48e5c9c291cd753af9e89`
- Kimi-K3 checkpoint revision:
  `eaf5a944bfc8c57438bbce226feef9f6bdbdaae1`
- Runtime: PyTorch `2.11.0+rocm7.2`, HIP `7.2.26015`,
  Transformers `5.12.0`, Triton `3.6.0`
- Parallelism: attention TP8, dense TP8, MoE TP8, EP1
- Collectives: RCCL plus fused Iris all-reduce kernels
- KV cache: FP8 E4M3; prefix cache and host KV store disabled
- Prefill: eager, 8192-token chunk and scheduling budget
- Decode: CUDA graphs at batch sizes 1, 2, 4, 8, and 16

`TORCH_NCCL_BLOCKING_WAIT=1` was required to avoid the ProcessGroupNCCL
watchdog polling an event while HIP graph capture was active.

## Real service metrics

The client used streaming raw `/v1/completions`, EvalScope random prompts,
greedy sampling, `ignore_eos=true`, closed-loop load, one full-concurrency
warmup, and three saturated measured waves.

| C | Success | Output tok/s | Total tok/s | Mean latency | TTFT p50 / p90 | TPOT p50 / p90 | Steady output tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3/3 | 78.22 | 391.08 | 13.092 s | 405.61 / 405.81 ms | 12.36 / 12.49 ms | 78.50 |
| 16 | 48/48 | 556.13 | 2,780.65 | 29.457 s | 4,596.48 / 4,727.20 ms | 24.28 / 27.60 ms | 816.84 |

Output throughput divides all completed output tokens by measured wall time,
so it includes both prefill and decode. The steady value is EvalScope's final
30-second-window estimate.

Mean request-time split:

| C | Mean TTFT | Mean post-TTFT | TTFT share | Post-TTFT share |
|---:|---:|---:|---:|---:|
| 1 | 0.405 s | 12.686 s | 3.10% | 96.90% |
| 16 | 4.030 s | 25.427 s | 13.68% | 86.32% |

The graph-mode server exposed 3,197,056 scheduler-visible cache tokens. A
fully resident C16 4K/1K workload needs at most 81,920 tokens, only 2.56% of
that capacity, so this result is not KV-capacity limited.

## Real service versus one logical rank

The one-GPU estimate executes the full 93-layer rank-0 model and production
decode graph but substitutes rank-spanning collectives locally. It therefore
excludes RCCL latency, synchronization, communication/compute overlap, HTTP,
and sampling.

| C | Real p50 TPOT | Logical-rank graph p50 | Real overhead | Real p50 decode tok/s | Logical-rank tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 12.36 ms | 10.736 ms | +15.12% | 80.93 | 93.13 |
| 16 | 24.28 ms | 18.470 ms | +31.46% | 41.18 | 54.13 |

At C16, real overall throughput is 35.79% below the 866.11 tok/s logical-rank
graph estimate, while the steady-window value is only 5.69% below it. The
difference between overall and steady throughput is prefill and request-wave
overhead, not a contradiction between the two rates.

Real p50 TTFT is 45.94% above the one-GPU first-token estimate at C1 and
135.55% above it at C16. Real queueing, rank synchronization, and the
multi-request prefill schedule are absent from the logical-rank run.

## One-GPU component breakdown

These are sampled eager component p50 values from the rank-local load path.
They aggregate each category across the complete 93-layer forward. The
instrumented profile uses one prefill sample at C1, eight prefill steps at
C16, and seven decode samples per setting.

| Setting and phase | Model p50 | MoE | KDA attention | MLA attention | Dense FFN | Hottest sampled module |
|---|---:|---:|---:|---:|---:|---|
| C1 prefill | 270.572 ms | 179.221 ms | 58.002 ms | 12.854 ms | 0.605 ms | layer 39 MoE, 2.087 ms |
| C1 decode | 68.415 ms | 29.225 ms | 15.517 ms | 10.520 ms | 0.129 ms | layer 67 MLA, 0.485 ms |
| C16 prefill step | 427.572 ms | 264.214 ms | 101.033 ms | 23.307 ms | 1.103 ms | layer 63 MoE, 2.947 ms |
| C16 decode | 81.635 ms | 36.496 ms | 12.956 ms | 13.943 ms | 0.116 ms | layer 39 MLA, 0.648 ms |

The top 15 layer/module records for each phase are retained in
`one_gpu_rank_local_4k_1k.json`.

## Eight-GPU kernel hotspot method

Service throughput was measured unprofiled in graph mode. Dynamic
PyTorch/roctracer profiling on this ROCm build does not expose kernels replayed
inside a CUDA graph. Kernel attribution therefore uses a separate,
otherwise-identical `--enforce-eager` server:

- prefill profiling is armed before request submission;
- decode captures exactly the first 64 forward batches;
- all eight TP-rank traces are included;
- percentages divide each exact kernel's summed duration by the stage's summed
  GPU kernel duration across all eight traces;
- complete exact-name CSVs retain calls, total milliseconds, GPU-time share,
  and average microseconds per call.

This follows the reporting contract used in
[TokenSpeed issue #78](https://github.com/raikonenfnu/tokenspeed/issues/78).
The graph observability caveat is also documented in
[issue #68](https://github.com/raikonenfnu/tokenspeed/issues/68).

Category summary:

| Setting and stage | Communication | MoE | KDA attention | MLA attention | Other / remaining |
|---|---:|---:|---:|---:|---:|
| C1 prefill | 21.41% | 35.16% | 12.20% | 7.14% | 24.09% |
| C1 decode, 64 batches | 88.55% | 1.51% | 2.04% | 2.88% | 5.02% |
| C16 prefill | 22.81% | 24.90% | 11.68% | 11.51% | 29.10% |
| C16 decode, 64 batches | 62.52% | 11.78% | 5.64% | 4.02% | 16.04% |

The C1 decode trace is dominated by three Iris collective kernels totaling
88.31% of summed GPU kernel time. At C16, communication remains the largest
decode category but falls to 62.52% as larger-batch MoE and attention work
become more significant.

Top exact-name kernels:

| Setting and stage | Exact kernel name | Calls | Total across ranks | GPU time |
|---|---|---:|---:|---:|
| C1 prefill | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1,528 | 601.47 ms | 20.77% |
| C1 prefill | `gluon_mxfp4_moe_stage1_kernel` | 736 | 496.32 ms | 17.14% |
| C1 prefill | `gluon_mxfp4_moe_stage2_1x2_kernel` | 736 | 375.09 ms | 12.95% |
| C1 decode | `iris_stage_one_shot_allreduce_residual_attnres_gluon_kernel` | 43,345 | 18,026.47 ms | 34.06% |
| C1 decode | `iris_stage_one_shot_allreduce_kernel` | 5,099 | 16,293.80 ms | 30.79% |
| C1 decode | `iris_reduce_symmetric_gluon_kernel` | 46,914 | 12,412.30 ms | 23.45% |
| C16 prefill | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 13,640 | 10,251.01 ms | 22.73% |
| C16 prefill | `gluon_mxfp4_moe_stage1_kernel` | 6,624 | 5,592.55 ms | 12.40% |
| C16 prefill | `_attn_res_rmsnorm_kernel` | 13,480 | 4,627.23 ms | 10.26% |
| C16 decode | `iris_stage_one_shot_allreduce_kernel` | 48,351 | 50,520.44 ms | 44.55% |
| C16 decode | `iris_reduce_symmetric_two_stage_gluon_kernel` | 46,814 | 11,846.94 ms | 10.45% |
| C16 decode | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 13,126 | 8,526.76 ms | 7.52% |

Summed kernel durations are hotspot weights, not critical-path wall time.
Collective kernel residency includes time waiting for peers. Rank-duration
spread and heavily perturbed eager timings must not be read as graph-mode
latency or pure compute imbalance. Category assignment is name-based; the
exact-name CSVs are the authoritative data.

## Rank-local artifact validation

The portable rank-local and full-source load paths still agree within 0.5%:

| C | Rank-local graph p50 | Full-source graph p50 | Difference |
|---:|---:|---:|---:|
| 1 | 10.736 ms | 10.691 ms | +0.42% |
| 16 | 18.470 ms | 18.546 ms | -0.41% |

The rank-local checkpoint loaded in 46.70 seconds versus 138.00 seconds for
the full source checkpoint on this run. Both retained 209.95 GiB of processed
model allocation and selected the same gfx950 kernels.

## Artifacts

Checked in:

- `real_8gpu_tp8ep1_4k_1k.json`: normalized unprofiled service data
- `one_gpu_rank_local_4k_1k.json`: portable-artifact estimator and components
- `one_gpu_full_source_4k_1k.json`: source-checkpoint reference
- `hotspots/eager_kernel_hotspots.json`: all categories, exact names, and ranks
- `hotspots/csv/{c1,c16}_{extend,decode}.csv`: complete exact-name tables
- `real_8gpu_graph_server.log` and `real_8gpu_eager_profile_server.log`
- `one_gpu_rank_local_4k_1k.log` and `one_gpu_full_source_4k_1k.log`

The complete EvalScope directory, 32 Chrome traces, per-profile manifests, and
failed-profiler diagnostic attempts remain under:

```text
/data/results/kimi-k3-real-tp8ep1-gfx950-0b1061eb/
```

## Validation

```text
python3 -m pytest -q -p no:cacheprovider toy_e2e/tests
8 passed

python3 -m ruff check toy_e2e
All checks passed!
```

Physical gfx1250 execution remains pending. Follow
[`../../docs/gfx1250-validation.md`](../../docs/gfx1250-validation.md) on one
MI450; do not infer gfx1250 performance from these gfx950 measurements.
