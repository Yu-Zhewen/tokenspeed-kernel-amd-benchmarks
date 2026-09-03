# Kimi-K3 toy-rank and real-TP8 test plan

## Goal

Maintain exactly three result targets for the same Kimi-K3 TP8/EP1 workload:

| ID | Target | Physical GPUs | TP ranks | Status |
|---|---|---:|---:|---|
| G950-T1 | gfx950 toy logical rank | 1 | rank 0 executed; seven ranks substituted | required and collected |
| G950-R8 | gfx950 real TP8/EP1 serving | 8 | ranks 0–7 executed | required and collected |
| G1250-T1 | gfx1250 toy logical rank | 1 | rank 0 executed; seven ranks substituted | required and collected |

“Toy 1-GPU” means one physical GPU running logical TP8 rank 0. It does not
mean TP1, and it is not an eight-GPU service measurement.

## Matched workload

All three targets must use:

- Kimi-K3 revision
  `eaf5a944bfc8c57438bbce226feef9f6bdbdaae1`;
- TokenSpeed revision
  `0b1061eb9fe1df36a4e48e5c9c291cd753af9e89`, unless a new result directory
  and report are created for another revision;
- attention TP8, dense TP8, MoE TP8, EP1;
- 4096 input tokens and 1024 output tokens;
- concurrency 1 and 16;
- an 8192-token prefill budget;
- FP8 E4M3 KV cache;
- no prefix cache and no host KV store;
- decode CUDA-graph buckets 1, 2, 4, 8, and 16 with eager prefill;
- one complete warmup wave followed by three measured closed-loop waves
  (C1: 1 + 3 requests; C16: 16 + 48 requests);
- separate unprofiled performance and eager hotspot runs.

The toy targets use the same portable raw rank-0 checkpoint. Architecture-
Toy prompts are deterministic varied token IDs (seed 7, IDs below 160,000),
not repeated token 1 and not EvalScope text. Their greedy decode outputs are
rank-local because seven TP ranks and their reductions are absent.

## Required performance output

Each target reports both C1 and C16:

- exact request/sequence completion and token counts;
- first-token p50 and p90;
- primary decode p50 and p90;
- rolling decode-input context samples from 4097 through 5119, with the final
  generated token completing context 5120;
- per-user decode rate;
- overall 4K/1K output throughput;
- steady decode capacity;
- complete raw JSON and console log.

This is a performance contract, not an output-accuracy qualification. Record
missing KV scales or other accuracy-relevant runtime warnings in each report.

For toy targets, primary decode latency comes from the complete rolling
`ModelExecutor` CUDA-graph path with KV, sequence, position, and sampled-token
state updated every step. The control loop uses TokenSpeed's depth-1 overlap:
it dispatches step N before committing step N-1, rather than synchronizing the
whole device after the current step. For the real target, primary decode is
request-level TPOT from EvalScope. The report must label this difference and
must not treat toy latency as physical TP8 serving latency.

The first 4097-context inter-token sample may include remaining C16 prompt
chunks for requests that produced their first token early. Preserve that
scheduling effect in context reporting, but exclude the first decode
transition from the separate steady-capacity denominator.

## Required hotspot output

Collect the same four eager profiles for every target:

Toy eager traces use the shared varied prompts. Because the bare profiling
runner intentionally bypasses `ModelExecutor` sampling, its decode input is
the documented deterministic rank-local token ID 1; these traces are for
kernel attribution, not performance or output semantics.

| Case | Concurrency | Stage | Required forwards | Rank traces |
|---|---:|---|---:|---:|
| P-C1 | 1 | prefill / `EXTEND` | complete prefill (1 at the validated budget) | all physical ranks |
| D-C1 | 1 | decode / `DECODE` | 64 | all physical ranks |
| P-C16 | 16 | prefill / `EXTEND` | complete prefill (8 at the validated budget) | all physical ranks |
| D-C16 | 16 | decode / `DECODE` | 64 | all physical ranks |

Each hotspot summary must contain:

- summed GPU-kernel time and kernel-call counts;
- semantic categories using the shared classifier;
- exact kernel names, calls, total milliseconds, GPU-time percentage, and
  average microseconds;
- the top 10 exact profiler kernel names for each C1/C16 prefill/decode
  setting in the result README;
- per-rank totals and imbalance for a multi-rank run;
- complete exact-name CSVs for all four profiles.

Percentages use summed kernel duration across all captured ranks as the
denominator. They are attribution weights, not wall-clock decomposition.

## Collection model

1. Run performance without any profiler.
2. Run a separate eager workload for stage attribution.
3. Start eager profiling immediately before the first target-stage forward.
4. Stop after the complete prefill or exactly 64 decode forwards.
5. Preserve raw traces outside git when they are too large.
6. Normalize traces with the same `summarize_gpu_hotspots.py` revision.
7. Complete a copy of [`RESULT_TEMPLATE.md`](RESULT_TEMPLATE.md), including
   unavailable and failed fields.

Do not compare profiled eager timings with unprofiled graph/service latency.
Do not sum concurrent rank durations and call the result elapsed time.

## Result layout

```text
toy_e2e/
  README.md
  TEST_PLAN.md
  RUNBOOK.md
  RESULT_TEMPLATE.md
  results/
    README.md
    gfx950_toy_1gpu_<TokenSpeed-short-SHA>/
      README.md
      result.json
      run.log
      hotspots/
        hotspots.json
        csv/{c1,c16}_{extend,decode}.csv
    gfx950_real_8gpu_<TokenSpeed-short-SHA>/
      README.md
      result.json
      logs/
      hotspots/
        hotspots.json
        csv/{c1,c16}_{extend,decode}.csv
    gfx1250_toy_1gpu_<TokenSpeed-short-SHA>/
      README.md
      result.json
      run.log
      hotspots/
        hotspots.json
        csv/{c1,c16}_{extend,decode}.csv
```

The collected G1250-T1 result uses the same revision-scoped naming and
complete artifact contract as the gfx950 toy target.

## Completion checklist

- The physical architecture string matches the target.
- Exactly one GPU is visible for toy targets and eight for the real target.
- Both C1 and C16 complete with exact 4096/1024 token counts.
- Toy graph capture contains batches 1, 2, 4, 8, and 16 and both measured
  concurrencies use their exact graph bucket.
- Toy decode progresses through input contexts 4097–5119 and finishes at
  context 5120 without using the old static first-decode snapshot.
- All four stage profiles exist with the required physical-rank count.
- Performance and profiling were collected in separate runs.
- The report follows `RESULT_TEMPLATE.md` without deleting unavailable rows.
- Commands, revisions, raw artifact locations, failures, and caveats are
  recorded.
