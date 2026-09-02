# Kimi-K3 `<gfx950|gfx1250>` `<toy 1-GPU|real 8-GPU>` result at `<TokenSpeed short SHA>`

## Status

- Collection date: `<UTC timestamp>`
- Target: `<gfx950 toy 1-GPU | gfx950 real 8-GPU | gfx1250 toy 1-GPU>`
- Overall status: `<complete | incomplete | blocked>`
- Performance cases: `<C1 status>`, `<C16 status>`
- Stage profiles: `<C1 prefill>`, `<C1 decode>`, `<C16 prefill>`,
  `<C16 decode>`
- Missing or invalid data: `<none or explicit list>`

## Software and hardware setup

| Field | Value |
|---|---|
| Device | `<physical product name>` |
| Architecture | `<full architecture string>` |
| Physical GPUs / ranks | `<count>` |
| Measurement environment | `<physical | FFM | AM>` |
| Host / container | `<host and image identifier>` |
| OS | `<distribution and version>` |
| ROCm / HIP | `<version>` |
| PyTorch | `<version>` |
| Transformers | `<version>` |
| Triton package / module | `<package and module version>` |
| EvalScope | `<version or unavailable>` |
| TokenSpeed commit | `<full SHA>` |
| Kimi-K3 revision | `<full revision>` |

## Workload and topology

| Field | Value |
|---|---|
| Checkpoint | `<path and format>` |
| Prompt / output | `4096 / 1024 tokens` |
| Concurrency | `1 and 16` |
| Prefill budget | `8192 tokens` |
| Attention / dense / MoE / EP | `<TP / TP / TP / EP>` |
| KV cache | `<dtype and capacity>` |
| Prefix cache / host KV | `<enabled or disabled>` |
| Sampling | `<greedy, ignore_eos=true>` |
| Prompt source | `<EvalScope text or deterministic varied synthetic IDs>` |
| Warmup / measured requests | `<C1 and C16 counts>` |
| Decode graphs / scheduling | `<capture buckets and overlap mode>` |
| Performance measurement | `<unprofiled scope>` |
| Hotspot measurement | `<separate eager scope>` |

For a toy result, state that one physical GPU executes logical TP rank 0 and
that rank-spanning collectives are local substitutes. For a real result, state
the physical TP ranks and collective implementation.

## Correctness

| C | Requests or sequences completed | Exact input length | Exact output length | Failures | Status |
|---:|---:|---:|---:|---:|---|
| 1 | `<value>` | `<value>` | `<value>` | `<value>` | `<pass/fail>` |
| 16 | `<value>` | `<value>` | `<value>` | `<value>` | `<pass/fail>` |

## Unprofiled performance

| C | First-token p50 / p90 | Primary decode p50 / p90 | Per-user decode | Overall output | Steady decode capacity | Scope |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `<ms>` | `<ms>` | `<tok/s>` | `<tok/s>` | `<tok/s>` | `<scope>` |
| 16 | `<ms>` | `<ms>` | `<tok/s>` | `<tok/s>` | `<tok/s>` | `<scope>` |

For toy runs, the primary decode metric is request TPOT from the complete
rolling `ModelExecutor` CUDA-graph workload. Steady capacity is
`batch / mean rolling decode-step latency` after the first decode transition;
the 4097-context sample separately preserves any prefill interference. Overall
output includes eager prefill plus all 1024 generated tokens across three
measured closed-loop waves. For real runs, primary decode is EvalScope request
TPOT, overall output is measured service throughput, and steady capacity is
EvalScope's final window. These scopes must be stated; do not present toy
latency as physical TP8 service latency.

For each toy result, report the sampled rolling decode contexts. A model
forward consumes contexts 4097–5119; its last output completes context 5120.

| C | Decode input context | Resulting context | Step p50 / p90 | Samples |
|---:|---:|---:|---:|---:|
| `<C>` | `<4097..5119 checkpoint>` | `<input + 1>` | `<ms>` | `<N>` |

## Stage hotspot summary

Use the separate eager traces, never the unprofiled performance run.
Percentages divide category duration by summed GPU-kernel duration across all
captured physical ranks.

| C | Stage | Ranks | Forwards | Summed GPU time | Communication | MoE | KDA | MLA / attention | GEMM / quant | Other |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | prefill | `<N>` | `<N>` | `<ms>` | `<%>` | `<%>` | `<%>` | `<%>` | `<%>` | `<%>` |
| 1 | decode | `<N>` | `64` | `<ms>` | `<%>` | `<%>` | `<%>` | `<%>` | `<%>` | `<%>` |
| 16 | prefill | `<N>` | `<N>` | `<ms>` | `<%>` | `<%>` | `<%>` | `<%>` | `<%>` | `<%>` |
| 16 | decode | `<N>` | `64` | `<ms>` | `<%>` | `<%>` | `<%>` | `<%>` | `<%>` | `<%>` |

`Other` includes normalization, elementwise/reduction, cache, sampling, and
unclassified kernels. The machine-readable hotspot JSON and exact-name CSVs
are authoritative.

## Top exact kernels

Include exactly the top ten GPU kernels by total profiled GPU duration for
every concurrency/stage pair. These must be exact profiler kernel names, not
model component, layer, or module names.

| C | Stage | Order | Exact kernel name | Calls | Total across ranks | GPU share | Average call |
|---:|---|---:|---|---:|---:|---:|---:|
| `<C>` | `<stage>` | `<1..10>` | `<exact profiler name>` | `<N>` | `<ms>` | `<%>` | `<µs>` |

## Incomplete or failed work

| Case | Stage | Status | Error or reason | Raw artifact |
|---|---|---|---|---|
| `<case or none>` | `<performance/profile>` | `<status>` | `<reason>` | `<path>` |

Do not omit pending, unavailable, or failed cases. Use `unavailable` rather
than leaving a required cell blank.

## Exact commands

### Performance

```bash
<exact command>
```

### Stage profiles

```bash
<exact command or commands>
```

### Hotspot aggregation

```bash
<exact summarize_gpu_hotspots.py command>
```

## Raw artifacts

- Primary result JSON: `<path>`
- Complete run log: `<path>`
- Hotspot summary: `<path>`
- Exact-name CSVs: `<path>`
- Raw traces and manifests: `<path or unavailable>`
- Service logs and EvalScope outputs: `<path or unavailable>`

## Conclusions and limitations

- `<primary performance conclusion>`
- `<primary prefill hotspot conclusion>`
- `<primary decode hotspot conclusion>`
- `<scope-specific limitation>`

Profiled GPU-duration sums are hotspot weights, not critical-path latency.
Collective residency may include peer waiting, and profiler instrumentation
perturbs execution. Never substitute profiled timing for the unprofiled
performance table.
