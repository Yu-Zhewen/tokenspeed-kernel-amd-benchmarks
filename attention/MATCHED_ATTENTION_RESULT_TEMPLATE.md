# Matched GFX950 and GFX1250 attention results

## Status

- Report owner: `<name or agent>`
- Collection date: `<UTC timestamp>`
- Overall status: `<complete | incomplete | blocked>`
- Missing or invalid cases: `<none or list>`

## Software and hardware setup

| Field | GFX950 | GFX1250 |
|---|---|---|
| Device | `<device name>` | `<device name or AM model>` |
| Architecture | `gfx950` | `gfx1250` |
| Measurement environment | `<physical>` | `<physical | FFM | AM>` |
| Host / container | `<host or container>` | `<host or zhewenyu-mi450-am>` |
| OS | `<distribution and version>` | `<distribution and version>` |
| ROCm / ROCDTIF | `<version>` | `<version>` |
| PyTorch | `<version>` | `<version>` |
| TokenSpeed commit SHA | `<full SHA>` | `<full SHA>` |
| tokenspeed-kernel commit SHA | `<full SHA>` | `<full SHA>` |
| tokenspeed-kernel-amd commit SHA | `<full SHA>` | `<full SHA>` |
| Kernel package / build ID | `<identifier>` | `<identifier>` |
| Triton package / build ID | `<identifier>` | `<identifier>` |
| Runner gist revision | `<gist revision SHA>` | `<same revision SHA>` |
| Warmups / repeats | `<warmups / repeats>` | `<warmups / repeats>` |

State any unavoidable setup difference:

`<none, or exact difference and why it does not change the logical workload>`

## Required test suite

| ID | Runner case | Required workload | GFX950 status | GFX1250 status |
|---|---|---|---|---|
| MLA-D1 | `mla-decode` | Kimi-K3 B1 decode, context 4096, FP8 absorbed Q and FP8 dense paged KV | `<pass/fail>` | `<pass/fail>` |
| MLA-P1 | `mla-prefill` | Kimi-K3 B1 pure prefill, extend 4096, causal FP8 Q/K/V | `<pass/fail>` | `<pass/fail>` |
| KDA-D1 | `kda-decode` | Kimi-K3 B1 one-token recurrent decode, BF16 inputs and FP32 indexed state | `<pass/fail>` | `<pass/fail>` |
| KDA-P1 | `kda-prefill` | Kimi-K3 B1 KDA pure prefill, extend 4096, BF16 inputs and FP32 state | `<pass/fail>` | `<pass/fail>` |
| DSA-D1 | `dsa-decode-pipeline` | GLM-5.2 B1 decode, context 4096, top-k 2048, live selected slots and dense FP8 KV | `<pass/fail>` | `<pass/fail>` |
| DSA-P1 | `dsa-prefill-pipeline-4k` | GLM-5.2 B1 pure prefill, extend 4096, causal top-k up to 2048 and dense FP8 KV | `<pass/fail>` | `<pass/fail>` |

## Physical workload latency

Use only unprofiled GPU-event measurements from physical hardware. Do not put
FFM or AM time in this table.

| ID | GFX950 latency (µs) | GFX1250 latency (µs) | Warmups / repeats | Status / notes |
|---|---:|---:|---|---|
| MLA-D1 | `<value>` | `<value or unavailable>` | `<N / N>` | `<status>` |
| MLA-P1 | `<value>` | `<value or unavailable>` | `<N / N>` | `<status>` |
| KDA-D1 | `<value>` | `<value or unavailable>` | `<N / N>` | `<status>` |
| KDA-P1 | `<value>` | `<value or unavailable>` | `<N / N>` | `<status>` |
| DSA-D1 | `<value>` | `<value or unavailable>` | `<N / N>` | `<status>` |
| DSA-P1 | `<value>` | `<value or unavailable>` | `<N / N>` | `<status>` |

## Dispatch results

Add one row for every dispatch emitted by every required workload. Repeat a
dispatch row when multiple profiler passes provide different counter groups.
Use `unavailable` rather than leaving required fields blank.

| Case ID | Order | Dispatch | Arch / environment | Time (µs) | AM cycles | HBM read / write bytes | HBM read / write GB/s | L2 hit | Occupancy / waves | MFMA / XDL util. | LDS conflict / stall | VMEM / LDS latency | VGPR / SGPR | LDS / scratch bytes | Status |
|---|---:|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---|
| `<ID>` | `<N>` | `<kernel name>` | `<gfx950 physical>` | `<value>` | `N/A` | `<read / write>` | `<read / write>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<VGPR / SGPR>` | `<LDS / scratch>` | `<pass/unavailable>` |
| `<ID>` | `<N>` | `<kernel name>` | `<gfx1250 AM>` | `N/A` | `<cycles>` | `<read / write>` | `<read / write>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<VGPR / SGPR>` | `<LDS / scratch>` | `<pass/incomplete>` |

## Counter definitions

| Report column | Raw counter(s) | Formula / definition | Unit | GFX950 available | GFX1250 available |
|---|---|---|---|---|---|
| `<column>` | `<raw names>` | `<exact definition>` | `<unit>` | `<yes/no>` | `<yes/no>` |

## Incomplete or failed work

| Case / dispatch | Architecture | Stage | Status | Error or stop reason | Raw artifact |
|---|---|---|---|---|---|
| `<case>` | `<arch>` | `<latency/profile/capture/replay>` | `<failed/incomplete>` | `<reason>` | `<path or URL>` |

## Commands

### GFX950

```bash
<exact latency and profiler commands>
```

### GFX1250 physical, FFM, or AM

```bash
<exact latency, capture, and replay commands>
```

## Raw artifacts

- Runner descriptions: `<path or URL>`
- Runner JSON: `<paths or URLs>`
- rocprofv3 CSV: `<paths or URLs>`
- ROCcap captures and logs: `<paths or URLs>`
- AM output: `<paths or URLs>`

## Conclusions

- `<Only compare physical latency when both values are physical measurements.>`
- `<Summarize dispatch-level bottlenecks without treating AM time as hardware latency.>`
