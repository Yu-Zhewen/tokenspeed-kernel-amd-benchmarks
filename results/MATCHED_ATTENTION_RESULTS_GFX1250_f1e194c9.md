# Matched GFX950 and GFX1250 attention results

## Status

- Report owner: Cursor agent
- Collection date: 2026-08-12 UTC
- Overall status: incomplete
- Missing or invalid cases: all GFX950 results, GFX1250 FFM/AM results, and several GFX1250 profiler metrics are unavailable
- GFX950 status: unavailable; not collected on this host
- GFX1250 status: all six required physical workloads completed; four reporting profiler passes were collected for every case

## Software and hardware setup

| Field | GFX950 | GFX1250 |
|---|---|---|
| Device | unavailable | MI455X-class device (PyTorch reports `AMD Radeon Graphics`) |
| Architecture | `gfx950` unavailable | `gfx1250` |
| Measurement environment | unavailable | physical |
| Host / container | unavailable | `heliosr-1b114-d04-1`, container `tokenspeed-kimi-smoke@sha256:77179418c81a72f003c08656b62b9d7dec3bbdb794a8d5af2f19cf37b2289405` |
| OS | unavailable | Ubuntu 24.04 host and container |
| ROCm / ROCDTIF | unavailable | ROCm 7.15.0; rocprofv3 1.3.5 (`44be71b52284948e58c93f65f46910399773fdcd`) |
| PyTorch | unavailable | 2.11.0+rocm7.15.0a20260728 |
| TokenSpeed commit SHA | unavailable | `f1e194c9c5617b95311d571032dbf94139f0fff9` |
| tokenspeed-kernel revision / tree | unavailable | tree `833f1037c0a1e9b65be292ec15d95e18b18ebbd4` within the TokenSpeed commit |
| tokenspeed-kernel-amd revision / tree | unavailable | tree `ea350b0b776a3a0702ad4d3e0a027e7aed6bd51a` within the TokenSpeed commit |
| Kernel package / build ID | unavailable | source trees mounted read-only through `PYTHONPATH` |
| Triton package / build ID | unavailable | tokenspeed-triton 3.8.10.post20260721 |
| Runner gist revision | unavailable | `29150c094de69af0c13fc39a92237ba47bd59fbc` |
| Warmups / repeats | unavailable | 2 / 5 latency; 0 / 1 profiler |

State any unavoidable setup difference:

`GFX950 was not collected. No cross-architecture comparison is valid yet.`

## Required test suite

| ID | Runner case | Required workload | GFX950 status | GFX1250 status |
|---|---|---|---|---|
| MLA-D1 | `mla-decode` | Kimi-K3 B1 decode, context 4096, FP8 absorbed Q and dense paged KV | unavailable | pass |
| MLA-P1 | `mla-prefill` | Kimi-K3 B1 pure prefill, extend 4096, causal FP8 Q/K/V | unavailable | pass |
| KDA-D1 | `kda-decode` | Kimi-K3 B1 one-token recurrent decode, BF16 inputs and FP32 indexed state | unavailable | pass |
| KDA-P1 | `kda-prefill` | Kimi-K3 B1 KDA pure prefill, extend 4096, BF16 inputs and FP32 state | unavailable | pass |
| DSA-D1 | `dsa-decode-pipeline` | GLM-5.2 B1 decode, context 4096, top-k 2048 and dense FP8 KV | unavailable | pass |
| DSA-P1 | `dsa-prefill-pipeline-4k` | GLM-5.2 B1 pure prefill, extend 4096, causal top-k and dense FP8 KV | unavailable | pass |

The runner completed each GFX1250 invocation and synchronization successfully.
It does not serialize a separate finite-output assertion or selected solution
name, so those two status details remain unavailable in the raw JSON.

## Physical workload latency

Unprofiled GPU-event averages:

| ID | GFX950 latency (µs) | GFX1250 latency (µs) | Warmups / repeats | Status / notes |
|---|---:|---:|---|---|
| MLA-D1 | unavailable | 221.308 | 2 / 5 | GFX1250 pass |
| MLA-P1 | unavailable | 163.998 | 2 / 5 | GFX1250 pass |
| KDA-D1 | unavailable | 54.594 | 2 / 5 | GFX1250 pass |
| KDA-P1 | unavailable | 324.174 | 2 / 5 | GFX1250 pass |
| DSA-D1 | unavailable | 134.361 | 2 / 5 | GFX1250 pass |
| DSA-P1 | unavailable | 40823.276 | 2 / 5 | GFX1250 pass |

## Dispatch results

These rows use the measured invocation, not the runner's preceding untimed setup
invocation. GFX1250 columns merge four compatible reporting passes.
`FetchSize` is converted from reported KiB to bytes using 1024 bytes/KiB.
GFX1250 has no exact HBM write-byte metric; its measured
`GL2C_WRITE_SECTORS` values are intentionally not relabeled as HBM writes.
Bandwidth uses duration from the corresponding counter pass. The HBM-read
column retains the template's naming convention; on GFX1250, `FetchSize`
specifically represents traffic leaving GL2 toward the external-memory
interface and is not proof that every byte reached DRAM. The LDS allocation
stall was zero for every dispatch.

| Case ID | Order | Dispatch | Arch / environment | Time (µs) | AM cycles | HBM read / write bytes | HBM read / write GB/s | L2 hit | Occupancy / waves | MFMA / XDL util. (%) | LDS conflict or allocation stall / memory stall | VMEM / LDS latency (cycles) | VGPR / SGPR | LDS / scratch bytes | Status |
|---|---:|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---|
| MLA-D1 | 1 | MLA decode main | gfx1250 physical | 6.529 | N/A | 765248 / unavailable | 132.648 / unavailable | 99.64% | unavailable / 128 | unavailable | 0.000% / unavailable | unavailable / unavailable | 264 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| MLA-D1 | 2 | MLA decode reduction | gfx1250 physical | 4.086 | N/A | 763648 / unavailable | 156.229 / unavailable | 59.48% | unavailable / 48 | unavailable | 0.000% / unavailable | unavailable / unavailable | 264 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| MLA-D1 | 3 | projected-value GEMM | gfx1250 physical | 6.169 | N/A | 30336 / unavailable | 4.733 / unavailable | 99.62% | unavailable / 96 | unavailable | 0.000% / unavailable | unavailable / unavailable | 128 / 128 | 8192 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| MLA-D1 | 4 | ROCm copyBuffer | gfx1250 physical | 1.202 | N/A | 1408 / unavailable | 1.301 / unavailable | 40.28% | unavailable / 16 | unavailable | 0.000% / unavailable | unavailable / unavailable | 8 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| MLA-D1 | 5 | sigmoid multiply | gfx1250 physical | 2.324 | N/A | 5312 / unavailable | 2.600 / unavailable | 77.53% | unavailable / 8 | unavailable | 0.000% / unavailable | unavailable / unavailable | 56 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| MLA-P1 | 1 | MLA prefill | gfx1250 physical | 152.669 | N/A | 4512 / unavailable | 0.029 / unavailable | 97.93% | unavailable / 1536 | unavailable | 0.000% / unavailable | unavailable / unavailable | 304 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| KDA-D1 | 1 | KDA recurrent decode | gfx1250 physical | 3.566 | N/A | 714880 / unavailable | 205.131 / unavailable | 89.68% | unavailable / 192 | unavailable | 0.000% / unavailable | unavailable / unavailable | 56 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| KDA-P1 | 1 | preprocess intra | gfx1250 physical | 101.112 | N/A | 69783072 / unavailable | 693.179 / unavailable | 82.45% | unavailable / 6144 | unavailable | 0.000% / unavailable | unavailable / unavailable | 168 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| KDA-P1 | 2 | BF16 workspace fill | gfx1250 physical | 3.124 | N/A | 6656 / unavailable | 2.051 / unavailable | 98.14% | unavailable / 12288 | unavailable | 0.000% / unavailable | unavailable / unavailable | 8 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| KDA-P1 | 3 | solve/merge 64 | gfx1250 physical | 11.937 | N/A | 6056224 / unavailable | 482.991 / unavailable | 92.08% | unavailable / 768 | unavailable | 0.000% / unavailable | unavailable / unavailable | 104 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| KDA-P1 | 4 | W/U vector | gfx1250 physical | 24.797 | N/A | 45775296 / unavailable | 1828.306 / unavailable | 91.56% | unavailable / 3072 | unavailable | 0.000% / unavailable | unavailable / unavailable | 232 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| KDA-P1 | 5 | state scan | gfx1250 physical | 167.251 | N/A | 123036256 / unavailable | 769.554 / unavailable | 88.52% | unavailable / 384 | unavailable | 0.000% / unavailable | unavailable / unavailable | 264 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| KDA-P1 | 6 | output | gfx1250 physical | 16.866 | N/A | 24938560 / unavailable | 1371.230 / unavailable | 92.96% | unavailable / 3072 | unavailable | 0.000% / unavailable | unavailable / unavailable | 264 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| DSA-D1 | 1 | decode logits FP8 | gfx1250 physical | 2.965 | N/A | 23008 / unavailable | 7.179 / unavailable | 99.70% | unavailable / 512 | unavailable | 0.000% / unavailable | unavailable / unavailable | 392 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| DSA-D1 | 2 | wave32 radix top-k | gfx1250 physical | 15.423 | N/A | 21376 / unavailable | 1.386 / unavailable | 87.67% | unavailable / 8 | unavailable | 0.000% / unavailable | unavailable / unavailable | 40 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| DSA-D1 | 3 | selected dense WMMA | gfx1250 physical | 81.442 | N/A | 896 / unavailable | 0.011 / unavailable | 99.62% | unavailable / 4 | unavailable | 0.000% / unavailable | unavailable / unavailable | 248 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| DSA-P1 | 1 | prefill logits FP8 | gfx1250 physical | 3790.639 | N/A | 62002560 / unavailable | 16.369 / unavailable | 99.76% | unavailable / 2097152 | unavailable | 0.000% / unavailable | unavailable / unavailable | 392 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| DSA-P1 | 2 | wave32 radix top-k | gfx1250 physical | 92.378 | N/A | 32428928 / unavailable | 321.234 / unavailable | 85.77% | unavailable / 32768 | unavailable | 0.000% / unavailable | unavailable / unavailable | 40 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |
| DSA-P1 | 3 | dense selected attention | gfx1250 physical | 36827.406 | N/A | 35536256 / unavailable | 0.965 / unavailable | 100.00% | unavailable / 1048576 | unavailable | 0.000% / unavailable | unavailable / unavailable | 48 / 128 | 0 / 0 | pass; exact HBM write/occupancy/WMMA/latency unavailable |

## Counter definitions

| Report column | Raw counter(s) | Formula / definition | Unit | GFX950 available | GFX1250 available |
|---|---|---|---|---|---|
| HBM read bytes | `FetchSize` / `FETCH_SIZE` | Reported KiB multiplied by 1024; on GFX1250 this is the weighted `GL2C_EA_RDREQ_*B` sum | bytes | unavailable | yes |
| HBM write bytes | `MemWrites32B` / `WRITE_REQ_32B` | Effective 32-byte memory write transactions multiplied by 32 | bytes | unavailable | no; `GL2C_WRITE_SECTORS` is GL2 traffic, not exact HBM traffic |
| Occupancy / waves | `MeanOccupancyPerActiveCU`; `SQ_WAVES` | Mean active waves where supported; total launched waves only on GFX1250 | waves | unavailable | launch count only |
| MFMA / XDL utilization | `MfmaUtil`; GFX1250 SQ WMMA events | Busy-cycle ratio | percent | unavailable | no; source events returned zero |
| LDS conflict | `LDSBankConflict` | Bank-conflict ratio | percent | unavailable | no; only zero LDS-capacity allocation stalls were observed |
| LDS latency | `LdsLatency`; accumulated `SQ_INST_LEVEL_LDS / SQ_INSTS_LDS` | Average instruction latency | cycles | unavailable | no; source events returned zero |
| Memory stall | `MemUnitStalled`; GFX1250 TCP stall events | Memory-unit stall ratio | percent | unavailable | no |
| L2 hit | `GL2C_HIT`, `GL2C_MISS` on GFX1250 | `hit / (hit + miss)` | percent | unavailable | yes |
| VMEM dependency latency | GFX1250 `TX_VMW_VMW_LATENCY / TX_VMW_VCA_REQ_STATE_READ` | Average VMEM/TCP latency | cycles | unavailable | no; source events returned zero |

## Incomplete or failed work

| Case / dispatch | Architecture | Stage | Status | Error or stop reason | Raw artifact |
|---|---|---|---|---|---|
| All six cases | gfx950 | latency / profile / capture / replay | not collected | No GFX950 device was available on this host | unavailable |
| All six cases | gfx1250 | FFM capture / AM replay | not collected | Physical GFX1250 run only | unavailable |
| All GFX1250 dispatches | gfx1250 | profile | incomplete metrics | Occupancy, WMMA, LDS-latency, and TCP-latency source events returned zero in isolated passes | `matched-attention-results/gfx1250/profiles-v2/` |

### Post-July 28 profiler validation

A complete, ABI-matched TheRock nightly stack was tested after the primary
collection: ROCm `10.1.0a20260812`, HIP `7.16.26315`, PyTorch
`2.13.0+rocm10.1.0a20260812`, and rocprofv3 1.3.5
(`5bc651a82683b2ae21acf14ffc9af35f5c2722ae`). The DSA decode pipeline ran
successfully under this stack. Its selected dense WMMA dispatch reported
`SQ_WAVES=4` and nonzero `SQ_BUSY_CYCLES`, confirming that counter collection
was active, but `SQ_INSTS_ALL`, `SQ_INSTS_VALU`, `SQ_WAVE_CYCLES`,
`SQ_INSTS_VEC32_VALU_WMMA`, `SQ_INST_CYCLES_VALU_WMMA`, and
`SQ_VALU_WMMA_FLOP_BF16` still returned zero. Cached compiler assembly confirms
that the kernel contains many `v_wmma_f32_16x16x32_bf16` instructions. The
previously tested `SQ_VALU_WMMA_FLOP_FP8` counter is expected to be zero for
this BF16 WMMA kernel and is not evidence of a profiler failure. The missing instruction,
occupancy, and WMMA metrics therefore persist with a complete post-July 28
runtime and are not caused solely by mixing profiler and ROCm versions.

An additional matched ROCm `10.1.0a20260805` stack with rocprofv3 commit
`3e226263752e1f07cfbcc04884ec6a2d789a7aab` reproduced zero for all three
BF16 WMMA counters. A matched July 29 stack with rocprofv3 commit
`920e5a8a70669d28b930c33fcef2b18c4b47d152` could not profile the TokenSpeed
workload because its available GFX1250 PyTorch builds abort during profiler
initialization with duplicate LLVM option registration. August 5 raw output is
under `profiler-version-validation/20260805/`.

A much older matched ROCm `7.13.0` stack from May 5 with rocprofv3 `1.2.2`
commit `c0b5138a275b3989251d76e86bcae90499c1e45e` also reproduced the failure.
For two `_dsa_selected_dense_wmma_kernel` dispatches, `SQ_WAVES` returned `4`
and `SQ_BUSY_CYCLES` returned approximately 54--55 million, while
`SQ_INSTS_VEC32_VALU_WMMA`, `SQ_INST_CYCLES_VALU_WMMA`, and
`SQ_VALU_WMMA_FLOP_BF16` all returned zero. The image's rocprofv3 launcher
contained a duplicated `--selected-regions-ref-count` argument; removing the
duplicate allowed the otherwise unmodified rocprofiler SDK to run. This shows
that the WMMA counter behavior predates the July and August profiler builds and
is not a recent rocprofv3 regression. Raw output is under
`profiler-version-validation/20260505/`.
This May counter table does not yet expose `GL2C_EA_WR_UNCACHED_32B` or
`GL2C_EA_WRREQ_DRAM`; its only write-related event is the
`GL1C_GL2_REQ_WRITE_LEVEL` in-flight-cycle counter, so the current write-count
fallbacks cannot be compared on that stack.

ROCm documents a fixed GPU performance level as a workaround for perfmon clock
gating on some GFX11/GFX12 devices. On this MI455X host, PMFW rejected
`STABLE_STD`, `STABLE_PEAK`, `HIGH`, `LOW`, `STABLE_MIN_MCLK`, and
`STABLE_MIN_SCLK` with `AMDSMI_STATUS_NOT_SUPPORTED`. `MANUAL` was accepted,
but an isolated rerun still produced zero for all six instruction, wave-cycle,
and WMMA events above while `SQ_WAVES` and `SQ_BUSY_CYCLES` remained nonzero.
The GPU was restored to `AUTO` afterward. Raw output is under
`profiler-complete-nightly-validation/profile-manual/`.

Public ROCm history points to event-specific GFX1250 profiler bring-up rather
than an application or permission failure. The working SQ counters use low
selectors (`SQ_BUSY_CYCLES=3`, `SQ_WAVES=4`), while the zero counters use
selectors 32, 64, 128, 237, 292, and 359. GFX1250's 10-bit selector field can
represent those values, but the original counter-table addition
([ROCm/rocm-systems#7767](https://github.com/ROCm/rocm-systems/pull/7767))
published no hardware validation results. A separate July fix
([#9252](https://github.com/ROCm/rocm-systems/pull/9252)) corrected other
GFX1250 counters that always returned zero, and an August fix
([#9522](https://github.com/ROCm/rocm-systems/pull/9522)) addresses another
event class whose request counters remain zero while cycle counters work.
The remaining SQ/WMMA failures are therefore most consistent with missing
event-class enablement or incorrect selector mappings, potentially specific
to the hardware stepping.

## Commands

### GFX950

```text
unavailable; not run on this host
```

### GFX1250 physical

The checkout was a clean clone at the requested TokenSpeed SHA. The source
trees were mounted read-only into the pinned container:

```bash
export PYTHONPATH=/workspace/tokenspeed/python:/workspace/tokenspeed/tokenspeed-kernel/python:/workspace/tokenspeed/tokenspeed-kernel-amd/python
python /workspace/gist/profile_matched_attention.py \
  --case all --expected-arch gfx1250 --environment physical \
  --warmup 2 --repeats 5 \
  --output /workspace/results/gfx1250/latency.json

APP="python /workspace/gist/profile_matched_attention.py \
  --case $CASE --expected-arch gfx1250 --environment physical \
  --warmup 0 --repeats 1"

rocprofv3 --kernel-trace --pmc FetchSize \
  --output-format csv --output-file fetch-size \
  --output-directory "/workspace/results/gfx1250/profiles-v2/$CASE/fetch-size" \
  -- $APP
```

The other reporting passes collected `GL2C_WRITE_SECTORS`, `GL2C_HIT`,
`GL2C_MISS`, `GL2C_EA_WRREQ_DRAM`, SQ events, and TCP/SPI diagnostics in
compatible groups. All GPU commands ran under `gpu-lock`.

## Raw artifacts

- GFX950 artifacts: unavailable
- GFX1250 runner JSON: `matched-attention-results/gfx1250/latency.json`
- GFX1250 rocprofv3 CSV: `matched-attention-results/gfx1250/profiles-v2/<case>/<pass>/`
- Post-July 28 validation CSV: `matched-attention-results/gfx1250/profiler-complete-nightly-validation/`
- May 5 rocprofv3 1.2.2 validation CSV: `matched-attention-results/gfx1250/profiler-version-validation/20260505/`
- GFX1250 counter definitions and validation: `matched-attention-results/gfx1250-extra-counters.yaml`
- ROCcap captures, FFM logs, and AM output: unavailable

## Conclusions

- The GFX1250 six-case physical suite completed at the requested TokenSpeed commit.
- GFX1250 `FetchSize * 1024`, L2 hit rate, and wave-launch counts are available; exact HBM writes, occupancy, WMMA utilization, and latency metrics remain unavailable.
- A complete 2026-08-12 ROCm/PyTorch/rocprofv3 stack reproduced the zero SQ instruction and WMMA counters, while `SQ_WAVES` and `SQ_BUSY_CYCLES` remained nonzero.
- No architecture speedup claim can be made until the matching GFX950 suite is collected.
