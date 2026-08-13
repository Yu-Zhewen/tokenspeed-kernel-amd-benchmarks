# Matched GFX950 and GFX1250 attention results

## Status

- Report owner: Cursor agent
- Collection date: 2026-08-11 UTC
- Overall status: incomplete
- Missing or invalid cases: all GFX1250 physical/FFM/AM results are unavailable
- GFX950 status: all six required workloads completed; seven single-counter profiler passes were collected for every case

## Software and hardware setup

| Field | GFX950 | GFX1250 |
|---|---|---|
| Device | AMD Instinct MI355X (PyTorch reports `AMD Radeon Graphics`) | unavailable |
| Architecture | `gfx950:sramecc+:xnack-` | `gfx1250` unavailable |
| Measurement environment | physical | unavailable |
| Host / container | host GPU, container `lightseekorg/tokenspeed-amd:tml` | unavailable |
| OS | Ubuntu 24.04.4 LTS (container) | unavailable |
| ROCm / ROCDTIF | ROCm 7.2.26015 | unavailable |
| PyTorch | 2.11.0+rocm7.2 | unavailable |
| TokenSpeed commit SHA | `f1e194c9c5617b95311d571032dbf94139f0fff9` | unavailable |
| tokenspeed-kernel commit SHA | `f1e194c9c5617b95311d571032dbf94139f0fff9` | unavailable |
| tokenspeed-kernel-amd commit SHA | `f1e194c9c5617b95311d571032dbf94139f0fff9` | unavailable |
| Kernel package / build ID | source checkout; installed metadata `0.1.0.dev20260715+git00000000` | unavailable |
| Triton package / build ID | tokenspeed-triton 3.8.10.post20260709; Triton 3.6.0+rocm7.2.4.git4ed88892 | unavailable |
| Runner gist revision | `71c78f192f9e5f0f5c099b7ad94fbff488d6fb29` | use same revision |
| Warmups / repeats | 2 / 5 latency; 0 / 1 profiler | unavailable |

State any unavoidable setup difference:

`GFX1250 has not been collected. No cross-architecture comparison is valid yet.`

## Required test suite

| ID | Runner case | Required workload | GFX950 status | GFX1250 status |
|---|---|---|---|---|
| MLA-D1 | `mla-decode` | Kimi-K3 B1 decode, context 4096, FP8 absorbed Q and dense paged KV | pass | unavailable |
| MLA-P1 | `mla-prefill` | Kimi-K3 B1 pure prefill, extend 4096, causal FP8 Q/K/V | pass | unavailable |
| KDA-D1 | `kda-decode` | Kimi-K3 B1 one-token recurrent decode, BF16 inputs and FP32 indexed state | pass | unavailable |
| KDA-P1 | `kda-prefill` | Kimi-K3 B1 KDA pure prefill, extend 4096, BF16 inputs and FP32 state | pass | unavailable |
| DSA-D1 | `dsa-decode-pipeline` | GLM-5.2 B1 decode, context 4096, top-k 2048 and dense FP8 KV | pass | unavailable |
| DSA-P1 | `dsa-prefill-pipeline-4k` | GLM-5.2 B1 pure prefill, extend 4096, causal top-k and dense FP8 KV | pass | unavailable |

The runner completed each GFX950 invocation and synchronization successfully.
It does not serialize a separate finite-output assertion or selected solution
name, so those two status details remain unavailable in the raw JSON.

## Physical workload latency

Unprofiled GPU-event averages:

| ID | GFX950 latency (µs) | GFX1250 latency (µs) | Warmups / repeats | Status / notes |
|---|---:|---:|---|---|
| MLA-D1 | 81.857 | unavailable | 2 / 5 | pass |
| MLA-P1 | 143.457 | unavailable | 2 / 5 | pass |
| KDA-D1 | 45.464 | unavailable | 2 / 5 | pass |
| KDA-P1 | 295.915 | unavailable | 2 / 5 | pass |
| DSA-D1 | 274164.282 | unavailable | 2 / 5 | pass |
| DSA-P1 | 105108.887 | unavailable | 2 / 5 | pass |

## Dispatch results

These rows use the measured invocation, not the runner's preceding untimed setup
invocation. Time and static resources come from the trace-only pass. Counter
columns merge seven separate identical single-PMC passes. `FetchSize` was
converted from reported KiB to bytes using 1024 bytes/KiB; `MemWrites32B` was
multiplied by 32. Bandwidth is derived from those bytes and trace duration.

| Case ID | Order | Dispatch | Arch / environment | Time (µs) | AM cycles | HBM read / write bytes | HBM read / write GB/s | L2 hit | Occupancy / waves | MFMA / XDL util. (%) | LDS conflict / memory stall | VMEM / LDS latency (cycles) | VGPR / SGPR | LDS / scratch bytes | Status |
|---|---:|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---|
| MLA-D1 | 1 | `_mla_decode_gluon` | gfx950 physical | 10.600 | N/A | 1298432 / 202752 | 122.494 / 19.128 | unavailable | 1.001 | 0.349 | 0.086% / 0.118% | unavailable / 58.930 | 104 / 80 | 0 / 0 | pass; L2/VMEM unavailable |
| MLA-D1 | 2 | `_mla_reduce_project_value_kernel` | gfx950 physical | 11.360 | N/A | 7113216 / 6144 | 626.163 / 0.541 | unavailable | 1.979 | 0.000 | 0.000% / 0.011% | unavailable / 0.000 | 16 / 32 | 0 / 0 | pass; L2/VMEM unavailable |
| MLA-P1 | 1 | `_mla_prefill_kernel` | gfx950 physical | 136.081 | N/A | 17086016 / 12582912 | 125.558 / 92.466 | unavailable | 1.000 | 19.499 | 8.713% / 0.002% | unavailable / 75.777 | 140 / 112 | 0 / 0 | pass; L2/VMEM unavailable |
| KDA-D1 | 1 | `_kda_recurrent_decode_kernel` | gfx950 physical | 6.280 | N/A | 499712 / 789504 | 79.572 / 125.717 | unavailable | 1.002 | 0.000 | 0.000% / 0.013% | unavailable / 0.000 | 92 / 48 | 0 / 0 | pass; L2/VMEM unavailable |
| KDA-P1 | 1 | `at::native::BF16 fill (internal)` | gfx950 physical | 4.760 | N/A | 8768 / 6291456 | 1.842 / 1321.734 | unavailable | 4.570 | 0.000 | 0.000% / 0.035% | unavailable / 0.000 | 4 / 16 | 0 / 0 | pass; L2/VMEM unavailable |
| KDA-P1 | 2 | `_preprocess_intra_fwd_kernel` | gfx950 physical | 68.281 | N/A | 19303808 / 62128128 | 282.711 / 909.889 | unavailable | 1.640 | 2.222 | 5.171% / 0.154% | unavailable / 64.339 | 108 / 48 | 0 / 0 | pass; L2/VMEM unavailable |
| KDA-P1 | 3 | `at::native::BF16 fill (internal)` | gfx950 physical | 4.200 | N/A | 8704 / 6291456 | 2.072 / 1497.966 | unavailable | 4.573 | 0.000 | 0.000% / 0.038% | unavailable / 0.000 | 4 / 16 | 0 / 0 | pass; L2/VMEM unavailable |
| KDA-P1 | 4 | `_solve_merge_64_fwd_kernel` | gfx950 physical | 24.720 | N/A | 4787712 / 3932160 | 193.678 / 159.068 | unavailable | 1.000 | 0.246 | 0.130% / 0.063% | unavailable / 44.990 | 36 / 32 | 0 / 0 | pass; L2/VMEM unavailable |
| KDA-P1 | 5 | `_wu_vector_fwd_kernel` | gfx950 physical | 25.680 | N/A | 31080256 / 37748736 | 1210.290 / 1469.966 | unavailable | 1.656 | 1.849 | 5.655% / 0.444% | unavailable / 152.429 | 96 / 48 | 0 / 0 | pass; L2/VMEM unavailable |
| KDA-P1 | 6 | `_state_scan_fwd_kernel` | gfx950 physical | 146.642 | N/A | 204512576 / 51118080 | 1394.638 / 348.591 | unavailable | 1.000 | 2.602 | 1.293% / 0.289% | unavailable / 43.825 | 60 / 112 | 0 / 0 | pass; L2/VMEM unavailable |
| KDA-P1 | 7 | `_output_fwd_kernel` | gfx950 physical | 12.600 | N/A | 15763456 / 12595840 | 1251.068 / 999.670 | unavailable | 2.552 | 1.440 | 3.035% / 0.585% | unavailable / 85.474 | 52 / 112 | 0 / 0 | pass; L2/VMEM unavailable |
| DSA-D1 | 1 | `_dsa_decode_logits_fp8_kernel` | gfx950 physical | 15.961 | N/A | 407040 / 16384 | 25.502 / 1.027 | unavailable | 1.001 | 0.000 | 0.000% / 0.007% | unavailable / 0.000 | 168 / 64 | 0 / 0 | pass; L2/VMEM unavailable |
| DSA-D1 | 2 | `_dsa_oneblock_manual_radix_topk_kernel` | gfx950 physical | 10.720 | N/A | 26368 / 8224 | 2.460 / 0.767 | unavailable | 3.969 | 0.000 | 0.018% / 0.026% | unavailable / 116.712 | 28 / 112 | 0 / 0 | pass; L2/VMEM unavailable |
| DSA-D1 | 3 | `_dsa_dense_kv_kernel` | gfx950 physical | 273410.163 | N/A | 21717047360 / 39225991168 | 79.430 / 143.469 | unavailable | 1.000 | 0.000 | 0.000% / 0.098% | unavailable / 42.633 | 256 / 80 | 0 / 18744 | pass; L2/VMEM unavailable |
| DSA-P1 | 1 | `_dsa_prefill_logits_fp8_kernel` | gfx950 physical | 15570.324 | N/A | 815667392 / 67108864 | 52.386 / 4.310 | unavailable | 1.001 | 0.000 | 0.000% / 0.003% | unavailable / 0.000 | 164 / 48 | 0 / 0 | pass; L2/VMEM unavailable |
| DSA-P1 | 2 | `_dsa_oneblock_manual_radix_topk_kernel` | gfx950 physical | 56.560 | N/A | 13006784 / 33685504 | 229.964 / 595.571 | unavailable | 7.043 | 0.000 | 11.677% / 0.056% | unavailable / 134.322 | 24 / 80 | 0 / 0 | pass; L2/VMEM unavailable |
| DSA-P1 | 3 | `_dsa_dense_kv_kernel` | gfx950 physical | 89135.898 | N/A | 1189723392 / 33554432 | 13.347 / 0.376 | unavailable | 6.963 | 0.000 | 2.957% / 0.000% | unavailable / 54.431 | 36 / 48 | 0 / 0 | pass; L2/VMEM unavailable |

## Counter definitions

| Report column | Raw counter(s) | Formula / definition | Unit | GFX950 available | GFX1250 available |
|---|---|---|---|---|---|
| HBM read bytes | `FetchSize` / `FETCH_SIZE` | Total video-memory fetches including cache and memory effects; reported KiB multiplied by 1024 | bytes | yes | unknown |
| HBM write bytes | `MemWrites32B` / `WRITE_REQ_32B` | Effective 32-byte memory write transactions multiplied by 32 | bytes | yes | unknown |
| Occupancy / waves | `MeanOccupancyPerActiveCU` | `reduce(accumulate(SQ_LEVEL_WAVES, LOW_RES),sum) / reduce(SQ_BUSY_CU_CYCLES,sum)` | waves | yes | unknown |
| MFMA utilization | `MfmaUtil` | `reduce(SQ_VALU_MFMA_BUSY_CYCLES,sum) / (reduce(GRBM_GUI_ACTIVE,max) * SIMD_NUM) * 100` | percent | yes | unknown |
| LDS conflict | `LDSBankConflict` | `100 * reduce(SQ_LDS_BANK_CONFLICT,sum) / reduce(GRBM_GUI_ACTIVE,max) / CU_NUM` | percent | yes | unknown |
| LDS latency | `LdsLatency` | `reduce(accumulate(SQ_INST_LEVEL_LDS, HIGH_RES),sum) / reduce(SQ_INSTS_LDS,sum)` | cycles | yes | unknown |
| Memory stall | `MemUnitStalled` | `100 * TCP_TCP_TA_DATA_STALL_CYCLES_max / reduce(GRBM_GUI_ACTIVE,max) / SE_NUM` | percent | yes | unknown |
| L2 hit | unavailable | No stable L2 hit counter was collected | unavailable | no | unknown |
| VMEM dependency latency | unavailable | No VMEM dependency-latency counter was collected | unavailable | no | unknown |

## Incomplete or failed work

| Case / dispatch | Architecture | Stage | Status | Error or stop reason | Raw artifact |
|---|---|---|---|---|---|
| All six cases | gfx1250 | latency/profile/capture/replay | incomplete | Not collected on this physical GFX950 host | unavailable |
| All GFX950 dispatches | gfx950 | profile | incomplete metrics | Stable L2-hit and VMEM dependency-latency counters were not collected | `gfx950/profiles-rev-71c78f1/` |

## Commands

### GFX950

The checkout was a clean detached worktree at the TokenSpeed SHA above. The
container used GPU 0, `/dev/kfd`, `/dev/dri`, host IPC/networking, and:

```bash
export PYTHONPATH=/workspace/python:/workspace/tokenspeed-kernel/python:/workspace/tokenspeed-kernel-amd/python
python /benchmark/gist/profile_matched_attention.py --describe \
  > /benchmark/results/matched-attention-cases.json
python /benchmark/gist/profile_matched_attention.py \
  --case all --expected-arch gfx950 --environment physical \
  --warmup 2 --repeats 5 \
  --output /benchmark/results/gfx950/latency-six.json
```

Each named case was traced once, followed by one identical pass for each of
`FetchSize`, `MemWrites32B`, `MeanOccupancyPerActiveCU`, `MfmaUtil`,
`LDSBankConflict`, `LdsLatency`, and `MemUnitStalled`:

```bash
rocprofv3 --kernel-trace --output-format csv \
  --output-directory "$CASE_DIR/trace" -- \
  python "$RUNNER" --case "$CASE" --expected-arch gfx950 \
    --environment physical --warmup 0 --repeats 1

rocprofv3 --kernel-trace --pmc "$COUNTER" --output-format csv \
  --output-directory "$CASE_DIR/counter-$COUNTER" -- \
  python "$RUNNER" --case "$CASE" --expected-arch gfx950 \
    --environment physical --warmup 0 --repeats 1
```

### GFX1250 physical, FFM, or AM

```text
unavailable; not run on this host
```

## Raw artifacts

- Runner descriptions: `/home/zhewenyu/matched-attention-main-20260811/results/matched-attention-cases.json`
- Runner JSON: `/home/zhewenyu/matched-attention-main-20260811/results/gfx950/latency-six.json`
- rocprofv3 CSV: `/home/zhewenyu/matched-attention-main-20260811/results/gfx950/profiles-rev-71c78f1/<case>/`
- Raw rocprofv3 data: `/home/zhewenyu/matched-attention-main-20260811/results/gfx950/rocprofv3-raw-rev-71c78f1`
- ROCcap captures and logs: unavailable
- AM output: unavailable

## Conclusions

- The GFX950 six-case suite completed at the latest runner revision.
- KDA-D1 physical latency was 45.464 µs.
- DSA decode and prefill dominate complete-workload latency on this GFX950 run.
- No physical speedup or architecture comparison can be claimed until matched GFX1250 data is collected.
