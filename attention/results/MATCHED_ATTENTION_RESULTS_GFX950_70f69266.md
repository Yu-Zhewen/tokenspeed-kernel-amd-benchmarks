# Latest-main GFX950 matched-attention results

## Status

- Collection date: 2026-08-20 UTC
- GFX950 physical collection: complete
- Overall matched-comparison status: incomplete; GFX1250 has not been collected at this revision
- Required workloads: 6/6 passed
- Profiled measured-scope dispatches: 19

## Software and hardware setup

| Field | GFX950 | GFX1250 |
|---|---|---|
| Device | AMD Instinct MI355X (PyTorch: `AMD Radeon Graphics`) | unavailable |
| Architecture | `gfx950:sramecc+:xnack-` | `gfx1250` |
| Measurement environment | physical, GPU 0 | unavailable |
| Host / container | `smci355-ccs-aus-n15-05` / `lightseekorg/tokenspeed-amd:tml` | unavailable |
| OS | Ubuntu 24.04 container; Linux 6.8.0-84 host | unavailable |
| ROCm / ROCDTIF | `7.2.26015` | unavailable |
| PyTorch | `2.11.0+rocm7.2` | unavailable |
| TokenSpeed commit SHA | `70f692669a76b6c317cec31b679d7f4fac5da9fa` | unavailable |
| `tokenspeed-kernel` source | same monorepo commit; tree `8c4c7564d4f1727e67e01855c801e5d44fc5baad` | unavailable |
| `tokenspeed-kernel-amd` source | same monorepo commit; tree `c139bcd7853eae1d0c843cc544ab11d144326cfc` | unavailable |
| Kernel package / build ID | source-mounted from the TokenSpeed checkout | unavailable |
| Triton package / build ID | `tokenspeed-triton 3.8.10.post20260721` (`triton.__version__` reports `3.6.0`) | unavailable |
| Runner repository revision | `0135b4c4ae7262d992ad5ce3d19a508e207bdd98` | same revision required |
| Runner SHA-256 | `f8d1fad666ec89eec46a05acf60a913b3c9e1e8dd7edb334efd4f525a59217dc` | same runner required |
| Latency warmups / repeats | 2 / 5 | unavailable |
| Profiler warmups / repeats | 0 / 1 | unavailable |

Container image: `lightseekorg/tokenspeed-amd@sha256:a6630bce4f1d9e0dd236aa90dcf56f342526ebac30ba3aaf04d17866c4b71fa1`
(image ID `sha256:274fb85449cb34e0ec2b1b728899dabf944ad24d61a269cff73f14490fff1fe4`).
The tracked TokenSpeed checkout was clean; unrelated untracked files were present and were not used or modified.

## Required test suite

| ID | Runner case | GFX950 status | GFX1250 status |
|---|---|---|---|
| MLA-D1 | `mla-decode` | pass | unavailable |
| MLA-P1 | `mla-prefill` | pass | unavailable |
| KDA-D1 | `kda-decode` | pass | unavailable |
| KDA-P1 | `kda-prefill` | pass | unavailable |
| DSA-D1 | `dsa-decode-pipeline` | pass | unavailable |
| DSA-P1 | `dsa-prefill-pipeline-4k` | pass | unavailable |

All cases completed without an exception. The runner reports successful API completion and selected
paths but has no separate finite-output boolean in its JSON schema.

## Physical workload latency

| ID | Runner case | GFX950 latency (µs) | GFX1250 latency (µs) | Warmups / repeats | Status |
|---|---|---:|---:|---|---|
| MLA-D1 | `mla-decode` | 83.937 | unavailable | 2 / 5 | pass |
| MLA-P1 | `mla-prefill` | 142.257 | unavailable | 2 / 5 | pass |
| KDA-D1 | `kda-decode` | 51.952 | unavailable | 2 / 5 | pass |
| KDA-P1 | `kda-prefill` | 293.306 | unavailable | 2 / 5 | pass |
| DSA-D1 | `dsa-decode-pipeline` | 124.953 | unavailable | 2 / 5 | pass |
| DSA-P1 | `dsa-prefill-pipeline-4k` | 2,311.904 | unavailable | 2 / 5 | pass |

Latency is the complete unprofiled API or pipeline time measured with GPU events. It is not a sum
of profiler dispatch durations.

## Per-dispatch physical results

The table includes every dispatch in the measured invocation suffix. The runner makes one untimed
setup invocation before the measured invocation; setup-only dispatches are retained in raw traces
but excluded here. Every counter was collected in a separate pass.

| Case | # | Dispatch | Duration µs | Grid | Workgroup | VGPR | SGPR | LDS B | Scratch B | Read MB | Write MB | Read GB/s | Write GB/s | Occupancy | MFMA % | LDS conflict % | LDS latency cyc | Mem stalled % |
|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLA-D1 | 1 | `_mla_decode_gluon` | 10.200 | 256×16×1 | 256×1×1 | 104 | 80 | 0 | 0 | 1.298 | 0.203 | 151.7 | 23.1 | 1.001 | 0.354 | 0.086 | 58.836 | 0.124 |
| MLA-D1 | 2 | `_mla_reduce_project_value_kernel` | 11.360 | 98304×1×1 | 512×1×1 | 16 | 32 | 0 | 0 | 7.113 | 0.006 | 583.1 | 0.5 | 1.979 | 0.000 | 0.000 | 0.000 | 0.026 |
| MLA-P1 | 1 | `_mla_prefill_kernel` | 134.681 | 131072×1×1 | 256×1×1 | 140 | 112 | 0 | 0 | 17.086 | 12.583 | 127.3 | 94.3 | 1.000 | 19.521 | 8.661 | 75.875 | 0.001 |
| KDA-D1 | 1 | `_kda_recurrent_decode_kernel` | 6.200 | 256×12×1 | 64×1×1 | 92 | 48 | 0 | 0 | 0.500 | 0.790 | 100.7 | 153.0 | 1.002 | 0.000 | 0.000 | 0.000 | 0.015 |
| KDA-P1 | 1 | `at::native::BF16 fill` | 4.080 | 393216×1×1 | 256×1×1 | 4 | 16 | 0 | 0 | 0.009 | 6.291 | 2.8 | 2097.2 | 4.663 | 0.000 | 0.000 | 0.000 | 0.035 |
| KDA-P1 | 2 | `_preprocess_intra_fwd_kernel` | 69.121 | 16384×12×1 | 256×1×1 | 108 | 48 | 0 | 0 | 19.309 | 62.128 | 290.6 | 928.4 | 1.641 | 2.211 | 5.244 | 64.294 | 0.144 |
| KDA-P1 | 3 | `at::native::BF16 fill` | 4.160 | 393216×1×1 | 256×1×1 | 4 | 16 | 0 | 0 | 0.009 | 6.291 | 2.7 | 1966.1 | 4.661 | 0.000 | 0.000 | 0.000 | 0.018 |
| KDA-P1 | 4 | `_solve_merge_64_fwd_kernel` | 25.200 | 4096×12×1 | 64×1×1 | 36 | 32 | 0 | 0 | 4.788 | 3.932 | 199.8 | 164.4 | 1.000 | 0.245 | 0.131 | 45.032 | 0.058 |
| KDA-P1 | 5 | `_wu_vector_fwd_kernel` | 23.960 | 16384×12×1 | 256×1×1 | 96 | 48 | 0 | 0 | 30.913 | 37.749 | 1163.9 | 1483.8 | 1.648 | 1.950 | 5.574 | 152.706 | 0.478 |
| KDA-P1 | 6 | `_state_scan_fwd_kernel` | 147.441 | 4096×12×1 | 256×1×1 | 60 | 112 | 0 | 0 | 204.513 | 51.118 | 1410.0 | 351.5 | 1.000 | 2.559 | 1.273 | 43.815 | 0.274 |
| KDA-P1 | 7 | `_output_fwd_kernel` | 12.880 | 16384×12×1 | 256×1×1 | 52 | 112 | 0 | 0 | 15.763 | 12.592 | 1187.0 | 959.7 | 2.545 | 1.479 | 2.602 | 84.729 | 0.493 |
| DSA-D1 | 1 | `_dsa_decode_logits_fp8_kernel` | 7.880 | 256×128×1 | 256×1×1 | 128 | 64 | 0 | 0 | 0.386 | 0.016 | 64.7 | 2.7 | 1.002 | 0.000 | 0.000 | 0.000 | 0.003 |
| DSA-D1 | 2 | `_dsa_oneblock_manual_radix_topk_kernel` | 10.520 | 1024×1×1 | 1024×1×1 | 28 | 112 | 0 | 0 | 0.025 | 0.008 | 2.4 | 0.8 | 3.983 | 0.000 | 0.018 | 118.303 | 0.021 |
| DSA-D1 | 3 | `_dsa_dense_mfma_kv_kernel` | 17.680 | 256×1×32 | 256×1×1 | 164 | 112 | 0 | 0 | 0.846 | 0.270 | 73.9 | 23.3 | 1.001 | 0.568 | 0.416 | 85.391 | 0.003 |
| DSA-D1 | 4 | `_dsa_dense_mfma_reduce_kernel` | 8.080 | 256×8×1 | 256×1×1 | 8 | 64 | 0 | 0 | 0.148 | 0.008 | 20.6 | 1.1 | 1.002 | 0.000 | 0.000 | 0.000 | 0.002 |
| DSA-P1 | 1 | `_combine_scoring_query_heads_kernel` | 25.001 | 1048576×1×1 | 256×1×1 | 72 | 80 | 0 | 0 | 17.378 | 2.097 | 758.2 | 93.8 | 1.820 | 0.000 | 0.000 | 0.000 | 0.006 |
| DSA-P1 | 2 | `_dsa_prefill_logits_fp8_kernel` | 479.043 | 1048576×128×1 | 256×1×1 | 20 | 48 | 0 | 0 | 3.623 | 67.109 | 7.5 | 139.9 | 6.678 | 0.000 | 0.000 | 0.000 | 0.173 |
| DSA-P1 | 3 | `_dsa_oneblock_manual_radix_topk_kernel` | 54.360 | 4194304×1×1 | 1024×1×1 | 24 | 80 | 0 | 0 | 13.003 | 33.686 | 240.4 | 590.1 | 7.043 | 0.000 | 11.794 | 134.335 | 0.061 |
| DSA-P1 | 4 | `_dsa_dense_mfma_kv_kernel` | 1836.652 | 1048576×1×1 | 256×1×1 | 164 | 112 | 0 | 0 | 35.861 | 33.554 | 19.6 | 18.4 | 1.000 | 26.476 | 12.732 | 92.052 | 0.000 |

No stable L2/cache-hit or VMEM dependency-latency counter was collected; those required fields are
`unavailable`. AM cycles are not applicable to this physical GFX950 collection.

## Counter definitions

| Report column | Raw counter | Definition |
|---|---|---|
| Read bytes / bandwidth | `FetchSize` | `FETCH_SIZE` in KiB; bytes = value × 1024; bandwidth uses the same pass duration. |
| Write bytes / bandwidth | `MemWrites32B` | `WRITE_REQ_32B`; bytes = value × 32; bandwidth uses the same pass duration. |
| Occupancy | `MeanOccupancyPerActiveCU` | Mean wave occupancy per active CU. |
| MFMA % | `MfmaUtil` | MFMA busy cycles / (`GRBM_GUI_ACTIVE` × SIMD count) × 100. |
| LDS conflict % | `LDSBankConflict` | GPU-time percentage stalled by LDS bank conflicts. |
| LDS latency | `LdsLatency` | Average LDS instruction dependency latency in cycles. |
| Mem stalled % | `MemUnitStalled` | GPU-time percentage for which the memory unit is stalled. |
| L2 hit | unavailable | No stable common counter collected. |
| VMEM dependency latency | unavailable | No stable counter collected. |

## Incomplete or failed work

| Case / dispatch | Architecture | Stage | Status | Reason | Raw artifact |
|---|---|---|---|---|---|
| All cases | GFX1250 | latency/profile/capture/replay | incomplete | No same-revision GFX1250 run was requested or collected | unavailable |

## Exact commands

### GFX950 latency

```bash
ROCR_VISIBLE_DEVICES=0 python profile_matched_attention.py \
  --case all \
  --expected-arch gfx950 \
  --environment physical \
  --warmup 2 \
  --repeats 5 \
  --output matched-attention-gfx950-70f69266.json
```

### GFX950 trace

```bash
ROCR_VISIBLE_DEVICES=0 rocprofv3 \
  --kernel-trace \
  --output-format csv \
  --output-directory "profiles/$CASE/trace" \
  -- \
  python profile_matched_attention.py \
    --case "$CASE" \
    --expected-arch gfx950 \
    --environment physical \
    --warmup 0 \
    --repeats 1
```

### GFX950 single-counter pass

```bash
ROCR_VISIBLE_DEVICES=0 rocprofv3 \
  --kernel-trace \
  --pmc "$COUNTER" \
  --output-format csv \
  --output-directory "profiles/$CASE/counter-$COUNTER" \
  -- \
  python profile_matched_attention.py \
    --case "$CASE" \
    --expected-arch gfx950 \
    --environment physical \
    --warmup 0 \
    --repeats 1
```

The trace and each of `FetchSize`, `MemWrites32B`, `MeanOccupancyPerActiveCU`, `MfmaUtil`,
`LDSBankConflict`, `LdsLatency`, and `MemUnitStalled` were collected for every case. Separate passes
avoid unsupported counter-group scheduling and preserve one unambiguous value per dispatch.

## Raw artifacts

- Runner description: `gfx950_70f69266/matched-attention-cases.json`
- Unprofiled runner JSON: `gfx950_70f69266/latency.json`
- Dispatch traces and seven counter passes per case: `gfx950_70f69266/profiles/`
- Available-counter listing: `gfx950_70f69266/rocprofv3-pmcs.txt`
- Exact counter definitions: `gfx950_70f69266/rocprofv3-counter-info.txt`

## Conclusions

- All six required latest-main workloads completed on physical MI355X.
- DSA-P1 was the slowest complete workload at 2,311.904 µs; its selected-attention kernel dominated
  the profiled dispatch time at 1,836.652 µs.
- KDA-P1 was dominated by `_state_scan_fwd_kernel` at 147.441 µs.
- A physical GFX950-versus-GFX1250 speed comparison is invalid until GFX1250 is collected at the
  same TokenSpeed and runner revisions.
