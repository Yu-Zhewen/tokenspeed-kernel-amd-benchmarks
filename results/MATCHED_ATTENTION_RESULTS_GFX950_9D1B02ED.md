# Latest-main GFX950 matched-attention results

## Status

- Collection date: 2026-08-13 UTC
- GFX950 physical collection: complete
- Overall matched-comparison status: incomplete; GFX1250 has not been collected at this revision
- Required workloads: 6/6 passed
- Profiled measured-scope dispatches: 20

## Software and hardware setup

| Field | Value |
|---|---|
| Device | AMD Instinct MI355X (PyTorch: `AMD Radeon Graphics`) |
| Architecture | `gfx950:sramecc+:xnack-` |
| Environment | physical, GPU 0 |
| Container image | `lightseekorg/tokenspeed-amd:tml` |
| PyTorch | `2.11.0+rocm7.2` |
| ROCm | `7.2.26015` |
| TokenSpeed commit | `9d1b02ed9c001544e35a41aa4b216dc8caa2240a` |
| Runner gist revision | `29150c094de69af0c13fc39a92237ba47bd59fbc` |
| Runner SHA-256 | `f8d1fad666ec89eec46a05acf60a913b3c9e1e8dd7edb334efd4f525a59217dc` |
| Latency warmups / repeats | 2 / 5 |
| Profiler warmups / repeats | 0 / 1 |

A clean detached worktree was used because the pulled checkout contained unrelated untracked artifacts.

## Physical workload latency

| ID | Runner case | Latency (µs) | Status |
|---|---|---:|---|
| MLA-D1 | `mla-decode` | 85.425 | pass |
| MLA-P1 | `mla-prefill` | 142.105 | pass |
| KDA-D1 | `kda-decode` | 45.216 | pass |
| KDA-P1 | `kda-prefill` | 292.954 | pass |
| DSA-D1 | `dsa-decode-pipeline` | 138.825 | pass |
| DSA-P1 | `dsa-prefill-pipeline-4k` | 2,327.593 | pass |

Latency is the complete unprofiled API/pipeline time measured with GPU events. It is not a sum of profiler dispatch durations.

## Per-dispatch physical results

The table includes every dispatch in the measured invocation suffix. DSA-P1 includes three framework utility dispatches that occur inside the timed pipeline before logits, radix selection, and dense attention.

| Case | # | Dispatch | Duration µs | Grid | Workgroup | VGPR | SGPR | LDS B | Scratch B | Read MB | Write MB | Read GB/s | Write GB/s | Occupancy | MFMA % | LDS conflict % | LDS latency cyc | Mem stalled % |
|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLA-D1 | 1 | `_mla_decode_gluon` | 10.320 | 256×16×1 | 256×1×1 | 104 | 80 | 0 | 0 | 1.298 | 0.203 | 147.5 | 22.6 | 1.001 | 0.307 | 0.085 | 58.895 | 0.121 |
| MLA-D1 | 2 | `_mla_reduce_project_value_kernel` | 11.240 | 98304×1×1 | 512×1×1 | 16 | 32 | 0 | 0 | 7.113 | 0.006 | 594.8 | 0.5 | 1.979 | 0.000 | 0.000 | 0.000 | 0.011 |
| MLA-P1 | 1 | `_mla_prefill_kernel` | 134.201 | 131072×1×1 | 256×1×1 | 140 | 112 | 0 | 0 | 17.086 | 12.583 | 127.9 | 94.1 | 1.000 | 19.392 | 8.634 | 76.007 | 0.002 |
| KDA-D1 | 1 | `_kda_recurrent_decode_kernel` | 6.160 | 256×12×1 | 64×1×1 | 92 | 48 | 0 | 0 | 0.500 | 0.790 | 96.8 | 151.8 | 1.002 | 0.000 | 0.000 | 0.000 | 0.015 |
| KDA-P1 | 1 | `at::native::BF16 fill` | 4.720 | 393216×1×1 | 256×1×1 | 4 | 16 | 0 | 0 | 0.009 | 6.291 | 2.9 | 2246.9 | 4.682 | 0.000 | 0.000 | 0.000 | 0.033 |
| KDA-P1 | 2 | `_preprocess_intra_fwd_kernel` | 73.681 | 16384×12×1 | 256×1×1 | 108 | 48 | 0 | 0 | 19.305 | 62.128 | 289.5 | 925.1 | 1.642 | 2.257 | 5.168 | 64.299 | 0.144 |
| KDA-P1 | 3 | `at::native::BF16 fill` | 4.240 | 393216×1×1 | 256×1×1 | 4 | 16 | 0 | 0 | 0.009 | 6.291 | 2.7 | 2097.2 | 4.659 | 0.000 | 0.000 | 0.000 | 0.034 |
| KDA-P1 | 4 | `_solve_merge_64_fwd_kernel` | 25.280 | 4096×12×1 | 64×1×1 | 36 | 32 | 0 | 0 | 4.788 | 3.932 | 198.8 | 163.8 | 1.000 | 0.244 | 0.129 | 45.000 | 0.060 |
| KDA-P1 | 5 | `_wu_vector_fwd_kernel` | 26.481 | 16384×12×1 | 256×1×1 | 96 | 48 | 0 | 0 | 31.837 | 37.749 | 1172.2 | 1398.1 | 1.643 | 1.792 | 5.766 | 152.557 | 0.461 |
| KDA-P1 | 6 | `_state_scan_fwd_kernel` | 148.521 | 4096×12×1 | 256×1×1 | 60 | 112 | 0 | 0 | 204.515 | 51.118 | 1410.8 | 354.0 | 1.000 | 2.577 | 1.263 | 43.827 | 0.271 |
| KDA-P1 | 7 | `_output_fwd_kernel` | 12.880 | 16384×12×1 | 256×1×1 | 52 | 112 | 0 | 0 | 15.763 | 12.589 | 1176.3 | 936.7 | 2.549 | 1.417 | 2.878 | 84.640 | 0.506 |
| DSA-D1 | 1 | `_dsa_decode_logits_fp8_kernel` | 7.840 | 256×128×1 | 256×1×1 | 128 | 64 | 0 | 0 | 0.386 | 0.016 | 62.6 | 2.7 | 1.002 | 0.000 | 0.000 | 0.000 | 0.003 |
| DSA-D1 | 2 | `_dsa_oneblock_manual_radix_topk_kernel` | 10.480 | 1024×1×1 | 1024×1×1 | 28 | 112 | 0 | 0 | 0.025 | 0.008 | 2.4 | 0.8 | 3.975 | 0.000 | 0.018 | 117.573 | 0.023 |
| DSA-D1 | 3 | `_dsa_dense_mfma_kv_kernel` | 106.521 | 256×1×1 | 256×1×1 | 164 | 112 | 0 | 0 | 0.650 | 0.008 | 5.9 | 0.1 | 1.000 | 0.097 | 0.045 | 96.903 | 0.000 |
| DSA-P1 | 1 | `at::native::vectorized_elementwise_kernel` | 17.440 | 2097152×1×1 | 256×1×1 | 8 | 32 | 0 | 0 | 16.795 | 67.109 | 1057.6 | 4092.0 | 6.973 | 0.000 | 0.000 | 0.000 | 0.987 |
| DSA-P1 | 2 | `at::native::elementwise_kernel_manual_unroll` | 23.520 | 4194304×1×1 | 128×1×1 | 16 | 80 | 0 | 0 | 35.692 | 67.109 | 1509.8 | 2829.1 | 7.209 | 0.000 | 0.000 | 0.000 | 0.615 |
| DSA-P1 | 3 | `at::native::reduce_kernel` | 17.200 | 131072×2×1 | 64×2×1 | 28 | 80 | 512 | 0 | 33.644 | 2.097 | 2113.3 | 128.8 | 3.635 | 0.000 | 0.000 | 973.630 | 0.035 |
| DSA-P1 | 4 | `_dsa_prefill_logits_fp8_kernel` | 490.524 | 1048576×128×1 | 256×1×1 | 20 | 48 | 0 | 0 | 3.619 | 67.109 | 7.6 | 139.8 | 6.678 | 0.000 | 0.000 | 0.000 | 0.174 |
| DSA-P1 | 5 | `_dsa_oneblock_manual_radix_topk_kernel` | 56.160 | 4194304×1×1 | 1024×1×1 | 24 | 80 | 0 | 0 | 13.007 | 33.686 | 240.5 | 618.8 | 7.020 | 0.000 | 11.641 | 134.291 | 0.057 |
| DSA-P1 | 6 | `_dsa_dense_mfma_kv_kernel` | 1870.734 | 1048576×1×1 | 256×1×1 | 164 | 112 | 0 | 0 | 35.856 | 33.554 | 19.4 | 18.4 | 1.000 | 26.785 | 12.715 | 96.901 | 0.000 |

No stable L2/cache-hit counter was collected; that required field is `unavailable`.

## Counter definitions

| Counter | Definition |
|---|---|
| `FetchSize` | Total kilobytes fetched from video memory, including cache and memory effects (`FETCH_SIZE`). |
| `MemWrites32B` | Effective 32-byte write transactions (`WRITE_REQ_32B`). |
| `MeanOccupancyPerActiveCU` | Mean wave occupancy per active CU. |
| `MfmaUtil` | MFMA busy cycles divided by active cycles and SIMD count, percent. |
| `LDSBankConflict` | Percentage of GPU time stalled by LDS bank conflicts. |
| `LdsLatency` | Average LDS instruction dependency latency, cycles. |
| `MemUnitStalled` | Percentage of GPU time the memory unit is stalled. |

Read bytes are `FetchSize × 1024`; write bytes are `MemWrites32B × 32`. Each bandwidth uses the duration from the same single-counter pass, not the separate trace duration.

## Exact command shapes

### Latency

```bash
python profile_matched_attention.py --case all --expected-arch gfx950 \
  --environment physical --warmup 2 --repeats 5 --output latency.json
```

### Trace

```bash
rocprofv3 --kernel-trace --output-format csv --output-directory "$CASE/trace" -- \
  python profile_matched_attention.py --case "$CASE" --expected-arch gfx950 \
    --environment physical --warmup 0 --repeats 1
```

### Single-counter pass

```bash
rocprofv3 --kernel-trace --pmc "$COUNTER" --output-format csv \
  --output-directory "$CASE/counter-$COUNTER" -- \
  python profile_matched_attention.py --case "$CASE" --expected-arch gfx950 \
    --environment physical --warmup 0 --repeats 1
```

Counters were collected in seven separate passes per case because this GFX950 profiler cannot schedule the full group in one hardware pass.

## Artifacts

- Latency JSON: `gfx950/latency.json`
- Latency log: `gfx950/latency.log`
- Dispatch traces and seven counter passes per case: `gfx950/profiles/`
- Available-counter listing: `gfx950/rocprofv3-pmcs.txt`
- Exact counter definitions: `gfx950/rocprofv3-counter-info.txt`
- Case description: `matched-attention-cases.json`

## Conclusions

- All six latest-main workloads completed successfully on physical GFX950.
- Complete workload latency is preserved separately from instrumented per-dispatch measurements.
- A matched comparison against GFX1250 is not valid until GFX1250 is run at the same `9d1b02ed` revision.
