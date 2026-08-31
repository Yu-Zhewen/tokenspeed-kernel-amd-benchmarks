# Latest-main GFX950 matched-attention results

## Status

- Collection date: 2026-08-31 UTC
- GFX950 physical collection: complete
- Overall matched-comparison status: incomplete; GFX1250 has not been collected at this revision
- Required workloads: 6/6 passed
- Profiled measured-scope dispatches: 16
- Previous collection for comparison: [GFX950 at `70f69266`](MATCHED_ATTENTION_RESULTS_GFX950_70f69266.md)
- The most recent complete matched pair remains `70f69266` ([GFX950](MATCHED_ATTENTION_RESULTS_GFX950_70f69266.md),
  [GFX1250](MATCHED_ATTENTION_RESULTS_GFX1250_70f69266.md)). Use that pair for cross-architecture
  comparison; use this report only for GFX950 revision-over-revision tracking.

## Software and hardware setup

| Field | GFX950 | GFX1250 |
|---|---|---|
| Device | AMD Instinct MI355X (PyTorch: `AMD Radeon Graphics`) | unavailable |
| Architecture | `gfx950:sramecc+:xnack-` | `gfx1250` |
| Measurement environment | physical, GPU 0 | unavailable |
| Host / container | `smci355-ccs-aus-n15-05` / `lightseekorg/tokenspeed-amd:tml` | unavailable |
| OS | Ubuntu 24.04.4 container; Linux 6.8.0-84 host | unavailable |
| ROCm / ROCDTIF | `7.2.26015` (`torch.version.hip`); `/opt/rocm` reports `7.2.4-93`, HIP `7.2.53211-97f5574fe2` | unavailable |
| PyTorch | `2.11.0+rocm7.2` | unavailable |
| TokenSpeed commit SHA | `63f45850a221791c219fe891023aae50ceff47e5` | unavailable |
| `tokenspeed-kernel` source | same monorepo commit; tree `9dadb3d6ea85ebebd7623cf192db14697b7190cd` | unavailable |
| `tokenspeed-kernel-amd` source | same monorepo commit; tree `1698cdff764f691e2adc0c0cadb8b366b197d088` | unavailable |
| Kernel package / build ID | source-mounted from the TokenSpeed checkout | unavailable |
| Triton package / build ID | `tokenspeed-triton 3.8.10.post20260709` (`triton.__version__` reports `3.6.0`) | unavailable |
| Runner repository revision | `765372a07beace13bea757fb63501389fa3eee80` plus the `cu_seqlens_cpu` change in this commit | same revision required |
| Runner SHA-256 | `03dd67927e74bd8246999cf58e7287572b432a8233a0dc3601b420dd8c7220dd` | same runner required |
| Latency warmups / repeats | 2 / 5 | unavailable |
| Profiler warmups / repeats | 0 / 1 | unavailable |

Container image: `lightseekorg/tokenspeed-amd@sha256:a6630bce4f1d9e0dd236aa90dcf56f342526ebac30ba3aaf04d17866c4b71fa1`
(image ID `sha256:274fb85449cb34e0ec2b1b728899dabf944ad24d61a269cff73f14490fff1fe4`). This is byte-identical
to the image used for the `70f69266` collection, so the OS, ROCm, PyTorch, and Triton stack are unchanged
and TokenSpeed source is the only moving part.

The tracked TokenSpeed checkout was clean apart from one unrelated untracked file (`hsakmt_counters.csv`),
which was neither read nor modified.

### Setup differences from the `70f69266` collection

Two differences exist and both are recorded here rather than hidden:

1. **Runner change (required by an upstream API break).** At `63f45850`, `kda_paged_prefill` rejects the
   old call and raises `TypeError: kda_paged_prefill() missing 1 required keyword-only argument:
   'cu_seqlens_cpu'`. TokenSpeed
   [#1120](https://github.com/lightseekorg/tokenspeed/pull/1120) made a host `int64` copy of `cu_seqlens`
   mandatory so every KDA prefill solution can plan chunk indices on the host instead of forcing a
   stream-synchronizing device-to-host read per layer. The runner now builds that host tensor once during
   case setup, outside the timed region, and passes it through. The gfx950 gluon solution
   (`gluon_kda_paged_prefill_gfx950`) pops and ignores the argument, so KDA-P1's device work is unchanged
   and remains comparable with `70f69266`.
2. **Profiler output directory name.** `rocprofv3` names its output subdirectory after the hostname it
   observes, which inside this container is the container ID `81b771ff473d`. Those subdirectories were
   renamed to the physical host name `smci355-ccs-aus-n15-05` so the artifact layout matches the previous
   collection. Only directory names were changed; no CSV contents were edited. The `run.log` files still
   contain the original in-container paths.

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
| MLA-D1 | `mla-decode` | 83.960 | unavailable | 2 / 5 | pass |
| MLA-P1 | `mla-prefill` | 141.649 | unavailable | 2 / 5 | pass |
| KDA-D1 | `kda-decode` | 52.520 | unavailable | 2 / 5 | pass |
| KDA-P1 | `kda-prefill` | 294.361 | unavailable | 2 / 5 | pass |
| DSA-D1 | `dsa-decode-pipeline` | 137.249 | unavailable | 2 / 5 | pass |
| DSA-P1 | `dsa-prefill-pipeline-4k` | 2,075.339 | unavailable | 2 / 5 | pass |

Latency is the complete unprofiled API or pipeline time measured with GPU events. It is not a sum
of profiler dispatch durations.

### Change against the `70f69266` collection

Both revisions ran on the same host, same GPU, and the same container image, so these deltas are
attributable to TokenSpeed source changes.

| ID | `70f69266` (µs) | `63f45850` (µs) | Delta |
|---|---:|---:|---:|
| MLA-D1 | 83.937 | 83.960 | +0.03% |
| MLA-P1 | 142.257 | 141.649 | -0.43% |
| KDA-D1 | 51.952 | 52.520 | +1.09% |
| KDA-P1 | 293.306 | 294.361 | +0.36% |
| DSA-D1 | 124.953 | 137.249 | +9.84% |
| DSA-P1 | 2,311.904 | 2,075.339 | -10.23% |

The four Kimi-K3 cases are within run-to-run noise. The two DSA deltas are reproducible: three
back-to-back repetitions of the DSA cases measured 137.625 / 138.625 / 139.641 µs for DSA-D1 and
2,084.499 / 2,088.723 / 2,098.068 µs for DSA-P1, a spread of roughly 1-2%, well inside both deltas.

Both DSA changes come from the same upstream commit,
[`8ed1a67b` "fix(kernel): Correct DSA scoring and add GFX950 standard-cache kernel" (#1196)](https://github.com/lightseekorg/tokenspeed/pull/1196),
which replaced the previously selected FP8 index-logits kernels with a new gfx950 standard-cache
logits kernel. This is a correctness fix that changed kernel selection, not a targeted performance
change, so the DSA-D1 regression and the DSA-P1 improvement are two sides of the same rework:

- DSA-D1: `_dsa_decode_logits_fp8_kernel` (grid 256×128×1, 7.880 µs) became
  `_dsa_standard_decode_logits_kernel` (grid 64×16×1, 12.040 µs). The much smaller grid leaves the
  MI355X largely idle for that dispatch and accounts for the pipeline regression.
- DSA-P1: the `_combine_scoring_query_heads_kernel` plus `_dsa_prefill_logits_fp8_kernel` pair
  (25.001 µs + 479.043 µs) collapsed into a single `_dsa_standard_prefill_logits_kernel` (238.881 µs),
  removing one dispatch and roughly 265 µs.

DSA-D1 is worth raising upstream: a 64×16×1 grid for the decode logits kernel is a plausible
launch-geometry oversight in the new standard-cache path rather than an inherent cost.

## Per-dispatch physical results

The table includes every dispatch in the measured invocation suffix. The runner makes one untimed
setup invocation before the measured invocation; setup-only dispatches are retained in raw traces
but excluded here. Every counter was collected in a separate pass.

| Case | # | Dispatch | Duration µs | Grid | Workgroup | VGPR | SGPR | LDS B | Scratch B | Read MB | Write MB | Read GB/s | Write GB/s | Occupancy | MFMA % | LDS conflict % | LDS latency cyc | Mem stalled % |
|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLA-D1 | 1 | `_mla_decode_gluon` | 10.320 | 256×16×1 | 256×1×1 | 104 | 80 | 0 | 0 | 1.298 | 0.203 | 147.5 | 23.0 | 1.001 | 0.320 | 0.085 | 58.862 | 0.118 |
| MLA-D1 | 2 | `_mla_reduce_project_value_kernel` | 11.400 | 98304×1×1 | 512×1×1 | 16 | 32 | 0 | 0 | 7.113 | 0.006 | 583.1 | 0.5 | 1.979 | 0.000 | 0.000 | 0.000 | 0.012 |
| MLA-P1 | 1 | `_mla_prefill_kernel` | 134.240 | 131072×1×1 | 256×1×1 | 140 | 112 | 0 | 0 | 17.086 | 12.583 | 127.8 | 93.6 | 1.000 | 19.446 | 8.716 | 75.948 | 0.002 |
| KDA-D1 | 1 | `_kda_recurrent_decode_kernel` | 6.080 | 256×12×1 | 64×1×1 | 84 | 48 | 0 | 0 | 0.496 | 0.790 | 91.9 | 151.8 | 1.002 | 0.000 | 0.000 | 0.000 | 0.084 |
| KDA-P1 | 1 | `_preprocess_intra_fwd_kernel` | 68.641 | 16384×12×1 | 256×1×1 | 108 | 48 | 0 | 0 | 19.307 | 62.128 | 288.3 | 936.2 | 1.641 | 2.309 | 5.149 | 64.274 | 0.144 |
| KDA-P1 | 2 | `_solve_merge_64_fwd_kernel` | 30.440 | 4096×12×1 | 64×1×1 | 36 | 32 | 0 | 0 | 4.788 | 3.932 | 172.5 | 163.0 | 1.000 | 0.253 | 0.127 | 44.985 | 0.067 |
| KDA-P1 | 3 | `_wu_vector_fwd_kernel` | 25.680 | 16384×12×1 | 256×1×1 | 96 | 48 | 0 | 0 | 32.321 | 37.749 | 1166.0 | 1346.2 | 1.645 | 1.874 | 5.578 | 148.016 | 0.476 |
| KDA-P1 | 4 | `_state_scan_fwd_kernel` | 152.041 | 4096×12×1 | 256×1×1 | 60 | 112 | 0 | 0 | 203.334 | 51.118 | 1339.5 | 336.7 | 1.000 | 2.448 | 0.000 | 50.083 | 0.136 |
| KDA-P1 | 5 | `_output_fwd_kernel` | 12.800 | 16384×12×1 | 256×1×1 | 52 | 112 | 0 | 0 | 15.770 | 12.583 | 1180.4 | 917.1 | 2.559 | 1.468 | 2.538 | 83.163 | 0.498 |
| DSA-D1 | 1 | `_dsa_standard_decode_logits_kernel` | 12.040 | 64×16×1 | 64×1×1 | 88 | 112 | 0 | 0 | 0.343 | 0.016 | 30.0 | 1.4 | 1.001 | 0.066 | 0.077 | 58.751 | 0.004 |
| DSA-D1 | 2 | `_dsa_oneblock_manual_radix_topk_kernel` | 10.800 | 1024×1×1 | 1024×1×1 | 28 | 112 | 0 | 0 | 0.026 | 0.008 | 2.3 | 0.7 | 3.981 | 0.000 | 0.019 | 117.702 | 0.018 |
| DSA-D1 | 3 | `_dsa_dense_mfma_kv_kernel` | 12.441 | 256×1×32 | 256×1×1 | 164 | 112 | 0 | 0 | 0.844 | 0.270 | 72.8 | 23.1 | 1.001 | 0.609 | 0.457 | 85.633 | 0.003 |
| DSA-D1 | 4 | `_dsa_dense_mfma_reduce_kernel` | 7.600 | 256×8×1 | 256×1×1 | 8 | 64 | 0 | 0 | 0.148 | 0.008 | 20.2 | 1.1 | 1.002 | 0.000 | 0.000 | 0.000 | 0.002 |
| DSA-P1 | 1 | `_dsa_standard_prefill_logits_kernel` | 238.881 | 1048576×1×1 | 256×1×1 | 52 | 112 | 0 | 0 | 19.629 | 33.620 | 82.5 | 139.2 | 1.880 | 12.311 | 12.287 | 76.509 | 0.025 |
| DSA-P1 | 2 | `_dsa_oneblock_manual_radix_topk_kernel` | 56.960 | 4194304×1×1 | 1024×1×1 | 24 | 80 | 0 | 0 | 13.010 | 33.686 | 233.0 | 581.2 | 7.036 | 0.000 | 11.385 | 135.078 | 0.056 |
| DSA-P1 | 3 | `_dsa_dense_mfma_kv_kernel` | 1825.251 | 1048576×1×1 | 256×1×1 | 164 | 112 | 0 | 0 | 35.860 | 33.554 | 19.6 | 18.3 | 1.000 | 26.502 | 12.707 | 92.051 | 0.000 |

No stable L2/cache-hit or VMEM dependency-latency counter was collected; those required fields are
`unavailable`. AM cycles are not applicable to this physical GFX950 collection.

### Dispatch-set changes against `70f69266`

The measured-scope dispatch count fell from 19 to 16. Every difference was confirmed against the raw
kernel traces rather than inferred, and no dispatch was dropped by the suffix-selection logic:

| Case | `70f69266` | `63f45850` | Change |
|---|---:|---:|---|
| MLA-D1 | 2 | 2 | unchanged |
| MLA-P1 | 1 | 1 | unchanged |
| KDA-D1 | 1 | 1 | unchanged |
| KDA-P1 | 7 | 5 | two in-pipeline `at::native::BF16 fill` dispatches no longer occur |
| DSA-D1 | 4 | 4 | FP8 logits kernel replaced by the standard-cache logits kernel |
| DSA-P1 | 4 | 3 | `_combine_scoring_query_heads_kernel` folded into the new logits kernel |

For KDA-P1, the two BF16 fill dispatches that previously ran inside the measured pipeline (4.080 µs and
4.160 µs, grid 393216×1×1, one full BF16 `[1, 4096, 12, 128]` tensor each) are gone, and the five compute
kernels are unchanged in name and ordering. gfx950 KDA prefill was restructured in this commit range by
[`074049d4` "perf(gfx950): use V-major recurrent state for KDA" (#1176)](https://github.com/lightseekorg/tokenspeed/pull/1176);
the exact commit that removed the fills was not isolated. End-to-end KDA-P1 latency is unchanged
(+0.36%), so the fills were overlapped or cheap relative to the scan.

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

Duration, grid, workgroup, VGPR, SGPR, LDS, and scratch come from the unprofiled-counter
`--kernel-trace` pass. Counter values come from their own single-counter passes.

## Incomplete or failed work

| Case / dispatch | Architecture | Stage | Status | Reason | Raw artifact |
|---|---|---|---|---|---|
| All cases | GFX1250 | latency/profile/capture/replay | incomplete | No same-revision GFX1250 run was requested or collected; the newest GFX1250 data is at `70f69266` | `MATCHED_ATTENTION_RESULTS_GFX1250_70f69266.md` |
| KDA-P1 | GFX950 | latency | resolved | First attempt aborted on the `cu_seqlens_cpu` API break; rerun after the runner was updated | `gfx950_63f45850/latency.json` |

All 48 profiler passes (6 cases × 1 trace + 7 counters) returned exit status 0.

## Exact commands

Run from inside the `lightseekorg/tokenspeed-amd:tml` container with the TokenSpeed checkout at
`/workspace`:

```bash
export TS=/workspace
export PYTHONPATH="$TS/python:$TS/tokenspeed-kernel/python:$TS/tokenspeed-kernel-amd/python"
export ROCR_VISIBLE_DEVICES=0
```

### GFX950 case description

```bash
python profile_matched_attention.py --describe > matched-attention-cases.json
```

### GFX950 latency

```bash
python profile_matched_attention.py \
  --case all \
  --expected-arch gfx950 \
  --environment physical \
  --warmup 2 \
  --repeats 5 \
  --output latency.json
```

### GFX950 trace

```bash
rocprofv3 \
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
rocprofv3 \
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

### GFX950 counter listings

```bash
rocprofv3-avail list --pmc > rocprofv3-pmcs.txt
rocprofv3-avail info --pmc FetchSize MemWrites32B MeanOccupancyPerActiveCU MfmaUtil \
  LDSBankConflict LdsLatency MemUnitStalled > rocprofv3-counter-info.txt
```

The trace and each of `FetchSize`, `MemWrites32B`, `MeanOccupancyPerActiveCU`, `MfmaUtil`,
`LDSBankConflict`, `LdsLatency`, and `MemUnitStalled` were collected for every case. Separate passes
avoid unsupported counter-group scheduling and preserve one unambiguous value per dispatch.

## Raw artifacts

- Runner description: `gfx950_63f45850/matched-attention-cases.json`
- Unprofiled runner JSON: `gfx950_63f45850/latency.json`
- Dispatch traces and seven counter passes per case: `gfx950_63f45850/profiles/`
- Available-counter listing: `gfx950_63f45850/rocprofv3-pmcs.txt`
- Exact counter definitions: `gfx950_63f45850/rocprofv3-counter-info.txt`

## Conclusions

- All six required latest-main workloads completed on physical MI355X at TokenSpeed `63f45850`.
- The Kimi-K3 MLA and KDA cases are unchanged from `70f69266` within run-to-run noise, despite the
  gfx950 KDA prefill rework and the mandatory `cu_seqlens_cpu` API change.
- DSA-P1 improved 10.23% to 2,075.339 µs and DSA-D1 regressed 9.84% to 137.249 µs. Both follow from
  the DSA scoring correctness fix in #1196 replacing the FP8 index-logits kernels with the new gfx950
  standard-cache logits kernel.
- DSA-P1 remains the slowest workload, and `_dsa_dense_mfma_kv_kernel` still dominates it at
  1,825.251 µs, 88% of the profiled dispatch time. That kernel is the highest-value optimization
  target and is unchanged by #1196.
- The DSA-D1 regression is concentrated in `_dsa_standard_decode_logits_kernel`, which launches only
  64×16×1 workgroups and leaves the GPU underutilized. This looks like a launch-geometry oversight in
  the new standard-cache path and should be raised upstream.
- A physical GFX950-versus-GFX1250 speed comparison is invalid until GFX1250 is collected at the
  same TokenSpeed and runner revisions.
