# Kimi-K3 gfx950 toy 1-GPU result at `0b1061eb`

## Status

- Collection date: 2026-09-02 UTC
- Target: gfx950 toy 1-GPU logical TP8 rank 0
- Overall status: **complete**
- Performance cases: C1 pass, C16 pass
- Stage profiles: C1 prefill pass, C1 decode pass, C16 prefill pass,
  C16 decode pass
- Missing or invalid data: none

## Software and hardware setup

| Field | Value |
|---|---|
| Device | AMD Instinct MI355X (PyTorch: `AMD Radeon Graphics`) |
| Architecture | `gfx950:sramecc+:xnack-` |
| Physical GPUs / ranks | 1 / 1 |
| Measurement environment | physical |
| Host / container | Linux 6.8.0-84 / `zhewenyu/kimi-k3-e2e:tokenspeed-0b1061eb` |
| OS | Ubuntu 24.04 container |
| ROCm / HIP | 7.2 / `7.2.26015` |
| PyTorch | `2.11.0+rocm7.2` |
| Transformers | `5.12.0` |
| Triton package / module | TokenSpeed Triton / `3.6.0` |
| EvalScope | unavailable; direct logical-rank harness |
| TokenSpeed commit | `0b1061eb9fe1df36a4e48e5c9c291cd753af9e89` |
| Kimi-K3 revision | `eaf5a944bfc8c57438bbce226feef9f6bdbdaae1` |

## Workload and topology

| Field | Value |
|---|---|
| Checkpoint | `/data/models/kimi-k3-tp8ep1-rank0`, portable raw-rank-state |
| Prompt / output | 4096 / 1024 tokens |
| Concurrency | 1 and 16 |
| Prefill budget | 8192 tokens |
| Attention / dense / MoE / EP | TP8 / TP8 / TP8 / EP1 |
| KV cache | FP8 E4M3, 32 GiB, 2,334,336 scheduler-visible tokens |
| Prefix cache / host KV | disabled / disabled |
| Sampling | deterministic token substitute; no HTTP sampler |
| Performance measurement | complete eager scheduler run plus 20 unprofiled decode-graph replays |
| Hotspot measurement | separate eager PyTorch/roctracer run |

One physical GPU executes all 93 layers for logical TP8 rank 0. Rank-spanning
collectives record shapes and bytes but use local substitutes, so no RCCL
kernel appears in this result.

## Correctness

| C | Requests or sequences completed | Exact input length | Exact output length | Failures | Status |
|---:|---:|---:|---:|---:|---|
| 1 | 1/1 | 4096 | 1024 | 0 | pass |
| 16 | 16/16 | 4096 | 1024 | 0 | pass |

The result also records 93 layers, 896 local experts, the gfx950 Gluon MoE
solution, 209.95 GiB model allocation, and a 244.69 GiB runtime peak.

## Unprofiled performance

| C | First-token p50 / p90 | Primary decode p50 / p90 | Per-user decode | Overall output | Steady decode capacity | Scope |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 277.58 / 277.58 ms | 10.711 / 10.727 ms | 93.35 tok/s | 15.20 tok/s | 93.35 tok/s | logical rank, graph B1 |
| 16 | 1,948.78 / 3,247.79 ms | 18.553 / 18.579 ms | 53.90 tok/s | 196.90 tok/s | 862.47 tok/s | logical rank, graph B16 |

Primary decode latency is the static first full decode batch after prefill.
Overall output is the complete 1024-token eager scheduler workload and
includes Python/launch overhead. Steady capacity is
`concurrency / mean graph latency`; it excludes real inter-rank communication
and serving overhead.

## Stage hotspot summary

These are separate eager traces normalized with the same classifier and CSV
schema as the real eight-GPU result.

| C | Stage | Ranks | Forwards | Summed GPU time | Communication | MoE | KDA | MLA / attention | GEMM / quant | Other |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | prefill | 1 | 1 | 269.41 ms | 0.00% | 45.45% | 15.70% | 8.61% | 21.16% | 9.09% |
| 1 | decode | 1 | 64 | 749.32 ms | 0.00% | 12.29% | 18.83% | 27.20% | 30.47% | 11.21% |
| 16 | prefill | 1 | 8 | 3,423.67 ms | 0.00% | 36.79% | 18.91% | 10.49% | 25.00% | 8.81% |
| 16 | decode | 1 | 64 | 1,313.21 ms | 0.00% | 24.45% | 13.43% | 11.50% | 37.57% | 13.05% |

`Other` combines unclassified, elementwise/reduction, normalization, cache,
and sampling categories. The exact-name files are authoritative.

## Top exact kernels

The top 10 kernels for each setting are ranked by total profiled GPU duration.
Names are copied verbatim from the GPU profiler; they are not model-component
or layer labels.

| C | Stage | Order | Exact kernel name | Calls | Total across ranks | GPU share | Average call |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | prefill | 1 | `gluon_mxfp4_moe_stage1_kernel` | 92 | 59.92 ms | 22.24% | 651.25 µs |
| 1 | prefill | 2 | `gluon_mxfp4_moe_stage2_1x2_kernel` | 92 | 45.57 ms | 16.91% | 495.29 µs |
| 1 | prefill | 3 | `_packed_input_projections_kernel` | 92 | 27.08 ms | 10.05% | 294.31 µs |
| 1 | prefill | 4 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 69 | 21.83 ms | 8.10% | 316.35 µs |
| 1 | prefill | 5 | `_attn_res_rmsnorm_kernel` | 187 | 18.93 ms | 7.03% | 101.23 µs |
| 1 | prefill | 6 | `_mfma_lds_largem_kernel` | 92 | 14.26 ms | 5.29% | 155.02 µs |
| 1 | prefill | 7 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1` | 186 | 12.40 ms | 4.60% | 66.67 µs |
| 1 | prefill | 8 | `gluon_mxfp4_moe_stage2_reduce_kernel` | 92 | 10.87 ms | 4.04% | 118.18 µs |
| 1 | prefill | 9 | `_state_scan_fwd_kernel` | 69 | 10.36 ms | 3.85% | 150.14 µs |
| 1 | prefill | 10 | `_gather_package_cdna4_scale_kernel` | 92 | 5.36 ms | 1.99% | 58.27 µs |
| 1 | decode | 1 | `_latent_input_decode_kernel` | 5,888 | 107.78 ms | 14.38% | 18.30 µs |
| 1 | decode | 2 | `_linear_attnres_partials_kernel` | 5,440 | 91.79 ms | 12.25% | 16.87 µs |
| 1 | decode | 3 | `_warp_decode_precomputed_situ_stage1_kernel` | 5,888 | 72.62 ms | 9.69% | 12.33 µs |
| 1 | decode | 4 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4` | 5,952 | 66.33 ms | 8.85% | 11.14 µs |
| 1 | decode | 5 | `_rowcta_gemv_add3_kernel` | 5,888 | 62.19 ms | 8.30% | 10.56 µs |
| 1 | decode | 6 | `_attnres_combine_kernel` | 11,840 | 47.31 ms | 6.31% | 4.00 µs |
| 1 | decode | 7 | `_warp_decode_stage2_fp8_mxfp4_kernel` | 5,888 | 36.49 ms | 4.87% | 6.20 µs |
| 1 | decode | 8 | `_kda_fused_decode_kernel` | 4,416 | 33.35 ms | 4.45% | 7.55 µs |
| 1 | decode | 9 | `_kimi3_projection_gemv_kernel` | 5,888 | 29.84 ms | 3.98% | 5.07 µs |
| 1 | decode | 10 | `_kimi3_sigmoid_bias_topk_kernel` | 5,888 | 25.34 ms | 3.38% | 4.30 µs |
| 16 | prefill | 1 | `gluon_mxfp4_moe_stage1_kernel` | 736 | 652.81 ms | 19.07% | 886.97 µs |
| 16 | prefill | 2 | `_packed_input_projections_kernel` | 736 | 426.70 ms | 12.46% | 579.75 µs |
| 16 | prefill | 3 | `gluon_mxfp4_moe_stage2_1x2_kernel` | 736 | 376.11 ms | 10.99% | 511.01 µs |
| 16 | prefill | 4 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 552 | 326.46 ms | 9.54% | 591.42 µs |
| 16 | prefill | 5 | `_attn_res_rmsnorm_kernel` | 1,496 | 293.69 ms | 8.58% | 196.31 µs |
| 16 | prefill | 6 | `_mfma_lds_largem_kernel` | 736 | 225.22 ms | 6.58% | 306.01 µs |
| 16 | prefill | 7 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1` | 1,488 | 187.49 ms | 5.48% | 126.00 µs |
| 16 | prefill | 8 | `gluon_mxfp4_moe_stage2_reduce_kernel` | 736 | 164.77 ms | 4.81% | 223.88 µs |
| 16 | prefill | 9 | `_state_scan_fwd_kernel` | 552 | 148.21 ms | 4.33% | 268.49 µs |
| 16 | prefill | 10 | `_preprocess_intra_fwd_kernel` | 552 | 72.54 ms | 2.12% | 131.42 µs |
| 16 | decode | 1 | `_warp_decode_precomputed_situ_stage1_kernel` | 5,888 | 220.70 ms | 16.81% | 37.48 µs |
| 16 | decode | 2 | `_packed_projection_gemm_kernel` | 5,888 | 130.80 ms | 9.96% | 22.21 µs |
| 16 | decode | 3 | `_attn_res_rmsnorm_kernel` | 11,968 | 115.42 ms | 8.79% | 9.64 µs |
| 16 | decode | 4 | `_warp_decode_stage2_fp8_mxfp4_kernel` | 5,888 | 112.29 ms | 8.55% | 19.07 µs |
| 16 | decode | 5 | `_mfma_lds_mediumm_kernel` | 5,888 | 99.61 ms | 7.58% | 16.92 µs |
| 16 | decode | 6 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4` | 4,416 | 92.62 ms | 7.05% | 20.97 µs |
| 16 | decode | 7 | `_sigmoid_bias_topk_route_prefill_kernel` | 5,888 | 57.13 ms | 4.35% | 9.70 µs |
| 16 | decode | 8 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4` | 6,016 | 57.11 ms | 4.35% | 9.49 µs |
| 16 | decode | 9 | `_dynamic_fp8_quantize_kernel` | 5,888 | 52.45 ms | 3.99% | 8.91 µs |
| 16 | decode | 10 | `_kda_fused_decode_kernel` | 4,416 | 45.62 ms | 3.47% | 10.33 µs |

## Incomplete or failed work

| Case | Stage | Status | Error or reason | Raw artifact |
|---|---|---|---|---|
| none | all | complete | none | `result.json` and `hotspots/` |

## Exact commands

### Performance

```bash
python3 toy_e2e/benchmark_logical_rank.py \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --expected-arch gfx950 \
  --tokenspeed-revision 0b1061eb9fe1df36a4e48e5c9c291cd753af9e89 \
  --model-revision eaf5a944bfc8c57438bbce226feef9f6bdbdaae1 \
  --container-image zhewenyu/kimi-k3-e2e:tokenspeed-0b1061eb \
  --prompt-tokens 4096 --output-tokens 1024 \
  --concurrency 1 16 --chunked-prefill-size 8192 --cache-gib 32 \
  --warmup-output-tokens 2 --profile-output-tokens 8 \
  --decode-graph-replays 20 --output result.json
```

### Stage profiles

```bash
TOKENSPEED_KERNEL_PROFILE_BACKEND=roctracer \
python3 toy_e2e/scripts/profile_logical_rank_stages.py \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --expected-arch gfx950 \
  --output-dir /data/results/kimi-k3-toy-1gpu-gfx950-0b1061eb-recollect/eager-profile \
  --tokenspeed-revision 0b1061eb9fe1df36a4e48e5c9c291cd753af9e89 \
  --model-revision eaf5a944bfc8c57438bbce226feef9f6bdbdaae1 \
  --prompt-tokens 4096 --concurrency 1 16 \
  --chunked-prefill-size 8192 --cache-gib 32 --decode-steps 64
```

### Hotspot aggregation

```bash
python3 toy_e2e/scripts/summarize_gpu_hotspots.py \
  --input /data/results/kimi-k3-toy-1gpu-gfx950-0b1061eb-recollect/eager-profile \
  --top-k 15 --csv-dir hotspots/csv --output hotspots/hotspots.json
```

See [`../../RUNBOOK.md`](../../RUNBOOK.md) for container, environment, and
validation setup.

## Raw artifacts

- Primary result JSON: [`result.json`](result.json)
- Complete run log: [`run.log`](run.log)
- Hotspot summary: [`hotspots/hotspots.json`](hotspots/hotspots.json)
- Exact-name CSVs: [`hotspots/csv/`](hotspots/csv/)
- Raw traces and manifest:
  `/data/results/kimi-k3-toy-1gpu-gfx950-0b1061eb-recollect/eager-profile/`
- Service logs and EvalScope outputs: unavailable for a direct logical-rank run

## Conclusions and limitations

- The graph estimate reaches 93.35 tok/s at C1 and 862.47 aggregate tok/s at
  C16.
- MoE is the largest prefill category: 45.45% at C1 and 36.79% at C16.
- GEMM/quant is the largest decode category: 30.47% at C1 and 37.57% at C16.
- Communication is intentionally absent. This result cannot measure RCCL,
  cross-rank synchronization, queueing, HTTP, tokenization, or sampling.
- FP8 KV scaling factors were unavailable and defaulted to 1.0. This is a
  performance/scheduling result, not a model-quality qualification.

Profiled GPU-duration sums are hotspot weights, not critical-path latency.
Profiler instrumentation perturbs execution and must not replace the
unprofiled graph measurements.
