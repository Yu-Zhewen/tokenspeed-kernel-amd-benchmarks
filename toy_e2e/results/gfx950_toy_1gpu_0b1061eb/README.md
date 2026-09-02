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
| Sampling | rank-local greedy, `ignore_eos=true` |
| Prompt source | deterministic varied IDs, seed 7, vocabulary range 160,000 |
| Warmup / measured requests | C1: 1 / 3; C16: 16 / 48 |
| Decode graphs / scheduling | buckets 1/2/4/8/16; depth-1 dispatch/commit overlap |
| Performance measurement | complete rolling `ModelExecutor` CUDA-graph workload |
| Hotspot measurement | separate eager PyTorch/roctracer run; varied prompts and deterministic decode input ID 1 |

One physical GPU executes all 93 layers for logical TP8 rank 0. Rank-spanning
collectives record shapes and bytes but use local substitutes, so no RCCL
kernel appears in this result.

## Correctness

| C | Requests or sequences completed | Exact input length | Exact output length | Failures | Status |
|---:|---:|---:|---:|---:|---|
| 1 | 3/3 after 1 warmup | 4096 | 1024 | 0 | pass |
| 16 | 48/48 after 16 warmups | 4096 | 1024 | 0 | pass |

The result also records 93 layers, 896 local experts, the gfx950 Gluon MoE
solution, 209.95 GiB model allocation, and a 244.86 GiB runtime peak.

## Unprofiled performance

| C | First-token p50 / p90 | Primary decode p50 / p90 | Per-user decode | Overall output | Steady decode capacity | Scope |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 286.60 / 286.63 ms | 12.531 / 12.547 ms | 79.78 tok/s | 78.12 tok/s | 79.78 tok/s | logical rolling graph B1 |
| 16 | 1,955.96 / 3,461.99 ms | 25.418 / 26.865 ms | 39.35 tok/s | 586.09 tok/s | 668.47 tok/s | logical rolling graph B16 |

Primary decode is request TPOT across all 1024 generated tokens. Overall
output includes eager prefill and three complete measured waves. Steady
capacity excludes the first decode transition, where C16 requests that finish
prefill early can wait for the remaining prompt chunks. All metrics exclude
real inter-rank communication and HTTP serving.

| C | Decode input context | Resulting context | Step p50 / p90 | Samples |
|---:|---:|---:|---:|---:|
| 1 | 4097 | 4098 | 12.466 / 12.480 ms | 3 |
| 1 | 4224 | 4225 | 12.496 / 12.521 ms | 3 |
| 1 | 4352 | 4353 | 12.461 / 12.490 ms | 3 |
| 1 | 4480 | 4481 | 12.461 / 12.575 ms | 3 |
| 1 | 4608 | 4609 | 12.500 / 12.501 ms | 3 |
| 1 | 4736 | 4737 | 12.462 / 12.516 ms | 3 |
| 1 | 4864 | 4865 | 12.502 / 12.578 ms | 3 |
| 1 | 4992 | 4993 | 12.497 / 12.542 ms | 3 |
| 1 | 5119 | 5120 | 12.551 / 12.611 ms | 3 |
| 16 | 4097 | 4098 | 1,534.804 / 3,042.407 ms | 48 |
| 16 | 4224 | 4225 | 23.772 / 24.079 ms | 48 |
| 16 | 4352 | 4353 | 23.828 / 24.040 ms | 48 |
| 16 | 4480 | 4481 | 23.798 / 23.819 ms | 48 |
| 16 | 4608 | 4609 | 23.774 / 23.853 ms | 48 |
| 16 | 4736 | 4737 | 23.874 / 23.888 ms | 48 |
| 16 | 4864 | 4865 | 23.806 / 23.882 ms | 48 |
| 16 | 4992 | 4993 | 23.784 / 23.785 ms | 48 |
| 16 | 5119 | 5120 | 23.944 / 24.095 ms | 48 |

The large C16 sample at 4097 is intentional: TTFT occurs as each request's
prefill chunk completes, but the first decode batch waits for all 16 prompts.
Request TPOT includes this scheduling gap; steady capacity does not.

## Stage hotspot summary

These are separate eager traces normalized with the same classifier and CSV
schema as the real eight-GPU result.

| C | Stage | Ranks | Forwards | Summed GPU time | Communication | MoE | KDA | MLA / attention | GEMM / quant | Other |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | prefill | 1 | 1 | 271.69 ms | 0.00% | 45.99% | 15.59% | 8.53% | 20.95% | 8.94% |
| 1 | decode | 1 | 64 | 743.61 ms | 0.00% | 12.19% | 18.58% | 27.51% | 30.56% | 11.16% |
| 16 | prefill | 1 | 8 | 5,847.08 ms | 0.00% | 22.02% | 11.08% | 6.13% | 14.60% | 46.16% |
| 16 | decode | 1 | 64 | 1,504.21 ms | 0.00% | 30.27% | 11.93% | 10.47% | 36.36% | 10.97% |

`Other` combines unclassified, elementwise/reduction, normalization, cache,
and sampling categories. The exact-name files are authoritative.

## Top exact kernels

The top 10 kernels for each setting are ranked by total profiled GPU duration.
Names are copied verbatim from the GPU profiler; they are not model-component
or layer labels.

| C | Stage | Order | Exact kernel name | Calls | Total across ranks | GPU share | Average call |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | prefill | 1 | `gluon_mxfp4_moe_stage1_kernel` | 92 | 61.45 ms | 22.62% | 667.97 µs |
| 1 | prefill | 2 | `gluon_mxfp4_moe_stage2_1x2_kernel` | 92 | 46.72 ms | 17.20% | 507.79 µs |
| 1 | prefill | 3 | `_packed_input_projections_kernel` | 92 | 27.06 ms | 9.96% | 294.18 µs |
| 1 | prefill | 4 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 69 | 21.79 ms | 8.02% | 315.82 µs |
| 1 | prefill | 5 | `_attn_res_rmsnorm_kernel` | 187 | 18.94 ms | 6.97% | 101.27 µs |
| 1 | prefill | 6 | `_mfma_lds_largem_kernel` | 92 | 14.23 ms | 5.24% | 154.65 µs |
| 1 | prefill | 7 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1` | 186 | 12.41 ms | 4.57% | 66.72 µs |
| 1 | prefill | 8 | `gluon_mxfp4_moe_stage2_reduce_kernel` | 92 | 10.74 ms | 3.95% | 116.79 µs |
| 1 | prefill | 9 | `_state_scan_fwd_kernel` | 69 | 10.39 ms | 3.82% | 150.61 µs |
| 1 | prefill | 10 | `_gather_package_cdna4_scale_kernel` | 92 | 5.30 ms | 1.95% | 57.61 µs |
| 1 | decode | 1 | `_latent_input_decode_kernel` | 5,888 | 105.43 ms | 14.18% | 17.91 µs |
| 1 | decode | 2 | `_linear_attnres_partials_kernel` | 5,440 | 92.39 ms | 12.43% | 16.98 µs |
| 1 | decode | 3 | `_warp_decode_precomputed_situ_stage1_kernel` | 5,888 | 72.41 ms | 9.74% | 12.30 µs |
| 1 | decode | 4 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4` | 5,952 | 66.04 ms | 8.88% | 11.10 µs |
| 1 | decode | 5 | `_rowcta_gemv_add3_kernel` | 5,888 | 62.18 ms | 8.36% | 10.56 µs |
| 1 | decode | 6 | `_attnres_combine_kernel` | 11,840 | 48.41 ms | 6.51% | 4.09 µs |
| 1 | decode | 7 | `_warp_decode_stage2_fp8_mxfp4_kernel` | 5,888 | 36.60 ms | 4.92% | 6.22 µs |
| 1 | decode | 8 | `_kda_fused_decode_kernel` | 4,416 | 32.74 ms | 4.40% | 7.41 µs |
| 1 | decode | 9 | `_kimi3_projection_gemv_kernel` | 5,888 | 28.32 ms | 3.81% | 4.81 µs |
| 1 | decode | 10 | `_kimi3_sigmoid_bias_topk_kernel` | 5,888 | 26.01 ms | 3.50% | 4.42 µs |
| 16 | prefill | 1 | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1})` | 1,133 | 2,401.62 ms | 41.07% | 2,119.70 µs |
| 16 | prefill | 2 | `gluon_mxfp4_moe_stage1_kernel` | 736 | 671.51 ms | 11.48% | 912.38 µs |
| 16 | prefill | 3 | `_packed_input_projections_kernel` | 736 | 427.92 ms | 7.32% | 581.42 µs |
| 16 | prefill | 4 | `gluon_mxfp4_moe_stage2_1x2_kernel` | 736 | 387.49 ms | 6.63% | 526.48 µs |
| 16 | prefill | 5 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 552 | 325.95 ms | 5.57% | 590.50 µs |
| 16 | prefill | 6 | `_attn_res_rmsnorm_kernel` | 1,496 | 293.72 ms | 5.02% | 196.34 µs |
| 16 | prefill | 7 | `_mfma_lds_largem_kernel` | 736 | 224.36 ms | 3.84% | 304.83 µs |
| 16 | prefill | 8 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1` | 1,488 | 186.80 ms | 3.19% | 125.54 µs |
| 16 | prefill | 9 | `gluon_mxfp4_moe_stage2_reduce_kernel` | 736 | 163.29 ms | 2.79% | 221.86 µs |
| 16 | prefill | 10 | `_state_scan_fwd_kernel` | 552 | 147.74 ms | 2.53% | 267.64 µs |
| 16 | decode | 1 | `_warp_decode_precomputed_situ_stage1_kernel` | 5,888 | 351.01 ms | 23.34% | 59.61 µs |
| 16 | decode | 2 | `_warp_decode_stage2_fp8_mxfp4_kernel` | 5,888 | 166.28 ms | 11.05% | 28.24 µs |
| 16 | decode | 3 | `_packed_projection_gemm_kernel` | 5,888 | 133.66 ms | 8.89% | 22.70 µs |
| 16 | decode | 4 | `_attn_res_rmsnorm_kernel` | 11,968 | 119.06 ms | 7.92% | 9.95 µs |
| 16 | decode | 5 | `_mfma_lds_mediumm_kernel` | 5,888 | 98.24 ms | 6.53% | 16.68 µs |
| 16 | decode | 6 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4` | 4,416 | 93.15 ms | 6.19% | 21.09 µs |
| 16 | decode | 7 | `_sigmoid_bias_topk_route_prefill_kernel` | 5,888 | 58.88 ms | 3.91% | 10.00 µs |
| 16 | decode | 8 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4` | 6,016 | 57.09 ms | 3.80% | 9.49 µs |
| 16 | decode | 9 | `_dynamic_fp8_quantize_kernel` | 5,888 | 54.84 ms | 3.65% | 9.31 µs |
| 16 | decode | 10 | `_kda_fused_decode_kernel` | 4,416 | 45.74 ms | 3.04% | 10.36 µs |

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
  --warmup-waves 1 --measurement-waves 3 \
  --prompt-seed 7 --synthetic-vocabulary-size 160000 \
  --output result.json
```

### Stage profiles

```bash
TOKENSPEED_KERNEL_PROFILE_BACKEND=roctracer \
python3 toy_e2e/scripts/profile_logical_rank_stages.py \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --expected-arch gfx950 \
  --output-dir /data/results/kimi-k3-toy-1gpu-gfx950-0b1061eb-rolling/eager-profile \
  --tokenspeed-revision 0b1061eb9fe1df36a4e48e5c9c291cd753af9e89 \
  --model-revision eaf5a944bfc8c57438bbce226feef9f6bdbdaae1 \
  --prompt-tokens 4096 --concurrency 1 16 \
  --chunked-prefill-size 8192 --cache-gib 32 \
  --prompt-seed 7 --synthetic-vocabulary-size 160000 \
  --decode-steps 64
```

### Hotspot aggregation

```bash
python3 toy_e2e/scripts/summarize_gpu_hotspots.py \
  --input /data/results/kimi-k3-toy-1gpu-gfx950-0b1061eb-rolling/eager-profile \
  --top-k 15 \
  --csv-dir toy_e2e/results/gfx950_toy_1gpu_0b1061eb/hotspots/csv \
  --output toy_e2e/results/gfx950_toy_1gpu_0b1061eb/hotspots/hotspots.json

python3 toy_e2e/scripts/update_result_readme_hotspots.py \
  --readme toy_e2e/results/gfx950_toy_1gpu_0b1061eb/README.md \
  --hotspots toy_e2e/results/gfx950_toy_1gpu_0b1061eb/hotspots/hotspots.json \
  --csv-dir toy_e2e/results/gfx950_toy_1gpu_0b1061eb/hotspots/csv \
  --profile-manifest /data/results/kimi-k3-toy-1gpu-gfx950-0b1061eb-rolling/eager-profile/profile_manifest.json
```

See [`../../RUNBOOK.md`](../../RUNBOOK.md) for container, environment, and
validation setup.

## Raw artifacts

- Primary result JSON: [`result.json`](result.json)
- Complete run log: [`run.log`](run.log)
- Hotspot summary: [`hotspots/hotspots.json`](hotspots/hotspots.json)
- Exact-name CSVs: [`hotspots/csv/`](hotspots/csv/)
- Raw traces and manifest:
  `/data/results/kimi-k3-toy-1gpu-gfx950-0b1061eb-rolling/eager-profile/`
- Service logs and EvalScope outputs: unavailable for a direct logical-rank run

## Conclusions and limitations

- Complete rolling output reaches 78.12 tok/s at C1 and 586.09 tok/s at C16;
  post-transition steady decode capacity is 79.78 and 668.47 tok/s.
- MoE is the largest C1 prefill category at 45.99%. At C16, a generic
  direct-copy kernel dominates the varied-token rank-local path, placing
  46.16% in `Other` and 22.02% in MoE.
- GEMM/quant is the largest decode category: 30.56% at C1 and 36.36% at C16.
- Communication is intentionally absent. This result cannot measure RCCL,
  cross-rank synchronization, HTTP, or tokenization. Rank-local greedy outputs
  and MoE routes are not semantically valid full-TP model outputs.
- FP8 KV scaling factors were unavailable and defaulted to 1.0. This is a
  performance/scheduling result, not a model-quality qualification.

Profiled GPU-duration sums are hotspot weights, not critical-path latency.
Profiler instrumentation perturbs execution and must not replace the
unprofiled graph measurements.
