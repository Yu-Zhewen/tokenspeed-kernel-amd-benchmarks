# Kimi-K3 TP8/EP1 logical rank 0: MI355X vs MI455X at `0a290207`

One physical GPU per architecture executes rank 0 of the TP8 model with
local substitutes for rank-spanning collectives. Both architectures ran
the same TokenSpeed commit, the same workload, and the same harness.

## Provenance

| Field | Value |
|---|---|
| TokenSpeed revision | `0a29020790de2e8b8be67335fa95c062c058aecd` |
| Commit date (UTC) | 2026-09-28 |
| Measured | 2026-09-28 |
| Model revision | `eaf5a944bfc8c57438bbce226feef9f6bdbdaae1` |
| Model | KimiK3ForConditionalGeneration, 93 layers, attn tp8 moe tp8 ep1 |
| Prompt / output tokens | 50,000 / 1,024 |
| Concurrencies | 1, 16 |
| Prefill budget | 8,192 tokens |
| Warmup / measured waves | 1 / 3 |
| MI355X | AMD Radeon Graphics, `gfx950:sramecc+:xnack-` |
| MI455X | AMD Radeon Graphics, `gfx1250` |
| Container images | MI355X `zhewenyu/kimi-k3-e2e:tokenspeed-ffa16b13`, MI455X `tokenspeed-kimi-gfx1250:tokenspeed-ffa16b13-torch213` |
| ROCm / PyTorch / Triton | 7.2.53211 / 2.13.0+rocm7.2 / 3.7.1 |

## End-to-end performance

Latency rows are lower-is-better and throughput rows higher-is-better;
the last column is always MI455X's advantage, so above 1.00x favours
MI455X either way.

| Metric | Batch | MI355X (gfx950) | MI455X (gfx1250) | MI455X advantage |
|---|---:|---:|---:|---:|
| prefill TTFT p50 (ms) | 1 | 2,783.5 | 2,003.9 | 1.39x |
| prefill step p50 (ms) | 1 | 830.6 | 581.6 | 1.43x |
| decode step p50 (ms) | 1 | 12.0 | 9.2 | 1.30x |
| decode tok/s per user | 1 | 83.2 | 108.9 | 1.31x |
| aggregate output tok/s | 1 | 67.9 | 89.8 | 1.32x |
| prefill TTFT p50 (ms) | 16 | 22,671.0 | 16,119.3 | 1.41x |
| prefill step p50 (ms) | 16 | 865.8 | 613.2 | 1.41x |
| decode step p50 (ms) | 16 | 23.8 | 16.0 | 1.49x |
| decode tok/s per user | 16 | 23.1 | 33.5 | 1.45x |
| aggregate output tok/s | 16 | 244.7 | 351.4 | 1.44x |

## Where the time goes

The final row is the measured step latency from the end-to-end table above, not a total of the rows, so the buckets can be read against what the machine actually reported.

Kernels are bucketed by function because the two architectures do not
split the work into the same kernels. Rows follow that category list,
the same order in every entry. Ratios are accumulated GPU kernel
duration, MI355X over MI455X, so above 1.00x means MI455X is ahead.

| Category | prefill c16 | prefill c1 | decode c16 | decode c1 |
|---|---:|---:|---:|---:|
| KDA state scan | 1.40x | 1.37x | — | — |
| MoE | 1.73x | 1.75x | 1.69x | 1.08x |
| dense GEMM | 1.20x | 1.19x | 1.26x | 1.56x |
| input projections | 1.25x | 1.25x | 1.95x | 1.74x |
| MLA attention | 1.09x | 1.21x | 1.53x | 1.94x |
| AttnRes | 1.36x | 1.36x | 1.49x | 1.25x |
| KDA other | 1.38x | 1.38x | 1.40x | 1.15x |
| add3 | 2.72x | 2.74x | — | — |
| rmsnorm | 1.68x | 1.65x | 1.22x | 1.00x |
| elementwise | 1.67x | 1.64x | 1.92x | 1.19x |
| other | 2.17x | 2.30x | 0.75x | 0.72x |
| **End-to-end (step p50)** | **1.41x** | **1.43x** | **1.49x** | **1.30x** |

## Heaviest kernels

Percent is the share of accumulated GPU kernel duration within the stage,
not wall time, and kernels may overlap. The heaviest
20 symbols appear below; the CSVs under
`kernel-tables/` keep every symbol.

### MI355X (gfx950) prefill c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel.kd` | 5,486.357 | 9,016 | 608.5 | 13.12% |
| 2 | input projections | `gluon_latent_input_largem_gfx950.kd` | 4,188.457 | 9,016 | 464.6 | 10.02% |
| 3 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel.kd` | 4,144.627 | 9,016 | 459.7 | 9.91% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 4,087.857 | 7,003 | 583.7 | 9.78% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 3,379.927 | 18,326 | 184.4 | 8.09% |
| 6 | MLA attention | `gluon_mla_prefill_8wave_gfx950.kd` | 2,926.150 | 5,784 | 505.9 | 7.00% |
| 7 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950.kd` | 2,454.205 | 7,866 | 312.0 | 5.87% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 2,407.915 | 16,833 | 143.0 | 5.76% |
| 9 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950.kd` | 2,339.041 | 7,640 | 306.2 | 5.60% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 1,950.199 | 9,016 | 216.3 | 4.67% |
| 11 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx950.kd` | 939.562 | 7,866 | 119.4 | 2.25% |
| 12 | add3 | `_add3_kernel.kd` | 759.518 | 9,016 | 84.2 | 1.82% |
| 13 | MoE | `_gather_package_cdna4_scale_kernel.kd` | 539.895 | 9,016 | 59.9 | 1.29% |
| 14 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x192x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 500.772 | 2,589 | 193.4 | 1.20% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x256x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 489.808 | 10,362 | 47.3 | 1.17% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 461.610 | 6,762 | 68.3 | 1.10% |
| 17 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB2_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_15_MO40_NTn1_NTA0_NTB0_NTC6_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO1_SRVW0_SSO1_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 448.559 | 3,275 | 137.0 | 1.07% |
| 18 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx950.kd` | 369.464 | 7,866 | 47.0 | 0.88% |
| 19 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 351.740 | 9,016 | 39.0 | 0.84% |
| 20 | KDA other | `gluon_kda_paged_prefill_solve_merge_gfx950.kd` | 253.366 | 7,866 | 32.2 | 0.61% |

### MI355X (gfx950) prefill c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel.kd` | 371.124 | 644 | 576.3 | 13.84% |
| 2 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel.kd` | 271.447 | 644 | 421.5 | 10.12% |
| 3 | input projections | `gluon_latent_input_largem_gfx950.kd` | 259.024 | 552 | 469.2 | 9.66% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 248.079 | 414 | 599.2 | 9.25% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 211.974 | 1,309 | 161.9 | 7.91% |
| 6 | MLA attention | `gluon_mla_prefill_8wave_gfx950.kd` | 187.053 | 336 | 556.7 | 6.98% |
| 7 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950.kd` | 171.469 | 552 | 310.6 | 6.40% |
| 8 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950.kd` | 156.130 | 552 | 282.8 | 5.82% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 139.858 | 1,116 | 125.3 | 5.22% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 129.154 | 644 | 200.6 | 4.82% |
| 11 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx950.kd` | 59.916 | 552 | 108.5 | 2.23% |
| 12 | add3 | `_add3_kernel.kd` | 47.808 | 644 | 74.2 | 1.78% |
| 13 | MoE | `_gather_package_cdna4_scale_kernel.kd` | 36.953 | 644 | 57.4 | 1.38% |
| 14 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x256x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 32.536 | 750 | 43.4 | 1.21% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x192x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 30.264 | 144 | 210.2 | 1.13% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 28.977 | 483 | 60.0 | 1.08% |
| 17 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx950.kd` | 23.350 | 552 | 42.3 | 0.87% |
| 18 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 22.658 | 644 | 35.2 | 0.85% |
| 19 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 17.061 | 1,203 | 14.2 | 0.64% |
| 20 | KDA other | `gluon_kda_paged_prefill_solve_merge_gfx950.kd` | 16.885 | 552 | 30.6 | 0.63% |

### MI355X (gfx950) decode c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_warp_decode_precomputed_situ_stage1_kernel.kd` | 347.835 | 5,888 | 59.1 | 21.75% |
| 2 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel.kd` | 166.204 | 5,888 | 28.2 | 10.39% |
| 3 | MLA attention | `_mla_decode_gluon.kd` | 138.185 | 1,536 | 90.0 | 8.64% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 117.490 | 11,968 | 9.8 | 7.35% |
| 5 | input projections | `gluon_latent_input_small_batch_gfx950.kd` | 116.853 | 5,888 | 19.8 | 7.31% |
| 6 | dense GEMM | `gluon_mm_a16w16_medium_gfx950.kd` | 99.317 | 5,888 | 16.9 | 6.21% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 91.776 | 4,416 | 20.8 | 5.74% |
| 8 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 58.727 | 5,888 | 10.0 | 3.67% |
| 9 | MoE | `_dynamic_fp8_single_pass_kernel.kd` | 55.315 | 5,888 | 9.4 | 3.46% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 54.399 | 6,016 | 9.0 | 3.40% |
| 11 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx950.kd` | 46.499 | 4,416 | 10.5 | 2.91% |
| 12 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x256_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB512_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 42.386 | 5,888 | 7.2 | 2.65% |
| 13 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 40.227 | 9,024 | 4.5 | 2.52% |
| 14 | MoE | `_moe_partial_reduce.kd` | 27.155 | 5,888 | 4.6 | 1.70% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 26.874 | 1,600 | 16.8 | 1.68% |
| 16 | elementwise | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 25.842 | 5,888 | 4.4 | 1.62% |
| 17 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 23.822 | 4,416 | 5.4 | 1.49% |
| 18 | input projections | `gluon_latent_input_small_batch_epilogue_gfx950.kd` | 22.498 | 5,888 | 3.8 | 1.41% |
| 19 | rmsnorm | `_rmsnorm_kernel.kd` | 20.027 | 5,888 | 3.4 | 1.25% |
| 20 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB256_LBSPPM0_LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 15.900 | 1,536 | 10.4 | 0.99% |

### MI355X (gfx950) decode c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | input projections | `gluon_latent_input_decode_gfx950.kd` | 107.583 | 5,888 | 18.3 | 13.03% |
| 2 | AttnRes | `gluon_linear_attnres_partials_gfx950.kd` | 97.475 | 5,440 | 17.9 | 11.81% |
| 3 | MLA attention | `_mla_decode_gluon.kd` | 94.619 | 1,536 | 61.6 | 11.46% |
| 4 | MoE | `_warp_decode_precomputed_situ_stage1_kernel.kd` | 71.941 | 5,888 | 12.2 | 8.71% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 65.767 | 5,952 | 11.0 | 7.97% |
| 6 | dense GEMM | `_rowcta_gemv_add3_kernel.kd` | 60.879 | 5,888 | 10.3 | 7.37% |
| 7 | AttnRes | `_attnres_combine_kernel.kd` | 46.898 | 11,840 | 4.0 | 5.68% |
| 8 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel.kd` | 35.889 | 5,888 | 6.1 | 4.35% |
| 9 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx950.kd` | 34.229 | 4,416 | 7.8 | 4.15% |
| 10 | dense GEMM | `_kimi3_projection_gemv_kernel.kd` | 29.455 | 5,888 | 5.0 | 3.57% |
| 11 | MoE | `_decode_sigmoid_bias_topk_kernel.kd` | 23.138 | 5,888 | 3.9 | 2.80% |
| 12 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 21.616 | 4,416 | 4.9 | 2.62% |
| 13 | MoE | `_moe_partial_reduce.kd` | 20.011 | 5,888 | 3.4 | 2.42% |
| 14 | elementwise | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>) [clone .kd]` | 19.369 | 5,504 | 3.5 | 2.35% |
| 15 | MLA attention | `_mla_reduce_project_value_kernel.kd` | 14.430 | 1,536 | 9.4 | 1.75% |
| 16 | rmsnorm | `_rmsnorm_kernel.kd` | 14.116 | 5,952 | 2.4 | 1.71% |
| 17 | MoE | `_dynamic_fp8_single_pass_kernel.kd` | 12.223 | 5,888 | 2.1 | 1.48% |
| 18 | other | `gluon_mla_normalize_project_query_gfx950.kd` | 10.714 | 1,536 | 7.0 | 1.30% |
| 19 | other | `_mla_rope_set_kv_buffer_kernel.kd` | 9.903 | 1,536 | 6.4 | 1.20% |
| 20 | dense GEMM | `gluon_bmm_a16w16_gfx950.kd` | 7.943 | 1,536 | 5.2 | 0.96% |

### MI455X (gfx1250) prefill c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel.kd` | 8,593.017 | 46,944 | 183.0 | 28.95% |
| 2 | MoE | `_matmul.kd` | 5,611.535 | 18,032 | 311.2 | 18.91% |
| 3 | input projections | `gluon_latent_input_largem_gfx1250.kd` | 3,335.722 | 9,016 | 370.0 | 11.24% |
| 4 | MLA attention | `gluon_mla_prefill_gfx1250.kd` | 2,752.780 | 5,856 | 470.1 | 9.27% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 2,480.748 | 18,326 | 135.4 | 8.36% |
| 6 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250.kd` | 1,752.516 | 7,866 | 222.8 | 5.90% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250.kd` | 872.893 | 7,866 | 111.0 | 2.94% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 576.012 | 9,016 | 63.9 | 1.94% |
| 9 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel.kd` | 312.260 | 9,016 | 34.6 | 1.05% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 295.671 | 8,036 | 36.8 | 1.00% |
| 11 | add3 | `_add3_kernel.kd` | 279.400 | 9,016 | 31.0 | 0.94% |
| 12 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx1250.kd` | 260.585 | 7,866 | 33.1 | 0.88% |
| 13 | MoE | `gluon_sigmoid_bias_topk_gfx1250.kd` | 237.746 | 9,016 | 26.4 | 0.80% |
| 14 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 222.654 | 6,762 | 32.9 | 0.75% |
| 15 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 214.855 | 14,925 | 14.4 | 0.72% |
| 16 | MoE | `_precomputed_topk_route_large_stage4_gfx1250_kernel.kd` | 209.902 | 9,016 | 23.3 | 0.71% |
| 17 | MoE | `_precomputed_topk_route_large_stage1_gfx1250_kernel.kd` | 145.007 | 9,016 | 16.1 | 0.49% |
| 18 | other | `gluon_kda_paged_prefill_gfx1250.kd` | 119.060 | 7,866 | 15.1 | 0.40% |
| 19 | other | `attn_merge_state_kernel.kd` | 116.780 | 3,504 | 33.3 | 0.39% |
| 20 | other | `_mla_nope_quantize_fp8_kernel.kd` | 111.109 | 2,352 | 47.2 | 0.37% |

### MI455X (gfx1250) prefill c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel.kd` | 543.246 | 3,300 | 164.6 | 28.87% |
| 2 | MoE | `_matmul.kd` | 345.176 | 1,104 | 312.7 | 18.34% |
| 3 | input projections | `gluon_latent_input_largem_gfx1250.kd` | 204.073 | 552 | 369.7 | 10.84% |
| 4 | MLA attention | `gluon_mla_prefill_gfx1250.kd` | 158.277 | 360 | 439.7 | 8.41% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 155.572 | 1,309 | 118.8 | 8.27% |
| 6 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250.kd` | 113.927 | 552 | 206.4 | 6.05% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250.kd` | 56.017 | 552 | 101.5 | 2.98% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 37.197 | 644 | 57.8 | 1.98% |
| 9 | MoE | `_matmul_decode.kd` | 23.712 | 184 | 128.9 | 1.26% |
| 10 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel.kd` | 22.213 | 644 | 34.5 | 1.18% |
| 11 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 19.233 | 559 | 34.4 | 1.02% |
| 12 | add3 | `_add3_kernel.kd` | 17.465 | 644 | 27.1 | 0.93% |
| 13 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx1250.kd` | 16.019 | 552 | 29.0 | 0.85% |
| 14 | MoE | `gluon_sigmoid_bias_topk_gfx1250.kd` | 15.376 | 644 | 23.9 | 0.82% |
| 15 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 14.282 | 483 | 29.6 | 0.76% |
| 16 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 14.234 | 1,203 | 11.8 | 0.76% |
| 17 | MoE | `_precomputed_topk_route_large_stage4_gfx1250_kernel.kd` | 13.828 | 644 | 21.5 | 0.73% |
| 18 | MoE | `_precomputed_topk_route_large_stage1_gfx1250_kernel.kd` | 10.361 | 644 | 16.1 | 0.55% |
| 19 | input projections | `_packed_input_projections_kernel.kd` | 8.280 | 92 | 90.0 | 0.44% |
| 20 | other | `gluon_kda_paged_prefill_gfx1250.kd` | 7.535 | 552 | 13.7 | 0.40% |

### MI455X (gfx1250) decode c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode.kd` | 281.927 | 11,776 | 23.9 | 26.88% |
| 2 | dense GEMM | `_wmma_tdm_dense_m16_kernel.kd` | 163.373 | 19,456 | 8.4 | 15.57% |
| 3 | dense GEMM | `_wmma_tdm_add3_m16_kernel.kd` | 91.039 | 5,888 | 15.5 | 8.68% |
| 4 | MLA attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_HEAD_BLOCKS_1_NUM_KV_SPLITS_32_num_warps_2_num_stages_2.kd` | 85.985 | 1,536 | 56.0 | 8.20% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 78.907 | 11,968 | 6.6 | 7.52% |
| 6 | MoE | `_precomputed_topk_route_small_m_gfx1250_kernel.kd` | 58.167 | 5,888 | 9.9 | 5.55% |
| 7 | input projections | `gluon_latent_input_decode_gfx1250.kd` | 54.939 | 5,888 | 9.3 | 5.24% |
| 8 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 33.874 | 9,024 | 3.8 | 3.23% |
| 9 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx1250.kd` | 33.148 | 4,416 | 7.5 | 3.16% |
| 10 | MoE | `_kimi3_sigmoid_bias_topk_kernel.kd` | 27.673 | 5,888 | 4.7 | 2.64% |
| 11 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 23.942 | 3,072 | 7.8 | 2.28% |
| 12 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 23.538 | 4,480 | 5.3 | 2.24% |
| 13 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 17.435 | 5,888 | 3.0 | 1.66% |
| 14 | input projections | `gluon_latent_input_decode_epilogue_gfx1250.kd` | 16.593 | 5,888 | 2.8 | 1.58% |
| 15 | rmsnorm | `_rmsnorm_kernel.kd` | 16.550 | 5,888 | 2.8 | 1.58% |
| 16 | other | `_fp8_quantize_kernel.kd` | 13.378 | 5,888 | 2.3 | 1.28% |
| 17 | rmsnorm | `_rmsnorm_fused_parallel_kernel.kd` | 5.182 | 1,536 | 3.4 | 0.49% |
| 18 | MLA attention | `_mla_decode_fwd_reduce_kernel_num_query_heads_12_TILE_SIZE_64_KV_LORA_RANK_512_NUM_KV_SPLITS_32_ALL_DECODE_1_HAS_LSE_0_num_warps_4.kd` | 4.408 | 1,536 | 2.9 | 0.42% |
| 19 | MoE | `_sigmoid_mul_kernel.kd` | 4.400 | 1,536 | 2.9 | 0.42% |
| 20 | other | `_mla_rope_set_kv_buffer_kernel.kd` | 4.364 | 1,536 | 2.8 | 0.42% |

### MI455X (gfx1250) decode c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode.kd` | 91.848 | 11,776 | 7.8 | 14.94% |
| 2 | AttnRes | `gluon_linear_attnres_partials_gfx1250.kd` | 79.681 | 5,440 | 14.6 | 12.96% |
| 3 | dense GEMM | `_rowcta_gemv_kernel.kd` | 56.494 | 12,480 | 4.5 | 9.19% |
| 4 | input projections | `gluon_latent_input_decode_gfx1250.kd` | 46.834 | 5,888 | 8.0 | 7.62% |
| 5 | dense GEMM | `_rowcta_gemv_add3_kernel.kd` | 41.081 | 5,888 | 7.0 | 6.68% |
| 6 | MLA attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_HEAD_BLOCKS_1_NUM_KV_SPLITS_64_num_warps_2_num_stages_2.kd` | 39.615 | 1,536 | 25.8 | 6.44% |
| 7 | AttnRes | `_attnres_combine_kernel.kd` | 34.818 | 11,840 | 2.9 | 5.66% |
| 8 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx1250.kd` | 29.842 | 4,416 | 6.8 | 4.85% |
| 9 | MoE | `_decode_sigmoid_bias_topk_kernel.kd` | 23.795 | 5,888 | 4.0 | 3.87% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 21.717 | 4,480 | 4.8 | 3.53% |
| 11 | MoE | `_precomputed_topk_route_m1_canonical_gfx1250_kernel.kd` | 18.502 | 5,888 | 3.1 | 3.01% |
| 12 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 17.609 | 5,888 | 3.0 | 2.86% |
| 13 | elementwise | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>) [clone .kd]` | 17.401 | 5,504 | 3.2 | 2.83% |
| 14 | MLA attention | `_mla_reduce_project_value_kernel.kd` | 16.651 | 1,536 | 10.8 | 2.71% |
| 15 | other | `gluon_mla_normalize_project_query_gfx1250.kd` | 15.669 | 1,536 | 10.2 | 2.55% |
| 16 | input projections | `gluon_latent_input_decode_epilogue_gfx1250.kd` | 15.029 | 5,888 | 2.6 | 2.44% |
| 17 | rmsnorm | `_rmsnorm_kernel.kd` | 14.063 | 5,952 | 2.4 | 2.29% |
| 18 | other | `_fp8_quantize_kernel.kd` | 9.758 | 5,888 | 1.7 | 1.59% |
| 19 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 7.841 | 1,536 | 5.1 | 1.28% |
| 20 | AttnRes | `_attnres_partial_kernel.kd` | 6.868 | 960 | 7.2 | 1.12% |

## Method and limitations

- Timing and profiling are separate runs. The performance tables come
  from an unprofiled run; the kernel tables come from a profiled run at
  the same commit and workload.
- Rank-local execution omits physical collectives, HTTP serving, and
  valid full-TP MoE routing, so absolute numbers are a compute estimate
  rather than serving performance.
- A `—` in the bucket tables means one architecture has no kernel in that
  bucket for that stage, so no ratio exists. That is a difference in how
  the work is split, not a measurement of zero.
- Dtype columns are omitted. The profiler symbols alone do not prove
  operand, accumulation, and output precision for most of these kernels.

Regenerate this document with:

```bash
python3 toy_e2e/scripts/generate_arch_comparison.py \
  --revision 0a29020790de2e8b8be67335fa95c062c058aecd \
  --commit-date 2026-09-28 \
  --gfx950-performance <gfx950 performance result.json> \
  --gfx950-hotspots <gfx950 hotspots.json> \
  --gfx1250-performance <gfx1250 performance result.json> \
  --gfx1250-hotspots <gfx1250 hotspots.json> \
  --output-dir toy_e2e/results/arch_compare_20260928_0a290207
```
