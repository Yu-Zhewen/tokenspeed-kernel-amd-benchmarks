# Kimi-K3 TP8/EP1 logical rank 0: MI355X vs MI455X at `2a8dbc6b`

One physical GPU per architecture executes rank 0 of the TP8 model with
local substitutes for rank-spanning collectives. Both architectures ran
the same TokenSpeed commit, the same workload, and the same harness.

## Provenance

| Field | Value |
|---|---|
| TokenSpeed revision | `2a8dbc6b4c809823543da4cfc4dea70ec3ba2fcc` |
| Commit date (UTC) | 2026-09-30 |
| Measured | 2026-09-30 |
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
| prefill TTFT p50 (ms) | 1 | 2,743.3 | 1,890.5 | 1.45x |
| prefill step p50 (ms) | 1 | 827.1 | 548.7 | 1.51x |
| decode step p50 (ms) | 1 | 11.9 | 8.6 | 1.39x |
| decode tok/s per user | 1 | 83.5 | 115.9 | 1.39x |
| aggregate output tok/s | 1 | 68.3 | 95.6 | 1.40x |
| prefill TTFT p50 (ms) | 16 | 22,588.8 | 15,140.4 | 1.49x |
| prefill step p50 (ms) | 16 | 863.7 | 574.3 | 1.50x |
| decode step p50 (ms) | 16 | 23.8 | 14.1 | 1.69x |
| decode tok/s per user | 16 | 23.1 | 36.9 | 1.60x |
| aggregate output tok/s | 16 | 245.1 | 382.1 | 1.56x |

## Where the time goes

The final row is the measured step latency from the end-to-end table above, not a total of the rows, so the buckets can be read against what the machine actually reported.

Kernels are bucketed by function because the two architectures do not
split the work into the same kernels. Rows follow that category list,
the same order in every entry. Ratios are accumulated GPU kernel
duration, MI355X over MI455X, so above 1.00x means MI455X is ahead.

| Category | prefill c16 | prefill c1 | decode c16 | decode c1 |
|---|---:|---:|---:|---:|
| KDA state scan | 1.36x | 1.33x | — | — |
| MoE | 1.67x | 1.63x | 1.67x | 1.10x |
| dense GEMM | 1.26x | 1.24x | 1.82x | 1.60x |
| input projections | 1.28x | 1.27x | 2.02x | 1.74x |
| MLA attention | 1.94x | 2.07x | 1.86x | 2.75x |
| AttnRes | 1.40x | 1.40x | 1.48x | 1.29x |
| KDA other | 1.37x | 1.38x | 1.45x | 1.08x |
| add3 | 2.74x | 2.75x | — | — |
| rmsnorm | 1.68x | 1.66x | 1.20x | 1.01x |
| elementwise | 1.67x | 1.59x | 1.89x | 1.17x |
| other | 2.20x | 2.32x | 0.82x | 0.77x |
| **End-to-end (step p50)** | **1.50x** | **1.51x** | **1.69x** | **1.39x** |

## Heaviest kernels

Percent is the share of accumulated GPU kernel duration within the stage,
not wall time, and kernels may overlap. The heaviest
20 symbols appear below; the CSVs under
`kernel-tables/` keep every symbol.

### MI355X (gfx950) prefill c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel.kd` | 5,481.559 | 9,016 | 608.0 | 13.20% |
| 2 | input projections | `gluon_latent_input_largem_gfx950.kd` | 4,201.791 | 9,016 | 466.0 | 10.12% |
| 3 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 4,134.107 | 7,003 | 590.3 | 9.95% |
| 4 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel.kd` | 4,114.207 | 9,016 | 456.3 | 9.91% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 3,379.401 | 18,326 | 184.4 | 8.14% |
| 6 | MLA attention | `gluon_mla_prefill_8wave_gfx950.kd` | 2,966.970 | 5,784 | 513.0 | 7.14% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 2,411.417 | 16,833 | 143.3 | 5.81% |
| 8 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950.kd` | 2,406.079 | 7,866 | 305.9 | 5.79% |
| 9 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950.kd` | 2,351.396 | 7,640 | 307.8 | 5.66% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 1,569.668 | 9,016 | 174.1 | 3.78% |
| 11 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx950.kd` | 956.413 | 7,866 | 121.6 | 2.30% |
| 12 | add3 | `_add3_kernel.kd` | 754.307 | 9,016 | 83.7 | 1.82% |
| 13 | MoE | `_gather_package_cdna4_scale_kernel.kd` | 540.617 | 9,016 | 60.0 | 1.30% |
| 14 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x192x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 503.340 | 2,589 | 194.4 | 1.21% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x256x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 495.165 | 10,362 | 47.8 | 1.19% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 468.712 | 6,762 | 69.3 | 1.13% |
| 17 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB2_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_15_MO40_NTn1_NTA0_NTB0_NTC6_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO1_SRVW0_SSO1_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 453.690 | 3,275 | 138.5 | 1.09% |
| 18 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx950.kd` | 372.387 | 7,866 | 47.3 | 0.90% |
| 19 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 351.556 | 9,016 | 39.0 | 0.85% |
| 20 | KDA other | `gluon_kda_paged_prefill_solve_merge_gfx950.kd` | 255.886 | 7,866 | 32.5 | 0.62% |

### MI355X (gfx950) prefill c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel.kd` | 356.752 | 644 | 554.0 | 13.58% |
| 2 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel.kd` | 262.700 | 644 | 407.9 | 10.00% |
| 3 | input projections | `gluon_latent_input_largem_gfx950.kd` | 259.721 | 552 | 470.5 | 9.89% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 249.871 | 414 | 603.6 | 9.51% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 211.936 | 1,309 | 161.9 | 8.07% |
| 6 | MLA attention | `gluon_mla_prefill_8wave_gfx950.kd` | 188.249 | 336 | 560.3 | 7.17% |
| 7 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950.kd` | 171.717 | 552 | 311.1 | 6.54% |
| 8 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950.kd` | 153.338 | 552 | 277.8 | 5.84% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 139.953 | 1,116 | 125.4 | 5.33% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 96.491 | 552 | 174.8 | 3.67% |
| 11 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx950.kd` | 60.775 | 552 | 110.1 | 2.31% |
| 12 | add3 | `_add3_kernel.kd` | 47.479 | 644 | 73.7 | 1.81% |
| 13 | MoE | `_gather_package_cdna4_scale_kernel.kd` | 34.640 | 644 | 53.8 | 1.32% |
| 14 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x256x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 32.732 | 750 | 43.6 | 1.25% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x192x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 30.287 | 144 | 210.3 | 1.15% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 29.372 | 483 | 60.8 | 1.12% |
| 17 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx950.kd` | 23.578 | 552 | 42.7 | 0.90% |
| 18 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 22.560 | 644 | 35.0 | 0.86% |
| 19 | KDA other | `gluon_kda_paged_prefill_solve_merge_gfx950.kd` | 17.028 | 552 | 30.8 | 0.65% |
| 20 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 16.789 | 1,203 | 14.0 | 0.64% |

### MI355X (gfx950) decode c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_warp_decode_precomputed_situ_stage1_kernel.kd` | 343.659 | 5,888 | 58.4 | 21.55% |
| 2 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel.kd` | 164.865 | 5,888 | 28.0 | 10.34% |
| 3 | MLA attention | `_mla_decode_gluon.kd` | 142.080 | 1,536 | 92.5 | 8.91% |
| 4 | input projections | `gluon_latent_input_small_batch_gfx950.kd` | 119.092 | 5,888 | 20.2 | 7.47% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 116.220 | 11,968 | 9.7 | 7.29% |
| 6 | dense GEMM | `gluon_mm_a16w16_medium_gfx950.kd` | 96.423 | 5,888 | 16.4 | 6.05% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 91.882 | 4,416 | 20.8 | 5.76% |
| 8 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 60.073 | 5,888 | 10.2 | 3.77% |
| 9 | MoE | `_dynamic_fp8_single_pass_kernel.kd` | 54.939 | 5,888 | 9.3 | 3.45% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 54.918 | 6,016 | 9.1 | 3.44% |
| 11 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx950.kd` | 46.302 | 4,416 | 10.5 | 2.90% |
| 12 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x256_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB512_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 42.298 | 5,888 | 7.2 | 2.65% |
| 13 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 41.299 | 9,024 | 4.6 | 2.59% |
| 14 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 27.075 | 5,888 | 4.6 | 1.70% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 26.751 | 1,600 | 16.7 | 1.68% |
| 16 | elementwise | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 25.456 | 5,888 | 4.3 | 1.60% |
| 17 | input projections | `gluon_latent_input_small_batch_epilogue_gfx950.kd` | 22.891 | 5,888 | 3.9 | 1.44% |
| 18 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 22.399 | 4,416 | 5.1 | 1.40% |
| 19 | rmsnorm | `_rmsnorm_kernel.kd` | 19.946 | 5,888 | 3.4 | 1.25% |
| 20 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB256_LBSPPM0_LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 15.997 | 1,536 | 10.4 | 1.00% |

### MI355X (gfx950) decode c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | input projections | `gluon_latent_input_decode_gfx950.kd` | 107.086 | 5,888 | 18.2 | 13.02% |
| 2 | AttnRes | `gluon_linear_attnres_partials_gfx950.kd` | 97.079 | 5,440 | 17.8 | 11.81% |
| 3 | MLA attention | `_mla_decode_gluon.kd` | 94.360 | 1,536 | 61.4 | 11.48% |
| 4 | MoE | `_warp_decode_precomputed_situ_stage1_kernel.kd` | 70.468 | 5,888 | 12.0 | 8.57% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 66.001 | 5,952 | 11.1 | 8.03% |
| 6 | dense GEMM | `_rowcta_gemv_add3_kernel.kd` | 62.044 | 5,888 | 10.5 | 7.55% |
| 7 | AttnRes | `_attnres_combine_kernel.kd` | 45.724 | 11,840 | 3.9 | 5.56% |
| 8 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel.kd` | 36.162 | 5,888 | 6.1 | 4.40% |
| 9 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx950.kd` | 31.908 | 4,416 | 7.2 | 3.88% |
| 10 | dense GEMM | `_kimi3_projection_gemv_kernel.kd` | 29.598 | 5,888 | 5.0 | 3.60% |
| 11 | MoE | `_decode_sigmoid_bias_topk_kernel.kd` | 23.781 | 5,888 | 4.0 | 2.89% |
| 12 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 21.640 | 4,416 | 4.9 | 2.63% |
| 13 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 19.637 | 5,888 | 3.3 | 2.39% |
| 14 | elementwise | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>) [clone .kd]` | 18.944 | 5,504 | 3.4 | 2.30% |
| 15 | MLA attention | `_mla_reduce_project_value_kernel.kd` | 14.528 | 1,536 | 9.5 | 1.77% |
| 16 | rmsnorm | `_rmsnorm_kernel.kd` | 14.396 | 5,952 | 2.4 | 1.75% |
| 17 | MoE | `_dynamic_fp8_single_pass_kernel.kd` | 11.863 | 5,888 | 2.0 | 1.44% |
| 18 | other | `gluon_mla_normalize_project_query_gfx950.kd` | 10.885 | 1,536 | 7.1 | 1.32% |
| 19 | other | `_mla_rope_set_kv_buffer_kernel.kd` | 10.734 | 1,536 | 7.0 | 1.31% |
| 20 | dense GEMM | `gluon_bmm_a16w16_gfx950.kd` | 7.899 | 1,536 | 5.1 | 0.96% |

### MI455X (gfx1250) prefill c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel.kd` | 8,235.281 | 46,944 | 175.4 | 29.36% |
| 2 | MoE | `_matmul.kd` | 5,613.585 | 18,032 | 311.3 | 20.01% |
| 3 | input projections | `gluon_latent_input_largem_gfx1250.kd` | 3,268.983 | 9,016 | 362.6 | 11.65% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 2,410.864 | 18,326 | 131.6 | 8.59% |
| 5 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250.kd` | 1,770.671 | 7,866 | 225.1 | 6.31% |
| 6 | MLA attention | `gluon_mla_prefill_gfx1250.kd` | 1,536.817 | 5,856 | 262.4 | 5.48% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250.kd` | 894.002 | 7,866 | 113.7 | 3.19% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 569.876 | 9,016 | 63.2 | 2.03% |
| 9 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel.kd` | 330.694 | 9,016 | 36.7 | 1.18% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 298.789 | 8,036 | 37.2 | 1.07% |
| 11 | add3 | `_add3_kernel.kd` | 274.949 | 9,016 | 30.5 | 0.98% |
| 12 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx1250.kd` | 264.551 | 7,866 | 33.6 | 0.94% |
| 13 | MoE | `gluon_sigmoid_bias_topk_gfx1250.kd` | 241.574 | 9,016 | 26.8 | 0.86% |
| 14 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 224.449 | 6,762 | 33.2 | 0.80% |
| 15 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 219.533 | 14,925 | 14.7 | 0.78% |
| 16 | MoE | `_precomputed_topk_route_large_stage4_gfx1250_kernel.kd` | 214.033 | 9,016 | 23.7 | 0.76% |
| 17 | MoE | `_precomputed_topk_route_large_stage1_gfx1250_kernel.kd` | 147.433 | 9,016 | 16.4 | 0.53% |
| 18 | other | `gluon_kda_paged_prefill_gfx1250.kd` | 120.595 | 7,866 | 15.3 | 0.43% |
| 19 | other | `attn_merge_state_kernel.kd` | 115.767 | 3,504 | 33.0 | 0.41% |
| 20 | KDA other | `gluon_kda_paged_prefill_solve_merge_gfx1250.kd` | 112.980 | 7,866 | 14.4 | 0.40% |

### MI455X (gfx1250) prefill c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel.kd` | 521.497 | 3,300 | 158.0 | 29.12% |
| 2 | MoE | `_matmul.kd` | 344.981 | 1,104 | 312.5 | 19.27% |
| 3 | input projections | `gluon_latent_input_largem_gfx1250.kd` | 201.237 | 552 | 364.6 | 11.24% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 151.461 | 1,309 | 115.7 | 8.46% |
| 5 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250.kd` | 115.014 | 552 | 208.4 | 6.42% |
| 6 | MLA attention | `gluon_mla_prefill_gfx1250.kd` | 91.669 | 360 | 254.6 | 5.12% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250.kd` | 56.114 | 552 | 101.7 | 3.13% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 36.174 | 644 | 56.2 | 2.02% |
| 9 | MoE | `_matmul_decode.kd` | 23.895 | 184 | 129.9 | 1.33% |
| 10 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel.kd` | 23.389 | 644 | 36.3 | 1.31% |
| 11 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 19.350 | 559 | 34.6 | 1.08% |
| 12 | add3 | `_add3_kernel.kd` | 17.258 | 644 | 26.8 | 0.96% |
| 13 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx1250.kd` | 16.836 | 552 | 30.5 | 0.94% |
| 14 | MoE | `gluon_sigmoid_bias_topk_gfx1250.kd` | 15.654 | 644 | 24.3 | 0.87% |
| 15 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 14.574 | 1,203 | 12.1 | 0.81% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 14.363 | 483 | 29.7 | 0.80% |
| 17 | MoE | `_precomputed_topk_route_large_stage4_gfx1250_kernel.kd` | 14.144 | 644 | 22.0 | 0.79% |
| 18 | MoE | `_precomputed_topk_route_large_stage1_gfx1250_kernel.kd` | 10.519 | 644 | 16.3 | 0.59% |
| 19 | input projections | `_packed_input_projections_kernel.kd` | 8.497 | 92 | 92.4 | 0.47% |
| 20 | other | `gluon_kda_paged_prefill_gfx1250.kd` | 7.674 | 552 | 13.9 | 0.43% |

### MI455X (gfx1250) decode c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode.kd` | 284.797 | 11,776 | 24.2 | 30.36% |
| 2 | dense GEMM | `_wmma_tdm_dense_m16_kernel.kd` | 118.544 | 19,456 | 6.1 | 12.64% |
| 3 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 78.469 | 11,968 | 6.6 | 8.37% |
| 4 | MLA attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_HEAD_BLOCKS_1_NUM_KV_SPLITS_32_num_warps_2_num_stages_2.kd` | 71.854 | 1,536 | 46.8 | 7.66% |
| 5 | MoE | `_precomputed_topk_route_small_m_gfx1250_kernel.kd` | 58.416 | 5,888 | 9.9 | 6.23% |
| 6 | input projections | `gluon_latent_input_decode_gfx1250.kd` | 53.892 | 5,888 | 9.2 | 5.75% |
| 7 | dense GEMM | `_wmma_tdm_add3_m16_kernel.kd` | 41.365 | 5,888 | 7.0 | 4.41% |
| 8 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 34.840 | 9,024 | 3.9 | 3.71% |
| 9 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx1250.kd` | 32.002 | 4,416 | 7.2 | 3.41% |
| 10 | MoE | `_kimi3_sigmoid_bias_topk_kernel.kd` | 27.895 | 5,888 | 4.7 | 2.97% |
| 11 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 23.108 | 4,480 | 5.2 | 2.46% |
| 12 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 23.023 | 3,072 | 7.5 | 2.45% |
| 13 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 17.668 | 5,888 | 3.0 | 1.88% |
| 14 | rmsnorm | `_rmsnorm_kernel.kd` | 16.705 | 5,888 | 2.8 | 1.78% |
| 15 | input projections | `gluon_latent_input_decode_epilogue_gfx1250.kd` | 16.327 | 5,888 | 2.8 | 1.74% |
| 16 | other | `_fp8_quantize_kernel.kd` | 14.346 | 5,888 | 2.4 | 1.53% |
| 17 | rmsnorm | `_rmsnorm_fused_parallel_kernel.kd` | 5.345 | 1,536 | 3.5 | 0.57% |
| 18 | MLA attention | `_mla_decode_fwd_reduce_kernel_num_query_heads_12_TILE_SIZE_64_KV_LORA_RANK_512_NUM_KV_SPLITS_32_ALL_DECODE_1_HAS_LSE_0_num_warps_4.kd` | 4.568 | 1,536 | 3.0 | 0.49% |
| 19 | other | `_mla_rope_set_kv_buffer_kernel.kd` | 4.390 | 1,536 | 2.9 | 0.47% |
| 20 | MoE | `_sigmoid_mul_kernel.kd` | 4.387 | 1,536 | 2.9 | 0.47% |

### MI455X (gfx1250) decode c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode.kd` | 87.205 | 11,776 | 7.4 | 14.92% |
| 2 | AttnRes | `gluon_linear_attnres_partials_gfx1250.kd` | 74.598 | 5,440 | 13.7 | 12.77% |
| 3 | dense GEMM | `_rowcta_gemv_kernel.kd` | 54.984 | 12,480 | 4.4 | 9.41% |
| 4 | input projections | `gluon_latent_input_decode_gfx1250.kd` | 46.282 | 5,888 | 7.9 | 7.92% |
| 5 | dense GEMM | `_rowcta_gemv_add3_kernel.kd` | 40.472 | 5,888 | 6.9 | 6.93% |
| 6 | AttnRes | `_attnres_combine_kernel.kd` | 35.090 | 11,840 | 3.0 | 6.00% |
| 7 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx1250.kd` | 29.485 | 4,416 | 6.7 | 5.05% |
| 8 | MoE | `_decode_sigmoid_bias_topk_kernel.kd` | 23.728 | 5,888 | 4.0 | 4.06% |
| 9 | MLA attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_HEAD_BLOCKS_1_NUM_KV_SPLITS_64_num_warps_2_num_stages_2.kd` | 23.075 | 1,536 | 15.0 | 3.95% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 21.550 | 4,480 | 4.8 | 3.69% |
| 11 | MoE | `_precomputed_topk_route_m1_canonical_gfx1250_kernel.kd` | 18.311 | 5,888 | 3.1 | 3.13% |
| 12 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 17.788 | 5,888 | 3.0 | 3.04% |
| 13 | elementwise | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>) [clone .kd]` | 17.143 | 5,504 | 3.1 | 2.93% |
| 14 | MLA attention | `_mla_reduce_project_value_kernel.kd` | 16.473 | 1,536 | 10.7 | 2.82% |
| 15 | input projections | `gluon_latent_input_decode_epilogue_gfx1250.kd` | 15.294 | 5,888 | 2.6 | 2.62% |
| 16 | other | `gluon_mla_normalize_project_query_gfx1250.kd` | 14.710 | 1,536 | 9.6 | 2.52% |
| 17 | rmsnorm | `_rmsnorm_kernel.kd` | 14.232 | 5,952 | 2.4 | 2.44% |
| 18 | other | `_fp8_quantize_kernel.kd` | 9.851 | 5,888 | 1.7 | 1.69% |
| 19 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 7.617 | 1,536 | 5.0 | 1.30% |
| 20 | AttnRes | `_attnres_partial_kernel.kd` | 6.462 | 960 | 6.7 | 1.11% |

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
  --revision 2a8dbc6b4c809823543da4cfc4dea70ec3ba2fcc \
  --commit-date 2026-09-30 \
  --gfx950-performance <gfx950 performance result.json> \
  --gfx950-hotspots <gfx950 hotspots.json> \
  --gfx1250-performance <gfx1250 performance result.json> \
  --gfx1250-hotspots <gfx1250 hotspots.json> \
  --output-dir toy_e2e/results/arch_compare_20260930_2a8dbc6b
```
