# Kimi-K3 TP8/EP1 logical rank 0: MI355X vs MI455X at `2199c66d`

One physical GPU per architecture executes rank 0 of the TP8 model with
local substitutes for rank-spanning collectives. Both architectures ran
the same TokenSpeed commit, the same workload, and the same harness.

## Provenance

| Field | Value |
|---|---|
| TokenSpeed revision | `2199c66d2a71590b14f74708f8db69693a855102` |
| Commit date (UTC) | 2026-09-29 |
| Measured | 2026-09-29 |
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
| prefill TTFT p50 (ms) | 1 | 2,766.8 | 1,964.0 | 1.41x |
| prefill step p50 (ms) | 1 | 831.4 | 568.2 | 1.46x |
| decode step p50 (ms) | 1 | 12.0 | 8.8 | 1.36x |
| decode tok/s per user | 1 | 83.2 | 114.2 | 1.37x |
| aggregate output tok/s | 1 | 68.0 | 93.7 | 1.38x |
| prefill TTFT p50 (ms) | 16 | 22,717.4 | 15,850.2 | 1.43x |
| prefill step p50 (ms) | 16 | 868.9 | 603.5 | 1.44x |
| decode step p50 (ms) | 16 | 23.8 | 15.4 | 1.55x |
| decode tok/s per user | 16 | 23.1 | 34.5 | 1.50x |
| aggregate output tok/s | 16 | 244.4 | 360.2 | 1.47x |

## Where the time goes

The final row is the measured step latency from the end-to-end table above, not a total of the rows, so the buckets can be read against what the machine actually reported.

Kernels are bucketed by function because the two architectures do not
split the work into the same kernels. Rows follow that category list,
the same order in every entry. Ratios are accumulated GPU kernel
duration, MI355X over MI455X, so above 1.00x means MI455X is ahead.

| Category | prefill c16 | prefill c1 | decode c16 | decode c1 |
|---|---:|---:|---:|---:|
| KDA state scan | 1.42x | 1.41x | — | — |
| MoE | 1.70x | 1.66x | 1.69x | 1.09x |
| dense GEMM | 1.22x | 1.21x | 1.38x | 1.71x |
| input projections | 1.26x | 1.26x | 2.01x | 1.82x |
| MLA attention | 1.10x | 1.22x | 1.60x | 2.07x |
| AttnRes | 1.45x | 1.44x | 1.52x | 1.22x |
| KDA other | 1.40x | 1.40x | 1.50x | 1.11x |
| add3 | 2.90x | 2.93x | — | — |
| rmsnorm | 1.73x | 1.72x | 1.20x | 0.73x |
| elementwise | 1.49x | 1.43x | 1.89x | 1.15x |
| other | 2.25x | 2.38x | 0.68x | 0.66x |
| **End-to-end (step p50)** | **1.44x** | **1.46x** | **1.55x** | **1.36x** |

## Heaviest kernels

Percent is the share of accumulated GPU kernel duration within the stage,
not wall time, and kernels may overlap. The heaviest
20 symbols appear below; the CSVs under
`kernel-tables/` keep every symbol.

### MI355X (gfx950) prefill c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel.kd` | 5,483.665 | 9,016 | 608.2 | 13.19% |
| 2 | input projections | `gluon_latent_input_largem_gfx950.kd` | 4,203.318 | 9,016 | 466.2 | 10.11% |
| 3 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 4,125.148 | 7,003 | 589.1 | 9.93% |
| 4 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel.kd` | 4,091.276 | 9,016 | 453.8 | 9.84% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 3,405.876 | 18,326 | 185.8 | 8.19% |
| 6 | MLA attention | `gluon_mla_prefill_8wave_gfx950.kd` | 2,955.283 | 5,784 | 510.9 | 7.11% |
| 7 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950.kd` | 2,440.741 | 7,866 | 310.3 | 5.87% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 2,402.577 | 16,833 | 142.7 | 5.78% |
| 9 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950.kd` | 2,349.357 | 7,640 | 307.5 | 5.65% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 1,567.496 | 9,016 | 173.9 | 3.77% |
| 11 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx950.kd` | 950.987 | 7,866 | 120.9 | 2.29% |
| 12 | add3 | `_add3_kernel.kd` | 797.444 | 9,016 | 88.4 | 1.92% |
| 13 | MoE | `_gather_package_cdna4_scale_kernel.kd` | 540.580 | 9,016 | 60.0 | 1.30% |
| 14 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x192x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 502.915 | 2,589 | 194.3 | 1.21% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x256x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 493.286 | 10,362 | 47.6 | 1.19% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 465.708 | 6,762 | 68.9 | 1.12% |
| 17 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB2_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_15_MO40_NTn1_NTA0_NTB0_NTC6_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO1_SRVW0_SSO1_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 449.997 | 3,275 | 137.4 | 1.08% |
| 18 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx950.kd` | 376.622 | 7,866 | 47.9 | 0.91% |
| 19 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 353.123 | 9,016 | 39.2 | 0.85% |
| 20 | KDA other | `gluon_kda_paged_prefill_solve_merge_gfx950.kd` | 256.259 | 7,866 | 32.6 | 0.62% |

### MI355X (gfx950) prefill c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel.kd` | 359.112 | 644 | 557.6 | 13.63% |
| 2 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel.kd` | 261.678 | 644 | 406.3 | 9.93% |
| 3 | input projections | `gluon_latent_input_largem_gfx950.kd` | 259.726 | 552 | 470.5 | 9.86% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 249.151 | 414 | 601.8 | 9.46% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 213.537 | 1,309 | 163.1 | 8.10% |
| 6 | MLA attention | `gluon_mla_prefill_8wave_gfx950.kd` | 187.965 | 336 | 559.4 | 7.13% |
| 7 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950.kd` | 172.011 | 552 | 311.6 | 6.53% |
| 8 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950.kd` | 157.148 | 552 | 284.7 | 5.96% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 139.732 | 1,116 | 125.2 | 5.30% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 96.090 | 552 | 174.1 | 3.65% |
| 11 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx950.kd` | 60.425 | 552 | 109.5 | 2.29% |
| 12 | add3 | `_add3_kernel.kd` | 50.248 | 644 | 78.0 | 1.91% |
| 13 | MoE | `_gather_package_cdna4_scale_kernel.kd` | 34.813 | 644 | 54.1 | 1.32% |
| 14 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x256x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 32.690 | 750 | 43.6 | 1.24% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x192x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 30.292 | 144 | 210.4 | 1.15% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 29.204 | 483 | 60.5 | 1.11% |
| 17 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx950.kd` | 23.829 | 552 | 43.2 | 0.90% |
| 18 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 22.704 | 644 | 35.3 | 0.86% |
| 19 | KDA other | `gluon_kda_paged_prefill_solve_merge_gfx950.kd` | 17.083 | 552 | 30.9 | 0.65% |
| 20 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 16.845 | 1,203 | 14.0 | 0.64% |

### MI355X (gfx950) decode c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_warp_decode_precomputed_situ_stage1_kernel.kd` | 344.376 | 5,888 | 58.5 | 21.64% |
| 2 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel.kd` | 165.640 | 5,888 | 28.1 | 10.41% |
| 3 | MLA attention | `_mla_decode_gluon.kd` | 139.167 | 1,536 | 90.6 | 8.74% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 118.857 | 11,968 | 9.9 | 7.47% |
| 5 | input projections | `gluon_latent_input_small_batch_gfx950.kd` | 117.328 | 5,888 | 19.9 | 7.37% |
| 6 | dense GEMM | `gluon_mm_a16w16_medium_gfx950.kd` | 98.663 | 5,888 | 16.8 | 6.20% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 92.497 | 4,416 | 20.9 | 5.81% |
| 8 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 58.638 | 5,888 | 10.0 | 3.68% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 55.603 | 6,016 | 9.2 | 3.49% |
| 10 | MoE | `_dynamic_fp8_single_pass_kernel.kd` | 53.721 | 5,888 | 9.1 | 3.38% |
| 11 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx950.kd` | 45.152 | 4,416 | 10.2 | 2.84% |
| 12 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x256_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB512_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 43.038 | 5,888 | 7.3 | 2.70% |
| 13 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 41.990 | 9,024 | 4.7 | 2.64% |
| 14 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 27.305 | 5,888 | 4.6 | 1.72% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 26.917 | 1,600 | 16.8 | 1.69% |
| 16 | elementwise | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 25.559 | 5,888 | 4.3 | 1.61% |
| 17 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 22.232 | 4,416 | 5.0 | 1.40% |
| 18 | input projections | `gluon_latent_input_small_batch_epilogue_gfx950.kd` | 22.090 | 5,888 | 3.8 | 1.39% |
| 19 | rmsnorm | `_rmsnorm_kernel.kd` | 19.105 | 5,888 | 3.2 | 1.20% |
| 20 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB256_LBSPPM0_LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 15.558 | 1,536 | 10.1 | 0.98% |

### MI355X (gfx950) decode c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | input projections | `gluon_latent_input_decode_gfx950.kd` | 106.912 | 5,888 | 18.2 | 12.94% |
| 2 | AttnRes | `gluon_linear_attnres_partials_gfx950.kd` | 98.117 | 5,440 | 18.0 | 11.88% |
| 3 | MLA attention | `_mla_decode_gluon.kd` | 94.424 | 1,536 | 61.5 | 11.43% |
| 4 | MoE | `_warp_decode_precomputed_situ_stage1_kernel.kd` | 71.472 | 5,888 | 12.1 | 8.65% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 66.642 | 5,952 | 11.2 | 8.07% |
| 6 | dense GEMM | `_rowcta_gemv_add3_kernel.kd` | 62.808 | 5,888 | 10.7 | 7.60% |
| 7 | AttnRes | `_attnres_combine_kernel.kd` | 45.799 | 11,840 | 3.9 | 5.54% |
| 8 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel.kd` | 35.968 | 5,888 | 6.1 | 4.35% |
| 9 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx950.kd` | 31.629 | 4,416 | 7.2 | 3.83% |
| 10 | dense GEMM | `_kimi3_projection_gemv_kernel.kd` | 30.186 | 5,888 | 5.1 | 3.65% |
| 11 | MoE | `_decode_sigmoid_bias_topk_kernel.kd` | 23.321 | 5,888 | 4.0 | 2.82% |
| 12 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 22.114 | 4,416 | 5.0 | 2.68% |
| 13 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 19.837 | 5,888 | 3.4 | 2.40% |
| 14 | elementwise | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>) [clone .kd]` | 19.506 | 5,504 | 3.5 | 2.36% |
| 15 | MLA attention | `_mla_reduce_project_value_kernel.kd` | 14.811 | 1,536 | 9.6 | 1.79% |
| 16 | rmsnorm | `_rmsnorm_kernel.kd` | 13.417 | 5,952 | 2.3 | 1.62% |
| 17 | MoE | `_dynamic_fp8_single_pass_kernel.kd` | 12.339 | 5,888 | 2.1 | 1.49% |
| 18 | other | `gluon_mla_normalize_project_query_gfx950.kd` | 11.111 | 1,536 | 7.2 | 1.35% |
| 19 | other | `_mla_rope_set_kv_buffer_kernel.kd` | 10.029 | 1,536 | 6.5 | 1.21% |
| 20 | dense GEMM | `gluon_bmm_a16w16_gfx950.kd` | 8.118 | 1,536 | 5.3 | 0.98% |

### MI455X (gfx1250) prefill c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel.kd` | 8,521.483 | 46,944 | 181.5 | 29.01% |
| 2 | MoE | `_matmul.kd` | 5,499.411 | 18,032 | 305.0 | 18.72% |
| 3 | input projections | `gluon_latent_input_largem_gfx1250.kd` | 3,336.506 | 9,016 | 370.1 | 11.36% |
| 4 | MLA attention | `gluon_mla_prefill_gfx1250.kd` | 2,749.040 | 5,856 | 469.4 | 9.36% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 2,356.629 | 18,326 | 128.6 | 8.02% |
| 6 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250.kd` | 1,719.420 | 7,866 | 218.6 | 5.85% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250.kd` | 876.971 | 7,866 | 111.5 | 2.99% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 577.742 | 9,016 | 64.1 | 1.97% |
| 9 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel.kd` | 316.176 | 9,016 | 35.1 | 1.08% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 304.628 | 8,036 | 37.9 | 1.04% |
| 11 | add3 | `_add3_kernel.kd` | 275.416 | 9,016 | 30.5 | 0.94% |
| 12 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx1250.kd` | 248.332 | 7,866 | 31.6 | 0.85% |
| 13 | MoE | `gluon_sigmoid_bias_topk_gfx1250.kd` | 241.526 | 9,016 | 26.8 | 0.82% |
| 14 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 224.942 | 6,762 | 33.3 | 0.77% |
| 15 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 213.214 | 14,925 | 14.3 | 0.73% |
| 16 | MoE | `_precomputed_topk_route_large_stage4_gfx1250_kernel.kd` | 208.631 | 9,016 | 23.1 | 0.71% |
| 17 | MoE | `_precomputed_topk_route_large_stage1_gfx1250_kernel.kd` | 144.585 | 9,016 | 16.0 | 0.49% |
| 18 | other | `gluon_kda_paged_prefill_gfx1250.kd` | 114.705 | 7,866 | 14.6 | 0.39% |
| 19 | KDA other | `gluon_kda_paged_prefill_solve_merge_gfx1250.kd` | 111.119 | 7,866 | 14.1 | 0.38% |
| 20 | other | `attn_merge_state_kernel.kd` | 110.907 | 3,504 | 31.7 | 0.38% |

### MI455X (gfx1250) prefill c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel.kd` | 538.595 | 3,300 | 163.2 | 28.94% |
| 2 | MoE | `_matmul.kd` | 338.303 | 1,104 | 306.4 | 18.18% |
| 3 | input projections | `gluon_latent_input_largem_gfx1250.kd` | 203.856 | 552 | 369.3 | 10.95% |
| 4 | MLA attention | `gluon_mla_prefill_gfx1250.kd` | 157.374 | 360 | 437.1 | 8.46% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 147.959 | 1,309 | 113.0 | 7.95% |
| 6 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250.kd` | 111.657 | 552 | 202.3 | 6.00% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250.kd` | 55.595 | 552 | 100.7 | 2.99% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 36.540 | 644 | 56.7 | 1.96% |
| 9 | MoE | `_matmul_decode.kd` | 23.549 | 184 | 128.0 | 1.27% |
| 10 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel.kd` | 22.291 | 644 | 34.6 | 1.20% |
| 11 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 19.691 | 559 | 35.2 | 1.06% |
| 12 | add3 | `_add3_kernel.kd` | 17.178 | 644 | 26.7 | 0.92% |
| 13 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx1250.kd` | 15.601 | 552 | 28.3 | 0.84% |
| 14 | MoE | `gluon_sigmoid_bias_topk_gfx1250.kd` | 15.594 | 644 | 24.2 | 0.84% |
| 15 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 14.396 | 483 | 29.8 | 0.77% |
| 16 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 14.247 | 1,203 | 11.8 | 0.77% |
| 17 | MoE | `_precomputed_topk_route_large_stage4_gfx1250_kernel.kd` | 13.586 | 644 | 21.1 | 0.73% |
| 18 | MoE | `_precomputed_topk_route_large_stage1_gfx1250_kernel.kd` | 10.339 | 644 | 16.1 | 0.56% |
| 19 | input projections | `_packed_input_projections_kernel.kd` | 8.400 | 92 | 91.3 | 0.45% |
| 20 | KDA other | `gluon_kda_paged_prefill_solve_merge_gfx1250.kd` | 7.428 | 552 | 13.5 | 0.40% |

### MI455X (gfx1250) decode c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode.kd` | 276.672 | 11,776 | 23.5 | 27.33% |
| 2 | dense GEMM | `_wmma_tdm_dense_m16_kernel.kd` | 145.802 | 19,456 | 7.5 | 14.40% |
| 3 | MLA attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_HEAD_BLOCKS_1_NUM_KV_SPLITS_32_num_warps_2_num_stages_2.kd` | 82.450 | 1,536 | 53.7 | 8.15% |
| 4 | dense GEMM | `_wmma_tdm_add3_m16_kernel.kd` | 80.588 | 5,888 | 13.7 | 7.96% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 77.994 | 11,968 | 6.5 | 7.70% |
| 6 | MoE | `_precomputed_topk_route_small_m_gfx1250_kernel.kd` | 61.853 | 5,888 | 10.5 | 6.11% |
| 7 | input projections | `gluon_latent_input_decode_gfx1250.kd` | 53.193 | 5,888 | 9.0 | 5.25% |
| 8 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 34.017 | 9,024 | 3.8 | 3.36% |
| 9 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx1250.kd` | 30.010 | 4,416 | 6.8 | 2.96% |
| 10 | MoE | `_kimi3_sigmoid_bias_topk_kernel.kd` | 27.971 | 5,888 | 4.8 | 2.76% |
| 11 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 24.202 | 4,480 | 5.4 | 2.39% |
| 12 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 24.017 | 3,072 | 7.8 | 2.37% |
| 13 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 17.436 | 5,888 | 3.0 | 1.72% |
| 14 | rmsnorm | `_rmsnorm_kernel.kd` | 16.080 | 5,888 | 2.7 | 1.59% |
| 15 | input projections | `gluon_latent_input_decode_epilogue_gfx1250.kd` | 16.004 | 5,888 | 2.7 | 1.58% |
| 16 | other | `_fp8_quantize_kernel.kd` | 14.278 | 5,888 | 2.4 | 1.41% |
| 17 | rmsnorm | `_rmsnorm_fused_parallel_kernel.kd` | 5.123 | 1,536 | 3.3 | 0.51% |
| 18 | MLA attention | `_mla_decode_fwd_reduce_kernel_num_query_heads_12_TILE_SIZE_64_KV_LORA_RANK_512_NUM_KV_SPLITS_32_ALL_DECODE_1_HAS_LSE_0_num_warps_4.kd` | 4.473 | 1,536 | 2.9 | 0.44% |
| 19 | other | `_mla_rope_set_kv_buffer_kernel.kd` | 4.308 | 1,536 | 2.8 | 0.43% |
| 20 | MoE | `_sigmoid_mul_kernel.kd` | 4.301 | 1,536 | 2.8 | 0.42% |

### MI455X (gfx1250) decode c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode.kd` | 82.656 | 11,776 | 7.0 | 13.60% |
| 2 | AttnRes | `gluon_linear_attnres_partials_gfx1250.kd` | 71.869 | 5,440 | 13.2 | 11.83% |
| 3 | dense GEMM | `_rowcta_gemv_kernel.kd` | 51.521 | 12,480 | 4.1 | 8.48% |
| 4 | AttnRes | `_attnres_combine_kernel.kd` | 44.610 | 11,840 | 3.8 | 7.34% |
| 5 | input projections | `gluon_latent_input_decode_gfx1250.kd` | 44.136 | 5,888 | 7.5 | 7.26% |
| 6 | dense GEMM | `_rowcta_gemv_add3_kernel.kd` | 38.385 | 5,888 | 6.5 | 6.32% |
| 7 | MLA attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_HEAD_BLOCKS_1_NUM_KV_SPLITS_64_num_warps_2_num_stages_2.kd` | 36.182 | 1,536 | 23.6 | 5.95% |
| 8 | MoE | `_decode_sigmoid_bias_topk_kernel.kd` | 28.798 | 5,888 | 4.9 | 4.74% |
| 9 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx1250.kd` | 28.563 | 4,416 | 6.5 | 4.70% |
| 10 | MoE | `_precomputed_topk_route_m1_canonical_gfx1250_kernel.kd` | 23.149 | 5,888 | 3.9 | 3.81% |
| 11 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 20.655 | 4,480 | 4.6 | 3.40% |
| 12 | rmsnorm | `_rmsnorm_kernel.kd` | 18.430 | 5,952 | 3.1 | 3.03% |
| 13 | elementwise | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>) [clone .kd]` | 17.279 | 5,504 | 3.1 | 2.84% |
| 14 | MLA attention | `_mla_reduce_project_value_kernel.kd` | 16.643 | 1,536 | 10.8 | 2.74% |
| 15 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 15.453 | 5,888 | 2.6 | 2.54% |
| 16 | input projections | `gluon_latent_input_decode_epilogue_gfx1250.kd` | 14.497 | 5,888 | 2.5 | 2.39% |
| 17 | other | `_fp8_quantize_kernel.kd` | 14.463 | 5,888 | 2.5 | 2.38% |
| 18 | other | `gluon_mla_normalize_project_query_gfx1250.kd` | 14.444 | 1,536 | 9.4 | 2.38% |
| 19 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 7.738 | 1,536 | 5.0 | 1.27% |
| 20 | AttnRes | `_attnres_partial_kernel.kd` | 7.531 | 960 | 7.8 | 1.24% |

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
  --revision 2199c66d2a71590b14f74708f8db69693a855102 \
  --commit-date 2026-09-29 \
  --gfx950-performance <gfx950 performance result.json> \
  --gfx950-hotspots <gfx950 hotspots.json> \
  --gfx1250-performance <gfx1250 performance result.json> \
  --gfx1250-hotspots <gfx1250 hotspots.json> \
  --output-dir toy_e2e/results/arch_compare_20260929_2199c66d
```
