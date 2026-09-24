# Kimi-K3 TP8/EP1 logical rank 0: MI355X vs MI455X at `42686d40`

One physical GPU per architecture executes rank 0 of the TP8 model with
local substitutes for rank-spanning collectives. Both architectures ran
the same TokenSpeed commit, the same workload, and the same harness.

## Provenance

| Field | Value |
|---|---|
| TokenSpeed revision | `42686d4031453fad63e397a7e5707c2f324c4704` |
| Commit date (UTC) | 2026-09-24 |
| Measured | 2026-09-24 |
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
| prefill TTFT p50 (ms) | 1 | 2,827.5 | 2,013.6 | 1.40x |
| prefill step p50 (ms) | 1 | 839.8 | 584.2 | 1.44x |
| decode step p50 (ms) | 1 | 12.1 | 8.8 | 1.37x |
| decode tok/s per user | 1 | 82.5 | 114.0 | 1.38x |
| aggregate output tok/s | 1 | 67.2 | 93.2 | 1.39x |
| prefill TTFT p50 (ms) | 16 | 23,034.3 | 16,203.4 | 1.42x |
| prefill step p50 (ms) | 16 | 882.2 | 617.7 | 1.43x |
| decode step p50 (ms) | 16 | 23.9 | 17.3 | 1.38x |
| decode tok/s per user | 16 | 22.9 | 32.1 | 1.40x |
| aggregate output tok/s | 16 | 241.9 | 340.5 | 1.41x |

## Where the time goes

The final row is the measured step latency from the end-to-end table above, not a total of the rows, so the buckets can be read against what the machine actually reported.

Kernels are bucketed by function because the two architectures do not
split the work into the same kernels. Ratios are accumulated GPU kernel
duration, MI355X over MI455X, so above 1.00x means MI455X is ahead.

| Category | prefill c16 | prefill c1 | decode c16 | decode c1 |
|---|---:|---:|---:|---:|
| dense GEMM | 1.22x | 1.21x | 1.10x | 1.71x |
| MoE | 1.76x | 1.79x | 1.69x | 1.13x |
| input projections | 1.26x | 1.27x | 1.96x | 1.83x |
| MLA attention | 1.23x | 1.37x | 1.58x | 2.08x |
| AttnRes | 1.43x | 1.43x | 1.19x | 1.36x |
| KDA state scan | 1.04x | 1.02x | — | — |
| KDA other | 1.42x | 1.42x | 1.44x | 1.16x |
| other | 2.24x | 2.37x | 0.82x | 0.81x |
| elementwise | 1.71x | 1.66x | 0.95x | 1.25x |
| add3 | 2.80x | 2.81x | — | — |
| rmsnorm | 1.69x | 1.66x | 1.19x | 1.02x |
| **End-to-end (step p50)** | **1.43x** | **1.44x** | **1.38x** | **1.37x** |

## Heaviest kernels

Percent is the share of accumulated GPU kernel duration within the stage,
not wall time, and kernels may overlap. The heaviest
20 symbols appear below; the CSVs under
`kernel-tables/` keep every symbol.

### MI355X (gfx950) prefill c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel.kd` | 5,490.159 | 9,016 | 608.9 | 12.97% |
| 2 | input projections | `gluon_latent_input_prefill_gfx950.kd` | 4,187.945 | 9,016 | 464.5 | 9.89% |
| 3 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel.kd` | 4,153.258 | 9,016 | 460.7 | 9.81% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 4,120.602 | 7,003 | 588.4 | 9.73% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 3,382.754 | 18,326 | 184.6 | 7.99% |
| 6 | MLA attention | `gluon_mla_prefill_gfx950.kd` | 3,328.363 | 5,856 | 568.4 | 7.86% |
| 7 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950.kd` | 2,479.386 | 7,866 | 315.2 | 5.86% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 2,404.768 | 16,833 | 142.9 | 5.68% |
| 9 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950.kd` | 2,352.034 | 7,640 | 307.9 | 5.56% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 1,951.312 | 9,016 | 216.4 | 4.61% |
| 11 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx950.kd` | 948.836 | 7,866 | 120.6 | 2.24% |
| 12 | add3 | `_add3_kernel.kd` | 760.219 | 9,016 | 84.3 | 1.80% |
| 13 | MoE | `_gather_package_cdna4_scale_kernel.kd` | 543.416 | 9,016 | 60.3 | 1.28% |
| 14 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x192x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 502.968 | 2,589 | 194.3 | 1.19% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x256x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 493.250 | 10,362 | 47.6 | 1.17% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 463.958 | 6,762 | 68.6 | 1.10% |
| 17 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB2_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_15_MO40_NTn1_NTA0_NTB0_NTC6_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO1_SRVW0_SSO1_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 452.276 | 3,275 | 138.1 | 1.07% |
| 18 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx950.kd` | 371.110 | 7,866 | 47.2 | 0.88% |
| 19 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 351.620 | 9,016 | 39.0 | 0.83% |
| 20 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 254.597 | 14,925 | 17.1 | 0.60% |

### MI355X (gfx950) prefill c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel.kd` | 371.378 | 644 | 576.7 | 13.70% |
| 2 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel.kd` | 271.893 | 644 | 422.2 | 10.03% |
| 3 | input projections | `gluon_latent_input_prefill_gfx950.kd` | 258.865 | 552 | 469.0 | 9.55% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 249.366 | 414 | 602.3 | 9.20% |
| 5 | MLA attention | `gluon_mla_prefill_gfx950.kd` | 212.794 | 360 | 591.1 | 7.85% |
| 6 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 212.400 | 1,309 | 162.3 | 7.83% |
| 7 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950.kd` | 171.901 | 552 | 311.4 | 6.34% |
| 8 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950.kd` | 157.375 | 552 | 285.1 | 5.80% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 139.424 | 1,116 | 124.9 | 5.14% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 129.174 | 644 | 200.6 | 4.76% |
| 11 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx950.kd` | 60.106 | 552 | 108.9 | 2.22% |
| 12 | add3 | `_add3_kernel.kd` | 47.888 | 644 | 74.4 | 1.77% |
| 13 | MoE | `_gather_package_cdna4_scale_kernel.kd` | 37.206 | 644 | 57.8 | 1.37% |
| 14 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x256x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 32.609 | 750 | 43.5 | 1.20% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x192x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 30.230 | 144 | 209.9 | 1.11% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 29.031 | 483 | 60.1 | 1.07% |
| 17 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx950.kd` | 23.414 | 552 | 42.4 | 0.86% |
| 18 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 22.665 | 644 | 35.2 | 0.84% |
| 19 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 17.153 | 1,203 | 14.3 | 0.63% |
| 20 | KDA other | `gluon_kda_paged_prefill_solve_merge_gfx950.kd` | 16.923 | 552 | 30.7 | 0.62% |

### MI355X (gfx950) decode c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_warp_decode_precomputed_situ_stage1_kernel.kd` | 345.735 | 5,888 | 58.7 | 21.68% |
| 2 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel.kd` | 166.500 | 5,888 | 28.3 | 10.44% |
| 3 | MLA attention | `_mla_decode_gluon.kd` | 138.604 | 1,536 | 90.2 | 8.69% |
| 4 | input projections | `gluon_latent_input_small_batch_gfx950.kd` | 116.749 | 5,888 | 19.8 | 7.32% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 116.513 | 11,968 | 9.7 | 7.31% |
| 6 | dense GEMM | `gluon_mm_a16w16_medium_gfx950.kd` | 98.323 | 5,888 | 16.7 | 6.17% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 92.121 | 4,416 | 20.9 | 5.78% |
| 8 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 58.606 | 5,888 | 10.0 | 3.68% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 55.649 | 6,016 | 9.3 | 3.49% |
| 10 | MoE | `_dynamic_fp8_single_pass_kernel.kd` | 53.585 | 5,888 | 9.1 | 3.36% |
| 11 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx950.kd` | 45.674 | 4,416 | 10.3 | 2.86% |
| 12 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x256_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB512_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 42.424 | 5,888 | 7.2 | 2.66% |
| 13 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 41.005 | 9,024 | 4.5 | 2.57% |
| 14 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 26.722 | 1,600 | 16.7 | 1.68% |
| 15 | MoE | `_moe_partial_reduce.kd` | 26.369 | 5,888 | 4.5 | 1.65% |
| 16 | elementwise | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 26.209 | 5,888 | 4.5 | 1.64% |
| 17 | input projections | `gluon_latent_input_small_batch_epilogue_gfx950.kd` | 22.347 | 5,888 | 3.8 | 1.40% |
| 18 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 22.213 | 4,416 | 5.0 | 1.39% |
| 19 | rmsnorm | `_rmsnorm_kernel.kd` | 20.866 | 5,888 | 3.5 | 1.31% |
| 20 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB256_LBSPPM0_LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 15.845 | 1,536 | 10.3 | 0.99% |

### MI355X (gfx950) decode c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | input projections | `gluon_latent_input_decode_gfx950.kd` | 106.529 | 5,888 | 18.1 | 12.96% |
| 2 | AttnRes | `gluon_linear_attnres_partials_gfx950.kd` | 98.492 | 5,440 | 18.1 | 11.98% |
| 3 | MLA attention | `_mla_decode_gluon.kd` | 94.338 | 1,536 | 61.4 | 11.47% |
| 4 | MoE | `_warp_decode_precomputed_situ_stage1_kernel.kd` | 71.817 | 5,888 | 12.2 | 8.73% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 66.160 | 5,952 | 11.1 | 8.05% |
| 6 | dense GEMM | `_rowcta_gemv_add3_kernel.kd` | 61.246 | 5,888 | 10.4 | 7.45% |
| 7 | AttnRes | `_attnres_combine_kernel.kd` | 46.864 | 11,840 | 4.0 | 5.70% |
| 8 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel.kd` | 35.063 | 5,888 | 6.0 | 4.26% |
| 9 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx950.kd` | 31.643 | 4,416 | 7.2 | 3.85% |
| 10 | dense GEMM | `_kimi3_projection_gemv_kernel.kd` | 29.834 | 5,888 | 5.1 | 3.63% |
| 11 | MoE | `_decode_sigmoid_bias_topk_kernel.kd` | 23.821 | 5,888 | 4.0 | 2.90% |
| 12 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 21.417 | 4,416 | 4.8 | 2.60% |
| 13 | elementwise | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>) [clone .kd]` | 19.935 | 5,504 | 3.6 | 2.42% |
| 14 | MoE | `_moe_partial_reduce.kd` | 17.895 | 5,888 | 3.0 | 2.18% |
| 15 | MLA attention | `_mla_reduce_project_value_kernel.kd` | 14.761 | 1,536 | 9.6 | 1.80% |
| 16 | rmsnorm | `_rmsnorm_kernel.kd` | 13.152 | 5,952 | 2.2 | 1.60% |
| 17 | MoE | `_dynamic_fp8_single_pass_kernel.kd` | 12.214 | 5,888 | 2.1 | 1.49% |
| 18 | other | `gluon_mla_normalize_project_query_gfx950.kd` | 11.344 | 1,536 | 7.4 | 1.38% |
| 19 | other | `_mla_rope_set_kv_buffer_kernel.kd` | 10.191 | 1,536 | 6.6 | 1.24% |
| 20 | dense GEMM | `gluon_bmm_a16w16_gfx950.kd` | 8.121 | 1,536 | 5.3 | 0.99% |

### MI455X (gfx1250) prefill c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel.kd` | 8,510.179 | 46,944 | 181.3 | 28.45% |
| 2 | MoE | `_matmul.kd` | 5,509.921 | 18,032 | 305.6 | 18.42% |
| 3 | input projections | `gluon_latent_input_prefill_gfx1250.kd` | 3,310.276 | 9,016 | 367.2 | 11.07% |
| 4 | MLA attention | `gluon_mla_prefill_gfx1250.kd` | 2,746.218 | 5,856 | 469.0 | 9.18% |
| 5 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250.kd` | 2,388.776 | 7,866 | 303.7 | 7.99% |
| 6 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 2,372.834 | 18,326 | 129.5 | 7.93% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250.kd` | 856.741 | 7,866 | 108.9 | 2.86% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 575.595 | 9,016 | 63.8 | 1.92% |
| 9 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel.kd` | 310.829 | 9,016 | 34.5 | 1.04% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 291.312 | 8,036 | 36.3 | 0.97% |
| 11 | add3 | `_add3_kernel.kd` | 271.873 | 9,016 | 30.2 | 0.91% |
| 12 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx1250.kd` | 255.217 | 7,866 | 32.4 | 0.85% |
| 13 | MoE | `gluon_sigmoid_bias_topk_gfx1250.kd` | 237.869 | 9,016 | 26.4 | 0.80% |
| 14 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 218.356 | 6,762 | 32.3 | 0.73% |
| 15 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 212.095 | 14,925 | 14.2 | 0.71% |
| 16 | MoE | `_precomputed_topk_route_large_stage4_gfx1250_kernel.kd` | 206.499 | 9,016 | 22.9 | 0.69% |
| 17 | MoE | `_precomputed_topk_route_large_stage1_gfx1250_kernel.kd` | 143.571 | 9,016 | 15.9 | 0.48% |
| 18 | other | `gluon_kda_paged_prefill_gfx1250.kd` | 120.345 | 7,866 | 15.3 | 0.40% |
| 19 | other | `attn_merge_state_kernel.kd` | 110.678 | 3,504 | 31.6 | 0.37% |
| 20 | other | `_mla_nope_quantize_fp8_kernel.kd` | 110.334 | 2,352 | 46.9 | 0.37% |

### MI455X (gfx1250) prefill c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel.kd` | 537.089 | 3,300 | 162.8 | 28.38% |
| 2 | MoE | `_matmul.kd` | 338.044 | 1,104 | 306.2 | 17.86% |
| 3 | input projections | `gluon_latent_input_prefill_gfx1250.kd` | 201.750 | 552 | 365.5 | 10.66% |
| 4 | MLA attention | `gluon_mla_prefill_gfx1250.kd` | 157.245 | 360 | 436.8 | 8.31% |
| 5 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250.kd` | 154.668 | 552 | 280.2 | 8.17% |
| 6 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 148.884 | 1,309 | 113.7 | 7.87% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250.kd` | 53.999 | 552 | 97.8 | 2.85% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 36.451 | 644 | 56.6 | 1.93% |
| 9 | MoE | `_matmul_decode.kd` | 23.527 | 184 | 127.9 | 1.24% |
| 10 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel.kd` | 21.887 | 644 | 34.0 | 1.16% |
| 11 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 18.901 | 559 | 33.8 | 1.00% |
| 12 | add3 | `_add3_kernel.kd` | 17.056 | 644 | 26.5 | 0.90% |
| 13 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx1250.kd` | 16.202 | 552 | 29.4 | 0.86% |
| 14 | MoE | `gluon_sigmoid_bias_topk_gfx1250.kd` | 15.352 | 644 | 23.8 | 0.81% |
| 15 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 14.090 | 1,203 | 11.7 | 0.74% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 13.996 | 483 | 29.0 | 0.74% |
| 17 | MoE | `_precomputed_topk_route_large_stage4_gfx1250_kernel.kd` | 13.330 | 644 | 20.7 | 0.70% |
| 18 | MoE | `_precomputed_topk_route_large_stage1_gfx1250_kernel.kd` | 10.265 | 644 | 15.9 | 0.54% |
| 19 | input projections | `_packed_input_projections_kernel.kd` | 8.213 | 92 | 89.3 | 0.43% |
| 20 | other | `gluon_kda_paged_prefill_gfx1250.kd` | 7.906 | 552 | 14.3 | 0.42% |

### MI455X (gfx1250) decode c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode.kd` | 278.922 | 11,776 | 23.7 | 24.36% |
| 2 | dense GEMM | `_wmma_tdm_dense_m16_kernel.kd` | 197.992 | 17,920 | 11.0 | 17.29% |
| 3 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 97.882 | 11,968 | 8.2 | 8.55% |
| 4 | MLA attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_HEAD_BLOCKS_1_NUM_KV_SPLITS_32_num_warps_2_num_stages_2.kd` | 83.524 | 1,536 | 54.4 | 7.29% |
| 5 | dense GEMM | `_wmma_tdm_add3_m16_kernel.kd` | 79.872 | 5,888 | 13.6 | 6.97% |
| 6 | MoE | `_precomputed_topk_route_small_m_gfx1250_kernel.kd` | 58.105 | 5,888 | 9.9 | 5.07% |
| 7 | input projections | `gluon_latent_input_decode_gfx1250.kd` | 53.795 | 5,888 | 9.1 | 4.70% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 45.990 | 6,016 | 7.6 | 4.02% |
| 9 | elementwise | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 38.533 | 5,888 | 6.5 | 3.36% |
| 10 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 34.887 | 9,024 | 3.9 | 3.05% |
| 11 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx1250.kd` | 31.645 | 4,416 | 7.2 | 2.76% |
| 12 | MoE | `_kimi3_sigmoid_bias_topk_kernel.kd` | 28.288 | 5,888 | 4.8 | 2.47% |
| 13 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 23.105 | 3,072 | 7.5 | 2.02% |
| 14 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 18.246 | 5,888 | 3.1 | 1.59% |
| 15 | rmsnorm | `_rmsnorm_kernel.kd` | 17.443 | 5,888 | 3.0 | 1.52% |
| 16 | input projections | `gluon_latent_input_decode_epilogue_gfx1250.kd` | 17.107 | 5,888 | 2.9 | 1.49% |
| 17 | other | `_fp8_quantize_kernel.kd` | 13.925 | 5,888 | 2.4 | 1.22% |
| 18 | rmsnorm | `_rmsnorm_fused_parallel_kernel.kd` | 5.657 | 1,536 | 3.7 | 0.49% |
| 19 | other | `_mla_rope_set_kv_buffer_kernel.kd` | 4.702 | 1,536 | 3.1 | 0.41% |
| 20 | MoE | `_sigmoid_mul_kernel.kd` | 4.634 | 1,536 | 3.0 | 0.40% |

### MI455X (gfx1250) decode c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode.kd` | 84.741 | 11,776 | 7.2 | 14.83% |
| 2 | AttnRes | `gluon_linear_attnres_partials_gfx1250.kd` | 70.923 | 5,440 | 13.0 | 12.41% |
| 3 | dense GEMM | `_rowcta_gemv_kernel.kd` | 50.734 | 12,480 | 4.1 | 8.88% |
| 4 | input projections | `gluon_latent_input_decode_gfx1250.kd` | 44.389 | 5,888 | 7.5 | 7.77% |
| 5 | dense GEMM | `_rowcta_gemv_add3_kernel.kd` | 38.012 | 5,888 | 6.5 | 6.65% |
| 6 | MLA attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_HEAD_BLOCKS_1_NUM_KV_SPLITS_64_num_warps_2_num_stages_2.kd` | 36.298 | 1,536 | 23.6 | 6.35% |
| 7 | AttnRes | `_attnres_combine_kernel.kd` | 34.419 | 11,840 | 2.9 | 6.02% |
| 8 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx1250.kd` | 27.383 | 4,416 | 6.2 | 4.79% |
| 9 | MoE | `_decode_sigmoid_bias_topk_kernel.kd` | 23.040 | 5,888 | 3.9 | 4.03% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 20.229 | 4,480 | 4.5 | 3.54% |
| 11 | MoE | `_precomputed_topk_route_m1_canonical_gfx1250_kernel.kd` | 17.625 | 5,888 | 3.0 | 3.08% |
| 12 | elementwise | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>) [clone .kd]` | 16.988 | 5,504 | 3.1 | 2.97% |
| 13 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 16.620 | 5,888 | 2.8 | 2.91% |
| 14 | MLA attention | `_mla_reduce_project_value_kernel.kd` | 16.183 | 1,536 | 10.5 | 2.83% |
| 15 | other | `gluon_mla_normalize_project_query_gfx1250.kd` | 14.347 | 1,536 | 9.3 | 2.51% |
| 16 | input projections | `gluon_latent_input_decode_epilogue_gfx1250.kd` | 13.916 | 5,888 | 2.4 | 2.44% |
| 17 | rmsnorm | `_rmsnorm_kernel.kd` | 12.953 | 5,952 | 2.2 | 2.27% |
| 18 | other | `_fp8_quantize_kernel.kd` | 9.256 | 5,888 | 1.6 | 1.62% |
| 19 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 7.415 | 1,536 | 4.8 | 1.30% |
| 20 | AttnRes | `_attnres_partial_kernel.kd` | 6.691 | 960 | 7.0 | 1.17% |

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
  --revision 42686d4031453fad63e397a7e5707c2f324c4704 \
  --commit-date 2026-09-24 \
  --gfx950-performance <gfx950 performance result.json> \
  --gfx950-hotspots <gfx950 hotspots.json> \
  --gfx1250-performance <gfx1250 performance result.json> \
  --gfx1250-hotspots <gfx1250 hotspots.json> \
  --output-dir toy_e2e/results/arch_compare_20260924_42686d40
```
