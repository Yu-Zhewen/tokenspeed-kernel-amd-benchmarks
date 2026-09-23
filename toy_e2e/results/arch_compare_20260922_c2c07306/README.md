# Kimi-K3 TP8/EP1 logical rank 0: MI355X vs MI455X at `c2c07306`

One physical GPU per architecture executes rank 0 of the TP8 model with
local substitutes for rank-spanning collectives. Both architectures ran
the same TokenSpeed commit, the same workload, and the same harness.

## Provenance

| Field | Value |
|---|---|
| TokenSpeed revision | `c2c0730670b818880dcaf1a026b3266eb3e110dd` |
| Commit date (UTC) | 2026-09-22 |
| Measured | 2026-09-23 |
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
| prefill TTFT p50 (ms) | 1 | 2,990.3 | 2,459.3 | 1.22x |
| prefill step p50 (ms) | 1 | 892.1 | 727.6 | 1.23x |
| decode step p50 (ms) | 1 | 11.9 | 9.2 | 1.30x |
| decode tok/s per user | 1 | 83.5 | 108.8 | 1.30x |
| aggregate output tok/s | 1 | 67.2 | 86.4 | 1.29x |
| prefill TTFT p50 (ms) | 16 | 24,425.8 | 19,974.3 | 1.22x |
| prefill step p50 (ms) | 16 | 935.8 | 759.0 | 1.23x |
| decode step p50 (ms) | 16 | 24.0 | 20.5 | 1.17x |
| decode tok/s per user | 16 | 22.2 | 26.6 | 1.20x |
| aggregate output tok/s | 16 | 232.4 | 280.2 | 1.21x |

## Where the time goes

Kernels are bucketed by function because the two architectures do not
split the work into the same kernels. Ratios are accumulated GPU kernel
duration, MI355X over MI455X, so above 1.00x means MI455X is ahead.

| Category | prefill c16 | prefill c1 | decode c16 | decode c1 |
|---|---:|---:|---:|---:|
| AttnRes | 0.65x | 0.65x | 0.58x | 0.44x |
| dense GEMM | 1.13x | 1.15x | 1.43x | 0.88x |
| input projections | 1.13x | 1.13x | 1.56x | 1.64x |
| KDA state scan | 1.00x | 0.98x | — | — |
| MLA attention | 1.23x | 1.37x | 1.58x | 2.63x |
| other | 1.21x | 1.22x | 0.50x | 1.29x |
| elementwise | 1.10x | 1.11x | 0.95x | 1.28x |
| KDA other | 1.41x | 1.41x | 1.52x | 1.15x |
| rmsnorm | 1.76x | 1.77x | 1.22x | 0.98x |
| add3 | 2.94x | 2.93x | — | — |
| MoE | 1.88x | 1.97x | 1.48x | 1.09x |

The same buckets, expressed as the milliseconds MI455X would save if a
bucket reached 1.5x, the ratio its strongest buckets already
reach. Negative means MI455X is already past that bar, so the bucket has
nothing to give relative to MI355X whatever its absolute cost.

| Category | prefill c16 | prefill c1 | decode c16 | decode c1 |
|---|---:|---:|---:|---:|
| AttnRes | 3,800 | 239 | 133 | 2 |
| dense GEMM | 2,406 | 138 | 12 | 49 |
| input projections | 1,137 | 71 | -4 | -6 |
| KDA state scan | 790 | 54 | — | — |
| MLA attention | 495 | 13 | -5 | -27 |
| other | 240 | 14 | 24 | 25 |
| elementwise | 179 | 12 | 29 | 3 |
| KDA other | 89 | 6 | -0 | 6 |
| rmsnorm | -35 | -2 | 4 | 5 |
| add3 | -257 | -16 | — | — |
| MoE | -1,806 | -145 | 7 | 41 |

## Heaviest kernels

Percent is the share of accumulated GPU kernel duration within the stage,
not wall time, and kernels may overlap. The heaviest
20 symbols appear below; the CSVs under
`kernel-tables/` keep every symbol.

### MI355X (gfx950) prefill c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel.kd` | 5,528.001 | 9,016 | 613.1 | 12.23% |
| 2 | input projections | `_packed_input_projections_kernel.kd` | 5,187.772 | 9,016 | 575.4 | 11.48% |
| 3 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel.kd` | 4,910.272 | 9,016 | 544.6 | 10.86% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 4,390.146 | 18,326 | 239.6 | 9.71% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 4,150.246 | 7,003 | 592.6 | 9.18% |
| 6 | MLA attention | `gluon_mla_prefill_gfx950.kd` | 3,329.974 | 5,856 | 568.6 | 7.37% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 2,409.330 | 16,833 | 143.1 | 5.33% |
| 8 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950.kd` | 2,393.559 | 7,640 | 313.3 | 5.30% |
| 9 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950.kd` | 2,377.333 | 7,866 | 302.2 | 5.26% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 1,971.208 | 9,016 | 218.6 | 4.36% |
| 11 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx950.kd` | 937.117 | 7,866 | 119.1 | 2.07% |
| 12 | add3 | `_add3_kernel.kd` | 785.777 | 9,016 | 87.2 | 1.74% |
| 13 | MoE | `_gather_package_cdna4_scale_kernel.kd` | 546.575 | 9,016 | 60.6 | 1.21% |
| 14 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x192x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 506.932 | 2,589 | 195.8 | 1.12% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x256x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 493.938 | 10,362 | 47.7 | 1.09% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 463.533 | 6,762 | 68.5 | 1.03% |
| 17 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB2_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_15_MO40_NTn1_NTA0_NTB0_NTC6_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO1_SRVW0_SSO1_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 450.361 | 3,275 | 137.5 | 1.00% |
| 18 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx950.kd` | 368.233 | 7,866 | 46.8 | 0.81% |
| 19 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 354.561 | 9,016 | 39.3 | 0.78% |
| 20 | other | `_mxfp4_quantize_cdna4_scale_tiled_kernel.kd` | 253.481 | 9,016 | 28.1 | 0.56% |

### MI355X (gfx950) prefill c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel.kd` | 372.861 | 644 | 579.0 | 12.87% |
| 2 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel.kd` | 330.533 | 644 | 513.2 | 11.41% |
| 3 | input projections | `_packed_input_projections_kernel.kd` | 327.954 | 644 | 509.2 | 11.32% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 274.636 | 1,309 | 209.8 | 9.48% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 250.690 | 414 | 605.5 | 8.65% |
| 6 | MLA attention | `gluon_mla_prefill_gfx950.kd` | 212.645 | 360 | 590.7 | 7.34% |
| 7 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950.kd` | 173.952 | 552 | 315.1 | 6.00% |
| 8 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950.kd` | 151.291 | 552 | 274.1 | 5.22% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 139.101 | 1,116 | 124.6 | 4.80% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 130.487 | 644 | 202.6 | 4.50% |
| 11 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx950.kd` | 59.589 | 552 | 108.0 | 2.06% |
| 12 | add3 | `_add3_kernel.kd` | 49.549 | 644 | 76.9 | 1.71% |
| 13 | MoE | `_gather_package_cdna4_scale_kernel.kd` | 37.247 | 644 | 57.8 | 1.29% |
| 14 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x256x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 32.610 | 750 | 43.5 | 1.13% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x192x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 30.474 | 144 | 211.6 | 1.05% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 28.956 | 483 | 60.0 | 1.00% |
| 17 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx950.kd` | 23.365 | 552 | 42.3 | 0.81% |
| 18 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 22.803 | 644 | 35.4 | 0.79% |
| 19 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 16.889 | 1,203 | 14.0 | 0.58% |
| 20 | KDA other | `gluon_kda_paged_prefill_solve_merge_gfx950.kd` | 16.877 | 552 | 30.6 | 0.58% |

### MI355X (gfx950) decode c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_warp_decode_precomputed_situ_stage1_kernel.kd` | 346.392 | 5,888 | 58.8 | 21.45% |
| 2 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel.kd` | 166.437 | 5,888 | 28.3 | 10.31% |
| 3 | MLA attention | `_mla_decode_gluon.kd` | 138.605 | 1,536 | 90.2 | 8.58% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 126.802 | 11,968 | 10.6 | 7.85% |
| 5 | input projections | `gluon_latent_input_small_batch_gfx950.kd` | 118.425 | 5,888 | 20.1 | 7.33% |
| 6 | dense GEMM | `gluon_mm_a16w16_medium_gfx950.kd` | 98.698 | 5,888 | 16.8 | 6.11% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 92.588 | 4,416 | 21.0 | 5.73% |
| 8 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 58.743 | 5,888 | 10.0 | 3.64% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 55.012 | 6,016 | 9.1 | 3.41% |
| 10 | MoE | `_dynamic_fp8_single_pass_kernel.kd` | 54.171 | 5,888 | 9.2 | 3.35% |
| 11 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx950.kd` | 45.989 | 4,416 | 10.4 | 2.85% |
| 12 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x256_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB512_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 42.573 | 5,888 | 7.2 | 2.64% |
| 13 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 41.593 | 9,024 | 4.6 | 2.58% |
| 14 | MoE | `_moe_partial_reduce.kd` | 26.849 | 5,888 | 4.6 | 1.66% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 26.844 | 1,600 | 16.8 | 1.66% |
| 16 | elementwise | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 26.505 | 5,888 | 4.5 | 1.64% |
| 17 | input projections | `gluon_latent_input_small_batch_epilogue_gfx950.kd` | 23.941 | 5,888 | 4.1 | 1.48% |
| 18 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 23.937 | 4,416 | 5.4 | 1.48% |
| 19 | rmsnorm | `_rmsnorm_kernel.kd` | 21.641 | 5,888 | 3.7 | 1.34% |
| 20 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB256_LBSPPM0_LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 16.145 | 1,536 | 10.5 | 1.00% |

### MI355X (gfx950) decode c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | input projections | `gluon_latent_input_decode_gfx950.kd` | 107.287 | 5,888 | 18.2 | 13.01% |
| 2 | other | `gluon_linear_attnres_partials_gfx950.kd` | 98.984 | 5,440 | 18.2 | 12.01% |
| 3 | MLA attention | `_mla_decode_gluon.kd` | 94.406 | 1,536 | 61.5 | 11.45% |
| 4 | MoE | `_warp_decode_precomputed_situ_stage1_kernel.kd` | 71.912 | 5,888 | 12.2 | 8.72% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 66.156 | 5,952 | 11.1 | 8.03% |
| 6 | add3 | `_rowcta_gemv_add3_kernel.kd` | 62.763 | 5,888 | 10.7 | 7.61% |
| 7 | other | `_attnres_combine_kernel.kd` | 44.153 | 11,840 | 3.7 | 5.36% |
| 8 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel.kd` | 36.223 | 5,888 | 6.2 | 4.39% |
| 9 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx950.kd` | 31.595 | 4,416 | 7.2 | 3.83% |
| 10 | other | `_kimi3_projection_gemv_kernel.kd` | 29.247 | 5,888 | 5.0 | 3.55% |
| 11 | MoE | `_kimi3_sigmoid_bias_topk_kernel.kd` | 22.997 | 5,888 | 3.9 | 2.79% |
| 12 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 21.893 | 4,416 | 5.0 | 2.66% |
| 13 | elementwise | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>) [clone .kd]` | 20.723 | 5,504 | 3.8 | 2.51% |
| 14 | MoE | `_moe_partial_reduce.kd` | 20.709 | 5,888 | 3.5 | 2.51% |
| 15 | other | `_mla_reduce_project_value_kernel.kd` | 14.427 | 1,536 | 9.4 | 1.75% |
| 16 | rmsnorm | `_rmsnorm_kernel.kd` | 13.127 | 5,952 | 2.2 | 1.59% |
| 17 | MoE | `_dynamic_fp8_single_pass_kernel.kd` | 12.306 | 5,888 | 2.1 | 1.49% |
| 18 | other | `gluon_mla_normalize_project_query_gfx950.kd` | 11.582 | 1,536 | 7.5 | 1.40% |
| 19 | other | `_mla_rope_set_kv_buffer_kernel.kd` | 8.075 | 1,536 | 5.3 | 0.98% |
| 20 | dense GEMM | `gluon_bmm_a16w16_gfx950.kd` | 7.864 | 1,536 | 5.1 | 0.95% |

### MI455X (gfx1250) prefill c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel.kd` | 8,726.239 | 46,944 | 185.9 | 23.48% |
| 2 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 6,726.874 | 18,326 | 367.1 | 18.10% |
| 3 | MoE | `_matmul.kd` | 5,539.896 | 18,032 | 307.2 | 14.90% |
| 4 | input projections | `_packed_input_projections_kernel.kd` | 4,595.642 | 9,016 | 509.7 | 12.36% |
| 5 | MLA attention | `gluon_mla_prefill_gfx1250.kd` | 2,714.522 | 5,856 | 463.5 | 7.30% |
| 6 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250.kd` | 2,374.907 | 7,866 | 301.9 | 6.39% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 900.030 | 22,478 | 40.0 | 2.42% |
| 8 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250.kd` | 847.329 | 7,866 | 107.7 | 2.28% |
| 9 | other | `_fp8_quantize_kernel.kd` | 714.112 | 20,384 | 35.0 | 1.92% |
| 10 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 551.552 | 9,016 | 61.2 | 1.48% |
| 11 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 455.883 | 28,448 | 16.0 | 1.23% |
| 12 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel.kd` | 317.580 | 9,016 | 35.2 | 0.85% |
| 13 | add3 | `_add3_kernel.kd` | 267.287 | 9,016 | 29.6 | 0.72% |
| 14 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx1250.kd` | 255.036 | 7,866 | 32.4 | 0.69% |
| 15 | MoE | `gluon_sigmoid_bias_topk_gfx1250.kd` | 234.607 | 9,016 | 26.0 | 0.63% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 217.656 | 6,762 | 32.2 | 0.59% |
| 17 | MoE | `_precomputed_topk_route_large_stage4_gfx1250_kernel.kd` | 204.918 | 9,016 | 22.7 | 0.55% |
| 18 | MoE | `_precomputed_topk_route_large_stage1_gfx1250_kernel.kd` | 143.686 | 9,016 | 15.9 | 0.39% |
| 19 | KDA other | `gluon_kda_paged_prefill_gfx1250.kd` | 119.196 | 7,866 | 15.2 | 0.32% |
| 20 | other | `_mla_nope_quantize_fp8_kernel.kd` | 110.560 | 2,352 | 47.0 | 0.30% |

### MI455X (gfx1250) prefill c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel.kd` | 528.906 | 3,300 | 160.3 | 22.85% |
| 2 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 422.547 | 1,309 | 322.8 | 18.26% |
| 3 | MoE | `_matmul.kd` | 325.762 | 1,104 | 295.1 | 14.07% |
| 4 | input projections | `_packed_input_projections_kernel.kd` | 289.399 | 644 | 449.4 | 12.50% |
| 5 | MLA attention | `gluon_mla_prefill_gfx1250.kd` | 154.789 | 360 | 430.0 | 6.69% |
| 6 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250.kd` | 154.604 | 552 | 280.1 | 6.68% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 59.758 | 1,486 | 40.2 | 2.58% |
| 8 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250.kd` | 54.183 | 552 | 98.2 | 2.34% |
| 9 | other | `_fp8_quantize_kernel.kd` | 44.875 | 1,456 | 30.8 | 1.94% |
| 10 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 34.562 | 644 | 53.7 | 1.49% |
| 11 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 29.609 | 2,169 | 13.7 | 1.28% |
| 12 | MoE | `_matmul_decode.kd` | 24.738 | 184 | 134.4 | 1.07% |
| 13 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel.kd` | 22.198 | 644 | 34.5 | 0.96% |
| 14 | add3 | `_add3_kernel.kd` | 16.902 | 644 | 26.2 | 0.73% |
| 15 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx1250.kd` | 15.804 | 552 | 28.6 | 0.68% |
| 16 | MoE | `gluon_sigmoid_bias_topk_gfx1250.kd` | 15.071 | 644 | 23.4 | 0.65% |
| 17 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 13.966 | 483 | 28.9 | 0.60% |
| 18 | MoE | `_precomputed_topk_route_large_stage4_gfx1250_kernel.kd` | 13.623 | 644 | 21.2 | 0.59% |
| 19 | MoE | `_precomputed_topk_route_large_stage1_gfx1250_kernel.kd` | 10.257 | 644 | 15.9 | 0.44% |
| 20 | KDA other | `gluon_kda_paged_prefill_gfx1250.kd` | 7.787 | 552 | 14.1 | 0.34% |

### MI455X (gfx1250) decode c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode.kd` | 304.152 | 11,776 | 25.8 | 22.43% |
| 2 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 217.598 | 11,968 | 18.2 | 16.05% |
| 3 | dense GEMM | `_wmma_tdm_dense_m16_kernel.kd` | 197.828 | 17,920 | 11.0 | 14.59% |
| 4 | input projections | `_packed_input_projections_kernel.kd` | 90.991 | 5,888 | 15.5 | 6.71% |
| 5 | MLA attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_HEAD_BLOCKS_1_NUM_KV_SPLITS_32_num_warps_2_num_stages_2.kd` | 83.232 | 1,536 | 54.2 | 6.14% |
| 6 | add3 | `_wmma_tdm_add3_m16_kernel.kd` | 79.909 | 5,888 | 13.6 | 5.89% |
| 7 | MoE | `_precomputed_topk_route_small_m_gfx1250_kernel.kd` | 57.764 | 5,888 | 9.8 | 4.26% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 45.733 | 6,016 | 7.6 | 3.37% |
| 9 | MoE | `gluon_sigmoid_bias_topk_gfx1250.kd` | 43.817 | 5,888 | 7.4 | 3.23% |
| 10 | elementwise | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 38.803 | 5,888 | 6.6 | 2.86% |
| 11 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 35.120 | 9,024 | 3.9 | 2.59% |
| 12 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx1250.kd` | 30.218 | 4,416 | 6.8 | 2.23% |
| 13 | other | `_fp8_quantize_kernel.kd` | 29.201 | 11,776 | 2.5 | 2.15% |
| 14 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 22.940 | 3,072 | 7.5 | 1.69% |
| 15 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 18.623 | 5,888 | 3.2 | 1.37% |
| 16 | rmsnorm | `_rmsnorm_kernel.kd` | 17.564 | 5,888 | 3.0 | 1.30% |
| 17 | MoE | `_situ_kernel.kd` | 16.608 | 5,888 | 2.8 | 1.22% |
| 18 | rmsnorm | `_rmsnorm_fused_parallel_kernel.kd` | 5.754 | 1,536 | 3.7 | 0.42% |
| 19 | other | `_mla_rope_set_kv_buffer_kernel.kd` | 4.745 | 1,536 | 3.1 | 0.35% |
| 20 | MoE | `_sigmoid_mul_kernel.kd` | 4.709 | 1,536 | 3.1 | 0.35% |

### MI455X (gfx1250) decode c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode.kd` | 91.734 | 11,776 | 7.8 | 15.19% |
| 2 | other | `gluon_linear_attnres_partials_gfx1250.kd` | 72.273 | 5,440 | 13.3 | 11.97% |
| 3 | input projections | `gluon_latent_input_decode_gfx1250.kd` | 65.263 | 5,888 | 11.1 | 10.81% |
| 4 | dense GEMM | `_rowcta_gemv_kernel.kd` | 50.917 | 12,480 | 4.1 | 8.43% |
| 5 | dense GEMM | `_rowcta_gemv_add3_kernel.kd` | 38.441 | 5,888 | 6.5 | 6.37% |
| 6 | MLA attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_HEAD_BLOCKS_1_NUM_KV_SPLITS_64_num_warps_2_num_stages_2.kd` | 35.915 | 1,536 | 23.4 | 5.95% |
| 7 | other | `_attnres_combine_kernel.kd` | 33.811 | 11,840 | 2.9 | 5.60% |
| 8 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx1250.kd` | 27.514 | 4,416 | 6.2 | 4.56% |
| 9 | other | `_fp8_quantize_kernel.kd` | 23.931 | 11,776 | 2.0 | 3.96% |
| 10 | MoE | `_decode_sigmoid_bias_topk_kernel.kd` | 23.419 | 5,888 | 4.0 | 3.88% |
| 11 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 20.233 | 4,480 | 4.5 | 3.35% |
| 12 | MoE | `_precomputed_topk_route_m1_canonical_gfx1250_kernel.kd` | 17.991 | 5,888 | 3.1 | 2.98% |
| 13 | elementwise | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>) [clone .kd]` | 17.171 | 5,504 | 3.1 | 2.84% |
| 14 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 16.801 | 5,888 | 2.9 | 2.78% |
| 15 | other | `_mla_reduce_project_value_kernel.kd` | 16.170 | 1,536 | 10.5 | 2.68% |
| 16 | other | `gluon_mla_normalize_project_query_gfx1250.kd` | 14.308 | 1,536 | 9.3 | 2.37% |
| 17 | rmsnorm | `_rmsnorm_kernel.kd` | 13.381 | 5,952 | 2.2 | 2.22% |
| 18 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 7.422 | 1,536 | 4.8 | 1.23% |
| 19 | other | `_attnres_partial_kernel.kd` | 6.461 | 960 | 6.7 | 1.07% |
| 20 | other | `_mla_rope_set_kv_buffer_kernel.kd` | 3.862 | 1,536 | 2.5 | 0.64% |

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
  --revision c2c0730670b818880dcaf1a026b3266eb3e110dd \
  --commit-date 2026-09-22 \
  --gfx950-performance <gfx950 performance result.json> \
  --gfx950-hotspots <gfx950 hotspots.json> \
  --gfx1250-performance <gfx1250 performance result.json> \
  --gfx1250-hotspots <gfx1250 hotspots.json> \
  --output-dir toy_e2e/results/arch_compare_20260922_c2c07306
```
