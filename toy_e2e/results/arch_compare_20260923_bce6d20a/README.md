# Kimi-K3 TP8/EP1 logical rank 0: MI355X vs MI455X at `bce6d20a`

One physical GPU per architecture executes rank 0 of the TP8 model with
local substitutes for rank-spanning collectives. Both architectures ran
the same TokenSpeed commit, the same workload, and the same harness.

## Provenance

| Field | Value |
|---|---|
| TokenSpeed revision | `bce6d20a0e64138c8a9efa83a1f522d7a6a671b7` |
| Commit date (UTC) | 2026-09-23 |
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
| prefill TTFT p50 (ms) | 1 | 2,816.5 | 2,107.5 | 1.34x |
| prefill step p50 (ms) | 1 | 836.3 | 616.3 | 1.36x |
| decode step p50 (ms) | 1 | 11.9 | 9.1 | 1.31x |
| decode tok/s per user | 1 | 83.4 | 109.7 | 1.31x |
| aggregate output tok/s | 1 | 67.9 | 89.6 | 1.32x |
| prefill TTFT p50 (ms) | 16 | 22,988.5 | 17,005.3 | 1.35x |
| prefill step p50 (ms) | 16 | 879.6 | 648.1 | 1.36x |
| decode step p50 (ms) | 16 | 23.9 | 18.4 | 1.30x |
| decode tok/s per user | 16 | 22.9 | 30.3 | 1.32x |
| aggregate output tok/s | 16 | 242.2 | 323.0 | 1.33x |

## Where the time goes

Kernels are bucketed by function because the two architectures do not
split the work into the same kernels. Ratios are accumulated GPU kernel
duration, MI355X over MI455X, so above 1.00x means MI455X is ahead.

| Category | prefill c16 | prefill c1 | decode c16 | decode c1 |
|---|---:|---:|---:|---:|
| dense GEMM | 1.19x | 1.18x | 1.43x | 1.67x |
| input projections | 0.93x | 0.95x | 1.54x | 1.66x |
| KDA state scan | 1.01x | 0.99x | — | — |
| MLA attention | 1.23x | 1.37x | 1.58x | 2.08x |
| AttnRes | 1.42x | 1.41x | 1.24x | 1.21x |
| KDA other | 1.40x | 1.41x | 1.48x | 1.16x |
| elementwise | 1.54x | 1.49x | 0.97x | 1.21x |
| rmsnorm | 1.74x | 1.71x | 1.34x | 0.70x |
| add3 | 2.81x | 2.80x | — | — |
| other | 2.23x | 2.38x | 0.94x | 0.65x |
| MoE | 1.74x | 1.77x | 1.50x | 1.04x |

The same buckets, expressed as the milliseconds MI455X would save if a
bucket reached 1.5x, the ratio its strongest buckets already
reach. Negative means MI455X is already past that bar, so the bucket has
nothing to give relative to MI355X whatever its absolute cost.

| Category | prefill c16 | prefill c1 | decode c16 | decode c1 |
|---|---:|---:|---:|---:|
| dense GEMM | 1,872 | 122 | 13 | -14 |
| input projections | 1,740 | 107 | -2 | -7 |
| KDA state scan | 789 | 54 | — | — |
| MLA attention | 502 | 13 | -5 | -20 |
| AttnRes | 124 | 9 | 16 | 24 |
| KDA other | 93 | 6 | 0 | 6 |
| elementwise | -14 | 0 | 27 | 4 |
| rmsnorm | -32 | -2 | 2 | 10 |
| add3 | -237 | -15 | — | — |
| other | -339 | -22 | 8 | 19 |
| MoE | -1,168 | -85 | 0 | 48 |

## Heaviest kernels

Percent is the share of accumulated GPU kernel duration within the stage,
not wall time, and kernels may overlap. The heaviest
20 symbols appear below; the CSVs under
`kernel-tables/` keep every symbol.

### MI355X (gfx950) prefill c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel.kd` | 5,484.313 | 9,016 | 608.3 | 12.96% |
| 2 | input projections | `gluon_latent_input_prefill_gfx950.kd` | 4,184.721 | 9,016 | 464.1 | 9.89% |
| 3 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel.kd` | 4,146.192 | 9,016 | 459.9 | 9.80% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 4,112.105 | 7,003 | 587.2 | 9.72% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 3,381.810 | 18,326 | 184.5 | 7.99% |
| 6 | MLA attention | `gluon_mla_prefill_gfx950.kd` | 3,331.384 | 5,856 | 568.9 | 7.88% |
| 7 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950.kd` | 2,457.491 | 7,866 | 312.4 | 5.81% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 2,410.044 | 16,833 | 143.2 | 5.70% |
| 9 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950.kd` | 2,351.541 | 7,640 | 307.8 | 5.56% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 1,952.359 | 9,016 | 216.5 | 4.62% |
| 11 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx950.kd` | 952.273 | 7,866 | 121.1 | 2.25% |
| 12 | add3 | `_add3_kernel.kd` | 760.735 | 9,016 | 84.4 | 1.80% |
| 13 | MoE | `_gather_package_cdna4_scale_kernel.kd` | 540.218 | 9,016 | 59.9 | 1.28% |
| 14 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x192x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 503.317 | 2,589 | 194.4 | 1.19% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x256x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 493.135 | 10,362 | 47.6 | 1.17% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 463.595 | 6,762 | 68.6 | 1.10% |
| 17 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_DTLB1_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB2_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_15_MO40_NTn1_NTA0_NTB0_NTC6_NTD4_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO1_SRVW0_SSO1_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 452.578 | 3,275 | 138.2 | 1.07% |
| 18 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx950.kd` | 369.707 | 7,866 | 47.0 | 0.87% |
| 19 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 351.559 | 9,016 | 39.0 | 0.83% |
| 20 | KDA other | `gluon_kda_paged_prefill_solve_merge_gfx950.kd` | 254.929 | 7,866 | 32.4 | 0.60% |

### MI355X (gfx950) prefill c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel.kd` | 370.962 | 644 | 576.0 | 13.69% |
| 2 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel.kd` | 271.453 | 644 | 421.5 | 10.02% |
| 3 | input projections | `gluon_latent_input_prefill_gfx950.kd` | 258.924 | 552 | 469.1 | 9.56% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1.kd` | 248.882 | 414 | 601.2 | 9.19% |
| 5 | MLA attention | `gluon_mla_prefill_gfx950.kd` | 212.738 | 360 | 590.9 | 7.85% |
| 6 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 212.110 | 1,309 | 162.0 | 7.83% |
| 7 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950.kd` | 171.773 | 552 | 311.2 | 6.34% |
| 8 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950.kd` | 156.077 | 552 | 282.7 | 5.76% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 140.014 | 1,116 | 125.5 | 5.17% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel.kd` | 129.309 | 644 | 200.8 | 4.77% |
| 11 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx950.kd` | 60.426 | 552 | 109.5 | 2.23% |
| 12 | add3 | `_add3_kernel.kd` | 47.885 | 644 | 74.4 | 1.77% |
| 13 | MoE | `_gather_package_cdna4_scale_kernel.kd` | 36.975 | 644 | 57.4 | 1.36% |
| 14 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x256x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 32.515 | 750 | 43.4 | 1.20% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x192x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT6_6_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 30.207 | 144 | 209.8 | 1.12% |
| 16 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 29.011 | 483 | 60.1 | 1.07% |
| 17 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx950.kd` | 23.399 | 552 | 42.4 | 0.86% |
| 18 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 22.629 | 644 | 35.1 | 0.84% |
| 19 | KDA other | `gluon_kda_paged_prefill_solve_merge_gfx950.kd` | 17.004 | 552 | 30.8 | 0.63% |
| 20 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 16.948 | 1,203 | 14.1 | 0.63% |

### MI355X (gfx950) decode c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_warp_decode_precomputed_situ_stage1_kernel.kd` | 344.577 | 5,888 | 58.5 | 21.52% |
| 2 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel.kd` | 165.835 | 5,888 | 28.2 | 10.36% |
| 3 | MLA attention | `_mla_decode_gluon.kd` | 138.412 | 1,536 | 90.1 | 8.65% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx950.kd` | 118.306 | 11,968 | 9.9 | 7.39% |
| 5 | input projections | `gluon_latent_input_small_batch_gfx950.kd` | 116.676 | 5,888 | 19.8 | 7.29% |
| 6 | dense GEMM | `gluon_mm_a16w16_medium_gfx950.kd` | 99.359 | 5,888 | 16.9 | 6.21% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 92.083 | 4,416 | 20.9 | 5.75% |
| 8 | MoE | `gluon_sigmoid_bias_topk_gfx950.kd` | 58.938 | 5,888 | 10.0 | 3.68% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x512_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 54.444 | 6,016 | 9.0 | 3.40% |
| 10 | MoE | `_dynamic_fp8_single_pass_kernel.kd` | 53.586 | 5,888 | 9.1 | 3.35% |
| 11 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx950.kd` | 45.748 | 4,416 | 10.4 | 2.86% |
| 12 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x256_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB512_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 42.892 | 5,888 | 7.3 | 2.68% |
| 13 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 41.055 | 9,024 | 4.5 | 2.56% |
| 14 | MoE | `_moe_partial_reduce.kd` | 27.023 | 5,888 | 4.6 | 1.69% |
| 15 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 26.907 | 1,600 | 16.8 | 1.68% |
| 16 | elementwise | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 26.070 | 5,888 | 4.4 | 1.63% |
| 17 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 23.413 | 4,416 | 5.3 | 1.46% |
| 18 | input projections | `gluon_latent_input_small_batch_epilogue_gfx950.kd` | 22.339 | 5,888 | 3.8 | 1.40% |
| 19 | rmsnorm | `_rmsnorm_kernel.kd` | 21.712 | 5,888 | 3.7 | 1.36% |
| 20 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x32x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB256_LBSPPM0_LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1.kd` | 16.179 | 1,536 | 10.5 | 1.01% |

### MI355X (gfx950) decode c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | input projections | `gluon_latent_input_decode_gfx950.kd` | 106.627 | 5,888 | 18.1 | 13.06% |
| 2 | AttnRes | `gluon_linear_attnres_partials_gfx950.kd` | 97.589 | 5,440 | 17.9 | 11.96% |
| 3 | MLA attention | `_mla_decode_gluon.kd` | 94.748 | 1,536 | 61.7 | 11.61% |
| 4 | MoE | `_warp_decode_precomputed_situ_stage1_kernel.kd` | 71.321 | 5,888 | 12.1 | 8.74% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 65.982 | 5,952 | 11.1 | 8.08% |
| 6 | dense GEMM | `_rowcta_gemv_add3_kernel.kd` | 61.037 | 5,888 | 10.4 | 7.48% |
| 7 | AttnRes | `_attnres_combine_kernel.kd` | 43.995 | 11,840 | 3.7 | 5.39% |
| 8 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel.kd` | 34.841 | 5,888 | 5.9 | 4.27% |
| 9 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx950.kd` | 32.840 | 4,416 | 7.4 | 4.02% |
| 10 | dense GEMM | `_kimi3_projection_gemv_kernel.kd` | 28.398 | 5,888 | 4.8 | 3.48% |
| 11 | MoE | `_kimi3_sigmoid_bias_topk_kernel.kd` | 23.100 | 5,888 | 3.9 | 2.83% |
| 12 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4.kd` | 20.896 | 4,416 | 4.7 | 2.56% |
| 13 | elementwise | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>) [clone .kd]` | 20.292 | 5,504 | 3.7 | 2.49% |
| 14 | MoE | `_moe_partial_reduce.kd` | 19.501 | 5,888 | 3.3 | 2.39% |
| 15 | MLA attention | `_mla_reduce_project_value_kernel.kd` | 14.641 | 1,536 | 9.5 | 1.79% |
| 16 | rmsnorm | `_rmsnorm_kernel.kd` | 12.928 | 5,952 | 2.2 | 1.58% |
| 17 | MoE | `_dynamic_fp8_single_pass_kernel.kd` | 11.882 | 5,888 | 2.0 | 1.46% |
| 18 | other | `gluon_mla_normalize_project_query_gfx950.kd` | 10.894 | 1,536 | 7.1 | 1.33% |
| 19 | other | `_mla_rope_set_kv_buffer_kernel.kd` | 9.163 | 1,536 | 6.0 | 1.12% |
| 20 | dense GEMM | `gluon_bmm_a16w16_gfx950.kd` | 7.875 | 1,536 | 5.1 | 0.96% |

### MI455X (gfx1250) prefill c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel.kd` | 8,735.742 | 46,944 | 186.1 | 27.71% |
| 2 | MoE | `_matmul.kd` | 5,508.768 | 18,032 | 305.5 | 17.47% |
| 3 | input projections | `_packed_input_projections_kernel.kd` | 4,599.855 | 9,016 | 510.2 | 14.59% |
| 4 | MLA attention | `gluon_mla_prefill_gfx1250.kd` | 2,749.053 | 5,856 | 469.4 | 8.72% |
| 5 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250.kd` | 2,426.916 | 7,866 | 308.5 | 7.70% |
| 6 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 2,378.269 | 18,326 | 129.8 | 7.54% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250.kd` | 870.774 | 7,866 | 110.7 | 2.76% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 552.671 | 9,016 | 61.3 | 1.75% |
| 9 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel.kd` | 315.230 | 9,016 | 35.0 | 1.00% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 303.461 | 8,036 | 37.8 | 0.96% |
| 11 | add3 | `_add3_kernel.kd` | 270.312 | 9,016 | 30.0 | 0.86% |
| 12 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx1250.kd` | 254.561 | 7,866 | 32.4 | 0.81% |
| 13 | MoE | `gluon_sigmoid_bias_topk_gfx1250.kd` | 239.058 | 9,016 | 26.5 | 0.76% |
| 14 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 221.210 | 6,762 | 32.7 | 0.70% |
| 15 | MoE | `_precomputed_topk_route_large_stage4_gfx1250_kernel.kd` | 207.443 | 9,016 | 23.0 | 0.66% |
| 16 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 203.842 | 14,925 | 13.7 | 0.65% |
| 17 | MoE | `_precomputed_topk_route_large_stage1_gfx1250_kernel.kd` | 144.038 | 9,016 | 16.0 | 0.46% |
| 18 | other | `gluon_kda_paged_prefill_gfx1250.kd` | 122.248 | 7,866 | 15.5 | 0.39% |
| 19 | other | `attn_merge_state_kernel.kd` | 110.634 | 3,504 | 31.6 | 0.35% |
| 20 | other | `_mla_nope_quantize_fp8_kernel.kd` | 109.235 | 2,352 | 46.4 | 0.35% |

### MI455X (gfx1250) prefill c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel.kd` | 550.154 | 3,300 | 166.7 | 27.60% |
| 2 | MoE | `_matmul.kd` | 337.687 | 1,104 | 305.9 | 16.94% |
| 3 | input projections | `_packed_input_projections_kernel.kd` | 289.375 | 644 | 449.3 | 14.52% |
| 4 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250.kd` | 157.801 | 552 | 285.9 | 7.92% |
| 5 | MLA attention | `gluon_mla_prefill_gfx1250.kd` | 157.032 | 360 | 436.2 | 7.88% |
| 6 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 150.320 | 1,309 | 114.8 | 7.54% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250.kd` | 55.266 | 552 | 100.1 | 2.77% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 34.567 | 644 | 53.7 | 1.73% |
| 9 | MoE | `_matmul_decode.kd` | 24.967 | 184 | 135.7 | 1.25% |
| 10 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel.kd` | 22.157 | 644 | 34.4 | 1.11% |
| 11 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 19.545 | 559 | 35.0 | 0.98% |
| 12 | add3 | `_add3_kernel.kd` | 17.112 | 644 | 26.6 | 0.86% |
| 13 | KDA other | `gluon_kda_paged_prefill_wu_vector_gfx1250.kd` | 15.589 | 552 | 28.2 | 0.78% |
| 14 | MoE | `gluon_sigmoid_bias_topk_gfx1250.kd` | 15.457 | 644 | 24.0 | 0.78% |
| 15 | KDA other | `_causal_conv1d_fwd_kernel.kd` | 14.153 | 483 | 29.3 | 0.71% |
| 16 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 13.603 | 1,203 | 11.3 | 0.68% |
| 17 | MoE | `_precomputed_topk_route_large_stage4_gfx1250_kernel.kd` | 13.522 | 644 | 21.0 | 0.68% |
| 18 | MoE | `_precomputed_topk_route_large_stage1_gfx1250_kernel.kd` | 10.248 | 644 | 15.9 | 0.51% |
| 19 | other | `gluon_kda_paged_prefill_gfx1250.kd` | 7.638 | 552 | 13.8 | 0.38% |
| 20 | KDA other | `gluon_kda_paged_prefill_solve_merge_gfx1250.kd` | 7.329 | 552 | 13.3 | 0.37% |

### MI455X (gfx1250) decode c16

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode.kd` | 295.015 | 11,776 | 25.1 | 24.45% |
| 2 | dense GEMM | `_wmma_tdm_dense_m16_kernel.kd` | 197.153 | 17,920 | 11.0 | 16.34% |
| 3 | AttnRes | `gluon_attn_res_fwd_gfx1250.kd` | 95.149 | 11,968 | 8.0 | 7.88% |
| 4 | input projections | `_packed_input_projections_kernel.kd` | 90.231 | 5,888 | 15.3 | 7.48% |
| 5 | MLA attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_HEAD_BLOCKS_1_NUM_KV_SPLITS_32_num_warps_2_num_stages_2.kd` | 83.004 | 1,536 | 54.0 | 6.88% |
| 6 | add3 | `_wmma_tdm_add3_m16_kernel.kd` | 80.283 | 5,888 | 13.6 | 6.65% |
| 7 | MoE | `_precomputed_topk_route_small_m_gfx1250_kernel.kd` | 62.313 | 5,888 | 10.6 | 5.16% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 46.635 | 6,016 | 7.8 | 3.86% |
| 9 | MoE | `gluon_sigmoid_bias_topk_gfx1250.kd` | 43.522 | 5,888 | 7.4 | 3.61% |
| 10 | elementwise | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int) [clone .kd]` | 37.470 | 5,888 | 6.4 | 3.10% |
| 11 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]` | 33.012 | 9,024 | 3.7 | 2.74% |
| 12 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx1250.kd` | 30.890 | 4,416 | 7.0 | 2.56% |
| 13 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 23.877 | 3,072 | 7.8 | 1.98% |
| 14 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 16.927 | 5,888 | 2.9 | 1.40% |
| 15 | rmsnorm | `_rmsnorm_kernel.kd` | 15.891 | 5,888 | 2.7 | 1.32% |
| 16 | other | `_fp8_quantize_kernel.kd` | 14.863 | 5,888 | 2.5 | 1.23% |
| 17 | MoE | `_situ_kernel.kd` | 14.672 | 5,888 | 2.5 | 1.22% |
| 18 | rmsnorm | `_rmsnorm_fused_parallel_kernel.kd` | 5.493 | 1,536 | 3.6 | 0.46% |
| 19 | MLA attention | `_mla_decode_fwd_reduce_kernel_num_query_heads_12_TILE_SIZE_64_KV_LORA_RANK_512_NUM_KV_SPLITS_32_ALL_DECODE_1_HAS_LSE_0_num_warps_4.kd` | 4.447 | 1,536 | 2.9 | 0.37% |
| 20 | MoE | `_sigmoid_mul_kernel.kd` | 4.211 | 1,536 | 2.7 | 0.35% |

### MI455X (gfx1250) decode c1

| Rank | Category | Exact kernel name | GPU duration (ms) | Calls | Mean/call (us) | Stage GPU time |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode.kd` | 88.223 | 11,776 | 7.5 | 14.37% |
| 2 | AttnRes | `gluon_linear_attnres_partials_gfx1250.kd` | 71.744 | 5,440 | 13.2 | 11.68% |
| 3 | input projections | `gluon_latent_input_decode_gfx1250.kd` | 64.402 | 5,888 | 10.9 | 10.49% |
| 4 | dense GEMM | `_rowcta_gemv_kernel.kd` | 51.202 | 12,480 | 4.1 | 8.34% |
| 5 | AttnRes | `_attnres_combine_kernel.kd` | 43.082 | 11,840 | 3.6 | 7.02% |
| 6 | dense GEMM | `_rowcta_gemv_add3_kernel.kd` | 37.749 | 5,888 | 6.4 | 6.15% |
| 7 | MLA attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_QK_ROPE_HEAD_DIM_64_BLOCK_Q_1_BLOCK_M_16_NUM_HEAD_BLOCKS_1_NUM_KV_SPLITS_64_num_warps_2_num_stages_2.kd` | 36.008 | 1,536 | 23.4 | 5.86% |
| 8 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx1250.kd` | 28.237 | 4,416 | 6.4 | 4.60% |
| 9 | MoE | `_decode_sigmoid_bias_topk_kernel.kd` | 27.947 | 5,888 | 4.7 | 4.55% |
| 10 | MoE | `_precomputed_topk_route_m1_canonical_gfx1250_kernel.kd` | 22.951 | 5,888 | 3.9 | 3.74% |
| 11 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 20.620 | 4,480 | 4.6 | 3.36% |
| 12 | rmsnorm | `_rmsnorm_kernel.kd` | 18.403 | 5,952 | 3.1 | 3.00% |
| 13 | elementwise | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>) [clone .kd]` | 16.846 | 5,504 | 3.1 | 2.74% |
| 14 | MLA attention | `_mla_reduce_project_value_kernel.kd` | 16.506 | 1,536 | 10.7 | 2.69% |
| 15 | MoE | `_weighted_topk_reduce_gfx1250_kernel.kd` | 16.023 | 5,888 | 2.7 | 2.61% |
| 16 | other | `gluon_mla_normalize_project_query_gfx1250.kd` | 14.428 | 1,536 | 9.4 | 2.35% |
| 17 | other | `_fp8_quantize_kernel.kd` | 13.714 | 5,888 | 2.3 | 2.23% |
| 18 | dense GEMM | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMLWS0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1.kd` | 7.687 | 1,536 | 5.0 | 1.25% |
| 19 | AttnRes | `_attnres_partial_kernel.kd` | 7.556 | 960 | 7.9 | 1.23% |
| 20 | other | `_mla_rope_set_kv_buffer_kernel.kd` | 3.611 | 1,536 | 2.4 | 0.59% |

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
  --revision bce6d20a0e64138c8a9efa83a1f522d7a6a671b7 \
  --commit-date 2026-09-23 \
  --gfx950-performance <gfx950 performance result.json> \
  --gfx950-hotspots <gfx950 hotspots.json> \
  --gfx1250-performance <gfx1250 performance result.json> \
  --gfx1250-hotspots <gfx1250 hotspots.json> \
  --output-dir toy_e2e/results/arch_compare_20260923_bce6d20a
```
