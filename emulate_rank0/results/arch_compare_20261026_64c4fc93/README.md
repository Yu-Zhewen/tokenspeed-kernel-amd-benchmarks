# Emulated rank 0 with speculative decoding: MI355X vs MI455X at `64c4fc93`

One GPU per architecture serves global rank 0 of each model's CI layout
with `tokenspeed serve --emulate-rank-zero`, dummy weights, uniform MoE
routing and a fixed simulated accept length. Collectives are local
substitutes, so the numbers are a compute estimate, not serving
performance. Both architectures ran the same tree and workload.

## Provenance

| Field | Value |
|---|---|
| TokenSpeed revision | `64c4fc93d5a407cba17ce50df1bbefe01f91ad98` |
| Commit date (UTC) | 2026-10-26 |
| Changes on top | 0 files, identical on both hosts |
| Measured | 2026-10-26 |
| Prompt / output tokens | 50,000 / 500 |
| Concurrencies | 1, 16 |
| Warmup / measured waves | 1 / 1 |
| MI355X | `gfx950:sramecc+:xnack-`, 2.14.0+rocm7.2, `zhewenyu/emulate-rank0:64c4fc93-gfx950` |
| MI455X | `gfx1250`, 2.14.0+rocm10.2.0a20260923, `tokenspeed-gfx1250:64c4fc93-torch214-tt3910-hip` |
| tokenspeed-triton | 3.9.10.post20261008 / 3.9.10.post20261008 |

## Simulated acceptance

Logged values are medians over the server's 40-step decode windows;
windows that straddle a prefill keep fewer tokens. See
`emulate_rank0/README.md` for where each simulated value comes from.

| Model | Layout emulated | Simulated | MI355X logged | MI455X logged |
|---|---|---:|---|---|
| GLM-5.3-Flash MTP | TP4 | 2.9 | 2.90 (18 windows) | 2.90 (18 windows) |
| DeepSeek-V4.1-Flash DSPARK | TP4 | 3.9 | 3.90 (13 windows) | 3.90 (13 windows) |
| Kimi-K3 EAGLE3 | TP8 | 3.75 | 3.75 (14 windows) | 3.75 (14 windows) |

Latency rows are lower-is-better and throughput rows higher-is-better;
the last column is always MI455X's advantage. TPOT p50 is per request
over its whole decode, as CI reports it. At batch 16 every request
arrives at once, so a request whose prompt finishes early decodes
between other requests' prefill chunks, and its TPOT mostly measures
prefill. Steady decode TPOT is the median over the server's 40-step
decode windows in which every request was decoding and no prefill ran;
decode tok/s per user and the decode columns below use it. Both include
the speculative speedup.

## GLM-5.3-Flash MTP

| Metric | Batch | MI355X (gfx950) | MI455X (gfx1250) | MI455X advantage |
|---|---:|---:|---:|---:|
| TTFT p50 (ms) | 1 | 1,556.4 | 1,098.6 | 1.42x |
| TPOT p50 (ms) | 1 | 4.1 | 2.2 | 1.85x |
| steady decode TPOT (ms) | 1 | 4.1 | 2.2 | 1.87x |
| decode tok/s per user | 1 | 245.6 | 459.9 | 1.87x |
| aggregate output tok/s | 1 | 139.0 | 227.4 | 1.64x |
| TTFT p50 (ms) | 16 | 12,774.6 | 8,883.0 | 1.44x |
| TPOT p50 (ms) | 16 | 30.7 | 19.6 | 1.57x |
| steady decode TPOT (ms) | 16 | 8.2 | 3.9 | 2.13x |
| decode tok/s per user | 16 | 121.2 | 258.3 | 2.13x |
| aggregate output tok/s | 16 | 284.6 | 428.7 | 1.51x |

Accumulated GPU kernel time per bucket, MI355X over MI455X, so
above 1.00x means MI455X is ahead. The last row compares the
measured TTFT p50 and steady decode TPOT from the table above; it
is not a total of the rows.

| Category | prefill c1 | prefill c16 | decode c1 | decode c16 |
|---|---:|---:|---:|---:|
| KDA state scan | 1.42x | 1.45x | — | — |
| KDA other | 1.85x | 1.87x | 1.53x | 1.55x |
| MoE | 1.44x | 1.41x | 1.67x | 2.61x |
| index top-k | 0.71x | 0.71x | 0.75x | 0.74x |
| index scoring | 0.86x | 0.88x | 1.52x | 2.81x |
| sparse attention | 3.20x | 3.20x | 5.78x | 1.26x |
| dense GEMM | 1.19x | 1.13x | 1.51x | 1.54x |
| attention | 1.31x | 1.24x | 0.99x | 1.14x |
| hyper-connections | 2.64x | 2.66x | 1.50x | 1.36x |
| quantize | 1.73x | 1.73x | 1.11x | 1.76x |
| rmsnorm | 1.67x | 1.71x | 1.06x | 1.10x |
| top-k and sort | 1.28x | 1.26x | — | — |
| sampling | 2.01x | 2.07x | 1.46x | 1.29x |
| elementwise | 1.41x | 1.41x | 1.23x | 1.58x |
| other | 2.13x | 2.16x | 1.09x | 1.42x |
| all kernels | 1.52x | 1.50x | 1.54x | 1.91x |
| **End-to-end** | **1.42x** (TTFT) | **1.44x** (TTFT) | **1.87x** (steady TPOT) | **2.13x** (steady TPOT) |

<details><summary>MI355X (gfx950) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | sparse attention | `_dsa_dense_mfma_kv_kernel` | 228.209 | 86 | 2,653.6 | 15.68% |
| 2 | MoE | `gluon_fp8_packed_large_stage1_gfx950` | 192.956 | 294 | 656.3 | 13.25% |
| 3 | MoE | `gluon_fp8_packed_large_stage2_gfx950` | 153.029 | 294 | 520.5 | 10.51% |
| 4 | hyper-connections | `gluon_mhc_prefill_project_gfx950` | 107.320 | 630 | 170.3 | 7.37% |
| 5 | index scoring | `gluon_kpool_prefill_topk_fp8_plan_gfx950` | 78.358 | 120 | 653.0 | 5.38% |
| 6 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT208x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 70.824 | 204 | 347.2 | 4.87% |
| 7 | dense GEMM | `gluon_mm_mxfp8_gfx950` | 70.109 | 895 | 78.3 | 4.82% |
| 8 | hyper-connections | `gluon_mhc_post_gfx950` | 65.354 | 630 | 103.7 | 4.49% |
| 9 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950` | 63.127 | 238 | 265.2 | 4.34% |
| 10 | index top-k | `_dsa_oneblock_manual_radix_topk_kernel` | 60.745 | 146 | 416.1 | 4.17% |

</details>

<details><summary>MI355X (gfx950) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | sparse attention | `_dsa_dense_mfma_kv_kernel` | 3,642.076 | 1,208 | 3,015.0 | 15.91% |
| 2 | MoE | `gluon_fp8_packed_large_stage1_gfx950` | 2,979.606 | 4,116 | 723.9 | 13.02% |
| 3 | MoE | `gluon_fp8_packed_large_stage2_gfx950` | 2,398.650 | 4,116 | 582.8 | 10.48% |
| 4 | hyper-connections | `gluon_mhc_prefill_project_gfx950` | 1,687.399 | 8,820 | 191.3 | 7.37% |
| 5 | index scoring | `gluon_kpool_prefill_topk_fp8_plan_gfx950` | 1,225.866 | 1,764 | 694.9 | 5.36% |
| 6 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT208x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 1,150.363 | 3,298 | 348.8 | 5.03% |
| 7 | dense GEMM | `gluon_mm_mxfp8_gfx950` | 1,098.849 | 12,542 | 87.6 | 4.80% |
| 8 | hyper-connections | `gluon_mhc_post_gfx950` | 1,049.194 | 8,820 | 119.0 | 4.58% |
| 9 | index top-k | `_dsa_oneblock_manual_radix_topk_kernel` | 1,033.290 | 2,372 | 435.6 | 4.52% |
| 10 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950` | 997.797 | 3,332 | 299.5 | 4.36% |

</details>

<details><summary>MI355X (gfx950) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | sparse attention | `_dsa_dense_mfma_kv_kernel` | 91.405 | 896 | 102.0 | 11.62% |
| 2 | MoE | `_stage1_fp8_warp_gemv` | 76.160 | 2,880 | 26.4 | 9.68% |
| 3 | hyper-connections | `gluon_mhc_pre_gfx950` | 59.415 | 5,760 | 10.3 | 7.55% |
| 4 | MoE | `_stage2_fp8_warp_gemv` | 55.736 | 2,880 | 19.4 | 7.08% |
| 5 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 53.191 | 5,760 | 9.2 | 6.76% |
| 6 | dense GEMM | `gluon_mm_mxfp8_gfx950` | 35.927 | 4,864 | 7.4 | 4.57% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_D...` | 31.141 | 2,368 | 13.2 | 3.96% |
| 8 | index top-k | `_dsa_oneblock_manual_radix_topk_kernel` | 29.647 | 896 | 33.1 | 3.77% |
| 9 | quantize | `_fp8_token_group_quantize_kernel` | 24.460 | 8,832 | 2.8 | 3.11% |
| 10 | dense GEMM | `gluon_mm_mxfp8_decode_gfx950` | 21.883 | 3,968 | 5.5 | 2.78% |

</details>

<details><summary>MI355X (gfx950) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_fp8_packed_large_stage1_gfx950` | 482.036 | 2,688 | 179.3 | 29.50% |
| 2 | MoE | `gluon_fp8_packed_large_stage2_gfx950` | 302.203 | 2,688 | 112.4 | 18.49% |
| 3 | index scoring | `_kpool_score_dense_mma_kernel` | 77.392 | 896 | 86.4 | 4.74% |
| 4 | hyper-connections | `gluon_mhc_pre_gfx950` | 60.705 | 5,760 | 10.5 | 3.71% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_D...` | 45.768 | 2,176 | 21.0 | 2.80% |
| 6 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 43.733 | 5,760 | 7.6 | 2.68% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 42.911 | 5,120 | 8.4 | 2.63% |
| 8 | quantize | `_fp8_token_group_quantize_kernel` | 42.129 | 11,712 | 3.6 | 2.58% |
| 9 | dense GEMM | `gluon_mm_mxfp8_gfx950` | 40.630 | 4,864 | 8.4 | 2.49% |
| 10 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx950` | 34.868 | 2,176 | 16.0 | 2.13% |

</details>

<details><summary>MI455X (gfx1250) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_fp8_block_down_gfx1250` | 119.935 | 303 | 395.8 | 12.52% |
| 2 | MoE | `_fp8_block_gate_up_gfx1250` | 107.772 | 303 | 355.7 | 11.25% |
| 3 | index top-k | `_dsa_wave32_radix_topk_kernel` | 93.917 | 146 | 643.3 | 9.81% |
| 4 | index scoring | `gluon_kpool_prefill_topk_fp8_plan_gfx1250` | 91.219 | 120 | 760.2 | 9.53% |
| 5 | sparse attention | `_dsa_selected_dense_wmma_kernel` | 71.084 | 86 | 826.6 | 7.42% |
| 6 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT208x256x128_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_AS...` | 53.444 | 204 | 262.0 | 5.58% |
| 7 | dense GEMM | `gluon_mm_fp8_blockscale_largem_gfx1250_kn` | 46.610 | 581 | 80.2 | 4.87% |
| 8 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250` | 44.494 | 238 | 186.9 | 4.65% |
| 9 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 32.519 | 630 | 51.6 | 3.40% |
| 10 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 29.924 | 630 | 47.5 | 3.12% |

</details>

<details><summary>MI455X (gfx1250) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_fp8_block_down_gfx1250` | 1,907.804 | 4,246 | 449.3 | 12.54% |
| 2 | MoE | `_fp8_block_gate_up_gfx1250` | 1,686.739 | 4,246 | 397.3 | 11.09% |
| 3 | index top-k | `_dsa_wave32_radix_topk_kernel` | 1,559.717 | 2,372 | 657.6 | 10.26% |
| 4 | index scoring | `gluon_kpool_prefill_topk_fp8_plan_gfx1250` | 1,402.933 | 1,764 | 795.3 | 9.22% |
| 5 | sparse attention | `_dsa_selected_dense_wmma_kernel` | 1,135.243 | 1,208 | 939.8 | 7.46% |
| 6 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT208x256x128_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_AS...` | 866.115 | 3,298 | 262.6 | 5.70% |
| 7 | dense GEMM | `gluon_mm_fp8_blockscale_largem_gfx1250_kn` | 738.962 | 8,134 | 90.8 | 4.86% |
| 8 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250` | 689.349 | 3,332 | 206.9 | 4.53% |
| 9 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 515.448 | 8,820 | 58.4 | 3.39% |
| 10 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 472.803 | 8,820 | 53.6 | 3.11% |

</details>

<details><summary>MI455X (gfx1250) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_kernel` | 41.800 | 9,408 | 4.4 | 8.17% |
| 2 | index top-k | `_dsa_wave32_radix_topk_kernel` | 40.925 | 896 | 45.7 | 8.00% |
| 3 | MoE | `_fp8_block_gate_up_gfx1250` | 36.595 | 2,880 | 12.7 | 7.15% |
| 4 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 30.393 | 5,760 | 5.3 | 5.94% |
| 5 | dense GEMM | `gluon_mm_fp8_blockscale_gfx1250_bn16_bk2048_buf3_sk1_tdmf0` | 27.087 | 4,864 | 5.6 | 5.29% |
| 6 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1...` | 25.599 | 3,200 | 8.0 | 5.00% |
| 7 | hyper-connections | `gluon_mhc_mixes_reduce_gfx1250` | 25.461 | 5,760 | 4.4 | 4.97% |
| 8 | MoE | `_fp8_block_down_gfx1250` | 23.538 | 2,880 | 8.2 | 4.60% |
| 9 | quantize | `_fp8_token_group_quantize_kernel` | 22.122 | 8,832 | 2.5 | 4.32% |
| 10 | rmsnorm | `_rmsnorm_kernel` | 20.221 | 6,784 | 3.0 | 3.95% |

</details>

<details><summary>MI455X (gfx1250) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_fp8_block_gate_up_gfx1250` | 187.930 | 2,880 | 65.3 | 21.95% |
| 2 | MoE | `_fp8_block_down_gfx1250` | 111.184 | 2,880 | 38.6 | 12.98% |
| 3 | dense GEMM | `_wmma_tdm_dense_kernel` | 69.416 | 10,112 | 6.9 | 8.11% |
| 4 | index top-k | `_dsa_wave32_radix_topk_kernel` | 43.154 | 896 | 48.2 | 5.04% |
| 5 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 29.941 | 5,760 | 5.2 | 3.50% |
| 6 | hyper-connections | `gluon_mhc_mixes_reduce_gfx1250` | 28.396 | 5,760 | 4.9 | 3.32% |
| 7 | sparse attention | `_dsa_selected_dense_wmma_kernel` | 26.827 | 896 | 29.9 | 3.13% |
| 8 | index scoring | `_kpool_score_dense_mma_kernel` | 25.463 | 896 | 28.4 | 2.97% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1...` | 25.396 | 2,944 | 8.6 | 2.97% |
| 10 | quantize | `_fp8_token_group_quantize_kernel` | 23.963 | 8,832 | 2.7 | 2.80% |

</details>

## DeepSeek-V4.1-Flash DSPARK

| Metric | Batch | MI355X (gfx950) | MI455X (gfx1250) | MI455X advantage |
|---|---:|---:|---:|---:|
| TTFT p50 (ms) | 1 | 789.3 | 373.7 | 2.11x |
| TPOT p50 (ms) | 1 | 2.6 | 1.7 | 1.50x |
| steady decode TPOT (ms) | 1 | 2.6 | 1.7 | 1.50x |
| decode tok/s per user | 1 | 385.7 | 578.7 | 1.50x |
| aggregate output tok/s | 1 | 238.9 | 401.6 | 1.68x |
| TTFT p50 (ms) | 16 | 6,554.6 | 3,029.1 | 2.16x |
| TPOT p50 (ms) | 16 | 18.2 | 8.6 | 2.12x |
| steady decode TPOT (ms) | 16 | 6.6 | 3.2 | 2.05x |
| decode tok/s per user | 16 | 152.1 | 312.6 | 2.05x |
| aggregate output tok/s | 16 | 511.5 | 1,093.4 | 2.14x |

Accumulated GPU kernel time per bucket, MI355X over MI455X, so
above 1.00x means MI455X is ahead. The last row compares the
measured TTFT p50 and steady decode TPOT from the table above; it
is not a total of the rows.

| Category | prefill c1 | prefill c16 | decode c1 | decode c16 |
|---|---:|---:|---:|---:|
| MoE | 2.20x | 2.22x | 1.48x | 2.26x |
| index top-k | 0.45x | 0.45x | 0.45x | 0.45x |
| index scoring | 5.27x | 5.29x | 2.47x | 4.76x |
| sparse attention | 1.79x | 1.79x | 1.86x | 1.71x |
| dense GEMM | 1.97x | 1.96x | 1.68x | 1.70x |
| hyper-connections | 1.63x | 1.59x | 1.21x | 1.30x |
| quantize | 2.07x | 2.06x | 2.76x | 3.37x |
| rmsnorm | 1.68x | 1.73x | 1.25x | 1.18x |
| top-k and sort | 1.19x | 1.20x | — | — |
| sampling | 1.81x | 1.83x | 1.50x | 1.37x |
| elementwise | 1.28x | 1.27x | 1.33x | 1.55x |
| other | 1.57x | 1.56x | 1.41x | 1.30x |
| all kernels | 2.08x | 2.08x | 1.30x | 1.84x |
| **End-to-end** | **2.11x** (TTFT) | **2.16x** (TTFT) | **1.50x** (steady TPOT) | **2.05x** (steady TPOT) |

<details><summary>MI355X (gfx950) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | index scoring | `gluon_dsv41_index_topk_gfx950` | 282.625 | 593 | 476.6 | 36.70% |
| 2 | MoE | `gluon_mxfp4_moe_stage1_kernel` | 74.419 | 160 | 465.1 | 9.66% |
| 3 | dense GEMM | `gluon_mm_mxfp8_gfx950` | 68.514 | 737 | 93.0 | 8.90% |
| 4 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel` | 60.679 | 160 | 379.2 | 7.88% |
| 5 | sparse attention | `gluon_dsv4_prefill_gfx950` | 55.183 | 160 | 344.9 | 7.17% |
| 6 | hyper-connections | `gluon_mhc_post_gfx950` | 33.117 | 320 | 103.5 | 4.30% |
| 7 | index top-k | `gluon_dsv41_index_topk_select_gfx950` | 30.556 | 593 | 51.5 | 3.97% |
| 8 | hyper-connections | `gluon_mhc_apply_pre_gfx950` | 20.583 | 327 | 62.9 | 2.67% |
| 9 | hyper-connections | `gluon_mhc_mixes_project_gfx950` | 19.669 | 320 | 61.5 | 2.55% |
| 10 | elementwise | `_rope_inplace_kernel` | 19.102 | 402 | 47.5 | 2.48% |

</details>

<details><summary>MI355X (gfx950) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | index scoring | `gluon_dsv41_index_topk_gfx950` | 4,534.671 | 9,566 | 474.0 | 37.30% |
| 2 | MoE | `gluon_mxfp4_moe_stage1_kernel` | 1,135.218 | 2,280 | 497.9 | 9.34% |
| 3 | dense GEMM | `gluon_mm_mxfp8_gfx950` | 1,063.713 | 10,322 | 103.1 | 8.75% |
| 4 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel` | 950.457 | 2,280 | 416.9 | 7.82% |
| 5 | sparse attention | `gluon_dsv4_prefill_gfx950` | 896.095 | 2,880 | 311.1 | 7.37% |
| 6 | hyper-connections | `gluon_mhc_post_gfx950` | 500.777 | 4,560 | 109.8 | 4.12% |
| 7 | index top-k | `gluon_dsv41_index_topk_select_gfx950` | 492.305 | 9,566 | 51.5 | 4.05% |
| 8 | hyper-connections | `gluon_mhc_apply_pre_gfx950` | 322.902 | 4,658 | 69.3 | 2.66% |
| 9 | hyper-connections | `gluon_mhc_mixes_project_gfx950` | 308.604 | 4,560 | 67.7 | 2.54% |
| 10 | elementwise | `_rope_inplace_kernel` | 302.670 | 5,718 | 52.9 | 2.49% |

</details>

<details><summary>MI355X (gfx950) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `gluon_mm_mxfp8_decode_gfx950` | 138.969 | 14,528 | 9.6 | 19.68% |
| 2 | MoE | `_stage1_mxfp4_direct_mfma_gluon` | 64.157 | 2,560 | 25.1 | 9.09% |
| 3 | dense GEMM | `gluon_mm_a16w16_medium_gfx950` | 56.509 | 5,312 | 10.6 | 8.00% |
| 4 | index top-k | `gluon_dsv41_index_topk_select_gfx950` | 46.616 | 512 | 91.0 | 6.60% |
| 5 | sparse attention | `gluon_dsv41_selected_attention_gfx950` | 43.230 | 2,560 | 16.9 | 6.12% |
| 6 | hyper-connections | `gluon_mhc_mixes_reduce_gfx950` | 40.313 | 5,504 | 7.3 | 5.71% |
| 7 | MoE | `_stage2_mxfp4_direct_mfma_gluon` | 39.890 | 2,560 | 15.6 | 5.65% |
| 8 | hyper-connections | `gluon_mhc_apply_pre_gfx950` | 28.168 | 5,632 | 5.0 | 3.99% |
| 9 | hyper-connections | `gluon_mhc_mixes_project_gfx950` | 26.424 | 5,504 | 4.8 | 3.74% |
| 10 | index scoring | `gluon_dsv41_index_topk_gfx950` | 20.038 | 512 | 39.1 | 2.84% |

</details>

<details><summary>MI355X (gfx950) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel` | 447.446 | 2,752 | 162.6 | 25.71% |
| 2 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel` | 243.611 | 2,752 | 88.5 | 14.00% |
| 3 | dense GEMM | `gluon_mm_mxfp8_decode_gfx950` | 224.999 | 14,336 | 15.7 | 12.93% |
| 4 | index scoring | `gluon_dsv41_index_topk_gfx950` | 199.310 | 512 | 389.3 | 11.45% |
| 5 | sparse attention | `gluon_dsv41_selected_attention_gfx950` | 69.554 | 2,560 | 27.2 | 4.00% |
| 6 | index top-k | `gluon_dsv41_index_topk_select_gfx950` | 47.050 | 512 | 91.9 | 2.70% |
| 7 | hyper-connections | `gluon_mhc_mixes_reduce_gfx950` | 39.510 | 5,504 | 7.2 | 2.27% |
| 8 | dense GEMM | `gluon_mm_a16w16_medium_gfx950` | 38.596 | 2,752 | 14.0 | 2.22% |
| 9 | hyper-connections | `gluon_mhc_mixes_project_gfx950` | 36.667 | 5,504 | 6.7 | 2.11% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA...` | 29.021 | 2,752 | 10.5 | 1.67% |

</details>

<details><summary>MI455X (gfx1250) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | index top-k | `gluon_dsv41_index_topk_select_gfx1250` | 68.485 | 593 | 115.5 | 18.49% |
| 2 | MoE | `_matmul` | 52.340 | 240 | 218.1 | 14.13% |
| 3 | index scoring | `gluon_dsv41_index_topk_gfx1250` | 50.448 | 593 | 85.1 | 13.62% |
| 4 | sparse attention | `gluon_dsv4_prefill_gfx1250` | 30.803 | 160 | 192.5 | 8.31% |
| 5 | dense GEMM | `gluon_mm_mxfp8_ue8m0_largem_bk256_gfx1250_c2` | 25.682 | 512 | 50.2 | 6.93% |
| 6 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 17.630 | 320 | 55.1 | 4.76% |
| 7 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 17.454 | 320 | 54.5 | 4.71% |
| 8 | elementwise | `_rope_inplace_kernel` | 11.217 | 402 | 27.9 | 3.03% |
| 9 | elementwise | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::n...` | 10.047 | 593 | 16.9 | 2.71% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x128_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_AS...` | 7.497 | 120 | 62.5 | 2.02% |

</details>

<details><summary>MI455X (gfx1250) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | index top-k | `gluon_dsv41_index_topk_select_gfx1250` | 1,098.978 | 9,566 | 114.9 | 18.77% |
| 2 | MoE | `_matmul` | 855.024 | 3,920 | 218.1 | 14.61% |
| 3 | index scoring | `gluon_dsv41_index_topk_gfx1250` | 804.181 | 9,566 | 84.1 | 13.74% |
| 4 | sparse attention | `gluon_dsv4_prefill_gfx1250` | 499.248 | 2,880 | 173.4 | 8.53% |
| 5 | dense GEMM | `gluon_mm_mxfp8_ue8m0_largem_bk256_gfx1250_c2` | 413.868 | 8,346 | 49.6 | 7.07% |
| 6 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 281.206 | 4,560 | 61.7 | 4.80% |
| 7 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 274.921 | 4,560 | 60.3 | 4.70% |
| 8 | elementwise | `_rope_inplace_kernel` | 178.075 | 5,718 | 31.1 | 3.04% |
| 9 | elementwise | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::n...` | 157.763 | 9,566 | 16.5 | 2.70% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x128_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_AS...` | 121.388 | 1,940 | 62.6 | 2.07% |

</details>

<details><summary>MI455X (gfx1250) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | index top-k | `gluon_dsv41_index_topk_select_gfx1250` | 110.472 | 512 | 215.8 | 20.36% |
| 2 | MoE | `_matmul_decode` | 60.901 | 5,504 | 11.1 | 11.22% |
| 3 | dense GEMM | `gluon_mm_mxfp8_ue8m0_gfx1250_bn16_bk512_buf3_sk1` | 52.555 | 11,776 | 4.5 | 9.69% |
| 4 | dense GEMM | `_wmma_tdm_dense_kernel` | 31.921 | 6,592 | 4.8 | 5.88% |
| 5 | hyper-connections | `gluon_mhc_mixes_reduce_gfx1250` | 30.728 | 5,504 | 5.6 | 5.66% |
| 6 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 23.815 | 5,504 | 4.3 | 4.39% |
| 7 | sparse attention | `gluon_dsv41_selected_attention_gfx1250` | 22.569 | 2,560 | 8.8 | 4.16% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASE...` | 19.827 | 64 | 309.8 | 3.65% |
| 9 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 15.300 | 5,504 | 2.8 | 2.82% |
| 10 | hyper-connections | `_mhc_pre_layer_norm_hc4_kernel` | 12.802 | 2,752 | 4.7 | 2.36% |

</details>

<details><summary>MI455X (gfx1250) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode` | 312.588 | 5,504 | 56.8 | 33.09% |
| 2 | index top-k | `gluon_dsv41_index_topk_select_gfx1250` | 109.983 | 512 | 214.8 | 11.64% |
| 3 | dense GEMM | `gluon_mm_mxfp8_ue8m0_largem_bk512_gfx1250_t64_c1_sk1` | 100.337 | 14,400 | 7.0 | 10.62% |
| 4 | index scoring | `gluon_dsv41_index_topk_gfx1250` | 40.532 | 512 | 79.2 | 4.29% |
| 5 | sparse attention | `gluon_dsv41_selected_attention_gfx1250` | 38.509 | 2,560 | 15.0 | 4.08% |
| 6 | dense GEMM | `_wmma_tdm_dense_kernel` | 38.115 | 5,696 | 6.7 | 4.03% |
| 7 | hyper-connections | `gluon_mhc_mixes_reduce_gfx1250` | 33.551 | 5,504 | 6.1 | 3.55% |
| 8 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 26.033 | 5,504 | 4.7 | 2.76% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASE...` | 19.935 | 64 | 311.5 | 2.11% |
| 10 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 18.438 | 5,504 | 3.3 | 1.95% |

</details>

## Kimi-K3 EAGLE3

| Metric | Batch | MI355X (gfx950) | MI455X (gfx1250) | MI455X advantage |
|---|---:|---:|---:|---:|
| TTFT p50 (ms) | 1 | 2,485.1 | 1,520.5 | 1.63x |
| TPOT p50 (ms) | 1 | 4.0 | 2.1 | 1.87x |
| steady decode TPOT (ms) | 1 | 3.9 | 2.1 | 1.87x |
| decode tok/s per user | 1 | 253.7 | 474.2 | 1.87x |
| aggregate output tok/s | 1 | 111.9 | 193.7 | 1.73x |
| TTFT p50 (ms) | 16 | 20,123.2 | 11,507.0 | 1.75x |
| TPOT p50 (ms) | 16 | 46.5 | 25.3 | 1.84x |
| steady decode TPOT (ms) | 16 | 11.0 | 4.9 | 2.23x |
| decode tok/s per user | 16 | 90.8 | 202.4 | 2.23x |
| aggregate output tok/s | 16 | 184.7 | 331.2 | 1.79x |

Accumulated GPU kernel time per bucket, MI355X over MI455X, so
above 1.00x means MI455X is ahead. The last row compares the
measured TTFT p50 and steady decode TPOT from the table above; it
is not a total of the rows.

| Category | prefill c1 | prefill c16 | decode c1 | decode c16 |
|---|---:|---:|---:|---:|
| KDA state scan | 1.52x | 1.31x | — | — |
| KDA other | 1.73x | 1.66x | 1.36x | 1.57x |
| MoE | 2.46x | 2.27x | 1.42x | 2.30x |
| input projections | 1.67x | 1.40x | 2.26x | 2.15x |
| dense GEMM | 2.06x | 1.81x | 1.77x | 1.99x |
| attention | 1.67x | 1.52x | 1.48x | 1.91x |
| AttnRes | 1.85x | 1.68x | 1.83x | 1.83x |
| add3 | 3.91x | 2.79x | — | — |
| quantize | 4.29x | 3.03x | 1.65x | 2.72x |
| rmsnorm | 2.57x | 1.84x | 1.03x | 1.40x |
| top-k and sort | 1.11x | 1.10x | — | — |
| sampling | 1.99x | 2.04x | 1.31x | 1.31x |
| elementwise | 1.72x | 1.61x | 3.97x | 2.70x |
| other | 3.19x | 1.81x | 0.94x | 1.09x |
| all kernels | 2.05x | 1.81x | 1.65x | 2.14x |
| **End-to-end** | **1.63x** (TTFT) | **1.75x** (TTFT) | **1.87x** (steady TPOT) | **2.23x** (steady TPOT) |

<details><summary>MI355X (gfx950) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel` | 429.723 | 644 | 667.3 | 16.08% |
| 2 | input projections | `gluon_latent_input_largem_gfx950` | 281.165 | 552 | 509.4 | 10.52% |
| 3 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel` | 277.292 | 644 | 430.6 | 10.38% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 235.032 | 414 | 567.7 | 8.80% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950` | 194.566 | 1,309 | 148.6 | 7.28% |
| 6 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 175.884 | 564 | 311.9 | 6.58% |
| 7 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950` | 151.572 | 1,254 | 120.9 | 5.67% |
| 8 | attention | `gluon_mla_prefill_8wave_gfx950` | 151.357 | 336 | 450.5 | 5.66% |
| 9 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950` | 140.034 | 483 | 289.9 | 5.24% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel` | 101.004 | 552 | 183.0 | 3.78% |

</details>

<details><summary>MI355X (gfx950) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel` | 5,991.026 | 9,016 | 664.5 | 16.22% |
| 2 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel` | 3,987.146 | 9,016 | 442.2 | 10.79% |
| 3 | input projections | `gluon_latent_input_largem_gfx950` | 3,775.942 | 9,016 | 418.8 | 10.22% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 3,494.097 | 6,833 | 511.4 | 9.46% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950` | 2,818.646 | 18,326 | 153.8 | 7.63% |
| 6 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 2,351.994 | 9,118 | 258.0 | 6.37% |
| 7 | attention | `gluon_mla_prefill_8wave_gfx950` | 2,266.623 | 5,784 | 391.9 | 6.14% |
| 8 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950` | 2,172.576 | 20,681 | 105.1 | 5.88% |
| 9 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950` | 1,883.790 | 6,762 | 278.6 | 5.10% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel` | 1,498.399 | 9,016 | 166.2 | 4.06% |

</details>

<details><summary>MI355X (gfx950) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `gluon_gemv_a16w16_gfx950` | 185.181 | 19,456 | 9.5 | 18.67% |
| 2 | MoE | `_warp_decode_precomputed_situ_stage1_kernel` | 113.658 | 5,888 | 19.3 | 11.46% |
| 3 | input projections | `gluon_latent_input_small_batch_gfx950` | 102.949 | 5,888 | 17.5 | 10.38% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx950` | 93.932 | 11,968 | 7.8 | 9.47% |
| 5 | dense GEMM | `gluon_mm_a16w16_medium_gfx950` | 79.154 | 5,952 | 13.3 | 7.98% |
| 6 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel` | 66.361 | 5,888 | 11.3 | 6.69% |
| 7 | MoE | `gluon_sigmoid_bias_topk_gfx950` | 54.466 | 5,888 | 9.3 | 5.49% |
| 8 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx950` | 44.385 | 4,416 | 10.1 | 4.48% |
| 9 | elementwise | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>,...` | 30.540 | 6,016 | 5.1 | 3.08% |
| 10 | attention | `gluon_mla_decode_fp8_query_blocks_gfx950` | 28.032 | 1,536 | 18.3 | 2.83% |

</details>

<details><summary>MI355X (gfx950) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_warp_decode_precomputed_situ_sorted_stage1_kernel` | 809.831 | 5,888 | 137.5 | 29.48% |
| 2 | MoE | `_warp_decode_sorted_stage2_fp8_mxfp4_kernel` | 523.864 | 5,888 | 89.0 | 19.07% |
| 3 | dense GEMM | `gluon_mm_a16w16_medium_gfx950` | 164.906 | 15,104 | 10.9 | 6.00% |
| 4 | attention | `gluon_mla_decode_fp8_query_blocks_gfx950` | 164.200 | 1,536 | 106.9 | 5.98% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT160x64x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 145.726 | 4,416 | 33.0 | 5.30% |
| 6 | input projections | `gluon_latent_input_small_batch_gfx950` | 133.380 | 5,888 | 22.7 | 4.85% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_D...` | 116.557 | 5,952 | 19.6 | 4.24% |
| 8 | AttnRes | `gluon_attn_res_fwd_gfx950` | 114.510 | 11,968 | 9.6 | 4.17% |
| 9 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx950` | 66.744 | 4,416 | 15.1 | 2.43% |
| 10 | MoE | `gluon_sigmoid_bias_topk_gfx950` | 60.051 | 5,888 | 10.2 | 2.19% |

</details>

<details><summary>MI455X (gfx1250) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_warp_pipeline_kernel` | 304.328 | 2,856 | 106.6 | 23.33% |
| 2 | MoE | `_matmul` | 254.088 | 1,104 | 230.2 | 19.48% |
| 3 | input projections | `gluon_latent_input_largem_gfx1250` | 169.581 | 552 | 307.2 | 13.00% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 105.447 | 1,309 | 80.6 | 8.08% |
| 5 | attention | `gluon_mla_prefill_gfx1250` | 92.382 | 360 | 256.6 | 7.08% |
| 6 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250` | 92.370 | 483 | 191.2 | 7.08% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250` | 37.607 | 483 | 77.9 | 2.88% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel` | 35.791 | 644 | 55.6 | 2.74% |
| 9 | MoE | `_matmul_decode` | 16.876 | 184 | 91.7 | 1.29% |
| 10 | add3 | `_add3_kernel` | 16.479 | 644 | 25.6 | 1.26% |

</details>

<details><summary>MI455X (gfx1250) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_warp_pipeline_kernel` | 4,961.814 | 46,704 | 106.2 | 24.35% |
| 2 | MoE | `_matmul` | 4,147.327 | 18,032 | 230.0 | 20.36% |
| 3 | input projections | `gluon_latent_input_largem_gfx1250` | 2,770.180 | 9,016 | 307.3 | 13.60% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 1,677.657 | 18,326 | 91.5 | 8.23% |
| 5 | attention | `gluon_mla_prefill_gfx1250` | 1,521.988 | 5,856 | 259.9 | 7.47% |
| 6 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250` | 1,438.767 | 6,762 | 212.8 | 7.06% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250` | 583.559 | 6,762 | 86.3 | 2.86% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel` | 569.196 | 9,016 | 63.1 | 2.79% |
| 9 | add3 | `_add3_kernel` | 261.399 | 9,016 | 29.0 | 1.28% |
| 10 | MoE | `gluon_sigmoid_bias_topk_gfx1250` | 251.521 | 9,016 | 27.9 | 1.23% |

</details>

<details><summary>MI455X (gfx1250) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_kernel` | 148.866 | 25,600 | 5.8 | 24.81% |
| 2 | MoE | `_matmul_decode` | 111.161 | 11,776 | 9.4 | 18.52% |
| 3 | input projections | `gluon_latent_input_decode_gfx1250` | 53.100 | 5,888 | 9.0 | 8.85% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 51.189 | 11,968 | 4.3 | 8.53% |
| 5 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx1250` | 33.001 | 4,416 | 7.5 | 5.50% |
| 6 | MoE | `_kimi3_sigmoid_bias_topk_kernel` | 24.304 | 5,888 | 4.1 | 4.05% |
| 7 | attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_NONE_TILE_SIZE_64_KV_LORA_RANK_5...` | 22.506 | 1,536 | 14.7 | 3.75% |
| 8 | MoE | `_precomputed_topk_route_small_m_gfx1250_kernel` | 21.307 | 5,888 | 3.6 | 3.55% |
| 9 | rmsnorm | `_rmsnorm_kernel` | 19.081 | 6,272 | 3.0 | 3.18% |
| 10 | MoE | `_weighted_topk_reduce_gfx1250_kernel` | 18.134 | 5,888 | 3.1 | 3.02% |

</details>

<details><summary>MI455X (gfx1250) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode` | 579.406 | 11,776 | 49.2 | 45.02% |
| 2 | dense GEMM | `_wmma_tdm_dense_kernel` | 217.428 | 26,496 | 8.2 | 16.90% |
| 3 | attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_NONE_TILE_SIZE_64_KV_LORA_RANK_5...` | 82.078 | 1,536 | 53.4 | 6.38% |
| 4 | input projections | `gluon_latent_input_decode_nosplit_gfx1250` | 78.923 | 5,888 | 13.4 | 6.13% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 62.559 | 11,968 | 5.2 | 4.86% |
| 6 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx1250` | 44.229 | 4,416 | 10.0 | 3.44% |
| 7 | MoE | `_kimi3_sigmoid_bias_topk_kernel` | 26.266 | 5,888 | 4.5 | 2.04% |
| 8 | MoE | `_precomputed_topk_route_small_m_gfx1250_kernel` | 23.625 | 5,888 | 4.0 | 1.84% |
| 9 | rmsnorm | `_rmsnorm_kernel` | 19.292 | 6,272 | 3.1 | 1.50% |
| 10 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_...` | 16.172 | 3,456 | 4.7 | 1.26% |

</details>

## Method and limitations

- Timing and profiling are separate server runs. Performance comes from
  a run with decode CUDA graphs; kernels come from an eager run without
  overlap scheduling, under rocprofv3.
- Emulation omits communication time, so multi-GPU collectives and
  their overlap with compute are not measured.
- Dummy weights with uniform routing give every expert equal load, which
  real routing does not.
- Accept lengths are fixed, so these runs compare the cost of a verify
  step at a given acceptance, not drafter quality.
- A `—` means one architecture has no kernel in that bucket for that
  stage, a difference in how the work is split rather than zero time.

Regenerate this document with:

```bash
python3 emulate_rank0/scripts/generate_arch_comparison.py \
  --gfx950 <MI355X result dir> --gfx1250 <MI455X result dir> \
  --commit-date 2026-10-26 \
  --output-dir emulate_rank0/results/arch_compare_20261026_64c4fc93
```
