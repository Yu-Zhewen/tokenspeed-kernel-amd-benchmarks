# Emulated rank 0 with speculative decoding: MI355X vs MI455X at `927562ef`

One GPU per architecture serves global rank 0 of each model's CI layout
with `tokenspeed serve --emulate-rank-zero`, dummy weights, uniform MoE
routing and a fixed simulated accept length. Collectives are local
substitutes, so the numbers are a compute estimate, not serving
performance. Both architectures ran the same tree and workload.

## Provenance

| Field | Value |
|---|---|
| TokenSpeed revision | `927562ef44c275d13aa87448d43a45d3e7ec68ec` |
| Commit date (UTC) | 2026-10-03 |
| Changes on top | 7 files, identical on both hosts |
| Measured | 2026-10-03 to 2026-10-07 |
| Prompt / output tokens | 50,000 / 500 |
| Concurrencies | 1, 16 |
| Warmup / measured waves | 1 / 1 |
| MI355X | `gfx950:sramecc+:xnack-`, 2.14.0+rocm7.2, `zhewenyu/emulate-rank0:927562ef-pr1936-gfx950` |
| MI455X | `gfx1250`, 2.14.0+rocm10.2.0a20260923, `tokenspeed-gfx1250:d36f9bc8-torch214` |
| tokenspeed-triton | 3.8.10.post20260920 / 3.8.10.post20260920 |

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
| TTFT p50 (ms) | 1 | 2,389.4 | 2,584.2 | 0.92x |
| TPOT p50 (ms) | 1 | 5.3 | 5.5 | 0.96x |
| steady decode TPOT (ms) | 1 | 5.2 | 5.4 | 0.96x |
| decode tok/s per user | 1 | 192.6 | 185.4 | 0.96x |
| aggregate output tok/s | 1 | 99.7 | 94.2 | 0.95x |
| TTFT p50 (ms) | 16 | 19,708.7 | 20,012.6 | 0.98x |
| TPOT p50 (ms) | 16 | 44.9 | 51.8 | 0.87x |
| steady decode TPOT (ms) | 16 | 9.8 | 16.4 | 0.60x |
| decode tok/s per user | 16 | 101.8 | 61.0 | 0.60x |
| aggregate output tok/s | 16 | 189.8 | 174.4 | 0.92x |

Accumulated GPU kernel time per bucket, MI355X over MI455X, so
above 1.00x means MI455X is ahead. The last row compares the
measured TTFT p50 and steady decode TPOT from the table above; it
is not a total of the rows.

| Category | prefill c1 | prefill c16 | decode c1 | decode c16 |
|---|---:|---:|---:|---:|
| KDA state scan | 1.61x | 1.64x | — | — |
| KDA other | 1.67x | 1.68x | 1.17x | 1.52x |
| MoE | 0.51x | 0.50x | 0.42x | 0.42x |
| sparse attention | 0.79x | 0.97x | 4.84x | 1.30x |
| dense GEMM | 2.37x | 2.40x | 1.89x | 0.96x |
| attention | 1.25x | 1.27x | 1.02x | 1.23x |
| hyper-connections | 2.42x | 2.43x | 0.66x | 0.63x |
| quantize | 1.47x | 1.51x | 1.42x | 1.63x |
| rmsnorm | 1.71x | 1.74x | 1.02x | 1.17x |
| top-k and sort | 0.03x | 0.03x | 0.80x | 0.52x |
| sampling | 1.28x | 1.31x | 1.30x | 1.74x |
| elementwise | 1.01x | 1.02x | 0.49x | 0.56x |
| other | 4.93x | 5.29x | 1.05x | 3.37x |
| all kernels | 0.93x | 1.00x | 0.95x | 0.59x |
| **End-to-end** | **0.92x** (TTFT) | **0.98x** (TTFT) | **0.96x** (steady TPOT) | **0.60x** (steady TPOT) |

<details><summary>MI355X (gfx950) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | sparse attention | `_dsa_dense_mfma_kv_kernel` | 381.154 | 86 | 4,432.0 | 16.63% |
| 2 | dense GEMM | `_w8a8_block_fp8_matmul` | 330.368 | 906 | 364.6 | 14.41% |
| 3 | MoE | `gluon_bf16_moe_stage1_kernel` | 323.693 | 294 | 1,101.0 | 14.12% |
| 4 | sparse attention | `_hadamard_128_kernel` | 182.547 | 86 | 2,122.6 | 7.96% |
| 5 | hyper-connections | `_mhc_prefill_project_hc4_kernel` | 159.810 | 630 | 253.7 | 6.97% |
| 6 | MoE | `gluon_bf16_moe_stage2_kernel` | 138.059 | 294 | 469.6 | 6.02% |
| 7 | sparse attention | `gluon_kpool_prefill_topk_fp8_plan_gfx950` | 104.814 | 120 | 873.4 | 4.57% |
| 8 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950` | 78.360 | 272 | 288.1 | 3.42% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT208x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 70.115 | 204 | 343.7 | 3.06% |
| 10 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 66.612 | 630 | 105.7 | 2.91% |

</details>

<details><summary>MI355X (gfx950) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | sparse attention | `_dsa_dense_mfma_kv_kernel` | 6,076.412 | 1,208 | 5,030.1 | 16.78% |
| 2 | dense GEMM | `_w8a8_block_fp8_matmul` | 5,291.126 | 12,704 | 416.5 | 14.61% |
| 3 | MoE | `gluon_bf16_moe_stage1_kernel` | 4,970.538 | 4,116 | 1,207.6 | 13.72% |
| 4 | sparse attention | `_hadamard_128_kernel` | 2,921.409 | 1,208 | 2,418.4 | 8.07% |
| 5 | hyper-connections | `_mhc_prefill_project_hc4_kernel` | 2,515.150 | 8,820 | 285.2 | 6.94% |
| 6 | MoE | `gluon_bf16_moe_stage2_kernel` | 2,147.670 | 4,116 | 521.8 | 5.93% |
| 7 | sparse attention | `gluon_kpool_prefill_topk_fp8_plan_gfx950` | 1,657.827 | 1,764 | 939.8 | 4.58% |
| 8 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950` | 1,227.390 | 3,876 | 316.7 | 3.39% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT208x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 1,143.415 | 3,332 | 343.2 | 3.16% |
| 10 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 1,068.981 | 8,820 | 121.2 | 2.95% |

</details>

<details><summary>MI355X (gfx950) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_w8a8_block_fp8_matmul` | 163.425 | 8,832 | 18.5 | 15.50% |
| 2 | sparse attention | `_dsa_dense_mfma_kv_kernel` | 158.590 | 896 | 177.0 | 15.05% |
| 3 | MoE | `_stage1_fp8_warp_gemv` | 83.852 | 2,880 | 29.1 | 7.95% |
| 4 | MoE | `_stage2_fp8_warp_gemv` | 58.402 | 2,880 | 20.3 | 5.54% |
| 5 | hyper-connections | `gluon_mhc_pre_gfx950` | 58.121 | 5,760 | 10.1 | 5.51% |
| 6 | top-k and sort | `void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>(at::cuda::detail::TensorInfo<float const, unsigned...` | 56.194 | 768 | 73.2 | 5.33% |
| 7 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 55.101 | 5,760 | 9.6 | 5.23% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_D...` | 31.849 | 2,368 | 13.4 | 3.02% |
| 9 | quantize | `_per_token_group_quant_8bit` | 27.591 | 8,832 | 3.1 | 2.62% |
| 10 | KDA other | `fused_recurrent_kda_mtp_fwd_kernel` | 23.284 | 2,176 | 10.7 | 2.21% |

</details>

<details><summary>MI355X (gfx950) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_bf16_moe_stage1_kernel` | 613.838 | 2,688 | 228.4 | 32.88% |
| 2 | MoE | `gluon_bf16_moe_stage2_kernel` | 243.248 | 2,688 | 90.5 | 13.03% |
| 3 | dense GEMM | `_w8a8_block_fp8_matmul` | 157.511 | 8,832 | 17.8 | 8.44% |
| 4 | sparse attention | `_kpool_score_dense_mma_kernel` | 77.421 | 896 | 86.4 | 4.15% |
| 5 | hyper-connections | `gluon_mhc_pre_gfx950` | 60.699 | 5,760 | 10.5 | 3.25% |
| 6 | KDA other | `fused_recurrent_kda_mtp_fwd_kernel` | 49.588 | 2,176 | 22.8 | 2.66% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_D...` | 45.336 | 2,176 | 20.8 | 2.43% |
| 8 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 43.703 | 5,760 | 7.6 | 2.34% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 43.253 | 5,120 | 8.4 | 2.32% |
| 10 | sparse attention | `_dsa_dense_mfma_kv_kernel` | 41.627 | 896 | 46.5 | 2.23% |

</details>

<details><summary>MI455X (gfx1250) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_stage1_kernel` | 505.376 | 303 | 1,667.9 | 20.54% |
| 2 | MoE | `_stage2_kernel` | 453.228 | 303 | 1,495.8 | 18.42% |
| 3 | sparse attention | `gluon_kpool_prefill_topk_fp8_plan_gfx1250` | 373.520 | 120 | 3,112.7 | 15.18% |
| 4 | sparse attention | `_dsa_selected_dense_wmma_kernel` | 324.844 | 86 | 3,777.3 | 13.20% |
| 5 | sparse attention | `_dsa_wave32_radix_topk_kernel` | 119.298 | 144 | 828.5 | 4.85% |
| 6 | sparse attention | `_hadamard_128_kernel` | 107.577 | 86 | 1,250.9 | 4.37% |
| 7 | dense GEMM | `_w8a8_block_fp8_matmul` | 86.098 | 875 | 98.4 | 3.50% |
| 8 | hyper-connections | `_mhc_prefill_project_hc4_kernel` | 60.062 | 630 | 95.3 | 2.44% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT208x256x128_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_AS...` | 50.937 | 204 | 249.7 | 2.07% |
| 10 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250` | 48.713 | 272 | 179.1 | 1.98% |

</details>

<details><summary>MI455X (gfx1250) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_stage1_kernel` | 7,894.823 | 4,246 | 1,859.4 | 21.90% |
| 2 | MoE | `_stage2_kernel` | 7,100.423 | 4,246 | 1,672.3 | 19.69% |
| 3 | sparse attention | `_dsa_selected_dense_wmma_kernel` | 5,188.387 | 1,208 | 4,295.0 | 14.39% |
| 4 | sparse attention | `gluon_kpool_prefill_topk_fp8_plan_gfx1250` | 3,177.212 | 1,764 | 1,801.1 | 8.81% |
| 5 | sparse attention | `_dsa_wave32_radix_topk_kernel` | 1,964.641 | 2,340 | 839.6 | 5.45% |
| 6 | sparse attention | `_hadamard_128_kernel` | 1,721.385 | 1,208 | 1,425.0 | 4.77% |
| 7 | dense GEMM | `_w8a8_block_fp8_matmul` | 1,361.353 | 12,250 | 111.1 | 3.78% |
| 8 | hyper-connections | `_mhc_prefill_project_hc4_kernel` | 950.283 | 8,820 | 107.7 | 2.64% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT208x256x128_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_AS...` | 829.714 | 3,332 | 249.0 | 2.30% |
| 10 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250` | 750.490 | 3,876 | 193.6 | 2.08% |

</details>

<details><summary>MI455X (gfx1250) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_stage1_kernel` | 198.842 | 2,880 | 69.0 | 17.99% |
| 2 | MoE | `_stage2_kernel` | 173.638 | 2,880 | 60.3 | 15.71% |
| 3 | hyper-connections | `_mhc_pre_mix_triton_kernel` | 131.846 | 5,760 | 22.9 | 11.93% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1...` | 80.942 | 8,256 | 9.8 | 7.32% |
| 5 | top-k and sort | `void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>(at::cuda::detail::TensorInfo<float const, unsigned...` | 69.790 | 3,456 | 20.2 | 6.31% |
| 6 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 40.070 | 5,760 | 7.0 | 3.63% |
| 7 | dense GEMM | `_wmma_tdm_dense_m16_kernel` | 32.361 | 6,720 | 4.8 | 2.93% |
| 8 | dense GEMM | `gluon_mm_fp8_blockscale_gfx1250_bn16_bk2048_buf3_sk1_tdmf0` | 26.473 | 4,864 | 5.4 | 2.40% |
| 9 | rmsnorm | `_rmsnorm_kernel` | 22.048 | 6,784 | 3.3 | 1.99% |
| 10 | quantize | `_per_token_group_quant_8bit` | 19.493 | 8,832 | 2.2 | 1.76% |

</details>

<details><summary>MI455X (gfx1250) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_stage1_kernel` | 1,203.191 | 2,880 | 417.8 | 37.79% |
| 2 | MoE | `_stage2_kernel` | 943.987 | 2,880 | 327.8 | 29.65% |
| 3 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1...` | 143.704 | 9,344 | 15.4 | 4.51% |
| 4 | hyper-connections | `_mhc_pre_mix_triton_kernel` | 142.085 | 5,760 | 24.7 | 4.46% |
| 5 | dense GEMM | `_w8a8_block_fp8_matmul` | 119.632 | 8,000 | 15.0 | 3.76% |
| 6 | sparse attention | `_dsa_selected_dense_wmma_kernel` | 66.803 | 896 | 74.6 | 2.10% |
| 7 | top-k and sort | `void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>(at::cuda::detail::TensorInfo<float const, unsigned...` | 49.343 | 3,008 | 16.4 | 1.55% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT80x64x256_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM...` | 38.778 | 2,176 | 17.8 | 1.22% |
| 9 | KDA other | `fused_recurrent_kda_mtp_fwd_kernel` | 27.101 | 2,176 | 12.5 | 0.85% |
| 10 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 26.691 | 5,760 | 4.6 | 0.84% |

</details>

## DeepSeek-V4.1-Flash DSPARK

| Metric | Batch | MI355X (gfx950) | MI455X (gfx1250) | MI455X advantage |
|---|---:|---:|---:|---:|
| TTFT p50 (ms) | 1 | 1,568.4 | 2,291.0 | 0.68x |
| TPOT p50 (ms) | 1 | 6.6 | 10.2 | 0.65x |
| steady decode TPOT (ms) | 1 | 6.5 | 10.1 | 0.65x |
| decode tok/s per user | 1 | 153.0 | 99.5 | 0.65x |
| aggregate output tok/s | 1 | 102.6 | 67.9 | 0.66x |
| TTFT p50 (ms) | 16 | 12,966.3 | 18,935.1 | 0.68x |
| TPOT p50 (ms) | 16 | 37.4 | 66.0 | 0.57x |
| steady decode TPOT (ms) | 16 | 14.0 | 31.9 | 0.44x |
| decode tok/s per user | 16 | 71.2 | 31.3 | 0.44x |
| aggregate output tok/s | 16 | 252.9 | 154.1 | 0.61x |

Accumulated GPU kernel time per bucket, MI355X over MI455X, so
above 1.00x means MI455X is ahead. The last row compares the
measured TTFT p50 and steady decode TPOT from the table above; it
is not a total of the rows.

| Category | prefill c1 | prefill c16 | decode c1 | decode c16 |
|---|---:|---:|---:|---:|
| MoE | 0.18x | 0.18x | 0.23x | 0.18x |
| sparse attention | 1.33x | 1.33x | 1.49x | 1.48x |
| dense GEMM | 0.61x | 0.59x | 0.58x | 0.49x |
| hyper-connections | 2.20x | 2.21x | 1.19x | 1.19x |
| quantize | 2.02x | 2.04x | 1.86x | 1.85x |
| rmsnorm | 1.42x | 1.42x | 1.23x | 1.38x |
| top-k and sort | 0.61x | 0.62x | 1.46x | 1.42x |
| sampling | 1.25x | 1.26x | 1.54x | 2.86x |
| elementwise | 1.91x | 1.92x | 1.36x | 3.35x |
| other | 1.67x | 1.65x | 1.17x | 2.01x |
| all kernels | 0.69x | 0.69x | 0.65x | 0.45x |
| **End-to-end** | **0.68x** (TTFT) | **0.68x** (TTFT) | **0.65x** (steady TPOT) | **0.44x** (steady TPOT) |

<details><summary>MI355X (gfx950) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | sparse attention | `gluon_dsv41_index_topk_gfx950` | 281.925 | 593 | 475.4 | 18.06% |
| 2 | dense GEMM | `_w8a8_block_fp8_matmul` | 279.447 | 452 | 618.2 | 17.90% |
| 3 | sparse attention | `gluon_dsv4_prefill_gfx950` | 212.609 | 160 | 1,328.8 | 13.62% |
| 4 | MoE | `_pipelined_moe_kernel_scaled_block_schedule` | 171.183 | 320 | 534.9 | 10.96% |
| 5 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 128.467 | 320 | 401.5 | 8.23% |
| 6 | quantize | `_fp8_group32_ue8m0_quantize` | 100.998 | 842 | 120.0 | 6.47% |
| 7 | dense GEMM | `gluon_mm_mxfp8_gfx950` | 45.448 | 390 | 116.5 | 2.91% |
| 8 | top-k and sort | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<f...` | 37.099 | 2,376 | 15.6 | 2.38% |
| 9 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 35.427 | 320 | 110.7 | 2.27% |
| 10 | elementwise | `_rope_inplace_kernel` | 18.118 | 402 | 45.1 | 1.16% |

</details>

<details><summary>MI355X (gfx950) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | sparse attention | `gluon_dsv41_index_topk_gfx950` | 4,507.116 | 9,583 | 470.3 | 18.27% |
| 2 | dense GEMM | `_w8a8_block_fp8_matmul` | 4,281.080 | 5,682 | 753.4 | 17.36% |
| 3 | sparse attention | `gluon_dsv4_prefill_gfx950` | 3,425.482 | 2,920 | 1,173.1 | 13.89% |
| 4 | MoE | `_pipelined_moe_kernel_scaled_block_schedule` | 2,615.552 | 4,560 | 573.6 | 10.60% |
| 5 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 2,061.703 | 4,560 | 452.1 | 8.36% |
| 6 | quantize | `_fp8_group32_ue8m0_quantize` | 1,618.405 | 12,002 | 134.8 | 6.56% |
| 7 | dense GEMM | `gluon_mm_mxfp8_gfx950` | 737.175 | 6,320 | 116.6 | 2.99% |
| 8 | top-k and sort | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<f...` | 587.700 | 36,300 | 16.2 | 2.38% |
| 9 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 566.798 | 4,560 | 124.3 | 2.30% |
| 10 | elementwise | `_rope_inplace_kernel` | 288.993 | 5,718 | 50.5 | 1.17% |

</details>

<details><summary>MI355X (gfx950) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_w8a8_block_fp8_matmul` | 540.376 | 14,528 | 37.2 | 31.07% |
| 2 | sparse attention | `gluon_dsv41_selected_attention_gfx950` | 368.223 | 2,560 | 143.8 | 21.17% |
| 3 | MoE | `_pipelined_moe_kernel_scaled_block_schedule` | 180.982 | 5,504 | 32.9 | 10.41% |
| 4 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 54.939 | 5,504 | 10.0 | 3.16% |
| 5 | hyper-connections | `_mhc_pre_mix_hc4_kernel` | 50.497 | 5,504 | 9.2 | 2.90% |
| 6 | quantize | `_fp8_group32_ue8m0_quantize` | 44.967 | 14,528 | 3.1 | 2.59% |
| 7 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_...` | 36.610 | 8,512 | 4.3 | 2.11% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0...` | 34.173 | 2,752 | 12.4 | 1.97% |
| 9 | top-k and sort | `void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>(at::cuda::detail::TensorInfo<float const, unsigned...` | 31.323 | 320 | 97.9 | 1.80% |
| 10 | MoE | `_fused_precomputed_topk_route_small_m` | 28.763 | 2,752 | 10.5 | 1.65% |

</details>

<details><summary>MI355X (gfx950) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_pipelined_moe_kernel_scaled_block_schedule` | 1,012.832 | 5,504 | 184.0 | 27.54% |
| 2 | sparse attention | `gluon_dsv41_selected_attention_gfx950` | 744.874 | 2,560 | 291.0 | 20.25% |
| 3 | dense GEMM | `_w8a8_block_fp8_matmul` | 643.055 | 14,528 | 44.3 | 17.48% |
| 4 | sparse attention | `gluon_dsv41_index_topk_gfx950` | 192.574 | 512 | 376.1 | 5.24% |
| 5 | quantize | `_fp8_group32_ue8m0_quantize` | 71.976 | 14,528 | 5.0 | 1.96% |
| 6 | top-k and sort | `void at::native::radixSortKVInPlace<-2, -1, 128, 8, long, long, unsigned int>(at::cuda::detail::TensorInfo<long, unsi...` | 67.520 | 2,752 | 24.5 | 1.84% |
| 7 | elementwise | `void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<true, in...` | 66.651 | 2,752 | 24.2 | 1.81% |
| 8 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 65.226 | 5,504 | 11.9 | 1.77% |
| 9 | hyper-connections | `_mhc_pre_mix_hc4_kernel` | 58.628 | 5,504 | 10.7 | 1.59% |
| 10 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_...` | 47.799 | 9,728 | 4.9 | 1.30% |

</details>

<details><summary>MI455X (gfx1250) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_stage1_kernel` | 544.100 | 160 | 3,400.6 | 23.93% |
| 2 | dense GEMM | `_w8a8_block_fp8_matmul` | 509.210 | 842 | 604.8 | 22.39% |
| 3 | MoE | `_stage2_kernel` | 411.005 | 160 | 2,568.8 | 18.08% |
| 4 | sparse attention | `gluon_dsv41_index_topk_gfx1250` | 306.754 | 593 | 517.3 | 13.49% |
| 5 | top-k and sort | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<f...` | 103.842 | 2,376 | 43.7 | 4.57% |
| 6 | sparse attention | `gluon_dsv4_prefill_gfx1250` | 64.107 | 160 | 400.7 | 2.82% |
| 7 | quantize | `_fp8_group32_ue8m0_quantize` | 58.495 | 842 | 69.5 | 2.57% |
| 8 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 56.568 | 320 | 176.8 | 2.49% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASE...` | 51.670 | 181 | 285.5 | 2.27% |
| 10 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 16.264 | 320 | 50.8 | 0.72% |

</details>

<details><summary>MI455X (gfx1250) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_stage1_kernel` | 8,322.349 | 2,280 | 3,650.2 | 23.34% |
| 2 | dense GEMM | `_w8a8_block_fp8_matmul` | 8,168.741 | 12,002 | 680.6 | 22.91% |
| 3 | MoE | `_stage2_kernel` | 6,326.494 | 2,280 | 2,774.8 | 17.75% |
| 4 | sparse attention | `gluon_dsv41_index_topk_gfx1250` | 4,915.449 | 9,575 | 513.4 | 13.79% |
| 5 | top-k and sort | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<f...` | 1,649.470 | 36,300 | 45.4 | 4.63% |
| 6 | sparse attention | `gluon_dsv4_prefill_gfx1250` | 1,037.894 | 2,880 | 360.4 | 2.91% |
| 7 | quantize | `_fp8_group32_ue8m0_quantize` | 929.346 | 12,002 | 77.4 | 2.61% |
| 8 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 905.935 | 4,560 | 198.7 | 2.54% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASE...` | 733.992 | 2,574 | 285.2 | 2.06% |
| 10 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 258.953 | 4,560 | 56.8 | 0.73% |

</details>

<details><summary>MI455X (gfx1250) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASE...` | 847.740 | 3,008 | 281.8 | 31.92% |
| 2 | MoE | `_stage1_kernel` | 604.939 | 2,752 | 219.8 | 22.78% |
| 3 | MoE | `_stage2_kernel` | 328.523 | 2,752 | 119.4 | 12.37% |
| 4 | sparse attention | `gluon_dsv41_selected_attention_gfx1250` | 233.725 | 2,560 | 91.3 | 8.80% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1...` | 140.433 | 2,752 | 51.0 | 5.29% |
| 6 | dense GEMM | `gluon_mm_mxfp8_ue8m0_gfx1250_bn16_bk512_buf3_sk1` | 50.243 | 11,776 | 4.3 | 1.89% |
| 7 | hyper-connections | `_mhc_pre_mix_hc4_kernel` | 47.029 | 5,504 | 8.5 | 1.77% |
| 8 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 39.248 | 5,504 | 7.1 | 1.48% |
| 9 | quantize | `_fp8_group32_ue8m0_quantize` | 33.248 | 14,528 | 2.3 | 1.25% |
| 10 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_...` | 31.999 | 8,512 | 3.8 | 1.20% |

</details>

<details><summary>MI455X (gfx1250) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_stage1_kernel` | 3,499.835 | 2,752 | 1,271.7 | 42.44% |
| 2 | MoE | `_stage2_kernel` | 2,109.373 | 2,752 | 766.5 | 25.58% |
| 3 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASE...` | 857.854 | 3,008 | 285.2 | 10.40% |
| 4 | dense GEMM | `_w8a8_block_fp8_matmul` | 471.563 | 14,528 | 32.5 | 5.72% |
| 5 | sparse attention | `gluon_dsv41_selected_attention_gfx1250` | 424.677 | 2,560 | 165.9 | 5.15% |
| 6 | sparse attention | `gluon_dsv41_index_topk_gfx1250` | 208.756 | 512 | 407.7 | 2.53% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1...` | 135.509 | 2,560 | 52.9 | 1.64% |
| 8 | hyper-connections | `_mhc_pre_mix_hc4_kernel` | 65.199 | 5,504 | 11.8 | 0.79% |
| 9 | top-k and sort | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<f...` | 64.201 | 2,304 | 27.9 | 0.78% |
| 10 | quantize | `_fp8_group32_ue8m0_quantize` | 56.763 | 14,528 | 3.9 | 0.69% |

</details>

## Kimi-K3 EAGLE3

| Metric | Batch | MI355X (gfx950) | MI455X (gfx1250) | MI455X advantage |
|---|---:|---:|---:|---:|
| TTFT p50 (ms) | 1 | 2,610.7 | 1,737.6 | 1.50x |
| TPOT p50 (ms) | 1 | 4.4 | 2.9 | 1.51x |
| steady decode TPOT (ms) | 1 | 4.3 | 2.8 | 1.52x |
| decode tok/s per user | 1 | 233.3 | 353.5 | 1.52x |
| aggregate output tok/s | 1 | 104.3 | 157.0 | 1.50x |
| TTFT p50 (ms) | 16 | 21,336.8 | 13,572.1 | 1.57x |
| TPOT p50 (ms) | 16 | 49.5 | 33.9 | 1.46x |
| steady decode TPOT (ms) | 16 | 11.2 | 9.2 | 1.22x |
| decode tok/s per user | 16 | 89.0 | 108.2 | 1.22x |
| aggregate output tok/s | 16 | 173.6 | 262.2 | 1.51x |

Accumulated GPU kernel time per bucket, MI355X over MI455X, so
above 1.00x means MI455X is ahead. The last row compares the
measured TTFT p50 and steady decode TPOT from the table above; it
is not a total of the rows.

| Category | prefill c1 | prefill c16 | decode c1 | decode c16 |
|---|---:|---:|---:|---:|
| KDA state scan | 1.35x | 1.37x | — | — |
| KDA other | 1.68x | 1.69x | 1.19x | 1.29x |
| MoE | 2.00x | 2.04x | 1.24x | 1.33x |
| input projections | 1.33x | 1.36x | 2.05x | 1.64x |
| dense GEMM | 1.33x | 1.36x | 1.84x | 0.97x |
| attention | 1.93x | 1.79x | 1.47x | 0.92x |
| AttnRes | 1.44x | 1.45x | 1.47x | 1.40x |
| add3 | 2.98x | 3.01x | 1.04x | 1.33x |
| quantize | 2.34x | 2.46x | 2.12x | 2.60x |
| rmsnorm | 1.62x | 1.66x | 1.18x | 1.22x |
| top-k and sort | 1.32x | 1.32x | — | — |
| sampling | 1.27x | 1.23x | 1.33x | 1.66x |
| elementwise | 1.41x | 1.42x | 1.51x | 0.97x |
| other | 1.49x | 1.48x | 1.09x | 1.02x |
| all kernels | 1.60x | 1.62x | 1.52x | 1.21x |
| **End-to-end** | **1.50x** (TTFT) | **1.57x** (TTFT) | **1.52x** (steady TPOT) | **1.22x** (steady TPOT) |

<details><summary>MI355X (gfx950) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel` | 394.307 | 644 | 612.3 | 15.99% |
| 2 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel` | 248.507 | 644 | 385.9 | 10.08% |
| 3 | input projections | `gluon_latent_input_largem_gfx950` | 221.305 | 552 | 400.9 | 8.97% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 212.434 | 414 | 513.1 | 8.62% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950` | 203.212 | 1,309 | 155.2 | 8.24% |
| 6 | attention | `gluon_mla_prefill_8wave_gfx950` | 151.713 | 336 | 451.5 | 6.15% |
| 7 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950` | 149.538 | 552 | 270.9 | 6.06% |
| 8 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950` | 146.191 | 552 | 264.8 | 5.93% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 122.967 | 1,122 | 109.6 | 4.99% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel` | 92.587 | 552 | 167.7 | 3.75% |

</details>

<details><summary>MI355X (gfx950) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel` | 6,015.710 | 9,016 | 667.2 | 15.49% |
| 2 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel` | 3,855.275 | 9,016 | 427.6 | 9.93% |
| 3 | input projections | `gluon_latent_input_largem_gfx950` | 3,646.632 | 9,016 | 404.5 | 9.39% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 3,521.614 | 7,003 | 502.9 | 9.07% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950` | 3,240.837 | 18,326 | 176.8 | 8.35% |
| 6 | attention | `gluon_mla_prefill_8wave_gfx950` | 2,346.458 | 5,784 | 405.7 | 6.04% |
| 7 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950` | 2,285.742 | 7,866 | 290.6 | 5.89% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 2,112.433 | 16,930 | 124.8 | 5.44% |
| 9 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950` | 2,051.921 | 7,640 | 268.6 | 5.28% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel` | 1,506.151 | 9,016 | 167.1 | 3.88% |

</details>

<details><summary>MI355X (gfx950) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `gluon_mm_a16w16_medium_gfx950` | 276.809 | 25,344 | 10.9 | 23.95% |
| 2 | MoE | `_warp_decode_precomputed_situ_stage1_kernel` | 116.261 | 5,888 | 19.7 | 10.06% |
| 3 | AttnRes | `gluon_attn_res_fwd_gfx950` | 115.821 | 11,968 | 9.7 | 10.02% |
| 4 | input projections | `gluon_latent_input_small_batch_gfx950` | 108.689 | 5,888 | 18.5 | 9.40% |
| 5 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel` | 70.048 | 5,888 | 11.9 | 6.06% |
| 6 | MoE | `gluon_sigmoid_bias_topk_gfx950` | 62.675 | 5,888 | 10.6 | 5.42% |
| 7 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx950` | 34.178 | 4,416 | 7.7 | 2.96% |
| 8 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_...` | 33.449 | 6,208 | 5.4 | 2.89% |
| 9 | elementwise | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>,...` | 32.442 | 6,016 | 5.4 | 2.81% |
| 10 | quantize | `_dynamic_fp8_single_pass_kernel` | 29.779 | 5,888 | 5.1 | 2.58% |

</details>

<details><summary>MI355X (gfx950) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_warp_decode_precomputed_situ_sorted_stage1_kernel` | 807.586 | 5,888 | 137.2 | 29.06% |
| 2 | MoE | `_warp_decode_sorted_stage2_fp8_mxfp4_kernel` | 509.783 | 5,888 | 86.6 | 18.34% |
| 3 | dense GEMM | `gluon_mm_a16w16_medium_gfx950` | 165.867 | 15,104 | 11.0 | 5.97% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT160x64x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 144.500 | 4,416 | 32.7 | 5.20% |
| 5 | attention | `gluon_mla_decode_fp8_query_blocks_gfx950` | 143.856 | 1,536 | 93.7 | 5.18% |
| 6 | input projections | `gluon_latent_input_small_batch_gfx950` | 133.601 | 5,888 | 22.7 | 4.81% |
| 7 | AttnRes | `gluon_attn_res_fwd_gfx950` | 119.184 | 11,968 | 10.0 | 4.29% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_D...` | 116.906 | 5,952 | 19.6 | 4.21% |
| 9 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx950` | 64.548 | 4,416 | 14.6 | 2.32% |
| 10 | MoE | `gluon_sigmoid_bias_topk_gfx950` | 61.280 | 5,888 | 10.4 | 2.20% |

</details>

<details><summary>MI455X (gfx1250) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel` | 434.168 | 3,300 | 131.6 | 28.23% |
| 2 | MoE | `_matmul` | 283.686 | 1,104 | 257.0 | 18.44% |
| 3 | input projections | `gluon_latent_input_largem_gfx1250` | 164.206 | 552 | 297.5 | 10.68% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 141.022 | 1,309 | 107.7 | 9.17% |
| 5 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250` | 108.071 | 552 | 195.8 | 7.03% |
| 6 | attention | `gluon_mla_prefill_gfx1250` | 77.887 | 360 | 216.4 | 5.06% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250` | 35.717 | 552 | 64.7 | 2.32% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel` | 34.116 | 644 | 53.0 | 2.22% |
| 9 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel` | 21.492 | 644 | 33.4 | 1.40% |
| 10 | MoE | `_matmul_decode` | 21.128 | 184 | 114.8 | 1.37% |

</details>

<details><summary>MI455X (gfx1250) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel` | 6,848.649 | 46,968 | 145.8 | 28.52% |
| 2 | MoE | `_matmul` | 4,613.750 | 18,032 | 255.9 | 19.21% |
| 3 | input projections | `gluon_latent_input_largem_gfx1250` | 2,680.152 | 9,016 | 297.3 | 11.16% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 2,236.703 | 18,326 | 122.1 | 9.31% |
| 5 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250` | 1,665.762 | 7,866 | 211.8 | 6.94% |
| 6 | attention | `gluon_mla_prefill_gfx1250` | 1,303.416 | 5,880 | 221.7 | 5.43% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250` | 549.978 | 7,866 | 69.9 | 2.29% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel` | 541.546 | 9,016 | 60.1 | 2.26% |
| 9 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel` | 306.998 | 9,016 | 34.1 | 1.28% |
| 10 | add3 | `_add3_kernel` | 270.063 | 9,016 | 30.0 | 1.12% |

</details>

<details><summary>MI455X (gfx1250) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_m16_kernel` | 148.458 | 25,600 | 5.8 | 19.52% |
| 2 | MoE | `_matmul_decode` | 117.944 | 11,776 | 10.0 | 15.51% |
| 3 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 78.546 | 11,968 | 6.6 | 10.33% |
| 4 | MoE | `_precomputed_topk_route_small_m_gfx1250_kernel` | 50.627 | 5,888 | 8.6 | 6.66% |
| 5 | input projections | `gluon_latent_input_decode_gfx1250` | 46.374 | 5,888 | 7.9 | 6.10% |
| 6 | MoE | `_kimi3_sigmoid_bias_topk_kernel` | 30.343 | 5,888 | 5.2 | 3.99% |
| 7 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx1250` | 29.260 | 4,416 | 6.6 | 3.85% |
| 8 | attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_...` | 23.537 | 1,536 | 15.3 | 3.09% |
| 9 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_...` | 22.391 | 6,208 | 3.6 | 2.94% |
| 10 | rmsnorm | `_rmsnorm_kernel` | 20.725 | 6,272 | 3.3 | 2.72% |

</details>

<details><summary>MI455X (gfx1250) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode` | 722.519 | 11,776 | 61.4 | 31.59% |
| 2 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x64x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM...` | 225.800 | 5,952 | 37.9 | 9.87% |
| 3 | attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_...` | 175.870 | 1,536 | 114.5 | 7.69% |
| 4 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel` | 171.533 | 5,888 | 29.1 | 7.50% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM...` | 169.164 | 17,728 | 9.5 | 7.40% |
| 6 | input projections | `_packed_input_projections_kernel` | 103.535 | 5,888 | 17.6 | 4.53% |
| 7 | MoE | `_precomputed_topk_route_large_stage1_gfx1250_kernel` | 94.595 | 5,888 | 16.1 | 4.14% |
| 8 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 85.264 | 11,968 | 7.1 | 3.73% |
| 9 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx1250` | 53.215 | 4,416 | 12.1 | 2.33% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x64x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM...` | 53.115 | 1,536 | 34.6 | 2.32% |

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
  --commit-date 2026-10-03 \
  --output-dir emulate_rank0/results/arch_compare_20261003_927562ef
```
