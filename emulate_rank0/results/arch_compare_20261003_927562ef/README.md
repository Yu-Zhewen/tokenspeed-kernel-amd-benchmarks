# Emulated rank 0 with speculative decoding: MI355X vs MI455X at `927562ef`

One GPU per architecture serves global rank 0 of a TP8 deployment with
`tokenspeed serve --emulate-rank-zero`, dummy weights, uniform MoE
routing and a fixed simulated accept length. Collectives are local
substitutes, so the numbers are a compute estimate, not serving
performance. Both architectures ran the same tree and workload.

## Provenance

| Field | Value |
|---|---|
| TokenSpeed revision | `927562ef44c275d13aa87448d43a45d3e7ec68ec` |
| Commit date (UTC) | 2026-10-03 |
| Changes on top | 7 files, identical on both hosts |
| Measured | 2026-10-03 |
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

| Model | Simulated | MI355X logged | MI455X logged |
|---|---:|---|---|
| GLM-5.3 MTP | 2.9 | 2.90 (18 windows) | 2.90 (18 windows) |
| DeepSeek-V4.1-Flash DSPARK | 3.9 | 3.90 (13 windows) | 3.90 (13 windows) |
| Kimi-K3 EAGLE3 | 3.75 | 3.75 (14 windows) | 3.75 (14 windows) |

Latency rows are lower-is-better and throughput rows higher-is-better;
the last column is always MI455X's advantage. TPOT p50 is per request
over its whole decode, as CI reports it. At batch 16 every request
arrives at once, so a request whose prompt finishes early decodes
between other requests' prefill chunks, and its TPOT mostly measures
prefill. Steady decode TPOT is the median over the server's 40-step
decode windows in which every request was decoding and no prefill ran;
decode tok/s per user and the decode columns below use it. Both include
the speculative speedup.

## GLM-5.3 MTP

| Metric | Batch | MI355X (gfx950) | MI455X (gfx1250) | MI455X advantage |
|---|---:|---:|---:|---:|
| TTFT p50 (ms) | 1 | 6,969.4 | 22,453.9 | 0.31x |
| TPOT p50 (ms) | 1 | 10.6 | 12.1 | 0.88x |
| steady decode TPOT (ms) | 1 | 10.5 | 11.7 | 0.90x |
| decode tok/s per user | 1 | 95.0 | 85.5 | 0.90x |
| aggregate output tok/s | 1 | 40.7 | 17.6 | 0.43x |
| TTFT p50 (ms) | 16 | 58,930.4 | 190,257.0 | 0.31x |
| TPOT p50 (ms) | 16 | 136.6 | 371.1 | 0.37x |
| steady decode TPOT (ms) | 16 | 32.5 | 34.7 | 0.94x |
| decode tok/s per user | 16 | 30.8 | 28.8 | 0.94x |
| aggregate output tok/s | 16 | 62.9 | 21.3 | 0.34x |

Accumulated GPU kernel time per bucket, MI355X over MI455X, so
above 1.00x means MI455X is ahead. The last row compares the
measured TTFT p50 and steady decode TPOT from the table above; it
is not a total of the rows.

| Category | prefill c1 | prefill c16 | decode c1 | decode c16 |
|---|---:|---:|---:|---:|
| MoE | 1.58x | 1.56x | 1.31x | 1.54x |
| sparse attention | 0.15x | 0.15x | 0.19x | 0.17x |
| dense GEMM | 3.16x | 3.20x | 1.98x | 1.17x |
| attention | 1.43x | 1.51x | 1.12x | 1.35x |
| quantize | 1.53x | 1.50x | 1.32x | 1.81x |
| rmsnorm | 1.74x | 1.76x | 1.00x | 1.35x |
| top-k and sort | — | — | — | — |
| sampling | 1.24x | 1.34x | 1.42x | 1.64x |
| elementwise | 1.56x | 1.59x | 0.57x | 0.75x |
| other | 0.72x | 0.74x | 0.96x | 1.05x |
| all kernels | 0.31x | 0.31x | 0.92x | 0.95x |
| **End-to-end** | **0.31x** (TTFT) | **0.31x** (TTFT) | **0.90x** (steady TPOT) | **0.94x** (steady TPOT) |

<details><summary>MI355X (gfx950) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | sparse attention | `_dsa_dense_mfma_kv_kernel` | 2,193.632 | 555 | 3,952.5 | 31.38% |
| 2 | dense GEMM | `_w8a8_block_fp8_matmul` | 1,405.883 | 3,085 | 455.7 | 20.11% |
| 3 | MoE | `_stage1_kernel` | 1,189.133 | 534 | 2,226.8 | 17.01% |
| 4 | MoE | `_stage2_kernel` | 868.206 | 534 | 1,625.9 | 12.42% |
| 5 | sparse attention | `gluon_dsa_prefill_topk_standard_gfx950` | 692.324 | 308 | 2,247.8 | 9.90% |
| 6 | quantize | `_per_token_group_quant_8bit` | 132.874 | 3,085 | 43.1 | 1.90% |
| 7 | MoE | `_combine_kernel` | 106.266 | 534 | 199.0 | 1.52% |
| 8 | rmsnorm | `_rmsnorm_kernel` | 59.896 | 1,144 | 52.4 | 0.86% |
| 9 | sparse attention | `_dsa_oneblock_manual_radix_topk_kernel` | 55.187 | 309 | 178.6 | 0.79% |
| 10 | elementwise | `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(in...` | 51.943 | 543 | 95.7 | 0.74% |

</details>

<details><summary>MI355X (gfx950) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | sparse attention | `_dsa_dense_mfma_kv_kernel` | 34,971.820 | 7,774 | 4,498.6 | 31.63% |
| 2 | dense GEMM | `_w8a8_block_fp8_matmul` | 22,243.672 | 43,214 | 514.7 | 20.12% |
| 3 | MoE | `_stage1_kernel` | 18,597.751 | 7,480 | 2,486.3 | 16.82% |
| 4 | MoE | `_stage2_kernel` | 13,562.482 | 7,480 | 1,813.2 | 12.27% |
| 5 | sparse attention | `gluon_dsa_prefill_topk_standard_gfx950` | 10,994.133 | 5,390 | 2,039.7 | 9.95% |
| 6 | quantize | `_per_token_group_quant_8bit` | 2,103.468 | 43,214 | 48.7 | 1.90% |
| 7 | MoE | `_combine_kernel` | 1,711.208 | 7,480 | 228.8 | 1.55% |
| 8 | rmsnorm | `_rmsnorm_kernel` | 951.622 | 16,036 | 59.3 | 0.86% |
| 9 | sparse attention | `_dsa_oneblock_manual_radix_topk_kernel` | 881.799 | 5,406 | 163.1 | 0.80% |
| 10 | elementwise | `void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(in...` | 824.496 | 7,610 | 108.3 | 0.75% |

</details>

<details><summary>MI355X (gfx950) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_w8a8_block_fp8_matmul` | 735.505 | 28,736 | 25.6 | 34.51% |
| 2 | MoE | `_stage1_kernel` | 362.788 | 4,992 | 72.7 | 17.02% |
| 3 | MoE | `_stage2_kernel` | 304.325 | 4,992 | 61.0 | 14.28% |
| 4 | quantize | `_per_token_group_quant_8bit` | 88.867 | 28,736 | 3.1 | 4.17% |
| 5 | sparse attention | `_dsa_dense_mfma_kv_kernel` | 72.639 | 5,184 | 14.0 | 3.41% |
| 6 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 56.520 | 5,056 | 11.2 | 2.65% |
| 7 | rmsnorm | `_rmsnorm_kernel` | 42.135 | 11,008 | 3.8 | 1.98% |
| 8 | sparse attention | `_dsa_oneblock_manual_radix_topk_kernel` | 41.493 | 1,408 | 29.5 | 1.95% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_DT...` | 40.962 | 4,992 | 8.2 | 1.92% |
| 10 | MoE | `_sigmoid_bias_topk_route_gluon_kernel` | 30.777 | 4,992 | 6.2 | 1.44% |

</details>

<details><summary>MI355X (gfx950) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_stage1_kernel` | 2,701.350 | 4,992 | 541.1 | 42.54% |
| 2 | MoE | `_stage2_kernel` | 1,838.505 | 4,992 | 368.3 | 28.95% |
| 3 | dense GEMM | `_w8a8_block_fp8_matmul` | 703.179 | 28,736 | 24.5 | 11.07% |
| 4 | sparse attention | `_dsa_dense_mfma_kv_kernel` | 206.489 | 5,184 | 39.8 | 3.25% |
| 5 | sparse attention | `gluon_dsa_decode_topk_standard_gfx950` | 153.097 | 1,408 | 108.7 | 2.41% |
| 6 | quantize | `_per_token_group_quant_8bit` | 126.121 | 28,736 | 4.4 | 1.99% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 64.492 | 6,208 | 10.4 | 1.02% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x64x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_D...` | 62.670 | 5,056 | 12.4 | 0.99% |
| 9 | rmsnorm | `_rmsnorm_kernel` | 49.409 | 11,008 | 4.5 | 0.78% |
| 10 | sparse attention | `_dsa_oneblock_manual_radix_topk_kernel` | 47.027 | 1,408 | 33.4 | 0.74% |

</details>

<details><summary>MI455X (gfx1250) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | sparse attention | `_dsa_selected_dense_wmma_kernel` | 19,480.619 | 555 | 35,100.2 | 86.73% |
| 2 | sparse attention | `gluon_dsa_prefill_topk_standard_gfx1250` | 793.970 | 308 | 2,577.8 | 3.53% |
| 3 | MoE | `_stage2_kernel` | 670.461 | 534 | 1,255.5 | 2.98% |
| 4 | MoE | `_stage1_kernel` | 662.582 | 534 | 1,240.8 | 2.95% |
| 5 | dense GEMM | `_w8a8_block_fp8_matmul` | 377.991 | 3,052 | 123.9 | 1.68% |
| 6 | quantize | `_per_token_group_quant_8bit` | 89.159 | 3,085 | 28.9 | 0.40% |
| 7 | sparse attention | `_dsa_wave32_radix_topk_kernel` | 83.375 | 309 | 269.8 | 0.37% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASE...` | 75.700 | 1,398 | 54.1 | 0.34% |
| 9 | MoE | `_routing_kernel` | 30.383 | 534 | 56.9 | 0.14% |
| 10 | rmsnorm | `_rmsnorm_kernel` | 29.018 | 1,144 | 25.4 | 0.13% |

</details>

<details><summary>MI455X (gfx1250) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | sparse attention | `_dsa_selected_dense_wmma_kernel` | 311,164.196 | 7,774 | 40,026.3 | 86.78% |
| 2 | sparse attention | `gluon_dsa_prefill_topk_standard_gfx1250` | 12,709.734 | 5,390 | 2,358.0 | 3.54% |
| 3 | MoE | `_stage2_kernel` | 10,828.626 | 7,480 | 1,447.7 | 3.02% |
| 4 | MoE | `_stage1_kernel` | 10,384.820 | 7,480 | 1,388.3 | 2.90% |
| 5 | dense GEMM | `_w8a8_block_fp8_matmul` | 5,941.301 | 42,728 | 139.0 | 1.66% |
| 6 | quantize | `_per_token_group_quant_8bit` | 1,444.253 | 43,214 | 33.4 | 0.40% |
| 7 | sparse attention | `_dsa_wave32_radix_topk_kernel` | 1,343.855 | 5,406 | 248.6 | 0.37% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASE...` | 1,219.086 | 22,601 | 53.9 | 0.34% |
| 9 | MoE | `_routing_kernel` | 507.875 | 7,480 | 67.9 | 0.14% |
| 10 | rmsnorm | `_rmsnorm_kernel` | 449.971 | 16,036 | 28.1 | 0.13% |

</details>

<details><summary>MI455X (gfx1250) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | sparse attention | `_dsa_selected_dense_wmma_kernel` | 753.059 | 5,184 | 145.3 | 32.44% |
| 2 | MoE | `_stage1_kernel` | 273.177 | 4,992 | 54.7 | 11.77% |
| 3 | MoE | `_stage2_kernel` | 266.448 | 4,992 | 53.4 | 11.48% |
| 4 | dense GEMM | `_w8a8_block_fp8_matmul` | 152.135 | 4,992 | 30.5 | 6.55% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1...` | 150.817 | 10,176 | 14.8 | 6.50% |
| 6 | sparse attention | `_dsa_wave32_radix_topk_kernel` | 69.360 | 1,408 | 49.3 | 2.99% |
| 7 | quantize | `_per_token_group_quant_8bit` | 66.286 | 28,736 | 2.3 | 2.86% |
| 8 | dense GEMM | `gluon_mm_fp8_blockscale_gfx1250_bn16_bk1024_buf3_sk1_tdmf0` | 47.427 | 11,776 | 4.0 | 2.04% |
| 9 | dense GEMM | `gluon_mm_fp8_blockscale_gfx1250_bn16_bk2048_buf3_sk1_tdmf0` | 46.525 | 6,784 | 6.9 | 2.00% |
| 10 | top-k and sort | `void at::native::warptopk::warpMergeSortTopK<2, 2, 256, 1, float, unsigned int, true, 32>(at::cuda::detail::TensorInf...` | 43.562 | 4,800 | 9.1 | 1.88% |

</details>

<details><summary>MI455X (gfx1250) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | sparse attention | `_dsa_selected_dense_wmma_kernel` | 2,035.410 | 5,184 | 392.6 | 30.37% |
| 2 | MoE | `_stage1_kernel` | 1,588.493 | 4,992 | 318.2 | 23.70% |
| 3 | MoE | `_stage2_kernel` | 1,373.060 | 4,992 | 275.1 | 20.49% |
| 4 | dense GEMM | `_w8a8_block_fp8_matmul` | 512.697 | 27,904 | 18.4 | 7.65% |
| 5 | sparse attention | `gluon_dsa_decode_topk_standard_gfx1250` | 381.450 | 1,408 | 270.9 | 5.69% |
| 6 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x64x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1...` | 125.090 | 4,800 | 26.1 | 1.87% |
| 7 | sparse attention | `_dsa_wave32_radix_topk_kernel` | 85.061 | 1,408 | 60.4 | 1.27% |
| 8 | quantize | `_per_token_group_quant_8bit` | 69.735 | 28,736 | 2.4 | 1.04% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1...` | 58.885 | 6,464 | 9.1 | 0.88% |
| 10 | top-k and sort | `void at::native::warptopk::warpMergeSortTopK<2, 2, 256, 1, float, unsigned int, true, 32>(at::cuda::detail::TensorInf...` | 44.658 | 4,992 | 8.9 | 0.67% |

</details>

## DeepSeek-V4.1-Flash DSPARK

| Metric | Batch | MI355X (gfx950) | MI455X (gfx1250) | MI455X advantage |
|---|---:|---:|---:|---:|
| TTFT p50 (ms) | 1 | 1,984.6 | 1,831.7 | 1.08x |
| TPOT p50 (ms) | 1 | 9.3 | 8.7 | 1.07x |
| steady decode TPOT (ms) | 1 | 9.2 | 8.6 | 1.07x |
| decode tok/s per user | 1 | 108.9 | 116.0 | 1.07x |
| aggregate output tok/s | 1 | 75.5 | 80.9 | 1.07x |
| TTFT p50 (ms) | 16 | 16,373.3 | 15,164.6 | 1.08x |
| TPOT p50 (ms) | 16 | 57.0 | 50.2 | 1.13x |
| steady decode TPOT (ms) | 16 | 27.5 | 22.9 | 1.20x |
| decode tok/s per user | 16 | 36.3 | 43.6 | 1.20x |
| aggregate output tok/s | 16 | 178.5 | 198.9 | 1.11x |

Accumulated GPU kernel time per bucket, MI355X over MI455X, so
above 1.00x means MI455X is ahead. The last row compares the
measured TTFT p50 and steady decode TPOT from the table above; it
is not a total of the rows.

| Category | prefill c1 | prefill c16 | decode c1 | decode c16 |
|---|---:|---:|---:|---:|
| MoE | 1.20x | 1.19x | 1.36x | 1.39x |
| sparse attention | 1.33x | 1.33x | 1.71x | 1.46x |
| dense GEMM | 0.49x | 0.47x | 0.57x | 0.48x |
| hyper-connections | 2.18x | 2.19x | 1.18x | 1.26x |
| quantize | 1.73x | 1.72x | 1.67x | 1.56x |
| rmsnorm | 1.04x | 0.92x | 1.24x | 1.97x |
| top-k and sort | 0.56x | 0.57x | 1.43x | 0.73x |
| sampling | 1.21x | 1.15x | 1.61x | 2.75x |
| elementwise | 1.59x | 1.58x | 1.36x | 2.02x |
| other | 1.46x | 1.45x | 1.11x | 1.49x |
| all kernels | 1.08x | 1.08x | 1.05x | 1.20x |
| **End-to-end** | **1.08x** (TTFT) | **1.08x** (TTFT) | **1.07x** (steady TPOT) | **1.20x** (steady TPOT) |

<details><summary>MI355X (gfx950) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_stage1_kernel` | 439.586 | 160 | 2,747.4 | 22.30% |
| 2 | MoE | `_stage2_kernel` | 333.427 | 160 | 2,083.9 | 16.91% |
| 3 | sparse attention | `gluon_dsv41_index_topk_gfx950` | 280.747 | 593 | 473.4 | 14.24% |
| 4 | sparse attention | `gluon_dsv4_prefill_gfx950` | 209.491 | 160 | 1,309.3 | 10.63% |
| 5 | dense GEMM | `_w8a8_block_fp8_matmul` | 153.735 | 452 | 340.1 | 7.80% |
| 6 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 127.241 | 320 | 397.6 | 6.45% |
| 7 | quantize | `_fp8_group32_ue8m0_quantize` | 92.022 | 842 | 109.3 | 4.67% |
| 8 | top-k and sort | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<f...` | 37.021 | 2,376 | 15.6 | 1.88% |
| 9 | dense GEMM | `gluon_mm_mxfp8_gfx950` | 35.775 | 390 | 91.7 | 1.81% |
| 10 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 33.994 | 320 | 106.2 | 1.72% |

</details>

<details><summary>MI355X (gfx950) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_stage1_kernel` | 6,731.454 | 2,280 | 2,952.4 | 21.80% |
| 2 | MoE | `_stage2_kernel` | 5,109.253 | 2,280 | 2,240.9 | 16.55% |
| 3 | sparse attention | `gluon_dsv41_index_topk_gfx950` | 4,494.088 | 9,583 | 469.0 | 14.55% |
| 4 | sparse attention | `gluon_dsv4_prefill_gfx950` | 3,379.433 | 2,920 | 1,157.3 | 10.94% |
| 5 | dense GEMM | `_w8a8_block_fp8_matmul` | 2,316.566 | 5,682 | 407.7 | 7.50% |
| 6 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 2,042.764 | 4,560 | 448.0 | 6.62% |
| 7 | quantize | `_fp8_group32_ue8m0_quantize` | 1,472.893 | 12,002 | 122.7 | 4.77% |
| 8 | top-k and sort | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<f...` | 586.774 | 36,300 | 16.2 | 1.90% |
| 9 | dense GEMM | `gluon_mm_mxfp8_gfx950` | 579.564 | 6,320 | 91.7 | 1.88% |
| 10 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 543.999 | 4,560 | 119.3 | 1.76% |

</details>

<details><summary>MI355X (gfx950) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_stage1_kernel` | 584.646 | 2,752 | 212.4 | 24.10% |
| 2 | dense GEMM | `_w8a8_block_fp8_matmul` | 491.540 | 14,528 | 33.8 | 20.26% |
| 3 | MoE | `_stage2_kernel` | 369.607 | 2,752 | 134.3 | 15.24% |
| 4 | sparse attention | `gluon_dsv41_selected_attention_gfx950` | 367.894 | 2,560 | 143.7 | 15.17% |
| 5 | quantize | `_fp8_group32_ue8m0_quantize` | 53.828 | 14,528 | 3.7 | 2.22% |
| 6 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 52.293 | 5,504 | 9.5 | 2.16% |
| 7 | hyper-connections | `_mhc_pre_mix_hc4_kernel` | 50.705 | 5,504 | 9.2 | 2.09% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_S_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0...` | 33.011 | 2,752 | 12.0 | 1.36% |
| 9 | top-k and sort | `void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>(at::cuda::detail::TensorInfo<float const, unsigned...` | 31.365 | 320 | 98.0 | 1.29% |
| 10 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_...` | 25.329 | 5,760 | 4.4 | 1.04% |

</details>

<details><summary>MI355X (gfx950) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_stage1_kernel` | 2,846.036 | 2,752 | 1,034.2 | 39.50% |
| 2 | MoE | `_stage2_kernel` | 2,046.326 | 2,752 | 743.6 | 28.40% |
| 3 | sparse attention | `gluon_dsv41_selected_attention_gfx950` | 732.674 | 2,560 | 286.2 | 10.17% |
| 4 | dense GEMM | `_w8a8_block_fp8_matmul` | 568.903 | 14,528 | 39.2 | 7.90% |
| 5 | sparse attention | `gluon_dsv41_index_topk_gfx950` | 189.107 | 512 | 369.4 | 2.62% |
| 6 | quantize | `_fp8_group32_ue8m0_quantize` | 89.760 | 14,528 | 6.2 | 1.25% |
| 7 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 66.518 | 5,504 | 12.1 | 0.92% |
| 8 | hyper-connections | `_mhc_pre_mix_hc4_kernel` | 60.220 | 5,504 | 10.9 | 0.84% |
| 9 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_...` | 34.668 | 5,760 | 6.0 | 0.48% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 33.605 | 2,752 | 12.2 | 0.47% |

</details>

<details><summary>MI455X (gfx1250) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_w8a8_block_fp8_matmul` | 374.776 | 842 | 445.1 | 20.60% |
| 2 | MoE | `_stage2_kernel` | 373.136 | 160 | 2,332.1 | 20.51% |
| 3 | sparse attention | `gluon_dsv41_index_topk_gfx1250` | 306.855 | 593 | 517.5 | 16.87% |
| 4 | MoE | `_stage1_kernel` | 277.896 | 160 | 1,736.9 | 15.28% |
| 5 | top-k and sort | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<f...` | 104.044 | 2,376 | 43.8 | 5.72% |
| 6 | sparse attention | `gluon_dsv4_prefill_gfx1250` | 62.130 | 160 | 388.3 | 3.42% |
| 7 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 56.334 | 320 | 176.0 | 3.10% |
| 8 | quantize | `_fp8_group32_ue8m0_quantize` | 53.251 | 842 | 63.2 | 2.93% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASE...` | 51.677 | 181 | 285.5 | 2.84% |
| 10 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 16.250 | 320 | 50.8 | 0.89% |

</details>

<details><summary>MI455X (gfx1250) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_w8a8_block_fp8_matmul` | 5,991.706 | 12,002 | 499.2 | 20.96% |
| 2 | MoE | `_stage2_kernel` | 5,759.305 | 2,280 | 2,526.0 | 20.15% |
| 3 | sparse attention | `gluon_dsv41_index_topk_gfx1250` | 4,916.631 | 9,583 | 513.1 | 17.20% |
| 4 | MoE | `_stage1_kernel` | 4,244.119 | 2,280 | 1,861.5 | 14.85% |
| 5 | top-k and sort | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<f...` | 1,649.953 | 36,300 | 45.5 | 5.77% |
| 6 | sparse attention | `gluon_dsv4_prefill_gfx1250` | 1,002.263 | 2,920 | 343.2 | 3.51% |
| 7 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 901.903 | 4,560 | 197.8 | 3.16% |
| 8 | quantize | `_fp8_group32_ue8m0_quantize` | 854.803 | 12,002 | 71.2 | 2.99% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASE...` | 734.897 | 2,574 | 285.5 | 2.57% |
| 10 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 258.722 | 4,560 | 56.7 | 0.91% |

</details>

<details><summary>MI455X (gfx1250) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASE...` | 842.239 | 3,008 | 280.0 | 36.48% |
| 2 | MoE | `_stage1_kernel` | 441.498 | 2,752 | 160.4 | 19.12% |
| 3 | MoE | `_stage2_kernel` | 262.715 | 2,752 | 95.5 | 11.38% |
| 4 | sparse attention | `gluon_dsv41_selected_attention_gfx1250` | 199.615 | 2,560 | 78.0 | 8.65% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1...` | 80.935 | 2,752 | 29.4 | 3.51% |
| 6 | hyper-connections | `_mhc_pre_mix_hc4_kernel` | 47.616 | 5,504 | 8.7 | 2.06% |
| 7 | hyper-connections | `_mhc_prenorm_gemm_triton_kernel` | 38.873 | 5,504 | 7.1 | 1.68% |
| 8 | quantize | `_fp8_group32_ue8m0_quantize` | 32.175 | 14,528 | 2.2 | 1.39% |
| 9 | dense GEMM | `gluon_mm_mxfp8_ue8m0_gfx1250_bn16_bk512_buf3_sk1` | 30.840 | 9,024 | 3.4 | 1.34% |
| 10 | sparse attention | `gluon_dsv41_index_topk_gfx1250` | 25.739 | 512 | 50.3 | 1.11% |

</details>

<details><summary>MI455X (gfx1250) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_stage1_kernel` | 1,795.392 | 2,752 | 652.4 | 29.90% |
| 2 | MoE | `_stage2_kernel` | 1,725.196 | 2,752 | 626.9 | 28.73% |
| 3 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASE...` | 857.950 | 3,008 | 285.2 | 14.29% |
| 4 | dense GEMM | `_w8a8_block_fp8_matmul` | 431.829 | 14,528 | 29.7 | 7.19% |
| 5 | sparse attention | `gluon_dsv41_selected_attention_gfx1250` | 421.394 | 2,560 | 164.6 | 7.02% |
| 6 | sparse attention | `gluon_dsv41_index_topk_gfx1250` | 209.042 | 512 | 408.3 | 3.48% |
| 7 | hyper-connections | `_mhc_pre_mix_hc4_kernel` | 64.441 | 5,504 | 11.7 | 1.07% |
| 8 | top-k and sort | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<f...` | 64.099 | 2,304 | 27.8 | 1.07% |
| 9 | quantize | `_fp8_group32_ue8m0_quantize` | 57.617 | 14,528 | 4.0 | 0.96% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x256_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM...` | 45.769 | 3,008 | 15.2 | 0.76% |

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
| KDA other | 1.68x | 1.70x | 1.19x | 1.29x |
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
| other | 1.48x | 1.53x | 1.09x | 1.02x |
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
