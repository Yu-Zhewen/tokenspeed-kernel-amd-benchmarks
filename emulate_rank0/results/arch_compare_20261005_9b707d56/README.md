# Emulated rank 0 with and without speculative decoding: MI355X vs MI455X at `9b707d56`

One GPU per architecture serves global rank 0 of a TP8 deployment with
`tokenspeed serve --emulate-rank-zero`, dummy weights and uniform MoE
routing. Speculative runs fix a simulated accept length. Collectives
are local substitutes, so the numbers are a compute estimate, not
serving performance. Both architectures ran the same tree and workload.

## Provenance

| Field | Value |
|---|---|
| TokenSpeed revision | `9b707d567a9a8523ab7ce33b9ae6f1b2792ce9fc` |
| Commit date (UTC) | 2026-10-05 |
| Changes on top | 0 files, identical on both hosts |
| Measured | 2026-10-05 |
| Prompt / output tokens | 50,000 / 500 |
| Concurrencies | 1, 16 |
| Warmup / measured waves | 1 / 1 |
| MI355X | `gfx950:sramecc+:xnack-`, 2.14.0+rocm7.2, `zhewenyu/emulate-rank0:9b707d56-gfx950` |
| MI455X | `gfx1250`, 2.14.0+rocm10.2.0a20260923, `tokenspeed-gfx1250:9b707d56-torch214` |
| tokenspeed-triton | 3.8.10.post20260920 / 3.8.10.post20260920 |

## Simulated acceptance

Logged values are medians over the server's 40-step decode windows;
windows that straddle a prefill keep fewer tokens. See
`emulate_rank0/README.md` for where each simulated value comes from.

| Model | Simulated | MI355X logged | MI455X logged |
|---|---:|---|---|
| Kimi-K3 EAGLE3 | 3.75 | 3.75 (14 windows) | 3.75 (14 windows) |

Latency rows are lower-is-better and throughput rows higher-is-better;
the last column is always MI455X's advantage. TPOT p50 is per request
over its whole decode, as CI reports it. At batch 16 every request
arrives at once, so a request whose prompt finishes early decodes
between other requests' prefill chunks, and its TPOT mostly measures
prefill. Steady decode TPOT is the median over the server's 40-step
decode windows in which every request was decoding and no prefill ran;
decode tok/s per user and the decode columns below use it. Both include
the speculative speedup when speculation is on.

## Kimi-K3 EAGLE3

| Metric | Batch | MI355X (gfx950) | MI455X (gfx1250) | MI455X advantage |
|---|---:|---:|---:|---:|
| TTFT p50 (ms) | 1 | 2,547.6 | 1,727.7 | 1.47x |
| TPOT p50 (ms) | 1 | 4.3 | 2.9 | 1.48x |
| steady decode TPOT (ms) | 1 | 4.2 | 2.8 | 1.49x |
| decode tok/s per user | 1 | 237.6 | 353.4 | 1.49x |
| aggregate output tok/s | 1 | 106.6 | 157.7 | 1.48x |
| TTFT p50 (ms) | 16 | 20,861.6 | 13,537.7 | 1.54x |
| TPOT p50 (ms) | 16 | 48.9 | 33.9 | 1.44x |
| steady decode TPOT (ms) | 16 | 11.3 | 9.2 | 1.22x |
| decode tok/s per user | 16 | 88.8 | 108.2 | 1.22x |
| aggregate output tok/s | 16 | 176.8 | 262.7 | 1.49x |

Accumulated GPU kernel time per bucket, MI355X over MI455X, so
above 1.00x means MI455X is ahead. The last row compares the
measured TTFT p50 and steady decode TPOT from the table above; it
is not a total of the rows.

| Category | prefill c1 | prefill c16 | decode c1 | decode c16 |
|---|---:|---:|---:|---:|
| KDA state scan | 1.10x | 1.13x | — | — |
| KDA other | 1.69x | 1.71x | 1.25x | 1.28x |
| MoE | 2.01x | 2.05x | 1.24x | 1.32x |
| input projections | 1.35x | 1.36x | 2.09x | 1.66x |
| dense GEMM | 1.34x | 1.34x | 1.87x | 0.99x |
| attention | 1.95x | 1.80x | 1.46x | 0.92x |
| AttnRes | 1.33x | 1.33x | 1.46x | 1.41x |
| add3 | 2.81x | 2.83x | — | 1.35x |
| quantize | 2.36x | 2.46x | 2.07x | 2.35x |
| rmsnorm | 1.62x | 1.64x | 1.13x | 1.21x |
| top-k and sort | 1.29x | 1.28x | — | — |
| sampling | 1.27x | 1.26x | 1.35x | 1.65x |
| elementwise | 1.39x | 1.40x | 1.51x | 0.98x |
| other | 1.49x | 1.53x | 1.02x | 1.04x |
| all kernels | 1.58x | 1.58x | 1.50x | 1.22x |
| **End-to-end** | **1.47x** (TTFT) | **1.54x** (TTFT) | **1.49x** (steady TPOT) | **1.22x** (steady TPOT) |

<details><summary>MI355X (gfx950) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel` | 394.190 | 644 | 612.1 | 16.20% |
| 2 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel` | 250.592 | 644 | 389.1 | 10.30% |
| 3 | input projections | `gluon_latent_input_largem_gfx950` | 224.166 | 552 | 406.1 | 9.21% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 216.784 | 414 | 523.6 | 8.91% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950` | 187.678 | 1,309 | 143.4 | 7.71% |
| 6 | attention | `gluon_mla_prefill_8wave_gfx950` | 153.234 | 336 | 456.1 | 6.30% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 147.017 | 564 | 260.7 | 6.04% |
| 8 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950` | 135.995 | 1,254 | 108.4 | 5.59% |
| 9 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950` | 118.887 | 552 | 215.4 | 4.89% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel` | 94.353 | 552 | 170.9 | 3.88% |

</details>

<details><summary>MI355X (gfx950) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel` | 6,012.995 | 9,016 | 666.9 | 15.81% |
| 2 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel` | 3,882.923 | 9,016 | 430.7 | 10.21% |
| 3 | input projections | `gluon_latent_input_largem_gfx950` | 3,651.196 | 9,016 | 405.0 | 9.60% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 3,530.804 | 6,787 | 520.2 | 9.28% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950` | 2,972.561 | 18,326 | 162.2 | 7.82% |
| 6 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 2,404.517 | 9,211 | 261.0 | 6.32% |
| 7 | attention | `gluon_mla_prefill_8wave_gfx950` | 2,345.485 | 5,784 | 405.5 | 6.17% |
| 8 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950` | 2,194.233 | 20,746 | 105.8 | 5.77% |
| 9 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950` | 1,874.139 | 7,866 | 238.3 | 4.93% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel` | 1,537.509 | 9,016 | 170.5 | 4.04% |

</details>

<details><summary>MI355X (gfx950) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `gluon_mm_a16w16_medium_gfx950` | 279.846 | 25,344 | 11.0 | 24.53% |
| 2 | AttnRes | `gluon_attn_res_fwd_gfx950` | 115.653 | 11,968 | 9.7 | 10.14% |
| 3 | MoE | `_warp_decode_precomputed_situ_stage1_kernel` | 115.506 | 5,888 | 19.6 | 10.12% |
| 4 | input projections | `gluon_latent_input_small_batch_gfx950` | 110.228 | 5,888 | 18.7 | 9.66% |
| 5 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel` | 70.232 | 5,888 | 11.9 | 6.16% |
| 6 | MoE | `gluon_sigmoid_bias_topk_gfx950` | 62.074 | 5,888 | 10.5 | 5.44% |
| 7 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx950` | 34.575 | 4,416 | 7.8 | 3.03% |
| 8 | elementwise | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>,...` | 34.486 | 6,016 | 5.7 | 3.02% |
| 9 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_...` | 33.860 | 6,208 | 5.5 | 2.97% |
| 10 | quantize | `_dynamic_fp8_single_pass_kernel` | 29.562 | 5,888 | 5.0 | 2.59% |

</details>

<details><summary>MI355X (gfx950) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_warp_decode_precomputed_situ_sorted_stage1_kernel` | 808.222 | 5,888 | 137.3 | 29.13% |
| 2 | MoE | `_warp_decode_sorted_stage2_fp8_mxfp4_kernel` | 503.913 | 5,888 | 85.6 | 18.16% |
| 3 | dense GEMM | `gluon_mm_a16w16_medium_gfx950` | 167.098 | 15,104 | 11.1 | 6.02% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT160x64x128_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 146.312 | 4,416 | 33.1 | 5.27% |
| 5 | attention | `gluon_mla_decode_fp8_query_blocks_gfx950` | 143.876 | 1,536 | 93.7 | 5.19% |
| 6 | input projections | `gluon_latent_input_small_batch_gfx950` | 134.711 | 5,888 | 22.9 | 4.86% |
| 7 | AttnRes | `gluon_attn_res_fwd_gfx950` | 120.627 | 11,968 | 10.1 | 4.35% |
| 8 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA1_D...` | 115.578 | 5,952 | 19.4 | 4.17% |
| 9 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx950` | 64.407 | 4,416 | 14.6 | 2.32% |
| 10 | MoE | `gluon_sigmoid_bias_topk_gfx950` | 61.754 | 5,888 | 10.5 | 2.23% |

</details>

<details><summary>MI455X (gfx1250) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel` | 432.916 | 3,300 | 131.2 | 28.18% |
| 2 | MoE | `_matmul` | 283.385 | 1,104 | 256.7 | 18.45% |
| 3 | input projections | `gluon_latent_input_largem_gfx1250` | 163.899 | 552 | 296.9 | 10.67% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 140.614 | 1,309 | 107.4 | 9.15% |
| 5 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250` | 108.061 | 552 | 195.8 | 7.03% |
| 6 | attention | `gluon_mla_prefill_gfx1250` | 78.008 | 360 | 216.7 | 5.08% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250` | 35.821 | 552 | 64.9 | 2.33% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel` | 34.112 | 644 | 53.0 | 2.22% |
| 9 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel` | 21.540 | 644 | 33.4 | 1.40% |
| 10 | MoE | `_matmul_decode` | 21.191 | 184 | 115.2 | 1.38% |

</details>

<details><summary>MI455X (gfx1250) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel` | 6,844.055 | 46,968 | 145.7 | 28.51% |
| 2 | MoE | `_matmul` | 4,613.888 | 18,032 | 255.9 | 19.22% |
| 3 | input projections | `gluon_latent_input_largem_gfx1250` | 2,677.352 | 9,016 | 297.0 | 11.15% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 2,233.510 | 18,326 | 121.9 | 9.30% |
| 5 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250` | 1,663.004 | 7,866 | 211.4 | 6.93% |
| 6 | attention | `gluon_mla_prefill_gfx1250` | 1,304.956 | 5,880 | 221.9 | 5.44% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250` | 548.707 | 7,866 | 69.8 | 2.29% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel` | 540.941 | 9,016 | 60.0 | 2.25% |
| 9 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel` | 306.043 | 9,016 | 33.9 | 1.27% |
| 10 | add3 | `_add3_kernel` | 266.675 | 9,016 | 29.6 | 1.11% |

</details>

<details><summary>MI455X (gfx1250) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_m16_kernel` | 147.267 | 25,600 | 5.8 | 19.40% |
| 2 | MoE | `_matmul_decode` | 116.405 | 11,776 | 9.9 | 15.33% |
| 3 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 79.483 | 11,968 | 6.6 | 10.47% |
| 4 | MoE | `_precomputed_topk_route_small_m_gfx1250_kernel` | 51.623 | 5,888 | 8.8 | 6.80% |
| 5 | input projections | `gluon_latent_input_decode_gfx1250` | 45.880 | 5,888 | 7.8 | 6.04% |
| 6 | MoE | `_kimi3_sigmoid_bias_topk_kernel` | 30.321 | 5,888 | 5.1 | 3.99% |
| 7 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx1250` | 28.095 | 4,416 | 6.4 | 3.70% |
| 8 | attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_...` | 23.321 | 1,536 | 15.2 | 3.07% |
| 9 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_...` | 22.321 | 6,208 | 3.6 | 2.94% |
| 10 | rmsnorm | `_rmsnorm_kernel` | 21.032 | 6,272 | 3.4 | 2.77% |

</details>

<details><summary>MI455X (gfx1250) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode` | 721.720 | 11,776 | 61.3 | 31.74% |
| 2 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT192x64x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM...` | 221.747 | 5,952 | 37.3 | 9.75% |
| 3 | attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_...` | 176.287 | 1,536 | 114.8 | 7.75% |
| 4 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel` | 171.169 | 5,888 | 29.1 | 7.53% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM...` | 165.534 | 17,728 | 9.3 | 7.28% |
| 6 | input projections | `_packed_input_projections_kernel` | 103.274 | 5,888 | 17.5 | 4.54% |
| 7 | MoE | `_precomputed_topk_route_large_stage1_gfx1250_kernel` | 94.452 | 5,888 | 16.0 | 4.15% |
| 8 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 85.512 | 11,968 | 7.1 | 3.76% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x64x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM...` | 53.429 | 1,536 | 34.8 | 2.35% |
| 10 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx1250` | 53.088 | 4,416 | 12.0 | 2.33% |

</details>

## Kimi-K3 without speculation

| Metric | Batch | MI355X (gfx950) | MI455X (gfx1250) | MI455X advantage |
|---|---:|---:|---:|---:|
| TTFT p50 (ms) | 1 | 2,546.5 | 1,714.0 | 1.49x |
| TPOT p50 (ms) | 1 | 11.8 | 8.6 | 1.37x |
| steady decode TPOT (ms) | 1 | 11.7 | 8.5 | 1.37x |
| decode tok/s per user | 1 | 85.4 | 117.2 | 1.37x |
| aggregate output tok/s | 1 | 59.3 | 83.4 | 1.41x |
| TTFT p50 (ms) | 16 | 20,735.7 | 13,402.7 | 1.55x |
| TPOT p50 (ms) | 16 | 61.2 | 38.5 | 1.59x |
| steady decode TPOT (ms) | 16 | 24.0 | 14.2 | 1.69x |
| decode tok/s per user | 16 | 41.6 | 70.3 | 1.69x |
| aggregate output tok/s | 16 | 156.0 | 245.1 | 1.57x |

Accumulated GPU kernel time per bucket, MI355X over MI455X, so
above 1.00x means MI455X is ahead. The last row compares the
measured TTFT p50 and steady decode TPOT from the table above; it
is not a total of the rows.

| Category | prefill c1 | prefill c16 | decode c1 | decode c16 |
|---|---:|---:|---:|---:|
| KDA state scan | 1.07x | 1.12x | — | — |
| KDA other | 1.70x | 1.70x | 1.18x | 1.50x |
| MoE | 2.02x | 2.05x | 0.98x | 1.45x |
| input projections | 1.36x | 1.36x | 1.72x | 1.85x |
| dense GEMM | 1.34x | 1.33x | 1.41x | 1.66x |
| attention | 1.97x | 1.82x | 2.19x | 2.17x |
| AttnRes | 1.32x | 1.32x | 1.25x | 1.46x |
| add3 | 2.75x | 2.79x | — | — |
| quantize | 2.38x | 2.50x | 1.04x | 3.83x |
| rmsnorm | 1.75x | 1.75x | 0.96x | 1.24x |
| top-k and sort | 1.16x | 1.15x | — | — |
| sampling | 1.28x | 1.30x | 1.23x | 1.59x |
| elementwise | 1.37x | 1.38x | 1.15x | 1.82x |
| other | 1.46x | 1.55x | 0.87x | 0.87x |
| all kernels | 1.58x | 1.58x | 1.33x | 1.62x |
| **End-to-end** | **1.49x** (TTFT) | **1.55x** (TTFT) | **1.37x** (steady TPOT) | **1.69x** (steady TPOT) |

<details><summary>MI355X (gfx950) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel` | 394.377 | 644 | 612.4 | 16.28% |
| 2 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel` | 249.093 | 644 | 386.8 | 10.28% |
| 3 | input projections | `gluon_latent_input_largem_gfx950` | 225.812 | 552 | 409.1 | 9.32% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 217.615 | 414 | 525.6 | 8.98% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950` | 186.374 | 1,309 | 142.4 | 7.69% |
| 6 | attention | `gluon_mla_prefill_8wave_gfx950` | 154.425 | 336 | 459.6 | 6.38% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 147.094 | 558 | 263.6 | 6.07% |
| 8 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950` | 136.051 | 1,254 | 108.5 | 5.62% |
| 9 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950` | 116.134 | 552 | 210.4 | 4.79% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel` | 98.002 | 552 | 177.5 | 4.05% |

</details>

<details><summary>MI355X (gfx950) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `gluon_mxfp4_moe_stage1_kernel` | 6,012.049 | 9,016 | 666.8 | 15.90% |
| 2 | MoE | `gluon_mxfp4_moe_stage2_1x2_kernel` | 3,853.887 | 9,016 | 427.4 | 10.19% |
| 3 | input projections | `gluon_latent_input_largem_gfx950` | 3,663.459 | 9,016 | 406.3 | 9.69% |
| 4 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 3,516.251 | 6,787 | 518.1 | 9.30% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950` | 2,957.136 | 18,326 | 161.4 | 7.82% |
| 6 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 2,383.281 | 9,114 | 261.5 | 6.30% |
| 7 | attention | `gluon_mla_prefill_8wave_gfx950` | 2,352.043 | 5,808 | 405.0 | 6.22% |
| 8 | dense GEMM | `gluon_mm_a16w16_prefill_gfx950` | 2,186.812 | 20,770 | 105.3 | 5.78% |
| 9 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx950` | 1,867.391 | 7,866 | 237.4 | 4.94% |
| 10 | MoE | `gluon_mxfp4_moe_stage2_reduce_kernel` | 1,591.955 | 9,016 | 176.6 | 4.21% |

</details>

<details><summary>MI355X (gfx950) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | input projections | `gluon_latent_input_decode_gfx950` | 105.460 | 5,888 | 17.9 | 13.11% |
| 2 | AttnRes | `gluon_linear_attnres_partials_gfx950` | 96.913 | 5,440 | 17.8 | 12.04% |
| 3 | attention | `_mla_decode_gluon` | 94.143 | 1,536 | 61.3 | 11.70% |
| 4 | MoE | `_warp_decode_precomputed_situ_stage1_kernel` | 69.835 | 5,888 | 11.9 | 8.68% |
| 5 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_D...` | 62.785 | 5,952 | 10.5 | 7.80% |
| 6 | dense GEMM | `_rowcta_gemv_add3_kernel` | 58.900 | 5,888 | 10.0 | 7.32% |
| 7 | AttnRes | `_attnres_combine_kernel` | 54.641 | 11,840 | 4.6 | 6.79% |
| 8 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel` | 34.148 | 5,888 | 5.8 | 4.24% |
| 9 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx950` | 32.746 | 4,416 | 7.4 | 4.07% |
| 10 | MoE | `_decode_sigmoid_bias_topk_kernel` | 28.357 | 5,888 | 4.8 | 3.52% |

</details>

<details><summary>MI355X (gfx950) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_warp_decode_precomputed_situ_stage1_kernel` | 351.700 | 5,888 | 59.7 | 22.42% |
| 2 | MoE | `_warp_decode_stage2_fp8_mxfp4_kernel` | 228.376 | 5,888 | 38.8 | 14.56% |
| 3 | dense GEMM | `gluon_mm_a16w16_medium_gfx950` | 178.235 | 19,456 | 9.2 | 11.36% |
| 4 | attention | `_mla_decode_gluon` | 134.700 | 1,536 | 87.7 | 8.59% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx950` | 115.793 | 11,968 | 9.7 | 7.38% |
| 6 | input projections | `gluon_latent_input_small_batch_gfx950` | 112.248 | 5,888 | 19.1 | 7.15% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_...` | 93.843 | 4,416 | 21.3 | 5.98% |
| 8 | MoE | `gluon_sigmoid_bias_topk_gfx950` | 60.082 | 5,888 | 10.2 | 3.83% |
| 9 | quantize | `_dynamic_fp8_single_pass_kernel` | 57.919 | 5,888 | 9.8 | 3.69% |
| 10 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx950` | 46.056 | 4,416 | 10.4 | 2.94% |

</details>

<details><summary>MI455X (gfx1250) prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel` | 434.300 | 3,300 | 131.6 | 28.41% |
| 2 | MoE | `_matmul` | 283.593 | 1,104 | 256.9 | 18.55% |
| 3 | input projections | `gluon_latent_input_largem_gfx1250` | 163.816 | 552 | 296.8 | 10.71% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 140.794 | 1,309 | 107.6 | 9.21% |
| 5 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250` | 108.805 | 552 | 197.1 | 7.12% |
| 6 | attention | `gluon_mla_prefill_gfx1250` | 77.950 | 360 | 216.5 | 5.10% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250` | 35.710 | 552 | 64.7 | 2.34% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel` | 34.129 | 644 | 53.0 | 2.23% |
| 9 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel` | 21.430 | 644 | 33.3 | 1.40% |
| 10 | MoE | `_matmul_decode` | 21.121 | 184 | 114.8 | 1.38% |

</details>

<details><summary>MI455X (gfx1250) prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | dense GEMM | `_wmma_tdm_dense_largem_kernel` | 6,862.757 | 46,944 | 146.2 | 28.69% |
| 2 | MoE | `_matmul` | 4,623.570 | 18,032 | 256.4 | 19.33% |
| 3 | input projections | `gluon_latent_input_largem_gfx1250` | 2,679.575 | 9,016 | 297.2 | 11.20% |
| 4 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 2,239.676 | 18,327 | 122.2 | 9.36% |
| 5 | KDA state scan | `gluon_kda_paged_prefill_state_scan_gfx1250` | 1,674.699 | 7,935 | 211.1 | 7.00% |
| 6 | attention | `gluon_mla_prefill_gfx1250` | 1,292.344 | 5,880 | 219.8 | 5.40% |
| 7 | KDA other | `gluon_kda_paged_prefill_preprocess_gfx1250` | 553.477 | 7,935 | 69.8 | 2.31% |
| 8 | MoE | `_weighted_topk_reduce_gfx1250_kernel` | 542.250 | 9,108 | 59.5 | 2.27% |
| 9 | MoE | `_precomputed_topk_route_large_stage2_gfx1250_kernel` | 304.659 | 9,016 | 33.8 | 1.27% |
| 10 | add3 | `_add3_kernel` | 261.987 | 9,016 | 29.1 | 1.10% |

</details>

<details><summary>MI455X (gfx1250) decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode` | 81.644 | 11,776 | 6.9 | 13.45% |
| 2 | AttnRes | `gluon_linear_attnres_partials_gfx1250` | 73.061 | 5,440 | 13.4 | 12.03% |
| 3 | dense GEMM | `_rowcta_gemv_kernel` | 55.317 | 12,544 | 4.4 | 9.11% |
| 4 | AttnRes | `_attnres_combine_kernel` | 46.666 | 11,840 | 3.9 | 7.69% |
| 5 | input projections | `gluon_latent_input_decode_gfx1250` | 45.733 | 5,888 | 7.8 | 7.53% |
| 6 | dense GEMM | `_rowcta_gemv_add3_kernel` | 39.298 | 5,888 | 6.7 | 6.47% |
| 7 | MoE | `_decode_sigmoid_bias_topk_kernel` | 29.278 | 5,888 | 5.0 | 4.82% |
| 8 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx1250` | 27.773 | 4,416 | 6.3 | 4.57% |
| 9 | MoE | `_precomputed_topk_route_m1_canonical_gfx1250_kernel` | 23.750 | 5,888 | 4.0 | 3.91% |
| 10 | attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_...` | 21.075 | 1,536 | 13.7 | 3.47% |

</details>

<details><summary>MI455X (gfx1250) decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_matmul_decode` | 326.129 | 11,776 | 27.7 | 33.73% |
| 2 | dense GEMM | `_wmma_tdm_dense_m16_kernel` | 119.571 | 19,520 | 6.1 | 12.37% |
| 3 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 79.106 | 11,968 | 6.6 | 8.18% |
| 4 | MoE | `_precomputed_topk_route_small_m_gfx1250_kernel` | 64.554 | 5,888 | 11.0 | 6.68% |
| 5 | attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_1_TILE_SIZE_64_KV_LORA_RANK_512_...` | 58.663 | 1,536 | 38.2 | 6.07% |
| 6 | input projections | `gluon_latent_input_decode_gfx1250` | 53.436 | 5,888 | 9.1 | 5.53% |
| 7 | dense GEMM | `_wmma_tdm_add3_m16_kernel` | 40.303 | 5,888 | 6.8 | 4.17% |
| 8 | elementwise | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_...` | 35.187 | 9,088 | 3.9 | 3.64% |
| 9 | KDA other | `gluon_kda_fused_paged_decode_vmajor_gfx1250` | 30.629 | 4,416 | 6.9 | 3.17% |
| 10 | MoE | `_kimi3_sigmoid_bias_topk_kernel` | 29.965 | 5,888 | 5.1 | 3.10% |

</details>

## Kimi-K3: EAGLE3 against no speculation

Each architecture's EAGLE3 run over its own run without
speculation, from the two sections above. Above 1.00x means EAGLE3
is ahead.

| Metric | Batch | MI355X off | MI355X EAGLE3 | MI355X gain | MI455X off | MI455X EAGLE3 | MI455X gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTFT p50 (ms) | 1 | 2,546.5 | 2,547.6 | 1.00x | 1,714.0 | 1,727.7 | 0.99x |
| TPOT p50 (ms) | 1 | 11.8 | 4.3 | 2.75x | 8.6 | 2.9 | 2.97x |
| steady decode TPOT (ms) | 1 | 11.7 | 4.2 | 2.78x | 8.5 | 2.8 | 3.02x |
| decode tok/s per user | 1 | 85.4 | 237.6 | 2.78x | 117.2 | 353.4 | 3.02x |
| aggregate output tok/s | 1 | 59.3 | 106.6 | 1.80x | 83.4 | 157.7 | 1.89x |
| TTFT p50 (ms) | 16 | 20,735.7 | 20,861.6 | 0.99x | 13,402.7 | 13,537.7 | 0.99x |
| TPOT p50 (ms) | 16 | 61.2 | 48.9 | 1.25x | 38.5 | 33.9 | 1.14x |
| steady decode TPOT (ms) | 16 | 24.0 | 11.3 | 2.14x | 14.2 | 9.2 | 1.54x |
| decode tok/s per user | 16 | 41.6 | 88.8 | 2.14x | 70.3 | 108.2 | 1.54x |
| aggregate output tok/s | 16 | 156.0 | 176.8 | 1.13x | 245.1 | 262.7 | 1.07x |

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
  --commit-date 2026-10-05 \
  --output-dir emulate_rank0/results/arch_compare_20261005_9b707d56
```
