# Kimi-K3 gfx1250 toy 1-GPU result at `0b1061eb`

## Status

- Collection date: 2026-09-03 UTC
- Target: gfx1250 toy 1-GPU logical TP8 rank 0
- Overall status: **complete**
- Performance cases: C1 pass, C16 pass
- Stage profiles: C1 prefill pass, C1 decode pass, C16 prefill pass,
  C16 decode pass
- Missing or invalid data: none

## Software and hardware setup

| Field | Value |
|---|---|
| Device | AMD Instinct MI450 (PyTorch: `AMD Radeon Graphics`) |
| Architecture | `gfx1250` |
| Physical GPUs / ranks | 1 / 1 |
| Measurement environment | physical |
| Host / container | `heliosr-1b114-d04-1` / `tokenspeed-kimi-gfx1250:tokenspeed-0b1061eb@sha256:15d09a4f39b938ec5c091e1b22128bf0aaa9246bdaefa803506fa27b44a49503` |
| OS | Ubuntu 24.04 container, Linux 7.1.0 |
| ROCm / HIP | 7.15 / `7.15.0` |
| PyTorch | `2.11.0+rocm7.15.0a20260728` |
| Transformers | `5.12.0` |
| Triton package / module | TokenSpeed Triton / `3.8.0` |
| TokenSpeed scheduler | `0.1.11` |
| EvalScope | unavailable; direct logical-rank harness |
| TokenSpeed commit | `0b1061eb9fe1df36a4e48e5c9c291cd753af9e89` |
| Kimi-K3 revision | `eaf5a944bfc8c57438bbce226feef9f6bdbdaae1` |

## Workload and topology

| Field | Value |
|---|---|
| Checkpoint | `/data/models/kimi-k3-tp8ep1-rank0`, portable raw-rank-state |
| Prompt / output | 4096 / 1024 tokens |
| Concurrency | 1 and 16 |
| Prefill budget | 8192 tokens |
| Attention / dense / MoE / EP | TP8 / TP8 / TP8 / EP1 |
| KV cache | FP8 E4M3, 32 GiB, 2,334,336 scheduler-visible tokens |
| Prefix cache / host KV | disabled / disabled |
| Sampling | rank-local greedy, `ignore_eos=true` |
| Prompt source | deterministic varied IDs, seed 7, vocabulary range 160,000 |
| Warmup / measured requests | C1: 1 / 3; C16: 16 / 48 |
| Decode graphs / scheduling | buckets 1/2/4/8/16; depth-1 dispatch/commit overlap |
| Performance measurement | complete rolling `ModelExecutor` CUDA-graph workload |
| Hotspot measurement | separate eager PyTorch/roctracer run; varied prompts and deterministic decode input ID 1 |

One physical GPU executes all 93 layers for logical TP8 rank 0. Rank-spanning
collectives record shapes and bytes but use local substitutes, so no RCCL
kernel appears in this result.

## Correctness

| C | Requests or sequences completed | Exact input length | Exact output length | Failures | Status |
|---:|---:|---:|---:|---:|---|
| 1 | 3/3 after 1 warmup | 4096 | 1024 | 0 | pass |
| 16 | 48/48 after 16 warmups | 4096 | 1024 | 0 | pass |

The result also records 93 layers, 896 local experts, the gfx1250 Gluon MoE
solution, 191.23 GiB model allocation, and a 229.72 GiB runtime peak.

## Unprofiled performance

| C | First-token p50 / p90 | Primary decode p50 / p90 | Per-user decode | Overall output | Steady decode capacity | Scope |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 659.30 / 661.28 ms | 14.654 / 14.655 ms | 68.24 tok/s | 65.44 tok/s | 68.24 tok/s | logical rolling graph B1 |
| 16 | 5,119.56 / 9,073.49 ms | 36.453 / 40.312 ms | 27.43 tok/s | 386.26 tok/s | 491.16 tok/s | logical rolling graph B16 |

Primary decode is request TPOT across all 1024 generated tokens. Overall
output includes eager prefill and three complete measured waves. Steady
capacity excludes the first decode transition, where C16 requests that finish
prefill early can wait for the remaining prompt chunks. All metrics exclude
real inter-rank communication and HTTP serving.

| C | Decode input context | Resulting context | Step p50 / p90 | Samples |
|---:|---:|---:|---:|---:|
| 1 | 4097 | 4098 | 14.667 / 14.675 ms | 3 |
| 1 | 4224 | 4225 | 14.663 / 14.673 ms | 3 |
| 1 | 4352 | 4353 | 14.669 / 14.671 ms | 3 |
| 1 | 4480 | 4481 | 14.646 / 14.658 ms | 3 |
| 1 | 4608 | 4609 | 14.667 / 14.671 ms | 3 |
| 1 | 4736 | 4737 | 14.677 / 14.685 ms | 3 |
| 1 | 4864 | 4865 | 14.677 / 14.677 ms | 3 |
| 1 | 4992 | 4993 | 14.670 / 14.691 ms | 3 |
| 1 | 5119 | 5120 | 14.672 / 14.684 ms | 3 |
| 16 | 4097 | 4098 | 4,001.135 / 7,959.249 ms | 48 |
| 16 | 4224 | 4225 | 32.796 / 32.817 ms | 48 |
| 16 | 4352 | 4353 | 32.819 / 32.819 ms | 48 |
| 16 | 4480 | 4481 | 32.741 / 32.820 ms | 48 |
| 16 | 4608 | 4609 | 32.776 / 32.830 ms | 48 |
| 16 | 4736 | 4737 | 32.824 / 32.834 ms | 48 |
| 16 | 4864 | 4865 | 32.800 / 32.801 ms | 48 |
| 16 | 4992 | 4993 | 32.775 / 32.832 ms | 48 |
| 16 | 5119 | 5120 | 32.554 / 32.627 ms | 48 |

The large C16 sample at 4097 is intentional: TTFT occurs as each request's
prefill chunk completes, but the first decode batch waits for all 16 prompts.
Request TPOT includes this scheduling gap; steady capacity does not.

## Stage hotspot summary

Use the separate eager traces, never the unprofiled performance run.
Percentages divide category duration by summed GPU-kernel duration across all
captured physical ranks.

| C | Stage | Ranks | Forwards | Summed GPU time | Communication | MoE | KDA | MLA / attention | GEMM / quant | Other |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | prefill | 1 | 1 | 612.91 ms | 0.00% | 0.08% | 6.15% | 5.98% | 74.32% | 13.47% |
| 1 | decode | 1 | 64 | 1,105.16 ms | 0.00% | 1.35% | 9.95% | 12.51% | 32.87% | 43.33% |
| 16 | prefill | 1 | 8 | 8,869.79 ms | 0.00% | 0.07% | 5.67% | 6.45% | 76.28% | 11.52% |
| 16 | decode | 1 | 64 | 2,253.97 ms | 0.00% | 0.67% | 5.18% | 10.58% | 47.50% | 36.07% |

`Other` includes normalization, elementwise/reduction, cache, sampling, and
unclassified kernels. The machine-readable hotspot JSON and exact-name CSVs
are authoritative.

## Top exact kernels

Include exactly the top ten GPU kernels by total profiled GPU duration for
every concurrency/stage pair. These must be exact profiler kernel names, not
model component, layer, or module names.

| C | Stage | Order | Exact kernel name | Calls | Total across ranks | GPU share | Average call |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | prefill | 1 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1` | 514 | 359.24 ms | 58.61% | 698.90 µs |
| 1 | prefill | 2 | `_matmul` | 184 | 96.27 ms | 15.71% | 523.21 µs |
| 1 | prefill | 3 | `_attn_res_rmsnorm_kernel` | 187 | 34.25 ms | 5.59% | 183.17 µs |
| 1 | prefill | 4 | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int, bool)#1})` | 92 | 23.50 ms | 3.83% | 255.44 µs |
| 1 | prefill | 5 | `_packed_input_projections_kernel` | 92 | 22.59 ms | 3.69% | 245.59 µs |
| 1 | prefill | 6 | `_state_scan_fwd_kernel` | 69 | 11.30 ms | 1.84% | 163.73 µs |
| 1 | prefill | 7 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)` | 92 | 11.25 ms | 1.83% | 122.24 µs |
| 1 | prefill | 8 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>)` | 92 | 5.69 ms | 0.93% | 61.90 µs |
| 1 | prefill | 9 | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*)` | 368 | 5.04 ms | 0.82% | 13.70 µs |
| 1 | prefill | 10 | `void at::native::mbtopk::computeBlockwiseWithinKCounts<unsigned int, float>(unsigned int*, short*, unsigned int*, unsigned int, int, bool, unsigned int*, float*, unsigned int*, unsigned int*, unsigned int*, unsigned int)` | 368 | 4.04 ms | 0.66% | 10.97 µs |
| 1 | decode | 1 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1` | 16,448 | 169.61 ms | 15.35% | 10.31 µs |
| 1 | decode | 2 | `_matmul_decode` | 11,776 | 87.37 ms | 7.91% | 7.42 µs |
| 1 | decode | 3 | `_packed_input_projections_kernel` | 5,888 | 81.79 ms | 7.40% | 13.89 µs |
| 1 | decode | 4 | `_rowcta_gemv_kernel` | 5,952 | 58.00 ms | 5.25% | 9.74 µs |
| 1 | decode | 5 | `_attnres_partial_dual_kernel` | 5,440 | 43.75 ms | 3.96% | 8.04 µs |
| 1 | decode | 6 | `_attnres_combine_kernel` | 11,840 | 43.15 ms | 3.90% | 3.64 µs |
| 1 | decode | 7 | `_rowcta_gemv_add3_kernel` | 5,888 | 40.29 ms | 3.65% | 6.84 µs |
| 1 | decode | 8 | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1})` | 17,664 | 35.19 ms | 3.18% | 1.99 µs |
| 1 | decode | 9 | `void at::native::vectorized_elementwise_kernel<4, at::native::compare_scalar_kernel<int>(at::TensorIteratorBase&, at::native::(anonymous namespace)::OpType, int)::{lambda(int)#1}, std::array<char*, 2ul> >(int, at::native::compare_scalar_kernel<int>(at::TensorIteratorBase&, at::native::(anonymous namespace)::OpType, int)::{lambda(int)#1}, std::array<char*, 2ul>)` | 11,839 | 29.30 ms | 2.65% | 2.47 µs |
| 1 | decode | 10 | `_decode_sigmoid_bias_topk_kernel` | 5,888 | 28.54 ms | 2.58% | 4.85 µs |
| 16 | prefill | 1 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1` | 4,112 | 5,750.57 ms | 64.83% | 1,398.48 µs |
| 16 | prefill | 2 | `_matmul` | 1,472 | 1,015.51 ms | 11.45% | 689.89 µs |
| 16 | prefill | 3 | `_attn_res_rmsnorm_kernel` | 1,496 | 541.53 ms | 6.11% | 361.98 µs |
| 16 | prefill | 4 | `_packed_input_projections_kernel` | 736 | 354.87 ms | 4.00% | 482.16 µs |
| 16 | prefill | 5 | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int, bool)#1})` | 736 | 300.81 ms | 3.39% | 408.71 µs |
| 16 | prefill | 6 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)` | 736 | 139.03 ms | 1.57% | 188.90 µs |
| 16 | prefill | 7 | `_state_scan_fwd_kernel` | 552 | 105.18 ms | 1.19% | 190.55 µs |
| 16 | prefill | 8 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>)` | 736 | 86.50 ms | 0.98% | 117.53 µs |
| 16 | prefill | 9 | `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int*, unsigned int, unsigned int, int, int, unsigned int, unsigned int, unsigned int*, short*)` | 2,944 | 69.30 ms | 0.78% | 23.54 µs |
| 16 | prefill | 10 | `_fp8_quantize_kernel` | 1,664 | 55.15 ms | 0.62% | 33.15 µs |
| 16 | decode | 1 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSUAMB_GLS0_HPLR0_ISA1250_ICIW1_IU1_K1_LDSTI0_LBSPPA128_LBSPPB128_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT1_1_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR0_PKA1_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_2_1_WGMXCC1` | 29,824 | 756.46 ms | 33.56% | 25.36 µs |
| 16 | decode | 2 | `_matmul_decode` | 11,776 | 289.90 ms | 12.86% | 24.62 µs |
| 16 | decode | 3 | `_attn_res_rmsnorm_kernel` | 11,968 | 212.27 ms | 9.42% | 17.74 µs |
| 16 | decode | 4 | `void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, bool, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, float*)` | 5,888 | 97.33 ms | 4.32% | 16.53 µs |
| 16 | decode | 5 | `_packed_input_projections_kernel` | 5,888 | 86.04 ms | 3.82% | 14.61 µs |
| 16 | decode | 6 | `void at::native::radixSortKVInPlace<-2, -1, 128, 8, long, long, unsigned int>(at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, bool)` | 5,888 | 75.60 ms | 3.35% | 12.84 µs |
| 16 | decode | 7 | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1})` | 23,552 | 56.55 ms | 2.51% | 2.40 µs |
| 16 | decode | 8 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)` | 5,952 | 38.25 ms | 1.70% | 6.43 µs |
| 16 | decode | 9 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1})` | 9,024 | 33.15 ms | 1.47% | 3.67 µs |
| 16 | decode | 10 | `_kda_fused_decode_kernel` | 4,416 | 30.74 ms | 1.36% | 6.96 µs |

## Incomplete or failed work

| Case | Stage | Status | Error or reason | Raw artifact |
|---|---|---|---|---|
| none | all | complete | none | `result.json` and `hotspots/` |

## Exact commands

### Performance

```bash
python3 toy_e2e/benchmark_logical_rank.py \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --expected-arch gfx1250 \
  --tokenspeed-revision 0b1061eb9fe1df36a4e48e5c9c291cd753af9e89 \
  --model-revision eaf5a944bfc8c57438bbce226feef9f6bdbdaae1 \
  --container-image tokenspeed-kimi-gfx1250:tokenspeed-0b1061eb@sha256:15d09a4f39b938ec5c091e1b22128bf0aaa9246bdaefa803506fa27b44a49503 \
  --prompt-tokens 4096 --output-tokens 1024 \
  --concurrency 1 16 --chunked-prefill-size 8192 --cache-gib 32 \
  --warmup-waves 1 --measurement-waves 3 \
  --prompt-seed 7 --synthetic-vocabulary-size 160000 \
  --output result.json
```

### Stage profiles

```bash
TOKENSPEED_KERNEL_PROFILE_BACKEND=roctracer \
python3 toy_e2e/scripts/profile_logical_rank_stages.py \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --expected-arch gfx1250 \
  --output-dir /data/results/kimi-k3-toy-1gpu-gfx1250-0b1061eb-rolling/eager-profile \
  --tokenspeed-revision 0b1061eb9fe1df36a4e48e5c9c291cd753af9e89 \
  --model-revision eaf5a944bfc8c57438bbce226feef9f6bdbdaae1 \
  --container-image tokenspeed-kimi-gfx1250:tokenspeed-0b1061eb@sha256:15d09a4f39b938ec5c091e1b22128bf0aaa9246bdaefa803506fa27b44a49503 \
  --prompt-tokens 4096 --concurrency 1 16 \
  --chunked-prefill-size 8192 --cache-gib 32 \
  --prompt-seed 7 --synthetic-vocabulary-size 160000 \
  --decode-steps 64
```

### Hotspot aggregation

```bash
python3 toy_e2e/scripts/summarize_gpu_hotspots.py \
  --input /data/results/kimi-k3-toy-1gpu-gfx1250-0b1061eb-rolling/eager-profile \
  --top-k 15 \
  --csv-dir toy_e2e/results/gfx1250_toy_1gpu_0b1061eb/hotspots/csv \
  --output toy_e2e/results/gfx1250_toy_1gpu_0b1061eb/hotspots/hotspots.json

python3 toy_e2e/scripts/update_result_readme_hotspots.py \
  --readme toy_e2e/results/gfx1250_toy_1gpu_0b1061eb/README.md \
  --hotspots toy_e2e/results/gfx1250_toy_1gpu_0b1061eb/hotspots/hotspots.json \
  --csv-dir toy_e2e/results/gfx1250_toy_1gpu_0b1061eb/hotspots/csv \
  --profile-manifest /data/results/kimi-k3-toy-1gpu-gfx1250-0b1061eb-rolling/eager-profile/profile_manifest.json
```

See [`../../RUNBOOK.md`](../../RUNBOOK.md) for container, environment, and
validation setup.

## Raw artifacts

- Primary result JSON: [`result.json`](result.json)
- Complete run log: [`run.log`](run.log)
- Hotspot summary: [`hotspots/hotspots.json`](hotspots/hotspots.json)
- Exact-name CSVs: [`hotspots/csv/`](hotspots/csv/)
- Raw traces and manifest:
  `/data/results/kimi-k3-toy-1gpu-gfx1250-0b1061eb-rolling/eager-profile/`
- Service logs and EvalScope outputs: unavailable for a direct logical-rank run

## Conclusions and limitations

- Complete rolling output reaches 65.44 tok/s at C1 and 386.26 tok/s at C16;
  post-transition steady decode capacity is 68.24 and 491.16 tok/s.
- GEMM/quant dominates prefill GPU duration: 74.32% at C1 and 76.28% at C16.
- C1 decode is split between `Other` (43.33%) and GEMM/quant (32.87%); at C16,
  GEMM/quant is the largest category at 47.50%.
- Communication is intentionally absent. This result cannot measure RCCL,
  cross-rank synchronization, HTTP, or tokenization. Rank-local greedy outputs
  and MoE routes are not semantically valid full-TP model outputs.
- FP8 KV scaling factors were unavailable and defaulted to 1.0. This is a
  performance/scheduling result, not a model-quality qualification.

Profiled GPU-duration sums are hotspot weights, not critical-path latency.
Profiler instrumentation perturbs execution and must not replace the
unprofiled graph measurements.
