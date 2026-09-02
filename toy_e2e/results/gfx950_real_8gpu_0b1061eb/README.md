# Kimi-K3 gfx950 real 8-GPU result at `0b1061eb`

## Status

- Collection date: 2026-09-02 UTC
- Target: gfx950 real 8-GPU TP8/EP1
- Overall status: **complete**
- Performance cases: C1 pass, C16 pass
- Stage profiles: C1 prefill pass, C1 decode pass, C16 prefill pass,
  C16 decode pass
- Missing or invalid data: none

## Software and hardware setup

| Field | Value |
|---|---|
| Device | 8× AMD Instinct MI355X |
| Architecture | `gfx950:sramecc+:xnack-` |
| Physical GPUs / ranks | 8 / 8 |
| Measurement environment | physical |
| Host / container | `smci355-ccs-aus-n15-05` / `zhewenyu/kimi-k3-e2e:tokenspeed-0b1061eb` |
| OS | Linux 6.8.0-84 host, Ubuntu 24.04 container |
| ROCm / HIP | 7.2 / `7.2.26015` |
| PyTorch | `2.11.0+rocm7.2` |
| Transformers | `5.12.0` |
| Triton package / module | TokenSpeed Triton / `3.6.0` |
| EvalScope | `1.9.1` |
| TokenSpeed commit | `0b1061eb9fe1df36a4e48e5c9c291cd753af9e89` |
| Kimi-K3 revision | `eaf5a944bfc8c57438bbce226feef9f6bdbdaae1` |

## Workload and topology

| Field | Value |
|---|---|
| Checkpoint | `/data/models/sunkist`, full source safetensors |
| Prompt / output | 4096 / 1024 tokens |
| Concurrency | 1 and 16 |
| Prefill budget | 8192 tokens |
| Attention / dense / MoE / EP | TP8 / TP8 / TP8 / EP1 |
| KV cache | FP8 E4M3, 3,197,056 scheduler-visible tokens |
| Prefix cache / host KV | disabled / disabled |
| Sampling | greedy streaming, `ignore_eos=true` |
| Prompt source | EvalScope random text, seed 1, exact 4096-token re-encode |
| Warmup / measured requests | C1: 1 / 3; C16: 16 / 48 |
| Decode graphs / scheduling | buckets 1/2/4/8/16; production overlap scheduler |
| Performance measurement | graph decode, eager prefill, EvalScope OpenAI completions |
| Hotspot measurement | separate otherwise-matched fully eager server |

All eight physical ranks execute with RCCL plus Iris all-reduce collectives.
Decode graphs cover batches 1, 2, 4, 8, and 16.

## Correctness

| C | Requests or sequences completed | Exact input length | Exact output length | Failures | Status |
|---:|---:|---:|---:|---:|---|
| 1 | 3/3 after 1 warmup | 4096 | 1024 | 0 | pass |
| 16 | 48/48 after 16 warmups | 4096 | 1024 | 0 | pass |

EvalScope emitted speculative-decoding fields even though speculation was
disabled. Those semantically invalid fields are intentionally absent from
`result.json`.

## Unprofiled performance

| C | First-token p50 / p90 | Primary decode p50 / p90 | Per-user decode | Overall output | Steady decode capacity | Scope |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 405.61 / 405.81 ms | 12.36 / 12.49 ms | 80.93 tok/s | 78.22 tok/s | 78.50 tok/s | real service |
| 16 | 4,596.48 / 4,727.20 ms | 24.28 / 27.60 ms | 41.18 tok/s | 556.13 tok/s | 816.84 tok/s | real service |

Primary decode is request-level TPOT. Overall output is all completed output
tokens divided by measured wall time, including prefill and wave boundaries.
Steady capacity is EvalScope's final 30-second completion window.

The toy result now matches graph buckets, request counts, warmup waves, and
rolling 4K/1K execution. It does not match prompt text, physical collectives,
HTTP, or full-model MoE semantics, so the two measurements remain diagnostic
rather than interchangeable.

## Stage hotspot summary

These separate eager traces use the same classifier and CSV schema as the toy
one-GPU result. GPU time is summed across all eight rank traces.

| C | Stage | Ranks | Forwards | Summed GPU time | Communication | MoE | KDA | MLA / attention | GEMM / quant | Other |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | prefill | 8 | 1 | 2,896.51 ms | 21.41% | 35.16% | 12.20% | 7.14% | 16.62% | 7.46% |
| 1 | decode | 8 | 64 | 52,919.97 ms | 88.55% | 1.51% | 2.04% | 2.88% | 3.41% | 1.61% |
| 16 | prefill | 8 | 8 | 45,103.51 ms | 22.81% | 24.90% | 11.68% | 11.51% | 15.36% | 13.74% |
| 16 | decode | 8 | 64 | 113,390.38 ms | 62.52% | 11.78% | 5.64% | 4.02% | 10.23% | 5.80% |

`Other` combines unclassified, elementwise/reduction, normalization, cache,
and sampling categories. Rank-summed GPU time is not wall time.

## Top exact kernels

The top 10 kernels for each setting are ranked by total profiled GPU duration
summed across all eight ranks. Names are copied verbatim from the GPU profiler;
they are not model-component or layer labels.

| C | Stage | Order | Exact kernel name | Calls | Total across ranks | GPU share | Average call |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | prefill | 1 | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1,528 | 601.47 ms | 20.77% | 393.63 µs |
| 1 | prefill | 2 | `gluon_mxfp4_moe_stage1_kernel` | 736 | 496.32 ms | 17.14% | 674.35 µs |
| 1 | prefill | 3 | `gluon_mxfp4_moe_stage2_1x2_kernel` | 736 | 375.09 ms | 12.95% | 509.64 µs |
| 1 | prefill | 4 | `_packed_input_projections_kernel` | 736 | 216.36 ms | 7.47% | 293.97 µs |
| 1 | prefill | 5 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 552 | 173.75 ms | 6.00% | 314.76 µs |
| 1 | prefill | 6 | `_attn_res_rmsnorm_kernel` | 1,504 | 149.82 ms | 5.17% | 99.61 µs |
| 1 | prefill | 7 | `_mfma_lds_largem_kernel` | 736 | 113.18 ms | 3.91% | 153.78 µs |
| 1 | prefill | 8 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1` | 1,488 | 100.63 ms | 3.47% | 67.63 µs |
| 1 | prefill | 9 | `gluon_mxfp4_moe_stage2_reduce_kernel` | 736 | 85.20 ms | 2.94% | 115.76 µs |
| 1 | prefill | 10 | `_state_scan_fwd_kernel` | 552 | 82.89 ms | 2.86% | 150.16 µs |
| 1 | decode | 1 | `iris_stage_one_shot_allreduce_residual_attnres_gluon_kernel` | 43,345 | 18,026.47 ms | 34.06% | 415.88 µs |
| 1 | decode | 2 | `iris_stage_one_shot_allreduce_kernel` | 5,099 | 16,293.80 ms | 30.79% | 3,195.49 µs |
| 1 | decode | 3 | `iris_reduce_symmetric_gluon_kernel` | 46,914 | 12,412.30 ms | 23.45% | 264.58 µs |
| 1 | decode | 4 | `_latent_input_decode_kernel` | 46,914 | 828.04 ms | 1.56% | 17.65 µs |
| 1 | decode | 5 | `_linear_attnres_partials_kernel` | 43,345 | 750.10 ms | 1.42% | 17.31 µs |
| 1 | decode | 6 | `_warp_decode_precomputed_situ_stage1_kernel` | 46,914 | 564.66 ms | 1.07% | 12.04 µs |
| 1 | decode | 7 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x128_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR0_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA256_LBSPPB256_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR0_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM8_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4` | 47,425 | 521.72 ms | 0.99% | 11.00 µs |
| 1 | decode | 8 | `_rowcta_gemv_add3_kernel` | 46,914 | 474.98 ms | 0.90% | 10.12 µs |
| 1 | decode | 9 | `_warp_decode_stage2_fp8_mxfp4_kernel` | 46,914 | 270.92 ms | 0.51% | 5.77 µs |
| 1 | decode | 10 | `_attnres_combine_kernel` | 50,993 | 266.14 ms | 0.50% | 5.22 µs |
| 16 | prefill | 1 | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 13,640 | 10,251.01 ms | 22.73% | 751.54 µs |
| 16 | prefill | 2 | `gluon_mxfp4_moe_stage1_kernel` | 6,624 | 5,592.55 ms | 12.40% | 844.29 µs |
| 16 | prefill | 3 | `_attn_res_rmsnorm_kernel` | 13,480 | 4,627.23 ms | 10.26% | 343.27 µs |
| 16 | prefill | 4 | `_packed_input_projections_kernel` | 6,624 | 3,423.87 ms | 7.59% | 516.89 µs |
| 16 | prefill | 5 | `gluon_mxfp4_moe_stage2_1x2_kernel` | 6,624 | 3,354.01 ms | 7.44% | 506.34 µs |
| 16 | prefill | 6 | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1})` | 10,280 | 3,218.13 ms | 7.13% | 313.05 µs |
| 16 | prefill | 7 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 4,968 | 2,579.74 ms | 5.72% | 519.27 µs |
| 16 | prefill | 8 | `_mfma_lds_largem_kernel` | 6,624 | 1,810.45 ms | 4.01% | 273.32 µs |
| 16 | prefill | 9 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT224x256x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA128_LBSPPB1024_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT7_8_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA1_VWB8_WSGRA0_WSGRB0_WS64_WG32_8_1` | 13,392 | 1,534.91 ms | 3.40% | 114.61 µs |
| 16 | prefill | 10 | `gluon_mxfp4_moe_stage2_reduce_kernel` | 6,624 | 1,239.99 ms | 2.75% | 187.20 µs |
| 16 | decode | 1 | `iris_stage_one_shot_allreduce_kernel` | 48,351 | 50,520.44 ms | 44.55% | 1,044.87 µs |
| 16 | decode | 2 | `iris_reduce_symmetric_two_stage_gluon_kernel` | 46,814 | 11,846.94 ms | 10.45% | 253.06 µs |
| 16 | decode | 3 | `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 13,126 | 8,526.76 ms | 7.52% | 649.61 µs |
| 16 | decode | 4 | `gluon_mxfp4_moe_stage1_kernel` | 5,888 | 5,095.53 ms | 4.49% | 865.41 µs |
| 16 | decode | 5 | `_attn_res_rmsnorm_kernel` | 107,133 | 3,637.83 ms | 3.21% | 33.96 µs |
| 16 | decode | 6 | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1})` | 9,144 | 3,216.98 ms | 2.84% | 351.81 µs |
| 16 | decode | 7 | `_packed_input_projections_kernel` | 5,888 | 3,206.19 ms | 2.83% | 544.53 µs |
| 16 | decode | 8 | `gluon_mxfp4_moe_stage2_1x2_kernel` | 5,888 | 2,977.40 ms | 2.63% | 505.67 µs |
| 16 | decode | 9 | `_warp_decode_precomputed_situ_stage1_kernel` | 46,840 | 2,687.39 ms | 2.37% | 57.37 µs |
| 16 | decode | 10 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB4_GSU0_GSUAMB_GLS0_ISA950_IU1_K1_LDSTI0_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_13_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK3_SKFTR0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGRO0_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG64_4_1` | 4,416 | 2,407.88 ms | 2.12% | 545.26 µs |

## Incomplete or failed work

| Case | Stage | Status | Error or reason | Raw artifact |
|---|---|---|---|---|
| none | all | complete | none | `result.json`, `logs/`, and `hotspots/` |

## Exact commands

### Performance

```bash
ts serve --model /data/models/sunkist --served-model-name kimi-k3 \
  --tp 8 --ep-size 1 --attention-backend mla --moe-backend auto \
  --kv-cache-dtype fp8 --mm-encoder-tp-mode data \
  --max-model-len 8192 --max-num-seqs 16 \
  --max-prefill-tokens 8192 --chunked-prefill-size 8192 \
  --max-cudagraph-capture-size 16 --cudagraph-capture-sizes 1 2 4 8 16 \
  --disable-prefill-graph --gpu-memory-utilization 0.92 \
  --trust-remote-code --sampling-backend greedy \
  --disable-kvstore --kvstore-ratio 0 --no-enable-prefix-caching \
  --enable-cache-report --enable-log-request-stats --enable-metrics \
  --host 127.0.0.1 --port 21000 --policy round_robin \
  --engine-startup-timeout 3000 --gateway-startup-timeout 600

EVALSCOPE_BIN=/data/zhewenyu-kimi-k3-gfx950-bench/evalscope-1.9.1/bin/evalscope \
bash toy_e2e/scripts/run_evalscope_4k1k.sh \
  http://127.0.0.1:21000/v1/completions \
  /data/results/kimi-k3-real-tp8ep1-gfx950-0b1061eb/evalscope \
  /data/models/sunkist

python3 toy_e2e/scripts/collect_real_serving_results.py \
  --input /data/results/kimi-k3-real-tp8ep1-gfx950-0b1061eb/evalscope \
  --output result.json \
  --tokenspeed-revision 0b1061eb9fe1df36a4e48e5c9c291cd753af9e89 \
  --model-revision eaf5a944bfc8c57438bbce226feef9f6bdbdaae1
```

### Stage profiles

The server used the same command with `--enforce-eager` and
`TOKENSPEED_KERNEL_PROFILE_BACKEND=roctracer`. The four capture invocations
were:

```bash
for C in 1 16; do
  python3 toy_e2e/scripts/profile_serving_stages.py \
    --output-dir "/data/results/kimi-k3-real-tp8ep1-gfx950-0b1061eb/eager-profile/c${C}/prefill" \
    --profile-id "c${C}-prefill" --concurrency "$C" --capture prefill
  python3 toy_e2e/scripts/profile_serving_stages.py \
    --output-dir "/data/results/kimi-k3-real-tp8ep1-gfx950-0b1061eb/eager-profile/c${C}/decode" \
    --profile-id "c${C}-decode" --concurrency "$C" --capture decode \
    --profile-steps 64
done
```

### Hotspot aggregation

```bash
python3 toy_e2e/scripts/summarize_gpu_hotspots.py \
  --input /data/results/kimi-k3-real-tp8ep1-gfx950-0b1061eb/eager-profile \
  --top-k 15 --csv-dir hotspots/csv --output hotspots/hotspots.json
```

See [`../../RUNBOOK.md`](../../RUNBOOK.md) for the complete server flags,
container environment, and checks.

## Raw artifacts

- Primary result JSON: [`result.json`](result.json)
- Complete run log: unavailable; graph server log retained instead
- Hotspot summary: [`hotspots/hotspots.json`](hotspots/hotspots.json)
- Exact-name CSVs: [`hotspots/csv/`](hotspots/csv/)
- Raw traces and manifests:
  `/data/results/kimi-k3-real-tp8ep1-gfx950-0b1061eb/eager-profile/`
- Service logs: [`logs/graph_server.log`](logs/graph_server.log) and
  [`logs/eager_profile_server.log`](logs/eager_profile_server.log)
- EvalScope outputs:
  `/data/results/kimi-k3-real-tp8ep1-gfx950-0b1061eb/evalscope/`

## Conclusions and limitations

- Real output throughput is 78.22 tok/s at C1 and 556.13 tok/s at C16; the
  saturated final window reaches 816.84 tok/s.
- MoE and communication dominate prefill, totaling 56.58% at C1 and 47.71%
  at C16.
- Communication dominates decode: 88.55% at C1 and 62.52% at C16.
- The eager hotspot server identifies optimization targets but does not time
  graph replay. EvalScope is the authoritative performance source.
- FP8 KV scaling factors were unavailable and defaulted to 1.0. This is a
  performance result, not a model-quality qualification.

Profiled GPU-duration sums are hotspot weights, not critical-path latency.
Collective kernels include peer-wait residency, and profiler instrumentation
perturbs execution.
