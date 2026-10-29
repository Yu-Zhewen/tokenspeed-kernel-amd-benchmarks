# Real-weight serving hotspots on MI455X at `f198a543`

Each model serves its real checkpoint over its CI job's parallel layout
on MI455X (gfx1250) GPUs, with real MoE routing and real collectives
(RCCL and Iris over the GPUs' scale-up fabric; `real_serving/README.md`
describes the nodes). The one fixed input is the accept length: with
`TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN` set, every verify step accepts
that many tokens whatever the drafter proposes, since random prompts
say little about a drafter. Kernel tables are rank 0's.

## Provenance

| Field | Value |
|---|---|
| TokenSpeed revision | `f198a543720bb417a1453611e13089a98ef69e6f` |
| Commit date (UTC) | 2026-10-29 |
| Measured | 2026-10-29 |
| Prompt / output tokens | 50,000 / 500 |
| Concurrencies | 1, 16 |
| Warmup / measured waves | 1 / 1 |
| GPU | `gfx1250`, 432.0 GiB |
| torch / tokenspeed-triton | 2.14.0+rocm10.2.0a20260923 / 3.9.10.post20261028 |
| Image | `tokenspeed-gfx1250:main-f198a543` |

## How each model was run

Each node runs one container (`real_serving/scripts/run.sh`) that owns
all of the node's GPUs and shares the host network. Inside it
`serve_bench.sh` starts `tokenspeed.cli serve`, whose gateway listens
on port 8000, and drives the benchmark client against it: a smoke
request, then the benchmark at each concurrency. Timing and profiling
are two server launches with the same command; the second runs under
`rocprofv3 --kernel-trace --marker-trace --selected-regions` with a
profiler hook in every rank, and the client opens one profiling
window per stage.

| Model | Layout | Node rank: host | Ranks per node | Profiled ranks | Fixed accept length |
|---|---|---|---:|---|---:|
| GLM-5.3-Flash MTP | TP4 (1 node, 4 GPUs) | 0: `ctheliosr-1b112-a31-2` | 4 | TP0, TP1, TP2, TP3 | 2.9 |
| DeepSeek-V4.1-Flash DSPARK | TP4 (1 node, 4 GPUs) | 0: `ctheliosr-1b112-a31-2` | 4 | TP0, TP1, TP2, TP3 | 3.9 |
| Kimi-K3 EAGLE3 | TP8 (2 nodes x 4 GPUs) | 0: `ctheliosr-1b112-a31-1`, 1: `ctheliosr-1b112-a31-2` | 4 | TP0, TP1, TP2, TP3 | 3.75 |

Every container also gets these settings, and `TOKENSPEED_HOST_IP`
set to its node's address (`real_serving/README.md` says why):

- `LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib`
- `GLOO_SOCKET_IFNAME=enp4s0np0`
- `NCCL_GRAPH_REGISTER=0`
- `NCCL_IB_GID_INDEX=1`
- `NCCL_IB_HCA=ionic_0`
- `NCCL_SOCKET_IFNAME=enp4s0np0`

### Kimi-K3 across 2 nodes

The 8 ranks form one TP8 group: TP0–TP3 on `ctheliosr-1b112-a31-1`, TP4–TP7 on `ctheliosr-1b112-a31-2`.
Every node runs the same `tokenspeed.cli serve` command except
`--node-rank`, binds `--host 0.0.0.0`, and joins the group at
node 0's `--dist-init-addr <node 0 ip>:25000`. The other
nodes serve engines only; node 0 also runs the gateway and the
client and holds the results, so the profiled ranks are node 0's
and the kernel tables are TP0's.

Each phase is its own launch, since the other nodes restart with
their phase's flags: `run.sh` with `ROLE=follower` and `NODE_RANK`
on every other node (detached), then `ROLE=leader` on node 0, which
returns when the phase ends; the follower containers are then
removed. Per phase, with `IMAGE_REF`, `OUT`, `MODELS_ROOT`, the
`NCCL_*`/`GLOO_*` settings above and `CONTAINER_LD_LIBRARY_PATH`
(which becomes the container's `LD_LIBRARY_PATH`) also set:

```bash
# node 1, ctheliosr-1b112-a31-2
MODEL=kimik3 ROLE=follower NNODES=2 NODE_RANK=1 \
HEAD_ADDR=<node 0 ip>:25000 NODE_IP=<node 1 ip> PHASES=<perf|hotspots> \
  real_serving/scripts/run.sh
# node 0, ctheliosr-1b112-a31-1
MODEL=kimik3 ROLE=leader NNODES=2 NODE_RANK=0 \
HEAD_ADDR=<node 0 ip>:25000 NODE_IP=<node 0 ip> PHASES=<perf|hotspots> \
  real_serving/scripts/run.sh
```

RCCL sees the GPUs of both a31 trays as one scale-up domain: it
reports `UALoE-enabled (aka MNNVL)` at 1.8 Tb/s, counts one node of
8 local ranks, and connects its rings `via P2P/CUMEMMNNVL`.

<details><summary>Serve commands</summary>

GLM-5.3-Flash MTP, node 0, for both phases:

```bash
TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN=2.9 python3 -m tokenspeed.cli serve /data/models/GLM-5.3-Flash \
  --language-model-only \
  --speculative-algorithm MTP \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1 \
  --served-model-name glm-5.3-flash \
  --trust-remote-code \
  --attn-tp-size 4 \
  --ep-size 1 \
  --max-model-len 65536 \
  --max-num-seqs 16 \
  --chunked-prefill-size 8192 \
  --disable-kvstore \
  --disable-prefix-caching \
  --disable-autotune \
  --disable-sampling-tp-sync \
  --sampling-backend greedy \
  --decode-log-interval 40 \
  --health-check-timeout-secs 1800 \
  --engine-startup-timeout 3000 \
  --host 127.0.0.1 \
  --port 8000 \
  --cudagraph-capture-sizes 1 2 4 8 16
```

DeepSeek-V4.1-Flash DSPARK, node 0, for both phases:

```bash
TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN=3.9 python3 -m tokenspeed.cli serve /data/models/DeepSeek-V4.1-Flash \
  --language-model-only \
  --dtype bfloat16 \
  --speculative-algorithm DSPARK \
  --served-model-name deepseek-v41-flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --max-num-seqs 16 \
  --chunked-prefill-size 8192 \
  --prefill-graph-max-tokens 8192 \
  --disable-kvstore \
  --disable-prefix-caching \
  --disable-autotune \
  --disable-sampling-tp-sync \
  --sampling-backend greedy \
  --decode-log-interval 40 \
  --health-check-timeout-secs 1800 \
  --engine-startup-timeout 3000 \
  --host 127.0.0.1 \
  --port 8000 \
  --cudagraph-capture-sizes 1 2 4 8 16
```

Kimi-K3 EAGLE3, node 0, for both phases:

```bash
TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN=3.75 python3 -m tokenspeed.cli serve /data/models/Kimi-K3 \
  --language-model-only \
  --attention-backend mla \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /data/models/kimi-k3-eagle3-mla \
  --speculative-draft-model-quantization unquant \
  --eagle3-layers-to-capture 2\,46\,90 \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1 \
  --served-model-name kimi-k3 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --nnodes 2 \
  --nprocs-per-node 4 \
  --world-size 8 \
  --dist-init-addr <node 0 ip>:25000 \
  --node-rank 0 \
  --max-model-len 65536 \
  --max-num-seqs 16 \
  --chunked-prefill-size 8192 \
  --disable-prefill-graph \
  --disable-kvstore \
  --disable-prefix-caching \
  --disable-autotune \
  --disable-sampling-tp-sync \
  --sampling-backend greedy \
  --decode-log-interval 40 \
  --health-check-timeout-secs 1800 \
  --engine-startup-timeout 3000 \
  --host 0.0.0.0 \
  --port 8000 \
  --cudagraph-capture-sizes 1 2 4 8 16
```

Node 1 runs the same command with `--node-rank 1`.

</details>

## Acceptance

Logged values are medians over the server's 40-step decode windows;
windows that straddle a prefill keep fewer tokens.

| Model | Layout | Fixed | Logged |
|---|---|---:|---:|
| GLM-5.3-Flash MTP | TP4 (1 node, 4 GPUs) | 2.9 | 2.9 |
| DeepSeek-V4.1-Flash DSPARK | TP4 (1 node, 4 GPUs) | 3.9 | 3.9 |
| Kimi-K3 EAGLE3 | TP8 (2 nodes x 4 GPUs) | 3.75 | 3.75 |

TPOT p50 is per request over its whole decode, as CI reports it. At
batch 16 every request arrives at once, so a request whose prompt
finishes early decodes between other requests' prefill chunks, and its
TPOT mostly measures prefill. Steady decode TPOT is the median over the
server's 40-step decode windows in which every request was decoding and
no prefill ran. A steady verify step is that window's wall time per
forward: the logged accept length times the steady TPOT.

## GLM-5.3-Flash MTP, TP4 (1 node, 4 GPUs)

| Metric | Batch 1 | Batch 16 |
|---|---:|---:|
| TTFT p50 (ms) | 1,074.3 | 8,777.6 |
| TPOT p50 (ms) | 3.3 | 20.5 |
| steady decode TPOT (ms) | 3.3 | 5.0 |
| decode tok/s per user | 302.3 | 201.9 |
| aggregate output tok/s | 182.7 | 420.5 |
| steady verify step (ms) | 9.59 | 14.36 |

Rank 0's accumulated GPU kernel time per bucket from the CUDA-graph
profiling run, with each bucket's share of the stage. Prefill
columns cover one wave's whole prefill (every chunk of every
request); decode columns are per verify step, averaged over 64.
The last row is the perf run's slowest TTFT in the wave, when its
prefill ends, and its steady verify step; it is not a total of
the rows. Communication kernels include the time they wait for
the other ranks.

| Category | prefill c1 | prefill c16 | decode c1 | decode c16 |
|---|---:|---:|---:|---:|
| **communication** | 128.06 (13%) | 1,705.48 (11%) | 3.83 (37%) | 4.27 (27%) |
| KDA other | 39.77 (4%) | 623.57 (4%) | 0.23 (2%) | 0.51 (3%) |
| MoE | 291.89 (30%) | 4,581.49 (30%) | 1.24 (12%) | 3.62 (23%) |
| index top-k | 19.78 (2%) | 318.20 (2%) | 0.26 (2%) | 0.27 (2%) |
| index scoring | 40.53 (4%) | 646.45 (4%) | 0.10 (1%) | 0.41 (3%) |
| sparse attention | 75.66 (8%) | 1,209.61 (8%) | 0.21 (2%) | 0.52 (3%) |
| dense GEMM | 221.72 (23%) | 3,447.84 (23%) | 2.11 (20%) | 3.72 (23%) |
| attention | 0.34 (0%) | 4.88 (0%) | 0.02 (0%) | 0.03 (0%) |
| hyper-connections | 85.74 (9%) | 1,345.61 (9%) | 1.12 (11%) | 1.11 (7%) |
| quantize | 8.12 (1%) | 121.07 (1%) | 0.26 (3%) | 0.41 (3%) |
| rmsnorm | 11.78 (1%) | 175.75 (1%) | 0.37 (4%) | 0.43 (3%) |
| sampling | 0.14 (0%) | 1.56 (0%) | 0.03 (0%) | 0.03 (0%) |
| elementwise | 51.12 (5%) | 775.63 (5%) | 0.55 (5%) | 0.64 (4%) |
| other | 8.07 (1%) | 123.14 (1%) | 0.06 (1%) | 0.09 (1%) |
| all kernels | 982.71 | 15,080.28 | 10.40 | 16.07 |
| **Measured, CUDA graphs** | **1,074.30** (TTFT max) | **16,509.28** (TTFT max) | **9.59** (verify step) | **14.36** (verify step) |

<details><summary>prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_fp8_block_down_gfx1250` | 135.601 | 348 | 389.7 | 13.80% |
| 2 | communication | `ncclDevKernel_Generic_32(ncclDevKernelArgsStorage<5120ul>)` | 121.669 | 664 | 183.2 | 12.38% |
| 3 | MoE | `_fp8_block_gate_up_gfx1250` | 120.879 | 348 | 347.4 | 12.30% |
| 4 | dense GEMM | `_w8a8_block_fp8_matmul` | 105.179 | 875 | 120.2 | 10.70% |
| 5 | sparse attention | `_dsa_selected_dense_wmma_kernel` | 75.217 | 100 | 752.2 | 7.65% |
| 6 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT208x256x128_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_AS...` | 53.932 | 204 | 264.4 | 5.49% |
| 7 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 35.611 | 720 | 49.5 | 3.62% |
| 8 | index scoring | `gluon_kpool_prefill_topk_fp8_plan_gfx1250` | 35.168 | 120 | 293.1 | 3.58% |
| 9 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 33.352 | 720 | 46.3 | 3.39% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x128_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_AS...` | 21.021 | 282 | 74.5 | 2.14% |

</details>

<details><summary>prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | MoE | `_fp8_block_down_gfx1250` | 2,147.215 | 4,291 | 500.4 | 14.24% |
| 2 | MoE | `_fp8_block_gate_up_gfx1250` | 1,876.554 | 4,291 | 437.3 | 12.44% |
| 3 | dense GEMM | `_w8a8_block_fp8_matmul` | 1,680.277 | 12,250 | 137.2 | 11.14% |
| 4 | communication | `ncclDevKernel_Generic_32(ncclDevKernelArgsStorage<5120ul>)` | 1,665.267 | 9,248 | 180.1 | 11.04% |
| 5 | sparse attention | `_dsa_selected_dense_wmma_kernel` | 1,204.547 | 1,222 | 985.7 | 7.99% |
| 6 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT208x256x128_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_AS...` | 876.266 | 3,298 | 265.7 | 5.81% |
| 7 | index scoring | `gluon_kpool_prefill_topk_fp8_plan_gfx1250` | 562.123 | 1,764 | 318.7 | 3.73% |
| 8 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 561.257 | 8,910 | 63.0 | 3.72% |
| 9 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 529.645 | 8,910 | 59.4 | 3.51% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x128_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_AS...` | 340.266 | 4,559 | 74.6 | 2.26% |

</details>

<details><summary>decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | communication | `iris_pull_oneshot_staged` | 217.961 | 5,922 | 36.8 | 32.75% |
| 2 | dense GEMM | `_wmma_tdm_dense_kernel` | 35.585 | 8,505 | 4.2 | 5.35% |
| 3 | MoE | `_fp8_block_gate_up_gfx1250` | 33.881 | 2,835 | 12.0 | 5.09% |
| 4 | dense GEMM | `gluon_mm_fp8_blockscale_gfx1250_bn16_bk2048_buf3_sk1_tdmf0` | 32.996 | 4,788 | 6.9 | 4.96% |
| 5 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 27.769 | 5,670 | 4.9 | 4.17% |
| 6 | hyper-connections | `gluon_mhc_mixes_reduce_gfx1250` | 22.482 | 5,670 | 4.0 | 3.38% |
| 7 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1...` | 22.421 | 3,150 | 7.1 | 3.37% |
| 8 | MoE | `_fp8_block_down_gfx1250` | 20.181 | 2,835 | 7.1 | 3.03% |
| 9 | quantize | `_fp8_token_group_quantize_kernel` | 16.941 | 8,694 | 1.9 | 2.55% |
| 10 | rmsnorm | `_rmsnorm_kernel` | 14.765 | 6,300 | 2.3 | 2.22% |

</details>

<details><summary>decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | communication | `iris_pull_oneshot_staged` | 240.202 | 5,922 | 40.6 | 23.35% |
| 2 | MoE | `_fp8_block_gate_up_gfx1250` | 125.504 | 2,835 | 44.3 | 12.20% |
| 3 | dense GEMM | `gluon_mm_fp8_blockscale_gfx1250_bn32_bk512_buf3_sk1_tdmf0` | 80.357 | 3,402 | 23.6 | 7.81% |
| 4 | MoE | `_fp8_block_down_gfx1250` | 75.790 | 2,835 | 26.7 | 7.37% |
| 5 | dense GEMM | `_wmma_tdm_dense_kernel` | 63.255 | 9,072 | 7.0 | 6.15% |
| 6 | dense GEMM | `gluon_mm_fp8_blockscale_gfx1250_bn16_bk1024_buf3_sk1_tdmf0` | 28.415 | 3,402 | 8.4 | 2.76% |
| 7 | quantize | `_fp8_token_group_quantize_kernel` | 26.302 | 8,694 | 3.0 | 2.56% |
| 8 | sparse attention | `_dsa_selected_dense_wmma_kernel` | 25.175 | 882 | 28.5 | 2.45% |
| 9 | hyper-connections | `gluon_mhc_mixes_reduce_gfx1250` | 24.564 | 5,670 | 4.3 | 2.39% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1...` | 24.408 | 2,898 | 8.4 | 2.37% |

</details>

## DeepSeek-V4.1-Flash DSPARK, TP4 (1 node, 4 GPUs)

| Metric | Batch 1 | Batch 16 |
|---|---:|---:|
| TTFT p50 (ms) | 399.9 | 3,248.9 |
| TPOT p50 (ms) | 2.2 | 10.5 |
| steady decode TPOT (ms) | 2.2 | 4.7 |
| decode tok/s per user | 451.2 | 212.4 |
| aggregate output tok/s | 329.8 | 940.8 |
| steady verify step (ms) | 8.64 | 18.36 |

Rank 0's accumulated GPU kernel time per bucket from the CUDA-graph
profiling run, with each bucket's share of the stage. Prefill
columns cover one wave's whole prefill (every chunk of every
request); decode columns are per verify step, averaged over 64.
The last row is the perf run's slowest TTFT in the wave, when its
prefill ends, and its steady verify step; it is not a total of
the rows. Communication kernels include the time they wait for
the other ranks.

| Category | prefill c1 | prefill c16 | decode c1 | decode c16 |
|---|---:|---:|---:|---:|
| **communication** | 63.52 (16%) | 1,164.84 (18%) | 4.17 (39%) | 10.31 (49%) |
| MoE | 71.89 (18%) | 1,106.30 (17%) | 1.09 (10%) | 2.69 (13%) |
| index top-k | 20.54 (5%) | 323.48 (5%) | 0.53 (5%) | 0.57 (3%) |
| index scoring | 56.53 (14%) | 906.24 (14%) | 0.13 (1%) | 0.68 (3%) |
| sparse attention | 32.92 (8%) | 525.57 (8%) | 0.53 (5%) | 1.06 (5%) |
| dense GEMM | 48.90 (12%) | 737.12 (12%) | 1.92 (18%) | 2.92 (14%) |
| hyper-connections | 50.34 (13%) | 782.10 (12%) | 1.37 (13%) | 1.55 (7%) |
| quantize | 3.94 (1%) | 60.56 (1%) | 0.08 (1%) | 0.10 (0%) |
| rmsnorm | 1.46 (0%) | 21.00 (0%) | 0.11 (1%) | 0.11 (1%) |
| top-k and sort | 0.26 (0%) | 4.38 (0%) | — | — |
| sampling | 0.01 (0%) | 0.11 (0%) | 0.01 (0%) | 0.01 (0%) |
| elementwise | 41.42 (10%) | 646.62 (10%) | 0.54 (5%) | 0.67 (3%) |
| other | 6.97 (2%) | 111.12 (2%) | 0.18 (2%) | 0.21 (1%) |
| all kernels | 398.72 | 6,389.45 | 10.66 | 20.88 |
| **Measured, CUDA graphs** | **399.91** (TTFT max) | **6,098.84** (TTFT max) | **8.64** (verify step) | **18.36** (verify step) |

<details><summary>prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | communication | `ncclDevKernel_Generic_32(ncclDevKernelArgsStorage<5120ul>)` | 59.987 | 348 | 172.4 | 15.04% |
| 2 | index scoring | `gluon_dsv41_index_topk_gfx1250` | 55.715 | 601 | 92.7 | 13.97% |
| 3 | MoE | `_matmul` | 53.298 | 240 | 222.1 | 13.37% |
| 4 | sparse attention | `gluon_dsv4_prefill_gfx1250` | 32.391 | 160 | 202.4 | 8.12% |
| 5 | dense GEMM | `gluon_mm_mxfp8_ue8m0_largem_bk256_gfx1250_c2` | 26.356 | 512 | 51.5 | 6.61% |
| 6 | index top-k | `gluon_dsv41_index_topk_select_gfx1250` | 20.408 | 601 | 34.0 | 5.12% |
| 7 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 19.845 | 406 | 48.9 | 4.98% |
| 8 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 18.345 | 406 | 45.2 | 4.60% |
| 9 | elementwise | `_rope_inplace_kernel` | 10.956 | 421 | 26.0 | 2.75% |
| 10 | elementwise | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::n...` | 10.470 | 601 | 17.4 | 2.63% |

</details>

<details><summary>prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | communication | `ncclDevKernel_Generic_32(ncclDevKernelArgsStorage<5120ul>)` | 1,161.223 | 4,876 | 238.2 | 18.17% |
| 2 | index scoring | `gluon_dsv41_index_topk_gfx1250` | 893.192 | 9,574 | 93.3 | 13.98% |
| 3 | MoE | `_matmul` | 881.649 | 3,920 | 224.9 | 13.80% |
| 4 | sparse attention | `gluon_dsv4_prefill_gfx1250` | 525.025 | 2,880 | 182.3 | 8.22% |
| 5 | dense GEMM | `gluon_mm_mxfp8_ue8m0_largem_bk256_gfx1250_c2` | 427.833 | 8,346 | 51.3 | 6.70% |
| 6 | index top-k | `gluon_dsv41_index_topk_select_gfx1250` | 322.189 | 9,574 | 33.7 | 5.04% |
| 7 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 313.414 | 4,646 | 67.5 | 4.91% |
| 8 | hyper-connections | `_mhc_post_hc4_triton_kernel` | 290.957 | 4,646 | 62.6 | 4.55% |
| 9 | elementwise | `_rope_inplace_kernel` | 174.031 | 5,737 | 30.3 | 2.72% |
| 10 | elementwise | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::n...` | 158.030 | 9,589 | 16.5 | 2.47% |

</details>

<details><summary>decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | communication | `iris_pull_oneshot_staged` | 254.803 | 5,670 | 44.9 | 37.36% |
| 2 | dense GEMM | `gluon_mm_mxfp8_ue8m0_gfx1250_bn16_bk512_buf3_sk1` | 54.612 | 11,592 | 4.7 | 8.01% |
| 3 | MoE | `_matmul_decode` | 44.856 | 5,418 | 8.3 | 6.58% |
| 4 | hyper-connections | `gluon_mhc_mixes_reduce_gfx1250` | 33.535 | 5,418 | 6.2 | 4.92% |
| 5 | index top-k | `gluon_dsv41_index_topk_select_gfx1250` | 29.651 | 504 | 58.8 | 4.35% |
| 6 | dense GEMM | `_wmma_tdm_dense_kernel` | 28.726 | 6,489 | 4.4 | 4.21% |
| 7 | sparse attention | `gluon_dsv41_selected_attention_gfx1250` | 23.031 | 2,520 | 9.1 | 3.38% |
| 8 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 20.841 | 5,418 | 3.8 | 3.06% |
| 9 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASE...` | 19.370 | 63 | 307.5 | 2.84% |
| 10 | communication | `ncclDevKernel_Generic_32(ncclDevKernelArgsStorage<5120ul>)` | 12.152 | 378 | 32.1 | 1.78% |

</details>

<details><summary>decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | communication | `ncclDevKernel_Generic_32(ncclDevKernelArgsStorage<5120ul>)` | 659.589 | 6,048 | 109.1 | 49.36% |
| 2 | MoE | `_matmul_decode` | 141.293 | 5,418 | 26.1 | 10.57% |
| 3 | dense GEMM | `gluon_mm_mxfp8_ue8m0_largem_bk512_gfx1250_t64_c1_sk1` | 94.054 | 14,175 | 6.6 | 7.04% |
| 4 | dense GEMM | `_wmma_tdm_dense_kernel` | 45.805 | 5,607 | 8.2 | 3.43% |
| 5 | index scoring | `gluon_dsv41_index_topk_gfx1250` | 42.176 | 504 | 83.7 | 3.16% |
| 6 | hyper-connections | `gluon_mhc_mixes_reduce_gfx1250` | 42.146 | 5,418 | 7.8 | 3.15% |
| 7 | sparse attention | `gluon_dsv41_selected_attention_gfx1250` | 41.214 | 2,520 | 16.4 | 3.08% |
| 8 | index top-k | `gluon_dsv41_index_topk_select_gfx1250` | 32.227 | 504 | 63.9 | 2.41% |
| 9 | hyper-connections | `gluon_mhc_mixes_project_gfx1250` | 23.079 | 5,418 | 4.3 | 1.73% |
| 10 | dense GEMM | `Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASE...` | 19.549 | 63 | 310.3 | 1.46% |

</details>

## Kimi-K3 EAGLE3, TP8 (2 nodes x 4 GPUs)

| Metric | Batch 1 | Batch 16 |
|---|---:|---:|
| TTFT p50 (ms) | 2,041.4 | 15,532.6 |
| TPOT p50 (ms) | 11.5 | 41.4 |
| steady decode TPOT (ms) | 11.4 | 14.0 |
| decode tok/s per user | 87.6 | 71.6 |
| aggregate output tok/s | 64.3 | 220.9 |
| steady verify step (ms) | 42.79 | 52.39 |

Rank 0's accumulated GPU kernel time per bucket from the CUDA-graph
profiling run, with each bucket's share of the stage. Prefill
columns cover one wave's whole prefill (every chunk of every
request); decode columns are per verify step, averaged over 64.
The last row is the perf run's slowest TTFT in the wave, when its
prefill ends, and its steady verify step; it is not a total of
the rows. Communication kernels include the time they wait for
the other ranks.

| Category | prefill c1 | prefill c16 | decode c1 | decode c16 |
|---|---:|---:|---:|---:|
| **communication** | 633.52 (33%) | 9,536.64 (32%) | 34.35 (80%) | 35.86 (70%) |
| KDA other | 72.54 (4%) | 1,124.94 (4%) | 0.54 (1%) | 0.94 (2%) |
| MoE | 393.56 (20%) | 6,038.07 (20%) | 2.25 (5%) | 5.92 (11%) |
| input projections | 195.75 (10%) | 3,032.04 (10%) | 0.84 (2%) | 1.26 (2%) |
| dense GEMM | 367.28 (19%) | 5,739.10 (19%) | 2.47 (6%) | 3.78 (7%) |
| attention | 114.24 (6%) | 1,907.84 (6%) | 0.76 (2%) | 1.84 (4%) |
| AttnRes | 116.05 (6%) | 1,827.71 (6%) | 0.70 (2%) | 0.84 (2%) |
| add3 | 16.62 (1%) | 264.07 (1%) | — | — |
| quantize | 5.78 (0%) | 84.37 (0%) | 0.16 (0%) | 0.18 (0%) |
| rmsnorm | 12.61 (1%) | 188.46 (1%) | 0.46 (1%) | 0.49 (1%) |
| sampling | 0.13 (0%) | 1.55 (0%) | 0.03 (0%) | 0.03 (0%) |
| elementwise | 6.89 (0%) | 99.57 (0%) | 0.13 (0%) | 0.35 (1%) |
| other | 8.20 (0%) | 123.90 (0%) | 0.07 (0%) | 0.10 (0%) |
| all kernels | 1,943.17 | 29,968.26 | 42.77 | 51.60 |
| **Measured, CUDA graphs** | **2,041.37** (TTFT max) | **29,181.27** (TTFT max) | **42.79** (verify step) | **52.39** (verify step) |

<details><summary>prefill c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | communication | `ncclDevKernel_Generic_32(ncclDevKernelArgsStorage<5120ul>)` | 633.519 | 1,548 | 409.2 | 32.60% |
| 2 | dense GEMM | `_wmma_tdm_dense_largem_warp_pipeline_kernel` | 342.157 | 2,856 | 119.8 | 17.61% |
| 3 | MoE | `_matmul` | 289.832 | 1,104 | 262.5 | 14.92% |
| 4 | input projections | `gluon_latent_input_largem_gfx1250` | 185.702 | 552 | 336.4 | 9.56% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 116.050 | 1,496 | 77.6 | 5.97% |
| 6 | attention | `gluon_mla_prefill_gfx1250` | 101.893 | 360 | 283.0 | 5.24% |
| 7 | MoE | `_weighted_topk_reduce_gfx1250_kernel` | 35.678 | 736 | 48.5 | 1.84% |
| 8 | KDA other | `gluon_kda_paged_prefill_prepare_gfx1250` | 29.108 | 483 | 60.3 | 1.50% |
| 9 | MoE | `_matmul_decode` | 20.113 | 368 | 54.7 | 1.04% |
| 10 | add3 | `_add3_kernel` | 16.622 | 644 | 25.8 | 0.86% |

</details>

<details><summary>prefill c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | communication | `ncclDevKernel_Generic_32(ncclDevKernelArgsStorage<5120ul>)` | 9,536.637 | 19,110 | 499.0 | 31.82% |
| 2 | dense GEMM | `_wmma_tdm_dense_largem_warp_pipeline_kernel` | 5,587.156 | 46,704 | 119.6 | 18.64% |
| 3 | MoE | `_matmul` | 4,737.575 | 18,032 | 262.7 | 15.81% |
| 4 | input projections | `gluon_latent_input_largem_gfx1250` | 3,031.186 | 9,016 | 336.2 | 10.11% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 1,827.713 | 18,513 | 98.7 | 6.10% |
| 6 | attention | `gluon_mla_prefill_gfx1250` | 1,686.181 | 5,856 | 287.9 | 5.63% |
| 7 | MoE | `_weighted_topk_reduce_gfx1250_kernel` | 570.965 | 9,108 | 62.7 | 1.91% |
| 8 | KDA other | `gluon_kda_paged_prefill_prepare_gfx1250` | 451.192 | 6,762 | 66.7 | 1.51% |
| 9 | add3 | `_add3_kernel` | 264.070 | 9,016 | 29.3 | 0.88% |
| 10 | MoE | `gluon_sigmoid_bias_topk_gfx1250` | 239.318 | 9,016 | 26.5 | 0.80% |

</details>

<details><summary>decode c1: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | communication | `ncclDevKernel_Generic_32(ncclDevKernelArgsStorage<5120ul>)` | 2,198.504 | 12,474 | 176.2 | 80.32% |
| 2 | dense GEMM | `_wmma_tdm_dense_kernel` | 139.415 | 25,200 | 5.5 | 5.09% |
| 3 | MoE | `_matmul_decode` | 95.019 | 11,592 | 8.2 | 3.47% |
| 4 | input projections | `gluon_latent_input_decode_gfx1250` | 53.908 | 5,796 | 9.3 | 1.97% |
| 5 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 44.848 | 11,781 | 3.8 | 1.64% |
| 6 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx1250` | 33.142 | 4,347 | 7.6 | 1.21% |
| 7 | attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_NONE_TILE_SIZE_64_KV_LORA_RANK_5...` | 26.059 | 1,512 | 17.2 | 0.95% |
| 8 | MoE | `_kimi3_sigmoid_bias_topk_kernel` | 19.802 | 5,796 | 3.4 | 0.72% |
| 9 | MoE | `_precomputed_topk_route_small_m_gfx1250_kernel` | 16.138 | 5,796 | 2.8 | 0.59% |
| 10 | rmsnorm | `_rmsnorm_kernel` | 15.007 | 6,174 | 2.4 | 0.55% |

</details>

<details><summary>decode c16: heaviest 10 kernels</summary>

| Rank | Category | Kernel | GPU time (ms) | Calls | Mean/call (us) | Share |
|---:|---|---|---:|---:|---:|---:|
| 1 | communication | `ncclDevKernel_Generic_32(ncclDevKernelArgsStorage<5120ul>)` | 2,295.247 | 12,474 | 184.0 | 69.51% |
| 2 | MoE | `_matmul_decode` | 313.355 | 11,592 | 27.0 | 9.49% |
| 3 | dense GEMM | `_wmma_tdm_dense_kernel` | 218.420 | 26,082 | 8.4 | 6.61% |
| 4 | attention | `_mla_decode_fwd_kernel_num_query_heads_12_num_queries_per_kv_NONE_num_tokens_per_seq_NONE_TILE_SIZE_64_KV_LORA_RANK_5...` | 94.449 | 1,512 | 62.5 | 2.86% |
| 5 | input projections | `gluon_latent_input_decode_nosplit_gfx1250` | 80.657 | 5,796 | 13.9 | 2.44% |
| 6 | AttnRes | `gluon_attn_res_fwd_gfx1250` | 53.705 | 11,781 | 4.6 | 1.63% |
| 7 | KDA other | `gluon_kda_fused_paged_verify_nostore_vmajor_gfx1250` | 45.574 | 4,347 | 10.5 | 1.38% |
| 8 | MoE | `_precomputed_topk_route_small_m_gfx1250_kernel` | 23.200 | 5,796 | 4.0 | 0.70% |
| 9 | MoE | `_kimi3_sigmoid_bias_topk_kernel` | 21.614 | 5,796 | 3.7 | 0.65% |
| 10 | MoE | `_weighted_topk_reduce_gfx1250_kernel` | 16.962 | 5,796 | 2.9 | 0.51% |

</details>

## Method and limitations

- Timing and profiling are separate server runs with the same flags.
  rocprofv3 traces the kernels inside the decode CUDA graphs too, so
  the profile sees the collectives as the perf run issues them.
- Kernel tables are rank 0's; the appendix covers every rank on
  node 0. A rank's communication kernels include the time it
  waits at each collective for the slowest rank.
- The isolated all-reduce sends hidden-size bf16 rows; the server's
  collectives can be wider or fused with a norm, so it shows what that
  many hidden-size rows cost on these GPUs, not what each call sends.
- Prompts are random token ids, so MoE routing is real but not that of
  natural text.
- Accept lengths are fixed, so these runs measure the cost of a verify
  step at a given acceptance, not drafter quality.

## Appendix: collectives

Communication's share of rank 0's kernel time per stage:

| Model | Layout | prefill c1 | prefill c16 | decode c1 | decode c16 |
|---|---|---:|---:|---:|---:|
| GLM-5.3-Flash MTP | TP4 (1 node, 4 GPUs) | 13% | 11% | 37% | 27% |
| DeepSeek-V4.1-Flash DSPARK | TP4 (1 node, 4 GPUs) | 16% | 18% | 39% | 49% |
| Kimi-K3 EAGLE3 | TP8 (2 nodes x 4 GPUs) | 33% | 32% | 80% | 70% |

Batch 1 decode collectives on rank 0 against the same GPUs'
all-reduce of four hidden-size rows (one verify step's tokens)
replayed from a CUDA graph outside the server:

| Model | Layout | Calls per step | In server (us per call) | Isolated, 4 rows (us) | Collectives per step (ms) |
|---|---|---:|---:|---:|---:|
| GLM-5.3-Flash MTP | TP4 (1 node, 4 GPUs) | 114 | 33.5 | 7.3 | 3.83 |
| DeepSeek-V4.1-Flash DSPARK | TP4 (1 node, 4 GPUs) | 94 | 44.1 | 7.3 | 4.17 |
| Kimi-K3 EAGLE3 | TP8 (2 nodes x 4 GPUs) | 195 | 176.2 | 11.9 | 34.35 |

### GLM-5.3-Flash MTP, TP4 (1 node, 4 GPUs)

Every profiled rank (all 4 ranks), from `rank_balance.txt`. A
collective ends on every rank together, so a rank that reached
it late would spend less time in it than the others.

| Rank | c1 decode: per call (us) | c1 decode: comm / compute per step (ms) | c16 decode: per call (us) | c16 decode: comm / compute per step (ms) |
|---|---:|---:|---:|---:|
| TP0 | 33.5 | 3.83 / 6.57 | 37.4 | 4.27 / 11.80 |
| TP1 | 33.1 | 3.78 / 6.65 | 35.8 | 4.09 / 11.99 |
| TP2 | 31.2 | 3.57 / 6.74 | 35.7 | 4.08 / 11.91 |
| TP3 | 32.3 | 3.69 / 6.62 | 34.8 | 3.97 / 11.94 |

A bf16 all-reduce of hidden-size rows over the same GPUs and
network setup outside the server (`allreduce_probe.txt`). Rank 0's
decode collectives inside the server average 33.5 us at c1 and 37.4 us at c16 per call. Some
server collectives carry fused norms or wider rows than the
hidden size.

| Tokens | Bytes | Back-to-back (us) | CUDA-graph replay (us) |
|---:|---:|---:|---:|
| 4 | 32,768 | 14.7 | 7.3 |
| 64 | 524,288 | 15.9 | 9.1 |
| 256 | 2,097,152 | 26.8 | 20.1 |
| 8,192 | 67,108,864 | 135.8 | — |

### DeepSeek-V4.1-Flash DSPARK, TP4 (1 node, 4 GPUs)

Every profiled rank (all 4 ranks), from `rank_balance.txt`. A
collective ends on every rank together, so a rank that reached
it late would spend less time in it than the others.

| Rank | c1 decode: per call (us) | c1 decode: comm / compute per step (ms) | c16 decode: per call (us) | c16 decode: comm / compute per step (ms) |
|---|---:|---:|---:|---:|
| TP0 | 44.1 | 4.17 / 6.48 | 109.1 | 10.31 / 10.57 |
| TP1 | 41.2 | 3.89 / 6.55 | 102.9 | 9.72 / 10.54 |
| TP2 | 40.9 | 3.87 / 6.47 | 105.2 | 9.94 / 10.48 |
| TP3 | 35.9 | 3.39 / 6.53 | 103.8 | 9.81 / 10.45 |

A bf16 all-reduce of hidden-size rows over the same GPUs and
network setup outside the server (`allreduce_probe.txt`). Rank 0's
decode collectives inside the server average 44.1 us at c1 and 109.1 us at c16 per call. Some
server collectives carry fused norms or wider rows than the
hidden size.

| Tokens | Bytes | Back-to-back (us) | CUDA-graph replay (us) |
|---:|---:|---:|---:|
| 4 | 40,960 | 14.4 | 7.3 |
| 64 | 655,360 | 16.0 | 9.8 |
| 256 | 2,621,440 | 24.9 | 18.5 |
| 8,192 | 83,886,080 | 159.3 | — |

### Kimi-K3 EAGLE3, TP8 (2 nodes x 4 GPUs)

Every profiled rank (the 4 ranks on node 0, of 8), from `rank_balance.txt`. A
collective ends on every rank together, so a rank that reached
it late would spend less time in it than the others.

| Rank | c1 decode: per call (us) | c1 decode: comm / compute per step (ms) | c16 decode: per call (us) | c16 decode: comm / compute per step (ms) |
|---|---:|---:|---:|---:|
| TP0 | 176.2 | 34.35 / 8.42 | 184.0 | 35.86 / 15.73 |
| TP1 | 176.7 | 34.44 / 8.34 | 183.7 | 35.81 / 15.81 |
| TP2 | 176.6 | 34.42 / 8.35 | 184.2 | 35.91 / 15.68 |
| TP3 | 176.0 | 34.31 / 8.44 | 184.0 | 35.86 / 15.68 |

A bf16 all-reduce of hidden-size rows over the same GPUs and
network setup outside the server (`allreduce_probe.txt`). Rank 0's
decode collectives inside the server average 176.2 us at c1 and 184.0 us at c16 per call. Some
server collectives carry fused norms or wider rows than the
hidden size.

| Tokens | Bytes | Back-to-back (us) | CUDA-graph replay (us) |
|---:|---:|---:|---:|
| 4 | 57,344 | 18.7 | 11.9 |
| 64 | 917,504 | 25.5 | 19.3 |
| 256 | 3,670,016 | 46.4 | 40.2 |
| 8,192 | 117,440,512 | 242.6 | — |

Regenerate this document with:

```bash
python3 real_serving/scripts/generate_hotspot_report.py \
  --run <MI455X result dir> \
  --commit-date 2026-10-29 \
  --output-dir real_serving/results/hotspots_20261029_f198a543
```
