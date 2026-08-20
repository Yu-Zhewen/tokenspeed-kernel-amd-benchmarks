# GFX1250 matched-attention results at `70f69266`

## Status

- Collection date: 2026-08-20 UTC
- GFX1250 physical collection: complete
- Required workloads: 6/6 passed
- Profiled measured-scope dispatches: 19
- Profiler passes per case: 6 (memory, cache, SQ, WMMA, TX, conflicts)
- TokenSpeed revision: `70f692669a76b6c317cec31b679d7f4fac5da9fa`
- GFX950 data at this revision: unavailable

## Software and hardware setup

| Field | Value |
|---|---|
| Device | MI455X-class device (PyTorch: `AMD Radeon Graphics`) |
| Architecture | `gfx1250` |
| Environment | physical, GPU 0 |
| Host | `heliosr-1b114-d04-1` |
| Container image | `tokenspeed-kimi-smoke@sha256:77179418c81a72f003c08656b62b9d7dec3bbdb794a8d5af2f19cf37b2289405` |
| OS | Linux 7.1.0 host; Ubuntu 24.04 container |
| PyTorch | `2.11.0+rocm7.15.0a20260728` |
| ROCm | `7.15.0` |
| rocprofv3 | `1.3.5` (`44be71b52284948e58c93f65f46910399773fdcd`) |
| tokenspeed-triton | `3.8.10.post20260721` |
| TokenSpeed commit | `70f692669a76b6c317cec31b679d7f4fac5da9fa` |
| TokenSpeed tree | `dbe507e0fd5d65366b10d63cef7028af5c7564e4` |
| tokenspeed-kernel tree | `8c4c7564d4f1727e67e01855c801e5d44fc5baad` |
| tokenspeed-kernel-amd tree | `c139bcd7853eae1d0c843cc544ab11d144326cfc` |
| Latency warmups / repeats | 2 / 5 |
| Profiler warmups / repeats | 0 / 1 |

The source checkout was clean for tracked files and mounted read-only. All GPU commands ran under the host `gpu-lock`. No process had `/dev/kfd` open before collection started.

## Physical workload latency

Complete unprofiled GPU-event averages:

| ID | Runner case | GFX1250 latency (µs) | Prior `9d1b02ed` (µs) | Change |
|---|---|---:|---:|---:|
| MLA-D1 | `mla-decode` | 106.520 | 119.931 | -11.18% |
| MLA-P1 | `mla-prefill` | 138.543 | 140.570 | -1.44% |
| KDA-D1 | `kda-decode` | 51.854 | 51.701 | +0.30% |
| KDA-P1 | `kda-prefill` | 286.301 | 323.683 | -11.55% |
| DSA-D1 | `dsa-decode-pipeline` | 162.123 | 131.484 | +23.30% |
| DSA-P1 | `dsa-prefill-pipeline-4k` | 1,023.501 | 2,470.439 | -58.57% |

## Per-dispatch physical results

Rows are the final repeated suffix corresponding to the measured invocation. Duration and `FetchSize` come from the memory pass; other values come from separate identical profiler reruns.

| Case | # | Dispatch | Duration µs | Grid items | WG | VGPR/SGPR | LDS/scratch B | Read MB | Read GB/s | L2 hit | Waves | WMMA insts | WMMA FLOPs | LDS bank conflict | LDS segment stall | VC segment stall |
|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| MLA-D1 | 1 | MLA decode main | 6.850 | 1024 | 64 | 280/128 | 0/0 | 0.382 | 55.7 | 99.94% | 32 | 6,656 | FP8:53,248 | 64 | 27,660 | 3,375 |
| MLA-D1 | 2 | reduce/project/gate | 5.128 | 49152 | 256 | 136/128 | 0/0 | 0.386 | 75.3 | 98.93% | 1,536 | 0 | 0 | 0 | 0 | 1,003,988 |
| MLA-P1 | 1 | MLA prefill | 127.671 | 49152 | 128 | 264/128 | 0/0 | 0.004 | 0.0 | 97.93% | 1,536 | 1,622,016 | FP8:12,976,128 | 98,304 | 3,721,181 | 337,717 |
| KDA-D1 | 1 | KDA recurrent decode | 3.405 | 6144 | 128 | 56/128 | 0/0 | 0.382 | 112.1 | 92.66% | 192 | 0 | 0 | 0 | 540 | 6,566 |
| KDA-P1 | 1 | BF16 workspace fill | 3.405 | 393216 | 256 | 8/128 | 0/0 | 0.008 | 2.4 | 94.02% | 12,288 | 0 | 0 | 0 | 0 | 0 |
| KDA-P1 | 2 | preprocess intra | 76.354 | 196608 | 256 | 128/128 | 0/0 | 82.482 | 1080.3 | 84.61% | 6,144 | 491,520 | BF16:3,932,160 | 3,538,944 | 5,603,278 | 214,572 |
| KDA-P1 | 3 | BF16 workspace fill | 3.004 | 393216 | 256 | 8/128 | 0/0 | 0.004 | 1.3 | 89.59% | 12,288 | 0 | 0 | 0 | 0 | 0 |
| KDA-P1 | 4 | solve/merge 64 | 12.579 | 24576 | 32 | 104/128 | 0/0 | 6.054 | 481.2 | 92.08% | 768 | 12,288 | BF16:98,304 | 73,728 | 0 | 30,644 |
| KDA-P1 | 5 | W/U vector | 24.637 | 98304 | 128 | 232/128 | 0/0 | 45.787 | 1858.5 | 91.05% | 3,072 | 98,304 | BF16:786,432 | 442,368 | 813,324 | 148,785 |
| KDA-P1 | 6 | state scan | 148.783 | 24576 | 128 | 176/128 | 0/0 | 114.778 | 771.4 | 93.13% | 768 | 589,824 | BF16:4,718,592 | 0 | 806,406 | 4,882,392 |
| KDA-P1 | 7 | output | 8.653 | 98304 | 128 | 104/128 | 0/0 | 24.606 | 2843.7 | 92.87% | 3,072 | 49,152 | BF16:393,216 | 49,152 | 91,105 | 250,751 |
| DSA-D1 | 1 | decode logits FP8 | 3.645 | 16384 | 128 | 408/128 | 0/0 | 0.028 | 7.6 | 99.09% | 512 | 0 | 0 | 0 | 0 | 67,708 |
| DSA-D1 | 2 | wave32 radix top-k | 17.186 | 256 | 256 | 40/128 | 0/0 | 0.014 | 0.8 | 86.70% | 8 | 0 | 0 | 3,393 | 1,638 | 877 |
| DSA-D1 | 3 | selected dense WMMA | 6.690 | 2048 | 128 | 264/128 | 0/0 | 0.169 | 25.2 | 98.67% | 64 | 5,632 | FP8:45,056 | 4,352 | 29,639 | 413 |
| DSA-D1 | 4 | selected WMMA reduce | 2.484 | 1024 | 128 | 48/128 | 0/0 | 0.127 | 51.3 | 63.70% | 32 | 0 | 0 | 0 | 1,008 | 0 |
| DSA-P1 | 1 | combine scoring query heads | 12.539 | 131072 | 32 | 384/128 | 0/0 | 2.813 | 224.3 | 96.37% | 4,096 | 0 | 0 | 0 | 0 | 5,502 |
| DSA-P1 | 2 | prefill logits FP8 | 200.981 | 33554432 | 256 | 48/128 | 0/0 | 59.820 | 297.6 | 97.33% | 1,048,576 | 0 | 0 | 0 | 0 | 7,788,211 |
| DSA-P1 | 3 | wave32 radix top-k | 84.527 | 1048576 | 256 | 40/128 | 0/0 | 32.314 | 382.3 | 79.88% | 32,768 | 0 | 0 | 6,862,486 | 2,350,085 | 947,257 |
| DSA-P1 | 4 | selected dense WMMA | 714.871 | 524288 | 128 | 296/128 | 0/0 | 4.162 | 5.8 | 99.48% | 16,384 | 20,660,224 | FP8:165,281,792 | 1,114,112 | 68,495,165 | 224,300 |

Profiler dispatch-duration sums are not substitutes for complete workload latency. Counter passes alter timing and were collected separately.

## Counter definitions and limitations

- Read bytes: `FetchSize * 1024`; rocprof reports `FetchSize` in KiB at the GL2 external interface. It is not proof that every byte reached DRAM.
- Read GB/s: derived read bytes divided by the same pass's dispatch duration.
- L2 hit: `GL2C_HIT / (GL2C_HIT + GL2C_MISS)`.
- Waves: raw `SQ_WAVES`, a launch count rather than mean occupancy.
- WMMA instructions: raw `SQ_INSTS_VEC32_VALU_WMMA`.
- WMMA FLOPs: matching nonzero raw FP8, FP16, or BF16 event. This is not a utilization percentage.
- LDS bank conflict: custom TCP event 132.
- LDS segment stall: custom TCP event 164.
- VC segment stall: custom TCP event 165.
- Exact HBM writes remain unavailable. `GL2C_WRITE_SECTORS` and `GL2C_EA_WRREQ_DRAM` were retained as raw counts.
- TX source counters were collected, but no VMEM dependency-latency value is reported because their reduction semantics were not validated.

The MGCG override in `attention/gfx1250_rocprof/` was enabled only for counter collection. It made the WMMA and custom conflict events nonzero. All eight registers were verified as `0x0` after collection.

## Collection notes

The first `kda-decode` memory pass terminated with a rocprofv3 `SIGTRAP` in `rocprofiler::common::container::pool<>::acquire()`. The GPU remained responsive, the MGCG cleanup ran successfully, and the missing pass was retried under a fresh process. All 36 final counter and kernel-trace artifacts are present.

## Exact command shapes

### Latency

```bash
gpu-lock docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /home/zhewenyu/tokenspeed:/workspace/tokenspeed:ro \
  -v /tmp/tokenspeed-kernel-amd-benchmarks:/workspace/benchmarks:ro \
  -e PYTHONPATH=/workspace/tokenspeed/python:/workspace/tokenspeed/tokenspeed-kernel/python:/workspace/tokenspeed/tokenspeed-kernel-amd/python \
  tokenspeed-kimi-smoke:local python \
  /workspace/benchmarks/attention/profile_matched_attention.py \
  --case all --expected-arch gfx1250 --environment physical \
  --warmup 2 --repeats 5 --output /results/latency.json
```

### Profiler passes

Each named case used `--warmup 0 --repeats 1` under `rocprofv3 --kernel-trace`. The six PMC groups were:

```text
FetchSize

GL2C_WRITE_SECTORS GL2C_HIT GL2C_MISS GL2C_EA_WRREQ_DRAM

SQ_WAVES SQ_INSTS_ALL SQ_WAVE_CYCLES SQ_BUSY_CYCLES

SQ_INST_CYCLES_VALU_WMMA SQ_INSTS_VEC32_VALU_WMMA
SQ_VALU_WMMA_FLOP_FP8 SQ_VALU_WMMA_FLOP_FP16 SQ_VALU_WMMA_FLOP_BF16

TX_VMW_VMW_LATENCY TX_VMW_VCA_REQ_STATE_READ TX_VMW_LFIFO_STALL
TX_VMW_READ_SETCONFLICT_STALL TX_VMW_WRITE_SETCONFLICT_STALL
SPI_RA_LDS_CU_FULL_CSN

TX_PERF_SEL_VMW_LDS_BANK_CONFLICT
TX_PERF_SEL_VMW_CROSS_PORT_SEGMENT_CONFLICT_LDS_STALLED_CYCLES
TX_PERF_SEL_VMW_CROSS_PORT_SEGMENT_CONFLICT_VC_STALLED_CYCLES
```

The final group used `-E attention/gfx1250_rocprof/extra.yaml`; all counter passes ran with the documented MGCG override.

## Raw artifacts

- Collection root: `/tmp/matched-attention-70f69266-20260820/`
- Case descriptions: `matched-attention-cases.json`
- Environment: `environment.json`
- Unprofiled latency: `latency.json`
- Normalized dispatch data: `profile-summary.json`
- Six profiler passes per case: `profiles/<case>/<pass>/`

Raw profiler CSVs remain on the collection host and are not committed.

## Conclusions

- All six required GFX1250 workloads passed at `70f69266`.
- DSA-P1 improved from 2,470.439 µs at `9d1b02ed` to 1,023.501 µs (-58.57%).
- KDA-P1 improved by 11.55% to 286.301 µs, and MLA-D1 improved by 11.18% to 106.520 µs.
- DSA-D1 regressed by 23.30% to 162.123 µs; this warrants follow-up.
- The MGCG override resolves the former zero-counter limitation for WMMA and custom LDS/segment-conflict PMCs on this stack.
