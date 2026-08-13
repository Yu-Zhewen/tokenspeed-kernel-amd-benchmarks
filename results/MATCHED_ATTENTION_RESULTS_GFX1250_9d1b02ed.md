# Latest-main GFX1250 matched-attention results

## Status

- Collection date: 2026-08-13 UTC
- GFX1250 physical collection: complete
- Required workloads: 6/6 passed
- Profiled measured-scope dispatches: 21
- TokenSpeed revision: `9d1b02ed9c001544e35a41aa4b216dc8caa2240a`
- Matched GFX950 data exists at the same TokenSpeed revision, but the hardware,
  ROCm, PyTorch, and container stacks differ, so the comparison is directional
  rather than a controlled architecture A/B.

## Software and hardware setup

| Field | Value |
|---|---|
| Device | MI455X-class device (PyTorch: `AMD Radeon Graphics`) |
| Architecture | `gfx1250` |
| Environment | physical, GPU 0 |
| Host | `heliosr-1b114-d04-1` |
| Container image | `tokenspeed-kimi-smoke@sha256:77179418c81a72f003c08656b62b9d7dec3bbdb794a8d5af2f19cf37b2289405` |
| OS | Ubuntu 24.04 host and container |
| PyTorch | `2.11.0+rocm7.15.0a20260728` |
| ROCm | `7.15.0` |
| rocprofv3 | 1.3.5 (`44be71b52284948e58c93f65f46910399773fdcd`) |
| tokenspeed-triton | `3.8.10.post20260721` |
| TokenSpeed commit | `9d1b02ed9c001544e35a41aa4b216dc8caa2240a` |
| TokenSpeed tree | `0f77cd559e67304eee7b1e0f6542bdb879fe52a2` |
| tokenspeed-kernel tree | `e8fef94593b47e7691b3a24290dc67e9e6d397f3` |
| tokenspeed-kernel-amd tree | `25d692089075c1764e46c1a5b3087c6921a77430` |
| Runner revision | benchmark repository `main` at collection time; runner content inherited from gist revision `29150c094de69af0c13fc39a92237ba47bd59fbc` |
| Latency warmups / repeats | 2 / 5 |
| Profiler warmups / repeats | 0 / 1 |

The source trees were mounted read-only through `PYTHONPATH`. Every physical
GPU command ran under the host `gpu-lock`.

## Physical workload latency

Complete unprofiled GPU-event averages:

| ID | Runner case | GFX1250 latency (µs) | Prior GFX1250 `f1e194c9` (µs) | Change |
|---|---|---:|---:|---:|
| MLA-D1 | `mla-decode` | 119.931 | 221.308 | -45.81% |
| MLA-P1 | `mla-prefill` | 140.570 | 163.998 | -14.29% |
| KDA-D1 | `kda-decode` | 51.701 | 54.594 | -5.30% |
| KDA-P1 | `kda-prefill` | 323.683 | 324.174 | -0.15% |
| DSA-D1 | `dsa-decode-pipeline` | 131.484 | 134.361 | -2.14% |
| DSA-P1 | `dsa-prefill-pipeline-4k` | 2,470.439 | 40,823.276 | -93.95% |

The current GFX1250 DSA-P1 result is 16.53 times faster than the earlier
GFX1250 collection. MLA-D1 is 1.85 times faster and now selects the fused
projected-value path.

For context, the separately collected GFX950 results at the same TokenSpeed
revision are 85.425, 142.105, 45.216, 292.954, 138.825, and 2,327.593 µs in
the same row order. Stack differences prevent attributing those deltas solely
to hardware architecture.

## Per-dispatch physical results

The rows below are the complete repeated suffix corresponding to the measured
invocation. MLA-D1 includes the query-offset construction emitted by the
selected implementation. DSA-P1 includes three framework utility dispatches
before logits, radix selection, and selected attention.

Duration and `FetchSize` come from the same profiler pass. `FetchSize` is
reported as GL2-to-external-interface traffic and is not proof that every byte
reached DRAM. GL2 write sectors and EA write requests are raw counts, not byte
values. Occupancy and WMMA utilization remain unavailable because the source
events return zero on this profiler stack.

| Case | # | Dispatch | Duration µs | Grid | Workgroup | VGPR | SGPR | LDS B | Scratch B | Read MB | Read GB/s | GL2 write sectors | EA write req | L2 hit | Waves |
|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLA-D1 | 1 | query-offset arange | 1.643 | 128×1×1 | 128×1×1 | 8 | 128 | 0 | 0 | 0.002 | 1.1 | 758 | 355 | 49.28% | 4 |
| MLA-D1 | 2 | MLA decode main | 5.168 | 64×1×64 | 64×1×1 | 280 | 128 | 0 | 0 | 1.534 | 296.8 | 137550 | 393 | 99.96% | 128 |
| MLA-D1 | 3 | fused reduce/project/gate | 27.160 | 49152×1×1 | 256×1×1 | 40 | 128 | 0 | 0 | 1.519 | 55.9 | 6918 | 366 | 97.91% | 1536 |
| MLA-P1 | 1 | MLA prefill | 127.550 | 128×12×32 | 128×1×1 | 264 | 128 | 0 | 0 | 0.006 | 0.1 | 377624 | 377225 | 97.93% | 1536 |
| KDA-D1 | 1 | KDA recurrent decode | 3.565 | 512×12×1 | 128×1×1 | 56 | 128 | 0 | 0 | 0.761 | 213.5 | 48260 | 324 | 90.92% | 192 |
| KDA-P1 | 1 | BF16 workspace fill | 3.606 | 393216×1×1 | 256×1×1 | 8 | 128 | 0 | 0 | 0.007 | 1.9 | 189036 | 188677 | 96.54% | 12288 |
| KDA-P1 | 2 | preprocess intra | 97.786 | 16384×12×1 | 256×1×1 | 168 | 128 | 0 | 0 | 70.325 | 719.2 | 4761302 | 1221958 | 81.95% | 6144 |
| KDA-P1 | 3 | BF16 workspace fill | 2.964 | 393216×1×1 | 256×1×1 | 8 | 128 | 0 | 0 | 0.005 | 1.7 | 189266 | 188961 | 93.96% | 12288 |
| KDA-P1 | 4 | solve/merge 64 | 12.458 | 2048×12×1 | 32×1×1 | 104 | 128 | 0 | 0 | 6.054 | 485.9 | 829570 | 36295 | 92.07% | 768 |
| KDA-P1 | 5 | W/U vector | 24.757 | 8192×12×1 | 128×1×1 | 232 | 128 | 0 | 0 | 45.441 | 1835.5 | 5405978 | 876346 | 91.62% | 3072 |
| KDA-P1 | 6 | state scan | 163.123 | 1024×12×1 | 128×1×1 | 264 | 128 | 0 | 0 | 123.149 | 754.9 | 5692400 | 1463166 | 88.53% | 384 |
| KDA-P1 | 7 | output | 18.026 | 8192×12×1 | 128×1×1 | 264 | 128 | 0 | 0 | 24.281 | 1347.0 | 2626284 | 92254 | 92.87% | 3072 |
| DSA-D1 | 1 | decode logits FP8 | 2.684 | 128×128×1 | 128×1×1 | 408 | 128 | 0 | 0 | 0.014 | 5.2 | 4692 | 324 | 99.72% | 512 |
| DSA-D1 | 2 | wave32 radix top-k | 15.624 | 256×1×1 | 256×1×1 | 40 | 128 | 0 | 0 | 0.021 | 1.4 | 2256 | 865 | 85.79% | 8 |
| DSA-D1 | 3 | selected dense WMMA | 46.550 | 128×1×1 | 128×1×1 | 264 | 128 | 0 | 0 | 0.008 | 0.2 | 1254 | 351 | 99.74% | 4 |
| DSA-P1 | 1 | BF16-to-FP32 copy | 37.415 | 2097152×1×1 | 256×1×1 | 8 | 128 | 0 | 0 | 24.416 | 652.6 | 2010196 | 2634428 | 24.30% | 65536 |
| DSA-P1 | 2 | index weighting | 33.610 | 4194304×1×1 | 128×1×1 | 16 | 128 | 0 | 0 | 128.990 | 3837.8 | 2010566 | 1295494 | 66.58% | 131072 |
| DSA-P1 | 3 | index reduction | 9.775 | 32768×4×1 | 32×4×1 | 32 | 128 | 512 | 0 | 33.107 | 3386.9 | 63604 | 364555 | 54.36% | 4096 |
| DSA-P1 | 4 | prefill logits FP8 | 1,585.405 | 524288×128×1 | 128×1×1 | 264 | 128 | 0 | 0 | 67.335 | 42.5 | 16059152 | 348609 | 97.45% | 2097152 |
| DSA-P1 | 5 | wave32 radix top-k | 92.939 | 1048576×1×1 | 256×1×1 | 40 | 128 | 0 | 0 | 32.257 | 347.1 | 3535282 | 1647847 | 83.98% | 32768 |
| DSA-P1 | 6 | selected dense WMMA | 707.976 | 524288×1×1 | 128×1×1 | 296 | 128 | 0 | 0 | 21.084 | 29.8 | 1005218 | 1627380 | 99.19% | 16384 |

Profiler dispatch-duration sums are not substitutes for complete workload
latency. They come from instrumented single-counter executions with different
runtime overhead.

## Counter definitions and limitations

| Report field | Raw counter or source | Definition |
|---|---|---|
| Read MB / GB/s | `FetchSize` | Reported KiB multiplied by 1024; traffic leaving GL2 toward the external-memory interface |
| GL2 write sectors | `GL2C_WRITE_SECTORS` | Raw GL2 write-sector count; not exact HBM bytes |
| EA write req | `GL2C_EA_WRREQ_DRAM` | Raw external-address write request count |
| L2 hit | `GL2C_HIT`, `GL2C_MISS` | `hit / (hit + miss)` |
| Waves | `SQ_WAVES` | Total launched waves, not occupancy |
| Duration | profiler kernel trace | `(End_Timestamp - Start_Timestamp) / 1000` |
| LDS / scratch / registers | profiler kernel trace | Static resource fields reported for the dispatch |

`SQ_INSTS_ALL`, `SQ_WAVE_CYCLES`, the WMMA instruction counters, and the TCP
latency source events still returned zero. This reproduces the previously
documented GFX1250 profiler limitation; no occupancy, WMMA utilization, or
dependency-latency value is inferred from those zeros.

## Exact command shapes

### Latency

```bash
python profile_matched_attention.py --case all --expected-arch gfx1250 \
  --environment physical --warmup 2 --repeats 5 --output latency.json
```

### Reporting profiler passes

For each named case, the runner used `--warmup 0 --repeats 1` under
`rocprofv3 --kernel-trace`, with these four counter groups:

```text
FetchSize

GL2C_WRITE_SECTORS GL2C_HIT GL2C_MISS GL2C_EA_WRREQ_DRAM

SQ_WAVES SQ_INSTS_ALL SQ_WAVE_CYCLES SQ_BUSY_CYCLES
SQ_INSTS_VEC32_VALU_WMMA SQ_VALU_WMMA_FLOP_FP8

TX_VMW_VMW_LATENCY TX_VMW_VCA_REQ_STATE_READ TX_VMW_LFIFO_STALL
TX_VMW_READ_SETCONFLICT_STALL TX_VMW_WRITE_SETCONFLICT_STALL
SPI_RA_LDS_CU_FULL_CSN
```

## Artifacts

- Case description:
  `matched-attention-9d1b02ed-20260813/matched-attention-cases.json`
- Unprofiled latency:
  `matched-attention-9d1b02ed-20260813/latency.json`
- Smoke result:
  `matched-attention-9d1b02ed-20260813/gfx1250-smoke.json`
- Four profiler passes per case:
  `matched-attention-9d1b02ed-20260813/profiles/<case>/<pass>/`

Raw profiler artifacts remain on the collection host and are not committed to
this report repository.

## Conclusions

- All six GFX1250 workloads completed at `9d1b02ed`.
- MLA-D1 now uses a fused reduce/project/gate dispatch and improved from
  221.308 to 119.931 µs relative to the earlier GFX1250 collection.
- DSA-P1 improved from 40.823 ms to 2.470 ms. Its measured dispatch sequence is
  1.585 ms logits, 0.708 ms selected attention, 0.093 ms radix top-k, plus
  approximately 0.081 ms of framework utility work.
- KDA-P1 remains dominated by the 163.123 µs state scan and 97.786 µs
  preprocess dispatches.
- MLA-P1 is 140.570 µs and now approximately matches the separately collected
  GFX950 result at this source revision.
- Cross-architecture deltas remain directional because the GFX950 and GFX1250
  software and hardware environments are not identical.
