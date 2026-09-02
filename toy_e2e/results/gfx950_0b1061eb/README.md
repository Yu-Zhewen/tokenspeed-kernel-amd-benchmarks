# Kimi-K3 TP8/EP1 on gfx950 at `0b1061eb`

This revision directory separates the logical-rank estimate from physical
eight-GPU serving. Each result set has its own report with the same analytical
structure: environment, measurement contract, performance, hotspots,
interpretation, artifacts, and limitations.

## Result sets

| Scope | Report | Primary measurement | Hotspot unit |
|---|---|---|---|
| One physical GPU, logical TP8 rank 0 | [`one_gpu/`](one_gpu/) | Rank-local model and CUDA-graph estimate | Timed model components and layer modules |
| Eight physical GPUs, real TP8/EP1 | [`real_8gpu/`](real_8gpu/) | HTTP serving with real RCCL/Iris collectives | Exact GPU kernels summed across all ranks |

Both result sets use:

- 4096 input and 1024 output tokens;
- concurrency 1 and 16;
- TokenSpeed `0b1061eb9fe1df36a4e48e5c9c291cd753af9e89`;
- Kimi-K3 `eaf5a944bfc8c57438bbce226feef9f6bdbdaae1`;
- AMD Instinct MI355X (`gfx950:sramecc+:xnack-`).

## Direct comparison

| C | One-GPU graph decode p50 | Real TP8 TPOT p50 | Real overhead | One-GPU aggregate estimate | Real overall output | Real steady output |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.736 ms | 12.36 ms | +15.12% | 93.13 tok/s | 78.22 tok/s | 78.50 tok/s |
| 16 | 18.470 ms | 24.28 ms | +31.46% | 866.11 tok/s | 556.13 tok/s | 816.84 tok/s |

The one-GPU result executes the full rank-0 compute graph but substitutes
rank-spanning collectives locally. It is a rank-compute estimate, not a second
serving result. The real result includes communication, synchronization,
queueing, HTTP, and sampling.

At C16, overall service throughput is 35.79% below the one-GPU graph estimate,
while the final steady window is 5.69% below it. Overall throughput includes
prefill and request-wave overhead; the steady window is decode dominated.

## Status

- [`one_gpu/README.md`](one_gpu/README.md): **PASS** for portable rank-local
  loading, full-source equivalence, 4K/1K execution, and graph decode.
- [`real_8gpu/README.md`](real_8gpu/README.md): **PASS** for 51/51 exact-token
  requests and stage-separated all-rank kernel capture.
- Physical gfx1250 execution remains pending; follow
  [`../../docs/gfx1250-validation.md`](../../docs/gfx1250-validation.md).
