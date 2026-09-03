# Kimi-K3 TP8/EP1 three-target performance comparison

This report compares the three completed targets at TokenSpeed revision
`0b1061eb9fe1df36a4e48e5c9c291cd753af9e89`:

- [gfx950 toy 1-GPU logical rank](gfx950_toy_1gpu_0b1061eb/)
- [gfx950 real 8-GPU TP8/EP1 service](gfx950_real_8gpu_0b1061eb/)
- [gfx1250 toy 1-GPU logical rank](gfx1250_toy_1gpu_0b1061eb/)

All three use 4,096 input tokens, 1,024 output tokens, concurrency 1 and 16,
an 8,192-token prefill budget, FP8 KV cache, one warmup wave, and three
measured waves. They also use the same Kimi-K3 revision
`eaf5a944bfc8c57438bbce226feef9f6bdbdaae1`.

## Comparability

The two toy targets are the direct comparison. Each uses one physical GPU to
execute logical TP8 rank 0 with deterministic synthetic prompts and local
substitutes for rank-spanning collectives.

The real gfx950 target is not a one-to-one hardware comparison. It uses eight
physical MI355X GPUs, all TP8 ranks, RCCL plus Iris collectives, HTTP serving,
and EvalScope text. It is included to show how the matched workload behaves in
production-style service.

| Target | Physical GPUs | Execution scope | Prompts | Collectives | HIP / Triton |
|---|---:|---|---|---|---|
| gfx950 toy | 1× MI355X | logical TP8 rank 0 | deterministic synthetic IDs | local substitutes | 7.2 / 3.6 |
| gfx950 real | 8× MI355X | complete TP8/EP1 service | EvalScope random text | RCCL + Iris | 7.2 / 3.6 |
| gfx1250 toy | 1× MI450 | logical TP8 rank 0 | deterministic synthetic IDs | local substitutes | 7.15 / 3.8 |

## Unprofiled end-to-end performance

Higher is better for throughput. Lower is better for TTFT and TPOT.

### Concurrency 1

| Target | Overall output | TTFT p50 / p90 | TPOT p50 / p90 | Per-user decode | Steady decode |
|---|---:|---:|---:|---:|---:|
| gfx950 toy | 78.12 tok/s | 286.60 / 286.63 ms | 12.531 / 12.547 ms | 79.78 tok/s | 79.78 tok/s |
| gfx950 real | 78.22 tok/s | 405.61 / 405.81 ms | 12.360 / 12.490 ms | 80.93 tok/s | 78.50 tok/s |
| gfx1250 toy | 65.44 tok/s | 659.30 / 661.28 ms | 14.654 / 14.655 ms | 68.24 tok/s | 68.24 tok/s |

### Concurrency 16

| Target | Overall output | TTFT p50 / p90 | TPOT p50 / p90 | Per-user decode | Steady decode |
|---|---:|---:|---:|---:|---:|
| gfx950 toy | 586.09 tok/s | 1,955.96 / 3,461.99 ms | 25.418 / 26.865 ms | 39.35 tok/s | 668.47 tok/s |
| gfx950 real | 556.13 tok/s | 4,596.48 / 4,727.20 ms | 24.280 / 27.600 ms | 41.18 tok/s | 816.84 tok/s |
| gfx1250 toy | 386.26 tok/s | 5,119.56 / 9,073.49 ms | 36.453 / 40.312 ms | 27.43 tok/s | 491.16 tok/s |

Toy steady decode is post-transition rolling graph capacity. Real steady
decode is EvalScope's final 30-second completion window. Those two steady
metrics have different scopes and should not be treated as identical.

## Direct gfx1250-to-gfx950 toy comparison

| Metric | C1 change on gfx1250 | C16 change on gfx1250 |
|---|---:|---:|
| Overall output throughput | 16.24% lower | 34.09% lower |
| Per-user decode throughput | 14.47% lower | 30.29% lower |
| Steady decode capacity | 14.47% lower | 26.53% lower |
| p50 TPOT | 16.94% higher | 43.42% higher |
| p50 TTFT | 130.05% higher | 161.74% higher |

At this revision, gfx950 is faster in every directly comparable toy metric.
The gap widens at C16: gfx1250 delivers 65.91% of gfx950's overall output
throughput and 73.47% of its steady decode capacity.

This measures the current architecture-specific kernels and runtime stacks,
not an immutable hardware limit. The gfx1250 run uses a newer HIP, PyTorch,
and Triton stack and remains an early optimization target.

## Gfx950 toy versus real service

The real eight-GPU service produces 0.12% more overall output than the gfx950
toy result at C1 and 5.11% less at C16. Its p50 TPOT is similar: 1.36% lower
at C1 and 4.48% lower at C16. TTFT is 41.53% higher at C1 and 135.00% higher
at C16.

The close output and decode rates do not make the two modes interchangeable.
The real result includes HTTP, prompt handling, eight-rank synchronization,
and physical communication; the toy result executes only rank-local compute
with synthetic prompts.

## Eager GPU hotspot comparison

| Target | C1 prefill dominant | C1 decode dominant | C16 prefill dominant | C16 decode dominant |
|---|---|---|---|---|
| gfx950 toy | MoE, 45.99% | GEMM / quant, 30.56% | Other, 46.16% | GEMM / quant, 36.36% |
| gfx950 real | MoE, 35.16% | Communication, 88.55% | MoE, 24.90% | Communication, 62.52% |
| gfx1250 toy | GEMM / quant, 74.32% | Other, 43.33% | GEMM / quant, 76.28% | GEMM / quant, 47.50% |

Key observations:

- Real gfx950 decode is communication-bound in rank-summed GPU time:
  communication accounts for 88.55% at C1 and 62.52% at C16.
- Gfx1250 prefill is dominated by generic GEMM/quant kernels at both
  concurrencies, identifying the clearest optimization area.
- Gfx1250 C16 decode spends 47.50% in GEMM/quant and 36.07% in the combined
  elementwise, reduction, normalization, cache, and unclassified categories.
- The gfx1250 profiler reports the leading routed-expert work under generic
  `Cijk...` GEMM names. Its small explicit `MoE` category does not mean MoE
  computation is absent.

Profile percentages are shares of summed eager GPU-kernel duration, not
critical-path wall time. They must not replace the unprofiled performance
measurements.

## Conclusion

For current rank-local kernel performance, gfx950 is decisively ahead of
gfx1250, especially at C16. The gfx1250 profile points first to prefill
GEMM/quant work, then C16 decode GEMM/quant and elementwise overhead.

The real gfx950 result remains the production reference. It demonstrates that
physical TP8 decode behavior is dominated by collective communication even
when aggregate output remains close to the rank-local diagnostic result.

All values are taken from the committed `result.json` and
`hotspots/hotspots.json` files in the three linked result directories.
