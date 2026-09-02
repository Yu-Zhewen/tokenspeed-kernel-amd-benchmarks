# Kimi-K3 one-GPU TP8/EP1 gfx950 results

## Outcome

Overall: **PASS** for full-depth TP8/EP1 rank-0 export, direct rank-state
loading, target-side gfx950 preprocessing, scheduler-driven 4K/1K execution,
CUDA-graph decode, and comparison with the full source checkpoint.

The portable artifact and the full-checkpoint reference produced identical:

- 93-layer model structure;
- attention TP8, MoE TP8, and EP1 mapping;
- 896 local expert IDs per MoE layer;
- gfx950 Gluon MoE solution;
- 209.95 GiB processed model allocation;
- 32 GiB hybrid-cache geometry and capacity;
- logical collective call counts and payload bytes.

Their graph-decode medians differ by less than 0.5% at both tested
concurrencies. This is the strongest timing check because it removes most
Python launch variance. The portable path therefore shows no material
steady-state compute regression against loading the full checkpoint.

## Environment

- Physical GPU: one MI355X
- Architecture: `gfx950:sramecc+:xnack-`
- TokenSpeed and kernel source:
  `0b1061eb9fe1df36a4e48e5c9c291cd753af9e89`
- PyTorch: `2.11.0+rocm7.2`
- HIP: `7.2.26015`
- Transformers: `5.12.0`
- Triton: `3.6.0`
- Kimi-K3 source revision:
  `eaf5a944bfc8c57438bbce226feef9f6bdbdaae1`
- Prompt/output: 4096/1024 tokens
- Concurrency: 1 and 16
- Chunked-prefill budget: 8192 tokens
- Configured KV cache: 32 GiB

Raw outputs:

- [rank-local checkpoint result](../results/gfx950-rank-local-4k-1k.json)
- [full source checkpoint result](../results/gfx950-full-source-4k-1k.json)

## Rank-local artifact

The MI355X export completed in 193.88 seconds and produced:

- format `tokenspeed_raw_rank_state_v1`;
- 114 safetensors parts;
- 2,342 state tensors;
- 205,057,976,816 tensor bytes (190.98 GiB);
- 191.28 GiB raw model allocation during export;
- all 93 layers and all 896 experts, with TP rank-0 slicing.

The direct loader rebuilt architecture-neutral derived weights, then applied
gfx950 kernel preprocessing. It completed in 43.39 seconds, used 214.28 GiB
peak HBM during load, and retained 209.95 GiB after loading.

The full source path completed in 134.82 seconds with the same 214.28 GiB load
peak and 209.95 GiB retained allocation. On this filesystem run, the artifact
loaded 3.11 times faster and saved 91.43 seconds (67.82%). Load time depends
strongly on storage and page-cache state and is outside request latency.

With the 32 GiB cache, runtime peak allocation was 244.69 GiB. The cache arena
contained 1,617 LCM blocks, 2,483,712 physical slots, and 2,334,336
scheduler-visible token slots after reserved/null capacity.

## Concurrency 1

Rank-local versus full-source measurements:

- first-token latency: 277.16 ms versus 278.05 ms;
- prefill model latency: 269.61 ms versus 270.31 ms (`-0.26%`);
- eager decode model p50: 65.05 ms versus 62.95 ms (`+3.35%`);
- eager decode step-wall p50: 66.11 ms versus 63.90 ms;
- full eager-workload output rate: 15.06 versus 15.57 tok/s;
- graph decode p50: 10.699 ms versus 10.744 ms (`-0.42%`);
- graph decode rate: 93.46 versus 93.07 tok/s.

The eager result covers all 1,023 decode steps after the first token. The
rank-local p95 was 66.60 ms and the source p95 was 64.11 ms. The graph result
uses 20 replays at sequence length 4,097 and was stable within 0.05 ms in each
run.

Sampled eager component p50, shown as rank-local versus full source:

- KDA attention: 17.054 ms versus 16.220 ms;
- MLA attention: 11.012 ms versus 10.779 ms;
- MoE: 30.843 ms versus 30.424 ms;
- dense FFN: 0.149 ms versus 0.136 ms.

These values are category totals across the complete 93-layer forward, not
single-layer latency.

## Concurrency 16

The scheduler processed the 65,536 prompt tokens in eight 8,192-token prefill
steps. Rank-local versus full-source measurements:

- first-token latency p50: 1,954.76 ms versus 1,956.18 ms;
- first-token latency maximum: 3,474.08 ms versus 3,477.46 ms;
- prefill model step p50: 427.246 ms versus 427.496 ms (`-0.06%`);
- eager decode model p50: 79.651 ms versus 76.536 ms (`+4.07%`);
- eager decode step-wall p50: 81.372 ms versus 78.416 ms;
- full eager-workload aggregate output: 188.25 versus 194.25 tok/s;
- graph decode p50: 18.514 ms versus 18.568 ms (`-0.29%`);
- graph per-user decode: 54.03 versus 53.83 tok/s;
- graph aggregate decode: 864.46 versus 861.34 tok/s.

The source eager run had one 497.71 ms decode outlier; its p95 remained
78.10 ms. Medians are used for the direct comparison.

Sampled eager component p50, shown as rank-local versus full source:

- KDA attention: 14.021 ms versus 13.297 ms;
- MLA attention: 14.644 ms versus 13.983 ms;
- MoE: 38.631 ms versus 36.745 ms;
- dense FFN: 0.122 ms versus 0.121 ms.

The component hooks add event overhead, so those values explain the eager
profile and must not be summed to predict the separately captured graph.

## Logical collective traffic

The two load paths recorded exactly the same traffic.

Each decode forward issued:

- 187 logical all-reduces;
- one logical all-gather-into-tensor.

At concurrency 1, a decode forward accounted for 3,340,288 all-reduce input
bytes and expanded 40,960 gather input bytes to 327,680 output bytes. At
concurrency 16, those payloads were 53,444,608 bytes and
655,360-to-5,242,880 bytes respectively.

The calls are executed by local substitutes. Their counts and payloads
describe the missing TP8 communication, but the measured latency does not
include RCCL transfer, cross-rank synchronization, or communication overlap.

## Interpretation

Use the graph-decode result as the best rank-compute estimate for a
production-shaped decode graph. Use the eager 1,024-token workload to validate
scheduler progression, sequence growth from 4,096 to 5,120, cache allocation,
and long-run stability. The 3–4% eager decode difference between the two load
paths is consistent with run-to-run launch/operating variance; it is not
present in graph decode, prefill, model allocation, cache geometry, or
collective traces.

These numbers are not eight-GPU TP8/EP1 serving throughput and do not include
sampling, tokenization, HTTP, or real collective latency.

## Validation

```text
python3 -m pytest -q -p no:cacheprovider toy_e2e/tests
5 passed

python3 -m ruff check toy_e2e
All checks passed!
```

No TokenSpeed runtime or kernel source was modified.

## gfx1250 status

Physical gfx1250 execution is intentionally not simulated on this machine.
The same raw rank-state artifact is ready for target-side MI450 preprocessing.
Follow [gfx1250-validation.md](gfx1250-validation.md) on the external physical
MI450 and add its complete JSON beside these gfx950 results.
