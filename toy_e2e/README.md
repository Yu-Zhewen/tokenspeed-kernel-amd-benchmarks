# Kimi-K3 one-GPU TP8/EP1 benchmark

This package estimates one rank of Kimi-K3 TP8/EP1 on one physical AMD GPU.
It constructs the complete 93-layer model with the production rank shape:

- attention TP8, dense TP8, and MoE TP8;
- EP1, so every layer has all 896 expert IDs and each expert is TP-sharded;
- rank 0 only, with shape-correct local collective substitutes;
- scheduler-driven 4096-token prefill and 1024-token decode workloads at
  concurrency 1 and 16.

The same code and rank-local checkpoint are intended for both `gfx950`
(MI355X) and `gfx1250` (MI450). The checked-in results are measured on one
physical MI355X. Follow the gfx1250 section below on a physical MI450; neither
FFM nor AM is part of this workflow.

## What is measured

The harness reports:

- full-model prefill and eager decode latency;
- steady-state CUDA-graph decode latency;
- sampled KDA attention, MLA attention, dense FFN, and MoE time;
- first-token latency, per-user decode rate, and aggregate output rate;
- logical collective call counts and payload bytes;
- model allocation and hybrid KDA/MLA cache geometry.

Both load paths execute the same full-depth model and real weights:

1. `raw-rank-state` loads the portable rank-0 artifact directly.
2. `safetensors` reads the full source checkpoint and TP-shards it while
   loading. This is the reference used to check the artifact path.

The word “logical” applies only to communication. No reduced layers, repeated
experts, dummy weights, or simulated GPU are used in the reported comparison.

This is a rank-compute estimator, not an eight-GPU benchmark. TP collectives
are replaced locally so their shapes and traffic can be recorded without
RCCL. The results therefore exclude communication latency, cross-rank
synchronization, and communication/compute overlap. They do not establish
model-output correctness or eight-GPU serving throughput.

## Validated environment

- GPU: one MI355X (`gfx950:sramecc+:xnack-`), 288 GiB HBM
- TokenSpeed and kernel sources:
  `0b1061eb9fe1df36a4e48e5c9c291cd753af9e89`
- PyTorch: `2.11.0+rocm7.2`
- HIP runtime reported by PyTorch: `7.2.26015`
- Transformers: `5.12.0`
- Triton: `3.6.0`
- Kimi-K3 source revision:
  `eaf5a944bfc8c57438bbce226feef9f6bdbdaae1`

Use a TokenSpeed environment that supports the target GPU and the pinned
source revision. The benchmark needs one visible GPU and approximately
191 GiB for the raw model before kernel preprocessing; reserve additional HBM
for processed weights, a KV cache, workspaces, and CUDA graphs. A 288 GiB GPU
is the validated minimum.

Set paths inside that environment:

```bash
export BENCHMARKS_ROOT=/workspace/tokenspeed-kernel-amd-benchmarks
export TOKENSPEED_ROOT=/workspace/tokenspeed
export PYTHONPATH="$BENCHMARKS_ROOT:$TOKENSPEED_ROOT/python:$TOKENSPEED_ROOT/tokenspeed-kernel/python:$TOKENSPEED_ROOT/tokenspeed-kernel-amd/python"
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
```

Only one device may be visible. Adapt the paths and container bind mounts to
the machine; do not change the TP8/EP1 settings in the scripts.

## Validate the package

From the benchmark repository root:

```bash
python3 -m pytest -q -p no:cacheprovider toy_e2e/tests

python3 -m ruff check \
  toy_e2e/rank_checkpoint.py \
  toy_e2e/logical_rank.py \
  toy_e2e/benchmark_logical_rank.py \
  toy_e2e/scripts/export_rank_local_checkpoint.py \
  toy_e2e/tests
```

The tests are CPU-only. They verify bounded checkpoint parts, manifest
metadata, pre-processing load order, exact tensor restoration, and logical
collective shapes and traffic accounting.

## Create the portable rank-local checkpoint

This step needs the full Kimi-K3 source checkpoint. It is run once on a
conversion machine; a benchmark machine then needs only the resulting
rank-local directory.

Allow roughly 1.5 TiB for the source plus 191 GiB for the output. The exporter
initializes TP8/EP1 rank 0, streams the source tensors through TokenSpeed's
normal sharding logic, and writes bounded safetensors parts before any
GPU-specific kernel preprocessing:

```bash
python3 "$BENCHMARKS_ROOT/toy_e2e/scripts/export_rank_local_checkpoint.py" \
  --source /data/models/Kimi-K3 \
  --output /data/models/kimi-k3-tp8ep1-rank0 \
  --part-gib 2 \
  --tokenspeed-revision \
    0b1061eb9fe1df36a4e48e5c9c291cd753af9e89 \
  --source-revision \
    eaf5a944bfc8c57438bbce226feef9f6bdbdaae1
```

The output directory must be absent or empty. Use `--overwrite` only when an
existing partial or obsolete export should be replaced.

The validated artifact contains 114 safetensors parts and 190.98 GiB of tensor
payload. Its `rank-local-manifest.json` is written last, so the absence of a
complete manifest means the export must not be used.

Copy the complete directory to another machine, for example:

```bash
rsync -a --info=progress2 \
  /data/models/kimi-k3-tp8ep1-rank0/ \
  mi450-host:/data/models/kimi-k3-tp8ep1-rank0/
```

Do not convert this artifact with a kernel-specific sharded-state exporter.
Kimi-K3 MXFP4 preprocessing is architecture-dependent. The custom format
stores raw TP-sliced weights and runs `process_weights_after_loading` on the
target GPU, which is what permits one artifact to serve gfx950 and gfx1250.
See [checkpoint-preparation.md](docs/checkpoint-preparation.md) for the format
contract and validation checklist.

## Run the rank-local benchmark

This is the primary one-GPU command:

```bash
python3 "$BENCHMARKS_ROOT/toy_e2e/benchmark_logical_rank.py" \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --prompt-tokens 4096 \
  --output-tokens 1024 \
  --concurrency 1 16 \
  --chunked-prefill-size 8192 \
  --cache-gib 32 \
  --warmup-output-tokens 2 \
  --profile-output-tokens 8 \
  --decode-graph-replays 20 \
  --output /data/results/kimi-k3-gfx950-rank-local-4k-1k.json
```

The eager workload advances the real TokenSpeed scheduler through prefill and
all 1024 requested output tokens. The graph section captures the first full
decode batch after prefill and replays that static batch to isolate
production-shaped graph execution from Python launch overhead.

## Run the full-checkpoint reference

On the conversion machine, repeat the same workload with the full source
checkpoint:

```bash
python3 "$BENCHMARKS_ROOT/toy_e2e/benchmark_logical_rank.py" \
  --checkpoint /data/models/Kimi-K3 \
  --load-format safetensors \
  --prompt-tokens 4096 \
  --output-tokens 1024 \
  --concurrency 1 16 \
  --chunked-prefill-size 8192 \
  --cache-gib 32 \
  --warmup-output-tokens 2 \
  --profile-output-tokens 8 \
  --decode-graph-replays 20 \
  --output /data/results/kimi-k3-gfx950-full-source-4k-1k.json
```

The two result files should report the same 93 layers, TP8/EP1 mapping, 896
local expert IDs, kernel solutions, cache geometry, collective counts, and
similar steady-state latency. Loading time is intentionally outside the
reported workload latency.

## Run on gfx1250

Use the copied `kimi-k3-tp8ep1-rank0` directory on a machine with one physical
MI450. Use the target machine's supported ROCm/TokenSpeed runtime; do not use
FFM or AM.

Before benchmarking, verify physical architecture detection:

```bash
python3 - <<'PY'
import torch

assert torch.cuda.device_count() == 1
arch = torch.cuda.get_device_properties(0).gcnArchName
print(torch.cuda.get_device_name(0), arch)
assert arch.startswith("gfx1250")
PY
```

Run the rank-local command unchanged except for the output filename:

```bash
python3 "$BENCHMARKS_ROOT/toy_e2e/benchmark_logical_rank.py" \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --prompt-tokens 4096 \
  --output-tokens 1024 \
  --concurrency 1 16 \
  --chunked-prefill-size 8192 \
  --cache-gib 32 \
  --warmup-output-tokens 2 \
  --profile-output-tokens 8 \
  --decode-graph-replays 20 \
  --output /data/results/kimi-k3-gfx1250-rank-local-4k-1k.json
```

Successful completion proves target-side gfx1250 preprocessing and execution
for the portable artifact. Preserve the complete JSON and the console log so
the result can be added beside the MI355X baseline.

## Files

- `rank_checkpoint.py`: portable raw rank-state writer and loader.
- `logical_rank.py`: TP8/EP1 rank-0 configuration and local collectives.
- `benchmark_logical_rank.py`: scheduler, cache, eager, graph, and breakdown
  benchmark.
- `scripts/export_rank_local_checkpoint.py`: full-source conversion CLI.
- `tests/test_rank_checkpoint.py`: CPU checkpoint-format tests.
- `docs/checkpoint-preparation.md`: artifact contract and transfer validation.
- `docs/gfx1250-validation.md`: physical MI450 handoff procedure.
- `docs/results.md`: measured gfx950 comparison and gfx1250 handoff status.
