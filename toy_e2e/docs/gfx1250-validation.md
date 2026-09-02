# gfx1250 physical-GPU validation

## Goal

Run the portable Kimi-K3 TP8/EP1 rank-0 artifact on one physical MI450 and
return a result directly comparable with the MI355X baseline.

This procedure does not use FFM or AM. It must execute on physical gfx1250
hardware with exactly one GPU visible.

## Inputs

Copy these inputs to the target:

- this benchmark repository;
- a compatible TokenSpeed source/runtime installation;
- the complete `kimi-k3-tp8ep1-rank0` directory;
- `SHA256SUMS`, if checksums were generated before transfer.

The full 1.5 TiB source checkpoint is not needed on the target.

The validated source baseline uses:

```text
TokenSpeed: 0b1061eb9fe1df36a4e48e5c9c291cd753af9e89
PyTorch: 2.11.0+rocm7.2
HIP: 7.2.26015
Transformers: 5.12.0
Triton: 3.6.0
```

Use the MI450 machine's supported physical-GPU package or container. Prefer
the project's prebuilt TokenSpeed Triton wheel. Do not rebuild Triton merely
to run this benchmark.

## 1. Verify the artifact

If checksums are present:

```bash
cd /data/models/kimi-k3-tp8ep1-rank0
sha256sum -c SHA256SUMS
```

Inspect the manifest:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("/data/models/kimi-k3-tp8ep1-rank0")
manifest = json.loads((root / "rank-local-manifest.json").read_text())
assert manifest["format"] == "tokenspeed_raw_rank_state_v1"
assert manifest["complete"] is True
assert manifest["tp_size"] == 8 and manifest["tp_rank"] == 0
assert manifest["ep_size"] == 1 and manifest["ep_rank"] == 0
assert manifest["raw_state_before_kernel_preprocessing"] is True
assert len(manifest["parts"]) == 114
assert all((root / part["filename"]).is_file() for part in manifest["parts"])
print(manifest["tensor_count"], manifest["tensor_bytes"] / (1 << 30))
PY
```

The validated artifact reports 190.98 GiB.

## 2. Verify the physical runtime

Make only one GPU visible:

```bash
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
```

Record and validate the environment:

```bash
git -C /workspace/tokenspeed rev-parse HEAD
python3 - <<'PY'
import torch
import transformers
import triton

assert torch.cuda.device_count() == 1
props = torch.cuda.get_device_properties(0)
assert props.gcnArchName.startswith("gfx1250"), props.gcnArchName
print("device:", torch.cuda.get_device_name(0))
print("architecture:", props.gcnArchName)
print("HBM GiB:", props.total_memory / (1 << 30))
print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("transformers:", transformers.__version__)
print("triton:", triton.__version__)
PY
```

Set the source paths used by the benchmark:

```bash
export BENCHMARKS_ROOT=/workspace/tokenspeed-kernel-amd-benchmarks
export TOKENSPEED_ROOT=/workspace/tokenspeed
export PYTHONPATH="$BENCHMARKS_ROOT:$TOKENSPEED_ROOT/python:$TOKENSPEED_ROOT/tokenspeed-kernel/python:$TOKENSPEED_ROOT/tokenspeed-kernel-amd/python"
export TS_SHORT_SHA="$(git -C "$TOKENSPEED_ROOT" rev-parse --short=8 HEAD)"
export RESULT_DIR="$BENCHMARKS_ROOT/toy_e2e/results/gfx1250_${TS_SHORT_SHA}"
mkdir -p "$RESULT_DIR"
```

## 3. Run CPU format tests

```bash
cd "$BENCHMARKS_ROOT"
python3 -m pytest -q -p no:cacheprovider toy_e2e/tests
```

Do not proceed with a corrupt transfer or failed format test.

## 4. Run a short load/execute smoke

```bash
mkdir -p /data/results

python3 "$BENCHMARKS_ROOT/toy_e2e/benchmark_logical_rank.py" \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --prompt-tokens 128 \
  --output-tokens 2 \
  --concurrency 1 \
  --cache-gib 4 \
  --warmup-output-tokens 1 \
  --profile-output-tokens 1 \
  --decode-graph-replays 3 \
  --output /data/results/kimi-k3-gfx1250-rank-local-smoke.json
```

Require `status: passed`, a `gfx1250` architecture, 93 layers, TP8/EP1, and 896
experts. A failure during MoE preprocessing is a portability failure; do not
replace the artifact with gfx1250-preprocessed weights.

## 5. Run the 4K/1K benchmark

```bash
set -o pipefail
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
  --output "$RESULT_DIR/one_gpu_rank_local_4k_1k.json" \
  2>&1 | tee "$RESULT_DIR/one_gpu_rank_local_4k_1k.log"
```

Do not run another GPU workload concurrently. Preserve the complete console
log even when the command fails. The JSON includes per-setting component
totals and the top timed layer/module hotspots for prefill and decode.

## 6. Return the result

Return:

- the complete `gfx1250_<TokenSpeed-short-SHA>/` result directory;
- the smoke JSON and complete 4K/1K console log;
- exact TokenSpeed, kernel, PyTorch, HIP, Transformers, and Triton versions;
- physical GPU name, architecture string, and HBM size;
- rank artifact manifest;
- the target-selected gfx1250 MoE solution;
- any first exception and peak-memory data.

Do not describe the result as eight-GPU throughput. It is one logical TP8/EP1
rank's compute time with local substitutes for rank-spanning collectives.
