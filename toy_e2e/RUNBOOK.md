# Kimi-K3 toy-rank and real-TP8 runbook

This runbook collects the three targets in [`TEST_PLAN.md`](TEST_PLAN.md).
Run performance without a profiler, then collect stage hotspots in a separate
eager run.

## 1. Common inputs

Use these exact revisions for the current matched set:

```bash
export TOKENSPEED_SHA=0b1061eb9fe1df36a4e48e5c9c291cd753af9e89
export MODEL_SHA=eaf5a944bfc8c57438bbce226feef9f6bdbdaae1
export TS_SHORT_SHA="${TOKENSPEED_SHA:0:8}"
export BENCHMARKS_ROOT=/workspace/tokenspeed-kernel-amd-benchmarks
export TOKENSPEED_ROOT=/workspace/tokenspeed
export PYTHONPATH="$BENCHMARKS_ROOT:$TOKENSPEED_ROOT/python:$TOKENSPEED_ROOT/tokenspeed-kernel/python:$TOKENSPEED_ROOT/tokenspeed-kernel-amd/python"
```

Required model inputs:

- full Kimi-K3 checkpoint for real eight-GPU serving;
- portable `kimi-k3-tp8ep1-rank0` checkpoint for either toy target.

The portable checkpoint must have
`format: tokenspeed_raw_rank_state_v1`,
`raw_state_before_kernel_preprocessing: true`, TP8 rank 0, EP1, 93 layers,
and 896 experts. See
[`docs/checkpoint-preparation.md`](docs/checkpoint-preparation.md).

Before any physical run:

```bash
git -C "$TOKENSPEED_ROOT" rev-parse HEAD
git -C "$BENCHMARKS_ROOT" rev-parse HEAD
```

Record the full SHAs, image ID, package versions, host, and UTC time in the
result report.

## 2. GFX950 toy 1-GPU logical rank

### 2.1 Container and device setup

The validated image is
`zhewenyu/kimi-k3-e2e:tokenspeed-0b1061eb`
(local image ID
`sha256:23011448679a86e14d46a7c0fe0493ce853cc57634274b5883611f99a2ea53c6`).
An equivalent supported TokenSpeed image may be used, but its identity and
every version difference must be reported.

Example container:

```bash
docker run -d --name kimi-k3-toy-gfx950 \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host \
  -v /home/zhewenyu/tokenspeed:/workspace/tokenspeed \
  -v /home/zhewenyu/tokenspeed-kernel-amd-benchmarks:/workspace/tokenspeed-kernel-amd-benchmarks \
  -v /data:/data \
  -v /mnt/disk_nvme1n1/data/models:/mnt/disk_nvme1n1/data/models \
  zhewenyu/kimi-k3-e2e:tokenspeed-0b1061eb \
  sleep infinity
```

The second model mount is required when `/data/models` is a symlink into the
NVMe tree.

Inside the container, expose exactly one physical GPU:

```bash
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

python3 - <<'PY'
import torch

assert torch.cuda.device_count() == 1
arch = torch.cuda.get_device_properties(0).gcnArchName
print(torch.cuda.get_device_name(0), arch)
assert arch.startswith("gfx950")
PY
```

### 2.2 Unprofiled performance

```bash
export RESULT_DIR="$BENCHMARKS_ROOT/toy_e2e/results/gfx950_toy_1gpu_${TS_SHORT_SHA}"
mkdir -p "$RESULT_DIR"
set -o pipefail

flock -n /tmp/zhewenyu-kimi-gpu0.lock \
python3 "$BENCHMARKS_ROOT/toy_e2e/benchmark_logical_rank.py" \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --expected-arch gfx950 \
  --tokenspeed-revision "$TOKENSPEED_SHA" \
  --model-revision "$MODEL_SHA" \
  --container-image zhewenyu/kimi-k3-e2e:tokenspeed-0b1061eb \
  --prompt-tokens 4096 \
  --output-tokens 1024 \
  --concurrency 1 16 \
  --chunked-prefill-size 8192 \
  --cache-gib 32 \
  --warmup-waves 1 \
  --measurement-waves 3 \
  --prompt-seed 7 \
  --synthetic-vocabulary-size 160000 \
  --output "$RESULT_DIR/result.json" \
  2>&1 | tee "$RESULT_DIR/run.log"
```

Require `status: passed`, gfx950, one physical rank, logical TP8 rank 0,
93 layers, 896 experts, graph captures for 1/2/4/8/16, C1 and C16 exact graph
buckets, and rolling decode-input contexts 4097–5119 ending at context 5120.
The harness runs a complete 4K/1K warmup wave before three measured
closed-loop waves, matching the real request counts.

### 2.3 Stage-separated kernel hotspots

Keep raw traces outside git:

```bash
export RAW_PROFILE_DIR="/data/results/kimi-k3-toy-1gpu-gfx950-${TS_SHORT_SHA}/eager-profile"
export TOKENSPEED_KERNEL_PROFILE_BACKEND=roctracer

flock -n /tmp/zhewenyu-kimi-gpu0.lock \
python3 "$BENCHMARKS_ROOT/toy_e2e/scripts/profile_logical_rank_stages.py" \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --expected-arch gfx950 \
  --output-dir "$RAW_PROFILE_DIR" \
  --tokenspeed-revision "$TOKENSPEED_SHA" \
  --model-revision "$MODEL_SHA" \
  --container-image zhewenyu/kimi-k3-e2e:tokenspeed-0b1061eb \
  --prompt-tokens 4096 \
  --concurrency 1 16 \
  --chunked-prefill-size 8192 \
  --cache-gib 32 \
  --prompt-seed 7 \
  --synthetic-vocabulary-size 160000 \
  --decode-steps 64

python3 "$BENCHMARKS_ROOT/toy_e2e/scripts/summarize_gpu_hotspots.py" \
  --input "$RAW_PROFILE_DIR" \
  --top-k 15 \
  --csv-dir "$RESULT_DIR/hotspots/csv" \
  --output "$RESULT_DIR/hotspots/hotspots.json"

python3 "$BENCHMARKS_ROOT/toy_e2e/scripts/update_result_readme_hotspots.py" \
  --readme "$RESULT_DIR/README.md" \
  --hotspots "$RESULT_DIR/hotspots/hotspots.json" \
  --csv-dir "$RESULT_DIR/hotspots/csv" \
  --profile-manifest "$RAW_PROFILE_DIR/profile_manifest.json"
```

Require four traces: C1/C16 × `EXTEND`/`DECODE`, each with rank count 1.
The eager profiler uses the same varied prompts as performance collection;
its bare-runner decode input is deterministic token ID 1 because sampling is
outside this attribution-only path.

## 3. GFX950 real 8-GPU TP8/EP1

### 3.1 Graph-mode performance server

Expose all eight MI355X GPUs. On a shared node, take a full-node GPU lock
before launching. A lock file may remain after a run; `flock`, not file
existence, determines whether it is held.

```bash
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export TORCH_NCCL_BLOCKING_WAIT=1

flock -n /tmp/zhewenyu-kimi-fullnode.lock \
ts serve \
  --model /data/models/sunkist \
  --served-model-name kimi-k3 \
  --tp 8 \
  --ep-size 1 \
  --attention-backend mla \
  --moe-backend auto \
  --kv-cache-dtype fp8 \
  --mm-encoder-tp-mode data \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --max-prefill-tokens 8192 \
  --chunked-prefill-size 8192 \
  --max-cudagraph-capture-size 16 \
  --cudagraph-capture-sizes 1 2 4 8 16 \
  --disable-prefill-graph \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  --sampling-backend greedy \
  --disable-kvstore \
  --kvstore-ratio 0 \
  --no-enable-prefix-caching \
  --enable-cache-report \
  --enable-log-request-stats \
  --host 127.0.0.1 \
  --port 21000 \
  --policy round_robin \
  --engine-startup-timeout 3000 \
  --gateway-startup-timeout 600
```

Do not start load generation until `/health` succeeds and all graph sizes
1/2/4/8/16 have been captured.

### 3.2 EvalScope C1 and C16

Use EvalScope 1.9.1:

```bash
export EVALSCOPE_BIN=/path/to/evalscope-1.9.1/bin/evalscope
export RAW_SERVING_DIR="/data/results/kimi-k3-real-tp8ep1-gfx950-${TS_SHORT_SHA}/evalscope"

bash "$BENCHMARKS_ROOT/toy_e2e/scripts/run_evalscope_4k1k.sh" \
  http://127.0.0.1:21000/v1/completions \
  "$RAW_SERVING_DIR" \
  /data/models/sunkist
```

The wrapper runs:

- C1: one warmup and three measured requests;
- C16: sixteen warmups and 48 measured requests;
- closed-loop random exact 4096/1024 requests;
- greedy streaming completion with `ignore_eos=true`.

Normalize the outputs:

```bash
export RESULT_DIR="$BENCHMARKS_ROOT/toy_e2e/results/gfx950_real_8gpu_${TS_SHORT_SHA}"
mkdir -p "$RESULT_DIR"

python3 "$BENCHMARKS_ROOT/toy_e2e/scripts/collect_real_serving_results.py" \
  --input "$RAW_SERVING_DIR" \
  --output "$RESULT_DIR/result.json" \
  --tokenspeed-revision "$TOKENSPEED_SHA" \
  --model-revision "$MODEL_SHA"
```

Require 3/3 successful C1 requests and 48/48 successful C16 requests with
exact token counts.

### 3.3 Eager hotspot server

Stop the graph-mode server. Launch an otherwise identical server with
`--enforce-eager`, without changing workload, cache, topology, or kernels:

```bash
export TOKENSPEED_KERNEL_PROFILE_BACKEND=roctracer
export TOKENSPEED_KERNEL_PROFILE_DATA=trace
export TOKENSPEED_KERNEL_PROFILE_OUTPUT_FORMAT=chrome_trace

flock -n /tmp/zhewenyu-kimi-fullnode.lock \
ts serve \
  --model /data/models/sunkist \
  --served-model-name kimi-k3 \
  --tp 8 \
  --ep-size 1 \
  --attention-backend mla \
  --moe-backend auto \
  --kv-cache-dtype fp8 \
  --mm-encoder-tp-mode data \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --max-prefill-tokens 8192 \
  --chunked-prefill-size 8192 \
  --enforce-eager \
  --disable-prefill-graph \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  --sampling-backend greedy \
  --disable-kvstore \
  --kvstore-ratio 0 \
  --no-enable-prefix-caching \
  --enable-cache-report \
  --enable-log-request-stats \
  --host 127.0.0.1 \
  --port 21000 \
  --policy round_robin \
  --engine-startup-timeout 3000 \
  --gateway-startup-timeout 600
```

### 3.4 Four all-rank stage profiles

```bash
export RAW_PROFILE_DIR="/data/results/kimi-k3-real-tp8ep1-gfx950-${TS_SHORT_SHA}/eager-profile"

python3 "$BENCHMARKS_ROOT/toy_e2e/scripts/profile_serving_stages.py" \
  --output-dir "$RAW_PROFILE_DIR/c1/prefill" \
  --profile-id c1-prefill \
  --concurrency 1 \
  --capture prefill

python3 "$BENCHMARKS_ROOT/toy_e2e/scripts/profile_serving_stages.py" \
  --output-dir "$RAW_PROFILE_DIR/c1/decode" \
  --profile-id c1-decode \
  --concurrency 1 \
  --capture decode \
  --profile-steps 64

python3 "$BENCHMARKS_ROOT/toy_e2e/scripts/profile_serving_stages.py" \
  --output-dir "$RAW_PROFILE_DIR/c16/prefill" \
  --profile-id c16-prefill \
  --concurrency 16 \
  --capture prefill

python3 "$BENCHMARKS_ROOT/toy_e2e/scripts/profile_serving_stages.py" \
  --output-dir "$RAW_PROFILE_DIR/c16/decode" \
  --profile-id c16-decode \
  --concurrency 16 \
  --capture decode \
  --profile-steps 64

python3 "$BENCHMARKS_ROOT/toy_e2e/scripts/summarize_gpu_hotspots.py" \
  --input "$RAW_PROFILE_DIR" \
  --top-k 15 \
  --csv-dir "$RESULT_DIR/hotspots/csv" \
  --output "$RESULT_DIR/hotspots/hotspots.json"
```

Require 32 retained traces: eight physical ranks for each of four profiles.
Decode transition-only `EXTEND` traces are discarded by the capture script.

## 4. GFX1250 toy 1-GPU logical rank

This target requires one physical gfx1250 GPU. FFM and AM are valid for
functional development but must not be reported as physical performance. The
reference result is under
[`results/gfx1250_toy_1gpu_0b1061eb/`](results/gfx1250_toy_1gpu_0b1061eb/).

### 4.1 Transfer and validate

Copy the benchmark repository and complete
`kimi-k3-tp8ep1-rank0` directory to the target. Verify checksums and the
manifest before loading.

Use the target's supported TokenSpeed package or container and the project's
prebuilt TokenSpeed Triton wheel. Do not rebuild Triton solely for this run.

```bash
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

python3 - <<'PY'
import torch

assert torch.cuda.device_count() == 1
props = torch.cuda.get_device_properties(0)
print(torch.cuda.get_device_name(0), props.gcnArchName)
assert props.gcnArchName.startswith("gfx1250")
PY
```

### 4.2 Collect performance and hotspots

Set the same common variables from section 1, then use a revision-specific
directory:

```bash
export RESULT_DIR="$BENCHMARKS_ROOT/toy_e2e/results/gfx1250_toy_1gpu_${TS_SHORT_SHA}"
export RAW_PROFILE_DIR="/data/results/kimi-k3-toy-1gpu-gfx1250-${TS_SHORT_SHA}-rolling/eager-profile"
mkdir -p "$RESULT_DIR"
set -o pipefail

python3 "$BENCHMARKS_ROOT/toy_e2e/benchmark_logical_rank.py" \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --expected-arch gfx1250 \
  --tokenspeed-revision "$TOKENSPEED_SHA" \
  --model-revision "$MODEL_SHA" \
  --container-image "<physical-gfx1250-image-and-ID>" \
  --prompt-tokens 4096 \
  --output-tokens 1024 \
  --concurrency 1 16 \
  --chunked-prefill-size 8192 \
  --cache-gib 32 \
  --warmup-waves 1 \
  --measurement-waves 3 \
  --prompt-seed 7 \
  --synthetic-vocabulary-size 160000 \
  --output "$RESULT_DIR/result.json" \
  2>&1 | tee "$RESULT_DIR/run.log"

export TOKENSPEED_KERNEL_PROFILE_BACKEND=roctracer
python3 "$BENCHMARKS_ROOT/toy_e2e/scripts/profile_logical_rank_stages.py" \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --expected-arch gfx1250 \
  --output-dir "$RAW_PROFILE_DIR" \
  --tokenspeed-revision "$TOKENSPEED_SHA" \
  --model-revision "$MODEL_SHA" \
  --container-image "<physical-gfx1250-image-and-ID>" \
  --prompt-tokens 4096 \
  --concurrency 1 16 \
  --chunked-prefill-size 8192 \
  --cache-gib 32 \
  --prompt-seed 7 \
  --synthetic-vocabulary-size 160000 \
  --decode-steps 64

python3 "$BENCHMARKS_ROOT/toy_e2e/scripts/summarize_gpu_hotspots.py" \
  --input "$RAW_PROFILE_DIR" \
  --top-k 15 \
  --csv-dir "$RESULT_DIR/hotspots/csv" \
  --output "$RESULT_DIR/hotspots/hotspots.json"

python3 "$BENCHMARKS_ROOT/toy_e2e/scripts/update_result_readme_hotspots.py" \
  --readme "$RESULT_DIR/README.md" \
  --hotspots "$RESULT_DIR/hotspots/hotspots.json" \
  --csv-dir "$RESULT_DIR/hotspots/csv" \
  --profile-manifest "$RAW_PROFILE_DIR/profile_manifest.json"
```

Copy [`RESULT_TEMPLATE.md`](RESULT_TEMPLATE.md) into the completed result
directory as `README.md`, fill every field, and update
[`results/README.md`](results/README.md) to reference it.

## 5. Final report validation

For each target:

```bash
python3 -m pytest -q -p no:cacheprovider "$BENCHMARKS_ROOT/toy_e2e/tests"
python3 -m ruff check "$BENCHMARKS_ROOT/toy_e2e"
```

Also verify:

- performance JSON and logs are from an unprofiled run;
- hotspot JSON contains four profiles with the expected rank count;
- exact-name CSV rows agree with hotspot JSON;
- README headings and table columns still match `RESULT_TEMPLATE.md`;
- unavailable or failed work is explicit;
- raw trace paths and container/software identities are recorded.
