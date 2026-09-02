# Portable TP8/EP1 rank checkpoint

## Contract

`tokenspeed_raw_rank_state_v1` is a rank-local checkpoint for one logical
Kimi-K3 TP8/EP1 rank:

- TP size 8, TP rank 0;
- EP size 1, EP rank 0;
- all 93 language layers;
- all 896 expert IDs in every MoE layer;
- each expert and other TP-aware weights sliced for TP rank 0;
- tensors saved before target-kernel weight preprocessing.

The final property is required for architecture portability. Kimi-K3's MXFP4
expert weights are transformed into architecture-specific kernel layouts
during `process_weights_after_loading`. A checkpoint saved after that transform
must not be moved between gfx950 and gfx1250.

The exporter instead captures the initialized model immediately after
TokenSpeed's normal `load_weights` step. `RawRankStateLoader` reconstructs that
raw state on the target, checks every tensor, rebuilds KDA convolution, MLA
absorption, and AttnRes derived tensors, and then runs the target installation's
normal kernel preprocessing. No TokenSpeed core patch is required.

## Export

Run inside the same TokenSpeed environment used for model loading, with one
physical GPU visible:

```bash
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

python3 toy_e2e/scripts/export_rank_local_checkpoint.py \
  --source /data/models/Kimi-K3 \
  --output /data/models/kimi-k3-tp8ep1-rank0 \
  --part-gib 2 \
  --tokenspeed-revision \
    0b1061eb9fe1df36a4e48e5c9c291cd753af9e89 \
  --source-revision \
    eaf5a944bfc8c57438bbce226feef9f6bdbdaae1
```

The source must contain the complete Kimi-K3 checkpoint and model metadata.
The output must be absent or empty. The exporter copies only runtime metadata,
not the full source index or source shards.

The validated MI355X export completed with:

- 114 bounded safetensors parts;
- 190.98 GiB tensor payload;
- 191.28 GiB raw model allocation;
- 93 model layers;
- 896 local experts per MoE layer;
- MoE TP8 and EP1.

The 190.98 GiB result is larger than the simple
`full-checkpoint-size / 8` estimate because not every state tensor is TP
sharded and the source filesystem size includes different framing overheads.

## Manifest and atomic completion

Each part is first written with a `.tmp` suffix and atomically renamed.
`rank-local-manifest.json` is written only after every tensor part succeeds and
contains:

- format and completion markers;
- topology and source architecture metadata;
- TokenSpeed and source revisions when supplied;
- tensor count and total tensor bytes;
- every part's filename, tensor count, and payload bytes;
- every tensor's name, shape, dtype, bytes, and part number.

Treat the directory as usable only when all of these are true:

```text
format == "tokenspeed_raw_rank_state_v1"
complete == true
tp_size == 8
tp_rank == 0
ep_size == 1
ep_rank == 0
raw_state_before_kernel_preprocessing == true
```

An interrupted export has no complete manifest. Remove it or rerun with
`--overwrite`; do not manually promote `.tmp` files or construct a manifest.

## Transfer validation

Keep all model metadata, all 114 parts, and the manifest together. For an
untrusted or resumable transfer, create external checksums after export:

```bash
cd /data/models/kimi-k3-tp8ep1-rank0
sha256sum model-rank-0-part-*.safetensors rank-local-manifest.json \
  > SHA256SUMS
```

After copying:

```bash
cd /data/models/kimi-k3-tp8ep1-rank0
sha256sum -c SHA256SUMS
```

Hashing reads the full 191 GiB artifact and is optional when the storage and
transport already provide end-to-end checksums.

## Load validation

The benchmark's `raw-rank-state` mode performs strict checks while loading:

- manifest format, completeness, and TP8/EP1 rank-0 topology;
- checkpoint names against the target model state;
- duplicate tensors;
- exact shape and dtype;
- loaded count against the manifest;
- missing model tensors.

It then applies architecture-specific preprocessing and builds the full model.
A quick physical-GPU validation is:

```bash
EXPECTED_ARCH=gfx950  # use gfx1250 on MI450
python3 toy_e2e/benchmark_logical_rank.py \
  --checkpoint /data/models/kimi-k3-tp8ep1-rank0 \
  --load-format raw-rank-state \
  --expected-arch "$EXPECTED_ARCH" \
  --tokenspeed-revision \
    0b1061eb9fe1df36a4e48e5c9c291cd753af9e89 \
  --model-revision \
    eaf5a944bfc8c57438bbce226feef9f6bdbdaae1 \
  --prompt-tokens 128 \
  --output-tokens 2 \
  --concurrency 1 \
  --cache-gib 4 \
  --warmup-waves 1 \
  --measurement-waves 1 \
  --prompt-seed 7 \
  --synthetic-vocabulary-size 160000 \
  --output /data/results/kimi-k3-rank-local-smoke.json
```

Check that the output reports:

```text
status: passed
num_layers: 93
attn_tp_size: 8
moe_tp_size: 8
moe_ep_size: 1
local_experts: [896]
```

On gfx950, `architecture` must start with `gfx950`. On gfx1250 it must start
with `gfx1250`; this proves that preprocessing occurred under the target
architecture rather than on the export machine.

## Compatibility boundaries

The format is portable across gfx950 and gfx1250, not across arbitrary model
or runtime changes. Keep the artifact and benchmark on compatible Kimi-K3
model definitions, quantization code, and TokenSpeed revisions. Regenerate the
artifact if a model-state name, raw tensor shape, raw dtype, TP/EP mapping, or
preprocessing contract changes.

The artifact contains only rank 0. It is sufficient for this one-GPU logical
benchmark and is not a complete eight-rank serving checkpoint.
