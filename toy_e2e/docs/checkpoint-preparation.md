# Kimi-K3 partial-checkpoint analysis

## Outcome

A four-layer, language-only checkpoint for logical TP8/EP8 rank 0 is practical:

- Whole Hugging Face shards that must be downloaded: **53.63 GiB**
- Repacked checkpoint tensor payload: **15.19 GiB**
- Full upstream checkpoint: **1,453.66 GiB**
- Required source shards: **5 of 96**
- Required tensors: **2,122 of 497,220**

The initial analysis read only the Hugging Face JSON index and small metadata
headers. The subsequent repack downloaded all five source shards and produced
the validated reduced checkpoint described below.

## Revisions

- TokenSpeed branch: `main`
- TokenSpeed commit: `d34dcf1aec3295019614bd9af53370f9ddaade64`
- Hugging Face repository: `moonshotai/Kimi-K3`
- Hugging Face revision:
  `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`

## Proposed reduced model

- Retain language-model layers 0 through 3.
- Preserve hidden size 7168.
- Preserve the production attention sequence: three KDA layers followed by one
  MLA layer.
- Preserve 96 global KDA heads, producing 12 heads on one logical TP8 rank.
- Preserve head dimension 128.
- Preserve the production first-dense-layer boundary:
  `first_k_dense_replace=1`.
- Preserve 896 global experts and retain experts 0 through 111 owned by EP8
  rank 0.
- Preserve routed latent size 3584 and expert intermediate size 3072.
- Retain embeddings, final norm, output AttnRes projection/norm, and LM head.
- Exclude vision weights and layers 4 through 92.

This model covers layer 0's dense MLP, routed MoE layers 1 through 3, KDA, and
MLA. It does not cross the 12-layer AttnRes boundary.

## Required shards

| Shard | Source size | Required tensors | Repacked payload |
|---|---:|---:|---:|
| `model-00001-of-000096.safetensors` | 2.18 GiB | 23 | 2.18 GiB |
| `model-00002-of-000096.safetensors` | 15.82 GiB | 700 | 3.01 GiB |
| `model-00003-of-000096.safetensors` | 15.82 GiB | 700 | 3.01 GiB |
| `model-00004-of-000096.safetensors` | 15.43 GiB | 694 | 2.62 GiB |
| `model-00094-of-000096.safetensors` | 4.38 GiB | 5 | 4.38 GiB |

The first four layer shards are organized cleanly: layer 0 is in shard 1,
layer 1 in shard 2, layer 2 in shard 3, and layer 3 in shard 4. Language-model
global tensors are in shard 94.

The three MoE source shards contain all 896 experts. Repacking only rank 0's
112 experts reduces those shards from 47.07 GiB to 8.64 GiB of tensor payload.

## Analyzer added

`toy_e2e/scripts/analyze_checkpoint.py`:

- resolves and pins the requested Hugging Face revision;
- reads `config.json` and `model.safetensors.index.json`;
- selects early language layers and one logical EP rank's experts;
- requests only the safetensors header byte ranges;
- refuses to continue if the server ignores HTTP range requests;
- calculates exact source-shard and repacked tensor sizes;
- does not read tensor payload bytes.

Unit coverage is in
`toy_e2e/tests/test_analyze_checkpoint.py`.

Command used:

```bash
source .venv/bin/activate
python3 toy_e2e/scripts/analyze_checkpoint.py \
  --revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  --num-layers 4 \
  --tp-size 8 \
  --tp-rank 0 \
  --ep-size 8 \
  --ep-rank 0
```

Validation:

```text
2 passed
```

## Download and repack result

`toy_e2e/scripts/repack_checkpoint.py` processed one shard at a time:

1. Download a required source shard.
2. Read its safetensors header without materializing tensors.
3. Copy only selected tensor byte ranges into a reduced safetensors shard.
4. Preserve global expert IDs 0 through 111 so the EP8 rank-0 loader can map
   them normally.
5. Delete the source shard before downloading the next one.
6. Build a filtered `model.safetensors.index.json`.
7. Copy only configuration, tokenizer, and processor metadata.
8. Save the upstream configuration as `config.original.json` and write a
   four-layer `config.json`.

Actual result:

- Output: `/data/models/kimi-k3-4layer-tp8ep8-rank0/`
- Final tensor payload: 16,314,581,504 bytes (15.19 GiB)
- Final filesystem usage: 16 GiB
- Repacked tensors: 2,122 in 5 shards
- Source download completed in 14 minutes 45 seconds
- Staging directory was empty after completion

The validator checked every index key against its shard header, contiguous
tensor offsets, exact shard file lengths, aggregate payload size, reduced layer
configuration, and staging cleanup. Unit coverage is in
`toy_e2e/tests/test_repack_checkpoint.py`.

Validation:

```text
4 passed
5 shards, 2,122 tensors, 16,314,581,504 payload bytes
```

## Important loader constraint

Expert tensors can be filtered to EP8 rank 0 because the logical rank does not
own the other experts. Attention, dense, embedding, and projection tensors
remain global in the reduced checkpoint so TokenSpeed's standard TP loader can
slice them for TP8 rank 0.

Further slicing those global tensors to rank-local storage could save more
space, but it would require a pre-sharded checkpoint format or a custom loader.
That is intentionally outside this first prototype.

## What real weights add

The reduced checkpoint can validate:

- real packed MXFP4 weights and scales;
- real weight preprocessing and loader names;
- production TP8/EP8 rank-local tensor geometry;
- model-level kernel registration and dispatch with real parameter layouts;
- rank-local prefill/decode performance.

It still cannot validate:

- RCCL collectives;
- cross-rank reductions;
- expert all-to-all;
- complete checkpoint loading;
- full-model output quality;
- eight-GPU throughput.

## Safety check

After the download and repack:

- the final checkpoint contained only the five filtered shards;
- every downloaded full source shard had been deleted;
- the local Hugging Face cache was not used;
- no GPU was used.
