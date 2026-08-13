# Kimi-K3 logical TP8/EP8 rank-0 toy E2E

This package runs production-shaped Kimi-K3 rank-0 work on one MI450
(`gfx1250`). It loads real weights from a reduced four-layer checkpoint,
constructs the production TP8/EP8 mapping, executes three KDA layers followed
by one MLA layer, and records model and kernel diagnostics.

This is a structural integration test, not an eight-rank numerical-parity test.
TP reductions are local identities, only EP rank 0 experts (0–111) execute, and
remote expert contributions are zero.

## Tested revisions

- TokenSpeed: `d34dcf1aec3295019614bd9af53370f9ddaade64`
- Kimi-K3 checkpoint:
  `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`
- Model repository: `moonshotai/Kimi-K3`
- GPU: one MI450 (`gfx1250`)
- Tested container image: `tokenspeed-kimi-smoke:local`

The container name is local to the original machine. Another environment is
valid if it supplies the ROCm, PyTorch, Transformers, TokenSpeed, and
TokenSpeed-kernel dependencies required by the pinned TokenSpeed revision.

## Package layout

- `run_kimi_k3_logical_rank.py`: real-weight logical-rank harness.
- `scripts/analyze_checkpoint.py`: metadata-only checkpoint size analysis.
- `scripts/repack_checkpoint.py`: resumable, one-shard-at-a-time downloader and
  reduced-checkpoint builder.
- `tests/`: CPU tests for checkpoint selection/repacking and logical adapters.
- `docs/results.md`: observed prefill/decode result and cache-harness correction.
- `docs/checkpoint-preparation.md`: exact checkpoint sizing and preparation
  record.
- `docs/smoke-plan.md`: broader staged Kimi-K3 smoke-test plan.

## 1. Check out the sources

```bash
git clone https://github.com/Yu-Zhewen/tokenspeed-kernel-amd-benchmarks.git
git clone https://github.com/lightseekorg/tokenspeed.git
git -C tokenspeed checkout d34dcf1aec3295019614bd9af53370f9ddaade64

export BENCHMARKS_ROOT="$PWD/tokenspeed-kernel-amd-benchmarks"
export TOKENSPEED_ROOT="$PWD/tokenspeed"
```

Use a Python virtual environment for local tooling. The checkpoint scripts use
only the Python standard library. `pytest` is needed for tests and the logical
adapter tests additionally require PyTorch.

## 2. Validate the package

Run from the benchmark repository root:

```bash
python3 -m pytest -q -p no:cacheprovider \
  toy_e2e/tests/test_analyze_checkpoint.py \
  toy_e2e/tests/test_repack_checkpoint.py

python3 -m pytest -q -p no:cacheprovider toy_e2e/tests
```

The first command is CPU-only and does not require TokenSpeed or a GPU. The
second command also imports PyTorch, but does not allocate a GPU.

## 3. Estimate the reduced checkpoint

The analyzer reads Hugging Face JSON files and safetensors headers only. It
refuses a server response that ignores byte-range requests.

```bash
cd "$BENCHMARKS_ROOT"
python3 toy_e2e/scripts/analyze_checkpoint.py \
  --revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  --num-layers 4 \
  --tp-size 8 \
  --tp-rank 0 \
  --ep-size 8 \
  --ep-rank 0
```

Expected estimate:

- five of 96 source shards;
- 53.63 GiB downloaded sequentially;
- 2,122 selected tensors;
- 15.19 GiB final tensor payload.

Set `HF_TOKEN` if Hugging Face requires authentication.

## 4. Build the reduced checkpoint

Use separate staging and output directories. The output directory must be
empty. Allow at least one source shard (up to about 16 GiB) plus 16 GiB for the
growing output. The script resumes partial downloads, repacks each shard
without loading tensors, and deletes that full source shard before continuing.

```bash
cd "$BENCHMARKS_ROOT"
python3 toy_e2e/scripts/repack_checkpoint.py \
  --revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  --num-layers 4 \
  --tp-size 8 \
  --tp-rank 0 \
  --ep-size 8 \
  --ep-rank 0 \
  --staging-dir /data/staging/kimi-k3-rank0 \
  --output-dir /data/models/kimi-k3-4layer-tp8ep8-rank0
```

The output keeps global TP tensors because TokenSpeed’s standard loader slices
them at load time. It filters expert tensors to EP8 rank 0 and rewrites
`config.json` to retain layers 0–3. Do not pre-slice the global TP tensors
unless using a pre-sharded format or custom loader.

## 5. Run on one MI450

First run load-only validation. Adapt the image and device-lock command to the
machine, but keep only one GPU visible.

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --ipc=host \
  -v "$TOKENSPEED_ROOT:/workspace" \
  -v "$BENCHMARKS_ROOT:/benchmarks:ro" \
  -v /data/models:/data/models:ro \
  -w /workspace \
  -e HIP_VISIBLE_DEVICES=0 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONPATH=/workspace/python:/workspace/tokenspeed-kernel/python:/workspace/tokenspeed-kernel-amd/python \
  tokenspeed-kimi-smoke:local \
  python3 /benchmarks/toy_e2e/run_kimi_k3_logical_rank.py \
    --checkpoint /data/models/kimi-k3-4layer-tp8ep8-rank0 \
    --phase load
```

Then execute eight-token prefill and one-token decode:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --ipc=host \
  -v "$TOKENSPEED_ROOT:/workspace" \
  -v "$BENCHMARKS_ROOT:/benchmarks:ro" \
  -v /data/models:/data/models:ro \
  -w /workspace \
  -e HIP_VISIBLE_DEVICES=0 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONPATH=/workspace/python:/workspace/tokenspeed-kernel/python:/workspace/tokenspeed-kernel-amd/python \
  tokenspeed-kimi-smoke:local \
  python3 /benchmarks/toy_e2e/run_kimi_k3_logical_rank.py \
    --checkpoint /data/models/kimi-k3-4layer-tp8ep8-rank0 \
    --phase prefill-decode \
    --prefill-tokens 8 \
    --mla-decode-mode projected \
    --mla-kernel-solution auto
```

At the tested revisions, the command exits with status 0: prefill, projected
gfx1250 MLA decode, the FP8 cache, and every layer output are finite. Use
`--mla-decode-mode composed --mla-kernel-solution triton` only to isolate the
generic MLA composition; it is not production dispatch.

## MoE finding

The gfx950 native Gluon path supports BF16 activations with MXFP4 weights
(A16W4) and SiTU. At the tested revision, gfx1250 has no SiTU-capable A16W4
implementation in either Gluon or Triton. The harness changes only its local
construction arguments to request dynamic MXFP4 activations, selecting the
Triton A4W4 SiTU path. It did not change the production runtime and did not use
an A16W16 fallback.

All three local 112-expert MoE layers executed with finite outputs during
prefill and decode.

## MLA and cache-allocation finding

The harness uses the real TokenSpeed scheduler to allocate one synthetic
request and consumes its prefill/decode `ForwardOp` cache tables. The scheduler
assigned separate LCM parents to the three KDA state groups and the MLA history
group. Eight-token prefill, the nine-row FP8 decode cache, the decode query,
absorbed weights, attention scores, the gfx1250 projected-value output, and a
torch reference were all finite.

An earlier harness revision manually assigned every group to LCM parent 1.
Because KDA and MLA fields intentionally alias within one parent, KDA decode
overwrote the MLA cache before MLA ran and created a false-positive kernel
failure. The corrected result provides no evidence of an MLA runtime or kernel
bug.

## Coverage limits

This package validates real weight names/layouts, rank-local production shapes,
model dispatch, KDA/MLA/MoE layer connectivity, and finite local intermediates.
It does not validate RCCL, all-to-all, cross-rank sums, output quality, the full
93-layer model, or eight-GPU throughput.
