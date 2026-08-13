# Kimi-K3 single-MI450 weightless serving test

## Objective

Validate as much of the Kimi-K3 TokenSpeed serving path as possible on one
MI450 (`gfx1250`) without downloading or copying the real checkpoint weights.

This is a runtime and kernel smoke test. It does **not** validate model quality,
real checkpoint loading, or the production TP8/EP8 configuration.

Because TP1 changes per-rank tensor shapes, use production-shape operator tests
alongside this E2E smoke to verify the kernels that a TP8 rank would select.
Treat the TP1 server as integration coverage, not as proof of TP8 dispatch.

## Important constraints

- Use exactly one MI450 GPU.
- Do not download Kimi-K3 weight shards.
- Use `--load-format dummy` to initialize synthetic weights.
- Do not attempt to instantiate full-size Kimi-K3. Dummy loading removes the
  checkpoint dependency, but it does not reduce model memory allocation.
- Keep Kimi-K3's real hidden and attention dimensions where possible so the
  gfx1250 kernels receive representative shapes.
- Run the smallest test first and stop to diagnose each failure before
  increasing coverage.
- Generated tokens are expected to be meaningless.

## Repository context

The Kimi-K3 runtime implementation is located at:

```text
python/tokenspeed/runtime/models/kimi_k3.py
python/tokenspeed/runtime/configs/kimi_k3_config.py
```

Relevant gfx1250 kernels are under:

```text
tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/
```

The existing production Kimi-K3 CI configuration targets eight gfx950 GPUs:

```text
test/ci/eval/kimi-k3-mxfp4-tp8ep8-evalscope-aime26-amd.yaml
```

Current gfx1250 support should not be assumed to match gfx950. In particular,
Kimi-specific SiTU MoE, sigmoid routing, AttnRes, and specialized projection
paths may be missing or may select generic fallbacks.

## 1. Verify the environment

Use the machine's approved MI450 runtime environment. Do not install or rebuild
large dependencies unless required by that environment.

Record:

```bash
pwd
git status --short
python3 --version
python3 -c 'import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0))'
rocminfo | rg 'gfx1250|Name:'
```

Also verify that `ts` resolves to this TokenSpeed checkout:

```bash
which ts
python3 -c 'import tokenspeed; print(tokenspeed.__file__)'
python3 -c 'import tokenspeed_kernel_amd; print(tokenspeed_kernel_amd.__file__)'
```

If the machine is offline, copy only the Kimi-K3 configuration, tokenizer, and
processor metadata to a local model directory. Do not copy `*.safetensors`,
`*.bin`, or `*.pt` files. Replace `moonshotai/Kimi-K3` in the commands below
with that local metadata directory.

## 2. Stage A: configuration-only test

Run the cheap Kimi-K3 configuration and registration tests first:

```bash
cd /path/to/tokenspeed
python3 -m unittest test.runtime.test_kimi_k3_config -v
```

This does not prove gfx1250 execution, but it catches configuration and model
registration problems before starting a server.

## 3. Stage B: four-layer KDA/MLA serving smoke

This variant has three KDA layers and one MLA layer. All four layers use dense
MLPs, which isolates attention and serving from routed-MoE failures.

The KDA layer count must remain divisible by three because of Kimi-K3's cache
layout grouping.

```bash
cd /path/to/tokenspeed

ts serve \
  --model moonshotai/Kimi-K3 \
  --served-model-name kimi-k3-smoke \
  --load-format dummy \
  --trust-remote-code \
  --language-model-only \
  --hf-overrides '{
    "num_hidden_layers": 4,
    "first_k_dense_replace": 4,
    "linear_attn_config": {
      "kda_layers": [1, 2, 3],
      "full_attn_layers": [4],
      "num_heads": 96,
      "head_dim": 128,
      "short_conv_kernel_size": 4,
      "use_full_rank_gate": true,
      "gate_lower_bound": -5.0
    }
  }' \
  --tp 1 \
  --ep-size 1 \
  --attention-backend mla \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --disable-autotune \
  --disable-prefill-graph \
  --disable-kvstore \
  --kvstore-ratio 0.0 \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.8 \
  --sampling-backend greedy \
  --host 127.0.0.1 \
  --port 8000
```

Wait for readiness:

```bash
curl --fail http://127.0.0.1:8000/readiness
```

Exercise prefill and decode:

```bash
curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kimi-k3-smoke",
    "prompt": "Write one short sentence about GPUs.",
    "max_tokens": 16,
    "temperature": 0
  }'
```

Success means the server becomes ready and returns the requested number of
tokens without a crash, hang, NaN-related runtime error, or unsupported-kernel
exception. The text itself is not checked.

## 4. Stage C: reduced routed-MoE smoke

After Stage B succeeds, stop that server and rerun the same command with the
following `--hf-overrides` value:

```json
{
  "num_hidden_layers": 4,
  "first_k_dense_replace": 1,
  "num_experts": 8,
  "num_experts_per_token": 2,
  "num_expert_group": 1,
  "topk_group": 1,
  "linear_attn_config": {
    "kda_layers": [1, 2, 3],
    "full_attn_layers": [4],
    "num_heads": 96,
    "head_dim": 128,
    "short_conv_kernel_size": 4,
    "use_full_rank_gate": true,
    "gate_lower_bound": -5.0
  }
}
```

Keep `--tp 1`, `--ep-size 1`, and `--moe-backend auto`. Add the latter
explicitly if desired:

```text
--moe-backend auto
```

Repeat the readiness and completion requests from Stage B.

If this fails while Stage B succeeds, identify the selected MoE/routing
implementation and determine whether the failure is:

1. an unsupported Kimi SiTU or sigmoid-routing capability on gfx1250;
2. an invalid registry selection;
3. a dummy initialization problem for packed MXFP4 weights or scales;
4. an actual gfx1250 kernel compilation or execution bug.

Do not hide an unsupported path by silently changing Kimi's activation,
quantization, or routing semantics. A fallback may be used only when it
preserves those semantics, and it must be reported.

## 5. Stage D: AttnRes coverage

Four layers do not cover cross-block AttnRes behavior because Kimi-K3 uses a
12-layer residual block. If Stages B and C work and memory permits, use a
16-layer reduced model:

```json
{
  "num_hidden_layers": 16,
  "first_k_dense_replace": 1,
  "num_experts": 8,
  "num_experts_per_token": 2,
  "num_expert_group": 1,
  "topk_group": 1,
  "linear_attn_config": {
    "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15],
    "full_attn_layers": [4, 8, 12, 16],
    "num_heads": 96,
    "head_dim": 128,
    "short_conv_kernel_size": 4,
    "use_full_rank_gate": true,
    "gate_lower_bound": -5.0
  }
}
```

This preserves the normal sequence of three KDA layers followed by one MLA
layer and gives 12 KDA layers, which satisfies the cache grouping constraint.

If memory is insufficient, do not shrink Kimi's hidden size before recording
the result. Reducing hidden dimensions can avoid the exact specialized kernel
shapes this test is intended to cover.

## 6. Stage E: real-weight logical TP8/EP8 rank 0

Use this isolated harness after creating the reduced checkpoint described in
[`checkpoint-preparation.md`](checkpoint-preparation.md). It constructs the production
TP8/EP8 mapping but executes only rank 0 on one GPU:

```bash
python3 toy_e2e/run_kimi_k3_logical_rank.py \
  --checkpoint /data/models/kimi-k3-4layer-tp8ep8-rank0 \
  --phase prefill-decode \
  --prefill-tokens 8 \
  --mla-decode-mode projected \
  --mla-kernel-solution auto
```

The harness preserves TP8-local attention and projection shapes, loads 112
rank-0 experts per MoE layer, masks remote expert selections, uses Triton's
dynamic-MXFP4 SiTU path, and replaces rank-spanning reductions with rank-0
identity reductions. Its output is therefore a structural integration result,
not a numerical parity result.

The command exits nonzero if prefill or decode contains nonfinite values.
`--phase load` checks construction and checkpoint loading without running
kernels. `--mla-decode-mode composed --mla-kernel-solution triton` can isolate
the generic MLA composition, but must not be reported as production dispatch.

## 7. Optional request-load smoke

After a server variant is stable, send a small random workload. Keep it tiny;
this is not a performance benchmark:

```bash
tokenspeed bench serve \
  --backend openai \
  --model kimi-k3-smoke \
  --dataset-name random \
  --num-prompts 8 \
  --request-rate 1
```

If the benchmark CLI requires explicit input/output lengths in this checkout,
inspect `tokenspeed bench serve --help` and use short values such as 128 input
tokens and 32 output tokens.

## 8. Diagnostics to capture

For every stage, record:

- exact command and environment variables;
- commit SHA and working-tree status;
- detected GPU architecture;
- server startup log;
- selected attention, MoE, routing, GEMM, and sampling backends;
- first complete Python exception and any kernel compilation diagnostics;
- GPU memory after model construction and after one request;
- whether readiness, prefill, and decode each succeeded;
- whether execution used a gfx1250 kernel or a generic fallback.

Useful commands:

```bash
git rev-parse HEAD
git status --short
rocm-smi --showmeminfo vram --showuse
```

Do not report random generated text as a correctness result.

## 9. What this test can and cannot establish

It can validate:

- Kimi-K3 model/config construction on gfx1250;
- dummy model loading without checkpoint shards;
- TokenSpeed HTTP startup, scheduling, tokenization, and sampling;
- single-GPU KDA and MLA prefill/decode execution;
- reduced single-GPU MoE execution, if a semantically valid backend exists;
- reduced AttnRes execution with the 16-layer variant.

It cannot validate:

- model output quality or numerical parity with the real checkpoint;
- real Kimi-K3 weight loading and preprocessing;
- the full 93-layer/896-expert memory layout;
- TP8/EP8 communication and expert placement;
- production throughput;
- exact parity with the existing eight-gfx950 CI job.

## 10. Expected handoff report

Return a concise report in this form:

```text
Environment:
- Commit:
- GPU/architecture:
- TokenSpeed package path:
- Kernel package path:

Stage A, config:
- PASS/FAIL:
- Details:

Stage B, KDA/MLA HTTP serving:
- Startup:
- Readiness:
- Prefill:
- Decode:
- Selected kernels/fallbacks:
- Peak memory:

Stage C, reduced MoE:
- PASS/FAIL:
- Selected MoE/routing backend:
- Failure location, if any:

Stage D, AttnRes:
- PASS/FAIL/NOT RUN:
- Details:

Required code changes:
- None, or list each proposed change and why it is gfx1250-specific.
```

Do not make broad refactors. If a failure requires code changes, first isolate
the smallest reproducer and explain the expected dispatch behavior before
editing.

Record the completed run in [`results.md`](results.md), including the
exact commit and working-tree status, commands run, code/test changes, TP8
per-rank shapes, selected kernel names and fallbacks, TP1 E2E results, and
coverage that still requires real TP8/EP8 hardware.
