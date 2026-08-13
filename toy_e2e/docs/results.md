# Kimi-K3 logical TP8/EP8 rank-0 result

## Scope

Revision `d34dcf1aec3295019614bd9af53370f9ddaade64` was tested on one
MI450 (`gfx1250`) with the real four-layer checkpoint at
`/data/models/kimi-k3-4layer-tp8ep8-rank0`.

The test models rank 0 of TP8/EP8. It uses identity TP reductions and executes
only experts 0–111; remote expert selections contribute zero. Numerical parity
with a real eight-rank run is therefore out of scope.

## Result

Overall: **FAIL at MLA decode due to a nonfinite current-token FP8 cache row.**

- Checkpoint load passed in about 2 seconds and allocated 7.03 GiB.
- The model contained three KDA layers followed by one MLA layer.
- Every attention layer used 12 rank-local heads.
- Each of the three MoE layers loaded 112 local experts and selected the Triton
  dynamic-MXFP4 SiTU solution.
- Eight-token prefill passed through all four layers with finite `[8, 7168]`
  output.
- Decode passed all three KDA+MoE layers with finite output, then the MLA
  attention returned 7,168 nonfinite values.
- The process exited and released the GPU after every probe.

The first diagnostic run reported 6.58 s for prefill and 2.45 s for decode,
including compilation and synchronous diagnostic hooks. These are not
steady-state performance measurements.

## Selected kernels

Shape capture confirmed:

- prefill: `gluon_mla_prefill_gfx1250`, 12 query/KV heads, head dimensions
  192/128, FP8 cache;
- decode query prep: `gluon_mla_normalize_project_query_gfx1250`;
- production decode: `gluon_mla_decode_projected_value_gfx1250`, 12 heads,
  page size 64, sequence length 9;
- MoE plan: `triton` for all three local 112-expert layers.

The KDA and MoE operations executed and remained finite, but their exact kernel
names were not emitted by the current shape-capture instrumentation.

## Failure isolation

The logical cache table resolved correctly:

- logical parent page: 1;
- full-attention child pages: 12–23;
- decode kernel pages: `[24, 25]` at kernel page size 64;
- current-token write location: 1544;
- decode sequence length: 9.

Immediately after prefill, all eight live MLA cache rows were finite
(`abs_max=2.25`). The absorbed BF16 query, latent row, output gate, `w_kc`, and
`w_vc` were also finite. The final FP8 query passed to decode was finite
(`abs_max=13.0`).

By the decode attention call, however, the nine-row cache contained nonfinite
data. A torch attention reference over that cache was nonfinite too. Forcing
the portable `triton_mla_decode_with_kvcache` path produced the same result.
This excludes the gfx1250 projected-value reducer as the primary cause and
places the defect in the preceding NoPE FP8 decode cache-write path.

## Reproduction

```bash
python3 toy_e2e/run_kimi_k3_logical_rank.py \
  --checkpoint /data/models/kimi-k3-4layer-tp8ep8-rank0 \
  --phase prefill-decode \
  --prefill-tokens 8 \
  --mla-decode-mode projected \
  --mla-kernel-solution auto
```

The harness exits nonzero when prefill or decode is nonfinite. Load-only
validation is available with `--phase load`.

## Validation

```text
python3 -m pytest -q -p no:cacheprovider \
  toy_e2e/tests/test_logical_rank.py
4 passed
```

No production runtime or kernel source was changed. The next fix should use a
small reproducer for the NoPE BF16-to-FP8 current-token cache write before
changing the attention kernels.
