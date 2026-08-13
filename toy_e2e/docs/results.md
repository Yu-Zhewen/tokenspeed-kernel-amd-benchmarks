# Kimi-K3 logical TP8/EP8 rank-0 result

## Scope

Revision `d34dcf1aec3295019614bd9af53370f9ddaade64` was tested on one
MI450 (`gfx1250`) with the real four-layer checkpoint at
`/data/models/kimi-k3-4layer-tp8ep8-rank0`.

The test models rank 0 of TP8/EP8. It uses identity TP reductions and executes
only experts 0–111; remote expert selections contribute zero. Numerical parity
with a real eight-rank run is therefore out of scope.

## Result

Overall: **PASS for structural logical-rank prefill and decode.**

- Checkpoint load passed in about 2 seconds and allocated 7.11 GiB.
- The model contained three KDA layers followed by one MLA layer.
- Every attention layer used 12 rank-local heads.
- Each of the three MoE layers loaded 112 local experts and selected the Triton
  dynamic-MXFP4 SiTU solution.
- Eight-token prefill passed through all four layers with finite `[8, 7168]`
  output.
- One-token decode passed through all four layers with finite `[1, 7168]`
  output.
- The production projected-value MLA kernel and a torch attention reference
  both produced finite output.
- The process exited and released the GPU after every probe.

The corrected diagnostic run reported 6.57 s for prefill and 2.42 s for decode,
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

All KDA, MLA, and MoE layer outputs remained finite. The exact KDA and MoE
kernel names were not emitted by the current shape-capture instrumentation.

## Cache allocation correction

The harness now constructs a real `tokenspeed_scheduler.Scheduler` from the
runtime cache contract and consumes its prefill/decode `ForwardOp` block tables.
The scheduler allocated separate LCM parents:

- full attention: child page `37` (LCM parent 4);
- KDA groups: pages `1`, `2`, and `3` (LCM parents 1–3);
- decode kernel pages: `[74, 75]` at kernel page size 64;
- current-token write location: `4744`;
- decode sequence length: 9.

All nine MLA cache rows were finite at decode (`abs_max=2.25`). The absorbed
query, latent row, output gate, `w_kc`, and `w_vc` were finite. Attention scores
were finite (`abs_max=13.76`), as were the projected gfx1250 kernel output and
the torch reference.

The earlier harness assigned all cache groups to LCM parent 1. KDA state and
MLA history fields intentionally alias storage within one parent, so every KDA
decode layer overwrote the MLA cache before MLA ran. Instrumentation showed the
cache was finite before KDA layer 0, changed immediately after that layer, and
was already nonfinite before the MLA QKV/cache-write path. The MLA write did
not change the corruption pattern.

Therefore, the previous reported MLA cache-write defect was a harness false
positive. This test found no MLA runtime or kernel defect.

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
6 passed
```

No production runtime or kernel source was changed.
