# GFX1250 rocprofv3 counter collection

## Summary

On the tested GFX1250 system, zero-valued WMMA and LDS-conflict counters were
caused by clock gating rather than the installed rocprofv3 version. Applying
the MGCG override before collection made both the built-in WMMA counters and
the custom TCP conflict events report nonzero values.

Tested software:

- ROCm `7.15.0a20260728`
- rocprofv3 `1.3.5`, revision `44be71b52284948e58c93f65f46910399773fdcd`
- GFX1250

The August 15 nightly (`10.1.0a20260815`, rocprofv3 revision
`af82d0abb401f1e0bccebdb69d5053371507b6b5`) exposed the same built-in WMMA
counters and no built-in LDS conflict counters. PMC collection could not be
validated with that build on this system because it failed with
`aqlprofile API table load failed`.

## Included files

- `extra.yaml`: custom GFX1250 TCP events for LDS bank and cross-port segment
  conflicts.
- `enable_mgcg_override.sh`: writes `0x400` to the eight tested register
  instances.
- `restore_mgcg_override.sh`: restores those registers to `0x0`.
- `run_with_mgcg_override.sh`: enables the override, runs a command, and
  restores the default values on exit.

## Requirements and safety

- Run on the physical GFX1250 host with `umr`, `sudo`, and rocprofv3 available.
- Use exclusive GPU access while changing the registers and collecting PMCs.
- The scripts default to UMR instance `1`. Override it with `UMR_INSTANCE` if
  the target GPU has a different UMR instance.
- Always restore the registers to `0x0` after collection. A reboot is not
  required.
- Counter values are workload-dependent; the values below only establish that
  collection works.

## Recommended usage

Run rocprofv3 through the wrapper so restoration also happens when profiling
fails:

```bash
cd attention/gfx1250_rocprof

./run_with_mgcg_override.sh -- \
  rocprofv3 \
    -E extra.yaml \
    --pmc \
      TX_PERF_SEL_VMW_LDS_BANK_CONFLICT \
      TX_PERF_SEL_VMW_CROSS_PORT_SEGMENT_CONFLICT_LDS_STALLED_CYCLES \
      TX_PERF_SEL_VMW_CROSS_PORT_SEGMENT_CONFLICT_VC_STALLED_CYCLES \
    --output-format csv \
    --output-directory rocprof-conflicts \
    -- python workload.py
```

Collect WMMA activity in a separate pass:

```bash
./run_with_mgcg_override.sh -- \
  rocprofv3 \
    --pmc \
      SQ_INST_CYCLES_VALU_WMMA \
      SQ_INSTS_VEC32_VALU_WMMA \
      SQ_VALU_WMMA_FLOP_FP16 \
      SQ_VALU_WMMA_FLOP_BF16 \
    --output-format csv \
    --output-directory rocprof-wmma \
    -- python workload.py
```

If a site-specific GPU lock is available, place it before the wrapper:

```bash
gpu-lock ./run_with_mgcg_override.sh -- rocprofv3 ... -- python workload.py
```

## Validation result

For the BF16 KDA prefill preprocess kernel used in the smoke test, collection
without the override returned zero for every counter below. With the override:

- `TX_PERF_SEL_VMW_LDS_BANK_CONFLICT`: `3,538,944`
- `TX_PERF_SEL_VMW_CROSS_PORT_SEGMENT_CONFLICT_LDS_STALLED_CYCLES`:
  `5,557,968`
- `TX_PERF_SEL_VMW_CROSS_PORT_SEGMENT_CONFLICT_VC_STALLED_CYCLES`: `259,137`
- `SQ_INST_CYCLES_VALU_WMMA`: `3,932,160`
- `SQ_INSTS_VEC32_VALU_WMMA`: `491,520`
- `SQ_VALU_WMMA_FLOP_FP16`: `0`, expected for a BF16 kernel
- `SQ_VALU_WMMA_FLOP_BF16`: `3,932,160`

This confirms that TCP event `164`,
`TX_PERF_SEL_VMW_CROSS_PORT_SEGMENT_CONFLICT_LDS_STALLED_CYCLES`, is valid and
useful on the tested GFX1250 system.

## Manual recovery

If a collection command is interrupted outside the wrapper, restore the
default register values explicitly:

```bash
./restore_mgcg_override.sh
```

The script reads each register after writing it. All eight final values should
be `0x00000000`.
