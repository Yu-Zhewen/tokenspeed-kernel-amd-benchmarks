# Matched GFX1250 and GFX950 attention test plan

## Purpose

This document defines the next measurement set without reporting results.
Every case must use the same logical workload, tensor shapes, dtypes, layouts,
kernel API, warmup count, and repeat count on GFX1250 and GFX950. Architecture
specific kernel implementations are expected, but changing a test input or
dispatch scope on only one architecture invalidates the comparison.

The shared runner is
`tokenspeed-kernel-amd/benchmarks/profile_matched_attention.py`. It uses the
capability-based TokenSpeed APIs and accepts only `gfx950` and `gfx1250`. It is
intended to run unchanged on a real physical GFX1250 system as well as GFX950;
`--expected-arch` prevents accidentally collecting a result on another target.

## Test cases

| ID / runner case | Model phase | Exact logical workload | Dtypes and layout | Timed/profiled scope | Required |
|---|---|---|---|---|---|
| MLA-D1 / `mla-decode` | Kimi-K3 decode | B1, context 4096, q_len 1, 12 heads, absorbed Q width 576 = rank 512 + rope 64 | FP8 absorbed Q, dense FP8 paged KV, page 64; BF16 value projection, raw sigmoid gate, and output | Complete projected-value MLA call. Includes main/reduction and composed projection dispatches when the architecture cannot fuse the epilogue. | Yes |
| MLA-P1 / `mla-prefill` | Kimi-K3 pure prefill | B1, prefix 0, extend 4096, 12 heads, Q/K width 192, V width 128, causal | FP8 Q/K/V, packed varlen sequence | Complete causal MLA prefill call | Yes |
| KDA-P1 / `kda-prefill` | Kimi-K3 pure prefill | B1, prefix 0, extend 4096, 12 heads, K/V width 128 | BF16 Q/K/V/gates, FP32 parameters and recurrent state | Complete KDA prefill pipeline, including preprocess, solve/merge, W/U generation, state scan, and output | Yes |
| DSA-D1 / `dsa-decode-pipeline` | GLM-5.2 complete decode attention pipeline | B1, context 4096, q_len 1, 32 index heads × 128, 8 attention heads, top-k 2048, absorbed width 576 | BF16 index Q, packed FP8 index-K with FP32 scales, page 64; live slots feed FP8 Q and dense FP8 MLA KV selected attention | Complete index-selection plus selected-attention pipeline. Report logits, radix selection, attention main, and reduction dispatches separately inside the call. | Yes |
| DSA-P1 / `dsa-prefill-pipeline-4k` | GLM-5.2 complete pure-prefill attention pipeline | B1, prefix 0, extend 4096, 32 index heads × 128, 8 attention heads, causal top-k up to 2048, absorbed width 576 | BF16 index Q, packed FP8 index-K with FP32 scales, page 64; live slots feed FP8 Q and dense FP8 MLA KV selected attention | Complete causal index-selection plus selected-attention pipeline. Report every component dispatch separately inside the call. | Yes |

DSA is represented at the same level as MLA: one production-path workload per
phase, with all internal/component dispatches reported under that case.
For DSA, the runner composes the separate top-k and selected-attention APIs and
passes the live selected slots between them. No packed sparse attention-KV case
is included because current GLM serving uses dense MLA KV for selected
attention. FP8 DSA prefill attention uses normal capability-based selection;
the current common production solution on both architectures is Triton rather
than forcing a BF16-only Gluon prefill registration.

## Metrics to collect

### Required for every case

- Device name, architecture string, ROCm version, PyTorch version, TokenSpeed
  commit, kernel package build, and command line.
- Complete workload latency from GPU events after warmup.
- Every dispatch name, dispatch count, and dispatch duration or AM cycle count
  inside the workload.
- Grid/workgroup dimensions and static kernel resources: VGPR, SGPR, LDS, and
  scratch usage.
- HBM read bytes, write bytes, and effective read/write bandwidth.
- L2/cache hit information when the profiler exposes a stable counter.
- Mean active waves or occupancy per CU.
- Matrix-pipeline utilization (`MFMA`/`XDL`) with the exact counter definition.
- LDS bank-conflict or LDS-stall metric.
- Average VMEM and LDS dependency latency where available.
- Correctness/status: successful completion, finite outputs, and the selected
  kernel solution or fallback path.

### Reporting rules

- Keep raw counter names and definitions beside derived percentages.
- Report both complete-call latency and per-dispatch data; do not substitute a
  main-kernel duration for end-to-end latency.
- AM model time is simulated time, not physical MI450 latency. Never divide AM
  model time by MI350X physical time to claim a hardware speedup.
- If a counter is unavailable on one architecture, report it as unavailable
  rather than replacing it with a differently defined metric.
- Record failures and stopped runs. Do not silently omit an impractically slow
  case.

## Shared runner usage

Set the same source revision and Python paths on both systems:

```bash
export TOKENSPEED_ROOT=/workspace
export PYTHONPATH="$TOKENSPEED_ROOT/python:$TOKENSPEED_ROOT/tokenspeed-kernel/python:$TOKENSPEED_ROOT/tokenspeed-kernel-amd/python"
python "$TOKENSPEED_ROOT/tokenspeed-kernel-amd/benchmarks/profile_matched_attention.py" \
  --describe
```

Run the default matched set:

```bash
python "$TOKENSPEED_ROOT/tokenspeed-kernel-amd/benchmarks/profile_matched_attention.py" \
  --case all \
  --warmup 2 \
  --repeats 5 \
  --output matched-attention.json
```

On a real GFX1250 system, make the target and timing source explicit:

```bash
python "$TOKENSPEED_ROOT/tokenspeed-kernel-amd/benchmarks/profile_matched_attention.py" \
  --case all \
  --expected-arch gfx1250 \
  --environment physical \
  --warmup 2 \
  --repeats 5 \
  --output matched-attention-gfx1250.json
```

The corresponding GFX950 command is identical except for
`--expected-arch gfx950` and its output filename.

Run one case for dispatch profiling:

```bash
python "$TOKENSPEED_ROOT/tokenspeed-kernel-amd/benchmarks/profile_matched_attention.py" \
  --case mla-decode \
  --warmup 0 \
  --repeats 1
```

The required pure-4K DSA pipeline is long-running. For dispatch profiling, start
with one invocation:

```bash
python "$TOKENSPEED_ROOT/tokenspeed-kernel-amd/benchmarks/profile_matched_attention.py" \
  --case dsa-prefill-pipeline-4k \
  --warmup 0 \
  --repeats 1
```

## Physical GFX950 or GFX1250 profiling

Use the same `rocprofv3` command on both physical architectures. Confirm metric
availability with `rocprofv3 --list-metrics` before choosing the PMC list.
Keep separate output directories per architecture and case.

```bash
rocprofv3 \
  --kernel-trace \
  --pmc FETCH_SIZE WRITE_SIZE MeanOccupancyPerActiveCU MfmaUtil LDSBankConflict \
  --output-format csv \
  --output-directory "profiles/$ARCH/mla-decode" \
  -- \
  python "$TOKENSPEED_ROOT/tokenspeed-kernel-amd/benchmarks/profile_matched_attention.py" \
    --case mla-decode \
    --warmup 0 \
    --repeats 1
```

If a listed PMC is unsupported on one device, remove it from the common pass
and collect architecture-specific supplemental counters in a second pass.

## GFX1250 FFM and AM capture

The same runner can generate the functional dispatch in the dedicated
`zhewenyu-mi450-am` container. Event timing printed under FFM must be discarded.
Use FFM only for functional validation and ROCcap capture, then replay the
captured dispatches under AM. Capture one named case at a time with
`--environment ffm --expected-arch gfx1250 --warmup 0 --repeats 1`, and record
every dispatch needed for the complete API scope. Non-physical runs emit
`latency_us: null` so simulated event timing cannot be mistaken for hardware
latency. Do not compare a single captured dispatch against a multi-dispatch
physical timing.

The current AM container mounts `tokenspeed-kernel-amd` but not necessarily the
core `tokenspeed-kernel` source tree. Before an FFM/ROCcap run, make the matching
core revision importable (for example at `/workspace/tokenspeed-kernel/python`)
without rebuilding Triton.

The pure-4K DSA selected-attention AM replay is required, but may take several
hours. A stopped or unfinished replay must be reported as incomplete rather
than omitted from the matched set.

## Rerun checklist

- Use one TokenSpeed commit and one kernel-package build across both runs.
- Verify the runner reports the expected architecture before collecting data.
- Save `--describe` output with each result set.
- Use identical case arguments, warmup, and repeats.
- Save event-timing JSON and raw profiler/AM files.
- Confirm MLA-D1 reports FP8 Q and FP8 KV on both architectures.
- Confirm DSA attention uses dense KV and not packed sparse attention-KV.
- Treat DSA-P1 as incomplete until its complete pipeline finishes on both
  architectures.
