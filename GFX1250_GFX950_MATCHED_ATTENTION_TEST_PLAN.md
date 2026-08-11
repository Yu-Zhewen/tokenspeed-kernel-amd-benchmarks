# Matched GFX1250 and GFX950 attention test plan

## Goal

Produce a matched GFX950-versus-GFX1250 data set for production-shaped Kimi-K3
and GLM-5.2 attention. This document defines workloads and required outputs; it
does not contain benchmark results.

A comparison is complete only when:

- all five cases below run on both architectures;
- both runs use the same TokenSpeed revision, kernel package, inputs, warmups,
  repeats, dtypes, layouts, and public API scope;
- unprofiled physical latency and per-dispatch profiler data are both saved;
- every unavailable metric, failure, or unfinished AM replay is explicitly
  recorded.

Use `profile_matched_attention.py` from this gist. It calls capability-based
TokenSpeed APIs and runs only on `gfx950` or `gfx1250`. Architecture-specific
kernel selection is expected. Changing a workload on only one architecture
invalidates the comparison.

## Required test set

| ID / `--case` | Workload | Shape | Dtypes and layout | Measurement scope |
|---|---|---|---|---|
| MLA-D1 / `mla-decode` | Kimi-K3 decode | B1, context 4096, q_len 1, 12 heads, absorbed width 576 = rank 512 + rope 64 | FP8 absorbed Q, dense FP8 paged KV, page 64; BF16 value projection, raw sigmoid gate, and output | Complete projected-value MLA workload, including all main, reduction, and epilogue dispatches |
| MLA-P1 / `mla-prefill` | Kimi-K3 pure prefill | B1, prefix 0, extend 4096, 12 heads, Q/K width 192, V width 128, causal | FP8 Q/K/V, packed variable-length sequence | Complete causal MLA prefill workload |
| KDA-P1 / `kda-prefill` | Kimi-K3 KDA pure prefill | B1, prefix 0, extend 4096, 12 heads, K/V width 128 | BF16 Q/K/V/gates, FP32 parameters and recurrent state | Complete pipeline: preprocess, solve/merge, W/U generation, state scan, and output |
| DSA-D1 / `dsa-decode-pipeline` | GLM-5.2 decode: top-k then selected attention | B1, context 4096, q_len 1, 32 index heads × 128, 8 attention heads, top-k 2048, absorbed width 576 | BF16 index Q, packed FP8 index-K with FP32 scales, page 64; live slots feed FP8 Q and dense FP8 MLA KV | Complete pipeline; report logits, radix selection, attention main, and reduction dispatches individually |
| DSA-P1 / `dsa-prefill-pipeline-4k` | GLM-5.2 pure prefill: causal top-k then selected attention | B1, prefix 0, extend 4096, 32 index heads × 128, 8 attention heads, causal top-k up to 2048, absorbed width 576 | BF16 index Q, packed FP8 index-K with FP32 scales, page 64; live slots feed FP8 Q and dense FP8 MLA KV | Complete pipeline and every component dispatch; this case is long-running but required |

## Collection model

Each table row is one official workload. Measure its complete latency in an
unprofiled runner invocation. Then run the same case under the profiler and
report every GPU dispatch separately; do not replace the pipeline case with
synthetic component-only workloads.

For DSA, the runner composes the top-k and selected-attention APIs and passes
the live selected slots between them. The profiler still reports the logits,
radix-selection, attention-main, and reduction dispatches individually. For
MLA, it similarly reports all dispatches emitted by the single MLA API call.

No packed sparse attention-KV case is included because current GLM serving uses
dense MLA KV for selected attention. FP8 DSA prefill uses normal
capability-based solution selection on both architectures.

## Required outputs

Every field below is required in the final report. If the installed profiler
does not expose a counter on an architecture, write `unavailable`; do not omit
the field or substitute a counter with a different definition.

### Per run

- Device name, architecture string, environment (`physical`, `ffm`, or `am`),
  ROCm version, PyTorch version, TokenSpeed commit, kernel package build, and
  exact command line.
- Warmup count, repeat count, and complete unprofiled workload latency in
  microseconds from GPU events. Physical runs only.
- Correctness/status: successful completion, finite outputs, and selected
  solution or fallback path.

### Per GPU dispatch

- Dispatch name, sequence order, and count; physical duration for hardware
  runs and AM cycle count for AM replays.
- Grid/workgroup dimensions and static kernel resources: VGPR, SGPR, LDS, and
  scratch usage.
- HBM read bytes, write bytes, and effective read/write bandwidth.
- L2/cache hit information when the profiler exposes a stable counter.
- Mean active waves or occupancy per CU.
- Matrix-pipeline utilization (`MFMA`/`XDL`) with the exact counter definition.
- LDS bank-conflict or LDS-stall metric.
- Average VMEM and LDS dependency latency where available.

### Reporting rules

- Collect latency without profiler instrumentation. Collect dispatch counters
  in separate reruns of the identical case.
- If all counters cannot be collected together, use multiple profiler passes
  with the same inputs and identify dispatches by name and order.
- Keep raw counter names and definitions beside derived percentages.
- Report both complete-workload latency and per-dispatch data; do not
  substitute a main-kernel duration for end-to-end latency.
- AM model time is simulated time, not physical MI450 latency. Never divide AM
  model time by MI350X physical time to claim a hardware speedup.
- Record failures and stopped runs. Do not silently omit an impractically slow
  case.

## Execution instructions

### 1. Prepare identical software

Use the same TokenSpeed commit and kernel-package build on both systems. Set
these paths for each checkout:

```bash
export TOKENSPEED_ROOT=/path/to/tokenspeed
export RUNNER=/path/to/profile_matched_attention.py
export PYTHONPATH="$TOKENSPEED_ROOT/python:$TOKENSPEED_ROOT/tokenspeed-kernel/python:$TOKENSPEED_ROOT/tokenspeed-kernel-amd/python"

python "$RUNNER" --describe > matched-attention-cases.json
```

Save the describe output. Before collecting data, confirm that `torch`,
`tokenspeed_kernel`, and `tokenspeed_kernel.ops.attention` import from the
intended environment.

### 2. Collect physical workload latency

Run the complete required set without a profiler on each physical system:

```bash
ARCH=gfx950  # use gfx1250 on the other system
python "$RUNNER" \
  --case all \
  --expected-arch "$ARCH" \
  --environment physical \
  --warmup 2 \
  --repeats 5 \
  --output "matched-attention-$ARCH.json"
```

`--case all` includes DSA-P1. Do not remove it because it is slow. The runner
reports average GPU-event latency in microseconds.

### 3. Collect physical dispatch traces and counters

Profile one named case at a time. First discover counters and verify that a
counter group can be collected together:

```bash
rocprofv3-avail list --pmc
rocprofv3-avail pmc-check COUNTER_1 COUNTER_2
```

Use a common supported counter group on both architectures. Replace the
placeholder names below with that group:

```bash
ARCH=gfx950
CASE=mla-decode
COMMON_PMCS=(COUNTER_1 COUNTER_2)

rocprofv3 \
  --kernel-trace \
  --pmc "${COMMON_PMCS[@]}" \
  --output-format csv \
  --output-directory "profiles/$ARCH/$CASE" \
  -- \
  python "$RUNNER" \
    --case "$CASE" \
    --expected-arch "$ARCH" \
    --environment physical \
    --warmup 0 \
    --repeats 1
```

Current ROCm uses `rocprofv3-avail`; the older
`rocprofv3 --list-metrics` form is obsolete. If counters cannot share one
hardware pass, use multiple `--pmc` groups or separate profiler runs with
identical inputs.

The runner performs one untimed setup invocation before its timed repeats.
Therefore, the profiler may show the workload dispatch sequence more than
once. Preserve sequence order, label the setup and measured sequences, and
never sum duplicate invocations as one pipeline.

### 4. Capture GFX1250 dispatches for FFM and AM

On this host, use only the dedicated `zhewenyu-mi450-am` container:

```bash
docker start zhewenyu-mi450-am
docker exec -it zhewenyu-mi450-am bash
source /opt/rocdtif/ffmlite_env.sh
```

Make the runner and matching TokenSpeed Python packages importable in the
container. Do not rebuild `/root/triton`. Use FFM to identify every dispatch in
one named case, then capture each dispatch separately by its ROCcap selector:

```bash
export RUNNER=/workspace/tokenspeed-kernel-amd/benchmarks/profile_matched_attention.py
export ROCCAP=/opt/rocdtif/tools/roccap/bin/roccap
export CASE=mla-decode
export DISPATCH='<kernel-name>/0'
export DISPATCH_TAG='<short-unique-name>'
export RUN_DIR="/tmp/am-run/$CASE-$DISPATCH_TAG"

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
source /opt/rocdtif/ffmlite_env.sh
unset HSA_TOOLS_LIB
export HSA_ENABLE_DTIF_FAST_COPY=0
export DtifFbBaseLocation=0x200000000

# Run once without capture to validate the full case and inspect its dispatches.
python "$RUNNER" \
  --case "$CASE" \
  --expected-arch gfx1250 \
  --environment ffm \
  --warmup 0 \
  --repeats 1 \
  --output ffm-result.json

"$ROCCAP" capture \
  --disp "$DISPATCH" \
  --file workload.cap \
  python "$RUNNER" \
    --case "$CASE" \
    --expected-arch gfx1250 \
    --environment ffm \
    --warmup 0 \
    --repeats 1
```

FFM is for functional validation and dispatch capture only. The runner emits
`latency_us: null` for FFM and AM because their event timing is not physical
hardware latency. Repeat the capture with a separate output directory and
dispatch selector for every component dispatch.

Replay each captured dispatch from a disposable AM output directory:

```bash
cd "$RUN_DIR"
source /opt/rocdtif/am_env.sh
export DtifFbBaseLocation=0x100000000
"$ROCCAP" play -r "0x100000000-0xE00000000" ./workload_0001.cap
```

Record AM cycles and available AM counters for every dispatch in the workload.
Do not compare one dispatch's AM time with a complete physical pipeline
latency. DSA-P1 AM replay is long-running but required; report an unfinished
replay as incomplete.

## Final validation

- Both architectures contain all five case IDs.
- Commands, inputs, dtypes, layouts, warmups, and repeats match.
- MLA-D1 uses FP8 absorbed Q and FP8 dense paged KV.
- DSA uses live top-k slots with dense FP8 MLA KV, not packed sparse
  attention-KV.
- Each case has physical workload latency and every component dispatch is
  represented in profiler or AM output.
- Raw JSON, profiler CSV, AM output, `--describe` output, software revisions,
  and failure notes are saved.
