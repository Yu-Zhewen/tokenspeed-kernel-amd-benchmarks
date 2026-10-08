# Emulated rank 0: speculative serving on one GPU per architecture

How to produce a `results/arch_compare_<date>_<short-sha>/` entry comparing
MI355X (`gfx950`) and MI455X (`gfx1250`) on the same TokenSpeed commit for
three speculative-decoding deployments:

| Model | Speculative algorithm | Layout emulated |
|---|---|---|
| GLM-5.3-Flash | MTP, 3 steps | TP4 (attention TP4, MoE TP4), rank 0 |
| DeepSeek-V4.1-Flash | DSPARK, block size 5 from the config | TP4, rank 0 |
| Kimi-K3 | EAGLE3 (`lightseekorg/kimi-k3-eagle3-mla`), 3 steps | TP8, rank 0 |

Each layout is the one the model's AMD CI job serves on MI35x.

An optional fourth run, `kimik3-nospec`, serves Kimi-K3 with the same flags
and no speculation, as the baseline for its EAGLE3 run.

The workload is 50,000 prompt tokens and 500 output tokens per request at
batch 1 and 16, the same shape as TokenSpeed's Kimi-K3 EAGLE3 CI perf job.
Written for an agent picking this up cold.

Unlike `toy_e2e/`, nothing here calls TokenSpeed internals. Each run is a real
`tokenspeed serve` with `--emulate-rank-zero`: one GPU executes global rank 0
of the model's layout with exact per-rank shapes, and collectives are replaced by
local substitutes. Weights are `--load-format dummy`, so only each model's
`config.json` and tokenizer files are needed, not a checkpoint. Two
simulations stand in for what random weights cannot reproduce:

- `TOKENSPEED_MOE_ROUTING_SIMULATION=uniform` spreads tokens evenly over the
  experts. Random router weights otherwise collapse onto a few experts.
- `TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN` fixes the mean tokens each verify step
  keeps per request. A randomly initialized drafter would accept almost
  nothing. Section 2 explains the value used for each model.

The deliverable is one run per architecture, each covering all three models
with a `perf` phase and a `hotspots` phase, then
`scripts/generate_arch_comparison.py` to turn the two result directories into
the document.

## 1. What you need

| Thing | MI355X | MI455X |
|---|---|---|
| Host | local workstation | `heliosr-1b114-d04-1.mnb.dcgpu`, remote |
| Container | `zhewenyu/emulate-rank0:<sha>-gfx950`, built from `docker/Dockerfile.amd` | `tokenspeed-gfx1250:d36f9bc8-torch214` |
| Model configs | `/data/models/zhewenyu-emulate-configs` | `/tdata/models/zhewenyu` |
| GPU serialization | `flock /tmp/zhewenyu-kimi-gpu0.lock`, and a GPU nobody else is using | `/usr/local/bin/gpu-lock`, node-wide |

`CONFIG_ROOT` must hold four config-only directories, mounted read-only at
`/data/models` in the container:

- `glm-5.3-flash-config`, the config, tokenizer, chat template, processor
  config and safetensors index of `zai-org/GLM-5.3-Flash` at `eb9eb208`
- `deepseek-v4.1-flash-config`
- `kimi-k3-config`
- `kimi-k3-eagle3-mla-config`, which is only the 1 KB `config.json` from
  `lightseekorg/kimi-k3-eagle3-mla`. The drafter loads through the same dummy
  loader, so its 6 GB `model.safetensors` is not needed.

The container supplies only the Python environment. `scripts/run.sh` mounts a
TokenSpeed worktree over it and puts the worktree's `python/` and kernel
packages first on `PYTHONPATH`. The MI455X image is older than the commit
under test, which works as long as no compiled package changed in between.
For MI355X, build the image from the commit under test so torch and the
compiled packages match:

```bash
cd "$TS_ROOT"
docker build --build-arg MAX_JOBS=16 -f docker/Dockerfile.amd \
  -t "zhewenyu/emulate-rank0:${SHORT}-gfx950" .
```

The first build downloads torch 2.14 (6.2 GB); later builds reuse that layer.

When `tokenspeed-scheduler` is the only compiled package that changed since
an earlier run's image, rebuilding just the scheduler is enough on either
architecture. Check with `git diff --name-only <image commit> HEAD`: changes
outside `tokenspeed-scheduler/` should be Python files, documentation, or CUDA
sources that AMD builds skip. Then derive the image:

```bash
mkdir -p /tmp/scheduler-image
git -C "$TS_ROOT" archive HEAD tokenspeed-scheduler | tar -x -C /tmp/scheduler-image
docker build --network=host --build-arg BASE=<earlier image> \
  -f emulate_rank0/scripts/Dockerfile.scheduler -t <new tag> /tmp/scheduler-image
```

### TokenSpeed requirements

Emulation arrived in TokenSpeed
[#1938](https://github.com/lightseekorg/tokenspeed/pull/1938), together with
the dummy-weight EAGLE3 embedding fix that Kimi-K3 needs at TP8. GLM-5.3-Flash's
block-FP8 MoE has no gfx1250 kernel before
[#1936](https://github.com/lightseekorg/tokenspeed/pull/1936), which merged
on 2026-10-03. #1938's branch includes it from `9b707d56`, where it merged
`main`, so from there the tree under test is #1938's head with nothing
applied. Before that, it was #1938's head with #1936's diff applied:

```bash
git worktree add --detach ~/worktrees/emulate-rank0-compare <1938-head>
cd ~/worktrees/emulate-rank0-compare
git -C ~/worktrees/triton-fp8-moe diff --binary <1936-base> HEAD > /tmp/pr1936.diff
git apply --index /tmp/pr1936.diff
sha256sum /tmp/pr1936.diff
```

Build the same tree on both hosts, and compare the diff hash. `run.sh` records
the commit, `git status` and the full diff in every result directory.

## 2. Accept lengths

Every value comes from the per-window `avg_accept_len` that TokenSpeed logs
during decode, in the real-weight CI jobs of
[AMD Tests run 37104243262](https://github.com/lightseekorg/tokenspeed/actions/runs/37104243262)
on `main` at `b2dd427d` (MI35x):

| Model | Value | Source job | Workload | Logged windows |
|---|---:|---|---|---|
| Kimi-K3 EAGLE3 | 3.75 | [`perf-kimi-k3-eagle3-mxfp4-tp8ep1-random-50k-500-mi35x`](https://github.com/lightseekorg/tokenspeed/actions/runs/37104243262/job/111153155794) | random 50k/500, batch 16 | 3.65, 3.85, 3.85 |
| GLM-5.3-Flash MTP | 2.9 | [`eval-glm-5.3-flash-fp8-mtp-aime26-amd`](https://github.com/lightseekorg/tokenspeed/actions/runs/37104243262/job/111153155786) | AIME26 | median 2.91 of 623, range 2.06 to 3.61 |
| DeepSeek-V4.1-Flash DSPARK | 3.9 | [`eval-deepseek-v4.1-flash-dspark-gsm8k-amd`](https://github.com/lightseekorg/tokenspeed/actions/runs/37104243262/job/111153155790) | gsm8k | median 3.93 of 68, range 3.69 to 4.20 |

Only Kimi-K3's value comes from this exact workload. Its first logged window,
1.94, is excluded: it straddles the prefill, so it averages over too few
verify steps.

The workload matters a lot. Kimi-K3's own AIME26 eval job logs a median of
2.41, against 3.75 on random prompts. GLM-5.3-Flash and DeepSeek-V4.1-Flash
have no random 50k/500 CI job, so their values describe real text and may not
match what this workload would accept. Override a value with
`TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN` when you have a better source.

The value must lie between 1 and the verify width, the most tokens one step
can keep. That width is 4 for MTP and EAGLE3 at 3 steps, and 6 for DSPARK.
DSPARK takes its step count from the checkpoint's block size (5) rather than
from the CLI. The CI log prints `speculative_num_steps=3` because it echoes
the arguments before they are resolved; passing 3 steps explicitly fails at
startup. The server logs the value it applied:

```text
Simulating speculative acceptance: every verify step keeps 3.9 tokens per request on average, of up to 6
```

## 3. Serve flags

`scripts/serve_bench.sh` holds every flag. Each model's flags follow its AMD
CI job, including its parallel layout (`--attn-tp-size 4 --ep-size 1` for
GLM-5.3-Flash, `--tensor-parallel-size 4` for DeepSeek-V4.1-Flash,
`--tensor-parallel-size 8` for Kimi-K3) and automatic MoE kernel selection,
with these differences:

| Difference | Why |
|---|---|
| `--language-model-only` for Kimi-K3, no `--mm-encoder-tp-mode data` | emulation rejects vision-encoder data parallelism |
| `--attention-backend mla` for Kimi-K3 | TokenSpeed's AMD recipe for Kimi-K3; CI's `gluon` was not tried under emulation |
| `--max-model-len 65536`, `--max-num-seqs 16`, no `--max-total-tokens` | room for 16 requests of 50,500 tokens; GLM-5.3-Flash's CI job caps the pool at 524,288 tokens, and DeepSeek-V4.1-Flash's allows 32 requests |
| prefix caching and KV store off | every request is unique, and a warm cache would hide prefill |
| `--sampling-backend greedy`, `--disable-sampling-tp-sync` | sampling is deterministic and has no peer ranks to sync with |

The `perf` phase serves with decode CUDA graphs at batch sizes 1, 2, 4, 8 and
16. The `hotspots` phase serves eagerly with `--disable-overlap-schedule`, so
every kernel is attributable to the forward that launched it.

## 4. Collect the runs

```bash
# MI455X, on heliosr-1b114-d04-1
cd <benchmarks worktree>/emulate_rank0/scripts
ARCH=gfx1250 IMAGE_REF=tokenspeed-gfx1250:d36f9bc8-torch214 \
TRITON_LIBHIP_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so \
TS_ROOT=$HOME/worktrees/emulate-rank0-compare LABEL=<short-sha> \
  nohup ./run.sh > $HOME/gfx1250/emulate-rank0-<short-sha>.log 2>&1 &

# MI355X, on the workstation
ARCH=gfx950 IMAGE_REF=zhewenyu/emulate-rank0:<short-sha>-gfx950 \
CONFIG_ROOT=/data/models/zhewenyu-emulate-configs RESULTS_ROOT=$HOME/gfx950/results \
GPU_INDEX=<idle GPU> GPU_LOCK="flock /tmp/zhewenyu-kimi-gpu0.lock" \
TS_ROOT=$HOME/worktrees/emulate-rank0-compare LABEL=<short-sha> \
  nohup ./run.sh > $HOME/gfx950/emulate-rank0-<short-sha>.log 2>&1 &
```

Results land in `$RESULTS_ROOT/emulate-rank0-<arch>-<label>/<model>/`:

| File | Contents |
|---|---|
| `serve_bench.json` | TTFT, TPOT and output throughput per batch size |
| `accept.txt` | simulated accept length against the logged per-window median |
| `batches.log` | the server's prefill and decode log lines, the source of the steady decode rate |
| `hotspots/hotspots.json` | per-kernel GPU time per stage and batch size |
| `serve.log`, `serve_hotspots.log` | server logs of the two phases |
| `serve_cmd.txt`, `env.txt`, `simulation.txt` | exact command, package versions, simulation settings |

`MODELS` picks a subset (`glm53flash dsv41 kimik3` by default; add
`kimik3-nospec` for the baseline without speculation) and `PHASES` picks
`perf`, `hotspots` or both. `BENCH_ARGS` passes extra client flags. A short check
with `BENCH_ARGS="--prompt-tokens 2048 --output-tokens 64"` catches flag or
startup problems before committing to full runs; for the two Flash models it
takes about 3 minutes. Their full runs at `927562ef` took 20 minutes on
MI355X and 18 on MI455X.

To redo one phase of one model, rerun with the same `LABEL` and, for example,
`MODELS=glm53flash PHASES=hotspots`. The hotspots phase clears its own outputs
first. `run.sh` rewrites the script copies at the top of the result
directory, so keep the first pass's copies if the scripts changed since.

Check `accept.txt` before trusting a run. In full runs the logged median
equals the simulated value, and only windows that straddle a prefill log
less. A short check logs only a few windows, so its median can sit lower. A
median far below the simulated value means the simulation did not apply.
For `kimik3-nospec`, `accept.txt` should read `no avg_accept_len in the decode
log`.

## 5. Generate the document

```bash
python3 emulate_rank0/scripts/generate_arch_comparison.py \
  --gfx950 <MI355X result dir> --gfx1250 <MI455X result dir> \
  --commit-date "$(TZ=UTC git -C "$TS_ROOT" show -s --format=%cd \
      --date=format-local:%Y-%m-%d HEAD)" \
  --output-dir "emulate_rank0/results/arch_compare_<YYYYMMDD>_<short-sha>"
```

The generator reads only `tokenspeed.rev`, `tokenspeed.diff` and `image.id`
at the top of each directory, and `serve_cmd.txt`, `env.txt`,
`simulation.txt`, `accept.txt`, `serve_bench.json`, `batches.log` and
`hotspots/hotspots.json` per model, so a local copy of the MI455X results
needs only those:

```bash
rsync -am --include='*/' --include='tokenspeed.*' --include='image.id' \
  --include='serve_cmd.txt' --include='copied-from.txt' \
  --include='env.txt' --include='simulation.txt' --include='accept.txt' \
  --include='serve_bench.json' --include='batches.log' --include='hotspots.json' \
  --exclude='*' <node>:/data/results/emulate-rank0-gfx1250-<label>/ <local copy>/
```

Decode is compared on the steady rate the server logs while every request is
decoding, not on TPOT p50. At batch 16 all requests arrive at once, and a
request whose prompt finishes early decodes between the other prompts'
prefill chunks, so its TPOT mostly measures prefill. The generated document
shows both.

The generator covers whichever models both directories hold. When they
include `kimik3-nospec`, the document adds a section comparing each
architecture's Kimi-K3 EAGLE3 run with its run without speculation.

The generator refuses to run when the two directories record different
commits, diffs or parallel layouts. It compares functional kernel groups rather than symbols,
for the same reason as the toy comparison: the two architectures split the
same work into different kernels.

## 6. Failure modes worth knowing

**A GPU in use by someone else ruins the numbers.** The local workstation's
lock file is ours alone and does not coordinate with other users. Before a
local run, check `rocm-smi --showpids` and pick a GPU with no other process.

**A crashed container can hold the MI455X lock indefinitely.** See the same
section in `toy_e2e/docs/arch-comparison.md`: verify the holder with
`fuser -v /data/lock/amd-gpu.lock`, not by counting your own processes.

**Never overwrite a script while bash is running it.** Bash reads a script
as it executes, so editing `serve_bench.sh` under a running container changes
the commands it runs next. Use a separate worktree for edits, or swap Python
files with an atomic `mv`.

**A long prefill wave makes the gateway drop the worker.** smg checks the
worker every 60 s with a 5 s timeout and refuses traffic after three misses.
Sixteen 50k prefills keep the scheduler busy for minutes, so with the
defaults the batch-16 warmup wave completes and the measured wave then fails
with HTTP 503 `no_available_workers`. `serve_bench.sh` passes
`--health-check-timeout-secs 1800`, which `ts serve` forwards to smg.

**The first request after startup can fail with `tokenizer_not_found`.** smg
answers `/v1/models` before it has registered the model's tokenizer. In one
MI455X run the readiness poll landed in that gap, and the first profiling
request, a second later, got HTTP 500 even though it sends token ids.
`wait_ready` therefore waits for a one-token `/generate` to succeed.

**rocprofv3 can write garbage for a kernel name.** In one MI355X Kimi-K3
profile, all 154 records of one kernel id held raw memory, NULs included, in
place of the name, and the CSV no longer decoded as UTF-8.
`rocprof_to_traces.py` decodes leniently, labels such records
`<unreadable name, kernel id N>` and prints how many it found. Check their
share of GPU time before trusting the hotspots; it was at most 0.02% of a
window there. Otherwise rerun the hotspots phase.

**rocprofv3 aborts at `import torch` with the PyTorch ROCm wheel.** The wheel
bundles its own `librocprofiler-sdk.so`, which `libtorch_cpu.so` finds through
`RPATH=$ORIGIN`, so a second copy of the SDK loads beside rocprofv3's and
fails with "Configuration request occurred outside of valid rocprofiler
configuration period". `LD_LIBRARY_PATH` cannot override an `RPATH`. The
hotspots phase symlinks the bundled copy to `/opt/rocm`'s inside the
disposable container; both are SDK 1.1.0 in the gfx950 image. The MI455X
image's torch bundles no SDK, so it is unaffected.

**The hotspots phase logs "scheduler still alive after SIGTERM".** The
scheduler is flushing rocprofv3 buffers. The script pauses the servicer so it
cannot kill the scheduler mid-flush, then cleans up. The message is harmless
when `hotspots exit 0` follows.

**`pkill -f` matches your own shell.** Use a bracket pattern such as
`pkill -f "run[.]sh"`.

## 7. Results

- [`results/arch_compare_20261008_0d579b89`](results/arch_compare_20261008_0d579b89/README.md):
  emulation base `d4a92893` plus the changes in `0d579b89`, all three models
  with speculative decoding. GLM-5.3-Flash on MI455X is copied from the
  earlier run of that same tree. The [bar chart](results/arch_compare_20261008_0d579b89/speedup.svg)
  compares MI455X's advantage over MI355X with the 2026-10-03 run.
- [`results/arch_compare_20261003_927562ef`](results/arch_compare_20261003_927562ef/README.md):
  #1938 at `927562ef` with #1936 applied. GLM-5.3-Flash and
  DeepSeek-V4.1-Flash were run at TP4 on 2026-10-07, replacing the earlier
  TP8 runs of GLM-5.3 and DeepSeek-V4.1-Flash. Kimi-K3 at TP8 is copied from
  the 2026-10-03 runs of the same tree, recorded in each `kimik3/copied-from.txt`;
  MI355X's Kimi-K3 hotspots were regenerated from its rocprofv3 output after
  the unreadable kernel name described in section 6.
- [`results/arch_compare_20261005_9b707d56`](results/arch_compare_20261005_9b707d56/README.md):
  #1938 at `9b707d56`, which includes #1936, with nothing applied. Kimi-K3
  only, with EAGLE3 and with `kimik3-nospec`. The MI455X image is
  `tokenspeed-gfx1250:d36f9bc8-torch214` with `tokenspeed-scheduler` rebuilt
  at `9b707d56`. The two runs took 24 minutes on MI355X and 18 on MI455X.
