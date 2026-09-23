# Cross-architecture comparison at one commit

How to produce a `results/arch_compare_<short-sha>/` entry comparing MI355X
(`gfx950`) and MI455X (`gfx1250`) on the same TokenSpeed commit. Written for an
agent picking this up cold.

The deliverable is four runs and one command: a `performance` run and a
`hotspots` run on each architecture, then
`scripts/generate_arch_comparison.py` to turn the four JSON files into the
document. Everything in the document is derived from those four files, so the
document is reproducible from them alone.

## 1. What you need

| Thing | MI355X | MI455X |
|---|---|---|
| Host | local workstation, 2x MI355X | `heliosr-1b114-d04-1.mnb.dcgpu`, remote |
| Runner | `gfx950/run_physical_gfx950_ab.sh` | `gfx1250/run_physical_gfx1250_ab.sh` |
| Container | `zhewenyu/kimi-k3-e2e:tokenspeed-ffa16b13` | `tokenspeed-kimi-gfx1250:tokenspeed-ffa16b13-torch213` |
| Rank-0 checkpoint | `/data/models/kimi-k3-tp8ep1-rank0` | `/tdata/models/kimi-k3-tp8ep1-rank0` |
| GPU serialization | `flock /tmp/zhewenyu-kimi-gpu0.lock` | `/usr/local/bin/gpu-lock`, node-wide |

The runner scripts live outside this repo because they carry host paths. They
mount this repo and a TokenSpeed worktree into the container; the container
supplies only the Python environment.

On MI455X the lock is shared with every other user of the node. Never bypass
it, and never leave it held: see the failure modes at the end.

## 2. Per-commit worktrees

Both architectures must run the identical commit, so create a detached
worktree per commit on each host rather than moving one checkout around.

```bash
# on each host, inside any existing tokenspeed worktree
git fetch origin main
for c in <sha-a> <sha-b>; do
  git worktree add --detach "$HOME/worktrees/ts-rev-${c:0:8}" "$c"
done
```

Confirm the worktree is clean and at the intended commit before every run. A
stale or dirty worktree is the single most common cause of a wasted run, and
the runner's SHA assertion only catches the mismatch if you passed the right
`TOKENSPEED_SHA_OVERRIDE`.

## 3. Harness compatibility

The benchmark harness calls TokenSpeed's scheduler and executor directly, so a
commit that adds a required argument breaks it. Three such arguments are
already handled in `benchmark_logical_rank.py`; each is set to the value that
matches what this workload does, so none of them changes what is measured:

| Argument | Value | Why that value |
|---|---|---|
| `enable_l3_storage` | `False` | the harness configures no KV store backend, zero host pages, and L2 and prefix caching disabled |
| `ngram_inputs` | `None` | what the engine produces when n-gram speculation is off |
| `submit_remote_prefill` | `True` | only consulted when a plan carries a remote prefill, which this single-rank `role="null"` harness never produces |

If a newer commit adds a fourth, find it in under a minute instead of after a
full checkpoint load by running with dummy weights:

```bash
python3 toy_e2e/benchmark_logical_rank.py \
  --checkpoint "$CHECKPOINT" --load-format dummy --expected-arch gfx950 \
  --tokenspeed-revision smoke --model-revision smoke \
  --prompt-tokens 512 --output-tokens 8 --concurrency 1 \
  --chunked-prefill-size 8192 --cache-gib 8 \
  --warmup-waves 1 --measurement-waves 1 \
  --prompt-seed 7 --synthetic-vocabulary-size 160000 \
  --output /tmp/smoke.json
```

`--load-format dummy` skips the checkpoint, turning a ten-minute discovery
cycle into about forty seconds. Do this for every new commit before queueing
real runs. `profile_logical_rank_stages.py` imports its forward path from
`benchmark_logical_rank.py`, so it inherits the same fixes and the same smoke
coverage.

## 4. Collect the four runs

Per commit, per architecture, run `performance` then `hotspots`. The overrides
let one runner script serve any worktree:

```bash
# MI355X
START_PHYSICAL_GFX950_BENCHMARK=YES ALLOW_DIRTY_BENCHMARKS=YES \
TOKENSPEED_ROOT_OVERRIDE="$HOME/worktrees/ts-rev-$SHORT" \
TOKENSPEED_SHA_OVERRIDE="$FULL_SHA" \
CONTAINER_NAME_OVERRIDE="kimi-k3-gfx950-$SHORT-performance" \
RUN_LABEL="rev-$SHORT-performance" \
PERFORMANCE_PROMPT_TOKENS=50000 PROFILE_PROMPT_TOKENS=50000 \
  ./run_physical_gfx950_ab.sh performance
```

Swap `performance` for `hotspots` and the label to match. The MI455X runner
takes the same variables plus `START_PHYSICAL_GFX1250_BENCHMARK=YES`.

`ALLOW_DIRTY_BENCHMARKS=YES` is required because this repo's working tree
carries the harness fixes above. The comparison stays valid as long as every
run in one document uses the same benchmarks tree, which the recorded
`benchmarks_worktree_sha256` lets you verify afterwards.

Budget roughly seven minutes per run: about a minute of checkpoint load, forty
seconds for batch 1, three and a half minutes for batch 16. Four runs per
architecture is around half an hour, and the two architectures are independent
hosts so they can run at the same time. On MI455X add unbounded queueing time
for the node lock.

## 5. Generate the document

```bash
python3 toy_e2e/scripts/generate_arch_comparison.py \
  --revision "$FULL_SHA" \
  --commit-date "$(TZ=UTC git show -s --format=%cd \
      --date=format-local:%Y-%m-%d "$FULL_SHA")" \
  --gfx950-performance  <gfx950 perf>/result.json \
  --gfx950-hotspots     <gfx950 hs>/hotspots/hotspots.json \
  --gfx1250-performance <gfx1250 perf>/result.json \
  --gfx1250-hotspots    <gfx1250 hs>/hotspots/hotspots.json \
  --output-dir "toy_e2e/results/arch_compare_$(date -u +%Y%m%d)_${FULL_SHA:0:8}"
```

Name the directory `arch_compare_<YYYYMMDD>_<short-sha>`, using the commit's
own UTC date, so entries sort by code history. Use UTC rather than the
committer's local date: two commits landing hours apart can otherwise share a
date and lose their ordering. The document records the measurement date
separately, read from the inputs.

All four inputs must come from the same commit. The generator does not check
this, because the recorded revision is the runner's metadata rather than a
measurement, so verify it yourself from `software.tokenspeed_worktree_sha256`
in each `result.json`.

The document contains end-to-end performance, a functional-group breakdown, and
the heaviest kernels per stage. Kernels are bucketed by function because the
two architectures do not split work into the same kernels: gfx950 runs the MoE
as staged kernels plus a reduce, gfx1250 as two matmuls plus a separate reduce.
Comparing individual symbols across architectures is meaningless; comparing
buckets is not. `CATEGORIES` in the generator is that mapping and is also the
`Category` column of the kernel tables, so the two cannot drift apart.

If you add a category, check that no kernel matches two patterns. First match
wins, so a broad pattern placed above a narrow one silently swallows it, and
the bug shows up as a group that looks too good.

## 6. Failure modes worth knowing

**A crashed container can hold the node lock indefinitely.** An HSA fault
during weight loading left a container that `docker kill` could not reap
because its GPU context was wedged; the wrapping `flock` never exited and
blocked another user for two hours. If a run dies, confirm the lock is
released. Killing the `flock` process frees the lock even when the container
will not die. A wedged container may need a host power cycle.

**Interrupting a local SSH does not kill the remote run.** The remote `flock`
survives and keeps queueing. Kill it on the remote host, or it competes with
your own next attempt.

**Killing the `flock` wrapper does not release the lock.** Its `docker run`
child inherits the open file descriptor and survives as an orphan, so the lock
stays held by a process that is not a `flock` process. Counting your own
`flock` processes will therefore report the lock free while another user waits
on it. Always verify with the descriptor holder, which names whoever actually
has it regardless of the process:

```bash
fuser -v /data/lock/amd-gpu.lock
```

Kill that PID, then re-run the same command and confirm it prints nothing.

**`pkill -f` matches your own shell.** The pattern appears in the invoking
command line, so `pkill -f run_matrix.sh` kills the shell running it before it
does anything. Use `pkill -f "run_matrix[.]sh"`.

**The container image and the source can drift apart.** The image is pinned to
an older revision and supplies only the Python environment, so a commit that
requires a newer dependency will fail at import. The dummy-weight smoke test
catches this too.

**Identical results across runs do not prove the kernel wrote them.** PyTorch's
caching allocator hands back the same block, so stale memory is deterministic.
To find out whether a kernel writes a buffer, free a same-shaped buffer full of
a sentinel first and check whether the sentinel survives.
