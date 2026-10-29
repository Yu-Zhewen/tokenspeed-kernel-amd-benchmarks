# Real serving: hotspots of the three speculative deployments on MI455X

How to produce a `results/hotspots_<date>_<short-sha>/` entry: the same three
deployments and workload as `emulate_rank0/`, served with real checkpoints
over their real multi-GPU layouts on MI455X (`gfx1250`), with rank 0's
kernels profiled.

| Model | Speculative algorithm | Layout | GPUs |
|---|---|---|---|
| GLM-5.3-Flash | MTP, 3 steps | TP4 (attention TP4, MoE TP4) | 4 on one node |
| DeepSeek-V4.1-Flash | DSPARK, block size 5 from the config | TP4 | 4 on one node |
| Kimi-K3 | EAGLE3 (`lightseekorg/kimi-k3-eagle3-mla`), 3 steps | TP8 | 4 on each of two nodes |

The workload is 50,000 prompt tokens and 500 output tokens per request at
batch 1 and 16. Written for an agent picking this up cold; read
`emulate_rank0/README.md` first, since this reuses its client, profiler hook,
trace splitter and kernel categories.

What changes against emulation:

- Real weights and real MoE routing; `TOKENSPEED_MOE_ROUTING_SIMULATION` is
  unset. Prompts are still random token ids.
- Real collectives, whose kernels form the `communication` bucket of the
  report. On the a31 trays the eight GPUs of both nodes form one scale-up
  domain: RCCL reports them as `UALoE-enabled (aka MNNVL)` at 1.8 Tb/s,
  counts one node of 8 local ranks, and connects its rings
  `via P2P/CUMEMMNNVL` rather than over the RoCE NIC. Within a node,
  TokenSpeed also runs Iris collectives; across nodes it uses RCCL.
- `TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN` still fixes the accept length, to
  emulate_rank0's values (2.9, 3.9, 3.75), so a verify step does the same
  work. It needs no emulation. `SIMULATE_ACCEPT=0` keeps the real
  acceptance, which on random prompts says little about the drafter.
- The hotspots phase profiles a server with the perf phase's flags, CUDA
  graphs included; rocprofv3 traces the kernels inside the graphs. Eager
  profiling, emulate_rank0's method, inflates the collectives: with eager
  launches a rank 0 all-reduce also waits for the other ranks' slower
  launches. GLM-5.3-Flash's decode all-reduce took 139 us per call eager and
  37 us with graphs. `HOTSPOT_GRAPHS=0` restores the eager profile.
- rocprofv3 writes one `<pid>_*.csv` set per rank. The profiler hook records
  each pid's rank tag, and only `TP0`'s records become traces.

## 1. What you need

- Nodes with 4 MI455X each. A two-node layout also needs RDMA NICs; on the
  a31 tray (AMD Pollara, `ionic_0`), `run.sh` mounts the host's ionic
  verbs provider, which the image lacks, and needs `NCCL_IB_HCA=ionic_0`,
  `NCCL_IB_GID_INDEX=1` and `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` set to
  the RoCE interface.
- An image built from the commit under test with
  `scripts/Dockerfile.gfx1250`, whose context `src/` is
  `git archive <sha> python tokenspeed-kernel tokenspeed-kernel-amd
  tokenspeed-scheduler tokenspeed-engine` plus a `REVISION` file holding the
  sha. Its constraint file keeps torch from the base image but lets
  `tokenspeed-triton` and `tokenspeed-proton` follow the tree's pins. The
  build takes about 5 minutes on 64 cores; update its version asserts when
  the pins move.
- Checkpoints under `MODELS_ROOT`, mounted read-only at `/data/models`:
  `GLM-5.3-Flash`, `DeepSeek-V4.1-Flash`, `Kimi-K3` and
  `kimi-k3-eagle3-mla`.

Three environment fixes the a31 nodes need, all passed by `run.sh`:

| Setting | Why |
|---|---|
| `CONTAINER_LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib` | `tokenspeed_kernel.platform` dlopens `libamdhip64.so` by name |
| `NODE_IP=<the node's routable address>` (becomes `TOKENSPEED_HOST_IP`) | the hostname resolves to 127.0.1.1, which the ranks would advertise; a two-node mesh then never connects |
| `NCCL_GRAPH_REGISTER=0` | RCCL's capture-time buffer registration fails the next launch on this stack |

## 2. Run

`scripts/run.sh` runs one model on one node in a container that owns all of
the node's GPUs, and refuses to start while another process holds
`/dev/kfd`. One-node models:

```bash
IMAGE_REF=<image> MODEL=glm53flash OUT=<results>/<label>/glm53flash \
MODELS_ROOT=<checkpoints> NODE_IP=<ip> \
CONTAINER_LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib \
NCCL_GRAPH_REGISTER=0 \
  nohup real_serving/scripts/run.sh > <results>/<label>/glm53flash.run.log 2>&1 &
```

Kimi-K3 runs one phase at a time, since node 1 serves only engines and has
to restart with each phase's flags. Per phase, start node 1 with
`ROLE=follower NNODES=2 NODE_RANK=1 HEAD_ADDR=<node 0 ip>:25000 PHASES=<phase>`
(it detaches), then node 0 with `ROLE=leader NNODES=2 NODE_RANK=0` and the same
`HEAD_ADDR` and `PHASES`, which blocks until the phase ends; then remove the
follower container. Node 0 runs the gateway and the client, and holds the
results.

`BENCH_ARGS="--prompt-tokens 2048 --output-tokens 64 --concurrency 1 --decode-steps 8"`
gives a short check. GLM-5.3-Flash takes about 8 minutes for both phases
of that check, most of it loading the checkpoint twice.

Results per model hold the same files as emulate_rank0's, plus
`profiled_ranks.txt` (pid and rank tag of each profiled process) and
`launch.txt` (the node's host and address, and the container's
`LD_LIBRARY_PATH`, `NCCL_*` and `GLOO_*` settings). For Kimi-K3, append
node 1's `node 1 <host> <ip>` line to node 0's `launch.txt`; the report's
"How each model was run" section reads it.

Two optional inputs feed the report's collectives appendix. After the
hotspots phase, on node 0:

```bash
python3 real_serving/scripts/rank_balance.py <results>/<label>/<model> \
  > <results>/<label>/<model>/rank_balance.txt
```

With the GPUs free, time the all-reduce of the model's hidden size outside
the server, in the serving image with the serving container's devices,
mounts and `NCCL_*` settings, one container per node started together:

```bash
python3 real_serving/scripts/allreduce_probe.py --hidden <hidden_size> \
  --nnodes <N> --node-rank <r> --master <node 0 ip> \
  > <results>/<label>/<model>/allreduce_probe.txt   # rank 0's node only
```

Hidden sizes: GLM-5.3-Flash 4096, DeepSeek-V4.1-Flash 5120, Kimi-K3 7168.

## 3. Generate the document

```bash
python3 real_serving/scripts/generate_hotspot_report.py \
  --run <results>/<label> \
  --commit-date "$(TZ=UTC git -C "$TS_ROOT" show -s --format=%cd \
      --date=format-local:%Y-%m-%d <sha>)" \
  --output-dir real_serving/results/hotspots_<YYYYMMDD>_<short-sha>
```

It needs per model `serve_cmd.txt`, `serve_hotspots_cmd.txt`, `env.txt`,
`image.id`, `simulation.txt`, `accept.txt`, `serve_bench.json`,
`batches.log` and `hotspots/hotspots.json`, and reads `launch.txt`,
`profiled_ranks.txt`, `rank_balance.txt` and `allreduce_probe.txt` when
present.

## 4. Failure modes worth knowing

Those of emulate_rank0 apply. In addition:

**"a rank is still alive after SIGTERM" after the hotspots phase.** One of
the ranks did not exit within 180 s while the others flushed. It is harmless
when `hotspots exit 0` follows and the trace split reports rank 0's kernels.

**A rank's communication time includes waiting.** A collective kernel starts
when its rank reaches it and ends when the slowest rank does, so rank 0's
communication bucket holds the other ranks' lag as well as the transfer.

## 5. Results

- [`results/hotspots_20261029_f198a543`](results/hotspots_20261029_f198a543/README.md):
  main at `f198a543` on the a31 trays, GLM-5.3-Flash and DeepSeek-V4.1-Flash
  on a31-2 and Kimi-K3 over a31-1 and a31-2. The image is
  `tokenspeed-gfx1250:main-f198a543` with
  `tokenspeed-triton` 3.9.10.post20261028 and every TokenSpeed package
  built from the tree.
