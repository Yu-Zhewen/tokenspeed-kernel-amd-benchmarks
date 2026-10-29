#!/usr/bin/env bash
# Container side: serve MODEL with real weights over all of this node's GPUs
# (and, for a two-node layout, the other node's), then the smoke test and the
# 50k/500 benchmark at batch 1 and 16, and a rocprofv3 capture of rank 0.
# The workload, serve flags and simulated acceptance follow
# emulate_rank0/scripts/serve_bench.sh, without its emulation.
#
# ROLE=leader (default) runs the phases in PHASES and writes <output-dir>.
# ROLE=follower serves node NODE_RANK of a multi-node layout until it is
# stopped; it needs the same PHASES (one phase) as its leader.
# Usage: serve_bench.sh <output-dir>
set -uo pipefail
out="$1"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
emulate="$here/../../emulate_rank0/scripts"
role="${ROLE:-leader}"
mkdir -p "$out"
export PYTHONDONTWRITEBYTECODE=1

# Accept lengths are the ones emulate_rank0 simulates, so a verify step here
# does the same work as there. SIMULATE_ACCEPT=0 keeps the real acceptance.
accept=""
case "${MODEL:-}" in
  glm53flash)
    name=glm-5.3-flash
    vocab=154880
    accept=2.9
    layout=(--attn-tp-size 4 --ep-size 1)
    prefill_graph=()
    model_args=(/data/models/GLM-5.3-Flash --language-model-only
      --speculative-algorithm MTP --speculative-num-steps 3
      --speculative-num-draft-tokens 4 --speculative-eagle-topk 1)
    ;;
  dsv41)
    name=deepseek-v41-flash
    vocab=129280
    accept=3.9
    layout=(--tensor-parallel-size 4)
    prefill_graph=(--prefill-graph-max-tokens 8192)
    model_args=(/data/models/DeepSeek-V4.1-Flash --language-model-only
      --dtype bfloat16 --speculative-algorithm DSPARK)
    ;;
  kimik3)
    name=kimi-k3
    vocab=160000
    accept=3.75
    layout=(--tensor-parallel-size 8)
    prefill_graph=(--disable-prefill-graph)
    model_args=(/data/models/Kimi-K3 --language-model-only
      --attention-backend mla --kv-cache-dtype fp8_e4m3
      --speculative-algorithm EAGLE3
      --speculative-draft-model-path /data/models/kimi-k3-eagle3-mla
      --speculative-draft-model-quantization unquant
      --eagle3-layers-to-capture 2,46,90
      --speculative-num-steps 3 --speculative-num-draft-tokens 4
      --speculative-eagle-topk 1)
    ;;
  *)
    echo "### unknown MODEL='${MODEL:-}', expected glm53flash, dsv41 or kimik3"
    exit 2
    ;;
esac
if [[ "${SIMULATE_ACCEPT:-1}" == 1 ]]; then
  export TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN="${TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN:-$accept}"
else
  export TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN=""
fi
# Real weights route for real.
unset TOKENSPEED_MOE_ROUTING_SIMULATION

# NNODES > 1 splits the layout over nodes; HEAD_ADDR is node 0's ip:port.
multinode=()
if [[ "${NNODES:-1}" -gt 1 ]]; then
  : "${HEAD_ADDR:?set HEAD_ADDR to the ip:port of node 0 for a multi-node layout}"
  gpus=$(python3 -c "import torch; print(torch.cuda.device_count())")
  multinode=(--nnodes "$NNODES" --nprocs-per-node "$gpus"
    --world-size "$(( NNODES * gpus ))" --dist-init-addr "$HEAD_ADDR"
    --node-rank "${NODE_RANK:-0}")
fi

common=(
  "${model_args[@]}"
  --served-model-name "$name"
  --trust-remote-code
  "${layout[@]}"
  "${multinode[@]}"
  --max-model-len 65536
  --max-num-seqs 16
  --chunked-prefill-size 8192
  "${prefill_graph[@]}"
  --disable-kvstore
  --disable-prefix-caching
  --disable-autotune
  --disable-sampling-tp-sync
  --sampling-backend greedy
  --decode-log-interval 40
  # A wave of 16 x 50k prefills outlasts the gateway's default 5 s health
  # check three times over, and the worker is then refused until it recovers.
  --health-check-timeout-secs 1800
  # Loading a real checkpoint, Kimi-K3's in particular, outlasts the default.
  --engine-startup-timeout 3000
  --host "${BIND_HOST:-127.0.0.1}"
  --port 8000
)
read -r -a extra_args <<< "${EXTRA_SERVE_ARGS:-}"
common+=("${extra_args[@]}")
read -r -a bench_args <<< "${BENCH_ARGS:-}"
client=(python3 "$emulate/client.py")
client_args=(--model "$name" --vocab "$vocab" "${bench_args[@]}")
phases=" ${PHASES:-perf hotspots} "

# The hotspots phase profiles the perf phase's server, CUDA graphs included:
# eager collectives also time how far the other ranks' kernel launches lag
# rank 0's. HOTSPOT_GRAPHS=0 profiles emulate_rank0's eager server instead.
phase_args() {
  case "$1" in
    perf) echo --cudagraph-capture-sizes 1 2 4 8 16 ;;
    hotspots)
      if [[ "${HOTSPOT_GRAPHS:-1}" == 1 ]]; then
        phase_args perf
      else
        echo --enforce-eager --disable-overlap-schedule
      fi
      ;;
  esac
}

if [[ "$role" == follower ]]; then
  read -r phase <<< "$phases"
  read -r -a args <<< "$(phase_args "$phase")"
  echo "### follower node ${NODE_RANK:-?} phase $phase"
  exec python3 -m tokenspeed.cli serve "${common[@]}" "${args[@]}"
fi

python3 - <<'PY' 2>&1 | tee "$out/env.txt"
import importlib.metadata as md

import torch

for name in ("torch", "tokenspeed-triton", "tokenspeed-scheduler", "tokenspeed-smg"):
    print(name, md.version(name))
props = torch.cuda.get_device_properties(0)
print("arch", props.gcnArchName, "memory_gib", round(props.total_memory / 2**30, 1))
print("gpus_per_node", torch.cuda.device_count())
PY
[[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "### env check failed"; exit 1; }
echo "nnodes ${NNODES:-1}" >> "$out/env.txt"
[[ -r /src/REVISION ]] && echo "tokenspeed $(cat /src/REVISION)" >> "$out/env.txt"
{
  echo "TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN=$TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN"
  echo "TOKENSPEED_MOE_ROUTING_SIMULATION="
} | tee "$out/simulation.txt"

server=""
wait_ready() {
  local started
  started=$(date +%s)
  for _ in $(seq 1 720); do
    if ! kill -0 "$server" 2>/dev/null; then
      echo "### server exited during startup"
      return 1
    fi
    # /v1/models answers before the gateway has registered the tokenizer, and
    # requests in that gap fail with tokenizer_not_found.
    if python3 - "$name" 2>/dev/null <<'PY'
import json, sys, urllib.request as u
body = {"model": sys.argv[1], "input_ids": [1, 2, 3, 4],
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}}
request = u.Request("http://127.0.0.1:8000/generate", json.dumps(body).encode(),
                    {"Content-Type": "application/json"})
u.urlopen(request, timeout=60).read()
PY
    then
      echo "### ready after $(( $(date +%s) - started ))s"
      return 0
    fi
    sleep 5
  done
  return 1
}

stop_server() {
  kill -TERM "$server" 2>/dev/null
  for _ in $(seq 1 60); do
    kill -0 "$server" 2>/dev/null || break
    sleep 2
  done
  kill -KILL "$server" 2>/dev/null
  pkill -KILL -f "smg_grpc_servicer|tokenspeed::|smg launch|ts-serve|ts-host|rocprofv3" 2>/dev/null
  wait "$server" 2>/dev/null
  sleep 5
}

accept_summary() {
  grep -oE "avg_accept_len: [0-9.]+" "$1" | awk '{print $2}' | sort -n |
    awk -v want="$TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN" '
      { v[NR] = $1 }
      END {
        if (NR) printf "simulated %s, logged median %s over %d windows (min %s, max %s)\n",
          (want == "" ? "off" : want), v[int((NR + 1) / 2)], NR, v[1], v[NR]
        else print "no avg_accept_len in the decode log"
      }'
}

status=0

if [[ "$phases" == *" perf "* ]]; then
  cd /tmp
  read -r -a args <<< "$(phase_args perf)"
  args=("${common[@]}" "${args[@]}")
  printf '%q ' python3 -m tokenspeed.cli serve "${args[@]}" > "$out/serve_cmd.txt"
  python3 -m tokenspeed.cli serve "${args[@]}" > "$out/serve.log" 2>&1 &
  server=$!
  rc=1
  if wait_ready; then
    "${client[@]}" smoke --model "$name" --vocab "$vocab" > "$out/smoke.log" 2>&1 &&
      "${client[@]}" bench "${client_args[@]}" --output "$out/serve_bench.json" \
        > "$out/bench.log" 2>&1
    rc=$?
  fi
  echo "### perf exit $rc"
  [[ $rc -eq 0 ]] || status=$rc
  stop_server
  grep -E "Decode batch|Prefill batch|Mix batch" "$out/serve.log" > "$out/batches.log"
  accept_summary "$out/batches.log" | tee "$out/accept.txt"
fi

if [[ "$phases" == *" hotspots "* ]]; then
  # rocprofv3 buffers records under ./.rocprofv3, so run from a writable directory.
  mkdir -p /tmp/rocprof-cwd
  cd /tmp/rocprof-cwd
  site=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  cp "$emulate/emulate_profiler_hook.py" "$site/"
  echo "import emulate_profiler_hook" > "$site/zz_emulate_profiler_hook.pth"
  # A torch wheel that bundles rocprofiler-sdk (found through RPATH, so
  # LD_LIBRARY_PATH cannot redirect it) loads a second copy of the SDK beside
  # rocprofv3's, which aborts at `import torch`. Both are SDK 1.1.0.
  torch_sdk="$(python3 -c 'import os, torch; print(os.path.dirname(torch.__file__))')/lib/librocprofiler-sdk.so"
  if [[ -f "$torch_sdk" && ! -L "$torch_sdk" && -e /opt/rocm/lib/librocprofiler-sdk.so.1 ]]; then
    ln -sf /opt/rocm/lib/librocprofiler-sdk.so.1 "$torch_sdk"
  fi
  rank_dir=/tmp/emulate-scheduler.d
  rm -rf "$out/rocprof" "$out/profile" "$out/hotspots" "$rank_dir"
  read -r -a args <<< "$(phase_args hotspots)"
  args=("${common[@]}" "${args[@]}")
  printf '%q ' python3 -m tokenspeed.cli serve "${args[@]}" > "$out/serve_hotspots_cmd.txt"
  EMULATE_PROFILER_HOOK=roctx rocprofv3 --kernel-trace --marker-trace --selected-regions \
    --output-format csv -d "$out/rocprof" -- \
    python3 -m tokenspeed.cli serve "${args[@]}" > "$out/serve_hotspots.log" 2>&1 &
  server=$!
  rc=1
  if wait_ready; then
    "${client[@]}" profile "${client_args[@]}" --profile-dir "$out/profile-windows" \
      > "$out/profile.log" 2>&1
    rc=$?
  fi
  echo "### profile exit $rc"
  # Every local rank flushes its own rocprofv3 records on SIGTERM. Stop their
  # parents first so they do not kill a rank mid-flush.
  ranks=()
  for f in "$rank_dir"/*; do
    [[ -f "$f" ]] && ranks+=("$(basename "$f")")
  done
  parents=()
  for pid in "${ranks[@]}"; do
    parent=$(ps -o ppid= -p "$pid" | tr -d " ")
    [[ -n "$parent" ]] && parents+=("$parent")
  done
  parents=($(printf '%s\n' "${parents[@]}" | sort -u))
  ((${#parents[@]})) && kill -STOP "${parents[@]}"
  ((${#ranks[@]})) && kill -TERM "${ranks[@]}"
  alive=0
  for _ in $(seq 1 90); do
    alive=0
    for pid in "${ranks[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    ((alive)) || break
    sleep 2
  done
  ((alive)) && echo "### a rank is still alive after SIGTERM"
  ((${#parents[@]})) && kill -CONT "${parents[@]}"
  stop_server
  rm -f "$site/zz_emulate_profiler_hook.pth"
  rank0=""
  for pid in "${ranks[@]}"; do
    tag=$(cat "$rank_dir/$pid")
    echo "rank $tag pid $pid" >> "$out/profiled_ranks.txt"
    [[ "$tag" =~ ^(DP0-)?TP0$ ]] && rank0=$pid
  done
  if [[ $rc -eq 0 && -z "$rank0" ]]; then
    echo "### no profiled process reported rank TP0"
    rc=1
  fi
  if [[ $rc -eq 0 ]]; then
    python3 "$emulate/rocprof_to_traces.py" "$out/rocprof" "$out/profile" --pid "$rank0" &&
      python3 "$here/../../toy_e2e/scripts/summarize_gpu_hotspots.py" \
        --input "$out/profile" --top-k 15 --csv-dir "$out/hotspots/csv" \
        --output "$out/hotspots/hotspots.json" > /dev/null
    rc=$?
    echo "### hotspots exit $rc"
  fi
  [[ $rc -eq 0 ]] || status=$rc
fi
echo "### session exit $status"
exit "$status"
