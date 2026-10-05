#!/usr/bin/env bash
# Container side: serve MODEL as emulated rank 0 of a TP8 deployment with dummy
# weights and simulated speculative acceptance, then a smoke test and the
# 50k/500 benchmark at batch 1 and 16. kimik3-nospec is kimik3 without
# speculation. Usage: serve_bench.sh <output-dir>
set -uo pipefail
out="$1"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$out"
export PYTHONPATH=/workspace/tokenspeed/python:/workspace/tokenspeed/tokenspeed-kernel/python:/workspace/tokenspeed/tokenspeed-kernel-amd/python
export PYTHONDONTWRITEBYTECODE=1

# Accept lengths are the mean tokens per verify step that the matching CI job
# logs on MI35x with real weights; see README.md for the source of each.
accept=""
case "${MODEL:-}" in
  glm53)
    name=glm-5.3
    vocab=154880
    accept=2.9
    model_args=(/data/models/glm-5.3-config --kv-cache-dtype fp8_e4m3
      --speculative-algorithm MTP --speculative-num-steps 3
      --speculative-num-draft-tokens 4 --speculative-eagle-topk 1
      --draft-model-path-use-base)
    ;;
  dsv41)
    name=deepseek-v41-flash
    vocab=129280
    accept=3.9
    model_args=(/data/models/deepseek-v4.1-flash-config --language-model-only
      --dtype bfloat16 --moe-backend triton
      --speculative-algorithm DSPARK)
    ;;
  kimik3 | kimik3-nospec)
    name=kimi-k3
    vocab=160000
    model_args=(/data/models/kimi-k3-config --language-model-only
      --attention-backend mla --kv-cache-dtype fp8_e4m3)
    if [[ "$MODEL" == kimik3 ]]; then
      accept=3.75
      model_args+=(--speculative-algorithm EAGLE3
        --speculative-draft-model-path /data/models/kimi-k3-eagle3-mla-config
        --speculative-draft-model-quantization unquant
        --eagle3-layers-to-capture 2,46,90
        --speculative-num-steps 3 --speculative-num-draft-tokens 4
        --speculative-eagle-topk 1)
    fi
    ;;
  *)
    echo "### unknown MODEL='${MODEL:-}', expected glm53, dsv41, kimik3 or kimik3-nospec"
    exit 2
    ;;
esac
if [[ -n "$accept" ]]; then
  export TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN="${TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN:-$accept}"
else
  # The server refuses a simulated accept length without speculation.
  export TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN=""
fi
export TOKENSPEED_MOE_ROUTING_SIMULATION="${TOKENSPEED_MOE_ROUTING_SIMULATION:-uniform}"

common=(
  "${model_args[@]}"
  --served-model-name "$name"
  --trust-remote-code
  --load-format dummy
  --tensor-parallel-size 8
  --emulate-rank-zero
  --max-model-len 65536
  --max-num-seqs 16
  --chunked-prefill-size 8192
  --disable-prefill-graph
  --disable-kvstore
  --disable-prefix-caching
  --disable-autotune
  --disable-sampling-tp-sync
  --sampling-backend greedy
  --decode-log-interval 40
  # A wave of 16 x 50k prefills outlasts the gateway's default 5 s health
  # check three times over, and the worker is then refused until it recovers.
  --health-check-timeout-secs 1800
  --host 127.0.0.1
  --port 8000
)
read -r -a extra_args <<< "${EXTRA_SERVE_ARGS:-}"
common+=("${extra_args[@]}")
read -r -a bench_args <<< "${BENCH_ARGS:-}"
client=(python3 "$here/client.py")
client_args=(--model "$name" --vocab "$vocab" "${bench_args[@]}")

python3 - <<'PY' 2>&1 | tee "$out/env.txt"
import importlib.metadata as md

import torch

for name in ("torch", "tokenspeed-triton", "tokenspeed-scheduler", "tokenspeed-smg"):
    print(name, md.version(name))
props = torch.cuda.get_device_properties(0)
print("arch", props.gcnArchName, "memory_gib", round(props.total_memory / 2**30, 1))
PY
[[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "### env check failed"; exit 1; }
{
  echo "TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN=$TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN"
  echo "TOKENSPEED_MOE_ROUTING_SIMULATION=$TOKENSPEED_MOE_ROUTING_SIMULATION"
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
  pkill -KILL -f "smg_grpc_servicer|tokenspeed::|smg launch|rocprofv3" 2>/dev/null
  wait "$server" 2>/dev/null
  sleep 5
}

accept_summary() {
  grep -oE "avg_accept_len: [0-9.]+" "$1" | awk '{print $2}' | sort -n |
    awk -v want="$TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN" '
      { v[NR] = $1 }
      END {
        if (NR) printf "simulated %s, logged median %s over %d windows (min %s, max %s)\n",
          want, v[int((NR + 1) / 2)], NR, v[1], v[NR]
        else print "no avg_accept_len in the decode log"
      }'
}

status=0
phases=" ${PHASES:-perf hotspots} "

if [[ "$phases" == *" perf "* ]]; then
  cd /workspace/tokenspeed
  args=("${common[@]}" --cudagraph-capture-sizes 1 2 4 8 16)
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
  cp "$here/emulate_profiler_hook.py" "$site/"
  echo "import emulate_profiler_hook" > "$site/zz_emulate_profiler_hook.pth"
  # A torch wheel that bundles rocprofiler-sdk (found through RPATH, so
  # LD_LIBRARY_PATH cannot redirect it) loads a second copy of the SDK beside
  # rocprofv3's, which aborts at `import torch`. Both are SDK 1.1.0.
  torch_sdk="$(python3 -c 'import os, torch; print(os.path.dirname(torch.__file__))')/lib/librocprofiler-sdk.so"
  if [[ -f "$torch_sdk" && ! -L "$torch_sdk" && -e /opt/rocm/lib/librocprofiler-sdk.so.1 ]]; then
    ln -sf /opt/rocm/lib/librocprofiler-sdk.so.1 "$torch_sdk"
  fi
  rm -rf "$out/rocprof" "$out/profile" "$out/hotspots"
  args=("${common[@]}" --enforce-eager --disable-overlap-schedule)
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
  pid_file=/tmp/emulate-scheduler.pid
  if [[ -s "$pid_file" ]]; then
    scheduler=$(cat "$pid_file")
    parent=$(ps -o ppid= -p "$scheduler" | tr -d " ")
    # Keep the servicer from killing the scheduler while rocprofv3 flushes.
    [[ -n "$parent" ]] && kill -STOP "$parent"
    kill -TERM "$scheduler"
    for _ in $(seq 1 90); do
      kill -0 "$scheduler" 2>/dev/null || break
      sleep 2
    done
    kill -0 "$scheduler" 2>/dev/null && echo "### scheduler still alive after SIGTERM"
    [[ -n "$parent" ]] && kill -CONT "$parent"
  fi
  stop_server
  rm -f "$site/zz_emulate_profiler_hook.pth"
  if [[ $rc -eq 0 ]]; then
    python3 "$here/rocprof_to_traces.py" "$out/rocprof" "$out/profile" &&
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
