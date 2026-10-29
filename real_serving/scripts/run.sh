#!/usr/bin/env bash
# Host side: run serve_bench.sh for MODEL in one dedicated container that owns
# every GPU of this node. A multi-node layout runs this once per node: first
# ROLE=follower (detached) on each other node, then ROLE=leader on node 0.
#
# Required: IMAGE_REF, MODEL, OUT (result directory for this model).
# Optional: ROLE, PHASES, NNODES, NODE_RANK, HEAD_ADDR (node 0 ip:port),
#   NODE_IP (address this node's ranks advertise), MODELS_ROOT, CACHE_ROOT,
#   CONTAINER_NAME, GPU_LOCK, ALLOW_BUSY_GPU=1, SIMULATE_ACCEPT, HOTSPOT_GRAPHS,
#   EXTRA_SERVE_ARGS, BENCH_ARGS, and NCCL_*/GLOO_* passed through.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
: "${IMAGE_REF:?set IMAGE_REF to the container image}"
: "${MODEL:?set MODEL to glm53flash, dsv41 or kimik3}"
: "${OUT:?set OUT to the result directory for this model}"
readonly ROLE="${ROLE:-leader}"
readonly MODELS_ROOT="${MODELS_ROOT:-/data/models}"
readonly CACHE_ROOT="${CACHE_ROOT:-$HOME/real-serving-cache}"
readonly CONTAINER_NAME="${CONTAINER_NAME:-$USER-real-serving-$ROLE}"
# Word-split on purpose: e.g. "/usr/local/bin/gpu-lock".
read -r -a gpu_lock <<< "${GPU_LOCK:-}"

# Refuse to share GPUs: any other /dev/kfd holder (inb-node-agent is the
# node's monitoring agent) or any GPU process.
holders="$(for p in $(sudo -n lsof -t /dev/kfd 2>/dev/null); do
  cat "/proc/$p/comm" 2>/dev/null; done | grep -vx inb-node-agent | sort -u | tr '\n' ' ' || true)"
amd_smi="$(command -v amd-smi || echo /opt/rocm/bin/amd-smi)"
# amd-smi reads stdin, which would swallow a script piped to bash.
procs="$("$amd_smi" process < /dev/null 2>/dev/null | grep -c 'PID:' || true)"
if [[ -n "${holders// }" || "${procs:-0}" -gt 0 ]]; then
  echo "### GPUs in use on $(hostname -s): kfd holders [$holders], $procs GPU processes"
  [[ "${ALLOW_BUSY_GPU:-0}" == 1 ]] || exit 64
fi
if [[ -n "$(docker ps -aq --filter "name=^/${CONTAINER_NAME}$")" ]]; then
  echo "### container $CONTAINER_NAME already exists; remove it first"
  exit 1
fi

mkdir -p "$OUT" "$CACHE_ROOT/cache" "$CACHE_ROOT/triton"
if [[ "$ROLE" == leader ]]; then
  docker image inspect "$IMAGE_REF" --format '{{.Id}}' > "$OUT/image.id"
  echo "$IMAGE_REF" >> "$OUT/image.id"
  git -C "$repo" rev-parse HEAD > "$OUT/benchmarks.rev" 2>/dev/null || true
fi

env_args=(-e MODEL="$MODEL" -e ROLE="$ROLE" -e PHASES="${PHASES:-perf hotspots}"
  -e NNODES="${NNODES:-1}" -e NODE_RANK="${NODE_RANK:-0}"
  -e HEAD_ADDR="${HEAD_ADDR:-}" -e SIMULATE_ACCEPT="${SIMULATE_ACCEPT:-1}"
  -e HOTSPOT_GRAPHS="${HOTSPOT_GRAPHS:-1}"
  -e EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}" -e BENCH_ARGS="${BENCH_ARGS:-}"
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)
[[ -n "${NODE_IP:-}" ]] && env_args+=(-e TOKENSPEED_HOST_IP="$NODE_IP")
[[ "${NNODES:-1}" -gt 1 ]] && env_args+=(-e BIND_HOST=0.0.0.0)
[[ -n "${CONTAINER_LD_LIBRARY_PATH:-}" ]] &&
  env_args+=(-e LD_LIBRARY_PATH="$CONTAINER_LD_LIBRARY_PATH")
while IFS='=' read -r key _; do
  env_args+=(-e "$key")
done < <(env | grep -E '^(NCCL|GLOO)_[A-Z_]*=' || true)

# The image's rdma-core has no provider for AMD Pollara (ionic) NICs; the
# host's has the same ABI.
net_args=()
lib=/usr/lib/x86_64-linux-gnu
if [[ -e /etc/libibverbs.d/ionic.driver && -e /dev/infiniband ]]; then
  ionic="$(ls "$lib"/libionic.so.1.* 2>/dev/null | head -1)"
  net_args+=(--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1
    -v "$lib/libibverbs/libionic-rdmav34.so:$lib/libibverbs/libionic-rdmav34.so:ro"
    -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro)
  [[ -n "$ionic" ]] && net_args+=(-v "$ionic:$lib/libionic.so.1:ro")
fi

detach=()
[[ "$ROLE" == follower ]] && detach=(-d)
echo "### $MODEL $ROLE node ${NODE_RANK:-0} start $(date -u +%H:%M:%S)"
"${gpu_lock[@]}" docker run --rm "${detach[@]}" \
  --name "$CONTAINER_NAME" \
  --user root \
  --network host \
  --ipc host \
  --shm-size 64g \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  "${net_args[@]}" \
  -e HOME=/root \
  "${env_args[@]}" \
  -v "$repo:/workspace/tokenspeed-kernel-amd-benchmarks:ro" \
  -v "$MODELS_ROOT:/data/models:ro" \
  -v "$OUT:$OUT" \
  -v "$CACHE_ROOT/cache:/root/.cache" \
  -v "$CACHE_ROOT/triton:/root/.triton" \
  -w /tmp \
  "$IMAGE_REF" \
  bash /workspace/tokenspeed-kernel-amd-benchmarks/real_serving/scripts/serve_bench.sh \
    "$OUT" || echo "### $MODEL container exit $?"
echo "### $MODEL $ROLE end $(date -u +%H:%M:%S)"
