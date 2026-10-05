#!/usr/bin/env bash
# Host side: run serve_bench.sh for each model in MODELS, one dedicated
# single-GPU container at a time, each under the host's GPU lock.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
: "${TS_ROOT:?set TS_ROOT to the TokenSpeed worktree under test}"
: "${IMAGE_REF:?set IMAGE_REF to the container image}"
: "${ARCH:?set ARCH to gfx950 or gfx1250}"
readonly CONFIG_ROOT="${CONFIG_ROOT:-/tdata/models/$USER}"
readonly CACHE_ROOT="${CACHE_ROOT:-$HOME/$ARCH/emulate-rank0-cache}"
readonly GPU_INDEX="${GPU_INDEX:-0}"
readonly CONTAINER_NAME="${CONTAINER_NAME:-$USER-emulate-rank0}"
readonly LABEL="${LABEL:-$(date -u +%Y%m%d-%H%M%S)}"
readonly OUT="${RESULTS_ROOT:-/data/results}/emulate-rank0-$ARCH-$LABEL"
# Word-split on purpose: e.g. "flock /tmp/gpu0.lock" or "/usr/local/bin/gpu-lock".
read -r -a gpu_lock <<< "${GPU_LOCK:-/usr/local/bin/gpu-lock}"
image_env=()
[[ -n "${TRITON_LIBHIP_PATH:-}" ]] && image_env+=(-e "TRITON_LIBHIP_PATH=$TRITON_LIBHIP_PATH")

mkdir -p "$OUT" "$CACHE_ROOT/cache" "$CACHE_ROOT/triton"
{
  echo "arch $ARCH"
  echo "tokenspeed $(git -C "$TS_ROOT" rev-parse HEAD)"
  git -C "$TS_ROOT" status --short
} > "$OUT/tokenspeed.rev"
git -C "$TS_ROOT" diff --binary HEAD > "$OUT/tokenspeed.diff"
git -C "$repo" rev-parse HEAD > "$OUT/benchmarks.rev" 2>/dev/null || true
docker image inspect "$IMAGE_REF" --format '{{.Id}}' > "$OUT/image.id"
echo "$IMAGE_REF" >> "$OUT/image.id"
cp "$here"/*.sh "$here"/*.py "$OUT/"
echo "### output $OUT"

for model in ${MODELS:-glm53 dsv41 kimik3}; do
  if [[ -n "$(docker ps -aq --filter "name=^/${CONTAINER_NAME}$")" ]]; then
    echo "### container $CONTAINER_NAME already exists; remove it first"
    exit 1
  fi
  mkdir -p "$OUT/$model"
  echo "### $model start $(date -u +%H:%M:%S)"
  "${gpu_lock[@]}" docker run --rm \
    --name "$CONTAINER_NAME" \
    --user root \
    --ipc=host \
    --shm-size=8g \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -e HOME=/root \
    -e ROCR_VISIBLE_DEVICES="$GPU_INDEX" \
    -e HIP_VISIBLE_DEVICES=0 \
    -e CUDA_VISIBLE_DEVICES=0 \
    "${image_env[@]}" \
    -e MODEL="$model" \
    -e PHASES="${PHASES:-perf hotspots}" \
    -e TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN="${TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN:-}" \
    -e TOKENSPEED_MOE_ROUTING_SIMULATION="${TOKENSPEED_MOE_ROUTING_SIMULATION:-}" \
    -e EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}" \
    -e BENCH_ARGS="${BENCH_ARGS:-}" \
    -v "$TS_ROOT:/workspace/tokenspeed:ro" \
    -v "$repo:/workspace/tokenspeed-kernel-amd-benchmarks:ro" \
    -v "$CONFIG_ROOT:/data/models:ro" \
    -v "$OUT:$OUT" \
    -v "$CACHE_ROOT/cache:/root/.cache" \
    -v "$CACHE_ROOT/triton:/root/.triton" \
    -w /workspace/tokenspeed \
    "$IMAGE_REF" \
    bash /workspace/tokenspeed-kernel-amd-benchmarks/emulate_rank0/scripts/serve_bench.sh \
      "$OUT/$model" || echo "### $model container exit $?"
  echo "### $model end $(date -u +%H:%M:%S)"
done
echo "### all done"
