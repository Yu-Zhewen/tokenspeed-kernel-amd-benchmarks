#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: bash run_evalscope_4k1k.sh <completions-url> <output-root> [tokenizer-path]" >&2
  exit 2
fi

url=$1
output_root=$2
tokenizer_path=${3:-/data/models/sunkist}
evalscope_bin=${EVALSCOPE_BIN:-evalscope}

common_args=(
  --model kimi-k3
  --url "$url"
  --api openai
  --dataset random
  --tokenizer-path "$tokenizer_path"
  --min-prompt-length 4096
  --max-prompt-length 4096
  --prefix-length 0
  --min-tokens 1024
  --max-tokens 1024
  --rate -1
  --seed 1
  --temperature 0
  --stream
  --no-timestamp
  --extra-args '{"ignore_eos": true}'
)

run_case() {
  local name=$1
  local parallel=$2
  local number=$3
  local warmup=$4

  "$evalscope_bin" perf \
    "${common_args[@]}" \
    --parallel "$parallel" \
    --number "$number" \
    --warmup-num "$warmup" \
    --outputs-dir "$output_root/$name"
}

run_case c1 1 3 1
run_case c16 16 48 16
