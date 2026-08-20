#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" != "--" || "$#" -lt 2 ]]; then
  echo "usage: $0 -- command [args ...]" >&2
  exit 2
fi
shift

restore() {
  "${script_dir}/restore_mgcg_override.sh"
}

trap restore EXIT
"${script_dir}/enable_mgcg_override.sh"
"$@"
