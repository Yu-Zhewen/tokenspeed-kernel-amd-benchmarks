#!/usr/bin/env bash

set -euo pipefail

umr_instance="${UMR_INSTANCE:-1}"
registers=(
  0x3b120
  0x7b120
  0xbb120
  0xfb120
  0x13b120
  0x17b120
  0x1bb120
  0x1fb120
)

for register in "${registers[@]}"; do
  sudo umr -i "${umr_instance}" -go 0 -vmp -r "${register}"
  sudo umr -i "${umr_instance}" -go 0 -vmp -w "${register}" 0x400
  sudo umr -i "${umr_instance}" -go 0 -vmp -r "${register}"
done
