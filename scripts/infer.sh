#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INPUT_DIR OUTPUT_DIR [CHECKPOINT] [extra infer.py args]" >&2
  exit 2
fi

input_dir=$1
output_dir=$2
checkpoint=${3:-checkpoints/mini_bsrnn_best.ckpt}
if [[ $# -ge 3 ]]; then
  shift 3
else
  shift 2
fi

python scripts/infer.py \
  --input-dir "${input_dir}" \
  --output-dir "${output_dir}" \
  --checkpoint "${checkpoint}" \
  "$@"
