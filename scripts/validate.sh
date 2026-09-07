#!/usr/bin/env bash
set -euo pipefail

checkpoint=${1:-checkpoints/mini_bsrnn_best.ckpt}
data_dir=${2:-data/validation_1000}
output_dir=${3:-runs/validation_1000}
if [[ $# -ge 3 ]]; then
  shift 3
elif [[ $# -eq 2 ]]; then
  shift 2
elif [[ $# -eq 1 ]]; then
  shift 1
fi

device=${DEVICE:-cuda}
batch_size=${VALIDATION_BATCH_SIZE:-1}
validate_args=(
  --checkpoint "${checkpoint}"
  --data-dir "${data_dir}"
  --output-dir "${output_dir}"
  --device "${device}"
  --batch-size "${batch_size}"
)
if [[ -n "${VALIDATION_LIMIT:-}" ]]; then
  validate_args+=(--limit "${VALIDATION_LIMIT}")
fi

python scripts/validate.py "${validate_args[@]}"
python scripts/evaluate.py \
  --reference-scp "${output_dir}/manifests/clean.scp" \
  --enhanced-scp "${output_dir}/manifests/enhanced.scp" \
  --output-dir "${output_dir}/metrics" \
  --device "${device}" \
  "$@"
