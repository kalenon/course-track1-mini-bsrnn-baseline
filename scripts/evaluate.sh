#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 REFERENCE_SCP ENHANCED_SCP OUTPUT_DIR [extra evaluate.py args]" >&2
  exit 2
fi

reference_scp=$1
enhanced_scp=$2
output_dir=$3
shift 3

python scripts/evaluate.py \
  --reference-scp "${reference_scp}" \
  --enhanced-scp "${enhanced_scp}" \
  --output-dir "${output_dir}" \
  "$@"
