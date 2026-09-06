#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
python scripts/train.py --config configs/mini_bsrnn.yaml --devices 4 "$@"
