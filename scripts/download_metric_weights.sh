#!/usr/bin/env bash
set -euo pipefail
target=${1:-checkpoints/metric_weights}
mkdir -p "$target"
curl -fL --retry 3 -o "$target/sig_bak_ovr.onnx" \
  https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
curl -fL --retry 3 -o "$target/model_v8.onnx" \
  https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/DNSMOS/DNSMOS/model_v8.onnx
