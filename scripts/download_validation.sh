#!/usr/bin/env bash
set -euo pipefail

output=${1:-data/downloads/validation_1000.zip}
file_id=1dPezrikPASvS2XfvceBF9VStflx3iVNj

mkdir -p "$(dirname "${output}")"
if [[ -s "${output}" ]]; then
  echo "Already downloaded: ${output}"
  exit 0
fi
python -m gdown \
  "https://drive.google.com/uc?id=${file_id}" \
  --output "${output}.download"
if [[ ! -s "${output}.download" ]]; then
  echo "Download is empty: ${output}.download" >&2
  exit 1
fi
mv "${output}.download" "${output}"
echo "Downloaded: ${output}"
