#!/usr/bin/env bash
set -euo pipefail

output=${1:-data/downloads/validation_1000.zip}
expected_sha256=edd77dccb6cc1d7c273f2a05a8daee0d26956bc748ac2472a1f9f7305a896080
file_id=1dPezrikPASvS2XfvceBF9VStflx3iVNj

mkdir -p "$(dirname "${output}")"
if [[ -f "${output}" ]] && \
  echo "${expected_sha256}  ${output}" | sha256sum --check --status; then
  echo "Already downloaded and verified: ${output}"
  exit 0
fi
python -m gdown \
  "https://drive.google.com/uc?id=${file_id}" \
  --output "${output}"
echo "${expected_sha256}  ${output}" | sha256sum --check --status
echo "Downloaded and verified: ${output}"
