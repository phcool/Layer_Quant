#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
flash_dir="${repo_root}/third_party/flash-attention"

if [[ ! -d "${flash_dir}" ]]; then
  echo "Missing vendored flash-attention directory: ${flash_dir}" >&2
  exit 1
fi

cd "${flash_dir}"
python -m pip install --no-build-isolation -e .
