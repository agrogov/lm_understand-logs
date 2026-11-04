#!/usr/bin/env bash
set -euo pipefail
WSP_ENGINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WSP_MACUDA_DIR="${WSP_MACUDA_DIR:-/Users/arogov/Downloads/vfio/macuda}"
mkdir -p "$WSP_ENGINE_DIR/build/hardware"
sh "$WSP_MACUDA_DIR/tools/gpu-lock.sh" acquire A "Qwen weight-space hardware validation" "$$"
trap 'sh "$WSP_MACUDA_DIR/tools/gpu-lock.sh" release A' EXIT
export BIN="$WSP_MACUDA_DIR/cuda-shim/build/bin-sm_89"
export PYTHONDONTWRITEBYTECODE=1
sh "$WSP_MACUDA_DIR/tools/nv_shim_step.sh" A "$@" \
  2>&1 | tee "$WSP_ENGINE_DIR/build/hardware/last-run.txt"
