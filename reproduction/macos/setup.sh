#!/usr/bin/env bash
set -euo pipefail
WSP_MAC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'This environment requires native Apple Silicon macOS (arm64).' >&2
  exit 1
fi
export UV_CACHE_DIR="$WSP_MAC_DIR/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WSP_MAC_DIR/.python"
export HF_HOME="$WSP_MAC_DIR/.cache/huggingface"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$WSP_MAC_DIR/artifacts"
if [[ ! -x "$WSP_MAC_DIR/.venv/bin/python" ]]; then
  uv venv --python 3.12 "$WSP_MAC_DIR/.venv"
fi
if [[ -f "$WSP_MAC_DIR/uv.lock" ]]; then
  uv sync --project "$WSP_MAC_DIR" --frozen
else
  uv sync --project "$WSP_MAC_DIR"
fi
"$WSP_MAC_DIR/.venv/bin/python" "$WSP_MAC_DIR/doctor.py"
