#!/usr/bin/env bash
set -euo pipefail
WSP_MAC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR="$WSP_MAC_DIR/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WSP_MAC_DIR/.python"
export HF_HOME="$WSP_MAC_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -x "$WSP_MAC_DIR/.venv/bin/python" ]]; then
  echo 'Run bash reproduction/macos/setup.sh first.' >&2
  exit 1
fi
exec uv run --project "$WSP_MAC_DIR" --frozen "$@"
