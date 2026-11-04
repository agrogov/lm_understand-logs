#!/usr/bin/env bash
# Run from the repo root after preparing GGUFs and the BF16 kernel override.
set -euo pipefail
WSP_ENGINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WSP_BENCH_TAG="${1:-$(date +%Y%m%d-%H%M%S)}"
WSP_BENCH_CASES="${2:-1:2048,4:4096,8:4096,4:32768}"
WSP_MACUDA_DIR="${WSP_MACUDA_DIR:-/Users/arogov/Downloads/vfio/macuda}"
test -f "$WSP_ENGINE_DIR/build/bf16-gemm/gemm_cubin.o"
bash "$WSP_ENGINE_DIR/build_link_probe.sh" "$WSP_MACUDA_DIR" inference_bench
"$WSP_ENGINE_DIR/converter/.venv/bin/python" "$WSP_ENGINE_DIR/prepare_benchmark.py"
for WSP_CHECKPOINT in base edited; do
  bash "$WSP_ENGINE_DIR/hardware_run.sh" probe \
    "$WSP_ENGINE_DIR/build/inference_bench" \
    "$WSP_ENGINE_DIR/build/hardware/qwen35-$WSP_CHECKPOINT-bf16.gguf" \
    "$WSP_ENGINE_DIR/build/benchmark-prompts" \
    "$WSP_ENGINE_DIR/build/benchmark-$WSP_CHECKPOINT-$WSP_BENCH_TAG" \
    "$WSP_BENCH_CASES"
done
"$WSP_ENGINE_DIR/converter/.venv/bin/python" "$WSP_ENGINE_DIR/summarize_benchmark.py" \
  --base "$WSP_ENGINE_DIR/build/benchmark-base-$WSP_BENCH_TAG" \
  --edited "$WSP_ENGINE_DIR/build/benchmark-edited-$WSP_BENCH_TAG" \
  --out "$WSP_ENGINE_DIR/build/benchmark-comparison-$WSP_BENCH_TAG.json"
