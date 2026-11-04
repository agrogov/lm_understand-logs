#!/usr/bin/env bash
set -euo pipefail
WSP_ENGINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WSP_MACUDA_DIR="${1:-/Users/arogov/Downloads/vfio/macuda}"
WSP_LLAMA_DIR="$WSP_MACUDA_DIR/llama.cpp"
WSP_SDK="$(xcrun --sdk macosx --show-sdk-path)"
WSP_CXX=/opt/homebrew/opt/llvm/bin/clang++
WSP_PROBE_NAME="${2:-link_probe}"
case "$WSP_PROBE_NAME" in link_probe|model_probe|inference_bench) ;; *) echo 'Unknown probe name' >&2; exit 2;; esac
mkdir -p "$WSP_ENGINE_DIR/build"
WSP_GEMM_EXTRA=()
if [[ -f "$WSP_ENGINE_DIR/build/bf16-gemm/gemm_cubin.o" ]]; then
  WSP_GEMM_EXTRA=("$WSP_ENGINE_DIR/build/bf16-gemm/gemm_cubin.o")
fi
"$WSP_CXX" -std=c++17 -O2 -arch arm64 -isysroot "$WSP_SDK" \
  -I"$WSP_LLAMA_DIR/include" -I"$WSP_LLAMA_DIR/ggml/include" \
  "$WSP_ENGINE_DIR/$WSP_PROBE_NAME.cpp" \
  "$WSP_LLAMA_DIR/build-null/src/libllama.a" \
  "$WSP_MACUDA_DIR/cuda-shim/build/ggml-backend-reg.cuda.o" \
  "$WSP_MACUDA_DIR/cuda-shim/build/libggml-cuda.sm_89.a" \
  "$WSP_MACUDA_DIR/cuda-shim/build/shim-sm_89/libtinycudart.a" \
  "${WSP_GEMM_EXTRA[@]}" \
  "$WSP_MACUDA_DIR/cuda-shim/build/shim-sm_89/libtinycublas.a" \
  "$WSP_MACUDA_DIR/cuda-shim/build/shim-sm_89/nv/libtinynv.a" \
  /opt/homebrew/opt/llvm/lib/libLLVMDemangle.a \
  "$WSP_LLAMA_DIR/build-null/ggml/src/libggml.a" \
  "$WSP_LLAMA_DIR/build-null/ggml/src/libggml-cpu.a" \
  "$WSP_LLAMA_DIR/build-null/ggml/src/libggml-base.a" \
  -framework Accelerate -framework Foundation -framework Security -lc++ -lm \
  -o "$WSP_ENGINE_DIR/build/$WSP_PROBE_NAME"
if [[ "$WSP_PROBE_NAME" == link_probe ]]; then
  env -u TINYNV_SOCKET -u TINYNV_PCI "$WSP_ENGINE_DIR/build/link_probe"
fi
