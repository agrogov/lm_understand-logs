#!/usr/bin/env bash
# Rebuild only the missing Ada BF16 kernels; keep the MacCUDA checkout untouched.
set -euo pipefail
WSP_ENGINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WSP_MACUDA_DIR="${1:-/Users/arogov/Downloads/vfio/macuda}"
WSP_GEMM_BUILD="$WSP_ENGINE_DIR/build/bf16-gemm"
mkdir -p "$WSP_GEMM_BUILD"
cp "$WSP_MACUDA_DIR/cuda-shim/libtinycublas/gemm.cu" "$WSP_GEMM_BUILD/gemm.cu"
container run --rm --network=none --platform linux/arm64 \
  --user "$(id -u):$(id -g)" -v "$WSP_GEMM_BUILD:/work" \
  nvidia/cuda:13.0.3-devel-ubuntu24.04 \
  nvcc -cubin -arch=sm_89 -O3 -o /work/gemm.sm_89.cubin /work/gemm.cu
strings "$WSP_GEMM_BUILD/gemm.sm_89.cubin" > "$WSP_GEMM_BUILD/symbols.txt"
grep -q '^tinyblas_gemm_bf16_tc2s_tn$' "$WSP_GEMM_BUILD/symbols.txt"
xxd -i -n gemm_cubin "$WSP_GEMM_BUILD/gemm.sm_89.cubin" > "$WSP_GEMM_BUILD/gemm_cubin.c"
WSP_SDK="$(xcrun --sdk macosx --show-sdk-path)"
/opt/homebrew/opt/llvm/bin/clang -O2 -arch arm64 -isysroot "$WSP_SDK" \
  -c "$WSP_GEMM_BUILD/gemm_cubin.c" -o "$WSP_GEMM_BUILD/gemm_cubin.o"
shasum -a 256 "$WSP_GEMM_BUILD/gemm.cu" "$WSP_GEMM_BUILD/gemm.sm_89.cubin" \
  > "$WSP_GEMM_BUILD/sha256.txt"
echo 'Built Ada BF16 GEMM cubin and native embedding object; external archives unchanged.'
