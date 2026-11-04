# Custom MacCUDA engine: local source audit and offline build proof

Audited 2026-11-04, using `/Users/arogov/Downloads/vfio/macuda` at commit
`e797e31`. The checkout has existing user changes; **none were modified**.
Existing build artifacts have their own provenance: the linked driver announces
build `ba4e38d`. Reusing those artifacts does not establish that they match every
line of the currently checked-out source.

## Conclusion

**Yes: a custom forward-inference research engine can use the same build and
runtime approach.** The best starting point is a narrow application over the
existing llama.cpp/GGML engine, with diagnostic tensor callbacks, rather than a
new general-purpose CUDA/PyTorch implementation.

I compiled and linked a custom native arm64 executable against the existing
**`sm_89` CUDA archive and shim libraries**, and ran its offline callback-selection
checks successfully. That verifies a concrete build/link path. It does **not**
validate an RTX 4090 forward pass, actual intermediate-tensor downloads, BF16
numerical parity, or full experiment performance.

The lack of gradient-based training is **helpful** here. We need forward
generation and teacher-forced diagnostics; CPU code can compute the edits and
statistics. Autograd, optimizer kernels and backward operators are unnecessary.

**Hardware follow-up:** the connected RTX 4090 has now passed operator checks
and custom BF16 base/edited model inference with real activation readback and
CPU numerical comparisons. An isolated rebuild of missing Ada BF16 GEMM kernels
was needed. See [the hardware validation report](HARDWARE_TEST.md). The limits
of the initial offline proof above should not be confused with this later test.

**Throughput follow-up:** a dedicated native sampling/batching benchmark is now
available in `inference_bench.cpp`, with a two-checkpoint runner and a workload
calculator. See [the connected-eGPU benchmark](BENCHMARK.md) for completed
measurements and the comparison against MLX. This is separate from the
32-token diagnostic smoke test.

## Source evidence

Paths below are relative to the audited MacCUDA checkout:

| Requirement | Existing implementation | Implication |
|---|---|---|
| Split CUDA compilation | `cuda-shim/build/tinycc` | Linux `nvcc` builds device fatbins; native Mac Clang builds host objects and embeds them |
| Application linking | `cuda-shim/build/link-null.sh` | A custom CMake executable can be linked with the GGML CUDA archive, runtime, BLAS and driver libraries |
| Ada/4090 selection | `setup.sh`, lines 18–39 | `ARCH=sm_89` selects distinct archives/build directories; helper cubins use the compatible Ampere set |
| BF16 matrix multiplication | `cuda-shim/libtinycublas/cublas.c`, lines 118–137 | BF16 GEMM with FP32 accumulation has an explicit accepted type contract |
| Qwen3.5 forward graph | `llama.cpp/src/models/qwen35.cpp` | Full-attention, Gated DeltaNet, dense MLPs and residual connections already exist |
| Gated DeltaNet CUDA dispatch | `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`, `GGML_OP_GATED_DELTA_NET` | The hybrid model does not require inventing this operator in a new engine |
| Evaluation callbacks | `llama.cpp/include/llama.h`, `cb_eval`; `ggml/include/ggml-backend.h`, lines 309–316 | Select graph tensors before execution and observe them after computation |
| Tensor downloads | `ggml_backend_tensor_get`; `llama.cpp/common/debug.cpp`, lines 143 onward | There is an existing pattern for reading selected GPU intermediate values |
| Unquantized conversion | `llama.cpp/convert_hf_to_gguf.py` and `conversion/qwen.py` | Qwen3.5 and BF16 output are recognized; converted tensor values/layouts still need verification |

Important for this checkout: `cuda-shim/build/libggml-cuda.a` currently points
to **`libggml-cuda.sm_120a.a`**, the Blackwell archive. A 4090 build must explicitly
use `libggml-cuda.sm_89.a` plus `build/shim-sm_89/`; do not silently use the default
symlink. The supplied probe does this explicitly.

## Where the experiment's tensors come from

The Qwen3.5 graph labels block output `l_out`, and the generic dense FFN builder
labels the SwiGLU intermediate `ffn_swiglu` **before** the down projection.
`llama-context.cpp` formats layer names as `<label>-<layer>`.

For the published intervention this gives:

- `l_out-15` and `l_out-19`: post-block residuals, corresponding to original
  hidden-state indices 16 and 20 when no control vectors are applied.
- `ffn_swiglu-15` and `ffn_swiglu-19`: activated MLP neurons entering `down_proj`.
- `result_output`: vocabulary logits; request the appropriate token rows for
  teacher-forced log-probabilities.

There is already enough graph visibility for the **central repetition-direction
method**. Full attention probability maps require more investigation: a fused
attention path may not materialize the softmax tensor. A diagnostic mode may
need to disable flash/fused attention or add an explicit graph node. Capturing
intermediates can also affect fusion and synchronization; keep it off for bulk
generation and validate captured values against a trusted reference.

## Offline proof included here

```bash
bash reproduction/macuda-engine/build_link_probe.sh
```

Or pass another existing MacCUDA checkout as its first argument. The script
writes only to `reproduction/macuda-engine/build/`, never to the source checkout.
It needs the already-built libraries, Mac SDK and existing Homebrew LLVM.

The probe:

1. Compiles our own C++ main and callback.
2. Resolves llama.cpp, GGML CUDA, and all three MacCUDA shim/driver archives.
3. Checks selection of the layer-15/19 diagnostic names with synthetic CPU
   tensor descriptors.
4. Runs with `TINYNV_SOCKET` and `TINYNV_PCI` removed so static CUDA registration
   uses the null-device path. The shim can initialize that null driver before
   `main`; this is **not** a real GPU execution test.

It deliberately does not call model loading or `llama_decode`, and does not
exercise the callback's GPU-download branch. A successful null-device check
must never be reported as successful inference.

## Proposed research worker

```text
Python controller / CPU weight analysis (existing uv environment)
          |
          | structured request files / local process protocol
          v
Native C++ research worker over llama.cpp + GGML CUDA
    generation: token IDs, texts, seeds, lengths
    teacher-forced: selected logits and intermediate tensors
          |
          v
MacCUDA runtime + BLAS + driver -> TinyGPU -> RTX 4090
```

Implement these pieces in order:

1. **Generation parity:** load unquantized BF16 GGUFs for base/edited models;
   preserve tokenizer, thinking template and sampling semantics. Verify the two
   modified matrices after layout conversion. Check logits and short greedy
   outputs against the same HF checkpoints on a trusted backend.
2. **Diagnostic mode:** install selective `cb_eval`; export dtype, shape,
   strides, token positions and arrays. First compare the eight recorded short
   contrast texts' residuals and MLP activations. Do not dump every tensor every
   decode step across Thunderbolt.
3. **Teacher forcing:** calculate next-token log-probabilities with the correct
   one-token offset and chunked vocabulary normalization. Keep dataset/problem
   boundaries and padding explicit. Validate against a reference.
4. **Candidate lifecycle:** compute edits on CPU, round to BF16, convert/reload a
   new model, and create fresh KV and recurrent state. Start with reloads rather
   than hot-patching weights inside a context with stale caches.
5. **Research protocol:** implement budgets, paired seeds, failures, independent
   holdouts, exact checkpoint hashes and append-safe logs. Validate the published
   edit before asking an editor to discover a new one.

New CUDA kernels would also need to stay within the shim's supported runtime
and BLAS subset. Existing GGML operations avoid much of that work. General
PyTorch/vLLM compatibility remains a separate, much larger project; it is not
required for this narrow engine.

## Remaining protocol validation

Use the checkout's preflight and single-client locking/run protocol when testing
real hardware. Require a real TinyGPU device and explicitly reject null-device
execution or CPU fallback in the actual inference worker. Validate operator
values, target-model outputs, long-context correctness, and stability on this
specific 4090/enclosure/macOS setup before publishing throughput estimates.

The initial source audit changed no live eGPU state. Subsequent authorized
hardware tests updated runtime logs and connection bookkeeping, as documented
in `HARDWARE_TEST.md`; original MacCUDA source and libraries remain unchanged.
The Mac benchmark in `../macos/BENCHMARK.md` measures the internal M4 Max GPU;
`BENCHMARK.md` measures the connected RTX 4090 through our native worker.
