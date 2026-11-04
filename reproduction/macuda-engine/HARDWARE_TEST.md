# RTX 4090 / MacCUDA hardware validation

Completed **2026-11-04** on the connected RTX 4090, using this M4 Max Mac.

**Result: the custom native engine successfully runs the BF16 base and published
edited models on the real eGPU and reads the required residual/MLP activations.**
This is a working forward-and-readback smoke test, not the full experiment harness.

## Hardware and checks

| Check | Result |
|---|---|
| GPU | NVIDIA PCI ID `0x2684`, AD102 / `sm_89`, RTX 4090 |
| Link | x4 at 8.0 GT/s, PCIe Gen3, through the connected Thunderbolt adapter |
| CUDA device memory reported | 24,258 MiB usable after firmware reservation |
| MacCUDA operator gate | **450/450 value-checked supported cases passed** |
| BF16 GGUF intervention tensors | Base and edited layer-15/19 down projections retain all four expected raw BF16 hashes |
| Real model forward execution | Both checkpoints passed; loader reports 26/26 offloadable layers on GPU, with the normal 24-layer Qwen trunk |
| Diagnostic prompt | First recorded productive contrast from `fwd_4.json.claimed`, 124 tokens |
| Residual captures | `l_out-15`, `l_out-19`, each `[1024,124,1,1]` in GGML order |
| MLP input captures | `ffn_swiglu-15`, `ffn_swiglu-19`, each `[3584,124,1,1]` |
| Device readback confirmation | All four selected tensors explicitly reported backend buffer **`CUDA0`** |
| CPU comparison | Both checkpoints: all 32 greedy continuation token IDs identical |
| Residual numerical agreement | Approximately 0.40–0.41% relative L2 difference |
| MLP numerical agreement | Approximately 0.54–0.55% relative L2 difference |
| Final-prompt logits | Approximately 0.31–0.32% relative L2 difference |
| Cleanup | Post-test preflight OK, no new DART fault, GPU lock free; successful processes unloaded GSP cleanly |

The numerical comparison uses CPU and CUDA backends of the **same GGUF model**.
It does not claim bitwise floating-point equality or agreement with every HF/MLX
operator. A 1% relative-L2 bound was used as an engineering smoke-test threshold;
both checkpoints passed. Validation is on one short contrast, not all prompts.

## Actual short-run timings

| Checkpoint | Model load | 124-token prefill including four captures | 32-token greedy decode | Decode rate |
|---|---:|---:|---:|---:|
| Base, final buffer-confirmed run | 1.365 s | 0.0737 s | 0.1286 s | **248.8 tokens/s** |
| Published edit | 1.402 s | 0.0736 s | 0.1303 s | **245.5 tokens/s** |

These are tiny, single-sequence greedy runs with a 512-token context, default
F16 KV caches, diagnostic capture during prefill, and no capture during decode.
They do **not** establish long-context, batched, or temperature-0.6 benchmark
throughput. They should not replace the measured Mac rates in the full-workload
runtime estimate or be compared as a matched speed test against the MLX runs.

A later [dedicated throughput benchmark](BENCHMARK.md) uses both checkpoints,
independent sequences, experiment sampling and the full 32,768-token output
cap. Use that report for runtime projections; the table above remains only
the original diagnostic smoke timing.

## What failed initially and how it was repaired

The stock Ada shim archive successfully handled the operator gate, but full BF16
model prefill stopped at:

```text
tinyblas_gemm_bf16_tc2s_tn not in the cubin
CUDA error: CUBLAS_STATUS_NOT_INITIALIZED
```

Its embedded `gemm.sm_86.cubin` lacked the BF16 tensor-core entry points used by
the current BLAS shim. Therefore a successful link or the initial operator gate
alone was insufficient to validate this model.

The fix is isolated in this reproduction folder:

1. Copy the existing `libtinycublas/gemm.cu` source into our ignored build folder.
2. Compile it to an **`sm_89` cubin** using the already-cached CUDA 13.0.3 Linux
   compiler image through Apple's `container` tool. No GPU is used for compilation.
3. Embed the rebuilt cubin in a native Mac object defining `gemm_cubin`.
4. Link that object before the existing `libtinycublas.a`, replacing the old
   embedding only inside our custom executable.
5. Re-run real model inference and compare the retrieved tensors against CPU.

**The original MacCUDA source files, archives and application binaries were not
rewritten.** Its existing user changes remain intact. Runtime logs and connection
bookkeeping were updated as part of the authorized hardware tests. The temporary
Apple compiler-container service was stopped afterward; the native inference
executable does not require that service.

The old connection record dated October was backed up and refreshed after the
user's reconnect, only after confirming a live link and no DART fault. No DART
fault was marked stale, no driver was installed, and no intentional GPU-fault
tests were run.

## Files and rerunning

All generated models, binaries, raw arrays and logs are under ignored `build/`.
The separate `converter/.venv` is managed by `uv` and does not alter the MLX venv.

```text
build/hardware/hardware-results.json           consolidated results
build/hardware/gguf-tensor-verification.json    intervention weight hashes
build/hardware/gpu-base-buffer-confirmed/      GPU captures, logits, tokens, metadata
build/hardware/gpu-edited-fixed/               edited GPU captures and tokens
build/hardware/cpu-base/                       CPU reference
build/hardware/cpu-edited/                     edited CPU reference
build/hardware/cpu-gpu-parity-*.json            numerical comparisons
build/hardware/logs/                          operator/model logs, including initial failure
build/bf16-gemm/sha256.txt                     copied source and rebuilt cubin hashes
```

From the repository root, build the custom executable after rebuilding the
missing kernels if the embedding object is not already present:

```bash
# Only needed when rebuilding the CUDA cubin; the image was already cached here.
container system start
bash reproduction/macuda-engine/build_bf16_gemm.sh
bash reproduction/macuda-engine/build_link_probe.sh \
  /Users/arogov/Downloads/vfio/macuda model_probe
```

For a hardware rerun, choose a **new output directory**:

```bash
bash reproduction/macuda-engine/hardware_run.sh probe \
  "$PWD/reproduction/macuda-engine/build/model_probe" \
  "$PWD/reproduction/macuda-engine/build/hardware/qwen35-base-bf16.gguf" \
  "$PWD/reproduction/macuda-engine/build/hardware/contrast-0.txt" \
  "$PWD/reproduction/macuda-engine/build/hardware/gpu-base-rerun"
```

The wrapper acquires the project GPU lock and uses the supplied MacCUDA
preflight/server runner. Do not launch a second GPU client concurrently. A
preflight failure must be inspected; do not automatically clear fault records.
The prototype refuses a missing/non-AD102 CUDA device and now requires the
captured activations themselves to be on `CUDA0`.

Compare with the saved CPU reference:

```bash
reproduction/macuda-engine/converter/.venv/bin/python \
  reproduction/macuda-engine/compare_probes.py \
  --gpu reproduction/macuda-engine/build/hardware/gpu-base-rerun \
  --cpu reproduction/macuda-engine/build/hardware/cpu-base \
  --out reproduction/macuda-engine/build/hardware/parity-base-rerun.json
```

For a fresh converter environment, use `uv sync --project
reproduction/macuda-engine/converter --python 3.12` with its cache inside that
folder. The local MacCUDA `llama.cpp/convert_hf_to_gguf.py` converts the already
downloaded HF checkpoints using `--outtype bf16`. No quantized replacement model
was downloaded for these tests.

## Remaining work for the full experiment

- Teacher-forced per-token log-probabilities and complete short-contrast batches.
- The full fixed sampling and paired-seed protocol, queues and pass-budget accounting.
- Bounded generation-time activation summaries and optional attention probes.
- Long-context, batched numerical/stability/performance tests on this exact stack.
- Independent confirmation and MATH/GSM evaluation, before claiming capability gains.

The hardware and custom-engine feasibility questions now have positive results;
the complete research controller remains to be implemented and validated.
