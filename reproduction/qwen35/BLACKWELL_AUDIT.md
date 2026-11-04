# Blackwell target audit: RTX 5090 image

Completed 2026-11-04 against remote image
`qwen35-reproduction:20261104-5090`, ID
`sha256:50e1d394e1aff89dbac695d20183fa70320e5ccc07d5c4056a8dcd41078da986`.

**The inspected core binaries and retained JIT artifacts include Blackwell
targets.** This establishes target inclusion; correctness of each individual kernel
requires runtime validation.

## Static findings

| Component | Observed targets/evidence |
|---|---|
| PyTorch 2.14.0, CUDA 13.0 | Compiled flags include `sm_120` |
| `libtorch_cuda.so` | Native cubins include `sm_120` and `sm_120a` |
| `libtorch_cuda_linalg.so` | Native cubins include `sm_120` |
| vLLM 0.33.0 `_C_stable_libtorch.abi3.so` | Native cubins include `sm_120` |
| vLLM `_flashkda_C` and `_moe_C_stable_libtorch` | Native cubins include `sm_120` |
| FlashAttention 2 `_vllm_fa2_C` | Native `sm_80` cubins plus `sm_80` PTX |
| Retained Triton metadata | **123** CUDA target records, all architecture **120** |
| Retained generated PTX | **123** files with target **`sm_120a`** |
| Installed compiler | NVCC **13.0.88**, supported target list includes `sm_120` |

The compiler/package versions were Torch 2.14.0, vLLM 0.33.0,
FlashInfer 0.7.2.post1, and Triton 3.8.0. PyTorch's compiled flag string was:

```text
sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
```

FlashAttention 2's older native cubin target does not by itself establish a
compatibility error: its accompanying generic PTX can be JIT-compiled by the
driver for a newer architecture. NVIDIA documents that PTX is forward-compatible,
while native cubins have stricter compatibility rules. Architecture-specific
targets such as `sm_90a` are different; an optional library containing only those
targets must have an appropriate platform dispatch guard.
See [NVIDIA's Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/12.9.0/blackwell-compatibility-guide/index.html).

The audit lists all vLLM extension libraries, including optional architecture-
specific components. A library without `sm_120` is not automatically a defect in
the dense BF16 Qwen3.5 path: it may be an inactive backend. Conversely, seeing
`sm_120` somewhere in a library does not prove every exported kernel supports it.

## No GPU execution

The package inspection used **`--runtime=runc`**, no `--gpus`, no network, a
read-only image filesystem, `NVIDIA_VISIBLE_DEVICES=void` and empty
`CUDA_VISIBLE_DEVICES`. It read compiled-in PyTorch flags and used `cuobjdump`
to list cubin/PTX targets. **`torch.cuda.is_initialized()` remained false.**
No `torch.cuda.is_available()` or other device query was needed.

The JIT audit read JSON metadata and PTX text under
`results-5090-retry/vllm-cache/`. Cached Python/pickle code was not imported or
executed. No benchmark or GPU kernel was launched during either inspection.

## Limits and evidence

Runtime-generated FlashInfer code outside the retained cache was not covered
by this inspection. Target inclusion verifies architecture availability for the
inspected binaries and artifacts.

- [Compact audit results](blackwell-audit-results.json).
- Raw local package audit: `remote-results-5090-retry/architecture-audit.json`.
- Raw local JIT target metadata: `remote-results-5090-retry/jit-targets.json`.
- Remote originals: `/root/qwen35-benchmark-20261104/architecture-audit/`.
- [Benchmark results](REMOTE_BENCHMARK.md).

## Repeat the package audit without GPU access

Copy `audit_cuda_targets.py` to the dedicated remote folder. Then execute on the
remote host (no image rebuild is required):

```bash
docker run --rm --runtime=runc --network=none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /root/qwen35-benchmark-20261104/audit_cuda_targets.py:/audit.py:ro \
  --entrypoint /opt/venv/bin/python \
  qwen35-reproduction:20261104-5090 /audit.py
```
