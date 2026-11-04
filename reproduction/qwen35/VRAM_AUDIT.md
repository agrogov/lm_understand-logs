# vLLM VRAM audit, 2026-11-04

The completed CPU-limited RTX 5090 benchmark showed **no physical VRAM
overcommit** in its one-second telemetry. Both checkpoints completed all cases.

## Configuration and allocation

`benchmark_cuda.py` explicitly sets `gpu_memory_utilization=0.85`, BF16 weights,
`max_model_len=34816`, at most eight sequences, and 4,096 batched prefill tokens.
The launcher runs base and edited checkpoints sequentially in separate containers.
No explicit KV-size override or CPU/KV offloading is configured.

The completed 5090 base initialization in `remote-results-5090-cpu-limited/base.log`
reported:

| Item | GiB |
|---|---:|
| Device memory used by vLLM's budget calculation | 31.45 |
| Free at startup | 30.79 |
| Requested executor budget (85%) | 26.73 |
| Weights plus non-Torch consumed memory | 1.57 |
| Peak profiled activations | 0.58 |
| Actual CUDA graph pool | 0.05 |
| Allocated KV-cache budget | 24.58 |

vLLM suggested 24.38 GiB KV memory to fit its requested budget more closely.
Thus its profiling estimate exceeded the requested executor budget slightly
(about 0.20 GiB); **85% is not a strict driver-enforced allocation ceiling**.
The reported cache capacity was 2,051,545 tokens versus at most
`8 * 34816 = 278528` tokens in the configured requests. This is the hybrid
cache manager's reported capacity, not an independent derivation of every
attention/recurrent-state allocation.

## Recorded whole-device usage

The peak recorded allocation across both checkpoints was **28,125 MiB**
(27.47 GiB), leaving approximately **4 GiB** against the roughly 31.45 GiB
usable device memory. GPU telemetry was sampled once per second and cannot
resolve subsecond allocation peaks.

See [completed benchmark results](remote-benchmark-5090-cpu-limited-results.json)
for CPU, system RAM and GPU measurements. The difference between the driver's
total memory and used-plus-free memory is reserved memory; exclude it from
allocatable headroom.

vLLM documents that the utilization budget applies per instance, rather than
coordinating a global budget across processes, and that an explicit
`kv_cache_memory_bytes` overrides it. Neither concurrent benchmark instances nor
that override is present in this launcher. See the
[vLLM 0.33.0 memory arguments](https://docs.vllm.ai/en/v0.33.0/configuration/engine_args/#cacheconfig).
