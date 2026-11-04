# Connected RTX 4090 benchmark

Measured on 2026-11-04: this M4 Max Mac, 36 GiB RAM, RTX 4090 AD102 over
x4 PCIe Gen3 (8 GT/s), using the custom MacCUDA/llama.cpp executable.
An extra inference run confirmed x4 at 8.0 GT/s through `system_profiler` while
the model was active; the idle/cold card reports Gen1 at 2.5 GT/s. The live
snapshot is retained in `build/hardware/benchmark-live-pci.txt`.
See [hardware validation](HARDWARE_TEST.md) for device checks, the isolated
BF16 GEMM repair, conversion hashes and short CPU/GPU numerical comparisons.

## Result and where to run

**Use the internal M4 Max with MLX for the current measured Mac workload.**
The connected 4090 through this MacCUDA runner did not improve the combined
generation-time projection: **310.1 hours (12.9 days)** versus **278.2 hours
(11.6 days)** on MLX, about **11.5% longer**. This is a point estimate from
one long batch per checkpoint, not evidence that the M4 Max hardware is
intrinsically faster than an RTX 4090.

**Later native Linux follow-up:** the remote RTX 4090 has now run both
checkpoints in the full pinned Docker/uv environment, built on the remote host.
It measured **967.6 base / 637.2 edited tokens/s**, projecting **123.3 generation
hours (5.1 days)**. Prefer that remote 4090 for the current reproduction; the
MLX preference above applies when choosing between the two tested Mac backends.
See [native Linux results and rerun instructions](../qwen35/REMOTE_BENCHMARK.md).
The 5090 remains unmeasured. The eGPU remains useful for native-engine development
and diagnostic readback.

All rates below are aggregate generated tokens/s, including prompt prefill.

| Checkpoint | Sequences | Output cap/sequence | Actual output tokens | Wall time | RTX 4090 tokens/s | Prior M4 Max tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 2,048 | 2,048 | 7.77 s | 263.6 | 174.6 |
| base | 4 | 4,096 | 16,286 | 28.26 s | 576.3 | 563.5 |
| base | 8 | 4,096 | 32,768 | 34.07 s | 961.9 | 945.5 |
| edited | 1 | 2,048 | 2,048 | 7.93 s | 258.3 | 174.5 |
| edited | 4 | 4,096 | 14,164 | 34.87 s | 406.2 | 568.4 |
| edited | 8 | 4,096 | 27,855 | 46.11 s | 604.1 | 890.7 |
| base | 4 | 32,768 | 105,890 | 338.64 s | **312.7** | **421.8** |
| edited | 4 | 32,768 | 85,139 | 249.62 s | **341.1** | **287.5** |

The 4090 base long case ended at lengths `[32768,32768,7586,32768]`; the
edited case at `[1073,32768,18530,32768]`. MLX generated different lengths
(base 131,072 total, edited 66,089), so occupancy and generated content are
part of the observed difference. Short-case timings also include different
EOS occupancy; they are not controlled kernel-speed comparisons.

| Recorded workload | RTX 4090 / MacCUDA projected hours | M4 Max / MLX projected hours |
|---|---:|---:|
| Discovery generation | 125.0 | 113.5 |
| MATH-500, four samples/problem | 90.7 | 83.2 |
| GSM8K, two samples/problem | 94.4 | 81.5 |
| Both evaluation sets | 185.1 | 164.7 |
| Discovery + both evaluation sets | **310.1** | **278.2** |

Both four-case benchmark jobs completed with exit code 0. Post-test preflight
reported no DART error, unchanged driver instance, and a free GPU lock. This
does not establish stability over a two-week run.

## Method

`inference_bench.cpp` loads unquantized BF16 weights on CUDA0 and explicitly
requires the real AD102 device and TinyGPU socket. Every case creates fresh
KV/recurrent state. Both K and V caches are BF16; flash attention is enabled.
All 26 offloadable layers are placed on the eGPU. No diagnostic callbacks run
during this throughput test. CPU sampling and GPU-to-host logits transfers
are included in wall time, as is prompt prefill. Model load and context
allocation are excluded.

The cases are `1:2048,4:4096,8:4096,4:32768`. Four independent sequences
with a 32,768-new-token cap form the sustained comparison. EOS stops a sequence;
finished slots are not refilled. The cap applies separately to each sequence.
The benchmark retains every output token ID and writes a completion marker
only after every requested case finishes. Each case has a 20-minute safety
limit; interrupted cases cannot enter the calculator.

Prompts alternate the same recorded GSM8K/MATH practice questions as the
[MLX measurements](../macos/BENCHMARK.md). Sampling uses temperature 0.6,
top-k 20, top-p 0.95, and presence penalty 1.5 over generated tokens only.
The C++ sampler uses independent seeds `1729 + slot`. Its RNG is different
from MLX's, so output lengths and active-slot occupancy differ. The first
GGUF prompt tokenizes to 100 tokens versus 99 in the prior MLX case. These
are comparable experiment workloads, not token-identical backend comparisons.

## Rerun

Use the existing dedicated `uv` converter environment and prepared BF16 GGUFs
from [the hardware setup](HARDWARE_TEST.md). From the repository root:

```bash
bash reproduction/macuda-engine/run_benchmark.sh rerun-01
```

The script builds the native worker, prepares prompts, acquires the MacCUDA
single-client lock, and runs both checkpoints sequentially. Use a new tag to
preserve previous results. No Docker runtime is needed on the Mac. The earlier
kernel compilation uses the cached Linux CUDA compiler image; the native
benchmark itself does not run in a container.

For the original completed outputs, recalculate with:

```bash
reproduction/macuda-engine/converter/.venv/bin/python \
  reproduction/macuda-engine/summarize_benchmark.py
```

Raw `build/benchmark-{base,edited}/cases.jsonl` files contain timings and token
IDs; `completed.txt` gates calculation. The compact `benchmark-results.json`
retains timing/length summaries, checkpoint and executable SHA-256 hashes,
and the original recorded workload counts. Model/binary/raw-output files are
gitignored.

Original driver logs, also copied into `build/hardware/logs/`:

- Base: `shim-probe-20261104-011802.log`.
- Edited: `shim-probe-20261104-012532.log`.

## Interpretation

This measures the implemented MacCUDA runner, including its sampling and
transfer costs. It does not establish the RTX 4090's native Linux speed,
optimal continuous batching, full-context numerical correctness, model accuracy,
or an end-to-end discovery run. One long batch per checkpoint provides a
configuration-specific estimate; there is no statistical confidence interval.
The full discovery controller and teacher-forced protocol still need integration.

Generation time is calculated separately for each model:

`hours = base_recorded_tokens / base_tokens_per_second / 3600 + edited_recorded_tokens / edited_tokens_per_second / 3600`

The core workload contains 226,823,922 base and 133,357,576 edited generated
tokens. Teacher forcing, grading, editor decisions, checkpoint IO, retries and
idle time are additional. Rates for intermediate edits are approximated by
the final edited checkpoint; actual new runs may generate different lengths.
