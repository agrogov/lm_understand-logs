# M4 Max measurements and runtime estimate

Completed 2026-11-04 on this Apple M4 Max, 36 GiB unified memory, macOS 26.6.2.

**Measured four-sequence projection: about 278 generation hours (11.6 days) for
replaying the original discovery workload plus both models on MATH-500 and GSM8K.**
This is a configuration-specific extrapolation, not a completed experiment or a
hardware speed limit. Teacher-forced probes, grading, checkpoint IO, editor
decisions, retries, and idle time are additional.

## What was actually measured

The pinned base was downloaded, the published edit rebuilt and hash-verified,
and both converted to unquantized MLX models. The two intervention matrices
also match the original BF16 hashes after conversion. MLX keeps 302 parameter
arrays in BF16 and 18 recurrent log-parameter arrays in FP32 as part of its
normal model implementation. No quantization was used.

Generation used the original thinking template and practice questions, temperature
0.6, top-k 20, top-p 0.95, and presence penalty 1.5 over generated tokens only.
The custom sampler applies temperature before top-k/top-p. Short runs are
calibration, not accuracy measurements. One long batch per checkpoint was run
with the full 32,768-new-token cap. Early EOS was respected; no output was forced
to continue for timing. Warm-up cases are excluded below.

| Checkpoint | Sequences | Output cap/sequence | Actual output tokens | Wall time incl. prefill | Aggregate tokens/s | Peak MLX allocation (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| base | 1 | 2,048 | 2,048 | 11.73 s | 174.6 | 1.56 |
| base | 4 | 4,096 | 16,384 | 29.08 s | 563.5 | 1.99 |
| base | 8 | 4,096 | 32,768 | 34.66 s | 945.5 | 2.48 |
| edited | 1 | 2,048 | 2,048 | 11.73 s | 174.5 | 1.58 |
| edited | 4 | 4,096 | 16,384 | 28.83 s | 568.4 | 2.07 |
| edited | 8 | 4,096 | 30,382 | 34.11 s | 890.7 | 2.48 |
| base | 4 | 32,768 | 131,072 | 310.73 s | 421.8 | 3.29 |
| edited | 4 | 32,768 | 66,089 | 229.90 s | 287.5 | 2.22 |

The base long run reached the cap on all four answers. The edited run completed
against that same cap, but three answers finished early and one reached the cap.
Its 287.5 tokens/s therefore includes a tail with only one active sequence.
Continuous batching with replacement work may improve utilization. The short
eight-sequence result is not evidence of sustained eight-sequence 32K throughput.

Peak MLX allocation is not total process or whole-system memory. Thermals and
other applications were not controlled, and long cases were not repeated, so
no statistical confidence interval is claimed.

## Calculation

For each checkpoint separately:

`hours = recorded_output_tokens / measured_aggregate_tokens_per_second / 3600`

The rates used are **421.8147 tokens/s for base** and **287.4624 for edited**.
The earlier 100-hour number assumed 1,000 aggregate tokens/s; it was not measured.

| Original recorded work | Base output tokens | Edited output tokens | Projected generation hours |
|---|---:|---:|---:|
| Discovery (34 generation + 16 teacher-forced passes) | 87,823,584 | 57,644,704 | 113.5 |
| MATH-500, 4 samples/problem | 61,736,869 | 44,074,295 | 83.2 |
| GSM8K, 2 samples/problem | 77,263,469 | 31,638,577 | 81.5 |
| Both evaluation sets | 139,000,338 | 75,712,872 | 164.7 |
| Discovery + both evaluation sets | 226,823,922 | 133,357,576 | 278.2 |

Only generated tokens enter this calculation; the 16 teacher-forced discovery
passes require additional inference time. Discovery totals come from successful
`exp8_math/runs/p50c/requests/*.done.json`; evaluation totals come from
`eval/{unedited,p50c}/merged.json`, multiplying mean length by problem/sample counts.
Rates for earlier candidate edits are approximated with the final edited model.
Actual token volumes on MLX can differ from those recorded on CUDA.

Using the short eight-sequence rates instead would predict **108.2 hours**
for the same token volume. That is an **optimistic short-context scenario**,
not an established full-context estimate. Neither 108 nor 278 hours is a
measured end-to-end job duration. For the tested fixed-batch implementation,
budget roughly **12 days of generation plus other work**; tune and remeasure
before treating continuous eight-sequence execution as a faster alternative.

## Additional eight-sequence probe: excluded

An extra calibration prefilling recorded 28,672-token continuations was stopped
after progressive slowdown (roughly 321 down to 68 tokens/s in observed windows).
Its cause was not established. It did not finish either a valid two-model
comparison or the intended full decode window. **It contributes no rate to the
estimate.** This is recorded in `artifacts/benchmark-late-aborted.json`. The
corresponding experimental script is not a recommended benchmark entry point.

## Evidence and rerunning

Local raw artifacts (gitignored, retained on this machine):

- `artifacts/benchmark-short.json`: timings, prompts identified by practice IDs, token IDs and responses.
- `artifacts/benchmark-long.json`: full-cap four-sequence measurements, progress windows and checkpoint hashes.
- `artifacts/tensor-verification.json`: BF16 hashes of both intervention matrices, both checkpoints.
- `artifacts/runtime-estimate.json`: source token counts and extrapolation arithmetic.
- `artifacts/environment-check.json`: hardware and dependency checks.

From the repository root, using new output names to preserve this evidence:

```bash
bash reproduction/macos/run.sh python reproduction/macos/benchmark.py \
  --cases 1:2048,4:4096,8:4096 --out reproduction/macos/artifacts/benchmark-short-rerun.json
bash reproduction/macos/run.sh python reproduction/macos/benchmark.py \
  --cases 4:32768 --out reproduction/macos/artifacts/benchmark-long-rerun.json
bash reproduction/macos/run.sh python reproduction/macos/verify_conversion.py
bash reproduction/macos/run.sh python reproduction/macos/estimate_runtime.py \
  --benchmark reproduction/macos/artifacts/benchmark-long-rerun.json \
  --short reproduction/macos/artifacts/benchmark-short-rerun.json \
  --out reproduction/macos/artifacts/runtime-estimate-rerun.json
```

No model-accuracy benchmark or full discovery run was performed. These MLX
measurements use the internal M4 Max GPU. The connected RTX 4090 subsequently
completed a [separate full-cap throughput benchmark](../macuda-engine/BENCHMARK.md)
with experiment sampling: **312.7 base / 341.1 edited aggregate tokens/s**.
Using the same recorded token counts projects **310.1 hours** on that MacCUDA
runner, versus **278.2 hours** here. Prefer MLX for the current measured Mac
workload; differing RNG/output lengths and fixed-batch occupancy limit the
comparison. A later [native Linux RTX 4090 test](../qwen35/REMOTE_BENCHMARK.md),
using the full Docker/uv image built remotely, measured **967.6 base / 637.2
edited tokens/s**, projecting **123.3 hours (5.1 days)**. Prefer that remote
4090 over either tested Mac backend for the current reproduction. See also
[4090 activation/numerical validation](../macuda-engine/HARDWARE_TEST.md).
See [environment](README.md) and [MacCUDA applicability](MACUDA.md).
