# Native Linux RTX 4090 and RTX 5090 benchmarks

Host: `root@192.168.31.52` (`nuc-13`), Linux x86-64, approximately 59 GiB RAM.
The image is built **on the remote Docker daemon**, with the repository's
`Dockerfile`, CUDA 13.0.2, `uv venv /opt/venv`, and hash-locked dependencies.
All Python inference runs inside that container. No local Docker build is used.

## Comparison of measured configurations

Aggregate output tokens/s for four sequences at a 32,768-new-token cap, including
prefill; generation-only projections use the same recorded token workload.

| Setup | Base tokens/s | Edited tokens/s | Projected generation time |
|---|---:|---:|---:|
| M4 Max / MLX | 421.8 | 287.5 | 278.2 h / 11.6 days |
| RTX 4090 / MacCUDA | 312.7 | 341.1 | 310.1 h / 12.9 days |
| RTX 4090 / Linux CUDA | 967.6 | 637.2 | 123.3 h / 5.1 days |
| RTX 5090 / Linux CUDA, CPU-limited compiled | 975.9 | 863.5 | 107.5 h / 4.5 days |

The completed 5090 run used a four-CPU quota, two math threads, one compilation
worker, and the host's 575 W limit. It is the fastest completed configuration
by this workload projection, about 12.8% less generation time than the 4090.
These are observed runner results with different sampled lengths and occupancy, not hardware limits
or end-to-end experiment durations.

## RTX 5090: completed CPU-limited benchmark

**Completed: all eight timing cases across both checkpoints.**
Both containers exited and the GPU returned to 2 MiB usage. Compact results:
[CPU-limited 5090 results](remote-benchmark-5090-cpu-limited-results.json).

| Checkpoint | Sequences | Output cap | Actual output tokens | Seconds | Tokens/s |
|---|---:|---:|---:|---:|---:|
| Base | 1 | 2,048 | 2,048 | 3.560 | 575.4 |
| Base | 4 | 4,096 | 14,597 | 10.179 | 1,434.1 |
| Base | 8 | 4,096 | 32,768 | 9.441 | 3,470.8 |
| Base | 4 | 32,768 | 73,657 | 75.472 | 975.9 |
| Edited | 1 | 2,048 | 2,048 | 3.610 | 567.3 |
| Edited | 4 | 4,096 | 16,384 | 10.381 | 1,578.3 |
| Edited | 8 | 4,096 | 30,877 | 9.514 | 3,245.4 |
| Edited | 4 | 32,768 | 61,455 | 71.166 | 863.5 |

The recorded workload projects to **64.56 base + 42.90 edited = 107.46 hours**
(4.48 days). This excludes teacher forcing, grading, editor time and startup.
Base long-case output lengths were `[5812,32768,2309,32768]`; edited lengths
were `[4788,19538,4361,32768]`. Different EOS/occupancy means these rates do
not isolate the CPU-limit effect or intrinsic hardware speed.

Resource telemetry: container CPU median **98.26%** and peak **223.84%**
(100% is one logical CPU), peak container RAM **5.075 GiB**, whole-host CPU
idle median **94%**, minimum **81%**, no observed swap-in/out, peak GPU memory
**28,125 MiB**. Container samples begin partway through base startup; host/GPU
samples cover the launcher run.

The benchmark uses the full compiled case list with these CPU resources:
Docker `--cpus=4`, affinity
`--cpuset-cpus=4-7` (four distinct physical cores on this host), two OpenMP,
MKL, OpenBLAS, NumExpr and Rayon threads, disabled tokenizer parallelism,
and one build/Inductor compilation worker. Docker quota and affinity were
verified with `docker inspect`. The GPU power limit was 575 W. Results are
under `results-5090-cpu-limited/`.
The launcher now defaults to four CPUs, two math threads, one compilation worker;
affinity is optional via `WSP_CPUSET` and must match the host topology.

Host `vmstat` logs and container CPU/RAM statistics accompany GPU telemetry.
Docker CPU percent counts 100% as one logical CPU, so the quota permits up to
400%; it does not mean four of the host's eight physical cores are always busy.
CPU limits can change throughput, so any completed numbers must be labeled
separately from the original 4090 configuration.

Reproduce the successful configuration on this host with its existing remote image:

```bash
WSP_IMAGE=qwen35-reproduction:20261104-5090 \
WSP_GPU=GPU-748da0db-f83b-2085-15a3-f726534efa44 WSP_EXPECTED_GPU=5090 \
WSP_CPU_LIMIT=4 WSP_CPUSET=4-7 WSP_CPU_THREADS=2 WSP_COMPILE_THREADS=1 \
WSP_RESULTS_DIR=/root/qwen35-benchmark-20261104/results-5090-cpu-limited-new \
bash /root/qwen35-benchmark-20261104/run_remote_benchmark.sh
```

The image was built on remote Docker as `qwen35-reproduction:20261104-5090`,
ID `sha256:50e1d394e1aff89dbac695d20183fa70320e5ccc07d5c4056a8dcd41078da986`.
Inference uses `enforce_eager=False`, torch compilation, CUDA graphs and BF16,
with the same package versions, prompts, seeds, sampling and case list as the
4090 benchmark. Select the card with `WSP_GPU` and `WSP_EXPECTED_GPU=5090`.

See the [VRAM allocation check](VRAM_AUDIT.md) and
[Blackwell target verification](BLACKWELL_AUDIT.md) for environment details.

## RTX 4090: completed results, 2026-11-04

The tested four-sequence configuration projects **123.3 generation hours (5.1 days)** for the recorded
core workload. The corresponding MLX projection is **278.2 hours (11.6 days)**;
the MacCUDA eGPU projection is **310.1 hours (12.9 days)**. That is about **2.26×
the throughput of the measured MLX configuration** for this weighted workload.
These compare complete implementations with differing sampled output lengths;
they do not isolate hardware speed. The 5090 results are reported separately above.

| Checkpoint | Sequences | Output cap/sequence | Actual output tokens | Wall incl. prefill | Aggregate tokens/s |
|---|---:|---:|---:|---:|---:|
| base | 1 | 2,048 | 2,048 | 4.73 s | 433.1 |
| base | 4 | 4,096 | 16,384 | 13.38 s | 1,224.7 |
| base | 8 | 4,096 | 32,217 | 12.78 s | 2,521.8 |
| edited | 1 | 2,048 | 2,048 | 4.73 s | 432.7 |
| edited | 4 | 4,096 | 14,309 | 13.09 s | 1,093.3 |
| edited | 8 | 4,096 | 31,589 | 12.73 s | 2,482.0 |
| base | 4 | 32,768 | 106,851 | 110.43 s | **967.6** |
| edited | 4 | 32,768 | 46,249 | 72.58 s | **637.2** |

The long base case produced lengths `[32768,32768,8547,32768]`; the edited case
produced `[7881,9208,2066,27094]`. All four edited answers ended naturally before
the cap; three base answers reached the cap. Timings include idle slots after
early finishes and the edited run's tail with one sequence active. No sequence
was forced past EOS. This is one long batch per checkpoint, without a confidence
interval. The eight-sequence cases test early context only; do not use their
2,500 tokens/s rates as a sustained full-context estimate.

| Original recorded work | Native Linux RTX 4090 generation hours |
|---|---:|
| Discovery generation | 50.3 |
| MATH-500, four samples/problem | 36.9 |
| GSM8K, two samples/problem | 36.0 |
| Discovery + both evaluation sets | **123.3** |

Calculation: `226823922 / 967.6078 / 3600 + 133357576 / 637.1903 / 3600`
= **65.12 base hours + 58.14 edited hours**. Teacher-forced probes, grading,
checkpoint IO, editor decisions, retries, and idle time remain additional.
Earlier candidate edits are approximated with the final edited checkpoint's
rate. Actual new runs can generate different token volumes.

Validated environment: NVIDIA driver **615.71.09**, vLLM **0.33.0**, PyTorch
**2.14.0**, Transformers **5.20.2**, CUDA **13.0.2** development base, Python
**3.12.3**, uv **0.9.5**, BF16, FlashAttention backend and FlashInfer sampling.
GPU telemetry peaked at **21,015 MiB allocated**, with 100% utilization observed
during generation; the existing **450 W** power limit was unchanged. CPU Torch
threads were set to eight, compiler jobs to four. This is not a tuned ceiling.
Only the 4090 was exposed to the benchmark containers; the 5090 was not tested.

Image: `qwen35-reproduction:20261104`, ID
`sha256:a6554437af836050cb2ca14d21ccb861e256ba2a0444e3e0f0e015291bca2511`.
The original CUDA base resolved to
`nvidia/cuda@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6`.
Both Docker jobs exited successfully; the benchmark containers were removed
and the 4090 returned to 1 MiB allocated.
The evidence includes checkpoints matching these full-file SHA-256 values:

```text
base:   04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
edited: fd1e42acdd118ac1b300d37500fd2885e75f5a09de3a33010d802bf7f737b553
```

See [compact results and provenance](remote-benchmark-results.json),
[prior MLX measurements](../macos/BENCHMARK.md), and
[prior MacCUDA measurements](../macuda-engine/BENCHMARK.md).

## Repeating the measurement

The dedicated remote directory is `/root/qwen35-benchmark-20261104`.
Its `full-context/reproduction/qwen35/` contains the Dockerfile, lock files,
benchmark and reproduction helpers. `data/` contains the already verified
base/edited BF16 checkpoints and matching practice prompts transferred from
this Mac. Preserve the original results; use a new results directory/output
names for another run.

Build remotely after copying those source files:

```bash
ssh root@192.168.31.52 \
  'cd /root/qwen35-benchmark-20261104/full-context && docker build \
   -f reproduction/qwen35/Dockerfile -t qwen35-reproduction:20261104 .'
```

The Dockerfile copies `benchmark_cuda.py` to `/opt/benchmark_cuda.py`. Run the
prepared launcher **on the remote host**:

```bash
ssh root@192.168.31.52 \
  'WSP_RESULTS_DIR=/root/qwen35-benchmark-20261104/results-rerun-01 \
   bash /root/qwen35-benchmark-20261104/run_remote_benchmark.sh'
```

The launcher exposes only RTX 4090 UUID
`GPU-2a5dea92-f35f-086a-776b-8c7531e9a802`. The benchmark also checks that exactly
one CUDA device is visible and it is a 4090. The 5090 is not used. Container
names are `qwen35-benchmark-base` and `qwen35-benchmark-edited`, sequentially.

Defaults are BF16 weights, automatic KV dtype matching the model, 34,816-token
context, maximum eight concurrent sequences, 4,096 scheduled tokens, 85% GPU
memory utilization, prefix caching disabled, and normal vLLM compile/CUDA-graph
execution. The engine is warmed with a separate 128-token case before timing.
Each checkpoint receives cases `1:2048,4:4096,8:4096,4:32768`. No replacement
requests are added when an answer stops. The four-sequence 32K-cap cases form
the workload projection, matching the earlier comparison's batch/cap choice.

The exact rendered prompts match the MacCUDA test and alternate the same
practice questions as MLX. Sampling is temperature 0.6, top-k 20, top-p 0.95,
presence penalty 1.5 over generated tokens, seeds `1729 + slot`, EOS respected.
Different backend RNG and numerical results mean different output lengths;
this compares implemented workloads, not token-identical kernel performance.

Results include prompt hashes, full checkpoint hashes, versions, all generated
token IDs, prompt/output counts, finish reasons, and wall time including prefill.
Engine startup, initial compilation, warm-up, and grading are excluded from
timed cases. Any further lazy shape/length-specific JIT during generation
remains included; these are observed runner rates, not a fully precompiled ceiling.
The launcher saves per-second 4090 utilization, memory, power and clocks to
`results/gpu-monitor.csv`, image metadata, package versions, and driver logs.
Completion is marked only after all requested cases finish.

After copying `results/base.json` and `results/edited.json` into local
`reproduction/qwen35/remote-results/`, calculate:

```bash
bash reproduction/macos/run.sh python \
  reproduction/qwen35/summarize_cuda_benchmark.py
```

The calculator requires completed 32K-cap results for both checkpoints and
uses the same recorded workload counts as the Mac reports. Raw output and
compiler caches can remain on the remote host; the compact summary is
`remote-benchmark-results.json`. These are generation-only projections, not
complete experiment durations or accuracy results. Teacher forcing, grading,
editor decisions, checkpoint IO, retries and idle time are additional.
