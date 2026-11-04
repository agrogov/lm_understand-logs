# Native macOS environment for Qwen3.5 weight-space experiments

Prepared and verified on **2026-11-04**, on this Apple M4 Max with **36 GiB
unified memory**, macOS 26.6.2, native arm64 Python 3.12.13. No Docker, CUDA,
Conda, or system-wide Python package installation is used.

## Installed and verified

The virtual environment is `.venv/` in this directory. `uv.lock` pins the
complete dependency resolution; subsequent setup uses `uv sync --frozen`.

| Component | Installed version |
|---|---|
| MLX / MLX Metal | 0.32.4 |
| MLX LM | 0.32.2 |
| Transformers | 5.22.1 |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| Safetensors | 0.8.0 |
| Hugging Face Hub | 2.4.0 |
| Datasets | 5.1.0 |
| math-verify | 0.9.0 |

Verified with an actual BF16 matrix multiplication on the **Metal GPU**, the
Qwen3.5 MLX implementation import, sampling API construction, a symbolic math
equivalence check, and a Safetensors round trip. The report is
`artifacts/environment-check.json`. Its short matmul timing includes warm-up;
it is **not** an LLM throughput benchmark.

MLX reports a recommended GPU working set of about **28.1 GiB** on this machine.
The remaining unified memory also serves macOS and other applications. Start
with single-sequence generation and small activation probes. No memory-limit
or system GPU-memory settings were changed.

The pinned base weights have now been downloaded, the published edit rebuilt
with the authors' BF16 hash verification, and both checkpoints converted to
unquantized MLX models. A bounded inference benchmark runs on practice questions;
no test benchmarks or training have been run. See [MacCUDA applicability](MACUDA.md)
for the external RTX 4090 option.

The CUDA/vLLM runner in `../qwen35/reproduce.py` has not been ported to Metal.
The native `benchmark.py` measures generation speed; a full Mac discovery run
still needs native activation instrumentation and a paired evaluation runner.
A throughput measurement is not a benchmark-accuracy reproduction.

## Measured inference and runtime

See [the completed benchmark report](BENCHMARK.md). The full-cap four-sequence
runs measured **421.8 aggregate tokens/s for the base model** and **287.5 for
its published edit**, including prefill and shrinking occupancy as answers end.
At those measured rates, the original recorded discovery plus MATH/GSM workload
projects to **278 generation hours (11.6 days)**, plus other work. The short
8-sequence rates (945.5 / 890.7 tokens/s) are not sustained full-context evidence.
The additional 8-sequence late-context probe was aborted and excluded.

## Run commands

From the repository root:

```bash
# Already completed; rerun to restore/check the locked environment.
bash reproduction/macos/setup.sh

# Recommended launcher: selects the venv and keeps caches inside this directory.
bash reproduction/macos/run.sh python reproduction/macos/doctor.py
bash reproduction/macos/run.sh mlx_lm.generate --help
bash reproduction/macos/run.sh mlx_lm.convert --help
```

The launcher's Python working directory stays wherever you invoked it. The
project and cache directories are resolved from the launcher's location.
You can use it from another directory with absolute paths.

For an interactive shell, activation is also supported:

```bash
source reproduction/macos/.venv/bin/activate
export HF_HOME="$PWD/reproduction/macos/.cache/huggingface"
python reproduction/macos/doctor.py
deactivate
```

The existing native Python 3.12 interpreter is reused; dependencies are isolated
in the local venv. If a compatible interpreter is absent on another machine,
`uv` can install one under this directory's `.python/` folder. No shell profile
was modified. Caches, downloaded assets, and the venv are locally gitignored.

```text
reproduction/macos/
  .venv/                         installed uv virtual environment
  uv.lock                        pinned dependency resolution
  pyproject.toml                 direct dependencies / Python requirement
  setup.sh                       create venv, sync locked dependencies, check GPU
  run.sh                         run inside the dedicated environment
  doctor.py                      environment / Metal validation
  .cache/uv/                     uv download cache
  .cache/huggingface/             future model and dataset cache
  .python/                       managed Python, only if needed
  artifacts/environment-check.json
```

## Recreate the downloaded base and the published edit

The download, reconstruction, and conversion commands below have been executed
successfully. Do not rerun reconstruction into an existing output directory:
it deliberately refuses to overwrite it. On a fresh environment these steps
download roughly 1.8 GB and retain several copies. Run from the repository root.
The final short CLI example is optional; the measured runs use `benchmark.py`.

```bash
export WSP_MAC_ARTIFACTS="$PWD/reproduction/macos/artifacts"

bash reproduction/macos/run.sh hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --local-dir "$WSP_MAC_ARTIFACTS/base-hf"

# CPU-only subcommand: works without the CUDA/Torch generation dependencies.
REPRO_REPO="$PWD" bash reproduction/macos/run.sh \
  python reproduction/qwen35/reproduce.py rebuild \
  --base "$WSP_MAC_ARTIFACTS/base-hf" \
  --out "$WSP_MAC_ARTIFACTS/published-edit-hf"

# Convert each HF checkpoint to an unquantized BF16 text model for MLX.
bash reproduction/macos/run.sh mlx_lm.convert \
  --hf-path "$WSP_MAC_ARTIFACTS/base-hf" \
  --mlx-path "$WSP_MAC_ARTIFACTS/base-mlx-bf16" --dtype bfloat16
bash reproduction/macos/run.sh mlx_lm.convert \
  --hf-path "$WSP_MAC_ARTIFACTS/published-edit-hf" \
  --mlx-path "$WSP_MAC_ARTIFACTS/published-edit-mlx-bf16" --dtype bfloat16

# Small inference smoke test; not the original benchmark protocol.
bash reproduction/macos/run.sh mlx_lm.generate \
  --model "$WSP_MAC_ARTIFACTS/published-edit-mlx-bf16" \
  --prompt 'What is 2 + 2?' --max-tokens 128 --seed 0 \
  --chat-template-config '{"enable_thinking":true}'
```

Keep the original HF checkpoints: hash verification applies to those tensors
**before** MLX sanitizes names/layouts during conversion. Do not use 4-bit
quantization for a faithful BF16 weight-edit comparison. Do not change the
tokenizer or chat template to stop loops instead of testing the weight edit.

The CLI smoke test above is deliberately short and is not a benchmark score.
For the experiment, a native Python runner must preserve thinking mode,
32,768 output tokens, temperature 0.6, top-p 0.95, top-k 20, and presence penalty
1.5 over the appropriate token history. The installed generation CLI does not
expose a presence-penalty option; use `mlx_lm.sample_utils.make_logits_processors`
in that runner and verify its history-window semantics. It must also preserve
problem/sample pairing, strict boxed-answer grading, and independent holdouts.
Numerical output and RNG trajectories can differ from CUDA even with the same
BF16 source weights.

For the experiment design and expected results, see
[`../../REPRODUCE_QWEN35_RTX.md`](../../REPRODUCE_QWEN35_RTX.md). Its Docker/CUDA
commands remain specific to the NVIDIA machine.

References: [MLX LM](https://github.com/ml-explore/mlx-lm),
[Qwen3.5 implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py),
[published edited model](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).
