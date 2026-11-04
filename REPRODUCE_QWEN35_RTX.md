# Reproducing Qwen3.5 weight-space programming on RTX 5090 / RTX 4090

Prepared 2026-11-03 for Linux x86-64, 64–128 GB system RAM, a desktop RTX 5090
(32 GB), and an optional desktop RTX 4090 (24 GB).

## Decision

**The RTX 5090 is sufficient by itself. The RTX 4090 is also sufficient by
itself at a smaller inference batch size.** Use BF16 weights and cache. Start
with eight concurrent sequences on the 5090 or four on the 4090. You do not
need to split this model across GPUs. Use the second card for independent jobs.

This experiment does **not train or fine-tune** Qwen. It diagnoses a failure,
computes direct weight edits in float32, and evaluates the edited BF16 model.
No optimizer, gradients, LoRA, or training data fitting is involved. The entire
Python experiment environment below lives in a **`uv venv` at `/opt/venv`
inside the Docker image**, including CUDA PyTorch, vLLM, Transformers, and grading.
Only the driver, Docker Engine, and NVIDIA Container Toolkit live on the host.

### What can and cannot be reproduced exactly

There are three different deliverables:

1. **Exact checkpoint reconstruction:** apply the published p50c diff to the
   pinned base. The supplied author script verifies the base and edited tensor
   hashes. This reproduces the evaluated weight values exactly.
2. **Method reproduction:** diagnose loops, collect contrasts, derive directions,
   select neurons, test doses and controls, freeze an edit, and confirm it on
   separate problems. The helpers here support this process with a human making
   the experimental decisions. They reconstruct the central mechanism from the
   released requests and code, rather than merely downloading the final model.
3. **The original autonomous Astra conversation:** the authors did not release
   the complete editor/kernel/forward-server harness. A bit-for-bit replay of
   its trajectory is not provided here. Recreating that agent experiment also
   needs access to the external frontier editor; these GPUs run the **0.8B target**,
   not the proprietary editor. Do not expect identical sampled generations,
   directions, neuron lists, or scores across backends, GPUs, and scheduler settings.

The helper uses vLLM for generation and Transformers/SDPA for short activation
probes. Recomputed candidate weights are **not claimed to be bit-exact** to the
authors' edit. Keep exact reconstruction and new discovery in separate directories.
The helpers have been statically checked and the Linux dependency lock resolved;
the image and CUDA execution have not been tested on your hardware.

## 1. Memory calculation

The pinned text config has width 1,024, MLP width 3,584, 24 layers, six full-attention
layers, two KV heads per full-attention layer, and head dimension 256. The other
18 layers use recurrent Gated DeltaNet state, not a full token-length KV cache.

| Allocation | Calculation / estimate |
|---|---|
| Target model weights | 0.8B × 2 bytes ≈ 1.49 GiB for text weights; allow about 2 GiB including the full released model |
| Full-attention KV cache per token per sequence | 6 layers × 2 (K,V) × 2 KV heads × 256 × 2 BF16 bytes = 12,288 bytes = 12 KiB |
| KV at 32,768 tokens | 384 MiB per sequence |
| KV at configured 34,816-token context | 408 MiB = 0.3984 GiB per sequence |
| Recurrent matrix state, rough lower-order term | 18 × 16 heads × 128 × 128 × 4 FP32 bytes ≈ 18 MiB per sequence, plus convolution/cache implementation overhead |
| Eight sequences at maximum context | About 3.19 GiB full-attention KV + 0.14 GiB recurrent matrix state |
| Runtime, temporary buffers, kernels, allocator | Budget another 4–8 GiB initially; actual use depends on the engine/backend |
| Planning total: one model, eight sequences | Roughly 10–15 GiB; this is an allocation estimate, not a measured peak |

With `gpu_memory_utilization=0.85`, vLLM may **reserve** roughly 27.2 GiB on the
5090 or 20.4 GiB on the 4090 for its weights, buffers, and cache pool. This is
not the same as the active tensors requiring that much. Inspect engine startup
logs and `nvidia-smi`; increase concurrency only after the long-context smoke test.

The main OOM traps are activation collection and vocabulary logits:

- A BF16 `[32768, 248320]` vocabulary-logit tensor alone is **15.16 GiB**;
  converting it to FP32 takes **30.31 GiB**. Do not materialize that for long texts.
- Saving all 24 MLP activation tensors plus 25 residual tensors at 32K tokens
  takes about **6.81 GiB per sequence**, even before attention and logits.
- A teacher-forced pass's allowance of 524,288 total tokens is a *logical work
  budget*, not an instruction to batch them all in VRAM. Microbatch and stream.
- Full attention maps scale quadratically. The original tool limited these to
  two texts of at most 2,048 tokens. Keep that restriction.

The supplied `probe` command handles only short contrasts (maximum 2,048 tokens),
runs one text at a time, copies activations to CPU, and computes log-probability
normalization in chunks. It does not implement the original long-generation
activation-averaging server. For the initial diagnosis, generated texts plus
the subsequent contrast probe are sufficient for the reconstructed method.

Use the CPU for weight arithmetic: each selected down-projection matrix is only
`1024 × 3584 × 4 = 14 MiB` in float32. System RAM of 64 GB is adequate with this
streamed workflow; 128 GB gives more room for retaining many probe results. The
original sandbox's 96 GB was a ceiling, not evidence that every operation needs it.
On a 64 GB machine, cap a replacement editor kernel below about 40 GB rather than
copying the original 96 GB limit.

Allow **100 GB free SSD for the core run**, including the CUDA image/build cache,
model snapshots, several candidate checkpoints, and generated text. Prefer
200 GB if retaining many candidates or broad activation dumps. Store artifacts
on disk under `/data`, never inside the disposable container layer.

Sources: [5090 specs](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/),
[4090 specs](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/),
[pinned model config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json),
[original experiment prompt](exp8_math/runs/p50c/turns/t001/prompt.txt).

## 2. Runtime calculation

GPU fit is not the main cost: the base model repeatedly generates close to its
32,768-token limit. Counts below come from the released logs, not GPU FLOPS.

| Workload | Approximate output tokens |
|---|---:|
| Successful 50-pass discovery run: 34 generation + 16 teacher-forced passes | 145,468,288 generated tokens across 5,984 answers |
| MATH-500: both models, 500 problems × 4 samples each | 105,811,164 |
| GSM8K: both models, 1,319 problems × 2 samples each | 108,902,046 |
| Discovery + those two benchmarks | **360,181,498** |
| All eight benchmarks reported for Qwen, both models, excluding AIME extras | 553,626,282 for evaluation alone |

Discovery tokens are estimated as `generations × mean_tokens` from **successful**
`requests/*.done.json`, summing original and edited separately. The request
directory also contains failed/retried requests; counting all `.claimed` files
would incorrectly give 69 passes rather than the 50 successful ones. Evaluation
counts use each `eval/*/merged.json` mean length and the published sample counts.

At sustained **aggregate output throughput** R across all active sequences:

`hours = output_tokens / (R × 3600)`

| Assumed measured R | Discovery only | MATH + GSM evaluation, both | Combined |
|---:|---:|---:|---:|
| 1,000 output tokens/s | 40.4 h | 59.6 h | 100.1 h |
| 2,500 output tokens/s | 16.2 h | 23.9 h | 40.0 h |
| 5,000 output tokens/s | 8.1 h | 11.9 h | 20.0 h |

**These are throughput scenarios, not RTX benchmarks or promised speeds.** Add
prefill, grading, startup/compilation, teacher-forced probes, candidate creation,
and human/editor reasoning time. New experiments may use fewer/more tokens. The
original run's 20.3 h elapsed included conversation overhead and a 4.7 h outage.

For context, the model card reports a full edited MATH-500 rerun of 43.4M output
tokens in 51 minutes on an H200: about 14,183 aggregate output tokens/s. Do not
transfer that rate to a 5090 using a simple FLOPS or memory-bandwidth ratio.
Measure your own long-output workload in section 5.

Source: [public checkpoint and independent rerun](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming),
[local final run notes](exp8_math/runs/p50c/work_notes/notes.md),
[evaluation protocol](exp8_math/README.md).

### Measured Mac and remote Linux configurations (2026-11-04)

The M4 Max / 36 GiB machine has now run real BF16 base and hash-verified edited
models through MLX, including generation at the full 32,768-token cap. The
measured rates and generation-time extrapolation are in
[the Mac benchmark report](reproduction/macos/BENCHMARK.md). The 100-hour row
above remains an **assumed 1,000 aggregate tokens/s scenario**, not a hardware
measurement. Use the Mac report for the measured four-sequence configuration;
short eight-sequence rates must not be substituted for sustained full-context
throughput without testing that configuration.

The connected **RTX 4090 eGPU has also now been benchmarked** through the custom
MacCUDA engine, including both BF16 checkpoints at the full 32,768-token cap.
Its four-sequence rates were **312.7 base / 341.1 edited tokens/s**, projecting
**310.1 generation hours (12.9 days)** for the recorded core workload versus
**278.2 hours (11.6 days)** on MLX. With these implementations, prefer MLX for
bulk work on the Mac. See [eGPU measurements, caveats and rerun commands](reproduction/macuda-engine/BENCHMARK.md).
The comparison includes different sampled outputs and EOS occupancy; it is not
a hardware speed limit.

The **remote Linux RTX 4090 at `192.168.31.52` has now been tested** with the
full pinned Docker/uv image, built on that host. It measured **967.6 base /
637.2 edited aggregate tokens/s** in the four-sequence 32K-cap cases, projecting
**123.3 generation hours (5.1 days)** for the same recorded workload. **Use that
remote 4090 for the current reproduction.** Both checkpoint checksums match.
Only the 4090 was exposed to the benchmark; the RTX 5090 remains unmeasured.
See [native Linux results, environment and remote rerun commands](reproduction/qwen35/REMOTE_BENCHMARK.md).
This is one batch per checkpoint, with different output lengths across engines;
teacher forcing, grading and editor time are additional. The earlier 100-hour
scenario was an assumption; the measured native-CUDA point estimate is 123 hours.

The [native macOS environment](reproduction/macos/README.md) uses its own locked
`uv venv`, not this CUDA image. Read the [MacCUDA / RTX 4090 applicability note](reproduction/macos/MACUDA.md)
before treating an external NVIDIA card as a replacement backend. This
experiment's GPU work is all forward inference, including activation probes;
there is no gradient-based training.

## 3. Build the Docker + uv environment

Run host commands from this repository's root. No host Python environment is
needed. Use a recent Linux NVIDIA driver compatible with **CUDA 13.0** (R580 or
newer, with the vendor's compatible patch release for your distribution/cards).
The RTX 5090 requires a Blackwell-compatible stack; old CUDA 12.1/12.4 images
are inappropriate. Driver compatibility must be checked on the actual host.

Install Docker Engine and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
using their distribution-specific instructions. After installation, configure
the runtime once:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```

Build the image:

```bash
docker build --platform linux/amd64 \
  -f reproduction/qwen35/Dockerfile -t qwen35-wsp:2026-11-03 .
```

Included files:

- `reproduction/qwen35/Dockerfile`: CUDA 13 developer image, Python 3.12,
  explicit `uv venv /opt/venv`, all packages installed there.
- `requirements.in`: direct dependencies; vLLM 0.33.0, Transformers 5.20.2,
  PyTorch 2.14.0, and math-verify 0.9.0.
- `requirements.lock`: hash-locked Linux x86-64 / Python 3.12 dependency resolution.
- `reproduce.py`: hardware check, asset preparation, exact rebuild, generation,
  probes, candidate construction, transfer comparison, and paired statistics.

The vLLM and Transformers versions match the public checkpoint rerun. The
published vLLM package pins PyTorch 2.14.0; that PyTorch package depends on CUDA
13.0 libraries. Hence this image deliberately uses CUDA 13, rather than combining
the current package pins with an old CUDA image. Let the lock install the matching
Torch/CUDA packages; do not independently replace Torch after installing vLLM.

After a successful build, save provenance outside the image:

```bash
mkdir -p reproduction-output
docker image inspect qwen35-wsp:2026-11-03 > reproduction-output/image-inspect.json
docker run --rm --entrypoint uv qwen35-wsp:2026-11-03 \
  pip freeze --python /opt/venv/bin/python > reproduction-output/installed.txt
git rev-parse HEAD > reproduction-output/source-commit.txt
git diff --binary > reproduction-output/local-changes.patch
```

The dependency lock is pinned, but Docker base tags remain mutable. Record their
resolved digests from the build, then replace both `FROM` image references with
`tag@sha256:...` for future rebuilds of an identical environment. Preserve the
built image as well; OS apt packages can change. Use `docker save` if moving it
to another offline machine. Do not confuse the Linux lock with a macOS environment.

References: [uv in Docker](https://docs.astral.sh/uv/guides/integration/docker/),
[vLLM GPU installation and binary compatibility](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/),
[vLLM 0.33.0 metadata](https://pypi.org/pypi/vllm/0.33.0/json),
[Torch 2.14.0 metadata](https://pypi.org/pypi/torch/2.14.0/json).

## 4. Establish the execution layout and download assets

Use GPU UUIDs to avoid assuming that index 0 is the 5090. Replace the UUIDs and
disk location in this host-shell setup. Run it again in each new terminal.

```bash
export WSP_REPO="$PWD"
export WSP_DATA=/mnt/ssd/qwen35-wsp
export WSP_5090=GPU-REPLACE-WITH-5090-UUID
export WSP_4090=GPU-REPLACE-WITH-4090-UUID
mkdir -p "$WSP_DATA"

wsp5090() {
  docker run --rm --gpus "device=$WSP_5090" --shm-size=8g \
    -v "$WSP_REPO:/workspace:ro" -v "$WSP_DATA:/data" \
    qwen35-wsp:2026-11-03 "$@"
}
wsp4090() {
  docker run --rm --gpus "device=$WSP_4090" --shm-size=8g \
    -v "$WSP_REPO:/workspace:ro" -v "$WSP_DATA:/data" \
    qwen35-wsp:2026-11-03 "$@"
}
wspcpu() {
  docker run --rm --shm-size=2g \
    -v "$WSP_REPO:/workspace:ro" -v "$WSP_DATA:/data" \
    qwen35-wsp:2026-11-03 "$@"
}

wsp5090 hardware
wsp4090 hardware
wspcpu prepare --out /data
wspcpu rebuild --out /data/published-edit
```

`prepare` downloads the original multimodal model at revision
`2fc06364715b967f1860aea9cf38778875588b17`, plus the MATH-500 and GSM8K test
datasets. It records resolved dataset revisions and exported-file hashes in
`/data/assets.json`. Keep the benchmark directory out of any editor's mount.
Public downloads need network initially; later runs can be offline after caches
and any required vLLM kernels are warm. If a download is interrupted before
`assets.json` is written, rerun `prepare`; after it exists use the saved assets.

`rebuild` calls the authors' `apply_edit.py`, verifies the original and rebuilt
tensor hashes, and copies the base tokenizer/config files. It does not depend
on the published edited checkpoint being unchanged on Hugging Face.

Expected edited BF16 tensor hashes:

```text
layers.15.mlp.down_proj.weight
82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53

layers.19.mlp.down_proj.weight
8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2
```

The full tensor names begin with `model.language_model.`. Only these two tensors
are replaced. Of 131,072 touched entries, 130,188 differ after BF16 rounding.
Hashes in `work_notes/frozen_weights_manifest.json` refer to float32 submissions;
use **`edit/manifest.json`** for the evaluated BF16 hashes above.

## 5. Smoke test and measure your GPU

First make the original 96-problem diagnostic set from the recorded request:

```bash
wspcpu select \
  --request /workspace/exp8_math/runs/p50c/requests/fwd_1.json.claimed \
  --out /data/sets/diagnostic96.jsonl

# Quick correctness/engine check; 1K outputs are NOT a result reproduction.
wsp5090 generate --model /data/base --rows /data/sets/diagnostic96.jsonl \
  --limit 8 --max-tokens 1024 --max-seqs 8 --out /data/smoke/base-1k.jsonl

# Long-context diagnosis and throughput calibration.
wsp5090 generate --model /data/base --rows /data/sets/diagnostic96.jsonl \
  --max-seqs 8 --out /data/discovery/diagnostic-base.jsonl
```

Use the same command on the 4090 with `--max-seqs 4` and a **different output
path** if measuring that card. The helper prints each queued batch's aggregate
output tokens/s. Discard the first batch for steady-state estimation, but also
measure total job wall time. Use enough prompts to saturate the chosen concurrency;
single-stream token rate is not aggregate throughput. Measure edited throughput
too because shorter outputs change batching and prefill overhead.

If startup/kernel compilation fails, first verify the hardware check, driver,
CUDA version, and installed package provenance. For graph/compile memory issues,
try a fresh output path with `--eager`. For OOM, reduce `--max-seqs` to 4, 2, or 1
and `--batch-tokens` from 4096 to 2048. On a display GPU leave additional headroom.
Do not change the 32,768 output cap, BF16 precision, or sampling protocol to obtain
a nicer headline score. A slow but correct run is useful evidence.

The first run may download/build architecture-specific kernels. Warm the exact
image on each GPU before offline use. There is no hardware-specific speed claim
until both the long-context test and an edited-model test have completed.

## 6. Reproduce diagnosis, localization, and edit construction

### 6.1 Diagnose from generated text

Read several incorrect, finished, and cut-off outputs in
`discovery/diagnostic-base.jsonl`. The original run got 11/96 correct and 85/96
cut off; your sampled counts will vary. Check that repeated reasoning dominates
the cut-off outputs. Inspect `tail_duplicate8`, `cut`, `thinking_closed`, and the
actual text; a correct boxed answer may appear even in a subsequently cut answer.
The helper deliberately grades such answers with the original extraction rule.

For a **fresh independent discovery**, construct your own matched productive and
repetitive continuations using only practice data, keep exact raw text, and do
not inspect evaluation problems or the published neuron IDs. The commands below
instead replay the authors' four published pairs for a traceable method reproduction.
This replay uses known layers 15/19; blind localization must scan all layers first.

### 6.2 Collect the eight short contrast traces

```bash
wsp5090 probe --model /data/base \
  --request /workspace/exp8_math/runs/p50c/requests/fwd_4.json.claimed \
  --out /data/probes/contrast-original
```

Expected token lengths for the pinned tokenizer: `124, 123, 131, 136, 107, 146,
98, 135`. The raw request already includes the chat markers and `<think>`; do
not apply the chat template a second time or insert a closed empty thinking block.

The probe records each block's residual output and each MLP's **input to
`down_proj`**, together with token log-probabilities. Hidden state index 16 is
the output of layer 15; hidden state index 20 is the output of layer 19.
The probe's index 24 is the raw final-block output, before final normalization;
the final edit uses indices 16 and 20, so this last-state convention does not
enter its construction.

To localize without assuming the answer, inspect every layer's change in the
productive/repetitive residual contrast, and its MLP contribution `W @ delta_a`.
Rank neurons using `delta_a[j] * dot(u, W[:, j])`. An activation correlation with
correctness alone is not causal evidence. The original implementation and
single-neuron test are in `tool_calls.jsonl` (search `Top MLP writers`).

### 6.3 Derive directions and write candidate checkpoints

At each selected layer:

1. Average the last **36 tokens** of each contrast's residual stream.
2. Subtract each productive average from its matched repetitive average.
3. Average the four differences and normalize to unit length: `u`.
4. Average the last **24 tokens** of the MLP activations and compute their paired
   repetitive-minus-productive contrast `delta_a`.
5. Rank neurons by `score[j] = delta_a[j] * (u @ W)[j]`; select the 64 highest.
6. Compute the edit from the **original** float32 weights and cast the result to BF16:

```python
projection = u @ W
W_new = W - alpha * torch.outer(u, mask64 * projection)
```

The `alpha=6` edit overcorrects the selected component; it is not simply a
projection removal. Restart from base weights for every candidate, not the last
edited checkpoint. Do not edit embeddings, output head, final norm, or globally
rescale an entire tensor.

```bash
# Reconstructed final-shaped candidate; no published diff is used here.
wspcpu candidate --probe /data/probes/contrast-original \
  --alpha 6 --topk 64 --out /data/candidates/top64-a6

# Earlier dense directional dose, useful for the causal sequence.
wspcpu candidate --probe /data/probes/contrast-original \
  --alpha 1 --topk 0 --out /data/candidates/dense-a1

# An equal-Frobenius-norm, orthogonal random-direction control.
wspcpu candidate --probe /data/probes/contrast-original \
  --alpha 6 --topk 64 --mode random --out /data/candidates/random-matched
```

Each candidate is a complete model directory and has `candidate.json` with
selected indices, relative change, and number of differing BF16 entries. The
helper is restricted to the full single-file base snapshot; it is not a general
sharded-checkpoint editor. New output directories must not already exist.

To reproduce the experimental logic, also test dense `alpha=1,2,4`, sparse
`alpha=4,6`, and single-layer variants (`--layers 15` or `--layers 19`). Each
candidate needs a unique name. Use practice development problems to compare;
do not choose a dose from benchmark scores. The original run also tested a
layer-19 neuron-11 ablation and attention-head alternatives. Those operations
are recorded in the action log, but are not separate helper subcommands.

### 6.4 Measure selectivity on new contrast texts

```bash
wsp5090 probe --model /data/base \
  --request /workspace/exp8_math/runs/p50c/requests/fwd_24.json.claimed \
  --out /data/probes/transfer-base
wsp5090 probe --model /data/candidates/top64-a6 \
  --request /workspace/exp8_math/runs/p50c/requests/fwd_24.json.claimed \
  --out /data/probes/transfer-candidate
wspcpu transfer --original /data/probes/transfer-base \
  --edited /data/probes/transfer-candidate \
  --bounds /workspace/exp8_math/runs/p50c/work_notes/transfer_probe_bounds.json \
  --out /data/discovery/transfer-top64-a6.json
```

Repeat for the dense doses, single-layer variants, and random control. The
reference final edit lowered repetitive-branch log-probability by about 0.103
nats/token and productive-branch log-probability by about 0.009. Look for
selective suppression and a dose response, not numerical equality. Premature
thinking-close controls should not reveal indiscriminate forced termination.
The helper checks token lengths against the recorded boundaries and refuses
to use them if tokenization differs; recompute boundaries before proceeding.

### 6.5 Generation tests on development problems

Use the diagnostic set for exploratory paired tests. Keep seeds, problem IDs,
sample counts, output cap, engine settings, and concurrency fixed between original
and edited jobs. A sample command pair:

```bash
wsp5090 generate --model /data/base --rows /data/sets/diagnostic96.jsonl \
  --n 1 --max-seqs 8 --out /data/dev/base.jsonl
wsp5090 generate --model /data/candidates/top64-a6 \
  --rows /data/sets/diagnostic96.jsonl --n 1 --max-seqs 8 \
  --out /data/dev/top64-a6.jsonl
wspcpu compare --original /data/dev/base.jsonl --edited /data/dev/top64-a6.jsonl \
  --out /data/dev/top64-a6-comparison.json
```

Repeat for controls and candidate strengths, recording decisions in a notebook.
Do not treat reruns of the same diagnostic questions as independent evidence.
The helper seeds each `(problem ID, sample index)` deterministically and uses
the same mapping across models. This is a documented replacement for the
unreleased original seed mapping. It does not guarantee identical random-number
trajectories across GPUs, batching, or resumed jobs.

For a strict 50-pass study, keep a ledger with request hash, candidate hash,
problem IDs, backend, token count, successful/failed status, and result paths.
Count one complete logical generation request, even if microbatched locally.
Preserve the original limits: at most 512 generations per request, or 256 per
model for a paired request; teacher-forced at most 1,024 texts and 524,288 total
tokens. CPU analysis is not a forward pass. The supplied helpers do not enforce
an autonomous 50-pass controller; you must manage that ledger explicitly.

## 7. Freeze the edit, then confirm on independent practice data

For a traceable replay, use the authors' five disjoint confirmation splits,
totalling 1,152 unique practice problems: 128 + 256 + 256 + 256 + 256. These
are from GSM8K/MATH **train**, not their test benchmarks.

```bash
for split in validation_3 validation_4 final_1 final_2 final_3; do
  wspcpu select \
    --request /workspace/exp8_math/runs/p50c/work_notes/fresh_splits.json \
    --key "$split" --out "/data/sets/$split.jsonl"
done

# Freeze before inspecting confirmation results. Hash the actual BF16 model.
sha256sum "$WSP_DATA/candidates/top64-a6/model.safetensors" \
  > "$WSP_DATA/frozen-candidate.sha256"
```

For each split run original and frozen candidate, then compare. For example:

```bash
wsp5090 generate --model /data/base --rows /data/sets/validation_3.jsonl \
  --max-seqs 8 --out /data/confirm/validation_3-base.jsonl
wsp5090 generate --model /data/candidates/top64-a6 \
  --rows /data/sets/validation_3.jsonl --max-seqs 8 \
  --out /data/confirm/validation_3-edit.jsonl
wspcpu compare --original /data/confirm/validation_3-base.jsonl \
  --edited /data/confirm/validation_3-edit.jsonl \
  --out /data/confirm/validation_3-comparison.json
```

Repeat for the other four names. Concatenate the five original files and the
five edited files separately, then run `compare` on those pooled files. It
rejects duplicate `(id,sample)` pairs or mismatched sets. For multiple samples,
its standard error is calculated across **problem means**, so repeated samples
are not falsely counted as independent questions. Do not concatenate retries
or include a partially completed split twice.

The original result was 10.6% → 60.4%, gain 49.8 percentage points, SE 1.5 points,
579 rescued samples and five regressions. Of those rescues, 576 originally hit
the token limit. Edited correct outputs still reasoned for thousands of tokens.
Inspect regressions and remaining whitespace/repetition failures; accuracy is
the primary criterion, not simply a lower truncation fraction.

If confirmation fails, report it. If you use those results to revise the edit,
the split becomes development data: reserve **new** disjoint confirmation
problems. A fresh scientific discovery must choose its own splits before tuning;
the recorded splits here are suitable for a replay whose known result is disclosed.

## 8. Reproduce final MATH-500 and GSM8K evaluation

First evaluate the **hash-verified published edit** as a baseline for the new
implementation. Evaluate the independently constructed candidate in another
directory only after freezing it. Use exactly:

- Thinking enabled through the pinned chat template.
- Problem + newline + `Please reason step by step, and put your final answer within \\boxed{}.`
  (In the actual Python string and prompt there is one backslash before `boxed`.)
- BF16, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5.
- 32,768 new tokens, with `max_model_len=34,816` leaving room for the prompt.
- MATH-500: 500 problems × 4 samples. GSM8K: 1,319 × 2 samples.
- Last boxed-answer extraction and math-verify equivalence checking, not
  format-agnostic/lenient grading.

```bash
wsp5090 generate --model /data/base --rows /data/bench/math500.jsonl \
  --n 4 --max-seqs 8 --out /data/eval/base-math500.jsonl
wsp5090 generate --model /data/published-edit --rows /data/bench/math500.jsonl \
  --n 4 --max-seqs 8 --out /data/eval/published-math500.jsonl
wspcpu compare --original /data/eval/base-math500.jsonl \
  --edited /data/eval/published-math500.jsonl --out /data/eval/math500-comparison.json

wsp5090 generate --model /data/base --rows /data/bench/gsm8k.jsonl \
  --n 2 --max-seqs 8 --out /data/eval/base-gsm8k.jsonl
wsp5090 generate --model /data/published-edit --rows /data/bench/gsm8k.jsonl \
  --n 2 --max-seqs 8 --out /data/eval/published-gsm8k.jsonl
wspcpu compare --original /data/eval/base-gsm8k.jsonl \
  --edited /data/eval/published-gsm8k.jsonl --out /data/eval/gsm8k-comparison.json
```

For a discovered candidate, substitute `/data/candidates/top64-a6` and use new
output filenames. The reported post scores are MATH-500 7.9% → 48.1%, GSM8K
13.8% → 67.1%. The public rerun scored the edited checkpoint at MATH-500 51.0%,
illustrating that exact weight hashes do not imply identical stochastic scores.

The helper writes every sample, grade, length, truncation flag, and response.
It reuses the released answer-extraction and math-verify functions without
importing the unavailable `weightenv` package. Sampling seeds, batching, and
the replacement runner differ from the original, so label the report accordingly.
Output metadata includes weights and input hashes, versions, seeds and engine
settings. Existing output with different settings is rejected; use a new path
for a changed configuration. Complete batches are append-saved; a malformed
last JSONL line after an interrupted write must be repaired before resuming.

### Broader benchmark extension

The core target experiment above reproduces discovery and MATH/GSM evaluation.
Reproducing **every** number in the public results figure additionally requires:

| Benchmark | Problems | Samples/problem | Grading |
|---|---:|---:|---|
| AMC23 | 40 | 16 | boxed answer + math-verify |
| Minerva Math | 272 | 4 | boxed answer + math-verify |
| OlympiadBench | 674 | 4 | English text-only open-ended; original answer formatting |
| MMLU-Pro | fixed 1,000-question subset | 1 | original multiple-choice prompt/extraction |
| GPQA Diamond | 198 | 4 | original multiple-choice prompt/extraction |
| IFEval | 541 | 2 | official strict prompt-level checkers on answer after `</think>` |

The released `code/evaluate.py` describes preparation and sampling, but imports
private `weightenv` task modules and IFEval/MC graders absent from this repository.
**It is not a turnkey entry point.** The supplied helper directly supports
boxed math datasets; export additional math datasets as `{id,source,problem,answer}`
JSONL and use it with the sample counts above. For the three non-math datasets,
port the released `grade_other` prompt/extraction contract to the official
benchmark implementations and record the exact subset/permutation/revisions.
An arbitrary new 1,000-question MMLU-Pro subset is not an exact reproduction.
GPQA requires the appropriate dataset access. Do not silently replace these
metrics with boxed math grading. These additional adapters are **not included**
in the core helpers. Consult [official IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval),
[MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro), and
[GPQA](https://github.com/idavidrein/gpqa).

## 9. Use the RTX 4090 productively

Recommended first pass: finish smoke tests and a short paired job on the 5090
alone. Then use the 4090 for an independent **paired** benchmark/dataset or
probe. Do not assume that the cards form a single 56 GB pool. Avoid tensor
parallelism between different GPU architectures for this small model.

A simple schedule uses the 5090 for paired MATH-500 and the 4090 for paired
GSM8K. On each card, run original then edited with matching settings. This
keeps a pair's backend and hardware consistent. CPU candidate construction
can run while a GPU evaluates a frozen candidate, provided RAM/disk allow it.

Alternatively split the *problem IDs* of one dataset across cards:

```bash
# Terminal A: problem shard 0 on the 5090.
wsp5090 generate --model /data/base --rows /data/bench/math500.jsonl \
  --n 4 --shards 2 --shard-index 0 --max-seqs 8 --out /data/shards/base-0.jsonl
wsp5090 generate --model /data/published-edit --rows /data/bench/math500.jsonl \
  --n 4 --shards 2 --shard-index 0 --max-seqs 8 --out /data/shards/edit-0.jsonl

# Terminal B: problem shard 1 on the 4090.
wsp4090 generate --model /data/base --rows /data/bench/math500.jsonl \
  --n 4 --shards 2 --shard-index 1 --max-seqs 4 --out /data/shards/base-1.jsonl
wsp4090 generate --model /data/published-edit --rows /data/bench/math500.jsonl \
  --n 4 --shards 2 --shard-index 1 --max-seqs 4 --out /data/shards/edit-1.jsonl
```

Concatenate original shards separately from edited shards and compare. Every
problem must appear exactly once per sample. Save both shard metadata files.
Equal halves may leave the faster GPU idle; balance shards after measuring
rates if needed, while keeping original/edited problem assignment identical.
The upper-bound combined throughput is `R5090 + R4090`, but sequential discovery
decisions and host/IO bottlenecks reduce whole-experiment speedup. Two GPUs also
need suitable power delivery, cooling, PCIe slots, and host RAM; running each
on a separate Linux machine is fine and avoids assuming a dual-card chassis.

## 10. If recreating the autonomous editor experiment

Do this only after the target-model pipeline works. A fresh editor must see the
task, target weights, and **practice** data, not this guide's known final edit,
published neuron lists, test datasets, or test scores. Give it the original
`turns/t001/prompt.txt` rules and maintain its own separate run directory.

Reimplement these missing harness services:

1. A persistent CPU Python kernel, with gradients disabled, exposing `NAMES`,
   `load`, `load_original`, `save_edit`, `drop_edit`, `edited`, and the tokenizer.
   Reject shape/dtype-invalid and nonfinite writes, edits to forbidden tensors,
   and whole-tensor scalar rescaling. Enforce the documented no-training rule.
2. A forward request controller: generation and teacher-forced modes, paired
   per-problem seeds, per-pass limits, failure accounting, atomic request/result
   files, and the 50-successful-pass budget.
3. Short probes plus chunked long-text teacher forcing; a separate implementation
   of generation-time residual/MLP averages for the first 8,192 answer tokens.
   The short `probe` helper is not that service.
4. An editor loop that executes one approved reply block, returns compact output,
   retains notes/checkpoints, counts Python runs (maximum 300), and preserves
   prompts, replies, request payloads, timing, hashes, and final decisions.

Run the CPU editor kernel in a separate container with no GPU access and no
network, mounting `/model` and `/practice` read-only and `/work` writable. Keep
the target GPU worker separate; only a trusted controller passes structured
requests/results. The external editor connection belongs in the controller,
not in the isolated weight-analysis kernel. All Python services should use the
same image's `/opt/venv` (or a separately hash-locked CPU `uv venv` image).

The helpers supplied here are transparent building blocks for a supervised
method reproduction, not a security sandbox for executing arbitrary generated
code. Recreating these missing services is additional implementation work;
the original logs do not make them automatically available.

## 11. Completion checklist

- Image, Linux lock, driver, GPU UUIDs, source commit and local changes recorded.
- Base model revision pinned; dataset snapshots and exported hashes retained.
- Published checkpoint rebuild passes the authors' BF16 tensor-hash checks.
- Long-context smoke test passes on each GPU used; runtime estimate uses measured
  aggregate throughput instead of advertised GPU arithmetic performance.
- Diagnosis is supported by actual failures; productive/repetitive raw contexts
  and tokenization retained; layer and neuron choices have causal controls.
- New candidates are constructed from original weights, cast to BF16 before
  generation, and kept separate from the exact published checkpoint.
- Candidate frozen before confirmation; no benchmark-guided tuning; independent
  confirmation reports accuracy, paired uncertainty, rescues and regressions.
- Final report names the exact runner, sampling, sample counts, shard assignments,
  missing optional benchmark adapters, and any deviations from the original.

Start with the **5090 alone**, verify the released edit, reproduce its mechanism,
then add the 4090 for throughput. Neither a second GPU nor training is needed
to fit the core experiment.
