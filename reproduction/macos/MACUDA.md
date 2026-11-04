# RTX 4090 eGPU and MacCUDA: applicability to this experiment

Checked 2026-11-04 against the project's current documentation. No eGPU driver
or MacCUDA installation was performed on this Mac.

**Local source audit update:** a custom native application has now been compiled
and linked against this checkout's existing `sm_89` archives, and its offline
callback-selection checks passed. The source exposes Qwen3.5 residual/MLP graph
tensors and evaluation callbacks, so a narrow research engine is feasible
without implementing PyTorch compatibility. See the
[source audit and build proof](../macuda-engine/README.md). This is an offline
link proof, not a real 4090 inference validation.

**Real hardware follow-up:** the connected 4090 subsequently passed 450 operator
cases and custom BF16 model/activation tests for base and edited checkpoints.
Missing Ada BF16 GEMM kernels were rebuilt locally. Both CPU comparisons matched
all 32 greedy token IDs, with activation relative-L2 differences below 0.56%.
See [the hardware report](../macuda-engine/HARDWARE_TEST.md). A subsequent
[full-cap throughput benchmark](../macuda-engine/BENCHMARK.md) completed both
checkpoints: 312.7 base / 341.1 edited aggregate tokens/s, projecting 310.1
generation hours versus 278.2 hours on MLX for the recorded workload. Full
protocol and long-context numerical/accuracy validation remain open.

## Is the GPU used only for inference?

Yes: the experiment has no gradient descent or optimizer. Its GPU workloads
are **forward inference**, which includes more than generating answers:

| Work | GPU needed for practical speed? | What the backend must expose |
|---|---|---|
| Practice/benchmark answer generation | Yes | Exact model, thinking template, sampling, and generated text |
| Teacher-forced diagnostics | Yes | Per-token log-probabilities, residual states, MLP inputs, sometimes attention |
| Weight analysis, direction construction, direct edits | No | CPU float32 arithmetic and checkpoint files |
| Grading, paired statistics, run control | No | CPU tools and saved results |

An endpoint that only returns chat completions covers the first row. It does
not automatically provide the activation measurements needed to discover the edit.

## Can the RTX 4090 work as an eGPU through MacCUDA?

**The project lists RTX 4090 (AD102, `sm_89`) support**, including an Ada build
selection. Its published development systems use macOS 27.0; compatibility with
this Mac's macOS 26.6.2 and a particular enclosure must be tested. The published
throughput tables do not establish this machine's RTX 4090 speed.

MacCUDA's scope is native `llama.cpp` and `stable-diffusion.cpp` CUDA backends,
using TinyGPU hardware access. It explicitly disclaims established PyTorch/vLLM
support. Thus an eGPU is a plausible **experimental generation backend**, not a
drop-in device for our Python CUDA runner or native MLX environment.

Its build still compiles GPU kernels with Linux `nvcc`, using a container or
an SSH compiler host; that build step does not require GPU access.

Source: [MacCUDA requirements, architectures and limitations](https://github.com/Davinchy/macuda#readme).

## Practical choices

1. **Current Mac environment:** use the internal M4 Max GPU through MLX for
   generation and port short activation probes to that same backend.
2. **4090/5090 Linux machine:** use the existing Docker/CUDA pipeline and control
   jobs from the Mac. A custom forward worker can return both text and diagnostic
   arrays; a chat-only API is insufficient for the complete experiment.
3. **MacCUDA eGPU experiment:** the card, conversion, short numerical comparisons,
   activation readback and long-output throughput have now been tested locally.
   This implementation did not improve the combined runtime projection over
   MLX. Use it for native-worker development; complete teacher-forced/controller
   integration and broader numerical/accuracy checks before full reproduction.

Do not infer a 4090 estimate from the project's 5090 or unrelated-model results.
An NVIDIA eGPU also does not become another MLX/Metal device simply by installing
this compatibility layer.

TinyGPU documents USB4/Thunderbolt hardware access for NVIDIA Ampere and newer,
and a macOS driver-extension approval step. That is a distinct runtime path
from ordinary native Apple GPU execution. See
[TinyGPU requirements and setup](https://docs.tinygrad.org/tinygpu/).
