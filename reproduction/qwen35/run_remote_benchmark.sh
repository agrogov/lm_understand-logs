#!/usr/bin/env bash
# Execute ON the Linux Docker host from its dedicated benchmark directory.
set -euo pipefail
WSP_REMOTE_ROOT="${1:-/root/qwen35-benchmark-20261104}"
WSP_IMAGE="${WSP_IMAGE:-qwen35-reproduction:20261104}"
WSP_GPU="${WSP_GPU:-GPU-2a5dea92-f35f-086a-776b-8c7531e9a802}"
WSP_EXPECTED_GPU="${WSP_EXPECTED_GPU:-4090}"
WSP_DEVICE_ARGS=()
if [[ "$WSP_EXPECTED_GPU" != 4090 ]]; then
  WSP_DEVICE_ARGS=(--expected-gpu "$WSP_EXPECTED_GPU")
fi
WSP_EXECUTION_ARGS=()
if [[ "${WSP_EAGER:-0}" == 1 ]]; then WSP_EXECUTION_ARGS=(--eager); fi
WSP_CASES="${WSP_CASES:-1:2048,4:4096,8:4096,4:32768}"
WSP_RESULTS_DIR="${WSP_RESULTS_DIR:-$WSP_REMOTE_ROOT/results}"
WSP_CPU_LIMIT="${WSP_CPU_LIMIT:-4}"
WSP_CPU_THREADS="${WSP_CPU_THREADS:-2}"
WSP_COMPILE_THREADS="${WSP_COMPILE_THREADS:-1}"
WSP_CPU_ARGS=(--cpus "$WSP_CPU_LIMIT")
if [[ -n "${WSP_CPUSET:-}" ]]; then WSP_CPU_ARGS+=(--cpuset-cpus "$WSP_CPUSET"); fi
mkdir -p "$WSP_RESULTS_DIR"
printf '%s\n' "CUDA_MODULE_LOADING=${WSP_CUDA_MODULE_LOADING:-LAZY}" \
  "WSP_EAGER=${WSP_EAGER:-0}" "WSP_GPU=$WSP_GPU" \
  "WSP_CASES=$WSP_CASES" "WSP_CPU_LIMIT=$WSP_CPU_LIMIT" \
  "WSP_CPUSET=${WSP_CPUSET:-}" "WSP_CPU_THREADS=$WSP_CPU_THREADS" \
  "WSP_COMPILE_THREADS=$WSP_COMPILE_THREADS" > "$WSP_RESULTS_DIR/run-config.txt"
docker image inspect "$WSP_IMAGE" > "$WSP_RESULTS_DIR/image-inspect.json"
docker run --rm --entrypoint sha256sum "$WSP_IMAGE" \
  /opt/benchmark_cuda.py /opt/requirements.lock > "$WSP_RESULTS_DIR/image-source-sha256.txt"
docker run --rm --entrypoint uv "$WSP_IMAGE" \
  pip freeze --python /opt/venv/bin/python > "$WSP_RESULTS_DIR/packages.txt"
nvidia-smi -i "$WSP_GPU" --query-gpu=timestamp,name,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm \
  --format=csv -l 1 > "$WSP_RESULTS_DIR/gpu-monitor.csv" &
WSP_MONITOR_PID=$!
vmstat -w -t 1 > "$WSP_RESULTS_DIR/host-cpu-memory.log" &
WSP_HOST_MONITOR_PID=$!
(
  while true; do
    date -u +%FT%TZ
    docker stats --no-stream --format '{{json .}}' 2>/dev/null || true
    sleep 2
  done
) > "$WSP_RESULTS_DIR/container-stats.log" &
WSP_CPU_MONITOR_PID=$!
trap 'kill "$WSP_MONITOR_PID" "$WSP_HOST_MONITOR_PID" "$WSP_CPU_MONITOR_PID" 2>/dev/null || true' EXIT
for WSP_MODEL in ${WSP_MODELS:-base edited}; do
  case "$WSP_MODEL" in base|edited) ;; *) echo 'Unknown checkpoint' >&2; exit 2;; esac
  WSP_MODEL_DIR=base-hf
  if [[ "$WSP_MODEL" == edited ]]; then WSP_MODEL_DIR=published-edit-hf; fi
  docker run --rm --name "qwen35-benchmark-$WSP_MODEL" \
    "${WSP_CPU_ARGS[@]}" \
    --gpus "device=$WSP_GPU" --shm-size=8g \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/results/hf \
    -e OMP_NUM_THREADS="$WSP_CPU_THREADS" -e MKL_NUM_THREADS="$WSP_CPU_THREADS" \
    -e OPENBLAS_NUM_THREADS="$WSP_CPU_THREADS" -e NUMEXPR_NUM_THREADS="$WSP_CPU_THREADS" \
    -e TOKENIZERS_PARALLELISM=false -e RAYON_NUM_THREADS="$WSP_CPU_THREADS" \
    -e MAX_JOBS="$WSP_COMPILE_THREADS" -e TORCHINDUCTOR_COMPILE_THREADS="$WSP_COMPILE_THREADS" \
    -e "CUDA_MODULE_LOADING=${WSP_CUDA_MODULE_LOADING:-LAZY}" \
    -e VLLM_CACHE_ROOT=/results/vllm-cache \
    -v "$WSP_REMOTE_ROOT/data:/data:ro" -v "$WSP_RESULTS_DIR:/results" \
    --entrypoint /opt/venv/bin/python "$WSP_IMAGE" /opt/benchmark_cuda.py \
    --model "/data/$WSP_MODEL_DIR" --prompts /data/benchmark-prompts \
    --out "/results/$WSP_MODEL.json" --cases "$WSP_CASES" "${WSP_DEVICE_ARGS[@]}" "${WSP_EXECUTION_ARGS[@]}" \
    2>&1 | tee "$WSP_RESULTS_DIR/$WSP_MODEL.log"
done
