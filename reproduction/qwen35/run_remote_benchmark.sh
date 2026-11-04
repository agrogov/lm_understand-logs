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
WSP_CASES="${WSP_CASES:-1:2048,4:4096,8:4096,4:32768}"
WSP_RESULTS_DIR="${WSP_RESULTS_DIR:-$WSP_REMOTE_ROOT/results}"
mkdir -p "$WSP_RESULTS_DIR"
docker image inspect "$WSP_IMAGE" > "$WSP_RESULTS_DIR/image-inspect.json"
docker run --rm --entrypoint uv "$WSP_IMAGE" \
  pip freeze --python /opt/venv/bin/python > "$WSP_RESULTS_DIR/packages.txt"
nvidia-smi -i "$WSP_GPU" --query-gpu=timestamp,name,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm \
  --format=csv -l 1 > "$WSP_RESULTS_DIR/gpu-monitor.csv" &
WSP_MONITOR_PID=$!
trap 'kill "$WSP_MONITOR_PID" 2>/dev/null || true' EXIT
for WSP_MODEL in base edited; do
  WSP_MODEL_DIR=base-hf
  if [[ "$WSP_MODEL" == edited ]]; then WSP_MODEL_DIR=published-edit-hf; fi
  docker run --rm --name "qwen35-benchmark-$WSP_MODEL" \
    --gpus "device=$WSP_GPU" --shm-size=8g \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/results/hf -e OMP_NUM_THREADS=8 \
    -e MAX_JOBS=4 -e TORCHINDUCTOR_COMPILE_THREADS=4 \
    -e VLLM_CACHE_ROOT=/results/vllm-cache \
    -v "$WSP_REMOTE_ROOT/data:/data:ro" -v "$WSP_RESULTS_DIR:/results" \
    --entrypoint /opt/venv/bin/python "$WSP_IMAGE" /opt/benchmark_cuda.py \
    --model "/data/$WSP_MODEL_DIR" --prompts /data/benchmark-prompts \
    --out "/results/$WSP_MODEL.json" --cases "$WSP_CASES" "${WSP_DEVICE_ARGS[@]}" \
    2>&1 | tee "$WSP_RESULTS_DIR/$WSP_MODEL.log"
done
