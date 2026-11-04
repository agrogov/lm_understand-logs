"""Verify the isolated macOS environment with a real Metal computation."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import mlx.core as mx
from mlx_lm.models import qwen3_5
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from math_verify import parse, verify
import numpy as np
from safetensors.numpy import save_file, load_file


def main():
    root = Path(__file__).resolve().parent
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Run native arm64 Python on macOS")
    if Path(sys.prefix).resolve() != (root / ".venv").resolve():
        raise RuntimeError(f"Wrong environment: {sys.prefix}")
    if not mx.metal.is_available():
        raise RuntimeError("Metal is not available to MLX")
    mx.set_default_device(mx.gpu)
    mx.random.seed(0)
    start = time.perf_counter()
    a = mx.random.normal((512, 512)).astype(mx.bfloat16)
    result = a @ a.T
    mx.eval(result)
    if not bool(mx.all(mx.isfinite(result)).item()):
        raise RuntimeError("Metal BF16 matmul produced nonfinite values")
    elapsed = time.perf_counter()-start
    if not verify(parse(r"\boxed{\frac{1}{2}}"), parse(r"\boxed{0.5}")):
        raise RuntimeError("math-verify equivalence check failed")
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    temporary = artifacts / "doctor-safetensors.tmp"
    try:
        save_file({"test": np.eye(2, dtype=np.float32)}, str(temporary))
        np.testing.assert_array_equal(load_file(str(temporary))["test"], np.eye(2, dtype=np.float32))
    finally:
        temporary.unlink(missing_ok=True)
    # Resolve the sampling API used by later native inference work.
    make_sampler(temp=.6, top_p=.95, top_k=20)
    make_logits_processors(presence_penalty=1.5)
    packages = ["mlx", "mlx-metal", "mlx-lm", "numpy", "scipy", "transformers",
                "huggingface-hub", "safetensors", "datasets", "math-verify"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "python": sys.version, "executable": sys.executable,
              "machine": platform.machine(), "macos": platform.mac_ver()[0],
              "physical_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
              "metal_device": mx.device_info(), "default_device": str(mx.default_device()),
              "bf16_matmul_passed": True, "matmul_seconds_including_warmup": elapsed,
              "qwen35_implementation": str(Path(qwen3_5.__file__).resolve()),
              "grading_passed": True, "safetensors_passed": True, "packages": versions}
    (artifacts / "environment-check.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
