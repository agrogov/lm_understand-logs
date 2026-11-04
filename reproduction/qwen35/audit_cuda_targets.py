"""Inspect packaged CUDA targets without initializing CUDA or executing kernels.

Run in the existing image with --runtime=runc, no --gpus, and no network.
Architecture lists describe containers, not coverage of every individual kernel.
"""
import concurrent.futures
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess


def listing(binary):
    result = {'file':str(binary)}
    for kind in ['elf','ptx']:
        p = subprocess.run(['cuobjdump',f'--list-{kind}',str(binary)],
                           text=True,capture_output=True,timeout=120)
        lines = p.stdout.splitlines()
        result[kind] = {'returncode':p.returncode,
            'architectures':sorted(set(re.findall(r'sm_\d+[af]?',p.stdout))),
            'entries':sum('file' in line.lower() for line in lines),
            'sm120_examples':[line.strip() for line in lines if 'sm_120' in line][:3],
            'stderr':p.stderr[-1500:]}
    return result


def main():
    import torch
    flags = torch._C._cuda_getArchFlags()  # Compiled-in string; no device query.
    torch_root = Path(torch.__file__).parent
    files = [p for p in (torch_root/'lib').glob('libtorch_cuda*.so')]
    distribution = importlib.metadata.distribution('vllm')
    files += [Path(distribution.locate_file(p)) for p in distribution.files
              if str(p).endswith('.so')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        libraries = list(pool.map(listing,sorted(set(files))))
    nvcc = subprocess.run(['nvcc','--version'],text=True,capture_output=True,check=True)
    targets = subprocess.run(['nvcc','--list-gpu-code'],text=True,capture_output=True,check=True)
    print(json.dumps({'gpu_exposure':'None: audit must run with runc and no GPU devices',
        'cuda_initialized':torch.cuda.is_initialized(),
        'versions':{p:importlib.metadata.version(p) for p in ['torch','vllm','flashinfer-python','triton']},
        'torch_cuda':torch.version.cuda,'torch_compiled_arch_flags':flags,
        'torch_build_config':torch.__config__.show(),
        'nvcc_version':nvcc.stdout.strip(),'nvcc_supported_targets':targets.stdout.split(),
        'cuobjdump':shutil.which('cuobjdump'),'libraries':libraries,
        'limitations':'No kernel was executed. Native target presence does not prove all kernel paths are correct; runtime-JIT outputs and driver stability are not validated.'},indent=2))


if __name__ == '__main__':
    main()
