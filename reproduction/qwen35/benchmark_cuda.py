"""Real BF16 fixed-batch vLLM benchmark; run separately for each checkpoint."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--prompts', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cases', default='1:2048,4:4096,8:4096,4:32768')
    p.add_argument('--memory', type=float, default=.85)
    p.add_argument('--eager', action='store_true')
    p.add_argument('--expected-gpu', choices=['4090','5090'], default='4090')
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise ValueError('Use a new output filename')
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = [tuple(map(int, c.split(':'))) for c in a.cases.split(',')]
    if any(not 1 <= b <= 32 or not 1 <= c <= 32768 for b,c in cases):
        raise ValueError('Invalid benchmark cases')
    import torch
    from vllm import LLM, SamplingParams
    if torch.cuda.device_count() != 1 or a.expected_gpu not in torch.cuda.get_device_name(0):
        raise RuntimeError(f'Expose only the requested RTX {a.expected_gpu} to this container')
    report = {'versions':{k:importlib.metadata.version(k) for k in ['vllm','torch','transformers']},
              'gpu':torch.cuda.get_device_name(0), 'model':a.model,
              'visible_devices':os.environ.get('NVIDIA_VISIBLE_DEVICES'),
              'max_model_len':34816,'weights_dtype':'bfloat16','kv_cache_dtype':'auto',
              'sampling':{'temperature':.6,'top_k':20,'top_p':.95,'presence_penalty':1.5},
              'eager':a.eager,'memory_fraction':a.memory,'results':[]}
    report['weights_sha256'] = {}
    for file in sorted(Path(a.model).glob('*.safetensors')):
        with file.open('rb') as stream:
            report['weights_sha256'][file.name] = hashlib.file_digest(stream,'sha256').hexdigest()
    texts = [(Path(a.prompts)/f'{i}.txt').read_text() for i in range(max(b for b,_ in cases))]
    report['prompt_sha256'] = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    started = time.perf_counter()
    llm = LLM(model=a.model, dtype='bfloat16', max_model_len=34816,
              gpu_memory_utilization=a.memory, max_num_seqs=max(b for b,_ in cases),
              max_num_batched_tokens=4096, enable_prefix_caching=False,
              enforce_eager=a.eager, seed=1729,
              limit_mm_per_prompt={'image':0,'video':0})
    report['model_and_engine_startup_seconds'] = time.perf_counter()-started
    warm = llm.generate(texts[:1], SamplingParams(max_tokens=128,seed=1729,**report['sampling']),use_tqdm=False)
    if not warm[0].outputs[0].token_ids:
        raise RuntimeError('Warm-up produced no tokens')
    for batch,cap in cases:
        params = [SamplingParams(max_tokens=cap,seed=1729+s,**report['sampling']) for s in range(batch)]
        torch.cuda.synchronize()
        begin = time.perf_counter()
        outputs = llm.generate(texts[:batch],params,use_tqdm=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-begin
        sequences = []
        for s,r in enumerate(outputs):
            o = r.outputs[0]
            if o.finish_reason not in ('stop','length'):
                raise RuntimeError(f'Incomplete generation: {o.finish_reason}')
            sequences.append({'slot':s,'prompt_tokens':len(r.prompt_token_ids),
                              'output_tokens':len(o.token_ids),'finish_reason':o.finish_reason,
                              'token_ids':list(o.token_ids)})
        count = sum(s['output_tokens'] for s in sequences)
        row = {'batch':batch,'max_new_tokens':cap,'output_tokens':count,
               'wall_seconds_including_prefill':elapsed,
               'aggregate_output_tps_including_prefill':count/elapsed,'sequences':sequences}
        report['results'].append(row)
        out.write_text(json.dumps(report,indent=2)+'\n')
        print('BENCH_RESULT '+json.dumps({k:v for k,v in row.items() if k != 'sequences'}),flush=True)
    report['completed'] = True
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
