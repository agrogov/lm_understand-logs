"""Bounded BF16 MLX throughput benchmark on practice questions, not a score run.

Run through run.sh. Cases are concurrent_sequences:max_new_tokens. Raw timings,
token IDs and responses are retained; no quantization or forced EOS suppression.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from functools import partial
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.generate import BatchGenerator

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):
            h.update(chunk)
    return h.hexdigest()


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def sampler(logprobs):
    # Temperature -> top-k -> top-p, rather than the different ordering in
    # mlx_lm.make_sampler. Reduce to top 20 before sorting for nucleus sampling.
    indices = mx.argpartition(logprobs, kth=logprobs.shape[-1]-20, axis=-1)[..., -20:]
    values = mx.take_along_axis(logprobs, indices, axis=-1).astype(mx.float32) / .6
    order = mx.argsort(-values, axis=-1)
    values = mx.take_along_axis(values, order, axis=-1)
    indices = mx.take_along_axis(indices, order, axis=-1)
    probabilities = mx.softmax(values, axis=-1)
    keep = mx.cumsum(probabilities, axis=-1)-probabilities < .95
    draw = mx.random.categorical(mx.where(keep, values, -float("inf")))
    return mx.take_along_axis(indices, draw[..., None], axis=-1).squeeze(-1)


class PresencePenalty:
    """Penalty once per token seen in generated output, excluding prompt tokens."""
    def __init__(self, prompt_length, vocab_size):
        self.cursor = prompt_length
        self.seen = mx.zeros((vocab_size,), dtype=mx.float32)

    def __call__(self, tokens, logits):
        if len(tokens) > self.cursor:
            new = tokens[self.cursor:]
            self.seen = mx.minimum(self.seen.at[new].add(1), 1)
            self.cursor = len(tokens)
        return logits.astype(mx.float32)-1.5*self.seen


def read_rows():
    request = json.loads((REPO/'exp8_math/runs/p50c/requests/fwd_1.json.claimed').read_text())
    ids = request['problems']
    bank = {}
    for file in (REPO/'exp8_math/practice').glob('*.jsonl'):
        for line in file.open():
            if line.strip():
                r = json.loads(line); bank[r['id']] = r
    # Alternate GSM/MATH; reuse identical questions across checkpoints.
    groups = [[bank[k] for k in ids if k.startswith(prefix)]
              for prefix in ['gsm8k_', 'math_']]
    return [r for pair in zip(*groups) for r in pair]


def run_case(model, tokenizer, rows, batch, cap, label, seed):
    instruction = 'Please reason step by step, and put your final answer within \\boxed{}.'
    prompts = [tokenizer.apply_chat_template(
        [{'role':'user', 'content':r['problem']+'\n'+instruction}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True) for r in rows[:batch]]
    original_lengths=list(map(len,prompts))
    prompts=[p+r.get('history_tokens',[]) for p,r in zip(prompts,rows[:batch])]
    mx.random.seed(seed)
    processors = [[PresencePenalty(n, 248320)] for n in original_lengths]
    gen = BatchGenerator(model, sampler=sampler, completion_batch_size=batch,
                         prefill_batch_size=batch, prefill_step_size=256,
                         stop_tokens=[[t] for t in tokenizer.eos_token_ids])
    tokens = {}; finishes = {}; windows = []
    mx.reset_peak_memory()
    start = time.perf_counter(); last_update = start; last_total = 0
    print(json.dumps({'event':'start', 'model':label,'batch':batch,'cap':cap}), flush=True)
    try:
        with gen.stats() as stats:
            uids = gen.insert(prompts, [cap]*batch, logits_processors=processors)
            tokens = {uid:[] for uid in uids}
            while responses := gen.next_generated():
                for r in responses:
                    if r.finish_reason != 'stop':
                        tokens[r.uid].append(r.token)
                    if r.finish_reason is not None:
                        finishes[r.uid] = r.finish_reason
                now = time.perf_counter()
                if now-last_update >= 20:
                    total = sum(map(len,tokens.values()))
                    window = {'elapsed_s':now-start,'output_tokens':total,
                              'window_output_tps':(total-last_total)/(now-last_update),
                              'max_sequence_tokens':max(map(len,tokens.values())),
                              'finished':len(finishes)}
                    windows.append(window)
                    print(json.dumps({'event':'progress','model':label,'batch':batch,**window}),flush=True)
                    last_update=now;last_total=total
            mx.synchronize()
        elapsed = time.perf_counter()-start
        peak = mx.get_peak_memory()
    finally:
        gen.close()
    count = sum(map(len,tokens.values()))
    samples = [{'id':row['id'],'tokens':len(tokens[uid]),'finish_reason':finishes[uid],
                'text':tokenizer.decode(tokens[uid]),'token_ids':tokens[uid]}
               for row,uid in zip(rows,uids)]
    result = {'model':label,'batch':batch,'max_new_tokens':cap,'seed':seed,
              'output_tokens':count,'wall_seconds_including_prefill':elapsed,
              'aggregate_output_tps_including_prefill':count/elapsed,
              'peak_mlx_memory_bytes':peak,'engine_stats':asdict(stats),
              'progress_windows':windows,'samples':samples,
              'prefilled_generated_history_tokens':[len(r.get('history_tokens',[])) for r in rows[:batch]]}
    print(json.dumps({'event':'result',**{k:v for k,v in result.items()
                                         if k not in ('samples','progress_windows')}}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',default='1:1024,4:4096,8:4096')
    p.add_argument('--models',default='base,edited')
    p.add_argument('--out',default=str(HERE/'artifacts/benchmark.json'))
    args=p.parse_args()
    cases=[tuple(map(int,c.split(':'))) for c in args.cases.split(',')]
    if any(b<1 or b>32 or n<1 or n>32768 for b,n in cases):
        raise ValueError('batch 1..32 and output cap 1..32768 required')
    paths={'base':HERE/'artifacts/base-mlx-bf16','edited':HERE/'artifacts/published-edit-mlx-bf16'}
    destination=Path(args.out)
    if destination.exists():
        raise ValueError('Output exists; choose a new path to preserve earlier evidence')
    rows=read_rows()
    report={'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'hardware':mx.device_info(),
            'versions':{n:importlib.metadata.version(n) for n in ['mlx','mlx-lm','transformers']},
            'sampling':{'temperature':.6,'top_k':20,'top_p':.95,'presence_penalty':1.5,
                        'penalty_history':'generated tokens only, full history',
                        'order':'presence penalty -> temperature -> top-k -> top-p'},
            'checkpoints':{},'results':[]}
    def save():
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2,default=str)+'\n')
        temporary.replace(destination)
    save()
    for label in args.models.split(','):
        path=paths[label]
        report['checkpoints'][label]={
            'path':str(path),'weights_sha256':file_sha256(path/'model.safetensors'),
            'config_sha256':file_sha256(path/'config.json'),
            'tokenizer_sha256':file_sha256(path/'tokenizer.json')}
        model,tokenizer=load(str(path))
        mx.eval(model.parameters())
        dtype_counts=Counter(str(w.dtype) for _,w in tree_flatten(model.parameters()))
        print(json.dumps({'event':'loaded','model':label,'dtypes':dtype_counts}),flush=True)
        if any('quant' in n for n in json.loads((path/'config.json').read_text())):
            raise ValueError('Quantized checkpoints are outside this benchmark')
        # Warm model/kernel path; exclude this tiny case from estimates.
        run_case(model,tokenizer,rows,1,16,label+'-warmup',0)
        for batch,cap in cases:
            report['results'].append(run_case(model,tokenizer,rows,batch,cap,label,1729))
            save()
            gc.collect();mx.clear_cache()
        del model,tokenizer
        gc.collect();mx.clear_cache()
    report['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');save()


if __name__=='__main__':
    main()
