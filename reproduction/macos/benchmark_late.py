"""Eight-sequence late-context decode probe using recorded base-model prefixes.

This is a throughput probe, not eight independent accuracy samples. Prefill
existing 28,672-token continuations, then measure up to 4,096 more tokens.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm import load
from benchmark import HERE, file_sha256, read_rows, run_case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default=str(HERE/'artifacts/benchmark-long.json'))
    p.add_argument('--out',default=str(HERE/'artifacts/benchmark-late.json'))
    args=p.parse_args()
    out=Path(args.out)
    if out.exists():raise ValueError('Choose a new output filename')
    source=json.loads(Path(args.source).read_text())
    samples=next(r['samples'] for r in source['results'] if r['model']=='base' and r['max_new_tokens']==32768)
    bank={r['id']:r for r in read_rows()}
    samples=[s for s in samples if s['tokens']>=28672]
    if not samples:raise ValueError('No sufficiently long recorded continuation')
    rows=[dict(bank[samples[i%len(samples)]['id']],
               history_tokens=samples[i%len(samples)]['token_ids'][:28672]) for i in range(8)]
    report={'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'hardware':mx.device_info(),
            'source_sha256':file_sha256(args.source),'prefix_tokens':28672,
            'scope':'synthetic repeated recorded prefixes, late-context throughput only; not accuracy',
            'results':[]}
    for label,path in [('base',HERE/'artifacts/base-mlx-bf16'),
                       ('edited',HERE/'artifacts/published-edit-mlx-bf16')]:
        model,tok=load(str(path));mx.eval(model.parameters())
        result=run_case(model,tok,rows,8,4096,label,2026)
        seconds=result['wall_seconds_including_prefill']-result['engine_stats']['prompt_time']
        result['late_decode_seconds_including_control']=seconds
        result['late_decode_output_tps']=result['output_tokens']/seconds
        result['checkpoint_sha256']=file_sha256(path/'model.safetensors')
        report['results'].append(result)
        out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'model':label,'late_decode_output_tps':result['late_decode_output_tps']}),flush=True)
        del model,tok;gc.collect();mx.clear_cache()
    report['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
