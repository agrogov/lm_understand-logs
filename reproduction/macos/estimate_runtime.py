"""Extrapolate the original recorded token workload using measured Mac rates.

This is generation time only, not an accuracy result or a full-run measurement.
"""
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',default=str(HERE/'artifacts/benchmark-long.json'))
    parser.add_argument('--short',default=str(HERE/'artifacts/benchmark-short.json'))
    parser.add_argument('--late',default=str(HERE/'artifacts/benchmark-late.json'))
    parser.add_argument('--out',default=str(HERE/'artifacts/runtime-estimate.json'))
    args=parser.parse_args()
    benchmark=json.loads(Path(args.benchmark).read_text())
    rates={}
    for r in benchmark['results']:
        if r['max_new_tokens']==32768:
            rates[r['model']]=r['aggregate_output_tps_including_prefill']
    if set(rates)!={'base','edited'}:
        raise ValueError('Need completed 32768-token-cap cases for both checkpoints')
    discovery={'base':0,'edited':0}
    passes={}
    for file in (REPO/'exp8_math/runs/p50c/requests').glob('*.done.json'):
        d=json.loads(file.read_text())
        if not d.get('ok'):continue
        typ=d['type'];passes[typ]=passes.get(typ,0)+1
        if typ!='generate':continue
        if d['model']=='both':
            for model,key in [('original','base'),('edited','edited')]:
                s=d[model];discovery[key]+=s['generations']*s['mean_tokens']
        else:
            key='base' if d['model']=='original' else 'edited'
            discovery[key]+=d['generations']*d['mean_tokens']
    workloads={'discovery':discovery}
    for bench,n in [('math500',4),('gsm8k',2)]:
        counts={}
        for model,key in [('unedited','base'),('p50c','edited')]:
            d=json.loads((REPO/f'exp8_math/eval/{model}/merged.json').read_text())[bench]
            counts[key]=round(d['problems']*n*d['mean_tokens'])
        workloads[bench]=counts
    workloads['evaluation_math_gsm']={k:workloads['math500'][k]+workloads['gsm8k'][k]
                                      for k in rates}
    workloads['discovery_plus_math_gsm']={k:discovery[k]+workloads['evaluation_math_gsm'][k]
                                         for k in rates}
    report={'benchmark':str(Path(args.benchmark).resolve()),
            'basis':'original recorded generated-token volumes / measured BF16 Mac batch throughput',
            'measured_aggregate_output_tps':rates,'original_successful_passes':passes,
            'excluded':'teacher-forced probes, grading, checkpoint IO, editor decisions, retries and idle time',
            'limitations':'one long-context batch per checkpoint; earlier edits approximated with final-edit rate; '
                          'actual Mac token volumes and continuous-batching occupancy may differ',
            'workloads':{}}
    for label,counts in workloads.items():
        per_model={k:counts[k]/rates[k]/3600 for k in rates}
        report['workloads'][label]={'output_tokens':counts,'generation_hours_per_model':per_model,
                                   'generation_hours':sum(per_model.values())}
    report['throughput_scenarios']={}
    if Path(args.short).exists():
        short=json.loads(Path(args.short).read_text())
        report['throughput_scenarios']['eight_early_context']={
            r['model']:r['aggregate_output_tps_including_prefill'] for r in short['results']
            if r['batch']==8 and r['max_new_tokens']==4096}
    if Path(args.late).exists():
        late=json.loads(Path(args.late).read_text())
        if 'finished_at' not in late:raise ValueError('Late-context benchmark is incomplete')
        report['throughput_scenarios']['eight_late_context']={
            r['model']:r['late_decode_output_tps'] for r in late['results']}
    report['scenario_generation_hours']={
        scenario:{label:sum(counts[k]/r[k]/3600 for k in r)
                  for label,counts in workloads.items()}
        for scenario,r in report['throughput_scenarios'].items()}
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
