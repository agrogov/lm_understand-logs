"""Compare completed native-CUDA benchmarks with the recorded Mac projections."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=HERE/'remote-results')
    p.add_argument('--out', type=Path, default=HERE/'remote-benchmark-results.json')
    p.add_argument('--reference-cuda', type=Path)
    a = p.parse_args()
    previous = json.loads((HERE/'../macuda-engine/benchmark-results.json').read_text())
    cases, rates, metadata = {}, {}, {}
    for model in ['base','edited']:
        report = json.loads((a.results/f'{model}.json').read_text())
        if not report.get('completed'):
            raise ValueError(f'{model} did not finish all cases')
        metadata[model] = {k:v for k,v in report.items() if k != 'results'}
        cases[model] = []
        for row in report['results']:
            if row['output_tokens'] != sum(s['output_tokens'] for s in row['sequences']):
                raise ValueError('Token-count mismatch')
            cases[model].append({**{k:v for k,v in row.items() if k != 'sequences'},
                'sequences':[{k:v for k,v in s.items() if k != 'token_ids'} for s in row['sequences']]})
        long = [r for r in report['results'] if r['batch']==4 and r['max_new_tokens']==32768]
        if len(long) != 1:
            raise ValueError('Need exactly one completed four-sequence full-cap case')
        rates[model] = long[0]['output_tokens']/long[0]['wall_seconds_including_prefill']
    if metadata['base']['gpu'] != metadata['edited']['gpu']:
        raise ValueError('Both checkpoints must be benchmarked on the same GPU')
    reference = json.loads(a.reference_cuda.read_text()) if a.reference_cuda else None
    workloads = {}
    for label, old in previous['workloads'].items():
        counts = old['output_tokens']
        per_model = {k:counts[k]/rates[k]/3600 for k in rates}
        workloads[label] = {**{k:v for k,v in old.items() if k != 'generation_hours_per_model'},
                            'egpu_generation_hours_per_model':old['generation_hours_per_model'],
                            'native_cuda_generation_hours_per_model':per_model,
                            'native_cuda_generation_hours':sum(per_model.values())}
        if reference:
            ref = reference['workloads'][label]
            if ref['output_tokens'] != counts:
                raise ValueError('Reference workload counts differ')
            workloads[label]['reference_cuda_generation_hours'] = ref['native_cuda_generation_hours']
    result = {'date':'2026-11-04','host':'root@192.168.31.52', 'metadata':metadata,
              'image_id':json.loads((a.results/'image-inspect.json').read_text())[0]['Id'],
              'local_source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                               for name in ['Dockerfile','benchmark_cuda.py','requirements.lock']},
              'cases':cases,'long_rates':rates,'workloads':workloads,
              'limitations':'Generation-only projection, one long batch per checkpoint; RNG/output lengths differ across backends; not an end-to-end experiment'}
    if reference:
        result['reference_cuda_gpu'] = reference['metadata']['base']['gpu']
    run_config = a.results/'run-config.txt'
    if run_config.exists():
        result['launch_configuration'] = dict(line.split('=', 1)
            for line in run_config.read_text().splitlines() if '=' in line)
    image_sources = a.results/'image-source-sha256.txt'
    if image_sources.exists():
        result['image_source_sha256'] = {Path(file).name:digest
            for digest,file in (line.split() for line in image_sources.read_text().splitlines() if line.strip())}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'rates':rates,'core':workloads['discovery_plus_math_gsm']},indent=2))


if __name__ == '__main__':
    main()
