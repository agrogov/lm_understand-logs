"""Validate completed MacCUDA runs and compare recorded workload projections."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=HERE/'build/benchmark-base')
    parser.add_argument('--edited', type=Path, default=HERE/'build/benchmark-edited')
    parser.add_argument('--out', type=Path, default=HERE/'benchmark-results.json')
    args = parser.parse_args()
    mac = json.loads((HERE/'../macos/artifacts/runtime-estimate.json').read_text())
    cases, rates = {}, {}
    for model, directory in [('base', args.base), ('edited', args.edited)]:
        if not (directory/'completed.txt').is_file():
            raise ValueError(f'Incomplete benchmark: {directory}')
        rows = [json.loads(line) for line in (directory/'cases.jsonl').read_text().splitlines()]
        cases[model] = []
        for row in rows:
            if row['output_tokens'] != sum(s['output_tokens'] for s in row['sequences']):
                raise ValueError('Inconsistent token counts')
            cases[model].append({**{k:v for k,v in row.items() if k != 'sequences'},
                'sequences': [{k:v for k,v in s.items() if k != 'token_ids'} for s in row['sequences']]})
        long = [r for r in rows if r['batch'] == 4 and r['max_new_tokens'] == 32768]
        if len(long) != 1:
            raise ValueError('Need exactly one four-sequence 32K case per checkpoint')
        rates[model] = long[0]['output_tokens']/long[0]['wall_seconds_including_prefill']
    workloads = {}
    for label, original in mac['workloads'].items():
        counts = original['output_tokens']
        per_model = {k:counts[k]/rates[k]/3600 for k in rates}
        workloads[label] = {'output_tokens':counts, 'generation_hours_per_model':per_model,
            'egpu_generation_hours':sum(per_model.values()),
            'mlx_generation_hours':original['generation_hours']}
    fingerprints = {}
    for name in ['inference_bench', 'hardware/qwen35-base-bf16.gguf', 'hardware/qwen35-edited-bf16.gguf']:
        file = HERE/'build'/name
        with file.open('rb') as stream:
            fingerprints[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'date':'2026-11-04', 'hardware':'RTX 4090 AD102, x4 PCIe Gen3 via MacCUDA on M4 Max 36 GiB',
        'basis':'Original recorded generated-token counts divided by measured fixed-batch rates',
        'excluded':'Teacher forcing, grading, editor decisions, checkpoint IO, retries and idle time',
        'limitations':'One long batch per checkpoint; different RNG/backends and EOS occupancy; not accuracy or end-to-end validation',
        'cases':cases, 'long_rates':rates, 'workloads':workloads, 'sha256':fingerprints}
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'long_rates':rates, 'core':workloads['discovery_plus_math_gsm']}, indent=2))


if __name__ == '__main__':
    main()
