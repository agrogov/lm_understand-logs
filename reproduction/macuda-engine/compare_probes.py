"""CPU/GPU numerical smoke-test comparison, not an exact cross-backend replay."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=Path,required=True)
    p.add_argument('--cpu',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    report={'gpu':str(a.gpu),'cpu':str(a.cpu),'threshold_relative_l2':.01,'tensors':[]}
    for name in ['l_out-15','l_out-19','ffn_swiglu-15','ffn_swiglu-19','last-prompt-logits']:
        x=np.fromfile(a.gpu/(name+'.f32'),dtype='<f4').astype(np.float64)
        y=np.fromfile(a.cpu/(name+'.f32'),dtype='<f4').astype(np.float64)
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid activation shapes or nonfinite values')
        relative=float(np.linalg.norm(x-y)/np.linalg.norm(y))
        report['tensors'].append({'name':name,'elements':x.size,'relative_l2':relative,
            'cosine':float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))),
            'max_abs':float(np.max(np.abs(x-y))),'passes_1pct_relative_l2':relative<.01})
    g=json.loads((a.gpu/'result.json').read_text());c=json.loads((a.cpu/'result.json').read_text())
    report['greedy_token_ids_identical']=g['token_ids']==c['token_ids']
    report['generated_tokens_compared']=len(g['token_ids'])
    report['prompt_tokens_identical']=g['prompt_tokens']==c['prompt_tokens']==124
    report['passed']=all(t['passes_1pct_relative_l2'] for t in report['tensors']) and report['greedy_token_ids_identical'] and report['prompt_tokens_identical']
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
