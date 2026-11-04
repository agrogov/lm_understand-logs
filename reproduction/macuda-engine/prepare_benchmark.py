"""Prepare the same alternating GSM/MATH practice prompts used by the MLX test."""
import json
from pathlib import Path
from transformers import AutoTokenizer

root=Path(__file__).resolve().parent.parent.parent
out=root/'reproduction/macuda-engine/build/benchmark-prompts'
out.mkdir(parents=True,exist_ok=True)
request=json.loads((root/'exp8_math/runs/p50c/requests/fwd_1.json.claimed').read_text())
bank={}
for file in (root/'exp8_math/practice').glob('*.jsonl'):
    for line in file.open():
        if line.strip():
            r=json.loads(line);bank[r['id']]=r
groups=[[bank[k] for k in request['problems'] if k.startswith(prefix)] for prefix in ['gsm8k_','math_']]
rows=[r for pair in zip(*groups) for r in pair][:32]
tok=AutoTokenizer.from_pretrained(root/'reproduction/macos/artifacts/base-hf')
for i,r in enumerate(rows):
    text=tok.apply_chat_template([{'role':'user','content':r['problem']+'\nPlease reason step by step, and put your final answer within \\boxed{}.'}],
                                  tokenize=False,add_generation_prompt=True,enable_thinking=True)
    (out/f'{i}.txt').write_text(text)
(out/'manifest.json').write_text(json.dumps([{'slot':i,'id':r['id']} for i,r in enumerate(rows)],indent=2)+'\n')
print(f'Prepared {len(rows)} pinned-tokenizer practice prompts in {out}')
