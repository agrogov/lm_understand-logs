"""CPU-only verification of the two intervention tensors after MLX conversion."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
spec=importlib.util.spec_from_file_location('st_numpy',REPO/'exp8_math/code/st_numpy.py')
st=importlib.util.module_from_spec(spec);spec.loader.exec_module(st)
manifest=json.loads((REPO/'exp8_math/runs/p50c/edit/manifest.json').read_text())
rows=[]
for variant,field in [('base','base_sha256'),('published-edit','edited_sha256')]:
    path=HERE/f'artifacts/{variant}-mlx-bf16/model.safetensors'
    names={t['name']:t['name'].replace('model.language_model.','language_model.model.')
           for t in manifest['tensors']}
    tensors=st.load(str(path),list(names.values()))
    for tensor in manifest['tensors']:
        name=names[tensor['name']]
        dtype,array=tensors[name]
        digest=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        if dtype!='BF16' or digest!=tensor[field]:
            raise RuntimeError(f'{variant}/{name}: BF16 hash mismatch')
        rows.append({'variant':variant,'name':name,'sha256':digest,'verified':True})
report={'tensors':rows,'scope':'the two intervention matrices; not every converted tensor'}
out=HERE/'artifacts/tensor-verification.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
