"""Compile/hash inputs and copy verified machinery BEFORE new model evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil
import numpy as np
from transformers import AutoTokenizer

HERE=Path(__file__).parent
ROOT=Path('C:/Users/farha/Research4/artifacts/norm_response_v2/prospective_20261012')
ROOT.mkdir(parents=True,exist_ok=False)
plan=json.loads((HERE/'PLAN.json').read_text())
checkpoint=Path('C:/Users/farha/Research4/cache/qwen35_08b_pinned')
tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
compiled=[]
for q in plan['questions']:
    ids=tokenizer.apply_chat_template([{'role':'system','content':plan['system']},{'role':'user','content':q['text']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if hasattr(ids,'keys'):ids=ids['input_ids']
    if hasattr(ids,'tolist'):ids=ids.tolist()
    if ids and isinstance(ids[0],list):assert len(ids)==1;ids=ids[0]
    assert 1<len(ids)<=96 and all(isinstance(t,int) for t in ids)
    compiled.append({**q,'prefix_ids':ids,'prefix_length':len(ids),'cached_token_id':760})
prior=Path('C:/Users/farha/Research4/artifacts/energy_response_v1/pilot_20261012')
prior_manifest=json.loads((prior/'EXPORT_HASHES.json').read_text())
prior_vectors=prior/'DIRECTIONS.npz'
assert hashlib.sha256(prior_vectors.read_bytes()).hexdigest()==prior_manifest['DIRECTIONS.npz']
with np.load(prior_vectors,allow_pickle=False) as z:old=z['q64'].copy()
draws=np.random.default_rng(plan['directions']['seed']).standard_normal((1024,4))
orthogonal=draws-old@(old.T@draws);q,_=np.linalg.qr(orthogonal)
for i in range(4):
    if q[:,i]@draws[:,i]<0:q[:,i]*=-1
np.savez_compressed(ROOT/'DIRECTIONS.npz',q64=q,directions_FP32=q.T.astype(np.float32))
(ROOT/'INPUTS.json').write_text(json.dumps({'system':plan['system'],'questions':compiled,'native_model_calls_before_freeze':0,
    'old_span_projection_error':float(np.linalg.norm(old.T@q)),'direction_Gram_error':float(np.linalg.norm(q.T@q-np.eye(4))),
    'prior_direction_file_sha256':hashlib.sha256(prior_vectors.read_bytes()).hexdigest()},indent=2)+'\n')
target=ROOT/'source';target.mkdir()
for name in ['PLAN.json','LITERATURE.json','prepare.py','equations.py','worker.py','native_v2.py']:
    p=HERE/name
    if p.suffix=='.py':ast.parse(p.read_text())
    shutil.copyfile(p,target/name)
for name in ['maths.py','model_pin.json']:
    oldfile=prior/'source'/name;assert hashlib.sha256(oldfile.read_bytes()).hexdigest()==prior_manifest['source/'+name]
    shutil.copyfile(oldfile,target/name)
files=[p for p in target.iterdir() if p.is_file()]+[ROOT/'INPUTS.json',ROOT/'DIRECTIONS.npz']
freeze={'frozen_UTC':datetime.now(timezone.utc).isoformat(),'sources':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'new_native_model_calls_before_freeze':0}
(ROOT/'EXECUTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
(HERE/'EXECUTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
print(json.dumps({'questions':len(compiled),'calibration':4,'held_out':4,'prefix_lengths':[p['prefix_length'] for p in compiled],
    'directions':4,'inputs_and_sources_frozen':len(files),'new_model_calls_before_freeze':0}))
