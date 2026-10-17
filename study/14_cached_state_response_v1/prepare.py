import ast,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
ROOT=Path('C:/Users/farha/Research4/artifacts/cached_state_response_v1/native_cpu_20261014')
ROOT.mkdir(parents=True,exist_ok=False)
plan=json.loads((HERE/'PLAN.json').read_text())
tokenizer=AutoTokenizer.from_pretrained(plan['checkpoint'],local_files_only=True,trust_remote_code=False)
compiled=[]
for q in plan['questions']:
    ids=tokenizer.apply_chat_template([{'role':'system','content':plan['system']},{'role':'user','content':q['text']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if hasattr(ids,'keys'):ids=ids['input_ids']
    if hasattr(ids,'tolist'):ids=ids.tolist()
    if ids and isinstance(ids[0],list):ids=ids[0]
    assert 1<len(ids)<=96;compiled.append({**q,'prefix_ids':ids,'prefix_length':len(ids)})
old=Path('C:/Users/farha/Research4/artifacts/finite_logit_response_v1/native_cpu_20261014')
manifest=json.loads((old/'EXPORT_HASHES.json').read_text())
assert hashlib.sha256((old/'DIRECTIONS.npz').read_bytes()).hexdigest()==manifest['DIRECTIONS.npz']
with np.load(old/'DIRECTIONS.npz',allow_pickle=False) as z:prior=z['q64']
draws=np.random.default_rng(plan['directions']['seed']).standard_normal((1024,2));draws-=prior@(prior.T@draws)
q,_=np.linalg.qr(draws)
for i in range(2):
    if q[:,i]@draws[:,i]<0:q[:,i]*=-1
np.savez_compressed(ROOT/'DIRECTIONS.npz',vectors=q.T.astype(np.float32),q64=q)
(ROOT/'INPUTS.json').write_text(json.dumps({'system':plan['system'],'questions':compiled,
    'gram_error':float(np.linalg.norm(q.T@q-np.eye(2))),
    'prior_span_projection_error':float(np.linalg.norm(prior.T@q)),'model_calls_before_freeze':0},indent=2)+'\n')
source=ROOT/'source';source.mkdir()
for name,target in [('native.py','base_native.py'),('model_pin.json','model_pin.json')]:
    p=old/'source'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['source/'+name]
    shutil.copyfile(p,source/target)
for name in ['PLAN.json','native.py','maths.py','prepare.py','worker.py']:
    p=HERE/name
    if p.suffix=='.py':ast.parse(p.read_text())
    shutil.copyfile(p,source/name)
files=list(source.iterdir())+[ROOT/'INPUTS.json',ROOT/'DIRECTIONS.npz']
freeze={'UTC':datetime.now(timezone.utc).isoformat(),'model_calls_before_freeze':0,
    'sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(ROOT/'EXECUTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
(HERE/'EXECUTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
print(json.dumps({'root':str(ROOT),'questions':6,'held_questions':4,'directions':2,'prefix_lengths':[q['prefix_length'] for q in compiled],'frozen_files':len(files)}))
