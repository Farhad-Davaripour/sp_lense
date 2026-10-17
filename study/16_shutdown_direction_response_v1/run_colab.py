"""Run the frozen bounded phase in the existing admitted CPU runtime."""
import os
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

BASE=Path('/content/research4/artifacts/shutdown_direction_response_v1')
ROOT=BASE/'colab_cpu_20261014'
PY=Path('/content/r4_cached_cpu_env/bin/python')
SOURCE=Path('/content/R4_SHUTDOWN_DIRECTION_SOURCE_20261014.zip')
SOURCE_SHA='b61b939d372551d8596108c6986fc4df8f03ad96fd487878a1655c66f28341aa'
PROOF=Path('/content/R4_STUDY14_DURABLE_PROOF_20261014.json')
assert PROOF.is_file() and json.loads(PROOF.read_text())['complete']
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
assert PY.is_file() and not ROOT.exists(),'Preserve any previous attempt; do not relaunch blindly'
BASE.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(SOURCE) as z:
    assert z.testzip() is None
    for n in z.namelist():
        assert (BASE/n).resolve().is_relative_to(BASE.resolve()) and n.startswith(ROOT.name+'/')
    z.extractall(BASE)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'4'}
with (BASE/'NATIVE_LOG_20261014.txt').open('x') as log:
    proc=subprocess.Popen([str(PY),'-B',str(ROOT/'source/worker.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    print('R4_SEMANTIC_PID',proc.pid,flush=True)
    for line in proc.stdout:log.write(line);log.flush();print(line.rstrip(),flush=True)
    code=proc.wait()
print('R4_SEMANTIC_NATIVE_TERMINAL',code,flush=True)
assert code==0,'Inspect preserved failure/export/call ledger before any further run'
subprocess.run([str(PY),'-B',str(ROOT/'source/analyze.py')],env=env,check=True)
out=BASE/'evaluated_20261014'
archive=BASE/'R4_SHUTDOWN_DIRECTION_FINDINGS_20261014.zip'
files={p.name:p for p in out.iterdir() if p.is_file()}
for name in ['NATIVE_EXPORT_RECEIPT_20261014.json','NATIVE_LOG_20261014.txt']:files['receipts/'+name]=BASE/name
manifest={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
    for n,p in files.items():z.write(p,n)
    z.writestr('CONTENTS_SHA256.json',json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
r=json.loads((out/'RESULTS.json').read_text())
print('R4_SEMANTIC_COMPLETE',json.dumps({'findings':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
    'model_worker_exited':True,'model_calls':r['model_calls'],'mathematical_primary_support':r['mathematical_primary_support'],
    'held_contexts_passing':r['held_contexts_passing'],'behavioral_selectivity_support':r['behavioral_selectivity_support'],
    'held_scenarios_selective':r['held_scenarios_selective'],'by_strength':r['by_strength'],'behavior':r['behavior'],
    'audit':r['audit'],'response_norm_accounting':r['response_norm_accounting']}),flush=True)
