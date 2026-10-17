"""One bounded CPU continuation in the existing pinned runtime, with preservation."""
import os,time,json,hashlib,subprocess,zipfile
from pathlib import Path
BEGIN=time.monotonic()
BASE=Path('/content/research4/artifacts/cached_memory_transport_v1');ROOT=BASE/'colab_cpu_20261014'
PY=Path('/content/r4_cached_cpu_env/bin/python');SOURCE=Path('/content/R4_MEMORY_TRANSPORT_SOURCE_20261014_V2.zip')
assert PY.is_file() and not ROOT.exists(),'Inspect any existing attempt before continuing; never overwrite or relaunch blindly'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='98b5b63ee261016c7be5d5cad6603f6d292c8005edf9309f4d2ca44ecf0977ed'
BASE.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(SOURCE) as z:
    assert z.testzip() is None
    for n in z.namelist():assert n.startswith(ROOT.name+'/') and (BASE/n).resolve().is_relative_to(BASE.resolve())
    z.extractall(BASE)
prep=time.monotonic()-BEGIN
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'4'}
with (BASE/'NATIVE_LOG_20261014.txt').open('x') as log:
    proc=subprocess.Popen([str(PY),'-B',str(ROOT/'source/worker.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    print('R4_MEMORY_PID',proc.pid,flush=True)
    for line in proc.stdout:log.write(line);log.flush();print(line.rstrip(),flush=True)
    code=proc.wait()
print('R4_MEMORY_NATIVE_TERMINAL',code,flush=True)
assert code==0,'Inspect preserved failure and cumulative ledger before any further work'
receipt=json.loads((BASE/'NATIVE_EXPORT_RECEIPT_20261014.json').read_text())
remaining=2700-receipt['shared_native_export_seconds']-prep
assert remaining>0,'Compute/export budget exhausted; native data remain preserved'
analysis_start=time.monotonic()
subprocess.run([str(PY),'-B',str(ROOT/'source/analyze.py')],check=True,env=env,timeout=max(1,remaining))
analysis_seconds=time.monotonic()-analysis_start
OUT=BASE/'evaluated_20261014'
archive=BASE/'R4_MEMORY_TRANSPORT_FINDINGS_20261014.zip'
files={p.name:p for p in OUT.iterdir() if p.is_file()}
for name in ['NATIVE_EXPORT_RECEIPT_20261014.json','NATIVE_LOG_20261014.txt']:files['receipts/'+name]=BASE/name
manifest={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
    for n,p in files.items():z.write(p,n)
    z.writestr('CONTENTS_SHA256.json',json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(archive) as z:assert z.testzip() is None and all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
shared_total=receipt['parent_native_export_seconds']+time.monotonic()-BEGIN
assert shared_total<2700
ledger={'source_preparation_seconds':prep,'analysis_seconds':analysis_seconds,'shared_compute_analysis_findings_seconds':shared_total,
    'shared_model_calls':receipt['model_calls'],'new_model_calls':receipt['new_calls'],'model_worker_exited':True,'GPU_allocations':0,
    'network_transfer_latency_not_in_compute_timer':True}
(BASE/'COMPUTE_LEDGER_20261014.json').write_text(json.dumps(ledger,indent=2)+'\n')
r=json.loads((OUT/'RESULTS.json').read_text())
print('R4_MEMORY_COMPLETE',json.dumps({'findings':str(archive),'findings_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
    'model_worker_exited':True,'compute_ledger':ledger,
    'summary':{k:r[k] for k in ['quadratic_primary_support','quadratic_held_questions_passing','cubic_secondary_support','cubic_held_questions_passing',
    'state_fields','memory_fields','EOS_fallback_cases','sampled_held_anisotropy','mixed_summary','by_strength']},
    'chain_pass':r['chain_control']['pass'],'chain_components':r['chain_control'].get('component_norms'),
    'conditional_cost':r['conditional_cost'],'audit':r['audit']}),flush=True)
