"""Freeze the remaining292-call native worker before fresh model execution."""
import ast,hashlib,io,json,shutil,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=Path('C:/Users/farha/Research4/artifacts/fresh_memory_boundary_response_v1')
ROOT=BASE/'fresh_prepared_20261014';SOURCE=ROOT/'fresh_source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    proof=json.loads((BASE/'REPAIR_FINAL_BACKUP_RECEIPT_20261014.json').read_text());assert proof['complete'] and proof['provider_metadata_verified']
    assert proof['phase17_calls']==8 and proof['remaining_phase17_calls']==292
    prior=json.loads((BASE/'FRESH_STAGE_PRIOR_LEDGER.json').read_text());assert prior['phase17_prior_calls']==8 and prior['phase17_remaining_calls']==292
    ROOT.mkdir(exist_ok=False);SOURCE.mkdir()
    raw=BASE/'R4_NATIVE_SEED_REPAIR_20261014.zip'
    assert raw.stat().st_size==76789098 and sha(raw)=='af1e34a1c6c869d78e97acf7ef436d758407ac32fc6e4a276b4f5a136820c775'
    with zipfile.ZipFile(raw) as z:
        manifest=json.loads(z.read('REPAIR_EXPORT_HASHES.json'))
        assert json.loads(z.read('REPAIR_RESULTS.json'))['full_control_pass']
        for name in ['native.py','base_native.py','model_pin.json','maths.py','transport_native.py']:
            key='source/'+name;data=z.read(key);assert hashlib.sha256(data).hexdigest()==manifest[key]
            (SOURCE/name).write_bytes(data)
        assert hashlib.sha256((HERE.parent/'15_cached_memory_transport_v1/transport_native.py').read_bytes()).hexdigest()==manifest['source/transport_native.py']
    for name in ['fresh_worker.py','analyze_fresh.py','boundary_native.py','response_math.py','point_store.py','lossless_arrays.py','PLAN.json']:
        shutil.copyfile(HERE/name,SOURCE/name)
    for name in ['FRESH_PREFLIGHT_20261014.json','FRESH_PREFLIGHT_FAULT_20261014.json']:
        shutil.copyfile(BASE/name,SOURCE/name)
    vector=Path('C:/Users/farha/Research4/artifacts/rotation_pilot_v1/inputs/vector_research2.json')
    assert sha(vector)=='3c611477efaa959314f4119d5b10b4dc9e3d5ebbfbfb49c4d5af0ef1c26e75aa'
    original=json.loads(vector.read_text());assert original['layer']==10 and original['model_revision']=='2fc06364715b967f1860aea9cf38778875588b17'
    v0=np.asarray(original['direction'],dtype=np.float32);assert v0.shape==(1024,)
    parent14=Path('C:/Users/farha/Research4/artifacts/cached_state_response_v1/R4_CACHED_STATE_NATIVE_20261014_COLAB_CPU_RESUME1.zip')
    with zipfile.ZipFile(parent14) as z:
        manifest=json.loads(z.read('EXPORT_HASHES.json'));data=z.read('DIRECTIONS.npz');assert hashlib.sha256(data).hexdigest()==manifest['DIRECTIONS.npz']
        with np.load(io.BytesIO(data),allow_pickle=False) as a:reference=a['vectors'][0].astype(np.float64)
    primary=v0.astype(np.float64);v1=reference-primary*(primary@reference)/(primary@primary);v1/=np.linalg.norm(v1)
    vectors=np.stack([v0,v1.astype(np.float32)]);gram=vectors.astype(np.float64)@vectors.astype(np.float64).T
    assert np.max(np.abs(gram-np.eye(2)))<2e-6 and np.array_equal(vectors[0],v0)
    np.savez_compressed(ROOT/'DIRECTIONS_FRESH.npz',vectors=vectors,gram=gram,reference_parent14=reference)
    (ROOT/'DIRECTION_PROVENANCE.json').write_text(json.dumps({'primary_file_sha256':sha(vector),'primary_original_bits_unchanged':True,
        'reference_parent14_file_sha256':manifest['DIRECTIONS.npz'],'basis_gram':gram.tolist(),'fresh_model_calls':0},indent=2)+'\n')
    shutil.copyfile(BASE/'FRESH_STAGE_PRIOR_LEDGER.json',ROOT/'FRESH_PRIOR_LEDGER.json')
    shutil.copyfile(BASE/'REPAIR_FINAL_BACKUP_RECEIPT_20261014.json',ROOT/'FRESH_PRIVATE_PROOF.json')
    for p in SOURCE.glob('*.py'):ast.parse(p.read_text())
    files=sorted(p for p in ROOT.rglob('*') if p.is_file())
    freeze={'UTC':datetime.now(timezone.utc).isoformat(),'historical_phase17_calls':8,'fresh_calls_before_freeze':0,'planned_fresh_calls':292,
        'planned_shared_phase_calls':300,'fresh_finite_target_rows':80,'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    (ROOT/'FRESH_EXECUTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
    archive=BASE/'R4_FRESH_MEMORY_BOUNDARY_SOURCE_20261014.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in sorted(ROOT.rglob('*')):
            if p.is_file():z.write(p,'colab_cpu_20261014/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    digest=sha(archive);launcher=BASE/'R4_FRESH_MEMORY_BOUNDARY_RUN_20261014.py'
    launcher.write_text('''import os,time,json,hashlib,subprocess,zipfile
from pathlib import Path
begin=time.monotonic()
base=Path('/content/research4/artifacts/fresh_memory_boundary_response_v1');root=base/'colab_cpu_20261014'
source=Path('/content/R4_FRESH_MEMORY_BOUNDARY_SOURCE_20261014.zip')
assert root.exists() and not (root/'fresh').exists() and not (root/'fresh_source').exists(),'Inspect any previous attempt; never overwrite or blindly restart'
assert json.loads((root/'REPAIR_RESULTS.json').read_text())['full_control_pass']
assert hashlib.sha256(source.read_bytes()).hexdigest()=='''+repr(digest)+'''
with zipfile.ZipFile(source) as z:
    assert z.testzip() is None
    assert all(n.startswith(root.name+'/') and (base/n).resolve().is_relative_to(base.resolve()) and not (base/n).exists() for n in z.namelist())
    z.extractall(base)
prep=time.monotonic()-begin
prior=json.loads((root/'FRESH_PRIOR_LEDGER.json').read_text())
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'4'}
with (base/'FRESH_NATIVE_LOG_20261014.txt').open('x') as log:
    proc=subprocess.Popen(['/content/r4_cached_cpu_env/bin/python','-B',str(root/'fresh_source/fresh_worker.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    print('R4_FRESH_PID',proc.pid,flush=True)
    for line in proc.stdout:log.write(line);log.flush();print(line.rstrip(),flush=True)
    code=proc.wait()
print('R4_FRESH_NATIVE_TERMINAL',code,flush=True)
(base/'FRESH_LAUNCH_LEDGER_20261014.json').write_text(json.dumps({'exit_code':code,'prior_calls':8,'preparation_seconds':prep,'elapsed_seconds':time.monotonic()-begin})+'\\n')
assert code==0,'Native files/failure remain preserved. Inspect actual call/time ledger before any recovery.'
receipt=json.loads((base/'FRESH_NATIVE_EXPORT_RECEIPT_20261014.json').read_text())
remaining=2700-receipt['shared_native_export_seconds']-prep
assert remaining>0,'Native export preserved; no analysis budget remains'
analysis_begin=time.monotonic()
with (base/'FRESH_ANALYSIS_LOG_20261014.txt').open('x') as log:
    subprocess.run(['/content/r4_cached_cpu_env/bin/python','-B',str(root/'fresh_source/analyze_fresh.py')],check=True,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
analysis_seconds=time.monotonic()-analysis_begin
out=base/'fresh_evaluated_20261014'
files={p.name:p for p in out.iterdir() if p.is_file()}
for name in ['FRESH_NATIVE_EXPORT_RECEIPT_20261014.json','FRESH_NATIVE_LOG_20261014.txt','FRESH_ANALYSIS_LOG_20261014.txt','FRESH_LAUNCH_LEDGER_20261014.json']:files['receipts/'+name]=base/name
manifest={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}
findings=base/'R4_FRESH_MEMORY_BOUNDARY_FINDINGS_20261014.zip'
with zipfile.ZipFile(findings,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
    for n,p in files.items():z.write(p,n)
    z.writestr('CONTENTS_SHA256.json',json.dumps(manifest,indent=2)+'\\n')
with zipfile.ZipFile(findings) as z:assert z.testzip() is None and all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
total=prior['phase17_prior_charged_seconds']+time.monotonic()-begin
ledger={'prior_phase_calls':8,'fresh_calls':292,'combined_calls':300,'GPU_allocations':0,'shared_compute_analysis_export_seconds':total,'analysis_seconds':analysis_seconds,'model_worker_exited':True,'network_transfer_latency_not_in_compute_timer':True}
(base/'FRESH_COMPUTE_LEDGER_20261014.json').write_text(json.dumps(ledger,indent=2)+'\\n')
assert total<2700
r=json.loads((out/'RESULTS.json').read_text())
print('R4_FRESH_COMPLETE',json.dumps({'findings':findings.name,'findings_bytes':findings.stat().st_size,'findings_sha256':hashlib.sha256(findings.read_bytes()).hexdigest(),'ledger':ledger,'primary_support':r['primary_support'],'mixed_support':r['mixed_support'],'fallbacks':[r['T1_EOS_fallbacks'],r['T2_EOS_fallbacks']],'audit':r['audit']}),flush=True)
''',encoding='utf-8')
    ast.parse(launcher.read_text())
    receipt={'source':str(archive),'source_bytes':archive.stat().st_size,'source_sha256':digest,'launcher':str(launcher),
             'launcher_bytes':launcher.stat().st_size,'launcher_sha256':sha(launcher),'fresh_native_calls_before_launch':0,'planned_fresh_calls':292,
             'shared_phase_calls':300,'prior_charged_seconds':prior['phase17_prior_charged_seconds']}
    (BASE/'FRESH_PREPARATION_RECEIPT_20261014.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
