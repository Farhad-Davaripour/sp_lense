"""Freeze stage one only after complete private preservation of study15."""
import ast,hashlib,json,shutil,zipfile
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('C:/Users/farha/Research4/artifacts/fresh_memory_boundary_response_v1')
P15=Path('C:/Users/farha/Research4/artifacts/cached_memory_transport_v1')
ROOT=BASE/'repair_prepared_20261014'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    proof=json.loads((P15/'FINAL_BACKUP_RECEIPT_20261014.json').read_text())
    assert proof['provider_metadata_verified'] and proof['private_Drive_full_readback_verified'] and proof['host_raw_transfer_complete']
    assert proof['files'][0]['bytes']==2735814453 and proof['files'][0]['sha256']=='be2cd385724c3b49ca6c27719d699bec93fee6a5a1c99bfc34eeb69da0e47658'
    ROOT.mkdir(exist_ok=False);source=ROOT/'source';source.mkdir()
    with zipfile.ZipFile(P15/'R4_MEMORY_TRANSPORT_NATIVE_20261014.zip') as z:
        manifest=json.loads(z.read('EXPORT_HASHES.json'))
        for name in ['base_native.py','native.py','model_pin.json']:
            key='source/'+name;data=z.read(key);assert hashlib.sha256(data).hexdigest()==manifest[key]
            (source/name).write_bytes(data)
    for name in ['repair_control.py','lossless_arrays.py','boundary_native.py','response_math.py']:
        shutil.copyfile(HERE/name,source/name)
    shutil.copyfile(HERE.parent/'15_cached_memory_transport_v1/transport_native.py',source/'transport_native.py')
    shutil.copyfile(HERE.parent/'14_cached_state_response_v1/maths.py',source/'maths.py')
    shutil.copyfile(HERE/'PLAN.json',source/'PLAN.json')
    shutil.copyfile(P15/'FINAL_BACKUP_RECEIPT_20261014.json',ROOT/'PRIOR_BACKUP.json')
    evidence=ROOT/'evidence';evidence.mkdir()
    for parent,names in [(P15,['SEED_STORAGE_OWNERSHIP_PROBE_20261014.json','SEED_OWNERSHIP_FIX_PLAN_20261014.json']),
                         (BASE,['LOSSLESS_FORMAT_PROBE_20261014_V2.json','MATH_FORMULA_PROBE_20261014.json','PRE_AMENDMENT_PLAN_20261014.json'])]:
        for name in names:shutil.copyfile(parent/name,evidence/name)
    for p in source.glob('*.py'):ast.parse(p.read_text())
    files=sorted(p for p in ROOT.rglob('*') if p.is_file())
    freeze={'UTC':datetime.now(timezone.utc).isoformat(),'phase17_model_calls_before_freeze':0,'fresh_model_calls_before_freeze':0,
            'stage_one_forward_ceiling':8,'shared_phase17_forward_ceiling':300,'stage_two_remaining_ceiling':292,
            'source_for_stage_two_will_be_frozen_before_fresh_execution':True,
            'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    (ROOT/'REPAIR_EXECUTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
    archive=BASE/'R4_NATIVE_SEED_REPAIR_SOURCE_20261014.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in sorted(ROOT.rglob('*')):
            if p.is_file():z.write(p,'colab_cpu_20261014/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    digest=sha(archive)
    launcher=BASE/'R4_NATIVE_SEED_REPAIR_RUN_20261014.py'
    launcher.write_text('''import os,time,json,hashlib,subprocess,zipfile
from pathlib import Path
begin=time.monotonic()
base=Path('/content/research4/artifacts/fresh_memory_boundary_response_v1');root=base/'colab_cpu_20261014'
source=Path('/content/R4_NATIVE_SEED_REPAIR_SOURCE_20261014.zip')
assert not root.exists(),'Inspect prior run and ledger; never blindly relaunch'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='''+repr(digest)+'''
base.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(source) as z:
    assert z.testzip() is None
    assert all(n.startswith(root.name+'/') and (base/n).resolve().is_relative_to(base.resolve()) for n in z.namelist())
    z.extractall(base)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'4'}
with (base/'REPAIR_NATIVE_LOG_20261014.txt').open('x') as log:
    proc=subprocess.Popen(['/content/r4_cached_cpu_env/bin/python','-B',str(root/'source/repair_control.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    print('R4_NATIVE_REPAIR_PID',proc.pid,flush=True)
    for line in proc.stdout:log.write(line);log.flush();print(line.rstrip(),flush=True)
    code=proc.wait()
print('R4_NATIVE_REPAIR_TERMINAL',code,flush=True)
receipt=json.loads((base/'REPAIR_EXPORT_RECEIPT_20261014.json').read_text())
receipt['stage_one_including_launcher_seconds']=time.monotonic()-begin
receipt['launcher_exit_code']=code
(base/'REPAIR_COMPUTE_LEDGER_20261014.json').write_text(json.dumps(receipt,indent=2)+'\\n')
if code==0:
    result=json.loads((root/'REPAIR_RESULTS.json').read_text())
    print('R4_NATIVE_REPAIR_COMPLETE',json.dumps({'receipt':receipt,'result':{k:v for k,v in result.items() if k!='storage'}}),flush=True)
assert code==0,'Preserve and inspect the fault before any continuation'
''',encoding='utf-8')
    ast.parse(launcher.read_text())
    receipt={'source':str(archive),'source_bytes':archive.stat().st_size,'source_sha256':digest,
             'launcher':str(launcher),'launcher_sha256':sha(launcher),'stage_one_calls':8,'shared_phase_ceiling':300,
             'model_calls_before_launch':0,'prior_private_backup_complete':True}
    (BASE/'REPAIR_PREPARATION_RECEIPT_20261014.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
