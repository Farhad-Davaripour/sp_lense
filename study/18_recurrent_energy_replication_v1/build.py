"""Freeze replay code, parameter fidelity and exact parent identities."""
import ast,hashlib,json,shutil,zipfile
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('C:/Users/farha/Research4/artifacts/recurrent_energy_replication_v1')
ROOT=BASE/'prepared_20261015';SRC=ROOT/'source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    proof=json.loads((BASE.parent/'fresh_memory_boundary_response_v1/FRESH_FINAL_BACKUP_RECEIPT_20261014.json').read_text())
    assert proof['complete'] and proof['provider_metadata_verified'] and proof['files'][0]['sha256']=='1364223250a6d98af7524b20b3bee328861481af7e0e6d57f9e808432e3577fb'
    pp=json.loads((BASE/'PARAMETER_PREP_LEDGER_20261015.json').read_text());assert pp['all115_parameter_arrays_bits_verified'] and pp['provider_size_parent_and_shared_false_verified']
    ROOT.mkdir(exist_ok=False);SRC.mkdir()
    for n in ['worker.py','memory_equations.py','PLAN.json']:shutil.copyfile(HERE/n,SRC/n)
    shutil.copyfile(HERE.parent/'17_fresh_memory_boundary_response_v1/output_store.py',SRC/'output_store.py')
    for n in ['EFFECTIVE_PARAMETERS.npz','PARAMETER_RECEIPT.json']:shutil.copyfile(BASE/n,ROOT/n)
    shutil.copyfile(BASE.parent/'fresh_memory_boundary_response_v1/FRESH_FINAL_BACKUP_RECEIPT_20261014.json',ROOT/'PARENT_PRIVATE_PROOF.json')
    shutil.copyfile(BASE/'PARAMETER_PREP_LEDGER_20261015.json',ROOT/'PRIOR_SETUP.json')
    preflight=json.loads((BASE/'REPLAY_PREFLIGHT_20261015.json').read_text())
    prior=json.loads((ROOT/'PRIOR_SETUP.json').read_text())
    prior['synthetic_preflight_seconds']=preflight['seconds'];prior['prior_setup_seconds']+=preflight['seconds']
    (ROOT/'PRIOR_SETUP.json').write_text(json.dumps(prior,indent=2)+'\n')
    shutil.copyfile(BASE/'REPLAY_PREFLIGHT_20261015.json',ROOT/'REPLAY_PREFLIGHT.json')
    for p in SRC.glob('*.py'):ast.parse(p.read_text())
    files=sorted(p for p in ROOT.rglob('*') if p.is_file())
    freeze={'UTC':datetime.now(timezone.utc).isoformat(),'new_backbone_forwards':0,'native_energy_replays_before_freeze':0,
            'expected_main_head_updates':26496,'expected_derivative_heads':4608,'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    (ROOT/'EXECUTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
    archive=BASE/'R4_RECURRENT_ENERGY_REPLAY_SOURCE_20261015.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in ROOT.rglob('*'):
            if p.is_file():z.write(p,'colab_cpu_20261015/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    digest=sha(archive);launcher=BASE/'R4_RECURRENT_ENERGY_REPLAY_RUN_20261015.py'
    launcher.write_text('''import os,time,json,hashlib,subprocess,zipfile
from pathlib import Path
begin=time.monotonic()
base=Path('/content/research4/artifacts/recurrent_energy_replication_v1');root=base/'colab_cpu_20261015'
source=Path('/content/R4_RECURRENT_ENERGY_REPLAY_SOURCE_20261015.zip')
assert not root.exists(),'Inspect prior native replay/ledger; do not overwrite or blindly repeat'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='''+repr(digest)+'''
with zipfile.ZipFile(source) as z:
    assert z.testzip() is None
    assert all(n.startswith(root.name+'/') and (base/n).resolve().is_relative_to(base.resolve()) for n in z.namelist())
    z.extractall(base)
with (base/'REPLAY_LOG_20261015.txt').open('x') as log:
    proc=subprocess.Popen(['/content/r4_cached_cpu_env/bin/python','-B',str(root/'source/worker.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'2'})
    print('R4_MEMORY_ENERGY_PID',proc.pid,flush=True)
    for line in proc.stdout:log.write(line);log.flush();print(line.rstrip(),flush=True)
    code=proc.wait()
print('R4_MEMORY_ENERGY_TERMINAL',code,flush=True)
assert code==0,'Original replay data/failure preserved; inspect before recovery'
r=json.loads((root/'RESULTS.json').read_text())
print('R4_MEMORY_ENERGY_COMPLETE',json.dumps({k:v for k,v in r.items() if k not in ['primary_failed_head_records','failed_derivative_records']}),flush=True)
''')
    ast.parse(launcher.read_text())
    receipt={'source':str(archive),'bytes':archive.stat().st_size,'sha256':digest,'launcher':str(launcher),'launcher_sha256':sha(launcher),'expected_new_backbone_forwards':0}
    (BASE/'PREPARATION_RECEIPT_20261015.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
