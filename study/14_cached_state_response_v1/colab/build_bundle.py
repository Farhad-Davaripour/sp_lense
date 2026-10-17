"""Mechanical Linux/Colab adaptation of the frozen, still-unexecuted test."""
import ast,hashlib,json,shutil,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent
BASE=Path('C:/Users/farha/Research4/artifacts/cached_state_response_v1')
OLD=BASE/'native_cpu_20261014_retry1';ROOT=BASE/'colab_prepared_20261014'/'colab_cpu_20261014'
ROOT.mkdir(parents=True,exist_ok=False);(ROOT/'source').mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((OLD/'EXPORT_HASHES.json').read_text());assert all(sha(OLD/n)==h for n,h in manifest.items())
for name in ['INPUTS.json','DIRECTIONS.npz']:shutil.copyfile(OLD/name,ROOT/name)
for name in ['base_native.py','model_pin.json','native.py','maths.py']:shutil.copyfile(OLD/'source'/name,ROOT/'source'/name)
for name in ['cached_geometry.py','POTENTIAL_DIAGNOSTIC_PLAN.json']:shutil.copyfile(SOURCE/name,ROOT/'source'/name)
shutil.copyfile(HERE/'bridge_check.py',ROOT/'source/bridge_check.py')
plan=json.loads((SOURCE/'PLAN.json').read_text());plan['checkpoint']='/content/r4_checkpoint'
plan['resources']['Colab']=True;plan['resources']['GPU_Colab_purchases_training_agents']=False
plan['resources']['platform_amendment']='Colab CPU explicitly authorized14Oct; original scientific questions/criteria unchanged;4 old-calibration bridge calls counted.'
(ROOT/'source/PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
worker=(SOURCE/'worker.py').read_text()
worker=worker.replace("'native_cpu_20261014_retry2']","'native_cpu_20261014_retry2','colab_cpu_20261014']")
worker=worker.replace("if ROOT.name.endswith('_retry2'):","if ROOT.name.endswith('_retry2') or ROOT.name=='colab_cpu_20261014':")
worker=worker.replace("def free_memory():\n", "def free_memory():\n    if os.name!='nt':return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024\n")
worker=worker.replace("suffix='_RETRY2' if", "suffix='_COLAB_CPU' if ROOT.name=='colab_cpu_20261014' else '_RETRY2' if")
worker=worker.replace("archive=ROOT.parent/f'R4_CACHED_STATE_NATIVE_20261014{suffix}.zip'", "archive=Path('/content/drive/MyDrive/sp_lense/research4/cached_state_response_v1')/f'R4_CACHED_STATE_NATIVE_20261014{suffix}.zip'")
worker=worker.replace("'Colab':False","'Colab':True")
worker=worker.replace("old=Path('C:/Users/farha/Research4/artifacts/finite_logit_response_v1')", "old=Path('/content/drive/MyDrive/sp_lense/research4/finite_logit_response_v1')")
worker=worker.replace("assert not torch.cuda.is_available() and '+cpu' in torch.__version__", "assert not torch.cuda.is_available() and torch.__version__=='2.13.0+cpu'\n    assert transformers.__version__=='5.19.0' and np.__version__=='2.5.4'")
worker=worker.replace("    for n in ['contexts','probes'", "    from bridge_check import run_bridge\n    bridge=run_bridge(rig,ROOT);report('platform_bridge_passed',model_calls=rig.calls)\n    for n in ['contexts','probes'")
# Include the small receipt in Drive while the original export timer is still live.
worker=worker.replace("    dump(ROOT.parent/f'NATIVE_EXPORT_RECEIPT_20261014{suffix}.json',receipt);", "    dump(ROOT.parent/f'NATIVE_EXPORT_RECEIPT_20261014{suffix}.json',receipt);dump(archive.parent/f'NATIVE_EXPORT_RECEIPT_20261014{suffix}.json',receipt);")
(ROOT/'source/worker.py').write_text(worker)
analyze=(SOURCE/'analyze.py').read_text().replace("BASE=Path('C:/Users/farha/Research4/artifacts/cached_state_response_v1')", "BASE=Path(__file__).resolve().parents[2]")
analyze=analyze.replace("else 'native_cpu_20261014_retry1'", "else 'colab_cpu_20261014'")
analyze=analyze.replace("'native_cpu_20261014_retry2']", "'native_cpu_20261014_retry2','colab_cpu_20261014']")
analyze=analyze.replace("suffix='_RETRY2' if", "suffix='_COLAB_CPU' if ROOT.name=='colab_cpu_20261014' else '_RETRY2' if")
analyze=analyze.replace("'Colab':False", "'Colab':True")
(ROOT/'source/analyze.py').write_text(analyze)
prior=Path('C:/Users/farha/Research4/artifacts/finite_logit_response_v1/native_cpu_20261014')
oldmanifest=json.loads((prior/'EXPORT_HASHES.json').read_text());pid='r13_cal_towel'
for name in ['baseline/'+pid+'.npz','actual/'+pid+'_d0.npz','coefficients/'+pid+'_d0.npz','DIRECTIONS.npz']:
    assert sha(prior/name)==oldmanifest[name]
with np.load(prior/'baseline'/f'{pid}.npz',allow_pickle=False) as z:reference={'base_'+n:z[n] for n in ['logits','input','middle','output','hidden']};reference['prefix_ids']=z['prefix_ids']
with np.load(prior/'actual'/f'{pid}_d0.npz',allow_pickle=False) as z:
    for alpha,label in [(.125,'plus'),(-.125,'minus')]:
        ai=list(z['alphas']).index(alpha)
        for n in ['logits','input','middle','output','hidden']:reference[label+'_'+n]=z[n][ai]
with np.load(prior/'coefficients'/f'{pid}_d0.npz',allow_pickle=False) as z:reference['q']=z['q']
with np.load(prior/'DIRECTIONS.npz',allow_pickle=False) as z:reference['direction']=z['q64'][:,0].astype(np.float32)
np.savez_compressed(ROOT/'BRIDGE_REFERENCE.npz',**reference)
probes=list(BASE.glob('RECOVERY_RUNTIME_ADMISSION_20261014*.json'))
prior_seconds=187.062+sum(json.loads(p.read_text())['preparation_seconds']+1. for p in probes)
(ROOT/'RECOVERY_BUDGET.json').write_text(json.dumps({'previous_attempt_model_calls':0,'previous_attempt_seconds':prior_seconds,
    'forward_limit':300,'bridge_calls_included':4,'total_seconds_limit':2700,'CPU_only':True,'Colab':True},indent=2)+'\n')
names=[p.name for p in Path('C:/Users/farha/Research4/cache/qwen35_08b_pinned').iterdir() if p.is_file()]
(ROOT/'source/MODEL_FILES.json').write_text(json.dumps(names,indent=2)+'\n')
amendment={'UTC':datetime.now(timezone.utc).isoformat(),'human_authorization':'Can you use colab to continue? ... Got for it.',
 'platform':'Colab Linux CPU, not GPU; same pinned torch2.13.0+cpu/transformers5.19.0/numpy2.5.4',
 'scientific_inputs_unchanged':{'INPUTS.json':sha(ROOT/'INPUTS.json'),'DIRECTIONS.npz':sha(ROOT/'DIRECTIONS.npz')},
 'changes':['Linux memory query and paths','Colab provenance/CPU archive name','Carry prior attempted time','Four counted old-calibration bridge calls before fresh held tests'],
 'original_source_sha256':{p.name:sha(p) for p in SOURCE.glob('*.py')},'model_calls_before_freeze':0,
 'primary_gates_strengths_questions_and_directions_unchanged':True,'no_Research3_or_old_monitor_changes':True}
(ROOT/'PLATFORM_AMENDMENT.json').write_text(json.dumps(amendment,indent=2)+'\n')
for p in (ROOT/'source').glob('*.py'):ast.parse(p.read_text())
files=[p for p in ROOT.rglob('*') if p.is_file()]
freeze={'UTC':datetime.now(timezone.utc).isoformat(),'model_calls_before_freeze':0,'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
(ROOT/'EXECUTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
archive=BASE/'R4_CACHED_STATE_COLAB_CPU_SOURCE_20261014.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
    for p in ROOT.rglob('*'):
        if p.is_file():z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
dest=Path('G:/My Drive/sp_lense/research4/cached_state_response_v1')/archive.name
with archive.open('rb') as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst,8*1024**2)
assert sha(dest)==sha(archive)
receipt={'path':str(archive),'drive_path':str(dest),'bytes':archive.stat().st_size,'sha256':sha(archive),'prior_charged_seconds':prior_seconds,'model_calls':0}
(BASE/'COLAB_SOURCE_RECEIPT_20261014.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
