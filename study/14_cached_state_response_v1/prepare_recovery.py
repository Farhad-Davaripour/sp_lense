"""Single bounded recovery after direct runtime-loaded memory admission."""
import os,time
PREP_STARTED=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4')
import ast,ctypes,hashlib,json,runpy,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
BASE=Path('C:/Users/farha/Research4/artifacts/cached_state_response_v1');OLD=BASE/'native_cpu_20261014_retry1'
ROOT=BASE/'native_cpu_20261014_retry2';HERE=Path(__file__).resolve().parent
assert not ROOT.exists(),'Inspect existing recovery rather than restart it'
sys.path.insert(0,str(OLD/'source'))
import numpy as np
import torch
import transformers
import native
import maths
assert '+cpu' in torch.__version__ and not torch.cuda.is_available()
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ['total','free','page','freepage','virtual','freevirtual','extended']]
m=Memory();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
admission={'UTC':datetime.now(timezone.utc).isoformat(),'free_memory_after_runtime_imports':int(m.free),'minimum_bytes':8589934592,
 'admitted':m.free>=8589934592,'model_calls':0,'torch':torch.__version__,'transformers':transformers.__version__,
 'preparation_seconds':time.monotonic()-PREP_STARTED,'CPU_only':True,'no_extra_before_import_memory_buffer':True}
PROBE=BASE/f"RECOVERY_RUNTIME_ADMISSION_20261014_{datetime.now(timezone.utc).strftime('%H%M%S%f')}.json"
with PROBE.open('x',encoding='utf-8') as f:f.write(json.dumps(admission,indent=2)+'\n')
if not admission['admitted']:
    print(json.dumps({'status':'not prepared: memory condition still blocked',**admission}))
    raise SystemExit(3)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((OLD/'EXPORT_HASHES.json').read_text());assert all(sha(OLD/n)==h for n,h in manifest.items())
prior=json.loads((BASE/'NATIVE_EXPORT_RECEIPT_20261014_RETRY1.json').read_text());assert prior['model_calls']==0
assert sha(Path(prior['path']))==prior['sha256']
ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir()
for name in ['INPUTS.json','DIRECTIONS.npz']:shutil.copyfile(OLD/name,ROOT/name)
for p in (OLD/'source').iterdir():
    use=HERE/p.name if p.name=='worker.py' else p
    if use.suffix=='.py':ast.parse(use.read_text(encoding='utf-8'))
    shutil.copyfile(use,ROOT/'source'/p.name)
for name in ['analyze.py','cached_geometry.py','POTENTIAL_DIAGNOSTIC_PLAN.json','prepare_recovery.py']:
    p=HERE/name
    if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
    shutil.copyfile(p,ROOT/'source'/name)
preparation_seconds=time.monotonic()-PREP_STARTED
prior_probe_paths=[p for p in BASE.glob('RECOVERY_RUNTIME_ADMISSION_20261014*.json') if p!=PROBE]
prior_probe_seconds=sum(float(json.loads(p.read_text())['preparation_seconds'])+1. for p in prior_probe_paths)
budget={'previous_attempt_seconds':prior['native_export_seconds']+prior_probe_seconds+preparation_seconds+1.,'previous_attempt_model_calls':0,
        'previous_workers_seconds':prior['native_export_seconds'],'preparation_seconds':preparation_seconds,'preparation_freeze_reserve_seconds':1.,
        'prior_failed_admission_probe_seconds_including_one_second_reserve_each':prior_probe_seconds,
        'prior_failed_admission_probes':[{'path':str(p),'sha256':sha(p)} for p in prior_probe_paths],
        'previous_attempt_archives':[{'path':str(BASE/n),'sha256':sha(BASE/n)} for n in ['R4_CACHED_STATE_NATIVE_20261014.zip','R4_CACHED_STATE_NATIVE_20261014_RETRY1.zip']],
        'fresh_free_memory_after_imports':int(m.free),'no_extra_before_import_memory_buffer':True,
        'after_import_min_bytes':8589934592,'combined_forward_ceiling':300,'combined_native_seconds':2400,'combined_total_seconds':2700}
(ROOT/'RECOVERY_BUDGET.json').write_text(json.dumps(budget,indent=2)+'\n')
shutil.copyfile(PROBE,ROOT/'RECOVERY_ADMISSION.json')
files=list((ROOT/'source').iterdir())+[ROOT/n for n in ['INPUTS.json','DIRECTIONS.npz','RECOVERY_BUDGET.json','RECOVERY_ADMISSION.json']]
freeze={'UTC':datetime.now(timezone.utc).isoformat(),'model_calls_before_freeze':0,
 'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
 'amendment':'Recovery allowed only after free RAM changes; unchanged questions/directions/primary criteria. Deduct all prior worker time from native and hard timers.'}
(ROOT/'EXECUTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
(HERE/'RETRY2_EXECUTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
print(json.dumps({'status':'prepared and frozen','root':str(ROOT),'budget':budget}))
if '--run' in sys.argv:
    sys.path.insert(0,str(ROOT/'source'))
    runpy.run_path(str(ROOT/'source/worker.py'),run_name='__main__')
