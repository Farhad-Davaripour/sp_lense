"""Fixed same-context hand-offs; exploratory diagnosis, not a rescue of primary."""
import os,time
START=time.monotonic()
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='4'
import ctypes,hashlib,json,threading,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
assert ROOT.name=='handoff_diagnostic_20261013' and ROOT.parent.name=='block_energy_replication_v1'
BASE=ROOT.parent;OLD=BASE/'native_cpu_20261013_retry1'
FREEZE=json.loads((ROOT/'EXECUTION_FREEZE.json').read_text())
PRIOR_SECONDS=FREEZE['prior_seconds'];PRIOR_CALLS=FREEZE['prior_calls']
STATUS={'status':'starting','PID':os.getpid(),'model_calls':0,'GPU_allocations':0,'original_primary_support':False}

def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()
def report(phase,**extra):
    STATUS.update(phase=phase,seconds=time.monotonic()-START,**extra)
    dump(ROOT/'STATUS.json',STATUS);print(json.dumps(STATUS),flush=True)
def deadline():
    if time.monotonic()-START+PRIOR_SECONDS>2400:raise TimeoutError('Combined native time budget reached')
    if 'rig' in globals() and PRIOR_CALLS+rig.calls>=300:raise RuntimeError('Combined backbone call ceiling reached')
def hard_stop():
    dump(ROOT/'HARD_STOP.json',{'PID':os.getpid()});os._exit(124)
TIMER=threading.Timer(2700-PRIOR_SECONDS,hard_stop);TIMER.daemon=True;TIMER.start()

def export():
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='EXPORT_HASHES.json')
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    dump(ROOT/'EXPORT_HASHES.json',hashes)
    archive=BASE/'R4_BLOCK_ENERGY_HANDOFF_20261013.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files+[ROOT/'EXPORT_HASHES.json']:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,expected in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==expected
    receipt={'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'members':len(files)+1,
        'all_members_and_CRC_verified':True,'model_calls':STATUS['model_calls'],
        'combined_model_calls':PRIOR_CALLS+STATUS['model_calls'],
        'native_and_export_seconds':time.monotonic()-START,
        'combined_native_and_export_seconds':PRIOR_SECONDS+time.monotonic()-START,'GPU_allocations':0,'status':STATUS['status']}
    dump(BASE/'HANDOFF_EXPORT_RECEIPT_20261013.json',receipt)
    print(json.dumps({'export_complete':receipt}),flush=True)

try:
    import numpy as np
    import torch
    from native import Rig
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available()
    for name,expected in FREEZE['source_sha256'].items():assert sha(ROOT/name)==expected
    assert sha(OLD/'EXPORT_HASHES.json')==FREEZE['original_manifest_sha256']
    assert sha(OLD/'PREDICTION_FREEZE.json')==FREEZE['original_prediction_freeze_sha256']
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ['total','available','page','avail_page','virtual','avail_virtual','extended']]
    memory=Memory();memory.length=ctypes.sizeof(memory)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    dump(ROOT/'ADMISSION.json',{'free_bytes':int(memory.available),'minimum_bytes':8589934592,
        'admitted':memory.available>=8589934592,'remaining_calls':300-PRIOR_CALLS,
        'remaining_total_native_export_seconds':2700-PRIOR_SECONDS})
    assert memory.available>=8589934592
    plan=json.loads((ROOT/'source/HANDOFF_PLAN.json').read_text())
    inputs=json.loads((OLD/'INPUTS.json').read_text())
    pin=json.loads((ROOT/'source/model_pin.json').read_text())
    weights=Path('C:/Users/farha/Research4/cache/qwen35_08b_pinned')
    for item in pin['files']:
        path=weights/item['name']
        if path.is_file() and item.get('sha256'):assert sha(path)==item['sha256']
    report('loading_model')
    rig=Rig(weights,deadline);rig.precision(torch.float32)
    (ROOT/'actual').mkdir()
    results=[];nulls=[]
    def logprob(logits):
        x=np.asarray(logits,dtype=np.float64);m=x.max();return x-(m+np.log(np.exp(x-m).sum()))
    def kl(a,b):
        lp=logprob(a);lq=logprob(b);value=float(np.exp(lp)@(lp-lq));assert value>=-1e-10;return max(0.,value)
    def response_error(base,actual,predicted):
        p=np.exp(logprob(base));a=np.asarray(actual,dtype=np.float64)-base;d=np.asarray(predicted,dtype=np.float64)-actual
        a-=p@a;d-=p@d
        return float(np.sqrt((p@(d*d))/max(p@(a*a),1e-30)))
    for question in inputs['questions']:
        pid=question['id']
        if pid not in plan['question_ids']:continue
        ids=question['prefix_ids']
        base=rig.value(ids)
        with np.load(OLD/'baseline'/f'{pid}_FP32.npz',allow_pickle=False) as z:
            assert np.array_equal(base['logits'],z['logits'])
            assert all(np.array_equal(base['points'][name],z[name]) for name in base['points'])
        null=rig.value(ids,'output',torch.zeros(1024),0.)
        assert np.array_equal(null['logits'],base['logits'])
        nulls.append({'question':pid,'baseline_exact':True,'null_exact':True})
        for di in plan['directions']:
            with np.load(OLD/'predictions'/f'{pid}_d{di}.npz',allow_pickle=False) as z:
                prediction={name:z[name].copy() for name in ['q','coefficients','directions']}
            with np.load(OLD/'actual'/f'{pid}_d{di}_FP32.npz',allow_pickle=False) as z:
                original={name:z[name].copy() for name in ['alphas','middle','output','logits']}
            for sign in plan['signs']:
                alpha=sign*plan['strength'];ai=int(np.where(original['alphas']==alpha)[0][0])
                truth_logits=original['logits'][ai]
                true_cost=kl(base['logits'],truth_logits)
                fisher=.5*alpha*alpha*prediction['coefficients'][0]
                linear_endpoint=kl(base['logits'],base['logits'].astype(np.float64)+alpha*prediction['q'][0])
                storage={};case={'question':pid,'direction':di,'alpha':alpha,'original_KL':true_cost,
                    'original_Fisher_cost':float(fisher),'linear_logit_endpoint_KL':linear_endpoint,'arms':{}}
                for arm in plan['arms']:
                    boundary='middle' if arm.endswith('middle') else 'output'
                    if arm.startswith('exact'):
                        displacement=original[boundary][ai].astype(np.float64)-base['points'][boundary]
                        vector=torch.tensor(displacement,dtype=torch.float32);scale=1.
                    else:
                        vector=torch.tensor(prediction['directions'][1 if boundary=='middle' else 2]);scale=alpha
                    measured=rig.value(ids,boundary,vector,scale)
                    assert measured['prefix_hashes']==base['prefix_hashes']
                    actual_state=original[boundary][ai].astype(np.float64)
                    target_error=float(np.linalg.norm(measured['points'][boundary].astype(np.float64)-actual_state)/max(1e-30,np.linalg.norm(actual_state-base['points'][boundary])))
                    actual_to_replayed=kl(truth_logits,measured['logits'])
                    cost=kl(base['logits'],measured['logits'])
                    info={'KL_base_to_handoff':cost,'KL_actual_to_handoff':actual_to_replayed,
                        'cost_relative_error_vs_original':abs(cost-true_cost)/true_cost,
                        'weighted_logit_response_error':response_error(base['logits'],truth_logits,measured['logits']),
                        'target_state_relative_error_vs_actual':target_error}
                    case['arms'][arm]=info
                    storage[arm+'_logits']=measured['logits']
                    storage[arm+'_middle']=measured['points']['middle']
                    storage[arm+'_output']=measured['points']['output']
                    if arm.startswith('exact'):
                        assert actual_to_replayed<=1e-7 and target_error<=1e-4,'Exact hand-off mechanics failed'
                results.append(case)
                np.savez_compressed(ROOT/'actual'/f'{pid}_d{di}_{"plus" if sign>0 else "minus"}.npz',
                    base_logits=base['logits'],truth_logits=truth_logits,**storage)
                dump(ROOT/'RESULTS.json',results)
                report('matched_handoffs',question=pid,direction=di,alpha=alpha,model_calls=rig.calls,
                    combined_model_calls=PRIOR_CALLS+rig.calls)
    assert rig.calls==36 and PRIOR_CALLS+rig.calls==300 and len(results)==8
    dump(ROOT/'MECHANICS.json',{'pass':True,'contexts':nulls,'matched_handoffs':16,
        'no_primary_result_relabeling':True,'all_cases_predeclared':True})
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in rig.model.modules())
    STATUS['status']='complete';report('complete_export_pending',model_calls=rig.calls,combined_model_calls=300)
except BaseException as error:
    if 'rig' in globals():STATUS['model_calls']=rig.calls
    STATUS['status']='fault_or_mechanics_stop';report('fault',error=type(error).__name__+': '+str(error))
    (ROOT/'FAULT.txt').write_text(traceback.format_exc());raise
finally:
    try:export()
    finally:TIMER.cancel()
