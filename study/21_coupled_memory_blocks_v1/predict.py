"""Sealed NumPy-only predictor: no access to native target records."""
import os,time
os.environ['OPENBLAS_NUM_THREADS']='2'
BEGIN=time.monotonic()
import sys,json,hashlib
from pathlib import Path
import numpy as np
from equations import trajectory,finite_terms,potential
ROOT=Path(sys.argv[1]);OUT=ROOT/'predictions';OUT.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,a):np.savez_compressed(p,**a)
freeze=read(ROOT/'NORMAL_INPUT_FREEZE.json')
for n,h in freeze['sha256'].items():assert sha(ROOT/n)==h,n
with np.load(ROOT/'PARAMETERS.npz',allow_pickle=False) as z:
    parameters={i:{n.split(':',1)[1]:z[n].astype(np.float64) for n in z.files if n.startswith(str(i)+':')} for i in [12,13]}
with np.load(ROOT/'DIRECTIONS.npz',allow_pickle=False) as z:directions=z['directions'].copy()
inputs=read(ROOT/'INPUTS.json');cases=read(ROOT/'CASES.json');metadata=[]
for question in inputs['questions']:
    pid=question['id']
    with np.load(ROOT/'normal_inputs'/f'{pid}.npz',allow_pickle=False) as z:
        assert set(z.files)=={'boundary_t1','boundary_t2','initial12:S','initial12:window','initial13:S','initial13:window'}
        boundaries=[z[f'boundary_t{t}'].copy() for t in [1,2]]
        initial={i:{n:z[f'initial{i}:{n}'].copy() for n in ['S','window']} for i in [12,13]}
    normal=trajectory(boundaries,initial,parameters)
    for ti in range(2):
        save(OUT/f'{pid}_normal_t{ti+1}.npz',{**normal[ti]['fields'],**{f'd{i}:{n}':a for i,d in normal[ti]['drivers'].items() for n,a in d.items()}})
    for c in [x for x in cases if x['question']==pid]:
        direction=directions[c['direction']];alpha=c['alpha'];closed=trajectory(boundaries,initial,parameters,direction,alpha)
        controls={k:trajectory(boundaries,initial,parameters,direction,alpha,k,normal) for k in ['frozenK','frozenKF','reset']}
        for ti in range(2):
            a={**closed[ti]['fields'],**{f'd{i}:{n}':v for i,d in closed[ti]['drivers'].items() for n,v in d.items()}}
            for mode,output in controls.items():
                for name in ['b13_S','b13_output']:a[mode+':'+name]=output[ti]['fields'][name]
            for i in [12,13]:
                old0=initial[i]['S'] if ti==0 else normal[ti-1]['state'][i]['S']
                olda=initial[i]['S'] if ti==0 else closed[ti-1]['state'][i]['S']
                terms=finite_terms(old0,olda,normal[ti]['drivers'][i],closed[ti]['drivers'][i])
                target=closed[ti]['state'][i]['S']-normal[ti]['state'][i]['S']
                total=sum(terms.values());err=float(np.linalg.norm(total-target));assert err<=1e-10*max(1.,float(np.linalg.norm(target)))
                metadata.append({**c,'token':ti+1,'layer':i,'finite_decomposition_error':err,'term_norms':{n:float(np.linalg.norm(v)) for n,v in terms.items()},'term_sha256':{n:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for n,v in terms.items()},'predicted_change_norm':float(np.linalg.norm(target)), 'full_term_storage':'Exactly reproducible from archived complete states/drivers and equations; full-vector digests retained.'})
            save(OUT/f'case_{c["index"]:03d}_t{ti+1}.npz',a)
    print('R4_COUPLED_PREDICTED',pid,flush=True)
dump(OUT/'FINITE_RESPONSE_TERMS.json',metadata)
assert len(metadata)==96 and 'torch' not in sys.modules
dump(OUT/'PREDICTOR_RECEIPT.json',{'complete':True,'seconds':time.monotonic()-BEGIN,'model_forwards':0,'numpy_only':True,'normal_question_packets':6,'trajectories':24,'all_predictions_before_native_targets':True,'input_packet_fields':'Only initial pair memories and normal upstream boundary inputs, plus original public parameters and known nudge.'})
dump(ROOT/'PREDICTION_FREEZE.json',{'all_targets_seen':0,'all_predictions_precede_any_native_steered_target':True,'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in OUT.iterdir() if p.is_file()}})
print('R4_COUPLED_PREDICTION_FREEZE',sha(ROOT/'PREDICTION_FREEZE.json'),flush=True)
