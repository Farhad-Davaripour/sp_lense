"""Independent vector/gate and dense-matrix energy audit of sealed forecasts."""
import os,time
os.environ['OPENBLAS_NUM_THREADS']='2'
BEGIN=time.monotonic()
import sys,json,hashlib
from pathlib import Path
import numpy as np
R=Path('/content/research4/artifacts/coupled_memory_blocks_v1/run_20261015')
S=R.parent/'run_20261015_resume1'
OUT=R.parent/'audit_20261015';OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(R/'source'))
from equations import finite_terms
def read(p):return json.loads(p.read_text())
def scalar(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=scalar)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
def norm(a):return float(np.linalg.norm(np.asarray(a,dtype=np.float64)))
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
for freeze in ['EXECUTION_FREEZE.json','NORMAL_INPUT_FREEZE.json','PREDICTION_FREEZE.json']:
    for n,h in read(R/freeze)['sha256'].items():assert sha(R/n)==h,(freeze,n)
for n,h in read(R/'NATIVE_BASELINE_FREEZE.json').items():assert sha(R/n)==h,n
for n,r in read(R.parent/'stopped_run_audit_20261015/ORIGINAL_STOP_HASHES.json').items():assert sha(R/n)==r['sha256'],n
for n,h in read(S/'EXPORT_HASHES.json').items():assert sha(S/n)==h,n
target_map=read(S/'TARGET_FILE_MAP.json')
assert len(target_map)==48 and target_map['0:1']==str(R/'native_targets/case_000_t1.npz')
pf=read(R/'PREDICTION_FREEZE.json');assert pf['all_predictions_precede_any_native_steered_target'] and pf['all_targets_seen']==0
pred_latest=max([p.stat().st_mtime_ns for p in (R/'predictions').iterdir()]+[(R/'PREDICTION_FREEZE.json').stat().st_mtime_ns])
target_earliest=min(Path(p).stat().st_mtime_ns for p in target_map.values());assert pred_latest<=target_earliest
receipt=read(R/'predictions/PREDICTOR_RECEIPT.json');assert receipt['numpy_only'] and receipt['model_forwards']==0
cases=read(R/'CASES.json');inputs=read(R/'INPUTS.json');native_summary=read(S/'RESULTS.json');assert len(cases)==24 and native_summary['native_backbone_calls']==76 and native_summary['resume_native_calls']==52
rows=read(S/'FIELD_COMPARISONS.json');lookup={(r['index'],r['token'],r['field']):r for r in rows};assert len(lookup)==672
energy_lookup={(r['question'],r.get('index',-1),r['token'],r['layer'],r['head']):r for r in read(S/'NATIVE_ENERGY_HEADS.json')}
term_lookup={(r['index'],r['token'],r['layer']):r for r in read(R/'predictions/FINITE_RESPONSE_TERMS.json')}
FIELDS=[f'b{i}_{n}' for i in [12,13] for n in ['input','mixer','middle','mlp','output','window','S']]
audit={'new_backbone_forwards':0,'GPU_allocations':0,'sealed_predictions_before_targets':True,'whitelisted_packet_fields_verified':True,'full_value_and_change_fields_rechecked':0,'baseline_fields_rechecked':0,'control_vectors_rechecked':0,'term_vectors_regenerated_and_hash_verified':0,'dense_energy_head_checks':0,'maximum_gate_scalar_discrepancy':0.,'all_original_gate_flags_reconfirmed':True,'all_dense_energy_checks_pass':True,'all_required_signals_resolved':True}
geometry=[];control_rows=[];moving=[];fields_failed=[]
def K(d):return (1-d['gamma'])[:,None,None]*np.eye(128)+(d['gamma']*d['beta'])[:,None,None]*np.einsum('hi,hj->hij',d['k'],d['k'])
def drive(point,i):return {n:point[f'd{i}:{n}'].astype(np.float64) for n in ['k','v','beta','gamma']}
def dense_energy(meta,old,new,d):
    matrix=K(d);f=d['beta'][:,None,None]*np.einsum('hi,hj->hij',d['k'],d['v']);step=new-old
    internal=matrix@old;res=step-f+internal;ks=matrix@step
    ns=lambda a:np.sqrt(np.sum(a*a,axis=(-2,-1)))
    force_error=ns(res)/np.maximum.reduce([np.ones(16),ns(f),ns(internal)])
    W=np.sum(f*step,axis=(-2,-1));U=np.sum(internal*step,axis=(-2,-1))+.5*np.sum(step*ks,axis=(-2,-1));D=np.sum(step*step,axis=(-2,-1))-.5*np.sum(step*ks,axis=(-2,-1))
    bound=ns(res)*ns(step)+100*np.finfo(np.float64).eps*np.maximum.reduce([np.ones(16),abs(W),abs(U),abs(D)])
    for h in range(16):
        original=energy_lookup[meta['question'],meta.get('index',-1),meta['token'],meta['layer'],h]
        assert original['pass'] and force_error[h]<=1e-5 and abs(W[h]-U[h]-D[h])<=bound[h]+1e-10 and D[h]>=-1e-10
        for key,value in [('work',W[h]),('stored_change',U[h]),('dissipation',D[h])]:assert abs(original[key]-value)<=1e-9*max(1.,abs(value)),key
        audit['dense_energy_head_checks']+=1
normal={};normal_pred={};packets={}
for q in inputs['questions']:
    pid=q['id'];packet=load(R/'normal_inputs'/f'{pid}.npz');packets[pid]=packet
    assert set(packet)=={'boundary_t1','boundary_t2','initial12:S','initial12:window','initial13:S','initial13:window'}
    for t in [1,2]:
        n=load(R/'native_baselines'/f'{pid}_t{t}.npz');p=load(R/'predictions'/f'{pid}_normal_t{t}.npz');normal[pid,t]=n;normal_pred[pid,t]=p
        assert np.array_equal(n['b12_input'],packet[f'boundary_t{t}'])
        for field in FIELDS:
            assert norm(p[field]-n[field])<=1e-5*max(1.,norm(n[field]));audit['baseline_fields_rechecked']+=1
        for i in [12,13]:
            old=packet[f'initial{i}:S'].astype(np.float64) if t==1 else normal[pid,1][f'b{i}_S'].astype(np.float64)
            dense_energy({'question':pid,'token':t,'layer':i},old,n[f'b{i}_S'].astype(np.float64),drive(n,i))
for c in cases:
    pid=c['question'];previous_native=None;previous_pred=None
    for t in [1,2]:
        n=load(Path(target_map[f'{c["index"]}:{t}']));p=load(R/'predictions'/f'case_{c["index"]:03d}_t{t}.npz');n0=normal[pid,t];p0=normal_pred[pid,t]
        assert np.array_equal(n['b12_output'],n['b13_input']) and np.array_equal(p['b12_output'],p['b13_input'])
        if t==2:assert np.array_equal(n['b12_input'],packets[pid]['boundary_t2'])
        for field in FIELDS:
            r=lookup[c['index'],t,field];v=norm(p[field]-n[field]);truth=n[field].astype(np.float64)-n0[field].astype(np.float64);pred=p[field]-p0[field]
            delta=norm(truth);floor=20*np.finfo(np.float32).eps*max(1.,norm(n[field]),norm(n0[field]));error=norm(pred-truth);bound=max(.01*delta,floor)
            for key,value in [('value_error',v),('native_change_norm',delta),('change_error',error),('numeric_floor',floor)]:
                diff=abs(r[key]-value)/max(1.,abs(value));audit['maximum_gate_scalar_discrepancy']=max(audit['maximum_gate_scalar_discrepancy'],diff);assert diff<1e-11
            assert (v<=1e-5*max(1.,norm(n[field])))==r['value_pass'] and (error<=bound)==r['change_pass'] and (delta>5*floor)==r['change_resolved']
            if c['split']=='held' and (not r['value_pass'] or not r['change_pass']):fields_failed.append(r)
            if c['split']=='held' and field in ['b13_S','b13_output']:audit['all_required_signals_resolved'] &= r['change_resolved']
            audit['full_value_and_change_fields_rechecked']+=1
        for mode in ['frozenK','frozenKF','reset']:
            for field in ['b13_S','b13_output']:
                true=n[field].astype(np.float64)-n0[field].astype(np.float64);forecast=p[mode+':'+field]-p0[field]
                floor=20*np.finfo(np.float32).eps*max(1.,norm(n[field]),norm(n0[field]));err=norm(forecast-true)/max(norm(true),floor)
                control_rows.append({**c,'token':t,'mode':mode,'field':field,'relative_change_error':err});audit['control_vectors_rechecked']+=1
        for i in [12,13]:
            native_old=packets[pid][f'initial{i}:S'].astype(np.float64) if t==1 else previous_native[f'b{i}_S'].astype(np.float64)
            dense_energy({**c,'token':t,'layer':i},native_old,n[f'b{i}_S'].astype(np.float64),drive(n,i))
            old0=packets[pid][f'initial{i}:S'] if t==1 else normal_pred[pid,1][f'b{i}_S'];olda=packets[pid][f'initial{i}:S'] if t==1 else previous_pred[f'b{i}_S']
            d0=drive(p0,i);da=drive(p,i);terms=finite_terms(old0,olda,d0,da);tr=term_lookup[c['index'],t,i]
            for name,value in terms.items():assert digest(value)==tr['term_sha256'][name];audit['term_vectors_regenerated_and_hash_verified']+=1
            ds=olda.astype(np.float64)-old0.astype(np.float64);K0=K(d0);Ka=K(da)
            df=da['beta'][:,None,None]*np.einsum('hi,hj->hij',da['k'],da['v'])-d0['beta'][:,None,None]*np.einsum('hi,hj->hij',d0['k'],d0['v'])
            explicit=ds-K0@ds+df-(Ka-K0)@old0-(Ka-K0)@ds
            target=p[f'b{i}_S']-p0[f'b{i}_S'];assert norm(explicit-target)<1e-10*max(1.,norm(target))
        previous_native=n;previous_pred=p
assert audit['baseline_fields_rechecked']==168 and audit['full_value_and_change_fields_rechecked']==672 and audit['dense_energy_head_checks']==1920 and audit['term_vectors_regenerated_and_hash_verified']==384
audit['all_held_value_change_gates_pass']=not fields_failed;audit['failed_held_fields']=fields_failed
control_original=read(S/'CONTROL_COMPARISONS.json');assert len(control_rows)==len(control_original)==288
for a,b in zip(control_rows,control_original):assert a['index']==b['index'] and a['token']==b['token'] and a['mode']==b['mode'] and a['field']==b['field'] and abs(a['relative_change_error']-b['relative_change_error'])<1e-11
mechanism=[]
for t in [1,2]:
    closed=float(np.median([r['change_error_over_scale'] for r in rows if r['split']=='held' and r['token']==t and r['field']=='b13_S']))
    shortcut=float(np.median([r['relative_change_error'] for r in control_rows if r['split']=='held' and r['token']==t and r['field']=='b13_S' and r['mode']=='frozenKF']))
    mechanism.append({'token':t,'closed_median':closed,'frozen_drivers_median':shortcut,'pass':closed<=.1*shortcut})
audit['mechanism_control_recheck']=mechanism
recomputed=not fields_failed and audit['all_required_signals_resolved'] and all(r['pass'] for r in mechanism)
assert bool(recomputed)==native_summary['primary_support'];audit['primary_support_reconfirmed']=bool(recomputed)
# Secondary geometry uses the predeclared finite two-axis plane, without extra native calls.
def angle(a,b):
    den=norm(a)*norm(b)
    return None if den==0 else float(np.degrees(np.arccos(np.clip(np.dot(a.ravel(),b.ravel())/den,-1.,1.))))
def plane_record(vectors,basis):
    M=np.stack([v.ravel() for v in vectors],axis=1);B=np.stack([v.ravel() for v in basis],axis=1)
    G=M.T@M;L=np.linalg.cholesky(B.T@B)
    transformed=np.linalg.solve(L,np.linalg.solve(L,G).T).T
    eigen=np.maximum(0.,np.linalg.eigvalsh((transformed+transformed.T)/2))
    return {'output_axis_angle_degrees':angle(vectors[0],vectors[1]),'axis_gains':[norm(vectors[j])/norm(basis[j]) for j in [0,1]],'principal_gains':np.sqrt(eigen).tolist(),'gram_eigenvalue_ratio':float(eigen[-1]/eigen[0]) if eigen[0]>0 else None}
for q in inputs['questions']:
    pid=q['id'];cs=[c for c in cases if c['question']==pid]
    actual={(c['direction'],c['alpha'],t):load(Path(target_map[f'{c["index"]}:{t}'])) for c in cs for t in [1,2]}
    predicted={(c['direction'],c['alpha'],t):load(R/'predictions'/f'case_{c["index"]:03d}_t{t}.npz') for c in cs for t in [1,2]}
    basis=[(actual[j,.125,1]['b12_input'].astype(np.float64)-actual[j,-.125,1]['b12_input'].astype(np.float64))/.25 for j in [0,1]]
    for t in [1,2]:
        for field in ['b12_output','b13_output','b12_S','b13_S']:
            responses={name:[(data[j,.125,t][field].astype(np.float64)-data[j,-.125,t][field].astype(np.float64))/.25 for j in [0,1]] for name,data in [('native',actual),('predicted',predicted)]}
            g={name:plane_record(v,basis) for name,v in responses.items()}
            asymmetric={}
            for name,data,normal_data in [('native',actual,normal),('predicted',predicted,normal_pred)]:
                base=normal_data[pid,t][field].astype(np.float64)
                asymmetric[name]=[norm(data[j,.125,t][field].astype(np.float64)+data[j,-.125,t][field].astype(np.float64)-2*base)/max(1e-30,norm(data[j,.125,t][field]-base)+norm(data[j,-.125,t][field]-base)) for j in [0,1]]
            geometry.append({'question':pid,'split':q['split'],'token':t,'field':field,**g,'signed_even_asymmetry_fraction':asymmetric,'response_vector_relative_errors':[norm(responses['predicted'][j]-responses['native'][j])/max(1e-30,norm(responses['native'][j])) for j in [0,1]],'turning_from_delivered_input_degrees':{name:[angle(v,basis[j]) for j,v in enumerate(vectors)] for name,vectors in responses.items()} if field.endswith('output') else None,'scope':'Finite signed secant at0.125 in fixed two-dimensional input plane; no ambient-spectrum or semantic claim.'})
    del actual,predicted
dump(OUT/'FINITE_TWO_AXIS_GEOMETRY.json',geometry)
audit['secondary_geometry_records']=len(geometry)
assert len(geometry)==48
audit['original_serialization_fault_preserved']=True
audit['combined_native_calls']=76
audit['six_exact_prefix_reconstructions_verified']=len(read(S/'PREFIX_RECONSTRUCTION.json'))==6 and all(r['full_prefix_cache_bitwise_hash_match'] for r in read(S/'PREFIX_RECONSTRUCTION.json'))
assert audit['six_exact_prefix_reconstructions_verified']
audit['seconds']=time.monotonic()-BEGIN;dump(OUT/'INDEPENDENT_AUDIT.json',audit)
print('R4_COUPLED_INDEPENDENT_AUDIT',json.dumps({k:v for k,v in audit.items() if k!='failed_held_fields'},default=scalar),flush=True)
