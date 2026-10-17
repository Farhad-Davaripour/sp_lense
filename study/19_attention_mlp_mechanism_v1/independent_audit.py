"""Recheck every saved full vector against native parents and causal sums."""
import os,time
os.environ['OPENBLAS_NUM_THREADS']='2'
BEGIN=time.monotonic()
import json,hashlib,sys,zipfile
from pathlib import Path
import numpy as np
R=Path('/content/research4/artifacts/attention_mlp_mechanism_v1/colab_cpu_20261015')
P=Path('/content/research4/artifacts/fresh_memory_boundary_response_v1/colab_cpu_20261014')
F=P/'fresh';E=Path('/content/research4/artifacts/recurrent_energy_replication_v1/colab_cpu_20261015')
sys.path.insert(0,str(R/'source'))
from output_store import OutputStore
store=OutputStore(F)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
for n,h in read(R/'PARENT_USED_HASHES.json').items():assert sha(F/n)==h,n
for n,h in read(R/'EXPORT_HASHES.json').items():assert sha(R/n)==h,n
cases=read(E/'DRIVER_CASES.json');meta=read(F/'METRICS.json');questions=read(F/'INPUTS.json')['questions']
field_rows=read(R/'FIELD_RESULTS.json');deriv_rows=read(R/'DERIVATIVE_RESULTS.json')
fl={(r['case'],r['layer'],r['module']):r for r in field_rows};dl={(r['case'],r['layer'],r['module']):r for r in deriv_rows}
with np.load(P/'DIRECTIONS_FRESH.npz',allow_pickle=False) as z:directions=z['vectors'].astype(np.float64)
with np.load(R/'FULL_REPLAY_ARRAYS.npz',allow_pickle=False) as z:arrays={n:z[n].copy() for n in z.files}
panel=[(q['id'],'primary',t) for q in questions for t in [1,2]]+[(pid,'reference',t) for pid in ['r17_cal_spoon','r17_held_panels'] for t in [1,2]]
names={f'b{i}_{n}' for i in range(24) for n in ['input','mixer','middle','mlp','output']}
audit={'new_backbone_forwards':0,'native_full_vector_forecasts_rechecked':0,'native_full_vector_derivatives_rechecked':0,'native_residual_field_sums_rechecked':0,'supplemental_interblock_field_sums_rechecked':0,'maximum_scalar_recheck_discrepancy':0.,'energy_head_cases':0,'finite_difference_probability_heads':0,'probability_FD_failures':[],'all_frozen_field_gates_reconfirmed':True}
def check(pred,actual,row,derivative):
    error=norm(pred-actual);den=norm(actual);bound=max(.001*den,1e-6) if derivative else 1e-5*max(1.,den)
    diff=abs(error-row['error_norm'])/max(1.,error);audit['maximum_scalar_recheck_discrepancy']=max(audit['maximum_scalar_recheck_discrepancy'],diff)
    # The worker added native FP32 residual fields; this audit adds them in
    # FP64. Reconfirm the fixed error gate rather than requiring identical
    # roundoff. Replayed module vectors share exactly the stored precision.
    if not row['module'].startswith('residual'):assert diff<1e-11
    assert error<=bound and row['pass']
    assert abs(bound-row['bound'])<=1e-12*max(1.,bound)
for ci,c in enumerate(cases):
    pid,t=c['question'],c['token'];target=None if c['label']=='normal' else meta[int(c['case'].split('_')[-1])]
    name=f'contexts/{pid}_t{t}_baseline.npz' if target is None else target['actual_file']
    point={n:a for n,a in [(k[2:],v.astype(np.float64)) for k,v in store._read(name,{'v:'+n for n in names}).items()]}
    for i in range(24):
        check(arrays[f'v{ci:03d}_l{i}_mlp'],point[f'b{i}_mlp'],fl[c['case'],i,'MLP'],False);audit['native_full_vector_forecasts_rechecked']+=1
        extra=np.zeros(1024)
        if target and t==1 and target['site']==['middle',i]:extra=target['alpha']*np.array(target['direction'])
        check(point[f'b{i}_input']+point[f'b{i}_mixer']+extra,point[f'b{i}_middle'],fl[c['case'],i,'residual_middle'],False)
        check(point[f'b{i}_middle']+point[f'b{i}_mlp'],point[f'b{i}_output'],fl[c['case'],i,'residual_output'],False);audit['native_residual_field_sums_rechecked']+=2
        if i>0:
            inject=np.zeros(1024)
            if target and t==1 and target['site']==['input',i]:inject=target['alpha']*np.array(target['direction'])
            assert norm(point[f'b{i}_input']-point[f'b{i-1}_output']-inject)<=1e-5*max(1.,norm(point[f'b{i}_input']))
            audit['supplemental_interblock_field_sums_rechecked']+=1
        if i in [3,7,11,15,19,23]:
            check(arrays[f'v{ci:03d}_l{i}_attention'],point[f'b{i}_mixer'],fl[c['case'],i,'attention'],False);audit['native_full_vector_forecasts_rechecked']+=1
for di,(pid,label,t) in enumerate(panel):
    name=f'coefficients/{pid}_{label}_t{t}.npz';q={k[2:]:v.astype(np.float64) for k,v in store._read(name,{'q:'+n for n in names}).items()};case=f'derivative_{di:02d}'
    for i in range(24):
        check(arrays[f'q{di:03d}_l{i}_mlp'],q[f'b{i}_mlp'],dl[case,i,'MLP'],True);audit['native_full_vector_derivatives_rechecked']+=1
        check(q[f'b{i}_input']+q[f'b{i}_mixer'],q[f'b{i}_middle'],dl[case,i,'residual_middle'],True)
        check(q[f'b{i}_middle']+q[f'b{i}_mlp'],q[f'b{i}_output'],dl[case,i,'residual_output'],True);audit['native_residual_field_sums_rechecked']+=2
        if i>0:
            inject=directions[0 if label=='primary' else 1] if i==11 and t==1 else 0.
            assert norm(q[f'b{i}_input']-q[f'b{i-1}_output']-inject)<=max(.001*norm(q[f'b{i}_input']),1e-6)
            audit['supplemental_interblock_field_sums_rechecked']+=1
        if i in [3,7,11,15,19,23]:
            check(arrays[f'q{di:03d}_l{i}_attention'],q[f'b{i}_mixer'],dl[case,i,'attention'],True);audit['native_full_vector_derivatives_rechecked']+=1
for key in sorted(k for k in arrays if k.endswith('_scores')):
    stem=key[:-7];s=arrays[key];p=arrays[stem+'_P'];lp=arrays[stem+'_logP']
    lz=np.max(s,axis=-1,keepdims=True)+np.log(np.exp(s-np.max(s,axis=-1,keepdims=True)).sum(axis=-1,keepdims=True))
    assert np.allclose(lp,s-lz,atol=1e-11,rtol=0) and np.allclose(p,np.exp(lp),atol=1e-12,rtol=1e-11)
    assert np.max(abs(p.sum(axis=-1)-1))<1e-11
    energy=np.sum(p*(lp-s),axis=-1);assert np.max(abs(energy+lz[:,0]))<1e-9*max(1.,np.max(abs(lz)))
    audit['energy_head_cases']+=8
    if stem+'_dP' in arrays:
        ds=arrays[stem+'_ds'];dp=arrays[stem+'_dP'];eps=1e-4
        def softmax(a):
            e=np.exp(a-np.max(a,axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)
        fd=(softmax(s+eps*ds)-softmax(s-eps*ds))/(2*eps)
        for h in range(8):
            error=norm(fd[h]-dp[h]);bound=max(.001*norm(dp[h]),1e-7)
            if error>bound:audit['probability_FD_failures'].append({'array':stem,'head':h,'error':error,'bound':bound})
            audit['finite_difference_probability_heads']+=1
assert audit['native_full_vector_forecasts_rechecked']==2760 and audit['native_full_vector_derivatives_rechecked']==480 and audit['energy_head_cases']==5184
audit['all_probability_FD_checks_pass']=not audit['probability_FD_failures'];audit['seconds']=time.monotonic()-BEGIN
(R.parent/'INDEPENDENT_AUDIT_20261015.json').write_text(json.dumps(audit,indent=2)+'\n')
print('R4_COMPONENT_INDEPENDENT_AUDIT',json.dumps(audit),flush=True)
