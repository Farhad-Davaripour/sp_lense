"""Audit original arrays after a terminal history-bookkeeping assertion failed."""
import os,time
os.environ['OPENBLAS_NUM_THREADS']='2'
BEGIN=time.monotonic()
import json,hashlib,sys
from pathlib import Path
import numpy as np
R=Path('/content/research4/artifacts/linear_and_output_readout_v1/colab_cpu_20261015')
OUT=R.parent/'recovery_20261015';OUT.mkdir(exist_ok=False)
F=Path('/content/research4/artifacts/fresh_memory_boundary_response_v1/colab_cpu_20261014/fresh')
P18=Path('/content/research4/artifacts/recurrent_energy_replication_v1/colab_cpu_20261015')
sys.path.insert(0,str(R/'source'))
from output_store import OutputStore
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
    return h.hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def norm(a):return float(np.linalg.norm(np.asarray(a,dtype=np.float64)))
for n,h in read(R/'EXPORT_HASHES.json').items():assert sha(R/n)==h,n
for n,h in read(R/'PARENT_USED_HASHES.json').items():assert sha(F/n)==h,n
status=read(R/'STATUS.json');failure=read(R/'FAILURE.json');assert status['primal_cases']==92 and status['derivative_cases']==16
assert 'not previous and not qprevious' in failure['traceback'] and status['status']=='failed'
cases=read(P18/'DRIVER_CASES.json');meta=read(F/'METRICS.json');questions=read(F/'INPUTS.json')['questions']
first={(c['question'],c['label'],c['alpha']) for c in cases if c['token']==1};second={(c['question'],c['label'],c['alpha']) for c in cases if c['token']==2};singletons=first-second
assert len(singletons)==24 and all(k[1] in ['early','middle','late'] for k in singletons)
recorded_rows=read(R/'FIELD_RESULTS.json');recorded_qrows=read(R/'DERIVATIVE_RESULTS.json');assert len(recorded_rows)==3496 and len(recorded_qrows)==608
lookup={(r['case'],r['layer'],r['module']):r for r in recorded_rows};qlookup={(r['case'],r['layer'],r['module']):r for r in recorded_qrows}
assert len(lookup)==3496 and len(qlookup)==608
layers=[i for i in range(24) if (i+1)%4!=0]
names={'hidden','logits'}|{f'b{i}_mixer' for i in layers}|{f'm{i}_conv_states_0' for i in layers}
store=OutputStore(F)
with np.load(R/'FULL_REPLAY_ARRAYS.npz',allow_pickle=False) as z:arrays={n:z[n].copy() for n in z.files}
audit={'classification':'Offline audit of original immutable vectors; terminal assertion corrected for24 intentionally token1-only targets, no scientific thresholds changed','original_run_exit':1,'original_failure_retained':True,'new_backbone_forwards':0,'GPU_allocations':0,'complete_primal_fields_rechecked':0,'complete_derivative_fields_rechecked':0,'maximum_scalar_recheck_discrepancy':0.,'failed_fields':[],'singleton_token1_keys':[list(k) for k in sorted(singletons)],'full_vocabulary_coordinates':248320}
digests=[];prob=[]
def check(pred,native,row,derivative):
    error=norm(pred-native);den=norm(native);bound=max(.001*den,1e-6) if derivative else 1e-5*max(1.,den)
    diff=abs(error-row['error_norm'])/max(1.,error);audit['maximum_scalar_recheck_discrepancy']=max(audit['maximum_scalar_recheck_discrepancy'],diff)
    assert diff<1e-11 and abs(bound-row['bound'])<1e-12*max(1.,bound)
    passed=error<=bound;assert bool(passed)==row['pass']
    if not passed:audit['failed_fields'].append(row)
def mappings(tag):
    a=[(f'{tag}_l{i}_mixer',f'b{i}_mixer',i,'linear_attention_readout') for i in layers]
    a +=[(f'{tag}_l{i}_window',f'm{i}_conv_states_0',i,'convolution_update') for i in layers]
    return a+[(tag+'_hidden','hidden',24,'hidden'),(tag+'_logits','logits',24,'logits')]
def logp(a):
    a=np.asarray(a,dtype=np.float64);v=a-a.max();return v-np.log(np.exp(v).sum())
for ci,c in enumerate(cases):
    name=f'contexts/{c["question"]}_t{c["token"]}_baseline.npz' if c['label']=='normal' else meta[int(c['case'].split('_')[-1])]['actual_file']
    a={k[2:]:v for k,v in store._read(name,{'v:'+n for n in names}).items()}
    for key,native,i,module in mappings(f'v{ci:03d}'):
        assert arrays[key].shape==a[native].shape
        check(arrays[key],a[native],lookup[c['case'],i,module],False);audit['complete_primal_fields_rechecked']+=1
    pred=arrays[f'v{ci:03d}_logits'];assert pred.shape==(248320,)
    lp=logp(a['logits']);lq=logp(pred);pa=np.exp(lp);pq=np.exp(lq)
    prob.append({**c,'TV':float(.5*np.sum(abs(pa-pq))),'KL_native_to_replay':float(np.sum(pa*(lp-lq)))})
    digests.append({'case':c['case'],'predicted_logit_dtype':str(pred.dtype),'predicted_full_logits_sha256':digest(pred),'native_full_logits_sha256':digest(a['logits']),'predicted_full_probability_sha256':digest(pq),'native_full_probability_sha256':digest(pa)})
panel=[(q['id'],'primary',t) for q in questions for t in [1,2]]+[(pid,'reference',t) for pid in ['r17_cal_spoon','r17_held_panels'] for t in [1,2]]
for di,(pid,label,t) in enumerate(panel):
    a={k[2:]:v for k,v in store._read(f'coefficients/{pid}_{label}_t{t}.npz',{'q:'+n for n in names}).items()};case=f'derivative_{di:02d}'
    for key,native,i,module in mappings(f'q{di:03d}'):
        assert arrays[key].shape==a[native].shape
        check(arrays[key],a[native],qlookup[case,i,module],True);audit['complete_derivative_fields_rechecked']+=1
    digests.append({'case':case,'native_full_logit_derivative_sha256':digest(a['logits']),'predicted_full_logit_derivative_sha256':digest(arrays[f'q{di:03d}_logits'])})
assert audit['complete_primal_fields_rechecked']==3496 and audit['complete_derivative_fields_rechecked']==608
original_prob=read(R/'FULL_PROBABILITY_COMPARISON.json');assert len(original_prob)==92
for a,b in zip(prob,original_prob):assert a['case']==b['case'] and abs(a['TV']-b['TV'])<1e-14 and abs(a['KL_native_to_replay']-b['KL_native_to_replay'])<1e-14
audit['all_original_scientific_field_gates_pass']=not audit['failed_fields'];audit['maximum_full_probability_TV']=max(r['TV'] for r in prob);audit['seconds']=time.monotonic()-BEGIN
dump(OUT/'INDEPENDENT_READOUT_AUDIT_20261015.json',audit);dump(OUT/'FULL_OUTPUT_DIGESTS.json',digests);dump(OUT/'FULL_PROBABILITY_RECHECK.json',prob)
print('R4_INDEPENDENT_READOUT_AUDIT',json.dumps(audit),flush=True)
