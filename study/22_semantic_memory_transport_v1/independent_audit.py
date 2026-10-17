"""Independent full-vector, dense-energy and held semantic checks; no model loading."""
import os,time
os.environ['OPENBLAS_NUM_THREADS']='2'
BEGIN=time.monotonic()
import sys,struct,hashlib
from pathlib import Path
import numpy as np
from common import read,dump,load,sha,digest,norm,FIELDS,word_scores
from equations import finite_terms
R=Path(sys.argv[1]);OUT=R.parent/'audit_20261015';OUT.mkdir(exist_ok=False)
plan=read(R/'source/PLAN.json');summary=read(R/'RESULTS.json');assert summary['native_backbone_calls']==222
for name in ['EXECUTION_FREEZE.json','NORMAL_INPUT_FREEZE.json','PREDICTION_FREEZE.json','DIRECTION_FREEZE.json']:
    for n,h in read(R/name)['sha256'].items():assert sha(R/n)==h,(name,n)
for name in ['NATIVE_BASELINE_FREEZE.json','EXPORT_HASHES.json']:
    for n,h in read(R/name).items():assert sha(R/n)==h,(name,n)
pf=read(R/'PREDICTION_FREEZE.json');assert pf['all_targets_seen']==0 and pf['all_predictions_precede_any_native_steered_target']
assert max(p.stat().st_mtime_ns for p in (R/'predictions').iterdir())<=min(p.stat().st_mtime_ns for p in (R/'native_targets').iterdir())
assert read(R/'predictions/PREDICTOR_RECEIPT.json')['numpy_only']
inputs=read(R/'INPUTS.json')['questions'];cases=read(R/'CASES.json');all_normal=read(R/'ALL_NORMAL_METADATA.json')
native_metadata={q['id']:q for q in all_normal};question_map={q['id']:q for q in inputs}
rows=read(R/'FIELD_COMPARISONS.json');lookup={(r['index'],r['token'],r['field']):r for r in rows}
assert len(lookup)==2016 and len(inputs)==18 and len(cases)==72
heads={(r['question'],r.get('index',-1),r['token'],r['layer'],r['head']):r for r in read(R/'NATIVE_ENERGY_HEADS.json')}
terms_saved={(r['index'],r['token'],r['layer']):r for r in read(R/'predictions/FINITE_RESPONSE_TERMS.json')}
with np.load(R/'DIRECTIONS.npz',allow_pickle=False) as z:directions=z['directions'].copy();alpha=float(z['alpha']);axes={n:z[n].copy() for n in ['b13_output','b13_S']}
differences={n:[] for n in ['b12_input','b13_output','b13_S']}
for j in range(6):
    qs=sorted([q for q in all_normal if q['pair_index']==j and q['split']!='held'],key=lambda q:q['number']);assert len(qs)==2
    x0=load(R/'native_baselines'/f'{qs[0]["id"]}_t1.npz');x1=load(R/'native_baselines'/f'{qs[1]["id"]}_t1.npz')
    for n in differences:differences[n].append(x1[n].astype(np.float64)-x0[n].astype(np.float64))
w=np.mean(differences['b12_input'],axis=0);w/=norm(w);rng=np.random.default_rng(2026101522);random=rng.normal(size=1024);random-=w*float(w@random);random/=norm(random)
assert digest(np.stack([w,random]).astype(np.float32))==digest(directions)
assert alpha==float(np.clip(.25*np.median([norm(a) for a in differences['b12_input']]),.125,2.))
for n in axes:
    ax=np.mean(differences[n],axis=0);ax/=norm(ax);assert digest(ax)==digest(axes[n])
assert read(R/'DIRECTION_FREEZE.json')['held_contexts_seen']==0
direction_time=(R/'DIRECTION_FREEZE.json').stat().st_mtime_ns
assert direction_time<=min((R/'native_baselines'/f'{q["id"]}_t1.npz').stat().st_mtime_ns for q in inputs if q['split']=='held')
geometry=load(R/'OUTPUT_GEOMETRY.npz');word_weights=geometry['selected_word_weights'].astype(np.float64)
C=geometry['covariance'];M=geometry['inverse_covariance'];assert np.linalg.norm(C@M-np.eye(1024))<1e-8
# Rebuild population covariance independently from the public BF16 bytes, using
# centered batch merging rather than the worker's second-moment calculation.
checkpoint=Path(sys.argv[2]);tensor_file=checkpoint/'model.safetensors-00001-of-00001.safetensors'
assert sha(tensor_file)=='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
with tensor_file.open('rb') as f:
    header_bytes=struct.unpack('<Q',f.read(8))[0];header=__import__('json').loads(f.read(header_bytes))
descriptor=header['model.language_model.embed_tokens.weight'];assert descriptor['dtype']=='BF16' and descriptor['shape']==[248320,1024]
packed=np.memmap(tensor_file,mode='r',dtype='<u2',offset=8+header_bytes+descriptor['data_offsets'][0],shape=(248320,1024))
decode=lambda a:(np.asarray(a,dtype=np.uint32)<<16).view(np.float32)
for pair in plan['word_pairs']:assert np.array_equal(decode(packed[pair['ids']]).astype(np.float64),word_weights[pair['pair_index']])
count=0;mean=np.zeros(1024);scatter=np.zeros((1024,1024));fp=hashlib.sha256()
for first in range(0,248320,8192):
    u=decode(packed[first:first+8192]);fp.update(np.ascontiguousarray(u).tobytes());x=u.astype(np.float64);n=len(x);mu=x.mean(0);center=x-mu;delta=mu-mean;new_count=count+n
    scatter+=center.T@center+np.outer(delta,delta)*(count*n/new_count);mean+=delta*(n/new_count);count=new_count
C_independent=scatter/count;covariance_error=norm(C_independent-C);assert covariance_error<=1e-10*max(1.,norm(C))
assert fp.hexdigest()==read(R/'OUTPUT_GEOMETRY_RESULTS.json')['output_weight_FP32_sha256']
del packed,x,u,center,scatter
gamma=(word_weights[:12,1]-word_weights[:12,0]).mean(0);assert np.allclose(gamma,geometry['mean_number_difference'],atol=1e-14,rtol=0)
metric_direction=M@gamma/np.sqrt(gamma@M@gamma);random_proj=geometry['random_pair_differences']@metric_direction
out_results=read(R/'OUTPUT_GEOMETRY_RESULTS.json');out_pair=[]
for j in range(12,16):
    score=float((word_weights[j,1]-word_weights[j,0])@metric_direction);percent=float(np.mean(random_proj<=score));out_pair.append({'pair_index':j,'score':score,'percentile':percent})
    old=next(r for r in out_results['held_pair_projections'] if r['pair_index']==j);assert abs(old['covariance_metric_projection']-score)<1e-10 and old['metric_random_percentile']==percent
output_subspace=all(r['score']>0 for r in out_pair) and np.median([r['percentile'] for r in out_pair])>=.95
v=M@gamma;v/=norm(v);slopes=[]
for j in range(12,16):
    a=word_weights[j]@v;slopes.append({'pair_index':j,'number_lower':float(a[1]-a[0]),'number_upper':float(a[3]-a[2]),'case_singular':float(a[2]-a[0]),'case_plural':float(a[3]-a[1])})
num=float(np.median([(r['number_lower']+r['number_upper'])/2 for r in slopes]));cap=float(np.median([max(abs(r['case_singular']),abs(r['case_plural'])) for r in slopes]))
output_selectivity=all(r['number_lower']>0 and r['number_upper']>0 for r in slopes) and cap<=.5*num
assert bool(output_subspace)==out_results['held_pair_subspace_gate_pass']
assert bool(output_selectivity)==next(r['pass'] for r in out_results['output_selectivity'] if r['method']=='covariance_mapped')
audit={'new_model_forwards':0,'GPU_allocations':0,'source_input_prediction_and_native_hashes_pass':True,'direction_recomputed_from_training_only':True,'directions_frozen_before_held_normals':True,'predictions_before_targets':True,'public_BF16_output_weights_and_FP32_fingerprint_verified':True,'population_covariance_independently_rebuilt_from_public_bytes':True,'covariance_reconstruction_Frobenius_error':covariance_error,'complete_baseline_fields_checked':0,'complete_target_fields_checked':0,'term_vector_hashes_regenerated':0,'dense_energy_heads_checked':0,'full_word_score_sets_checked':0,'word_readouts_checked_against_original_selected_weights':0,'all_numeric_flags_reconfirmed':True}
def driver(p,i):return {n:p[f'd{i}:{n}'].astype(np.float64) for n in ['k','v','beta','gamma']}
def K(d):return (1-d['gamma'])[:,None,None]*np.eye(128)+(d['gamma']*d['beta'])[:,None,None]*np.einsum('hi,hj->hij',d['k'],d['k'])
def dense_energy(meta,old,new,d):
    matrix=K(d);f=d['beta'][:,None,None]*np.einsum('hi,hj->hij',d['k'],d['v']);step=new-old;internal=matrix@old;residual=step-f+internal;ks=matrix@step
    n=lambda a:np.sqrt(np.sum(a*a,axis=(-2,-1)))
    force=n(residual)/np.maximum.reduce([np.ones(16),n(f),n(internal)])
    W=np.sum(f*step,axis=(-2,-1));U=np.sum(internal*step,axis=(-2,-1))+.5*np.sum(step*ks,axis=(-2,-1));D=np.sum(step*step,axis=(-2,-1))-.5*np.sum(step*ks,axis=(-2,-1))
    bound=n(residual)*n(step)+100*np.finfo(np.float64).eps*np.maximum.reduce([np.ones(16),abs(W),abs(U),abs(D)])
    for h in range(16):
        r=heads[meta['question'],meta.get('index',-1),meta['token'],meta['layer'],h]
        assert r['pass'] and force[h]<=1e-5 and abs(W[h]-U[h]-D[h])<=bound[h]+1e-10 and D[h]>=-1e-10
        for name,x in [('work',W[h]),('stored_change',U[h]),('dissipation',D[h])]:assert abs(r[name]-x)<=1e-9*max(1.,abs(x))
        audit['dense_energy_heads_checked']+=1
def keyword_check(point,q):
    selected=point['logits'][q['word_ids']].astype(np.float64);direct=word_weights[q['pair_index']]@point['final_embedding'].astype(np.float64)
    assert norm(selected-direct)<=1e-5*max(1.,norm(selected));audit['word_readouts_checked_against_original_selected_weights']+=1
normals={};forecasts={};packets={}
for q in inputs:
    pid=q['id'];packet=load(R/'normal_inputs'/f'{pid}.npz');packets[pid]=packet
    assert set(packet)=={'boundary_t1','boundary_t2','initial12:S','initial12:window','initial13:S','initial13:window'}
    for t in [1,2]:
        n=load(R/'native_baselines'/f'{pid}_t{t}.npz');p=load(R/'predictions'/f'{pid}_normal_t{t}.npz');normals[pid,t]=n;forecasts[pid,t]=p;keyword_check(n,q)
        assert np.array_equal(n['b12_input'],packet[f'boundary_t{t}'])
        for field in FIELDS:assert norm(p[field]-n[field])<=1e-5*max(1.,norm(n[field]));audit['complete_baseline_fields_checked']+=1
        for i in [12,13]:dense_energy({'question':pid,'token':t,'layer':i},(packet[f'initial{i}:S'] if t==1 else normals[pid,1][f'b{i}_S']).astype(np.float64),n[f'b{i}_S'].astype(np.float64),driver(n,i))
effects=[];numeric_failed=[];resolved=True;projected=[]
for c in cases:
    pid=c['question'];q=question_map[pid];previous=None;previous_pred=None
    for t in [1,2]:
        n=load(R/'native_targets'/f'case_{c["index"]:03d}_t{t}.npz');p=load(R/'predictions'/f'case_{c["index"]:03d}_t{t}.npz');n0=normals[pid,t];p0=forecasts[pid,t];keyword_check(n,q)
        assert np.array_equal(n['b12_output'],n['b13_input']) and np.array_equal(p['b12_output'],p['b13_input'])
        if t==2:assert np.array_equal(n['b12_input'],packets[pid]['boundary_t2'])
        for field in FIELDS:
            r=lookup[c['index'],t,field];value=norm(p[field]-n[field]);truth=n[field].astype(np.float64)-n0[field].astype(np.float64);pred=p[field]-p0[field]
            size=norm(truth);floor=float(20*np.finfo(np.float32).eps)*max(1.,norm(n[field]),norm(n0[field]));err=norm(pred-truth);bound=max(.01*size,floor)
            assert abs(r['value_error']-value)<1e-11 and abs(r['change_error']-err)<1e-11 and abs(r['numeric_floor']-floor)<1e-11
            assert bool(value<=1e-5*max(1.,norm(n[field])))==r['value_pass'] and bool(err<=bound)==r['change_pass'] and bool(size>5*floor)==r['change_resolved']
            if c['split']=='held' and (not r['value_pass'] or not r['change_pass']):numeric_failed.append(r)
            if c['split']=='held' and c['direction']==0 and field in ['b13_S','b13_output']:resolved=resolved and r['change_resolved']
            audit['complete_target_fields_checked']+=1
        for i in [12,13]:
            old=packets[pid][f'initial{i}:S'] if t==1 else previous[f'b{i}_S'];dense_energy({**c,'token':t,'layer':i},old.astype(np.float64),n[f'b{i}_S'].astype(np.float64),driver(n,i))
            old0=packets[pid][f'initial{i}:S'] if t==1 else forecasts[pid,1][f'b{i}_S'];olda=packets[pid][f'initial{i}:S'] if t==1 else previous_pred[f'b{i}_S']
            d0=driver(p0,i);da=driver(p,i);terms=finite_terms(old0,olda,d0,da);saved=terms_saved[c['index'],t,i]
            for name,a in terms.items():assert digest(a)==saved['term_sha256'][name];audit['term_vector_hashes_regenerated']+=1
            ds=olda-old0;K0=K(d0);Ka=K(da);f0=d0['beta'][:,None,None]*np.einsum('hi,hj->hij',d0['k'],d0['v']);fa=da['beta'][:,None,None]*np.einsum('hi,hj->hij',da['k'],da['v'])
            exact=ds-K0@ds+(fa-f0)-(Ka-K0)@old0-(Ka-K0)@ds;assert norm(exact-(p[f'b{i}_S']-p0[f'b{i}_S']))<1e-10*max(1.,norm(exact))
        for name in axes:projected.append({**c,'token':t,'field':name,'native_projection':float(np.sum((n[name].astype(np.float64)-n0[name])*axes[name])),'predicted_projection':float(np.sum((p[name]-p0[name])*axes[name]))})
        if t==1:effects.append({**c,**word_scores(n['logits'],q['word_ids'])});audit['full_word_score_sets_checked']+=1
        previous=n;previous_pred=p
assert audit['complete_baseline_fields_checked']==504 and audit['complete_target_fields_checked']==2016 and audit['dense_energy_heads_checked']==5760 and audit['term_vector_hashes_regenerated']==1152 and audit['word_readouts_checked_against_original_selected_weights']==180
original_effects=read(R/'NATIVE_WORD_EFFECTS.json')
for a,b in zip(effects,original_effects):
    for n in ['number_log_odds','number_lower','number_upper','case_singular','case_plural','four_word_probability_mass']:assert abs(a[n]-b[n])<1e-11
original_projected=read(R/'SEMANTIC_AXIS_PROJECTIONS.json');assert len(projected)==len(original_projected)==288
for a,b in zip(projected,original_projected):assert abs(a['native_projection']-b['native_projection'])<1e-11 and abs(a['predicted_projection']-b['predicted_projection'])<1e-11
held_contexts=[q for q in inputs if q['split']=='held'];normal_correct=sum(q['normal_word_scores']['number_log_odds']>0 if q['number'] else q['normal_word_scores']['number_log_odds']<0 for q in held_contexts)
normal_mass=float(np.median([q['normal_word_scores']['four_word_probability_mass'] for q in held_contexts]));pairs=[]
for j in range(12,16):
    for template in ['a','b']:
        qs=sorted([q for q in held_contexts if q['pair_index']==j and q['template']==template],key=lambda q:q['number']);pairs.append(qs[1]['normal_word_scores']['number_log_odds']-qs[0]['normal_word_scores']['number_log_odds'])
natural=normal_correct>=13 and normal_mass>=.05 and all(v>0 for v in pairs)
lookup_effect={(r['question'],r['direction'],r['alpha']>0):r for r in effects};semantic_cases=[]
for q in held_contexts:
    pid=q['id'];p=lookup_effect[pid,0,True];m=lookup_effect[pid,0,False];rp=lookup_effect[pid,1,True];rm=lookup_effect[pid,1,False]
    swing=.5*(p['number_log_odds']-m['number_log_odds']);case=max(.5*abs(p['case_singular']-m['case_singular']),.5*abs(p['case_plural']-m['case_plural']))
    semantic_cases.append({'question':pid,'pair_index':q['pair_index'],'number':q['number'],'template':q['template'],'normal_log_odds':q['normal_word_scores']['number_log_odds'],'positive_log_odds':p['number_log_odds'],'negative_log_odds':m['number_log_odds'],'monotonic_plural_shift':p['number_log_odds']>q['normal_word_scores']['number_log_odds']>m['number_log_odds'],'number_half_swing':swing,'random_absolute_half_swing':abs(.5*(rp['number_log_odds']-rm['number_log_odds'])),'capitalization_half_swing':case})
monotonic=sum(r['monotonic_plural_shift'] for r in semantic_cases);swing=float(np.median([r['number_half_swing'] for r in semantic_cases]));random=float(np.median([r['random_absolute_half_swing'] for r in semantic_cases]));case=float(np.median([r['capitalization_half_swing'] for r in semantic_cases]))
causal=monotonic>=13 and swing>=.01 and swing>=2*random;selective=case<=.5*swing
semantic={'output_subspace_gate_pass':bool(output_subspace),'output_intervention_selectivity_pass':bool(output_selectivity),'natural_context_gate_pass':bool(natural),'natural_count_choices_correct':normal_correct,'normal_contexts':16,'median_four_word_mass':normal_mass,'all8_natural_pair_differences_positive':all(v>0 for v in pairs),'internal_causal_direction_gate_pass':bool(causal),'internal_selectivity_gate_pass':bool(selective),'held_monotonic_plural_shifts':monotonic,'median_number_half_swing_nats':swing,'median_random_absolute_half_swing_nats':random,'median_capitalization_half_swing_nats':case,'capitalization_to_number_ratio':case/max(swing,1e-30),'held_context_details':semantic_cases,'semantic_support':bool(output_subspace and output_selectivity and natural and causal and selective),'interpretation':'All five gates required for a selective semantic-transport claim; failures preserved. Native word effects measured on the entire original backbone. The two-block equations predict local full states, not whole-backbone final-word probabilities.'}
audit.update(numerical_support_reconfirmed=bool(not numeric_failed and resolved),all_required_signals_resolved=bool(resolved),numeric_failed_fields=numeric_failed,output_geometry_independently_recomputed=True,all_semantic_flags_computed_from_held_archived_targets=True,seconds=time.monotonic()-BEGIN)
assert audit['numerical_support_reconfirmed']==summary['numerical_support']
dump(OUT/'INDEPENDENT_AUDIT.json',audit);dump(OUT/'SEMANTIC_RESULTS.json',semantic)
print('R4_SEMANTIC_AUDIT',__import__('json').dumps({k:v for k,v in audit.items() if k!='numeric_failed_fields'}),flush=True)
print('R4_SEMANTIC_MEANING',__import__('json').dumps({k:v for k,v in semantic.items() if k!='held_context_details'}),flush=True)
