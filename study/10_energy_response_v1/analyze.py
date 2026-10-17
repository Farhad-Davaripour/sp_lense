"""Independent saved-array evaluation of frozen response/energy predictions."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np

BASE=Path('C:/Users/farha/Research4/artifacts/energy_response_v1')
PILOT=BASE/'pilot_20261012';MATCHED=BASE/'matched_snapshot_20261012';OUT=BASE/'evaluated_20261012'
OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def js(root,name):return json.loads((root/name).read_text())
def size(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def cosine(a,b):
    den=size(a)*size(b)
    return float(np.clip(np.sum(np.asarray(a,dtype=float)*np.asarray(b,dtype=float))/den,-1,1)) if den>1e-20 else None
def parity(p,m):
    p=np.asarray(p,dtype=float);m=np.asarray(m,dtype=float);o=(p-m)/2;e=(p+m)/2
    oe=float(np.sum(o*o));ee=float(np.sum(e*e));c=cosine(p,m)
    return dict(response_cosine=c,response_angle_degrees=math.degrees(math.acos(c)) if c is not None else None,
        opposition_cosine=-c if c is not None else None,odd_norm=math.sqrt(oe),even_norm=math.sqrt(ee),
        sign_even_energy_fraction=ee/(ee+oe) if ee+oe>1e-20 else None),o,e
def summary(values):
    a=[float(v) for v in values if v is not None]
    return dict(n=len(a),min=min(a),median=float(np.median(a)),max=max(a)) if a else dict(n=0,min=None,median=None,max=None)
def csvwrite(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def log_probs(logits):
    x=np.asarray(logits,dtype=np.float64);mx=np.max(x,axis=-1,keepdims=True)
    return x-mx-np.log(np.exp(x-mx).sum(axis=-1,keepdims=True))
def KL(b,c):
    lb=log_probs(b);lc=log_probs(c)
    return float(np.sum(np.exp(lb)*(lb-lc)))

verified={}
for root in [PILOT,MATCHED]:
    assert js(root,'STATUS.json')['status']=='complete'
    manifest=js(root,'EXPORT_HASHES.json')
    for member,digest in manifest.items():assert sha(root/member)==digest,member
    verified[str(root)]=len(manifest)
for name,digest in js(PILOT,'MLP_PREDICTION_FREEZE.json').items():assert sha(PILOT/'MLP_predictions'/name)==digest

mlp=js(PILOT,'MLP_RESPONSE_METRICS.json');energy=js(PILOT,'FUNCTIONAL_ENERGY.json')
mlp_support=[]
for kind in ['branch','complete']:
    a=[r for r in mlp if r['split']=='held_out' and r['direction_split']=='held_out' and r['precision']=='FP32' and r['map']==kind and r['strength'] in [.0078125,.03125]]
    resolved=[r for r in a if r['resolved']]
    mlp_support.append(dict(map=kind,rows=len(a),resolved=len(resolved),support_count=sum(r['first_relative_error']<=.1 for r in resolved),
        absolute_error=summary(r['first_absolute_error'] for r in resolved),relative_error=summary(r['first_relative_error'] for r in resolved),
        all_declared_resolved_cases_pass=len(resolved)==len(a) and all(r['first_relative_error']<=.1 for r in resolved)))
energy_support=[]
for eps in [.03125,.125,.5]:
    a=[r for r in energy if r['split']=='held_out' and r['strength']==eps];resolved=[r for r in a if r['resolved_KL']]
    energy_support.append(dict(strength=eps,rows=len(a),resolved=len(resolved),support_count_20pct=sum(r['relative_error']<=.2 for r in resolved),
        absolute_error=summary(r['absolute_error'] for r in resolved),relative_error=summary(r['relative_error'] for r in resolved),
        measured_KL_nats=summary(r['measured_KL_raw'] for r in resolved),
        all_pass_20pct=len(resolved)==len(a) and all(r['relative_error']<=.2 for r in resolved)))
max_kl_error=0.;max_variance_error=0.
for row in energy:
    pid=row['question'];di=row['direction'];eps=row['strength'];sign=row['sign']
    pred=PILOT/'functional_energy'/f'{pid}_d{di}_prediction.npz'
    assert sha(pred)==js(PILOT/'functional_energy',f'{pid}_d{di}_prediction.json')['prediction_sha256']
    with np.load(pred,allow_pickle=False) as z:b=z['base_logits'].astype(float);q=z['q_logits'].astype(float)
    with np.load(PILOT/'functional_energy'/f'{pid}_d{di}_eps{eps}_sign{sign}.npz',allow_pickle=False) as z:c=z['logits'].astype(float)
    p=np.exp(log_probs(b));center=q-np.sum(p*q);variance=float(np.sum(p*center*center))
    measured=KL(b,c);max_kl_error=max(max_kl_error,abs(measured-row['measured_KL_raw']))
    max_variance_error=max(max_variance_error,abs(variance-row['fisher_variance']))
assert max_kl_error<1e-11 and max_variance_error<1e-11

mlp_parity=[]
for p in sorted((PILOT/'MLP_responses').glob('*_parity.json')):
    r=json.loads(p.read_text())
    if r['question'] not in ['energy03_python','energy04_packing'] or r['direction'] not in [2,3]:continue
    mlp_parity.append(r)
mlp_ranges=[]
for precision in ['FP32','BF16']:
    for kind in ['branch','complete']:
        for eps in [.001953125,.0078125,.03125,.125,.5,2.]:
            a=[r for r in mlp_parity if r['precision']==precision and r['map']==kind and r['strength']==eps]
            b=[r for r in mlp if r['split']=='held_out' and r['direction_split']=='held_out' and r['precision']==precision and r['map']==kind and r['strength']==eps]
            mlp_ranges.append(dict(precision=precision,map=kind,strength=eps,opposition_cosine=summary(r['opposition_cosine'] for r in a),
                sign_even_fraction=summary(r['even_energy_fraction'] for r in a),
                first_error=summary(r['first_relative_error'] for r in b if r['resolved']),second_error=summary(r['second_relative_error'] for r in b if r['resolved']),
                zero_input_rows=sum(r['zero_delivered'] for r in b),zero_output_rows=sum(r['zero_output'] for r in b),unresolved_rows=sum(not r['resolved'] for r in b)))

memory=[];lookup={}
for row in js(PILOT,'CAUSAL_MEMORY_INDEX.json'):
    with np.load(PILOT/'causal_memory'/row['file'],allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    lookup[row['question'],row['precision'],row['mode'],row['position'],row['alpha']]=data
for pid in ['energy03_python','energy04_packing']:
    for precision in ['FP32','BF16']:
        for mode in ['held_fixed_state','persistent_last','persistent_all']:
            positions=['prefill','decode1'] if mode=='held_fixed_state' else ['four_steps']
            for position in positions:
                base=lookup[pid,precision,mode if mode!='persistent_last' else 'persistent_all',position,0.]
                for eps in [.03125,.5]:
                    plus=lookup[pid,precision,mode,position,eps];minus=lookup[pid,precision,mode,position,-eps]
                    for name in ['site10_before']:
                        assert np.array_equal(base[name],plus[name]) and np.array_equal(base[name],minus[name]),(pid,precision,mode,position)
                    dp=plus['readout15'].astype(float)-base['readout15'].astype(float)
                    dm=minus['readout15'].astype(float)-base['readout15'].astype(float)
                    phase_slices=[('prefill',slice(0,1)),('decode1',slice(1,2)),('decode_all3',slice(1,4))] if dp.ndim==2 else [(position,None)]
                    for phase,sl in phase_slices:
                        p=dp if sl is None else dp[sl];m=dm if sl is None else dm[sl]
                        metrics,odd,even=parity(p,m)
                        memory.append(dict(question=pid,precision=precision,mode=mode,phase=phase,strength=eps,**metrics))
csvwrite(OUT/'CAUSAL_MEMORY_PARITY.csv',memory)

snapshot_index=js(MATCHED,'SNAPSHOT_INDEX.json');snapshot_lookup={}
for row in snapshot_index:
    with np.load(MATCHED/'snapshots'/row['file'],allow_pickle=False) as z:data={k:z[k].astype(float) for k in z.files}
    snapshot_lookup[row['question'],row['position'],row['schedule'],row['strength'],row['sign'],row['precision']]=data
matched_rows=[]
for key in sorted(set((r['question'],r['position'],r['schedule'],r['strength']) for r in snapshot_index)):
    pid,position,schedule,eps=key
    sets={precision:[snapshot_lookup[pid,position,schedule,eps,sign,precision] for sign in [1,-1]] for precision in ['FP32','BF16']}
    fp32,_,e32=parity(sets['FP32'][0]['response'],sets['FP32'][1]['response'])
    bf16,_,e16=parity(sets['BF16'][0]['response'],sets['BF16'][1]['response'])
    linear,_,elin=parity(sets['FP32'][0]['J_actual_delta'],sets['FP32'][1]['J_actual_delta'])
    smooth=e32-elin;precision_difference=e16-e32
    assert size(e16-(elin+smooth+precision_difference))<1e-10
    denom=float(np.sum(e16*e16))
    projections={name:float(np.sum(vector*e16)/denom) if denom>1e-20 else None for name,vector in [
        ('linear_actual_input',elin),('FP32_finite_remainder',smooth),('downstream_precision_difference',precision_difference)]}
    with np.load(MATCHED/'snapshots'/f'{pid}_{position}_{schedule}_eps{eps}_sign1_prediction.npz',allow_pickle=False) as z:ip=z['actual_delta'].copy()
    with np.load(MATCHED/'snapshots'/f'{pid}_{position}_{schedule}_eps{eps}_sign-1_prediction.npz',allow_pickle=False) as z:im=z['actual_delta'].copy()
    input_metrics,_,ei=parity(ip,im)
    matched_rows.append(dict(question=pid,position=position,schedule=schedule,strength=eps,
        actual_input_opposition_cosine=input_metrics['opposition_cosine'],actual_input_even_norm=size(ei),
        FP32_opposition_cosine=fp32['opposition_cosine'],BF16_opposition_cosine=bf16['opposition_cosine'],
        linear_prediction_opposition_cosine=linear['opposition_cosine'],FP32_even_energy_fraction=fp32['sign_even_energy_fraction'],
        BF16_even_energy_fraction=bf16['sign_even_energy_fraction'],native_even_norm=size(e16),linear_input_even_output_norm=size(elin),
        linear_even_gain=size(elin)/size(ei) if size(ei)>1e-20 else None,FP32_finite_even_remainder_norm=size(smooth),
        precision_even_difference_norm=size(precision_difference),
        linear_input_signed_projection=projections['linear_actual_input'],FP32_remainder_signed_projection=projections['FP32_finite_remainder'],
        precision_difference_signed_projection=projections['downstream_precision_difference'],
        signed_projection_sum=sum(projections.values()) if denom>1e-20 else None))
csvwrite(OUT/'MATCHED_SNAPSHOT_DECOMPOSITION.csv',matched_rows)

old=[]
for precision in ['BF16','FP32']:
    arrays={}
    for condition in ['base','plus_v','minus_v']:
        with np.load(MATCHED/'historical_reproduction'/f'{precision}_{condition}.npz',allow_pickle=False) as z:arrays[condition]={k:z[k].copy() for k in z.files}
    assert all(np.array_equal(arrays['base']['forced_ids'],arrays[a]['forced_ids']) and np.array_equal(arrays['base']['site10_before'],arrays[a]['site10_before']) for a in ['plus_v','minus_v'])
    for phase,sl in [('prefill',slice(0,1)),('decode',slice(1,28)),('all',slice(None))]:
        dp=arrays['plus_v']['readout15'][sl].astype(float)-arrays['base']['readout15'][sl].astype(float)
        dm=arrays['minus_v']['readout15'][sl].astype(float)-arrays['base']['readout15'][sl].astype(float)
        metrics,_,_=parity(dp,dm)
        token_cos=[cosine(a,b) for a,b in zip(dp,dm)]
        delivered_plus=arrays['plus_v']['delivered'][sl];delivered_minus=arrays['minus_v']['delivered'][sl]
        iparity,_,_=parity(delivered_plus,delivered_minus)
        old.append(dict(precision=precision,phase=phase,positions=len(dp),mean_token_response_cosine=float(np.mean(token_cos)),
            mean_token_opposition_cosine=-float(np.mean(token_cos)),site_opposition_cosine=iparity['opposition_cosine'],
            **metrics))
dump(OUT/'HISTORICAL_REPRODUCTION.json',old)

checks={'functional_KL_independent_max_absolute_error':max_kl_error,'functional_Fisher_variance_independent_max_absolute_error':max_variance_error,
    'current_site_input_identical_across_signed_causal_paths':True,'matched_snapshot_even_decomposition_exact':True,
    'all_prediction_and_native_manifest_hashes_verified':verified}
results={'date':'2026-10-12','status':'bounded staged study complete','fresh_independent_questions':4,'calibration_questions':2,'held_out_questions':2,
    'held_out_directions':2,'historical_reproduction_questions':1,'GPU_allocations':0,
    'pilot_native_backbone_calls':js(PILOT,'STATUS.json')['whole_backbone_calls'],'followup_native_backbone_calls':js(MATCHED,'STATUS.json')['whole_backbone_calls'],
    'followup_tail_evaluations':js(MATCHED,'STATUS.json')['tail_evaluations'],
    'MLP_numerical_gate':js(PILOT,'MLP_NUMERICAL_GATE.json'),'functional_numerical_gate':js(PILOT,'FUNCTIONAL_NUMERICAL_GATE.json'),
    'attention_numerical_gate':js(PILOT,'ATTENTION_NUMERICAL_GATE.json'),'matched_snapshot_numerical_gate':js(MATCHED,'NUMERICAL_GATE.json'),
    'MLP_prespecified_small_strength_support':mlp_support,'functional_energy_support_by_strength':energy_support,
    'MLP_validity_ranges':mlp_ranges,'matched_snapshot_summary':{
        name:summary(r[name] for r in matched_rows) for name in ['FP32_opposition_cosine','BF16_opposition_cosine','linear_prediction_opposition_cosine',
            'linear_input_signed_projection','FP32_remainder_signed_projection','precision_difference_signed_projection']},
    'historical_reproduction':old,'checks':checks,'combined_CPU_window_seconds':js(MATCHED,'EXPORT_RECEIPT.json')['combined_elapsed_seconds'],
    'scope':'Conditional local response and output cost, not classifier/global Lyapunov law/full-space rotation. Original failures and stopped monitor unchanged.'}
dump(OUT/'RESULTS.json',results);dump(OUT/'MLP_VALIDITY_RANGES.json',mlp_ranges)
print(json.dumps({'MLP_support':mlp_support,'energy_support':energy_support,'matched_snapshot_summary':results['matched_snapshot_summary'],'historical_reproduction_decode':[r for r in old if r['phase']=='decode']},indent=2))
