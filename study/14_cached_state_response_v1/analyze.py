"""Frozen-rule scoring and independent saved-array audit, without model calls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import hashlib,json,sys
from pathlib import Path
import numpy as np
from cached_geometry import potential_and_identities

BASE=Path('C:/Users/farha/Research4/artifacts/cached_state_response_v1')
ROOT=BASE/(sys.argv[1] if len(sys.argv)>1 else 'native_cpu_20261014_retry1')
assert ROOT.parent==BASE and ROOT.name in ['native_cpu_20261014_retry1','native_cpu_20261014_retry2']
OUT=BASE/'evaluated_20261014';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def dump(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
def stats(v):
    a=np.array([x for x in v if x is not None],dtype=np.float64)
    return {'n':len(a),'minimum':float(a.min()),'median':float(np.median(a)),'maximum':float(a.max())} if len(a) else {'n':0}
def lse(x):
    x=np.asarray(x,dtype=np.float64);m=x.max();return m+np.log(np.exp(x-m).sum())
def prob(x):return np.exp(np.asarray(x,dtype=np.float64)-lse(x))
def KL(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return max(0.,float(lse(b)-lse(a)-prob(a)@(b-a)))
def wn(p,x):
    x=np.asarray(x,dtype=np.float64);return float(np.sqrt(p@((x-p@x)**2)))
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
STATUS=read(ROOT/'STATUS.json');assert STATUS['status']=='native_complete'
PLAN=read(ROOT/'source/PLAN.json');INPUTS=read(ROOT/'INPUTS.json');SCHEMA=read(ROOT/'STATE_SCHEMA.json')
SELECT=read(ROOT/'DERIVATIVE_SELECTION_LOCK.json');ROWS=read(ROOT/'METRICS.json');assert len(ROWS)==48
GROUPS=['hidden','residual','mixer','mlp','memory_kv','memory_conv','memory_recurrent']
METHODS=['quadratic_Fisher','first_order_logits','second_order_logits']
MANIFEST=read(ROOT/'EXPORT_HASHES.json');assert all(sha(ROOT/n)==h for n,h in MANIFEST.items())
FREEZE=read(ROOT/'PREDICTION_FREEZE.json');assert FREEZE['all_forecasts_precede_endpoints'] and FREEZE['target_endpoints_seen']==0
assert sha(ROOT/'DERIVATIVE_SELECTION_LOCK.json')==FREEZE['selection_sha256']
assert all(sha(ROOT/n)==h for n,h in FREEZE['sha256'].items())
audit={'main_endpoints':0,'mixed_endpoints':0,'stress_endpoints':0,'fields_checked':0,
       'max_KL_difference':0.,'max_forecast_cost_difference':0.,'max_weighted_logit_error_difference':0.,
       'max_state_error_difference':0.,'manifest_files_verified':len(MANIFEST),'new_model_calls':0}

def audit_row(row,actual,base,c):
    alpha=row['alpha'];p=prob(base['logits']);q=c['q'].astype(np.float64);r=c['r'].astype(np.float64)
    measured=KL(base['logits'],actual['logits'])
    audit['max_KL_difference']=max(audit['max_KL_difference'],abs(measured-row['actual_KL']))
    costs={'quadratic_Fisher':.5*alpha**2*wn(p,q)**2}
    for method,order in [('first_order_logits',1),('second_order_logits',2)]:
        delta=alpha*q+(.5*alpha**2*r if order==2 else 0.)
        forecast=base['logits'].astype(np.float64)+delta
        costs[method]=KL(base['logits'],forecast)
        true=actual['logits'].astype(np.float64)-base['logits']
        error=wn(p,delta-true)/max(wn(p,true),1e-30)
        audit['max_weighted_logit_error_difference']=max(audit['max_weighted_logit_error_difference'],abs(error-row['weighted_logit_errors'][method]))
        audit['max_KL_difference']=max(audit['max_KL_difference'],abs(KL(actual['logits'],forecast)-row['distribution_KL_actual_to_forecast'][method]))
        group_sums={g:[0.,0.] for g in GROUPS}
        for n,schema in SCHEMA.items():
            true_field=actual[n].astype(np.float64)-base[n]
            forecast_field=alpha*c[n+'_first'].astype(np.float64)+(.5*alpha**2*c[n+'_second'] if order==2 else 0.)
            en=norm(forecast_field-true_field);tn=norm(true_field);group=schema['group']
            if group in group_sums:group_sums[group][0]+=tn*tn;group_sums[group][1]+=en*en
            if order==2:
                f=row['fields'][n];resolved=tn>max(1e-10,20*np.finfo(np.float32).eps*norm(base[n]))
                assert f['resolved']==resolved and f['exactly_unchanged']==np.array_equal(actual[n],base[n])
                audit['max_state_error_difference']=max(audit['max_state_error_difference'],abs(f['change_norm']-tn),abs(f['error_norm']-en))
                if resolved:audit['max_state_error_difference']=max(audit['max_state_error_difference'],abs(f['relative_error']-en/tn))
                audit['fields_checked']+=1
        for group,(tn2,en2) in group_sums.items():
            if np.sqrt(tn2)>1e-12:
                error=float(np.sqrt(en2/tn2))
                audit['max_state_error_difference']=max(audit['max_state_error_difference'],abs(error-row['state_errors'][str(order)][group]['relative_error']))
    for method,cost in costs.items():audit['max_forecast_cost_difference']=max(audit['max_forecast_cost_difference'],abs(cost-row['cost_predictions'][method]))

geometry=[];residual_identities=[];potential_symmetry=[]
for question in INPUTS['questions']:
    pid=question['id'];base=load(ROOT/'contexts'/f'{pid}_baseline.npz')
    coeff=[load(ROOT/'coefficients'/f'{pid}_d{di}.npz') for di in range(2)]
    identities,potential=potential_and_identities(ROOT,pid,coeff,SELECT)
    residual_identities.extend(identities)
    potential_symmetry.extend([{**r,'split':question['split']} for r in potential])
    p=prob(base['logits']);q=np.stack([c['q'].astype(np.float64) for c in coeff]);q-=q@p[:,None]
    G=(q*p)@q.T;lam,vec=np.linalg.eigh(G)
    row={'question':pid,'split':question['split'],'Fisher_metric_in_sampled_plane':G.tolist(),
         'plane_eigenvalues':lam.tolist(),'plane_stiffness_ratio':float(lam[-1]/lam[0]) if lam[0]>1e-15 else None,
         'high_cost_plane_direction_coordinates':vec[:,-1].tolist(),'full_space_spectrum':False,'semantic_direction_labels':False,
         'layer_plane_metrics':[]}
    for layer in range(11,24):
        n=f'b{layer}_output';J=np.stack([c[n+'_first'].astype(np.float64) for c in coeff]);gram=J@J.T
        row['layer_plane_metrics'].append({'layer':layer,'response_gram':gram.tolist(),'eigenvalues':np.linalg.eigvalsh(gram).tolist(),
            'curvature_norms':[norm(c[n+'_second']) for c in coeff]})
    geometry.append(row)
    for actual_row in [r for r in ROWS if r['question']==pid]:
        di=actual_row['direction'];a=actual_row['alpha'];actual=load(ROOT/'actual'/f'{pid}_d{di}_a{a}.npz')
        audit_row(actual_row,actual,base,coeff[di]);audit['main_endpoints']+=1
    for kind,filename in [('stress','STRESS_METRICS.json'),('mixed','MIXED_METRICS.json')]:
        if not (ROOT/filename).exists():continue
        for actual_row in [r for r in read(ROOT/filename) if r['question']==pid]:
            a=actual_row['alpha'];di=actual_row['direction']
            if kind=='stress':actual=load(ROOT/'stress'/f'{pid}_a{a}.npz');c=coeff[di]
            else:actual=load(ROOT/'mixed'/f'{pid}_{di}_a{a}_actual.npz');c=load(ROOT/'mixed'/f'{pid}_{di}.npz')
            audit_row(actual_row,actual,base,c);audit[kind+'_endpoints']+=1
    del base,coeff
    print(json.dumps({'audited_question':pid,**audit}),flush=True)
assert max(audit['max_KL_difference'],audit['max_forecast_cost_difference'],audit['max_weighted_logit_error_difference'],audit['max_state_error_difference'])<1e-9
dump('INDEPENDENT_AUDIT.json',audit);dump('SAMPLED_PLANE_GEOMETRY.json',geometry)
dump('RESIDUAL_DERIVATIVE_IDENTITIES.json',residual_identities)
dump('POTENTIAL_SYMMETRY.json',potential_symmetry)
supplementary={'residual_identity_discrepancies':{order:stats([r['relative_discrepancy'] for r in residual_identities if r['order']==order and not r['zero_response']]) for order in ['first','second']},
 'held_potential_checks':{name:{'informative':sum(r['informative'] for r in potential_symmetry if r['split']=='held_out' and r['submap']==name),
 'evidence_against_simple_gradient':sum(r['evidence_against_simple_Euclidean_gradient_form'] for r in potential_symmetry if r['split']=='held_out' and r['submap']==name)} for name in ['attention_plus_norm','whole_block_update','MLP_plus_norm']},
 'new_model_calls':0,'primary_rules_unchanged':True,'no_physical_energy_conservation_claim':True}
dump('SUPPLEMENTARY_GEOMETRY_SUMMARY.json',supplementary)

HELD=[r for r in ROWS if r['split']=='held_out'];assert len(HELD)==32
def median(rows,field,method):return float(np.median([r[field][method] for r in rows]))
support=[]
for question in INPUTS['questions']:
    if question['split']!='held_out':continue
    panel=[r for r in HELD if r['question']==question['id']];mid=[r for r in panel if r['strength']==.125];weak=[r for r in panel if r['strength']==.03125]
    resolved=all(r['resolved'] and not r['zero_delivery'] for r in panel)
    gates={'signed_resolved':resolved,'numerical_validation':all(r['derivative_validated'] for r in panel)}
    if resolved:
        gates.update(cost_each_le_20pct=all(r['cost_errors']['second_order_logits']<=.2 for r in panel),
            weighted_logits_each_le_15pct=all(r['weighted_logit_errors']['second_order_logits']<=.15 for r in panel),
            cost_median_improves_vs_Fisher=median(mid,'cost_errors','second_order_logits')<=.75*median(mid,'cost_errors','quadratic_Fisher'),
            cost_median_improves_vs_first=median(mid,'cost_errors','second_order_logits')<=.75*median(mid,'cost_errors','first_order_logits'),
            logit_median_improves_vs_first=median(mid,'weighted_logit_errors','second_order_logits')<=.75*median(mid,'weighted_logit_errors','first_order_logits'),
            weak_cost_not_degraded=median(weak,'cost_errors','second_order_logits')<=median(weak,'cost_errors','first_order_logits')+.005,
            weak_logits_not_degraded=median(weak,'weighted_logit_errors','second_order_logits')<=median(weak,'weighted_logit_errors','first_order_logits')+.005)
    for g in PLAN['primary_support']['required_state_groups']:
        values=[r['state_errors']['2'][g]['relative_error'] for r in panel]
        gates[g+'_each_le_10pct']=all(v is not None and v<=.10 for v in values)
    fields=[f for r in panel for f in r['fields'].values() if f['group'] in PLAN['primary_support']['required_state_groups'] and f['resolved']]
    gates['resolved_output_and_memory_fields_le_15pct']=bool(fields) and all(f['relative_error']<=.15 for f in fields)
    support.append({'question':question['id'],'family':question['family'],'all_rules_pass':all(gates.values()),'gates':gates,'resolved_field_cases':len(fields)})
dump('QUESTION_SUPPORT.json',support)
by_strength={}
for strength in PLAN['primary_strengths']:
    panel=[r for r in HELD if r['strength']==strength]
    by_strength[str(strength)]={'cases':len(panel),'cost_errors':{m:stats([r['cost_errors'][m] for r in panel]) for m in METHODS},
      'weighted_logit_errors':{m:stats([r['weighted_logit_errors'][m] for r in panel]) for m in METHODS[1:]},
      'state_errors':{g:{order:stats([r['state_errors'][order][g]['relative_error'] for r in panel]) for order in ['1','2']} for g in GROUPS},
      'resolved_fields_error':{g:stats([f['relative_error'] for r in panel for f in r['fields'].values() if f['group']==g and f['resolved']]) for g in GROUPS}}
mixed=read(ROOT/'MIXED_METRICS.json') if (ROOT/'MIXED_METRICS.json').exists() else []
mix_summary={}
for split in ['calibration','held_out']:
    for direction in ['sum','difference','weighted']:
        panel=[r for r in mixed if r['split']==split and r['direction']==direction]
        if panel:mix_summary[split+'_'+direction]={'cases':len(panel),'validated':all(r['derivative_validated'] for r in panel),
          'cost_full_interaction':stats([r['cost_errors']['second_order_logits'] for r in panel]),
          'cost_without_interaction':stats([r['no_interaction_cost_error'] for r in panel]),
          'weighted_logit_error':stats([r['weighted_logit_errors']['second_order_logits'] for r in panel]),
          'not_differentiated_at_target_direction':direction!='sum'}
dump('MIXED_SUMMARY.json',mix_summary)
stress=read(ROOT/'STRESS_METRICS.json') if (ROOT/'STRESS_METRICS.json').exists() else []
localization=[]
for r in stress:
    trajectory=[{'layer':i,**r['fields'][f'b{i}_output']} for i in range(11,24)]
    over=[x['layer'] for x in trajectory if x['resolved'] and x['relative_error']>.15]
    localization.append({'question':r['question'],'split':r['split'],'alpha':r['alpha'],'cost_error':r['cost_errors']['second_order_logits'],
       'first_observed_output_over_15pct':over[0] if over else None,'trajectory':trajectory,
       'cumulative_response_error_not_causal_local_attribution':True})
dump('STRESS_LOCALIZATION.json',localization)
suffix='_RETRY2' if ROOT.name.endswith('_retry2') else '_RETRY1'
receipt=read(BASE/f'NATIVE_EXPORT_RECEIPT_20261014{suffix}.json')
result={'primary_support':all(x['all_rules_pass'] for x in support),'held_questions_passing':sum(x['all_rules_pass'] for x in support),
 'held_unique_questions':4,'calibration_unique_questions':2,'main_endpoints':48,'held_signed_endpoints':32,'by_strength':by_strength,
 'derivative_methods':SELECT['modes'],'derivative_attempts':SELECT['attempts'],'probe_steps':[SELECT['coarse'],SELECT['fine']],
 'field_count':len(SCHEMA),'memory_field_count':sum(s['group'].startswith('memory_') for s in SCHEMA.values()),
 'mixed_summary':mix_summary,'stress_cases':len(stress),'audit':audit,'model_calls':STATUS['model_calls'],
 'supplementary_geometry':supplementary,
 'phase_native_export_seconds_including_first_stop':receipt['native_export_seconds'],
 'CPU_only':True,'GPU_allocations':0,'Colab':False,'limits':PLAN['limits'],'broad_goal_achieved':False}
dump('RESULTS.json',result)
lines=['CACHED DECODING RESPONSE — 14 October 2026','',
 f"Prospective primary: {result['primary_support']}; held questions passing {result['held_questions_passing']}/4.",
 'Six fresh questions: two calibration, four held out. Two fresh known directions, both signs.',
 'Actual complete incoming cache cloned per call; same normal first generated token forced for all variants.',
 'One intervention at block11 input; one cached prediction position. Original frozen model operations.',
 f"Readouts: {len(SCHEMA)} activation/memory fields, including {result['memory_field_count']} relevant outgoing memory tensors.",
 'Equation for every measured state: y(alpha) ≈ y(0) + alpha*q + alpha²*r/2.',
 'Output probabilities: softmax of predicted full-vocabulary logits.',
 f"Derivative methods locked on calibration: {SELECT['modes']}; steps {SELECT['coarse']}/{SELECT['fine']}.",'']
for s,d in by_strength.items():
    lines.append(f'Strength {s}, {d["cases"]} held signed cases:')
    for m,v in d['cost_errors'].items():lines.append(f'  {m} cost error median/max {100*v["median"]:.4g}% / {100*v["maximum"]:.4g}%')
    v=d['weighted_logit_errors']['second_order_logits'];lines.append(f'  second weighted-logit error median/max {100*v["median"]:.4g}% / {100*v["maximum"]:.4g}%')
    for g in PLAN['primary_support']['required_state_groups']:
        v=d['state_errors'][g]['2'];lines.append(f'  {g} second response error median/max {100*v["median"]:.4g}% / {100*v["maximum"]:.4g}%')
lines+=['','Failed gates:']+[f'  {s["question"]}: {[k for k,v in s["gates"].items() if not v]}' for s in support]
lines+=['','Geometry: the measured 2D Fisher matrix assigns different output costs to equal-sized pushes;',
 'the mixed second derivative tests whether two directions combine additively.',
 'These are local response geometry measurements, not a physical conservation or equilibrium law.',
 'Mixed-plane difference/weighted directions are assembled from the plane equation, not independently differentiated.',
 'Stress results and where errors accumulate are saved separately; they do not rescue failed primary gates.',
 f'Independent audit: {audit}.',f"Resources: {STATUS['model_calls']} forwards; {receipt['native_export_seconds']:.3f} seconds including prior zero-call stop; CPU only.",
 'Remaining: other intervention sites/positions, stronger-change remedy with fresh confirmation, broader directions and contexts.',
 'No universal equation, unknown-steering detector, semantic direction dictionary, speedup or physical energy claim.']
(OUT/'FINAL_FINDINGS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'result':str(OUT/'RESULTS.json'),'primary_support':result['primary_support'],'held_passing':result['held_questions_passing'],'audit':audit}))
