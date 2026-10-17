"""Independent state/logit audit, prospective gates, geometry and conditional cost."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import hashlib,json,time
from pathlib import Path
import numpy as np
START=time.monotonic();ROOT=Path(__file__).resolve().parent.parent;BASE=ROOT.parent;OUT=BASE/'evaluated_20261014';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
def lse(x):
    x=np.asarray(x,dtype=np.float64);m=x.max();return m+np.log(np.exp(x-m).sum())
def prob(x):return np.exp(np.asarray(x,dtype=np.float64)-lse(x))
def KL(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);return max(0.,float(lse(b)-lse(a)-prob(a)@(b-a)))
def wn(p,x):
    x=np.asarray(x,dtype=np.float64);return float(np.sqrt(max(0.,p@((x-p@x)**2))))
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def stats(xs):
    x=np.asarray([v for v in xs if v is not None],dtype=np.float64)
    return {'n':len(x),'median':float(np.median(x)),'maximum':float(x.max()),'minimum':float(x.min())} if len(x) else {'n':0}
STATUS=read(ROOT/'STATUS.json');assert STATUS['status']=='native_complete'
PLAN=read(ROOT/'source/PLAN.json');SCHEMA=read(ROOT/'STATE_SCHEMA.json');NAMES=list(SCHEMA);INPUTS=read(ROOT/'INPUTS.json')
ROWS=read(ROOT/'METRICS.json');HELD=[r for r in ROWS if r['split']=='held_out'];assert len(HELD)==32
GROUPS=['hidden','residual','mixer','mlp','memory_kv','memory_conv','memory_recurrent']
CHAIN=read(ROOT/'CHAIN_RULE_CONTROL.json');CUBIC=read(ROOT/'CUBIC_SELECTION_LOCK.json');ALLOC=read(ROOT/'CALL_ALLOCATION_LOCK.json')
FREEZE=read(ROOT/'PREDICTION_FREEZE.json');assert FREEZE['T2_target_endpoints_seen']==0 and FREEZE['all_forecasts_precede_endpoints']
assert all(sha(ROOT/n)==h for n,h in FREEZE['sha256'].items())
assert sha(ROOT/'DERIVATIVE_SELECTION_LOCK.json')==FREEZE['selection_sha256']
assert sha(ROOT/'CUBIC_SELECTION_LOCK.json')==FREEZE['cubic_selection_sha256'] and sha(ROOT/'CALL_ALLOCATION_LOCK.json')==FREEZE['allocation_sha256']
MANIFEST=read(ROOT/'EXPORT_HASHES.json');assert all(sha(ROOT/n)==h for n,h in MANIFEST.items())
assert read(ROOT/'CACHE_RECONSTRUCTION_CONTROL.json')['exact'] and read(ROOT/'NULL_CONTROL.json')['exact']
CARRY=read(ROOT/'MEMORY_CARRY_CHECKS.json');assert all(r['previous_KV_exact']==8 and r['b11_input_tangent_zero'] for r in CARRY)
AUDIT={'endpoints':0,'field_order_cases':0,'frozen_state_predictions_verified':0,'manifest_files':len(MANIFEST),'new_model_calls':0,
    'max_KL_difference':0.,'max_scaled_weighted_error_difference':0.,'max_scaled_state_error_difference':0.}
def audited(row,actual,base,c,forecasts=None,diagonal=False):
    a=row['alpha'];p=prob(base['logits']);cost=KL(base['logits'],actual['logits'])
    AUDIT['max_KL_difference']=max(AUDIT['max_KL_difference'],abs(cost-row['actual_KL']))
    assert np.array_equal(actual['b11_input'],base['b11_input']) and row['current_b11_input_exactly_unchanged']
    for order_text,record in row['orders'].items():
        order=int(order_text)
        delta=a*c['q'].astype(np.float64)
        if order>=2:delta+=.5*a*a*c['r'].astype(np.float64)
        if order==3:delta+=a**3*c['t']/6
        logits=base['logits'].astype(np.float64)+delta
        predicted=KL(base['logits'],logits)
        AUDIT['max_KL_difference']=max(AUDIT['max_KL_difference'],abs(predicted-record['predicted_KL']),abs(KL(actual['logits'],logits)-record['actual_to_prediction_KL']))
        truth=actual['logits'].astype(np.float64)-base['logits']
        err=wn(p,logits-base['logits']-truth)/max(wn(p,truth),1e-30)
        AUDIT['max_scaled_weighted_error_difference']=max(AUDIT['max_scaled_weighted_error_difference'],abs(err-record['weighted_logit_error'])/max(1.,abs(record['weighted_logit_error'])))
        group_sums={g:[0.,0.] for g in GROUPS};state_delta={}
        for n in NAMES:
            d=a*c[n+'_first'].astype(np.float64)
            if order>=2:d+=.5*a*a*c[n+'_second'].astype(np.float64)
            if order==3:d+=a**3*c[n+'_third']/6
            state_delta[n]=d;true=actual[n].astype(np.float64)-base[n];tn=norm(true);en=norm(d-true);f=record['fields'][n]
            resolved=tn>max(1e-10,20*np.finfo(np.float32).eps*norm(base[n]))
            assert f['resolved']==bool(resolved) and f['exactly_unchanged']==bool(np.array_equal(actual[n],base[n]))
            diffs=[abs(tn-f['change_norm'])/max(1.,tn),abs(en-f['error_norm'])/max(1.,en)]
            if resolved:diffs.append(abs(en/tn-f['relative_error'])/max(1.,abs(f['relative_error'])))
            AUDIT['max_scaled_state_error_difference']=max(AUDIT['max_scaled_state_error_difference'],*diffs)
            group=SCHEMA[n]['group']
            if group in group_sums:group_sums[group][0]+=tn*tn;group_sums[group][1]+=en*en
            AUDIT['field_order_cases']+=1
        for g,(t,e) in group_sums.items():
            if t>1e-24:
                value=float(np.sqrt(e/t));ref=record['groups'][g]['relative_error']
                AUDIT['max_scaled_state_error_difference']=max(AUDIT['max_scaled_state_error_difference'],abs(value-ref)/max(1.,abs(ref)))
        if order>=2 and forecasts is not None:
            h=hashlib.sha256()
            for n in sorted(state_delta):h.update(n.encode()+b'\0');h.update(state_delta[n].tobytes())
            expected=forecasts[str(order)]['state_delta_sha256'] if str(order) in forecasts else forecasts['diagonal_state_delta_sha256' if diagonal else 'full_state_delta_sha256']
            assert h.hexdigest()==expected;AUDIT['frozen_state_predictions_verified']+=1
            frozen_cost=forecasts[str(order)]['cost'] if str(order) in forecasts else forecasts['diagonal_cost' if diagonal else 'full_cost']
            AUDIT['max_KL_difference']=max(AUDIT['max_KL_difference'],abs(predicted-frozen_cost))
    AUDIT['endpoints']+=1
for q in INPUTS['questions']:
    pid=q['id'];base=load(ROOT/'contexts'/f'{pid}_baseline.npz')
    for di in range(2):
        c=load(ROOT/'coefficients'/f'{pid}_d{di}.npz');fs=read(ROOT/'predictions'/f'{pid}_d{di}.json')
        for row in [r for r in ROWS if r['question']==pid and r['direction']==di]:
            actual=load(ROOT/'actual'/f'{pid}_d{di}_a{row["alpha"]}.npz');f=next(x for x in fs if x['alpha']==row['alpha']);audited(row,actual,base,c,f)
MIXED=read(ROOT/'MIXED_METRICS.json') if (ROOT/'MIXED_METRICS.json').exists() else []
if ALLOC['full_plane_admitted']:
    for row in MIXED:
        pid=row['question'];name=row['direction'];a=row['alpha'];base=load(ROOT/'contexts'/f'{pid}_baseline.npz')
        actual=load(ROOT/'mixed'/f'{pid}_{name}_a{a}_actual.npz');c=load(ROOT/'mixed'/f'{pid}_{name}.npz')
        f=next(x for x in read(ROOT/'mixed'/f'{pid}_{name}_forecasts.json') if x['alpha']==a);audited(row,actual,base,c,f)
        diag=load(ROOT/'mixed'/f'{pid}_{name}_diagonal.npz');drow={**row,'orders':{'2':row['diagonal']}};audited(drow,actual,base,diag,f,True)
elif MIXED:
    row=MIXED[0];base=load(ROOT/'contexts'/f'{row["question"]}_baseline.npz');c=load(ROOT/'mixed'/f'{row["question"]}_sum_diagonal_only.npz')
    audited(row,load(ROOT/'mixed/SINGLE_DIAGONAL_ACTUAL.npz'),base,c)
assert AUDIT['max_KL_difference']<1e-10 and AUDIT['max_scaled_weighted_error_difference']<1e-8 and AUDIT['max_scaled_state_error_difference']<1e-8
dump('INDEPENDENT_AUDIT.json',AUDIT)
def gates(panel,order,strict=False):
    key=str(order);strong=[r for r in panel if r['strength']==.125];weak=[r for r in panel if r['strength']==.03125]
    g={'resolved_delivery':all(r['resolved'] and not r['zero_delivery'] for r in panel),
       'current_residual_unchanged':all(r['current_b11_input_exactly_unchanged'] for r in panel),
       'derivative_validation':all(r['derivative_validated'] for r in panel),'chain_control':CHAIN['pass']}
    if order==3:g['calibration_cubic_lock']=CUBIC['enabled'];g['held_third_validation']=all(r['third_validated'] for r in panel)
    available=all(key in r['orders'] for r in panel);g['forecast_available']=available
    if not available:return g
    o=[r['orders'][key] for r in panel]
    g['signed_cost_le20pct']=all(r['cost_error'] is not None and r['cost_error']<=.20 for r in o)
    g['signed_weighted_logit_le15pct']=all(r['weighted_logit_error']<=.15 for r in o)
    g['current_KV_only_le10pct']=all(r['current_KV_only_error'] is not None and r['current_KV_only_error']<=.10 for r in o)
    for group in GROUPS if strict else ['residual','memory_kv','memory_conv','memory_recurrent']:
        values=[r['groups'][group]['relative_error'] for r in o];g[group+'_le10pct']=all(v is not None and v<=.10 for v in values)
    fields=[f for r in o for f in r['fields'].values() if f['resolved'] and (strict or f['group'] in ['residual','memory_kv','memory_conv','memory_recurrent'])]
    g['resolved_fields_le15pct']=bool(fields) and all(f['relative_error']<=.15 for f in fields)
    def median(rows,which):
        v=[r['orders'][key]['cost_error'] if which=='cost' else r['orders'][key]['weighted_logit_error'] for r in rows]
        return float(np.median(v)) if all(x is not None for x in v) else float('inf')
    g['strong_cost_improves_Fisher']=median(strong,'cost')<=.75*float(np.median([r['Fisher_cost_error'] for r in strong])) if all(r['Fisher_cost_error'] is not None for r in strong) else False
    g['strong_cost_improves_first']=median(strong,'cost')<=.75*float(np.median([r['orders']['1']['cost_error'] for r in strong])) if all(r['orders']['1']['cost_error'] is not None for r in strong) else False
    g['strong_logit_improves_first']=median(strong,'weighted')<=.75*float(np.median([r['orders']['1']['weighted_logit_error'] for r in strong]))
    g['weak_cost_not_degraded']=median(weak,'cost')<=float(np.median([r['orders']['1']['cost_error'] for r in weak]))+.005 if all(r['orders']['1']['cost_error'] is not None for r in weak) else False
    g['weak_logit_not_degraded']=median(weak,'weighted')<=float(np.median([r['orders']['1']['weighted_logit_error'] for r in weak]))+.005
    return g
SUPPORT={}
for order in [2,3]:
    support=[]
    for q in INPUTS['questions']:
        if q['split']!='held_out':continue
        panel=[r for r in HELD if r['question']==q['id']];g=gates(panel,order,order==3)
        support.append({'question':q['id'],'family':q['family'],'pass':all(g.values()),'gates':g})
    SUPPORT[str(order)]=support
dump('QUESTION_SUPPORT.json',SUPPORT)
SUMMARY={}
for strength in [.03125,.125]:
    panel=[r for r in HELD if r['strength']==strength];by_order={}
    for order in ['1','2','3']:
        selected=[r['orders'][order] for r in panel if order in r['orders']]
        by_order[order]={'cases':len(selected),'cost_error':stats([r['cost_error'] for r in selected]),
            'weighted_logit_error':stats([r['weighted_logit_error'] for r in selected]),
            'state_errors':{g:stats([r['groups'][g]['relative_error'] for r in selected]) for g in GROUPS},
            'current_KV_only_error':stats([r['current_KV_only_error'] for r in selected]),
            'resolved_field_error':stats([f['relative_error'] for r in selected for f in r['fields'].values() if f['resolved']])}
    SUMMARY[str(strength)]={'cases':len(panel),'orders':by_order,'Fisher_cost_error':stats([r['Fisher_cost_error'] for r in panel])}
geometry=[]
for q in INPUTS['questions']:
    pid=q['id'];base=load(ROOT/'contexts'/f'{pid}_baseline.npz');p=prob(base['logits']);c=[load(ROOT/'coefficients'/f'{pid}_d{di}.npz') for di in range(2)]
    Q=np.stack([v['q'].astype(np.float64) for v in c]);centered=Q-(Q@p)[:,None];G=(centered*p)@centered.T;lam,vec=np.linalg.eigh(G)
    row={'question':pid,'split':q['split'],'sampled_Fisher_metric':G.tolist(),'eigenvalues':lam.tolist(),
        'eigenvalue_ratio':float(lam[-1]/lam[0]) if lam[0]>1e-15 else None,'high_cost_axis_coordinates':vec[:,-1].tolist(),
        'conditional_on_fixed_second_token':True,'full_space_spectrum':False,'layer_metrics':[]}
    for layer in range(11,24):
        n=f'b{layer}_output';J=np.stack([v[n+'_first'].astype(np.float64) for v in c]);row['layer_metrics'].append({'layer':layer,'response_gram':(J@J.T).tolist(),
            'curvature_norms':[norm(v[n+'_second']) for v in c],'third_norms':[norm(v[n+'_third']) if n+'_third' in v else None for v in c]})
    geometry.append(row)
dump('SAMPLED_GEOMETRY.json',geometry)
mix_summary={'full_plane_admitted':ALLOC['full_plane_admitted'],'held_targets':sum(r['split']=='held_out' for r in MIXED),'support':False}
if ALLOC['full_plane_admitted']:
    held=[r for r in MIXED if r['split']=='held_out'];checks=[]
    for row in held:
        o=row['orders']['2'];fields=[f for f in o['fields'].values() if f['resolved'] and f['group'] in ['residual','memory_kv','memory_conv','memory_recurrent']]
        checks.append(row['derivative_validated'] and row['resolved'] and o['cost_error']<=.2 and o['weighted_logit_error']<=.15
            and all(o['groups'][g]['relative_error'] is not None and o['groups'][g]['relative_error']<=.10 for g in ['residual','memory_kv','memory_conv','memory_recurrent'])
            and bool(fields) and all(f['relative_error']<=.15 for f in fields))
    full=[r['orders']['2']['weighted_logit_error'] for r in held];diag=[r['diagonal']['weighted_logit_error'] for r in held]
    gain=float(np.median(full))/max(float(np.median(diag)),1e-30)
    mix_summary.update(support=bool(all(checks) and gain<=.75),all_endpoint_bounds_pass=bool(all(checks)),weighted_error_full=stats(full),
        weighted_error_diagonal=stats(diag),median_error_ratio_full_to_diagonal=gain,
        per_direction={name:{'full_weighted_error':stats([r['orders']['2']['weighted_logit_error'] for r in held if r['direction']==name]),
            'diagonal_weighted_error':stats([r['diagonal']['weighted_logit_error'] for r in held if r['direction']==name])} for name in ['sum','difference','weighted']},
        distinct_held_questions=1,unprobed_at_T2_directions=['difference','weighted'])
dump('MIXED_SUMMARY.json',mix_summary)
timings=read(ROOT/'CALL_TIMINGS.json');forecasts=read(ROOT/'FORECAST_TIMINGS.json');setup=read(ROOT/'JET_SETUP_TIMINGS.json')
native=[t['seconds'] for t in timings if t['label']=='held_actual' and t['kind']=='value']
cost={'conditional_native_value_seconds':stats(native),'forecast_seconds':{str(o):stats([t['seconds'] for t in forecasts if t['order']==o]) for o in [2,3]},
    'basis_setup_seconds_including_input_loading_and_probe_storage':stats([t['seconds_including_parent_loading_assembly_and_probe_storage'] for t in setup if t['kind']=='basis']),
    'parent_first_step_probe_acquisition_cost_not_included':True,'not_end_to_end_generation_speedup':True,'same_fixed_context_direction_and_token_assumption':True}
for order in ['2','3']:
    f=cost['forecast_seconds'][order]
    if f['n'] and native:
        saving=float(np.median(native))-f['median'];cost['conditional_native_to_forecast_ratio_'+order]=float(np.median(native))/max(f['median'],1e-30)
        cost['conditional_setup_amortization_queries_'+order]=cost['basis_setup_seconds_including_input_loading_and_probe_storage']['median']/saving if saving>0 else None
dump('CONDITIONAL_COST.json',cost)
tokens=[{'question':q['id'],**{k:v for k,v in read(ROOT/'contexts'/f'{q["id"]}.json').items() if k in ['first_token_id','normal_second_candidate','forced_second_token','forced_second_text','EOS_fallback']}} for q in INPUTS['questions']]
result={'quadratic_primary_support':all(r['pass'] for r in SUPPORT['2']),'quadratic_held_questions_passing':sum(r['pass'] for r in SUPPORT['2']),
    'cubic_secondary_support':all(r['pass'] for r in SUPPORT['3']),'cubic_held_questions_passing':sum(r['pass'] for r in SUPPORT['3']),
    'held_known_questions_at_new_position':4,'new_question_wordings':0,'new_primary_held_endpoints':32,'calibration_targets':sum(r['split']=='calibration' for r in ROWS),
    'by_strength':SUMMARY,'state_fields':len(SCHEMA),'memory_fields':sum(v['group'].startswith('memory') for v in SCHEMA.values()),
    'chain_control':CHAIN,'cubic_calibration_lock':CUBIC,'mixed_summary':mix_summary,'tokens':tokens,'EOS_fallback_cases':sum(t['EOS_fallback'] for t in tokens),
    'sampled_held_anisotropy':stats([r['eigenvalue_ratio'] for r in geometry if r['split']=='held_out']),
    'conditional_cost':cost,'audit':AUDIT,'call_allocation':ALLOC,'model_calls_combined':STATUS['model_calls'],
    'new_model_calls':STATUS['new_calls'],'parent_calls':166,'CPU_only':True,'GPU_allocations':0,'no_new_residual_intervention_at_second_step':True,
    'physical_energy_conservation_claim':False,'semantic_preservation_claim':False,'broad_goal_achieved':False,
    'analysis_seconds':time.monotonic()-START,'limits':PLAN['limits']}
dump('RESULTS.json',result)
lines=['CACHED MEMORY TRANSPORT — PROSPECTIVE SECOND-STEP RESPONSE','',
    'Original steering enters only the preceding token at block11 input. Step two receives a cloned changed memory and the same fixed token; no new residual nudge.',
    'Six known questions, two calibration and four held at this new prediction position; two older frozen directions. These are not new independent question wordings.',
    f"Quadratic primary support: {result['quadratic_primary_support']}; {result['quadratic_held_questions_passing']}/4 held questions pass.",
    f"Cubic prospective secondary support: {result['cubic_secondary_support']}; {result['cubic_held_questions_passing']}/4 held questions pass stricter full-state gates.",
    f"Chain control: {CHAIN['pass']}; mixed-plane secondary: {mix_summary['support']}; forced EOS continuations: {result['EOS_fallback_cases']}/6."]
for strength,summary in SUMMARY.items():
    for order in ['2','3']:
        v=summary['orders'][order]
        if v['cases']:
            ce=v['cost_error'];cost_text=f"{100*ce['median']:.4g}%/{100*ce['maximum']:.4g}%" if ce['n'] else 'unresolved'
            lines.append(f"Strength{strength},order{order}: weighted-logit error median/max {100*v['weighted_logit_error']['median']:.4g}%/{100*v['weighted_logit_error']['maximum']:.4g}%; cost {cost_text}.")
lines += [f"Readouts: {len(SCHEMA)} fields; {result['memory_fields']} memory fields, including previous-token and new-token K/V separately.",
    f"Calls: {STATUS['new_calls']} new, {STATUS['model_calls']} combined with parent166; ceiling300. CPU only.",
    'The second derivative separates transported incoming curvature J*r1 from new decoder nonlinearity H[q1,q1]. Component norms/cosines and numerical checks are retained.',
    'The sampled Fisher matrix describes direction-dependent output sensitivity; it is not the full-space spectrum or physical energy.',
    'Conditional timing assumes first-step memory/probes already exist; it does not prove end-to-end compression or generation speedup.',
    'Earlier/later intervention sites, new wording/direction transfer, and task-matched semantic-direction correction remain unresolved. Original study16 failures are unchanged.']
(OUT/'FINAL_FINDINGS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('R4_MEMORY_ANALYSIS',json.dumps({k:result[k] for k in ['quadratic_primary_support','quadratic_held_questions_passing','cubic_secondary_support','cubic_held_questions_passing','new_model_calls','model_calls_combined','state_fields','EOS_fallback_cases']}),flush=True)
