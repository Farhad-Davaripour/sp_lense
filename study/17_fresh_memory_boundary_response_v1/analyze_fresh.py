"""Independent full-array audit, frozen gates, residual bookkeeping and cost."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import hashlib,json,time
from pathlib import Path
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt=None
from point_store import PointStore
ROOT=Path(__file__).resolve().parent.parent;RUN=ROOT/'fresh';BASE=ROOT.parent;OUT=BASE/'fresh_evaluated_20261014'
START=time.monotonic();OUT.mkdir(exist_ok=False)
GROUPS=['hidden','residual','mixer','mlp','memory_kv','memory_conv','memory_recurrent']
def read(p):return json.loads(p.read_text())
def scalar(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False,default=scalar)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def prob(x):
    x=np.asarray(x,dtype=np.float64);m=x.max();lp=x-(m+np.log(np.exp(x-m).sum()));return np.exp(lp)
def lse(x):
    x=np.asarray(x,dtype=np.float64);m=x.max();return float(m+np.log(np.exp(x-m).sum()))
def KL(a,b):return max(0.,float(lse(b)-lse(a)-prob(a)@(np.asarray(b,dtype=np.float64)-a)))
def wn(p,x):
    x=np.asarray(x,dtype=np.float64);return float(np.sqrt(max(0.,p@((x-p@x)**2))))
def stats(x):
    v=[float(a) for a in x if a is not None]
    return {'n':len(v),'median':float(np.median(v)),'minimum':min(v),'maximum':max(v)} if v else {'n':0}
def digest(delta):
    h=hashlib.sha256()
    for n in sorted(delta):h.update(n.encode()+b'\0');h.update(np.ascontiguousarray(delta[n]).tobytes())
    return h.hexdigest()
def flat(point):return {**point['values'],'logits':point['logits']}
def delta(j,a,order):
    out={}
    for n in j['q']:
        # Match the frozen recipe's initial sum accumulator, including -0.0.
        x=0.+np.asarray(j['q'][n],dtype=np.float64)*a
        if order>=2:x=x+(a*a/2)*j['r'][n]
        if order>=3:x=x+(a**3/6)*j['t'][n]
        out[n]=x
    return out
STATUS=read(RUN/'STATUS.json');assert STATUS['status']=='native_complete' and STATUS['model_calls']==300
MANIFEST=read(RUN/'EXPORT_HASHES.json');assert all(sha(ROOT/n)==h for n,h in MANIFEST.items())
FROZEN=read(RUN/'PREDICTION_FREEZE.json');assert FROZEN['target_rows_seen']==0 and FROZEN['calls_at_freeze']==216 and FROZEN['all_predictions_precede_targets']
assert all(sha(RUN/n)==h for n,h in FROZEN['sha256'].items())
INPUTS=read(RUN/'INPUTS.json');questions=INPUTS['questions'];SCHEMA={t:read(RUN/f'STATE_SCHEMA_T{t}.json') for t in [1,2]}
assert len(SCHEMA[1])==169 and len(SCHEMA[2])==181
ROWS=read(RUN/'METRICS.json');PREDICTIONS=read(RUN/'PREDICTIONS.json');assert len(ROWS)==len(PREDICTIONS)==80
assert {k:sum(r['category']==k for r in ROWS) for k in ['primary','alternate_boundary','reference','mixed']}=={'primary':32,'alternate_boundary':24,'reference':8,'mixed':16}
assert len({r['question'] for r in ROWS if r['category']=='primary'})==4
CHAIN=read(RUN/'CHAIN_CONTROLS.json');assert len(CHAIN)==2
NULL=read(RUN/'NULL_CONTROLS.json');assert len(NULL)==2 and all(x['exact_T1'] and x['exact_T2'] and x['whole_cache_exact'] for x in NULL)
CARRY=read(RUN/'MEMORY_CARRY_CHECKS.json');assert CARRY and all(x['previous_KV_fields_exact']==12 and x['b11_input_tangent_zero'] for x in CARRY)
CAUSAL=read(RUN/'CAUSAL_DELIVERY_CHECKS.json');assert CAUSAL and all(x['pass'] for x in CAUSAL)
LOCK=read(RUN/'CUBIC_CALIBRATION_LOCKS.json');store=PointStore(RUN)
AUDIT={'endpoint_rows':0,'field_order_cases':0,'full_delta_digests_verified':0,'full_probability_digests_verified':0,
       'manifest_files_verified':len(MANIFEST),'max_KL_difference':0.,'max_scaled_state_error_difference':0.,
       'max_weighted_error_difference':0.,'exact_actual_previous_KV_fields':0,'new_model_calls':0}
BALANCES=[]
previous_actual=None;previous_row=None
for row,frozen in zip(ROWS,PREDICTIONS):
    pid,t,a=row['question'],row['token'],row['alpha'];schema=SCHEMA[t]
    assert all(row[k]==frozen[k] for k in ['question','tag','token','alpha','category'])
    base=flat(store.point(f'contexts/{pid}_t{t}_baseline.npz'));actual=flat(store.point(row['actual_file']));j=store.load_jet(row['coefficient_file'])
    p=prob(base['logits']);actual_cost=KL(base['logits'],actual['logits']);truth={n:actual[n].astype(np.float64)-base[n] for n in base}
    AUDIT['max_KL_difference']=max(AUDIT['max_KL_difference'],abs(actual_cost-row['actual_KL']))
    if t==2:
        assert previous_row is not None and all(row[k]==previous_row[k] for k in ['question','tag','alpha'])
        for n in schema:
            if n.endswith('_previous1'):
                assert np.array_equal(actual[n],previous_actual[n.replace('_previous1','_current')]);AUDIT['exact_actual_previous_KV_fields']+=1
    else:previous_actual=actual;previous_row=row
    frontier=row['site'][1]+(row['site'][0]=='middle')
    upstream=[n for n,s in schema.items() if n.startswith('m') and s['layer']<frontier]
    assert all(np.array_equal(actual[n],base[n]) for n in upstream)
    for order in [1,2,3]:
        d=delta(j,a,order);record=row['orders'][str(order)];pred=base['logits'].astype(np.float64)+d['logits'];pc=KL(base['logits'],pred)
        assert digest(d)==frozen['orders'][str(order)]['full_delta_sha256'];AUDIT['full_delta_digests_verified']+=1
        pp=prob(pred);assert hashlib.sha256(pp.tobytes()).hexdigest()==frozen['orders'][str(order)]['probability_sha256'];AUDIT['full_probability_digests_verified']+=1
        AUDIT['max_KL_difference']=max(AUDIT['max_KL_difference'],abs(pc-record['predicted_KL']),abs(pc-frozen['orders'][str(order)]['cost']),
                                         abs(KL(actual['logits'],pred)-record['actual_to_prediction_KL']))
        weighted=wn(p,d['logits']-truth['logits'])/max(wn(p,truth['logits']),1e-30)
        AUDIT['max_weighted_error_difference']=max(AUDIT['max_weighted_error_difference'],abs(weighted-record['weighted_logit_error']))
        sums={g:[0.,0.] for g in GROUPS};curr=[0.,0.]
        for n,s in schema.items():
            tn=norm(truth[n]);en=norm(d[n]-truth[n]);f=record['fields'][n]
            resolved=tn>max(1e-10,20*np.finfo(np.float32).eps*norm(base[n]))
            assert bool(resolved)==f['resolved'] and bool(np.array_equal(actual[n],base[n]))==f['exactly_unchanged']
            discrepancy=max(abs(tn-f['change_norm'])/max(1.,tn),abs(en-f['error_norm'])/max(1.,en))
            if resolved:discrepancy=max(discrepancy,abs(en/tn-f['relative_error'])/max(1.,abs(f['relative_error'])))
            AUDIT['max_scaled_state_error_difference']=max(AUDIT['max_scaled_state_error_difference'],discrepancy)
            if s['group'] in sums:sums[s['group']][0]+=tn*tn;sums[s['group']][1]+=en*en
            if n.startswith('m') and n.endswith('_current'):curr[0]+=tn*tn;curr[1]+=en*en
            AUDIT['field_order_cases']+=1
        for g,(tn,en) in sums.items():
            if tn>1e-24:
                relative=float(np.sqrt(en/tn));r=record['groups'][g]['relative_error']
                AUDIT['max_scaled_state_error_difference']=max(AUDIT['max_scaled_state_error_difference'],abs(relative-r)/max(1.,abs(r)))
        if curr[0]>1e-24:
            err=float(np.sqrt(curr[1]/curr[0]));r=record['current_KV_only_error']
            assert abs(err-r)<1e-8*max(1.,abs(r))
    if row['category']=='mixed':
        dj=store.load_jet(row['diagonal_coefficient_file'])
        for order in [2,3]:
            d=delta(dj,a,order);ref=row['diagonal'][str(order)];pred=base['logits'].astype(np.float64)+d['logits']
            assert digest(d)==frozen['diagonal'][str(order)]['full_delta_sha256'];AUDIT['full_delta_digests_verified']+=1
            weighted=wn(p,d['logits']-truth['logits'])/max(wn(p,truth['logits']),1e-30)
            assert abs(weighted-ref['weighted_logit_error'])<1e-8
            AUDIT['max_KL_difference']=max(AUDIT['max_KL_difference'],abs(KL(base['logits'],pred)-ref['predicted_KL']))
    # An exact algebraic residual balance, not a learned conservation law.
    jdelta=delta(j,a,3);block_rows=[]
    for i in range(24):
        di=truth[f'b{i}_input'];dm=truth[f'b{i}_mixer'];dp=truth[f'b{i}_mlp'];dmid=truth[f'b{i}_middle'];do=truth[f'b{i}_output']
        external=np.asarray(row['direction'],dtype=np.float64)*a if t==1 and row['site']==['middle',i] else np.zeros_like(di)
        u=dm+dp+external;balance_error=norm(do-di-u)
        floor=100*np.finfo(np.float32).eps*max(1.,norm(base[f'b{i}_input']),norm(base[f'b{i}_output']))
        assert balance_error<=floor and norm(dmid-di-dm-external)<=floor
        input_energy=.5*norm(di)**2;output_energy=.5*norm(do)**2
        aligned_work=float(di@u);branch_term=.5*norm(u)**2;energy_error=abs(output_energy-input_energy-aligned_work-branch_term)
        predicted_energy=.5*norm(jdelta[f'b{i}_output'])**2
        block_rows.append({'layer':i,'input_perturbation_energy':input_energy,'output_perturbation_energy':output_energy,
            'alignment_term':aligned_work,'branch_squared_term':branch_term,'balance_vector_error':balance_error,'roundoff_bound':floor,
            'algebraic_energy_balance_error':energy_error,'cubic_predicted_output_perturbation_energy':predicted_energy,
            'cubic_energy_relative_error':abs(predicted_energy-output_energy)/output_energy if output_energy>max(1e-12,floor**2) else None})
    BALANCES.append({'question':pid,'token':t,'category':row['category'],'alpha':a,'site':row['site'],'blocks':block_rows})
    AUDIT['endpoint_rows']+=1
assert AUDIT['max_KL_difference']<1e-10 and AUDIT['max_scaled_state_error_difference']<1e-8 and AUDIT['max_weighted_error_difference']<1e-8
dump('INDEPENDENT_AUDIT.json',AUDIT)
dump('RESIDUAL_BALANCE.json',{'definition':'E=||steered residual-normal residual||^2/2',
    'identity':'E_out-E_in = delta_input dot delta_branch + ||delta_branch||^2/2; include external middle nudge in delta_branch',
    'interpretation':'Exact algebraic bookkeeping, not a physical conservation law or separate predictive success. The frozen vector equation predicts the terms; their alignment can amplify or damp changes.',
    'rows':BALANCES})

def gates(panel,order,label,token):
    strong=[r for r in panel if r['strength']==.125];weak=[r for r in panel if r['strength']==.03125];key=str(order)
    checks={'complete_signed_panel':len(panel)==4,'resolved_output':all(r['resolved'] for r in panel),
        'nonzero_delivery':all(r['delivered_perturbation_norm'] is not None and r['delivered_perturbation_norm']>1e-12 for r in panel),
        'derivatives_numerically_validated':all(r['derivative_validated'] for r in panel),'strict_null_controls':all(n['exact_T1'] and n['exact_T2'] for n in NULL),
        'historical_storage_repair':read(ROOT/'FRESH_PRIOR_LEDGER.json')['stage_one_full_control_pass']}
    if token==2:checks['fresh_chain_controls']=all(c['pass'] for c in CHAIN)
    if order==3:
        checks['calibration_cubic_lock']=LOCK[f'{label}_t{token}'];checks['held_third_validated']=all(r['third_validated'] for r in panel)
    o=[r['orders'][key] for r in panel]
    checks['signed_cost_le20pct']=all(r['cost_error'] is not None and r['cost_error']<=.20 for r in o)
    checks['signed_weighted_logit_le15pct']=all(r['weighted_logit_error']<=.15 for r in o)
    checks['current_KV_only_le10pct']=all(r['current_KV_only_error'] is not None and r['current_KV_only_error']<=.10 for r in o)
    for g in GROUPS:checks[g+'_le10pct']=all(r['groups'][g]['relative_error'] is not None and r['groups'][g]['relative_error']<=.10 for r in o)
    fields=[f for r in o for f in r['fields'].values() if f['resolved']];checks['all_resolved_fields_le15pct']=bool(fields) and all(f['relative_error']<=.15 for f in fields)
    def med(rows,metric,which=key):
        values=[r['orders'][which][metric] for r in rows]
        return float(np.median(values)) if values and all(v is not None for v in values) else None
    cost,first=med(strong,'cost_error'),med(strong,'cost_error','1')
    fisher=[r['Fisher_cost_error'] for r in strong]
    checks['strong_cost_improves_Fisher']=bool(cost is not None and all(x is not None for x in fisher) and cost<=.75*np.median(fisher))
    checks['strong_cost_improves_first']=bool(cost is not None and first is not None and cost<=.75*first)
    checks['strong_logit_improves_first']=med(strong,'weighted_logit_error')<=.75*med(strong,'weighted_logit_error','1')
    cost,first=med(weak,'cost_error'),med(weak,'cost_error','1')
    checks['weak_cost_not_degraded']=bool(cost is not None and first is not None and cost<=first+.005)
    checks['weak_logit_not_degraded']=med(weak,'weighted_logit_error')<=med(weak,'weighted_logit_error','1')+.005
    return {k:bool(v) for k,v in checks.items()}

SUPPORT=[]
for label in ['primary','early','middle','late','reference']:
    for t in ([1,2] if label in ['primary','reference'] else [1]):
        pids=sorted({r['question'] for r in ROWS if r['label']==label and r['token']==t})
        for pid in pids:
            panel=[r for r in ROWS if r['label']==label and r['token']==t and r['question']==pid]
            for order in [2,3]:
                g=gates(panel,order,label,t);SUPPORT.append({'question':pid,'label':label,'token':t,'order':order,'pass':all(g.values()),'gates':g})
dump('QUESTION_SUPPORT.json',SUPPORT)

MIXED=[]
for t in [1,2]:
    panel=[r for r in ROWS if r['category']=='mixed' and r['token']==t];assert len(panel)==8
    for order in [2,3]:
        full=[r['orders'][str(order)]['weighted_logit_error'] for r in panel];diag=[r['diagonal'][str(order)]['weighted_logit_error'] for r in panel]
        gain=float(np.median(full)/max(np.median(diag),1e-30))
        checks={'eight_signed_endpoints':len(panel)==8,'derivative_validation':all(r['derivative_validated'] for r in panel),
            'all_endpoint_bounds':all(r['orders'][str(order)]['endpoint_bounds_pass'] for r in panel),
            'nonzero_delivery_and_resolved':all(r['resolved'] and r['delivered_perturbation_norm']>1e-12 for r in panel),
            'median_logit_improves_diagonal':gain<=.75}
        if t==2:checks['fresh_chain_controls']=all(c['pass'] for c in CHAIN)
        if order==3:
            checks['held_mixed_third_validation']=all(r['third_validated'] for r in panel)
            checks['calibration_mixed_third_lock']=all(read(RUN/'coefficients'/f'r17_cal_spoon_mix_{label}_t{t}.json')['third_validated'] for label in ['sum','difference','weighted01','weighted10'])
        MIXED.append({'token':t,'order':order,'pass':all(checks.values()),'gates':checks,'held_questions':1,
            'full_weighted_error':stats(full),'diagonal_weighted_error':stats(diag),'full_to_diagonal_median_error_ratio':gain,
            'directions_not_differentiated':['weighted01','weighted10']})
dump('MIXED_SUPPORT.json',MIXED)

SUMMARY={}
for label in ['primary','early','middle','late','reference','mixed']:
    SUMMARY[label]={}
    for t in sorted({r['token'] for r in ROWS if r['label']==label or r['category']==label}):
        panel=[r for r in ROWS if (r['label']==label or r['category']==label) and r['token']==t]
        SUMMARY[label][str(t)]={}
        for strength in sorted({r['strength'] for r in panel}):
            subset=[r for r in panel if r['strength']==strength]
            SUMMARY[label][str(t)][str(strength)]={str(order):{'weighted_logit_error':stats([r['orders'][str(order)]['weighted_logit_error'] for r in subset]),
                'cost_error':stats([r['orders'][str(order)]['cost_error'] for r in subset]),
                'state_error':{g:stats([r['orders'][str(order)]['groups'][g]['relative_error'] for r in subset]) for g in GROUPS},
                'current_KV_only_error':stats([r['orders'][str(order)]['current_KV_only_error'] for r in subset]),
                'resolved_field_error':stats([f['relative_error'] for r in subset for f in r['orders'][str(order)]['fields'].values() if f['resolved']])} for order in [1,2,3]}
dump('ERROR_SUMMARY.json',SUMMARY)
contexts=[read(RUN/'contexts'/f'{q["id"]}.json') for q in questions]
primary_support={f't{t}_order{order}':{'passed':sum(r['pass'] for r in SUPPORT if r['label']=='primary' and r['token']==t and r['order']==order),
    'total':4,'support':all(r['pass'] for r in SUPPORT if r['label']=='primary' and r['token']==t and r['order']==order)} for t in [1,2] for order in [2,3]}
cost={'native_value_seconds':{str(t):stats([r['seconds'] for r in read(RUN/'CALL_TIMINGS.json') if r['label']==f'held_actual_T{t}']) for t in [1,2]},
      'conditional_forecast_seconds':{f't{t}_order{o}':stats([r['seconds_full_state_logits_softmax'] for r in read(RUN/'FORECAST_TIMINGS.json') if r['token']==t and r['order']==o]) for t in [1,2] for o in [1,2,3]},
      'context_setup_seconds':stats([r['seconds'] for r in read(RUN/'CONTEXT_SETUP_TIMINGS.json')]),
      'paired_derivative_acquisition_and_storage_seconds':stats([r['seconds_including_both_steps_and_storage'] for r in read(RUN/'JET_SETUP_TIMINGS.json')]),
      'coefficient_and_numerical_validation_seconds_total':sum(r['seconds'] for r in read(RUN/'COEFFICIENT_TIMINGS.json')),
      'historical_repair_setup_seconds_included_in_phase':read(ROOT/'FRESH_PRIOR_LEDGER.json')['phase17_prior_charged_seconds'],
      'no_unconditional_generation_or_compression_speedup_claim':True}
dump('COST.json',cost)
result={'status':'complete_and_audited','primary_support':primary_support,'mixed_support':MIXED,'unique_fresh_questions':6,'unique_fresh_held_questions':4,
        'state_fields':{'T1':169,'T2':181},'current_memory_fields':48,'previous_KV_T2_fields':12,'main_signed_output_rows':32,'all_signed_output_rows':80,
        'phase_calls':300,'fresh_calls':292,'historical_repair_calls':8,'listed_cumulative_calls':1438,'GPU_allocations':0,
        'T1_EOS_fallbacks':sum(c['token1_fallback'] for c in contexts),'T2_EOS_fallbacks':sum(c['token2_fallback'] for c in contexts),
        'actual_T1_argmax_branch_changes':sum(r['argmax_branch_changed'] for r in ROWS if r['token']==1),
        'fresh_chain_controls':CHAIN,'audit':AUDIT,'cost':cost,'source_scope':'Fixed-token inference on one frozen model and precision, one primary numeric action-gradient axis plus a sampled reference plane; no semantic selectivity, unknown-steering detector, full-space spectrum, physical equilibrium or exact universal model claim.',
        'method_transfer_scope':'Same equation and calibration-selected numerical rules across questions; each question has its own locally measured coefficients. This is not transfer of one coefficient set to unseen questions.',
        'original_prior_failures_unchanged':True,'fresh_analysis_seconds':time.monotonic()-START}
dump('RESULTS.json',result)
lines=['FRESH TWO-TOKEN RESPONSE AND BOUNDARY TEST','',f'Phase17 completed300 model runs:8 historical repair controls plus292 fresh runs. CPU only; no GPU.',
       'Six fresh questions: two calibration and four held-out questions. The model and its weights are unchanged.',
       'The same equation is used across questions, with local coefficients measured separately for each question before finite held nudges. This tests transfer of the method, not one universal set of coefficients.']
for t in [1,2]:
    for order in [2,3]:
        r=primary_support[f't{t}_order{order}'];e=SUMMARY['primary'][str(t)]['0.125'][str(order)]['weighted_logit_error']
        lines.append(f'Token{t},order{order}: {r["passed"]}/4 held questions passed all formal gates. Moderate-strength full-output error median/max {100*e["median"]:.3g}%/{100*e["maximum"]:.3g}%.')
lines+=['',f'Forced EOS replacement: token1 {result["T1_EOS_fallbacks"]}/6, token2 {result["T2_EOS_fallbacks"]}/6. Actual nudges changed the greedy next-token choice in{result["actual_T1_argmax_branch_changes"]} first-step targets. Following-token tests still force the preselected normal token.',
        'Earlier/middle/later boundaries and complete quadratic/cubic direction interactions have separate formal results. One held question supports the sampled plane; no full-space claim.',
        'The residual norm balance is exact algebraic bookkeeping. Its alignment terms help explain amplification/damping, but are not an independently discovered physical conservation law.',
        'Prediction cost is conditional on measured setup and the fixed context. No end-to-end generation or model compression speedup is established.',
        'Original failed studies remain failed. Read QUESTION_SUPPORT.json for every failed gate; no held question or site was omitted.']
(OUT/'SIMPLE_FINDINGS.txt').write_text('\n'.join(lines)+'\n')
if plt is not None:
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for t,ax in zip([1,2],axes):
        med=[];hi=[]
        for order in [1,2,3]:
            e=SUMMARY['primary'][str(t)]['0.125'][str(order)]['weighted_logit_error'];med.append(100*e['median']);hi.append(100*e['maximum'])
        ax.bar([1,2,3],med,color=['#9ba9b8','#3779ae','#d68b35']);ax.scatter([1,2,3],hi,color='#333333',s=24,label='Maximum')
        ax.set_xticks([1,2,3],['Linear','Quadratic','Cubic']);ax.set_ylabel('Full-output change error (%)');ax.set_title(f'Token {t}');ax.spines[['top','right']].set_visible(False)
    axes[1].legend(frameon=False);fig.suptitle('Frozen forecasts on four fresh held questions',fontsize=14)
    fig.text(.5,.01,'Moderate strength0.125, both signs. Formal support additionally requires every numerical/state/control gate.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.93]);fig.savefig(OUT/'FRESH_OUTPUT_ERRORS.png',dpi=170);plt.close(fig)
dump('PLOT_STATUS.json',{'rendered':plt is not None,'host_render_from_error_summary_supported':True})
print('R4_FRESH_ANALYSIS',json.dumps({'primary_support':primary_support,'mixed_support':MIXED,'audit':AUDIT,'T1_EOS_fallbacks':result['T1_EOS_fallbacks'],'T2_EOS_fallbacks':result['T2_EOS_fallbacks'],'analysis_seconds':time.monotonic()-START}),flush=True)
