"""Pre-target supplemental audit of scalar gates and complete output vectors.

No model calls, fitting, selected token subset or changed scientific threshold.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from point_store import PointStore
BEGIN=time.monotonic()
ROOT=Path('/content/research4/artifacts/fresh_memory_boundary_response_v1/colab_cpu_20261014')
RUN=ROOT/'fresh';OUT=ROOT.parent/'fresh_evaluated_20261014'
def read(p):return json.loads(p.read_text())
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def lp(x):
    x=np.asarray(x,dtype=np.float64);m=x.max();return x-(m+np.log(np.exp(x-m).sum()))
def KL(a,b):
    la,lb=lp(a),lp(b);return max(0.,float(np.exp(la)@(la-lb)))
def stats(values):
    x=[float(v) for v in values if v is not None]
    return {'n':len(x),'median':float(np.median(x)),'maximum':max(x)} if x else {'n':0}
rows=read(RUN/'METRICS.json');assert len(rows)==80
store=PointStore(RUN);records=[];maximum_cost_error_discrepancy=0.;maximum_fisher_error_discrepancy=0.;comparators=0
for row in rows:
    pid,t,a=row['question'],row['token'],row['alpha']
    base=store.point(f'contexts/{pid}_t{t}_baseline.npz')['logits'].astype(np.float64)
    actual=store.point(row['actual_file'])['logits'].astype(np.float64);jet=store.load_jet(row['coefficient_file'])
    actual_cost=KL(base,actual);assert abs(actual_cost-row['actual_KL'])<1e-10
    p=np.exp(lp(base));pa=np.exp(lp(actual));truth=actual-base;q=np.asarray(jet['q']['logits'],dtype=np.float64)
    fisher=.5*a*a*float(p@((q-p@q)**2));fe=abs(fisher-actual_cost)/actual_cost if actual_cost>1e-10 else None
    if fe is not None:
        error=abs(fe-row['Fisher_cost_error'])/max(1.,abs(fe));maximum_fisher_error_discrepancy=max(maximum_fisher_error_discrepancy,error);assert error<1e-8
    result={k:row[k] for k in ['question','label','category','token','alpha','strength']};result['orders']={}
    for order in [1,2,3]:
        delta=0.+a*q
        if order>=2:delta=delta+.5*a*a*jet['r']['logits']
        if order>=3:delta=delta+(a**3/6)*jet['t']['logits']
        predicted=base+delta;cost=KL(base,predicted);ce=abs(cost-actual_cost)/actual_cost if actual_cost>1e-10 else None
        old=row['orders'][str(order)]
        assert (ce is None)==(old['cost_error'] is None)
        if ce is not None:
            discrepancy=abs(ce-old['cost_error'])/max(1.,abs(ce));maximum_cost_error_discrepancy=max(maximum_cost_error_discrepancy,discrepancy);assert discrepancy<1e-8
        error=delta-truth;pp=np.exp(lp(predicted))
        bounds=bool(ce is not None and ce<=.20 and old['weighted_logit_error']<=.15
            and all(v['relative_error'] is None or v['relative_error']<=.10 for v in old['groups'].values())
            and all(not v['resolved'] or v['relative_error']<=.15 for v in old['fields'].values())
            and (old['current_KV_only_error'] is None or old['current_KV_only_error']<=.10))
        assert bounds==old['endpoint_bounds_pass']
        result['orders'][str(order)]={'cost_error':ce,'raw_unweighted_full_logit_relative_error':norm(error)/max(norm(truth),1e-30),
            'baseline_probability_weighted_mean_logit_error':float(p@error),'unweighted_mean_logit_error':float(error.mean()),
            'full_probability_total_variation':float(.5*np.abs(pp-pa).sum()),'maximum_single_token_probability_error':float(np.max(np.abs(pp-pa))),
            'weighted_logit_error':old['weighted_logit_error']}
    records.append(result)
    (OUT/'SUPPLEMENTAL_OUTPUT_AUDIT_PROGRESS.json').write_text(json.dumps({'completed_rows':len(records),'native_model_calls':0,'seconds':time.monotonic()-BEGIN})+'\n')
# Recheck every original median comparator, including weak-strength degradation.
support=read(OUT/'QUESTION_SUPPORT.json')
for decision in support:
    panel=[r for r in rows if r['question']==decision['question'] and r['label']==decision['label'] and r['token']==decision['token']]
    assert len(panel)==4;strong=[r for r in panel if r['strength']==.125];weak=[r for r in panel if r['strength']==.03125];order=str(decision['order'])
    def med(which,field,subset):
        values=[r['orders'][which][field] for r in subset];return float(np.median(values)) if all(v is not None for v in values) else None
    current=med(order,'cost_error',strong);first=med('1','cost_error',strong);fisher=[r['Fisher_cost_error'] for r in strong]
    expected={'strong_cost_improves_Fisher':bool(current is not None and all(v is not None for v in fisher) and current<=.75*np.median(fisher)),
        'strong_cost_improves_first':bool(current is not None and first is not None and current<=.75*first),
        'strong_logit_improves_first':med(order,'weighted_logit_error',strong)<=.75*med('1','weighted_logit_error',strong)}
    current=med(order,'cost_error',weak);first=med('1','cost_error',weak)
    expected.update(weak_cost_not_degraded=bool(current is not None and first is not None and current<=first+.005),
                    weak_logit_not_degraded=med(order,'weighted_logit_error',weak)<=med('1','weighted_logit_error',weak)+.005)
    assert all(bool(v)==decision['gates'][k] for k,v in expected.items());comparators+=len(expected)
summary={}
for category in ['primary','alternate_boundary','reference','mixed']:
    summary[category]={}
    for t in [1,2]:
        panel=[r for r in records if r['category']==category and r['token']==t and r['strength']==.125]
        if panel:summary[category][str(t)]={str(order):{k:stats([r['orders'][str(order)][k] for r in panel]) for k in ['raw_unweighted_full_logit_relative_error','full_probability_total_variation','maximum_single_token_probability_error']} for order in [1,2,3]}
result={'classification':'supplemental frozen-before-target audit; original thresholds and results unchanged','native_model_calls':0,
    'all80_endpoint_cost_and_bounds_checks_passed':True,'median_comparator_gates_verified':comparators,
    'maximum_scaled_cost_error_discrepancy':maximum_cost_error_discrepancy,'maximum_scaled_Fisher_error_discrepancy':maximum_fisher_error_discrepancy,
    'all_vocabulary_coordinates_used':True,'formal_output_norm_remains_baseline_probability_weighted_and_centered':True,
    'raw_unweighted_logits_and_full_probability_metrics_are_descriptive':True,'summary':summary,'rows':records,'seconds':time.monotonic()-BEGIN}
(OUT/'SUPPLEMENTAL_OUTPUT_AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('R4_FRESH_SUPPLEMENTAL_OUTPUT_AUDIT',json.dumps({k:v for k,v in result.items() if k not in ['rows','summary']}),flush=True)
