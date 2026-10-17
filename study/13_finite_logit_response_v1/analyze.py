"""Fixed prospective scoring; independent full-vocabulary saved-array audit."""
import csv,hashlib,json,os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

BASE=Path('C:/Users/farha/Research4/artifacts/finite_logit_response_v1')
ROOT=BASE/'native_cpu_20261014';OUT=BASE/'evaluated_20261014';OUT.mkdir(exist_ok=True)
STATUS=json.loads((ROOT/'STATUS.json').read_text())
assert STATUS['status']=='native_complete'
PLAN=json.loads((ROOT/'source/PLAN.json').read_text());INPUTS=json.loads((ROOT/'INPUTS.json').read_text())
SELECT=json.loads((ROOT/'DERIVATIVE_SELECTION_LOCK.json').read_text())
ROWS=json.loads((ROOT/'METRICS.json').read_text());assert len(ROWS)==96
METHODS=['quadratic_Fisher','first_order_logits','second_order_logits']

def dump(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()
manifest=json.loads((ROOT/'EXPORT_HASHES.json').read_text())
assert all(sha(ROOT/n)==h for n,h in manifest.items())
prediction_freeze=json.loads((ROOT/'PREDICTION_FREEZE.json').read_text())
assert prediction_freeze['all_forecasts_saved_before_primary_endpoints']
assert all(sha(ROOT/n)==h for n,h in prediction_freeze['sha256'].items())

def logz(x):
    x=np.asarray(x,dtype=np.float64);m=x.max();return m+np.log(np.exp(x-m).sum())
def prob(x):
    x=np.asarray(x,dtype=np.float64);return np.exp(x-logz(x))
def KL(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return max(0.,float(logz(b)-logz(a)-prob(a)@(b-a)))
def wn(p,x):
    x=np.asarray(x,dtype=np.float64);x=x-p@x;return float(np.sqrt(max(0.,p@(x*x))))
def stats(values):
    a=np.asarray([x for x in values if x is not None],dtype=np.float64)
    return {'n':len(a),'minimum':float(a.min()),'median':float(np.median(a)),'maximum':float(a.max())} if len(a) else {'n':0}
def med(rows,field,method):return float(np.median([r[field][method] for r in rows]))
def ratio(a,b):return float(a/b) if b>1e-15 else None

lookup={(r['question'],r['direction'],r['alpha']):r for r in ROWS}
audit={'main_endpoints_verified':0,'post_endpoints_verified':0,'max_KL_difference':0.,
       'max_forecast_cost_difference':0.,'max_weighted_logit_error_difference':0.,'max_hidden_error_difference':0.}
coefficients={};baselines={}
for question in INPUTS['questions']:
    pid=question['id']
    with np.load(ROOT/'baseline'/f'{pid}.npz',allow_pickle=False) as z:
        base={n:z[n].copy() for n in ['logits','hidden']}
    baselines[pid]=base;p=prob(base['logits'])
    for di in range(2):
        with np.load(ROOT/'coefficients'/f'{pid}_d{di}.npz',allow_pickle=False) as z:
            c={n:z[n].copy() for n in ['q','r','hidden_first','hidden_second','output_first','output_second']}
        coefficients[pid,di]=c
        with np.load(ROOT/'actual'/f'{pid}_d{di}.npz',allow_pickle=False) as z:
            for ai,alpha in enumerate(z['alphas']):
                row=lookup[pid,di,float(alpha)];actual=z['logits'][ai]
                measured=KL(base['logits'],actual)
                audit['max_KL_difference']=max(audit['max_KL_difference'],abs(measured-row['actual_KL']))
                q=c['q'].astype(np.float64);r=c['r'].astype(np.float64)
                fisher=.5*alpha**2*(p@(q*q)-(p@q)**2)
                predictions={'quadratic_Fisher':fisher}
                for name,order in [('first_order_logits',1),('second_order_logits',2)]:
                    delta=alpha*q+(0.5*alpha**2*r if order==2 else 0.)
                    endpoint=base['logits'].astype(np.float64)+delta
                    predictions[name]=KL(base['logits'],endpoint)
                    true=actual.astype(np.float64)-base['logits']
                    error=wn(p,delta-true)/max(wn(p,true),1e-30)
                    audit['max_weighted_logit_error_difference']=max(audit['max_weighted_logit_error_difference'],abs(error-row['weighted_logit_errors'][name]))
                    dh=alpha*c['hidden_first'].astype(np.float64)+(0.5*alpha**2*c['hidden_second'] if order==2 else 0.)
                    th=z['hidden'][ai].astype(np.float64)-base['hidden']
                    if np.linalg.norm(th)>1e-12:
                        err=float(np.linalg.norm(dh-th)/np.linalg.norm(th))
                        audit['max_hidden_error_difference']=max(audit['max_hidden_error_difference'],abs(err-row['hidden_errors'][name]))
                    distribution_error=KL(actual,endpoint)
                    audit['max_KL_difference']=max(audit['max_KL_difference'],abs(distribution_error-row['distribution_KL_actual_to_forecast'][name]))
                for name,value in predictions.items():
                    audit['max_forecast_cost_difference']=max(audit['max_forecast_cost_difference'],abs(value-row['cost_predictions'][name]))
                audit['main_endpoints_verified']+=1
assert audit['max_KL_difference']<1e-10 and audit['max_forecast_cost_difference']<1e-10
assert audit['max_weighted_logit_error_difference']<1e-10 and audit['max_hidden_error_difference']<1e-10

HELD=[r for r in ROWS if r['split']=='held_out'];assert len(HELD)==72
PRIMARY=[r for r in HELD if r['strength'] in PLAN['primary_strengths']];assert len(PRIMARY)==48
support=[]
for question in INPUTS['questions']:
    if question['split']!='held_out':continue
    pid=question['id'];panel=[r for r in PRIMARY if r['question']==pid]
    mid=[r for r in panel if r['strength']==.125];weak=[r for r in panel if r['strength']==.03125]
    assert len(panel)==8 and len(mid)==4 and len(weak)==4
    resolved=all(r['resolved'] and not r['zero_delivery'] for r in panel)
    # Do not silently drop unresolved signed endpoints.
    finite=all(r['cost_relative_errors'][n] is not None for r in panel for n in METHODS)
    if not finite:
        support.append({'question':pid,'family':question['family'],'all_rules_pass':False,'reason':'unresolved primary endpoint','resolved':resolved});continue
    c2=med(mid,'cost_relative_errors','second_order_logits')
    c1=med(mid,'cost_relative_errors','first_order_logits');cf=med(mid,'cost_relative_errors','quadratic_Fisher')
    w2=med(mid,'weighted_logit_errors','second_order_logits');w1=med(mid,'weighted_logit_errors','first_order_logits')
    gates={'all_signed_endpoints_resolved':resolved,
        'cost_each_primary_case_le_20pct':all(r['cost_relative_errors']['second_order_logits']<=.20 for r in panel),
        'mid_median_cost_improves_25pct_vs_first':c2<=.75*c1,
        'mid_median_cost_improves_25pct_vs_Fisher':c2<=.75*cf,
        'weighted_logit_each_case_le_15pct':all(r['weighted_logit_errors']['second_order_logits']<=.15 for r in panel),
        'mid_median_weighted_logit_improves_25pct':w2<=.75*w1,
        'weak_median_cost_not_degraded':med(weak,'cost_relative_errors','second_order_logits')<=med(weak,'cost_relative_errors','first_order_logits')+.005,
        'weak_median_weighted_logit_not_degraded':med(weak,'weighted_logit_errors','second_order_logits')<=med(weak,'weighted_logit_errors','first_order_logits')+.005}
    support.append({'question':pid,'family':question['family'],'primary_cases':8,'mid_cases':4,'gates':gates,
        'all_rules_pass':all(gates.values()),'mid_cost_median':{'Fisher':cf,'first':c1,'second':c2},
        'mid_weighted_logit_median':{'first':w1,'second':w2},
        'mid_cost_error_ratios':{'second_to_first':ratio(c2,c1),'second_to_Fisher':ratio(c2,cf)},
        'primary_second_cost_error_max':max(r['cost_relative_errors']['second_order_logits'] for r in panel),
        'primary_second_weighted_logit_error_max':max(r['weighted_logit_errors']['second_order_logits'] for r in panel)})
dump('QUESTION_SUPPORT.json',support)
by_strength={}
for strength in PLAN['strengths']:
    panel=[r for r in HELD if r['strength']==strength]
    by_strength[str(strength)]={'signed_cases':len(panel),'resolved':sum(r['resolved'] for r in panel),
        'cost_errors':{n:stats([r['cost_relative_errors'][n] for r in panel]) for n in METHODS},
        'weighted_logit_errors':{n:stats([r['weighted_logit_errors'][n] for r in panel]) for n in METHODS[1:]},
        'raw_logit_errors':{n:stats([r['raw_logit_errors'][n] for r in panel]) for n in METHODS[1:]},
        'hidden_errors':{n:stats([r['hidden_errors'][n] for r in panel]) for n in METHODS[1:]},
        'distribution_KL_actual_to_forecast':{n:stats([r['distribution_KL_actual_to_forecast'][n] for r in panel]) for n in METHODS[1:]},
        'total_variation':{n:stats([r['total_variation'][n] for r in panel]) for n in METHODS[1:]}}

post_summary=None
if (ROOT/'POST_METRICS.json').exists():
    post=json.loads((ROOT/'POST_METRICS.json').read_text());assert len(post)==8
    post_lookup={(r['question'],r['alpha']):r for r in post}
    for question in INPUTS['questions']:
        pid=question['id'];path=ROOT/'post'/f'{pid}_actual.npz'
        if not path.exists():continue
        with np.load(ROOT/'post'/f'{pid}_coefficients.npz',allow_pickle=False) as z:q=z['q'].astype(np.float64);r=z['r'].astype(np.float64)
        base=baselines[pid]['logits']
        with np.load(path,allow_pickle=False) as z:
            for ai,alpha in enumerate(z['alphas']):
                row=post_lookup[pid,float(alpha)];measured=KL(base,z['logits'][ai])
                audit['max_KL_difference']=max(audit['max_KL_difference'],abs(measured-row['actual_KL']))
                for name,order in [('first_order_logits',1),('second_order_logits',2)]:
                    endpoint=base.astype(np.float64)+alpha*q+(0.5*alpha**2*r if order==2 else 0.)
                    audit['max_forecast_cost_difference']=max(audit['max_forecast_cost_difference'],abs(KL(base,endpoint)-row['cost_predictions'][name]))
                audit['post_endpoints_verified']+=1
    chain=json.loads((ROOT/'CURVATURE_CHAIN.json').read_text())
    post_summary={'contexts':4,'held_questions':2,'signed_endpoints':8,
        'cost_errors_all':{n:stats([r['cost_relative_errors'][n] for r in post]) for n in METHODS},
        'cost_errors_held':{n:stats([r['cost_relative_errors'][n] for r in post if r['split']=='held_out']) for n in METHODS},
        'chain_q_relative':stats([r['q_transport_relative'] for r in chain]),
        'chain_r_relative':stats([r['second_chain_weighted_relative'] for r in chain]),
        'separate_post_block_input_contract':True,'not_additive_causal_percentages':True}
    dump('POST_BLOCK_SUMMARY.json',post_summary)
assert audit['max_KL_difference']<1e-10 and audit['max_forecast_cost_difference']<1e-10
audit.update(native_manifest_files_verified=len(manifest),no_new_model_calls=True)
dump('INDEPENDENT_AUDIT.json',audit)
result={'status':'complete_prospective_analysis','primary_support':all(s['all_rules_pass'] for s in support),
    'held_families_passing':sum(s['all_rules_pass'] for s in support),'held_unique_questions':6,'calibration_unique_questions':2,
    'directions':2,'primary_held_signed_endpoints':48,'all_held_signed_endpoints':72,
    'method':SELECT['method'],'coarse_validation_step':SELECT['coarse_step'],'fine_validation_step':SELECT['fine_step'],
    'by_strength':by_strength,'post_block_summary':post_summary,'audit':audit,
    'model_calls':STATUS['model_calls'],'GPU_allocations':0,'Colab':False,
    'stress_is_not_primary':True,'old_study12_primary_failure_preserved':True,'limits':PLAN['limits']}
dump('RESULTS.json',result)
with (OUT/'ALL_METRICS.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.writer(f);writer.writerow(['question','split','family','direction','alpha','actual_KL']+
        [n+'_cost_relative_error' for n in METHODS]+[n+'_weighted_logit_error' for n in METHODS[1:]])
    for row in ROWS:writer.writerow([row[n] for n in ['question','split','family','direction','alpha','actual_KL']]+
        [row['cost_relative_errors'][n] for n in METHODS]+[row['weighted_logit_errors'][n] for n in METHODS[1:]])

# Secondary signed-cubic diagnostic from the already frozen normal-state q,r.
# Its endpoint is distinct from the predeclared individual-signed primary test.
cubic=[]
for question in INPUTS['questions']:
    if question['split']!='held_out':continue
    pid=question['id'];p=prob(baselines[pid]['logits'])
    for di in range(2):
        c=coefficients[pid,di];q=c['q'].astype(np.float64);r=c['r'].astype(np.float64)
        qc=q-p@q;rc=r-p@r
        network=.5*float(p@(qc*rc));softmax=float(p@(qc**3))/6
        for strength in PLAN['strengths']:
            cp=lookup[pid,di,strength]['actual_KL'];cm=lookup[pid,di,-strength]['actual_KL']
            odd=(cp-cm)/2;prediction=strength**3*(network+softmax)
            cubic.append({'question':pid,'direction':di,'strength':strength,'actual_signed_odd_cost':odd,
                'predicted_leading_odd_cost':prediction,'network_coefficient':network,'softmax_coefficient':softmax,
                'absolute_error':abs(prediction-odd),'relative_to_even_cost_error':abs(prediction-odd)/max((cp+cm)/2,1e-30)})
dump('SECONDARY_SIGNED_CUBIC.json',{'primary_endpoint_unchanged':True,'model_calls':0,
    'formula':'alpha^3 [Cov_p(q,r)/2 + E_p[(q-E_p q)^3]/6]',
    'classification':'secondary mathematical diagnostic, not a newly selected primary forecast or causal energy shares',
    'rows':cubic,'error_relative_to_even_cost':{str(s):stats([r['relative_to_even_cost_error'] for r in cubic if r['strength']==s]) for s in PLAN['strengths']}})

colors={'quadratic_Fisher':'#b25b76','first_order_logits':'#cb913b','second_order_logits':'#277f91'}
labels={'quadratic_Fisher':'Quadratic Fisher cost','first_order_logits':'First-order logits + softmax','second_order_logits':'Second-order logits + softmax'}
fig,axes=plt.subplots(2,2,figsize=(13.5,9.5),layout='constrained')
for ax,field,title,names in [(axes[0,0],'cost_errors','Signed output-cost prediction',METHODS),
        (axes[0,1],'weighted_logit_errors','Full-vocabulary weighted logit response',METHODS[1:])]:
    for name in names:
        panels=[by_strength[str(s)][field][name] for s in PLAN['strengths']]
        median=[max(1e-5,100*p['median']) for p in panels]
        low=[max(1e-5,100*p['minimum']) for p in panels];high=[max(1e-5,100*p['maximum']) for p in panels]
        ax.plot(PLAN['strengths'],median,'o-',color=colors[name],label=labels[name])
        ax.fill_between(PLAN['strengths'],low,high,color=colors[name],alpha=.1)
    ax.set(xscale='log',yscale='log',xlabel='Steering strength',ylabel='Relative error (%)',title=title)
    ax.set_xticks(PLAN['strengths'],[f'{s:g}' for s in PLAN['strengths']]);ax.xaxis.set_minor_formatter(NullFormatter())
    ax.axhline(20 if field=='cost_errors' else 15,color='#777777',ls='--',lw=.8)
    ax.grid(alpha=.15,which='both');ax.legend(fontsize=8)
ax=axes[1,0];xs=np.arange(6);width=.24
for j,name in enumerate(METHODS):
    values=[]
    for question in [q for q in INPUTS['questions'] if q['split']=='held_out']:
        panel=[r for r in HELD if r['question']==question['id'] and r['strength']==.125]
        values.append(100*med(panel,'cost_relative_errors',name))
    ax.bar(xs+(j-1)*width,values,width,color=colors[name],label=labels[name])
ax.set_xticks(xs,['Grammar','Logic','Code','Polite','Analogy','Fact'],fontsize=9)
ax.set(ylabel='Median cost error (%)',title='Each held-out question family at strength 0.125')
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(fontsize=8)
ax=axes[1,1]
for strength,color in zip(PLAN['strengths'],['#3d89ad','#d59043','#b65274']):
    panel=[r for r in HELD if r['strength']==strength and r['resolved']]
    ax.scatter([r['actual_KL'] for r in panel],[r['cost_predictions']['second_order_logits'] for r in panel],color=color,s=28,alpha=.8,label=f'strength {strength:g}')
v=[r['actual_KL'] for r in HELD]+[r['cost_predictions']['second_order_logits'] for r in HELD]
line=np.geomspace(min(x for x in v if x>0)*.7,max(v)*1.4,100)
ax.plot(line,line,color='#333333',lw=1);ax.set(xscale='log',yscale='log',xlabel='Actual KL (nats)',ylabel='Second-order forecast KL (nats)',title='Signed endpoint predictions')
ax.grid(alpha=.15,which='both');ax.legend(fontsize=8)
fig.suptitle('Finite logit-response forecast test — Qwen3.5-0.8B / local CPU\nSix fresh held-out questions, two directions, both signs. Shading shows case ranges, not uncertainty.',fontsize=13)
fig.savefig(OUT/'finite_logit_forecasts.png',dpi=170);plt.close(fig)

receipt=json.loads((BASE/'NATIVE_EXPORT_RECEIPT_20261014.json').read_text())
lines=['FINITE SIGNED OUTPUT RESPONSE —14 October2026','',
    f'Prospective primary support: {result["primary_support"]}; held families passing {result["held_families_passing"]}/6.',
    'Eight fresh questions:2 numerical calibration,6 independently held-out question',
    'families. Two fresh directions, both signs, full248320-token output vocabulary.',
    'Primary strengths0.03125/0.125; strength0.5 is a separately declared stress test.',
    'Intervention: last prompt-token input before block11 attention RMSNorm.',
    'All original attention, MLP, normalization and residual operations are retained.',
    'Actual full-prefix FP32 computation, native masks/RoPE/reference recurrent kernels,',
    'use_cache=False. No decoding/cache substitute, model training or classifier.','',
    'Forecasts saved before every target endpoint',
    'l_hat1(alpha)=l0+alpha*q; l_hat2(alpha)=l0+alpha*q+alpha^2*r/2.',
    'Both logit forecasts use exact softmax. Compare KL(p_base||p_forecast) with',
    'actual KL(p_base||p_changed), retaining each sign individually.',
    f'Derivative method: {SELECT["method"]}.',
    f'Calibration-locked validation stencil: {SELECT["coarse_step"]}/{SELECT["fine_step"]}.',
    'Derivative probes were declared and separate from all target strengths.',
    'If exact automatic derivatives are selected, finite differences validate them',
    'and do not supply the second-derivative forecast. Native FP32 arithmetic still',
    'limits numerical representability; these are derivatives of its smooth graph.','',
    'Held-out results: median / maximum over24 repeated signed cases per strength.']
for strength,panel in by_strength.items():
    lines.append(f'Strength {strength}:')
    for name in METHODS:
        s=panel['cost_errors'][name];lines.append(f'  {name} scalar cost error: {100*s["median"]:.4f}% / {100*s["maximum"]:.4f}%.')
    for name in METHODS[1:]:
        w=panel['weighted_logit_errors'][name];h=panel['hidden_errors'][name];raw=panel['raw_logit_errors'][name]
        lines.append(f'  {name} weighted logit vector: {100*w["median"]:.4f}% / {100*w["maximum"]:.4f}%; raw logit vector: {100*raw["median"]:.4f}% / {100*raw["maximum"]:.4f}%; final hidden vector: {100*h["median"]:.4f}% / {100*h["maximum"]:.4f}%.')
lines+=['','Per-family fixed-rule outcomes:']
for s in support:lines.append(f'{s["family"]}: {s["all_rules_pass"]}; '+json.dumps(s.get('gates',{})))
if post_summary:
    lines+=['','Separate post-block boundary check',
        'Four fixed contexts(two calibration/two held), one mapped direction and',
        'both signs at0.125. These are separate post-block interventions, not an',
        'actual nonlinear tail evaluated on predicted activations counted as a forecast.',
        'The second derivative decomposes as r_full=r_tail(A*v)+J_tail(H_block[v,v]).',
        'This is a derivative chain-rule check, not energy conservation or additive',
        'scalar causal shares.',json.dumps(post_summary,indent=2)]
lines+=['','Independent raw-array numerical verification',json.dumps(audit,indent=2),'',
    'Preservation and resources',
    'Study12 failed primary remains unchanged. Its reused eight-case26%-to7%',
    'hand-off improvement remains exploratory motivation, not prospective evidence.',
    f'Whole-backbone forwards(all probes/attempts/main/post endpoints): {STATUS["model_calls"]}/300.',
    'Exact nested differentiation additionally performs backward AD passes; forward',
    'counts alone are not FLOP or wall-time comparisons.',
    f'Native imports/model evaluation/export: {receipt["native_and_export_seconds"]:.2f}sec, within2700sec including export reserve.',
    'CPU threads4; GPU allocations0; Colab0; no new agents, training or purchases.',
    'Prior native archives unchanged, old consistency monitor remains stopped,',
    'and no deferred three-block expansion was started.','',
    'Limits',*PLAN['limits'],'',
    'Remaining unexecuted next step: test the same forecast at one cached decoding',
    'position with actual cloned KV/recurrent state and fresh prompts, before',
    'claiming transfer to generation trajectories or a broader model-wide law.']
(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
dump('ARTIFACT_HASHES.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='ARTIFACT_HASHES.json'})
print(json.dumps({'primary_support':result['primary_support'],'families_passing':result['held_families_passing'],
    'method':result['method'],'by_strength':by_strength,'family_support':support,'post':post_summary,'audit':audit},indent=2))
