"""One bounded higher-order remedy using existing derivative probes only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('C:/Users/farha/Research4/artifacts/finite_logit_response_v1/native_cpu_20261014')
OUT=Path('C:/Users/farha/Research4/artifacts/response_geometry_v1/cubic_diagnostic_20261014')
OUT.mkdir(exist_ok=False)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {n:z[n].astype(np.float64) for n in z.files}
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prob(x):
    y=x-x.max();p=np.exp(y);return p/p.sum()
def lse(x):
    m=x.max();return m+np.log(np.exp(x-m).sum())
def KL(a,b):return max(0.,float(lse(b)-lse(a)-prob(a)@(b-a)))
def wn(p,x):return float(np.sqrt(p@((x-p@x)**2)))
def norm(x):return float(np.linalg.norm(x))
def stats(x):
    x=[v for v in x if v is not None]
    return {'n':len(x),'median':float(np.median(x)),'maximum':float(max(x)),'minimum':float(min(x))} if x else {'n':0}
INPUTS=json.loads((ROOT/'INPUTS.json').read_text());MANIFEST=json.loads((ROOT/'EXPORT_HASHES.json').read_text());verified={}
def checked(p):
    h=sha(p);assert h==MANIFEST[p.relative_to(ROOT).as_posix()];verified[p.relative_to(ROOT).as_posix()]=h
    return load(p)
plan={'model_calls':0,'classification':'Secondary diagnostic on study13 data whose outcomes were already known; not a new confirmation.',
 'remedy':'Add a cubic term, with third derivative estimated by symmetric second differences of saved forward-AD slopes.',
 'equation':'y(alpha)=y0+alpha*q+alpha^2*r/2+alpha^3*t/6; t≈[q(h)+q(-h)-2q(0)]/h^2',
 'fixed_steps':[.015625,.0078125],'calibration_gate':'Both original calibration questions/directions: coarse/fine third derivative <=10% relative in weighted logits and middle/output/hidden vector norms. Use fine step only if all pass.',
 'no_step_or_method_search':True,'held_validation_failure':'Retain diagnostic coefficients, mark validation false, never silently exclude or retune.',
 'prediction_strengths':[.03125,.125,.5],'frozen_original_second_order_baseline':True,
 'fresh_confirmation_required_before_positive_claim':True}
dump('ANALYSIS_FREEZE.json',{**plan,'source_sha256':sha(Path(__file__))})
def third(pid,di,step):
    z=checked(ROOT/'probes'/f'{pid}_d{di}_q0.npz')
    a=checked(ROOT/'probes'/f'{pid}_d{di}_s{step}_plus.npz');b=checked(ROOT/'probes'/f'{pid}_d{di}_s{step}_minus.npz')
    return {k:(a[k]+b[k]-2*z[k])/(step*step) for k in ['q','input_first','middle_first','output_first','hidden_first']}
def numerics(pid,di,p):
    coarse=third(pid,di,.015625);fine=third(pid,di,.0078125)
    tests={'weighted_logits':wn(p,coarse['q']-fine['q'])/max(wn(p,fine['q']),1e-30)}
    for field in ['middle','output','hidden']:
        name=field+'_first';tests[field]=norm(coarse[name]-fine[name])/max(norm(fine[name]),1e-30)
    return fine,{'question':pid,'direction':di,'relative_differences':tests,'pass':all(v<=.10 for v in tests.values())}
calibration=[];THIRD={};BASES={}
for question in INPUTS['questions']:
    if question['split']!='calibration':continue
    pid=question['id'];base=checked(ROOT/'baseline'/f'{pid}.npz');BASES[pid]=base
    for di in range(2):
        t,row=numerics(pid,di,prob(base['logits']));THIRD[pid,di]=t;calibration.append(row)
passed=all(r['pass'] for r in calibration)
dump('CALIBRATION_LOCK.json',{'pass':passed,'fine_step':.0078125,'coarse_step':.015625,'rows':calibration,'target_values_not_loaded_by_this_analysis':True})
if not passed:
    dump('RESULTS.json',{'status':'numerical_calibration_failed_candidate_not_evaluated','calibration':calibration,'model_calls':0,
                        'not_a_scientific_forecast_failure':True,'fresh_confirmation_required':True})
    dump('SOURCE_IMMUTABILITY.json',verified)
    print(json.dumps({'calibration_pass':False,'rows':calibration}));raise SystemExit(0)
validations=calibration.copy();jets={}
for question in INPUTS['questions']:
    pid=question['id'];base=BASES.get(pid)
    if base is None:base=checked(ROOT/'baseline'/f'{pid}.npz');BASES[pid]=base
    for di in range(2):
        t=THIRD.get((pid,di));valid=True
        if t is None:
            t,row=numerics(pid,di,prob(base['logits']));valid=row['pass'];validations.append(row)
        c=checked(ROOT/'coefficients'/f'{pid}_d{di}.npz');jets[pid,di]=(c,t,valid)
        np.savez_compressed(OUT/f'{pid}_d{di}_third.npz',**t)
dump('COEFFICIENTS_FREEZE.json',{'target_values_not_loaded_by_this_analysis':True,'calibration_pass':True,
 'sha256':{p.name:sha(p) for p in OUT.glob('*_third.npz')},'validation':validations,'previous_outcomes_already_known':True})
rows=[];localization=[]
for question in INPUTS['questions']:
    pid=question['id'];base=BASES[pid];p=prob(base['logits'])
    for di in range(2):
        c,t,valid=jets[pid,di];actual=checked(ROOT/'actual'/f'{pid}_d{di}.npz')
        for ai,a in enumerate(actual['alphas']):
            truth=actual['logits'][ai]-base['logits'];cost=KL(base['logits'],actual['logits'][ai])
            row={'question':pid,'split':question['split'],'direction':di,'alpha':float(a),'strength':abs(float(a)),
                 'third_numerically_validated':valid,'actual_KL':cost,'cost_errors':{},'weighted_logit_errors':{},'field_errors':{}}
            for order in [2,3]:
                delta=a*c['q']+.5*a*a*c['r']+(a**3*t['q']/6 if order==3 else 0.)
                forecast=base['logits']+delta
                row['cost_errors'][str(order)]=abs(KL(base['logits'],forecast)-cost)/cost if cost>1e-10 else None
                row['weighted_logit_errors'][str(order)]=wn(p,delta-truth)/max(wn(p,truth),1e-30)
                row['field_errors'][str(order)]={}
                for field in ['input','middle','output','hidden']:
                    change=actual[field][ai]-base[field]
                    predicted=a*c[field+'_first']+.5*a*a*c[field+'_second']+(a**3*t[field+'_first']/6 if order==3 else 0.)
                    row['field_errors'][str(order)][field]=norm(predicted-change)/max(norm(change),1e-30)
            rows.append(row)
            if abs(a)==.5:
                localization.append({**{n:row[n] for n in ['question','split','direction','alpha','third_numerically_validated']},
                    'second_order_interface_errors':row['field_errors']['2'],'cubic_interface_errors':row['field_errors']['3'],
                    'first_recorded_interface_over_15pct':next((f for f in ['middle','output','hidden'] if row['field_errors']['2'][f]>.15),None),
                    'not_causal_local_error_attribution':True})
dump('ALL_METRICS.json',rows);dump('STRESS_INTERFACE_LOCALIZATION.json',localization)
summary={'status':'complete_secondary_diagnostic','model_calls':0,'fresh_confirmation_required':True,'all_held_numeric_validation_pass':all(r['pass'] for r in validations if 'test' in r['question']),
         'calibration':calibration,'by_strength':{}}
for strength in [.03125,.125,.5]:
    panel=[r for r in rows if r['split']=='held_out' and r['strength']==strength]
    summary['by_strength'][str(strength)]={'held_cases':len(panel),
        'cost_errors':{str(order):stats([r['cost_errors'][str(order)] for r in panel]) for order in [2,3]},
        'weighted_logit_errors':{str(order):stats([r['weighted_logit_errors'][str(order)] for r in panel]) for order in [2,3]},
        'field_errors':{f:{str(order):stats([r['field_errors'][str(order)][f] for r in panel]) for order in [2,3]} for f in ['middle','output','hidden']}}
dump('RESULTS.json',summary);dump('SOURCE_IMMUTABILITY.json',verified)
(OUT/'FINDINGS.txt').write_text('Secondary cubic diagnostic; no new model calls.\n'+json.dumps(summary,indent=2)+'\nFresh native confirmation is still required.\n')
print(json.dumps(summary))
