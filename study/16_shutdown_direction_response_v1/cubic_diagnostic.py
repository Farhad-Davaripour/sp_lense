"""Fixed extra Taylor term from saved slopes; no model loading or refitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np

START=time.monotonic()
BASE=Path('C:/Users/farha/Research4/artifacts/shutdown_direction_response_v1')
RAW=BASE/'R4_SHUTDOWN_DIRECTION_NATIVE_20261014_RETRY1.zip'
OUT=BASE/'cubic_diagnostic_20261014';OUT.mkdir(exist_ok=False)
PLAN=json.loads((Path(__file__).with_name('CUBIC_DIAGNOSTIC_PLAN.json')).read_text(encoding='utf-8'))
receipt=json.loads((BASE/'UI_VERIFIED_EXPORT_RECEIPT_20261014.json').read_text(encoding='utf-8'))
assert receipt['sha256']==PLAN['source_archive_sha256'] and RAW.stat().st_size==receipt['bytes']
def prob(x):
    x=np.asarray(x,dtype=np.float64);m=x.max();lp=x-(m+np.log(np.exp(x-m).sum()));return np.exp(lp),lp
def KL(a,b):
    p,lp=prob(a);_,lq=prob(b);return max(0.,float(p@(lp-lq)))
def wn(p,x):
    x=np.asarray(x,dtype=np.float64);return float(np.sqrt(max(0.,p@((x-p@x)**2))))
def stats(xs):
    x=np.asarray([v for v in xs if v is not None],dtype=np.float64)
    return {'n':len(x),'median':float(np.median(x)),'maximum':float(x.max()),'minimum':float(x.min())} if len(x) else {'n':0}
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
with zipfile.ZipFile(RAW) as z:
    manifest=json.loads(z.read('EXPORT_HASHES.json'))
    def read_bytes(n):
        data=z.read(n);assert hashlib.sha256(data).hexdigest()==manifest[n];return data
    def load(n,keys):
        with np.load(io.BytesIO(read_bytes(n)),allow_pickle=False) as values:return {k:values[k].astype(np.float64) for k in keys}
    inputs=json.loads(read_bytes('INPUTS.json'));selection=json.loads(read_bytes('DERIVATIVE_SELECTION_LOCK.json'))
    source_rows=json.loads(read_bytes('METRICS.json'))
    coarse,fine=selection['coarse'],selection['fine'];validation=[];rows=[]
    for question in inputs['questions']:
        pid=question['id'];baseline=load(f'contexts/{pid}_baseline.npz',['logits'])['logits'];p,_=prob(baseline)
        c=load(f'coefficients/{pid}_d0.npz',['q','r'])
        q0=load(f'probes/{pid}_d0_q0.npz',['q'])['q'];third=[]
        for step in [coarse,fine]:
            plus=load(f'probes/{pid}_d0_s{step}_plus.npz',['q'])['q']
            minus=load(f'probes/{pid}_d0_s{step}_minus.npz',['q'])['q']
            third.append((plus+minus-2*q0)/(step*step))
        disagreement=wn(p,third[0]-third[1])/max(wn(p,third[1]),1e-30)
        validation.append({'question':pid,'split':question['split'],'coarse':coarse,'fine':fine,
            'weighted_third_derivative_disagreement':disagreement,'pass':bool(disagreement<=.10)})
        for old in [r for r in source_rows if r['question']==pid]:
            alpha=old['alpha'];actual=load(f'actual/{pid}_d0_a{alpha}.npz',['logits'])['logits']
            delta=alpha*c['q']+.5*alpha*alpha*c['r']+alpha**3*third[1]/6.
            cost=KL(baseline,actual);predicted_cost=KL(baseline,baseline+delta)
            error=wn(p,delta-(actual-baseline))/max(wn(p,actual-baseline),1e-30)
            ce=abs(predicted_cost-cost)/cost if cost>1e-10 else None
            rows.append({'question':pid,'split':question['split'],'scenario':question['scenario'],'condition':question['condition'],
                'alpha':alpha,'strength':abs(alpha),'third_numerically_validated':bool(disagreement<=.10),
                'second_weighted_logit_error':old['weighted_logit_errors']['second_order_logits'],'cubic_weighted_logit_error':error,
                'second_cost_error':old['cost_errors']['second_order_logits'],'cubic_cost_error':ce,
                'weighted_improves':bool(error<old['weighted_logit_errors']['second_order_logits']),
                'cost_improves':bool(ce is not None and ce<old['cost_errors']['second_order_logits'])})
dump('THIRD_DERIVATIVE_VALIDATION.json',validation);dump('ENDPOINTS.json',rows)
held=[r for r in rows if r['split']=='held_out'];summary={}
for strength in [.03125,.125]:
    panel=[r for r in held if r['strength']==strength]
    summary[str(strength)]={'cases':len(panel),'all_third_numerically_validated':all(r['third_numerically_validated'] for r in panel),
        'second_weighted_logit_error':stats([r['second_weighted_logit_error'] for r in panel]),
        'cubic_weighted_logit_error':stats([r['cubic_weighted_logit_error'] for r in panel]),
        'second_cost_error':stats([r['second_cost_error'] for r in panel]),'cubic_cost_error':stats([r['cubic_cost_error'] for r in panel]),
        'weighted_cases_improved':sum(r['weighted_improves'] for r in panel),'cost_cases_improved':sum(r['cost_improves'] for r in panel)}
result={'classification':PLAN['classification'],'source_sha256':receipt['sha256'],'by_strength':summary,
        'contexts_numerically_validated':sum(r['pass'] for r in validation),'contexts':len(validation),
        'held_contexts_numerically_validated':sum(r['pass'] for r in validation if r['split']=='held_out'),
        'new_model_calls':0,'GPU_allocations':0,'seconds':time.monotonic()-START,
        'original_primary_failures_unchanged':True,'fresh_prospective_confirmation_required':True,'internal_state_forecast_not_tested':True}
dump('RESULTS.json',result);dump('PLAN.json',PLAN)
print(json.dumps(result),flush=True)
