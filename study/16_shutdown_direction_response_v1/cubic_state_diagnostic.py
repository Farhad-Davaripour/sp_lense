"""State/memory cubic term from the same saved pre-target slopes; no model."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np

START=time.monotonic()
BASE=Path('C:/Users/farha/Research4/artifacts/shutdown_direction_response_v1')
RAW=BASE/'R4_SHUTDOWN_DIRECTION_NATIVE_20261014_RETRY1.zip'
OUT=BASE/'cubic_state_diagnostic_20261014';OUT.mkdir(exist_ok=False)
PLAN=json.loads(Path(__file__).with_name('CUBIC_STATE_DIAGNOSTIC_PLAN.json').read_text(encoding='utf-8'))
GROUPS=['hidden','residual','mixer','mlp','memory_kv','memory_conv','memory_recurrent']
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def stats(xs):
    a=np.asarray([x for x in xs if x is not None],dtype=np.float64)
    return {'n':len(a),'median':float(np.median(a)),'maximum':float(a.max()),'minimum':float(a.min())} if len(a) else {'n':0}
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
with zipfile.ZipFile(RAW) as z:
    manifest=json.loads(z.read('EXPORT_HASHES.json'))
    def data(n):
        b=z.read(n);assert hashlib.sha256(b).hexdigest()==manifest[n];return b
    schema=json.loads(data('STATE_SCHEMA.json'));names=list(schema)
    def load(n,keys):
        with np.load(io.BytesIO(data(n)),allow_pickle=False) as p:return {k:p[k].astype(np.float64) for k in keys}
    inputs=json.loads(data('INPUTS.json'));selection=json.loads(data('DERIVATIVE_SELECTION_LOCK.json'));oldrows=json.loads(data('METRICS.json'))
    coarse,fine=selection['coarse'],selection['fine'];validation=[];rows=[]
    for q in inputs['questions']:
        pid=q['id'];base=load(f'contexts/{pid}_baseline.npz',names)
        c=load(f'coefficients/{pid}_d0.npz',[n+s for n in names for s in ['_first','_second']])
        keys=[n+'_first' for n in names];zero=load(f'probes/{pid}_d0_q0.npz',keys);third=[]
        for step in [coarse,fine]:
            plus=load(f'probes/{pid}_d0_s{step}_plus.npz',keys);minus=load(f'probes/{pid}_d0_s{step}_minus.npz',keys)
            third.append({n:(plus[n+'_first']+minus[n+'_first']-2*zero[n+'_first'])/(step*step) for n in names})
        checks={}
        for group in GROUPS:
            selected=[n for n in names if schema[n]['group']==group]
            denominator=float(np.sqrt(sum(norm(third[1][n])**2 for n in selected)))
            error=float(np.sqrt(sum(norm(third[0][n]-third[1][n])**2 for n in selected)))
            checks[group]={'fine_third_norm':denominator,'discrepancy':error,'relative':error/max(denominator,1e-30),'pass':bool(error<=max(.10*denominator,1e-6))}
        validation.append({'question':pid,'split':q['split'],'checks':checks,'all_original_groups_pass':all(v['pass'] for v in checks.values())})
        del plus,minus,zero
        for old in [r for r in oldrows if r['question']==pid]:
            alpha=old['alpha'];actual=load(f'actual/{pid}_d0_a{alpha}.npz',names)
            sums={g:[0.,0.] for g in GROUPS};fields=[]
            for n in names:
                truth=actual[n]-base[n]
                delta=alpha*c[n+'_first']+.5*alpha*alpha*c[n+'_second']+alpha**3*third[1][n]/6.
                tn=norm(truth);en=norm(delta-truth);group=schema[n]['group']
                if group in sums:sums[group][0]+=tn*tn;sums[group][1]+=en*en
                if tn>max(1e-10,20*np.finfo(np.float32).eps*norm(base[n])):
                    fields.append({'field':n,'group':group,'layer':schema[n]['layer'],'relative_error':en/tn})
            errors={g:float(np.sqrt(en2/tn2)) if tn2>1e-24 else None for g,(tn2,en2) in sums.items()}
            rows.append({'question':pid,'split':q['split'],'alpha':alpha,'strength':abs(alpha),
                'cubic_group_errors':errors,'second_group_errors':{g:old['state_errors']['2'][g]['relative_error'] for g in GROUPS},
                'third_groups_validated':{g:checks[g]['pass'] for g in GROUPS},'resolved_fields':fields})
dump('THIRD_DERIVATIVE_VALIDATION.json',validation);dump('ENDPOINTS.json',rows);dump('PLAN.json',PLAN)
held=[r for r in rows if r['split']=='held_out'];summary={}
for strength in [.03125,.125]:
    panel=[r for r in held if r['strength']==strength]
    summary[str(strength)]={'cases':len(panel),'groups':{g:{
        'second_error':stats([r['second_group_errors'][g] for r in panel]),
        'cubic_error':stats([r['cubic_group_errors'][g] for r in panel]),
        'all_third_numerically_validated':all(r['third_groups_validated'][g] for r in panel),
        'resolved_field_cubic_error':stats([f['relative_error'] for r in panel for f in r['resolved_fields'] if f['group']==g])} for g in GROUPS}}
result={'classification':PLAN['classification'],'source_archive_sha256':PLAN['source_archive_sha256'],
    'by_strength':summary,'all_group_contexts_validated':sum(r['all_original_groups_pass'] for r in validation),
    'contexts':len(validation),'held_all_group_contexts_validated':sum(r['all_original_groups_pass'] for r in validation if r['split']=='held_out'),
    'new_model_calls':0,'GPU_allocations':0,'seconds':time.monotonic()-START,'original_primary_failures_unchanged':True,
    'fresh_confirmation_required':True}
dump('RESULTS.json',result);print(json.dumps(result),flush=True)
