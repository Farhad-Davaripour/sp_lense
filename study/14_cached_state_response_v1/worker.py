import os,time
START=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4')
import ctypes,gc,hashlib,json,shutil,threading,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
assert ROOT.name in ['native_cpu_20261014','native_cpu_20261014_retry1','native_cpu_20261014_retry2'] and ROOT.parent.name=='cached_state_response_v1'
if ROOT.name.endswith('_retry2'):
    recovery=json.loads((ROOT/'RECOVERY_BUDGET.json').read_text())
    assert recovery['previous_attempt_model_calls']==0
    PRIOR_SECONDS=float(recovery['previous_attempt_seconds'])
else:PRIOR_SECONDS=3.547 if ROOT.name.endswith('_retry1') else 0.
START-=PRIOR_SECONDS
SOURCE=Path(__file__).resolve().parent
STATUS={'status':'starting','PID':os.getpid(),'model_calls':0,'main_endpoints':0,'mixed_endpoints':0,'stress_endpoints':0,'GPU_allocations':0,'Colab':False}
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def report(phase,**extra):
    STATUS.update(phase=phase,seconds=time.monotonic()-START,**extra);dump(ROOT/'STATUS.json',STATUS);print(json.dumps(STATUS),flush=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def free_memory():
    class M(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ['total','free','page','freepage','virtual','freevirtual','extended']]
    m=M();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return int(m.free)
def deadline():
    if time.monotonic()-START>2400:raise TimeoutError('Native2400-second ceiling')
    if 'rig' in globals() and rig.calls>=300:raise RuntimeError('Combined300-forward ceiling')
def hardstop():dump(ROOT/'HARD_STOP.json',{'PID':os.getpid()});os._exit(124)
TIMER=threading.Timer(max(1.,2700-PRIOR_SECONDS),hardstop);TIMER.daemon=True;TIMER.start()
def export():
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='EXPORT_HASHES.json')
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files};dump(ROOT/'EXPORT_HASHES.json',manifest)
    suffix='_RETRY2' if ROOT.name.endswith('_retry2') else '_RETRY1' if ROOT.name.endswith('_retry1') else ''
    archive=ROOT.parent/f'R4_CACHED_STATE_NATIVE_20261014{suffix}.zip'
    with zipfile.ZipFile(archive,'x') as z:
        for p in files+[ROOT/'EXPORT_HASHES.json']:
            if p.suffix=='.npz':z.write(p,p.relative_to(ROOT).as_posix(),compress_type=zipfile.ZIP_STORED)
            else:z.write(p,p.relative_to(ROOT).as_posix(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=2)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
    receipt={'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'members':len(files)+1,
        'model_calls':STATUS['model_calls'],'status':STATUS['status'],'native_export_seconds':time.monotonic()-START,
        'all_members_and_CRC_verified':True,'GPU_allocations':0,'Colab':False}
    dump(ROOT.parent/f'NATIVE_EXPORT_RECEIPT_20261014{suffix}.json',receipt);print(json.dumps({'export_complete':receipt}),flush=True)

try:
    import numpy as np
    import torch
    import transformers
    from native import CachedRig,cache_hash
    from maths import GROUPS,probability,kl,wn,norm,group_norm,subtract,central,checks,evaluate
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available() and '+cpu' in torch.__version__
    PLAN=json.loads((SOURCE/'PLAN.json').read_text());INPUTS=json.loads((ROOT/'INPUTS.json').read_text());FREEZE=json.loads((ROOT/'EXECUTION_FREEZE.json').read_text())
    for n,h in FREEZE['sha256'].items():assert sha(ROOT/n)==h
    admission_samples=[];admission_start=time.monotonic()
    while True:
        available=free_memory();admission_samples.append({'seconds':time.monotonic()-START,'free_memory_bytes':available})
        if available>=8589934592 or time.monotonic()-admission_start>=180:break
        report('waiting_for_memory',free_memory_bytes=available,minimum_bytes=8589934592)
        time.sleep(10)
    admission={'free_memory_bytes':available,'minimum_bytes':8589934592,'disk_free_bytes':shutil.disk_usage(ROOT).free,
        'disk_min_bytes':17179869184,'CPU_threads':4,'call_limit':300,'native_seconds':2400,'export_reserve_seconds':300}
    admission.update(samples=admission_samples,previous_attempt_seconds=PRIOR_SECONDS,previous_attempt_model_calls=0)
    admission['admitted']=admission['free_memory_bytes']>=8589934592 and admission['disk_free_bytes']>=17179869184
    dump(ROOT/'ADMISSION.json',admission);assert admission['admitted']
    pin=json.loads((SOURCE/'model_pin.json').read_text());weights=Path(PLAN['checkpoint']);assert pin['revision']==PLAN['revision']
    verified=[]
    for item in pin['files']:
        p=weights/item['name']
        if not p.is_file():continue
        assert p.stat().st_size==item['size']
        if item.get('sha256'):assert sha(p)==item['sha256']
        else:
            data=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['git_blob_id']
        verified.append(item['name'])
    assert len(verified)==8;dump(ROOT/'IDENTITY.json',{'revision':pin['revision'],'files':verified})
    old=Path('C:/Users/farha/Research4/artifacts/finite_logit_response_v1')
    immutable={p.name:sha(p) for p in old.glob('R4_FINITE_LOGIT_*.zip')};assert len(immutable)==2
    report('loading_model');rig=CachedRig(weights,deadline);rig.precision(torch.float32)
    dump(ROOT/'RUNTIME.json',{'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,
        'CPU_threads':4,'precision':'FP32 promoted stored weights','use_cache':True,'original_reference_kernels':True,'GPU_allocations':0})
    for n in ['contexts','probes','coefficients','predictions','actual','mixed','stress']:(ROOT/n).mkdir()
    with np.load(ROOT/'DIRECTIONS.npz',allow_pickle=False) as z:V=z['vectors'].copy()
    QUESTIONS=INPUTS['questions'];STEPS=PLAN['derivatives']['probe_steps'];BASES={};ENTRIES={};JETS={}
    exact_available=None;forward_available=None;attempts=[]
    def save_value(p,value):np.savez_compressed(p,logits=value['logits'],**value['values'])
    def save_probe(p,value):
        data={'logits':value['logits'],**{n+'_value':x for n,x in value['values'].items()}}
        if 'first' in value:data.update({n+'_first':x for n,x in value['first'].items()});data['q']=value['q']
        np.savez_compressed(p,**data)
    def prepare_context(q):
        ctx=rig.prepare_context(q);rig.activate(q['id']);base=rig.value();BASES[q['id']]=base
        torch.save(ctx['cache'],ROOT/'contexts'/f'{q["id"]}_incoming_cache.pt')
        save_value(ROOT/'contexts'/f'{q["id"]}_baseline.npz',base)
        dump(ROOT/'contexts'/f'{q["id"]}.json',{'token_id':ctx['token'],'token_text':rig.tokenizer.decode([ctx['token']]),'EOS_fallback':ctx['fallback'],
            'incoming_cache_hash':ctx['hash'],'prefix_length':q['prefix_length'],'outgoing_constant_invariants':base['invariants']})
        dump(ROOT/'STATE_SCHEMA.json',rig.schema)
    def primal_ok(value,base):
        errors=[norm(value['values'][n]-base['values'][n])/max(1.,norm(base['values'][n])) for n in base['values']]
        errors.append(norm(value['logits']-base['logits'])/max(1.,norm(base['logits'])))
        assert max(errors)<=1e-6;return max(errors)
    def sample(q,direction,tag,steps,try_exact=False):
        global exact_available,forward_available
        rig.activate(q['id']);base=BASES[q['id']];exact=None;first=None
        if try_exact and exact_available is not False:
            before=rig.calls;begin=time.monotonic()
            try:
                exact=rig.exact_jet(direction);primal_ok(exact,base)
                exact_available=exact['seconds']<=45
                attempts.append({'kind':'exact','tag':tag,'success':True,'seconds':exact['seconds'],'calls':rig.calls-before})
                if not exact_available:exact=None
            except (RuntimeError,NotImplementedError,MemoryError) as e:
                message=str(e).lower()
                if not any(s in message for s in ['grad','backward','forward','inplace','in-place','not implemented','memory','jvp']):raise
                attempts.append({'kind':'exact','tag':tag,'success':False,'seconds':time.monotonic()-begin,'calls':rig.calls-before,'error':type(e).__name__+': '+str(e)})
                exact_available=False;gc.collect()
        if forward_available is not False:
            before=rig.calls
            try:
                first=rig.first_at(direction);primal_ok(first,base);forward_available=True
                save_probe(ROOT/'probes'/f'{tag}_q0.npz',first)
            except (RuntimeError,NotImplementedError,MemoryError) as e:
                if not any(s in str(e).lower() for s in ['grad','forward','not implemented','memory','jvp','tangent']):raise
                attempts.append({'kind':'forward_AD','tag':tag,'success':False,'calls':rig.calls-before,'error':type(e).__name__+': '+str(e)})
                forward_available=False;gc.collect()
        fds={}
        for step in steps:
            plus=rig.first_at(direction,step) if forward_available else rig.value(direction,step)
            minus=rig.first_at(direction,-step) if forward_available else rig.value(direction,-step)
            save_probe(ROOT/'probes'/f'{tag}_s{step}_plus.npz',plus);save_probe(ROOT/'probes'/f'{tag}_s{step}_minus.npz',minus)
            fds[step]=central(plus,minus,base,step)
        if exact is not None:
            np.savez_compressed(ROOT/'probes'/f'{tag}_exact.npz',logits=exact['logits'],q=exact['q'],r=exact['r'],
                **{n+'_first':x for n,x in exact['first'].items()},**{n+'_second':x for n,x in exact['second'].items()})
        return {'first':first,'exact':exact,'fd':fds}

    for q in QUESTIONS:
        if q['split']!='calibration':continue
        prepare_context(q)
        for di,v in enumerate(V):
            ENTRIES[q['id'],di]=sample(q,torch.tensor(v),f'{q["id"]}_d{di}',STEPS,try_exact=True)
            report('calibration',question=q['id'],direction=di,model_calls=rig.calls,exact_available=exact_available,forward_available=forward_available)
    SELECTION=None;candidate_checks=[]
    for coarse,fine in zip(STEPS[:-1],STEPS[1:]):
        modes={};detail={}
        for group in GROUPS:
            alternatives=[]
            for mode in ['exact','slope_difference','value_difference']:
                available=True;tests=[]
                for (pid,di),entry in ENTRIES.items():
                    if mode=='exact':
                        if entry['exact'] is None:available=False;break
                        qcoef=entry['exact']['first'];rcoef=entry['exact']['second']
                        reference=entry['fd'][fine]['slope_second'] or entry['fd'][fine]['value_second']
                        qtest=checks(qcoef,entry['fd'][fine]['value_first'],rig.schema,.03)[group]
                        rtest=checks(rcoef,reference,rig.schema,.05)[group]
                    elif mode=='slope_difference':
                        if entry['first'] is None or entry['fd'][fine]['slope_second'] is None:available=False;break
                        qcoef=entry['first']['first'];rcoef=entry['fd'][fine]['slope_second']
                        qtest=checks(qcoef,entry['fd'][fine]['value_first'],rig.schema,.03)[group]
                        rtest=checks(entry['fd'][coarse]['slope_second'],rcoef,rig.schema,.05)[group]
                    else:
                        qtest=checks(entry['fd'][coarse]['value_first'],entry['fd'][fine]['value_first'],rig.schema,.03)[group]
                        rtest=checks(entry['fd'][coarse]['value_second'],entry['fd'][fine]['value_second'],rig.schema,.05)[group]
                    if group=='hidden':
                        selected_q=(entry['exact']['q'] if mode=='exact' else entry['first']['q'] if mode=='slope_difference' else rig.head(entry['fd'][fine]['value_first']['hidden']))
                        reference_q=entry['fd'][fine]['q_fd'];base_logits=BASES[pid]['logits']
                        weighted_error=wn(base_logits,selected_q-reference_q)/max(wn(base_logits,reference_q),1e-30)
                        qtest['weighted_logits_relative']=weighted_error
                        qtest['pass']=qtest['pass'] and weighted_error<=.02
                    tests.append({'question':pid,'direction':di,'q':qtest,'r':rtest})
                alternatives.append({'mode':mode,'available':available,'pass':available and all(t['q']['pass'] and t['r']['pass'] for t in tests),'tests':tests})
                if alternatives[-1]['pass'] and group not in modes:modes[group]=mode
            detail[group]=alternatives
        candidate_checks.append({'coarse':coarse,'fine':fine,'modes':modes,'checks':detail})
        if SELECTION is None and len(modes)==len(GROUPS):SELECTION={'coarse':coarse,'fine':fine,'modes':modes}
    dump(ROOT/'CALIBRATION_NUMERICS.json',candidate_checks)
    assert SELECTION is not None,'No declared numerical scheme validated; retain calibration diagnostics for bounded continuation'
    SELECTION.update(attempts=attempts,forward_available=forward_available,exact_available=exact_available,target_endpoints_seen=0)
    dump(ROOT/'DERIVATIVE_SELECTION_LOCK.json',SELECTION);selection_sha=sha(ROOT/'DERIVATIVE_SELECTION_LOCK.json')
    coarse,fine=SELECTION['coarse'],SELECTION['fine']
    def jet_from(q,entry):
        first={};second={};validations={};valid=True
        for group,mode in SELECTION['modes'].items():
            if mode=='exact':
                assert entry['exact'] is not None;qcoef=entry['exact']['first'];rcoef=entry['exact']['second']
                qt=checks(qcoef,entry['fd'][fine]['value_first'],rig.schema,.03)[group]
                rt=checks(rcoef,entry['fd'][fine]['slope_second'] or entry['fd'][fine]['value_second'],rig.schema,.05)[group]
            elif mode=='slope_difference':
                qcoef=entry['first']['first'];rcoef=entry['fd'][fine]['slope_second']
                qt=checks(qcoef,entry['fd'][fine]['value_first'],rig.schema,.03)[group]
                rt=checks(entry['fd'][coarse]['slope_second'],rcoef,rig.schema,.05)[group]
            else:
                qcoef=entry['fd'][fine]['value_first'];rcoef=entry['fd'][fine]['value_second']
                qt=checks(entry['fd'][coarse]['value_first'],qcoef,rig.schema,.03)[group]
                rt=checks(entry['fd'][coarse]['value_second'],rcoef,rig.schema,.05)[group]
            if group=='hidden':
                weighted_error=wn(BASES[q['id']]['logits'],rig.head(qcoef['hidden'])-entry['fd'][fine]['q_fd'])/max(wn(BASES[q['id']]['logits'],entry['fd'][fine]['q_fd']),1e-30)
                qt['weighted_logits_relative']=weighted_error;qt['pass']=qt['pass'] and weighted_error<=.02
            validations[group]={'q':qt,'r':rt,'mode':mode};valid=valid and qt['pass'] and rt['pass']
            for n in qcoef:
                if rig.schema[n]['group']==group:first[n]=qcoef[n];second[n]=rcoef[n]
        # Interface fields use the residual method; group gates above cover residual outputs.
        for n in rig.names:
            if n in first:continue
            mode=SELECTION['modes']['residual']
            first[n]=(entry['exact']['first'][n] if mode=='exact' else entry['first']['first'][n] if mode=='slope_difference' else entry['fd'][fine]['value_first'][n])
            second[n]=(entry['exact']['second'][n] if mode=='exact' else entry['fd'][fine]['slope_second'][n] if mode=='slope_difference' else entry['fd'][fine]['value_second'][n])
        return {'first':first,'second':second,'q':rig.head(first['hidden']),'r':rig.head(second['hidden']),'validation':validations,'validated':bool(valid)}
    def save_jet(path,jet):
        np.savez_compressed(path,q=jet['q'],r=jet['r'],**{n+'_first':x for n,x in jet['first'].items()},**{n+'_second':x for n,x in jet['second'].items()})
        dump(path.with_suffix('.json'),{'validated':jet['validated'],'validation':jet['validation'],'methods':SELECTION['modes']})
    exact_needed=any(m=='exact' for m in SELECTION['modes'].values())
    for q in QUESTIONS:
        pid=q['id']
        if q['split']=='held_out':prepare_context(q)
        rig.activate(pid)
        for di,v in enumerate(V):
            entry=ENTRIES[pid,di] if q['split']=='calibration' else sample(q,torch.tensor(v),f'{pid}_d{di}',[coarse,fine],try_exact=exact_needed)
            jet=jet_from(q,entry);JETS[pid,di]=jet;save_jet(ROOT/'coefficients'/f'{pid}_d{di}.npz',jet)
            report('forecast_coefficients',question=pid,direction=di,model_calls=rig.calls,validated=jet['validated'])
    ENTRIES.clear();gc.collect()
    def forecasts(base,jet,alphas):
        l0=base['logits'].astype(np.float64);q=jet['q'].astype(np.float64);r=jet['r'].astype(np.float64)
        return [{'alpha':a,'quadratic_Fisher':.5*a*a*wn(l0,q)**2,
            'first_order_logits':kl(l0,l0+a*q),'second_order_logits':kl(l0,l0+a*q+.5*a*a*r)} for a in alphas]
    alphas=[sign*s for s in PLAN['primary_strengths'] for sign in PLAN['signs']];FORECASTS={}
    for q in QUESTIONS:
        for di in range(2):
            pid=q['id'];f=forecasts(BASES[pid],JETS[pid,di],alphas);FORECASTS[pid,di]=f
            dump(ROOT/'predictions'/f'{pid}_d{di}.json',f)
    pred_paths=list((ROOT/'coefficients').glob('*'))+list((ROOT/'predictions').glob('*'))
    pred_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in pred_paths}
    dump(ROOT/'PREDICTION_FREEZE.json',{'sha256':pred_hashes,'selection_sha256':selection_sha,'target_endpoints_seen':0,'all_forecasts_precede_endpoints':True})
    prediction_sha=sha(ROOT/'PREDICTION_FREEZE.json')
    def score(q,di,a,actual,jet,f):
        base=BASES[q['id']];cost=kl(base['logits'],actual['logits']);state,fields=evaluate(actual['values'],base['values'],jet['first'],jet['second'],a,rig.schema)
        row={'question':q['id'],'split':q['split'],'family':q['family'],'direction':di,'alpha':a,'strength':abs(a),
            'actual_KL':cost,'resolved':cost>1e-10,'derivative_validated':jet['validated'],'cost_predictions':{},'cost_errors':{},
            'weighted_logit_errors':{},'distribution_KL_actual_to_forecast':{},'state_errors':state,'fields':fields}
        delivered=actual['values']['b11_input'].astype(np.float64)-base['values']['b11_input']
        requested=actual['requested_vector'];projection=float(delivered@requested/max(float(requested@requested),1e-30))
        row.update(actual_input_norm=norm(delivered),requested_input_norm=abs(a)*norm(requested),
            delivered_projection=projection,off_direction_norm=norm(delivered-projection*requested),zero_delivery=norm(delivered)==0.)
        for name in ['quadratic_Fisher','first_order_logits','second_order_logits']:
            row['cost_predictions'][name]=f[name];row['cost_errors'][name]=abs(f[name]-cost)/cost if cost>1e-10 else None
        true=actual['logits'].astype(np.float64)-base['logits']
        for name,order in [('first_order_logits',1),('second_order_logits',2)]:
            delta=a*jet['q'].astype(np.float64)+(0.5*a*a*jet['r'] if order==2 else 0.)
            row['weighted_logit_errors'][name]=wn(base['logits'],delta-true)/max(wn(base['logits'],true),1e-30)
            row['distribution_KL_actual_to_forecast'][name]=kl(actual['logits'],base['logits'].astype(np.float64)+delta)
        return row
    metrics=[];nulls=[]
    for q in QUESTIONS:
        pid=q['id'];rig.activate(pid);base=BASES[pid]
        null=rig.value(torch.tensor(V[0]),0.);assert np.array_equal(null['logits'],base['logits']) and all(np.array_equal(null['values'][n],base['values'][n]) for n in rig.names)
        nulls.append({'question':pid,'exact':True})
        for di,v in enumerate(V):
            for ai,a in enumerate(alphas):
                actual=rig.value(torch.tensor(v),a);metrics.append(score(q,di,a,actual,JETS[pid,di],FORECASTS[pid,di][ai]))
                save_value(ROOT/'actual'/f'{pid}_d{di}_a{a}.npz',actual);STATUS['main_endpoints']+=1
            dump(ROOT/'METRICS.json',metrics);report('main_endpoints',question=pid,direction=di,model_calls=rig.calls)
    dump(ROOT/'NULL_CHECKS.json',nulls);assert len(metrics)==48
    assert sha(ROOT/'PREDICTION_FREEZE.json')==prediction_sha and sha(ROOT/'DERIVATIVE_SELECTION_LOCK.json')==selection_sha
    for n,h in pred_hashes.items():assert sha(ROOT/n)==h
    # Compact, prospectively selected stress localization while the model is loaded.
    stress=[]
    if 300-rig.calls>=4 and 2400-(time.monotonic()-START)>=120:
        for q in QUESTIONS:
            if q['id'] not in PLAN['optional_stress']['questions']:continue
            pid=q['id'];rig.activate(pid);jet=JETS[pid,0];fs=forecasts(BASES[pid],jet,[.5,-.5])
            dump(ROOT/'stress'/f'{pid}_forecasts_before.json',fs)
            for i,a in enumerate([.5,-.5]):
                actual=rig.value(torch.tensor(V[0]),a);stress.append(score(q,0,a,actual,jet,fs[i]))
                save_value(ROOT/'stress'/f'{pid}_a{a}.npz',actual);STATUS['stress_endpoints']+=1
        dump(ROOT/'STRESS_METRICS.json',stress)
    mixed_admission={'remaining_calls':300-rig.calls,'remaining_seconds':2400-(time.monotonic()-START),'free_bytes':free_memory()}
    mixed_admission['admitted']=mixed_admission['remaining_calls']>=25 and mixed_admission['remaining_seconds']>=300 and mixed_admission['free_bytes']>=4294967296
    dump(ROOT/'MIXED_ADMISSION.json',mixed_admission)
    if mixed_admission['admitted']:
        mix_jets={};mix_vectors={'sum':(V[0]+V[1])/np.sqrt(2),'difference':(V[0]-V[1])/np.sqrt(2),'weighted':(2*V[0]+V[1])/np.sqrt(5)}
        mixed_questions=[q for q in QUESTIONS if q['id'] in PLAN['optional_mixed_plane']['questions']]
        mix_forecasts={};interaction=[]
        for q in mixed_questions:
            pid=q['id'];rig.activate(pid);entry=sample(q,torch.tensor(mix_vectors['sum']),f'mix_{pid}',[coarse,fine],try_exact=exact_needed)
            jm=jet_from(q,entry);j0=JETS[pid,0];j1=JETS[pid,1]
            cross={n:(jm['second'][n].astype(np.float64)-.5*(j0['second'][n].astype(np.float64)+j1['second'][n])).astype(np.float32) for n in rig.names}
            interaction.append({'question':pid,'cross_norms':{g:group_norm(cross,rig.schema,g) for g in GROUPS},'not_full_space_Hessian':True})
            for name,coeff in [('sum',(1/np.sqrt(2),1/np.sqrt(2))),('difference',(1/np.sqrt(2),-1/np.sqrt(2))),('weighted',(2/np.sqrt(5),1/np.sqrt(5)))]:
                aa,bb=coeff
                first={n:(aa*j0['first'][n]+bb*j1['first'][n]).astype(np.float32) for n in rig.names}
                second={n:(aa*aa*j0['second'][n]+2*aa*bb*cross[n]+bb*bb*j1['second'][n]).astype(np.float32) for n in rig.names}
                diagonal={n:(aa*aa*j0['second'][n]+bb*bb*j1['second'][n]).astype(np.float32) for n in rig.names}
                jet={'first':first,'second':second,'q':rig.head(first['hidden']),'r':rig.head(second['hidden']),
                    'validated':jm['validated'] and j0['validated'] and j1['validated'],'validation':jm['validation']}
                mix_jets[pid,name]=jet;f=forecasts(BASES[pid],jet,[.125,-.125]);mix_forecasts[pid,name]=f
                save_jet(ROOT/'mixed'/f'{pid}_{name}.npz',jet)
                dump(ROOT/'mixed'/f'{pid}_{name}_forecasts.json',f)
                np.savez_compressed(ROOT/'mixed'/f'{pid}_{name}_diagonal.npz',r=rig.head(diagonal['hidden']))
        mix_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'mixed').iterdir()}
        dump(ROOT/'MIXED_PREDICTION_FREEZE.json',{'sha256':mix_hashes,'mixed_endpoints_seen':0,'unprobed_directions':['difference','weighted']})
        mix_rows=[]
        for q in mixed_questions:
            pid=q['id'];rig.activate(pid)
            for name,v in mix_vectors.items():
                for ai,a in enumerate([.125,-.125]):
                    actual=rig.value(torch.tensor(v),a);row=score(q,name,a,actual,mix_jets[pid,name],mix_forecasts[pid,name][ai])
                    with np.load(ROOT/'mixed'/f'{pid}_{name}_diagonal.npz',allow_pickle=False) as z:rdiag=z['r']
                    ldiag=BASES[pid]['logits'].astype(np.float64)+a*mix_jets[pid,name]['q']+.5*a*a*rdiag
                    row['no_interaction_cost_error']=abs(kl(BASES[pid]['logits'],ldiag)-row['actual_KL'])/row['actual_KL']
                    mix_rows.append(row);save_value(ROOT/'mixed'/f'{pid}_{name}_a{a}_actual.npz',actual);STATUS['mixed_endpoints']+=1
        dump(ROOT/'MIXED_METRICS.json',mix_rows);dump(ROOT/'DIRECTION_INTERACTION.json',interaction)
        for n,h in mix_hashes.items():assert sha(ROOT/n)==h
    assert all(sha(old/n)==h for n,h in immutable.items())
    dump(ROOT/'BASELINE_IMMUTABILITY.json',{'study13_archives_unchanged':True,'sha256':immutable})
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in rig.model.modules())
    STATUS['status']='native_complete';report('native_complete_export_pending',model_calls=rig.calls,methods=SELECTION['modes'])
except BaseException as e:
    if 'rig' in globals():STATUS['model_calls']=rig.calls
    STATUS['status']='fault_or_calibration_stop';report('fault',error=type(e).__name__+': '+str(e))
    (ROOT/'FAULT.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    if 'rig' in globals():del rig;gc.collect()
    try:export()
    finally:TIMER.cancel()
