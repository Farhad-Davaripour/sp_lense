"""Frozen prospective finite-logit response test; CPU only, capped300 forwards."""
import os,time
START=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4')
import ctypes,gc,hashlib,json,threading,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
assert ROOT.name=='native_cpu_20261014' and ROOT.parent.name=='finite_logit_response_v1'
SOURCE=Path(__file__).resolve().parent
STATUS={'status':'starting','PID':os.getpid(),'model_calls':0,'primary_endpoints':0,'post_endpoints':0,'GPU_allocations':0,'Colab':False}

def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def report(phase,**extra):
    STATUS.update(phase=phase,seconds=time.monotonic()-START,**extra)
    dump(ROOT/'STATUS.json',STATUS);print(json.dumps(STATUS,allow_nan=False),flush=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()
def memory():
    class M(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ['total','free','page','freepage','virtual','freevirtual','extended']]
    m=M();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return int(m.free)
def deadline():
    if time.monotonic()-START>2400:raise TimeoutError('Native2400-second ceiling')
    if 'rig' in globals() and rig.calls>=300:raise RuntimeError('Combined300-call ceiling')
def hardstop():
    dump(ROOT/'HARD_STOP.json',{'PID':os.getpid(),'seconds':time.monotonic()-START});os._exit(124)
TIMER=threading.Timer(2700,hardstop);TIMER.daemon=True;TIMER.start()

def export():
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='EXPORT_HASHES.json')
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    dump(ROOT/'EXPORT_HASHES.json',manifest)
    archive=ROOT.parent/'R4_FINITE_LOGIT_NATIVE_20261014.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files+[ROOT/'EXPORT_HASHES.json']:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==expected for n,expected in manifest.items())
    receipt={'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'members':len(files)+1,
        'all_members_and_CRC_verified':True,'model_calls':STATUS['model_calls'],'status':STATUS['status'],
        'native_and_export_seconds':time.monotonic()-START,'GPU_allocations':0,'Colab':False}
    dump(ROOT.parent/'NATIVE_EXPORT_RECEIPT_20261014.json',receipt)
    print(json.dumps({'export_complete':receipt}),flush=True)

try:
    import numpy as np
    import torch
    import transformers
    from derivatives import JetRig,kl,probability,weighted_norm,relative,consistency
    from native import NAMES
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available() and '+cpu' in torch.__version__
    PLAN=json.loads((SOURCE/'PLAN.json').read_text())
    INPUTS=json.loads((ROOT/'INPUTS.json').read_text())
    FREEZE=json.loads((ROOT/'EXECUTION_FREEZE.json').read_text())
    for name,expected in FREEZE['sha256'].items():assert sha(ROOT/name)==expected
    admission={'free_bytes':memory(),'minimum_bytes':8589934592,'threads':4,'interop_threads':1,'CPU_only':True,
        'call_limit':300,'native_limit_seconds':2400,'export_reserve_seconds':300}
    admission['admitted']=admission['free_bytes']>=admission['minimum_bytes']
    dump(ROOT/'ADMISSION.json',admission);assert admission['admitted'],'Fresh memory admission failed'
    checkpoint=Path(PLAN['checkpoint']);pin=json.loads((SOURCE/'model_pin.json').read_text())
    assert pin['revision']==PLAN['revision']
    verified=[]
    for item in pin['files']:
        p=checkpoint/item['name']
        if not p.is_file():continue
        assert p.stat().st_size==item['size']
        if item.get('sha256'):assert sha(p)==item['sha256']
        else:
            data=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['git_blob_id']
        verified.append(item['name'])
    assert len(verified)==8
    dump(ROOT/'IDENTITY.json',{'revision':pin['revision'],'verified_checkpoint_files':verified})
    baseline_dir=Path('C:/Users/farha/Research4/artifacts/block_energy_replication_v1')
    immutable={p.name:sha(p) for p in baseline_dir.glob('R4_BLOCK_ENERGY_*.zip')}
    assert len(immutable)==4
    dump(ROOT/'BASELINE_HASHES_BEFORE.json',immutable)
    stopped=Path('C:/Users/farha/Research4/artifacts/consistency_monitor_v1/native_cpu_20261012/continuation/STARTED.json')
    assert not stopped.exists()
    report('loading_model')
    rig=JetRig(checkpoint,deadline);rig.precision(torch.float32)
    dump(ROOT/'RUNTIME.json',{'torch':torch.__version__,'transformers':transformers.__version__,
        'numpy':np.__version__,'device':'CPU','precision':'FP32 promoted stored weights',
        'threads':4,'interop_threads':1,'attention':'SDPA MATH and reference gated-delta kernels',
        'use_cache':False,'GPU_allocations':0,'model_training':False})
    with np.load(ROOT/'DIRECTIONS.npz',allow_pickle=False) as z:V=z['vectors'].copy()
    QUESTIONS=INPUTS['questions'];STEPS=PLAN['derivatives']['probe_steps']
    contexts={};raw={};jets={};numerics=[];exact_available=None;exact_attempts=[]
    for name in ['baseline','probes','coefficients','predictions','actual','post']: (ROOT/name).mkdir()

    def save_first(path,value):
        np.savez_compressed(path,logits=value['logits'],q=value['q'],
            **{n+'_primal':x for n,x in value['primal'].items()},
            **{n+'_first':x for n,x in value['first'].items()})
    def check_primal(value,base):
        assert value['prefix_hashes']==base['prefix_hashes']
        errors=[float(np.linalg.norm(value['primal'][n]-base['points'][n])/max(1.,np.linalg.norm(base['points'][n]))) for n in NAMES]
        errors.append(float(np.linalg.norm(value['logits']-base['logits'])/max(1.,np.linalg.norm(base['logits']))))
        assert max(errors)<=1e-6
        return {'relative_max':max(errors),'logits_exact':bool(np.array_equal(value['logits'],base['logits']))}
    def probes(question,di,site,direction,steps,tag):
        base=contexts[question['id']];ids=question['prefix_ids']
        first=rig.first_at(ids,site,direction)
        primal=check_primal(first,base)
        assert weighted_norm(base['logits'],first['q'])>1e-8
        save_first(ROOT/'probes'/f'{tag}_q0.npz',first)
        values={};q_checks={}
        for step in steps:
            plus=rig.first_at(ids,site,direction,step);minus=rig.first_at(ids,site,direction,-step)
            assert plus['prefix_hashes']==base['prefix_hashes'] and minus['prefix_hashes']==base['prefix_hashes']
            save_first(ROOT/'probes'/f'{tag}_s{step}_plus.npz',plus)
            save_first(ROOT/'probes'/f'{tag}_s{step}_minus.npz',minus)
            second=rig.central_second(plus,minus,step)
            fdq=(plus['logits'].astype(np.float64)-minus['logits'])/(2*step)
            diff=weighted_norm(base['logits'],fdq-first['q']);den=weighted_norm(base['logits'],first['q'])
            q_checks[step]={'weighted_difference':diff,'weighted_relative':diff/max(den,1e-30),
                'pass':bool(diff<=max(.02*den,1e-6))}
            second['head_order_weighted_relative']=weighted_norm(base['logits'],second['r']-second['r_head_first'])/max(weighted_norm(base['logits'],second['r']),1e-30)
            values[step]=second
        return {'first':first,'seconds':values,'q_checks':q_checks,'primal':primal}

    for question in QUESTIONS:
        if question['split']!='calibration':continue
        ids=rig.tokenizer.apply_chat_template([{'role':'system','content':INPUTS['system']},{'role':'user','content':question['text']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if hasattr(ids,'keys'):ids=ids['input_ids']
        if hasattr(ids,'tolist'):ids=ids.tolist()
        if ids and isinstance(ids[0],list):ids=ids[0]
        assert ids==question['prefix_ids']
        base=rig.value(ids);contexts[question['id']]=base
        np.savez_compressed(ROOT/'baseline'/f'{question["id"]}.npz',logits=base['logits'],**base['points'],prefix_ids=np.asarray(ids))
        for di,vector in enumerate(V):
            direction=torch.tensor(vector);tag=f'{question["id"]}_d{di}'
            exact=None
            if exact_available is not False:
                call_before=rig.calls;begin=time.monotonic()
                try:
                    exact=rig.exact_jet(ids,'input',direction)
                    np.savez_compressed(ROOT/'probes'/f'{tag}_exact_jet.npz',
                        logits=exact['logits'],q=exact['q'],r=exact['r'],
                        **{n+'_first':x for n,x in exact['first'].items()},
                        **{n+'_second':x for n,x in exact['second'].items()})
                    practical=exact['seconds']<=45
                    exact_attempts.append({'question':question['id'],'direction':di,'success':True,'practical':practical,
                        'seconds':exact['seconds'],'backbone_calls':rig.calls-call_before})
                    exact_available=practical
                    if not practical:exact=None
                except (NotImplementedError,RuntimeError) as error:
                    message=str(error).lower()
                    if 'memory' in message or not any(w in message for w in ['derivative','backward','forward','not implemented','grad','jvp']):raise
                    exact_attempts.append({'question':question['id'],'direction':di,'success':False,
                        'seconds':time.monotonic()-begin,'backbone_calls':rig.calls-call_before,
                        'error':type(error).__name__+': '+str(error)})
                    exact_available=False;gc.collect()
            entry=probes(question,di,'input',direction,STEPS,tag)
            entry['exact']=exact;raw[question['id'],di]=entry
            if exact is not None:
                check_primal(exact,base)
                qerr=weighted_norm(base['logits'],exact['q']-entry['first']['q'])/weighted_norm(base['logits'],entry['first']['q'])
                entry['exact_q_relative']=qerr
            report('calibrating_derivatives',question=question['id'],direction=di,model_calls=rig.calls,
                   exact_second_currently_available=exact_available)
    selected_pair=None
    for coarse,fine in zip(STEPS[:-1],STEPS[1:]):
        pair_checks=[]
        for (pid,di),entry in raw.items():
            c=consistency(contexts[pid]['logits'],entry['seconds'][coarse],entry['seconds'][fine])
            c.update(question=pid,direction=di,coarse=coarse,fine=fine,
                q_pass=entry['q_checks'][coarse]['pass'] and entry['q_checks'][fine]['pass'])
            pair_checks.append(c)
        numerics.append({'coarse':coarse,'fine':fine,'checks':pair_checks})
        if selected_pair is None and all(c['pass'] and c['q_pass'] for c in pair_checks):selected_pair=(coarse,fine)
    assert selected_pair is not None,'No declared calibration stencil pair passed; no target endpoints'
    coarse,fine=selected_pair
    exact_checks=[]
    exact_ok=exact_available is True and all(e['exact'] is not None for e in raw.values())
    if exact_ok:
        for (pid,di),entry in raw.items():
            c=consistency(contexts[pid]['logits'],entry['exact'],entry['seconds'][fine])
            c.update(question=pid,direction=di,q_relative=entry['exact_q_relative'])
            exact_checks.append(c)
        exact_ok=all(c['pass'] and c['q_relative']<=1e-4 for c in exact_checks)
    METHOD='exact_nested_directional_AD' if exact_ok else 'central_difference_of_forward_AD'
    selection={'method':METHOD,'coarse_step':coarse,'fine_step':fine,
        'calibration_questions':2,'calibration_base_direction_pairs':4,'target_endpoints_seen':0,
        'exact_attempts':exact_attempts,'stencil_checks':numerics,'exact_checks':exact_checks,
        'selected_before_held_derivative_probes_and_all_target_endpoints':True}
    dump(ROOT/'DERIVATIVE_SELECTION_LOCK.json',selection);selection_sha=sha(ROOT/'DERIVATIVE_SELECTION_LOCK.json')

    def finish_jet(question,di,entry,exact):
        pid=question['id'];base=contexts[pid]
        c=consistency(base['logits'],entry['seconds'][coarse],entry['seconds'][fine])
        assert c['pass'] and all(entry['q_checks'][s]['pass'] for s in [coarse,fine]),'Locked derivative consistency failed; no retuning'
        if METHOD=='exact_nested_directional_AD':
            assert exact is not None
            check_primal(exact,base)
            c_exact=consistency(base['logits'],exact,entry['seconds'][fine]);assert c_exact['pass']
            qerr=weighted_norm(base['logits'],exact['q']-entry['first']['q'])/weighted_norm(base['logits'],entry['first']['q'])
            assert qerr<=1e-4
            # Use first forward-AD derivative to preserve its independent validation.
            second=exact['second'];r=exact['r']
        else:
            second=entry['seconds'][fine]['second'];r=entry['seconds'][fine]['r'];c_exact=None
        result={'first':entry['first']['first'],'second':second,'q':entry['first']['q'],'r':r,
            'consistency':c,'exact_consistency':c_exact}
        return result

    for question in QUESTIONS:
        pid=question['id'];ids=question['prefix_ids']
        if question['split']=='held_out':
            base=rig.value(ids);contexts[pid]=base
            np.savez_compressed(ROOT/'baseline'/f'{pid}.npz',logits=base['logits'],**base['points'],prefix_ids=np.asarray(ids))
        for di,vector in enumerate(V):
            if question['split']=='calibration':entry=raw[pid,di];exact=entry['exact']
            else:
                direction=torch.tensor(vector);exact=rig.exact_jet(ids,'input',direction) if METHOD=='exact_nested_directional_AD' else None
                entry=probes(question,di,'input',direction,[coarse,fine],f'{pid}_d{di}')
            jet=finish_jet(question,di,entry,exact);jets[pid,di]=jet
            path=ROOT/'coefficients'/f'{pid}_d{di}.npz'
            np.savez_compressed(path,q=jet['q'],r=jet['r'],
                **{n+'_first':x for n,x in jet['first'].items()},**{n+'_second':x for n,x in jet['second'].items()})
            dump(path.with_suffix('.json'),{'method':METHOD,'coarse_step':coarse,'fine_step':fine,
                'consistency':jet['consistency'],'exact_consistency':jet['exact_consistency']})
            report('freezing_coefficients',question=pid,direction=di,model_calls=rig.calls,method=METHOD)
    del raw;gc.collect()

    alphas=[sign*s for s in PLAN['strengths'] for sign in PLAN['signs']]
    def forecasts(base,jet,values):
        q=jet['q'].astype(np.float64);r=jet['r'].astype(np.float64);l0=base['logits'].astype(np.float64)
        fisher_coefficient=weighted_norm(l0,q)**2
        rows=[]
        for alpha in values:
            l1=l0+alpha*q;l2=l1+.5*alpha*alpha*r
            rows.append({'alpha':alpha,'quadratic_Fisher':.5*alpha*alpha*fisher_coefficient,
                'first_order_logits':kl(l0,l1),'second_order_logits':kl(l0,l2)})
        return rows
    forecast_map={}
    for question in QUESTIONS:
        for di in range(2):
            pid=question['id'];f=forecasts(contexts[pid],jets[pid,di],alphas)
            forecast_map[pid,di]=f;dump(ROOT/'predictions'/f'{pid}_d{di}.json',f)
    forecast_files=list((ROOT/'coefficients').glob('*'))+list((ROOT/'predictions').glob('*'))
    freeze_predictions={p.relative_to(ROOT).as_posix():sha(p) for p in forecast_files}
    dump(ROOT/'PREDICTION_FREEZE.json',{'method':METHOD,'selection_sha256':selection_sha,'sha256':freeze_predictions,
        'primary_target_endpoints_seen':0,'all_forecasts_saved_before_primary_endpoints':True})
    prediction_freeze_sha=sha(ROOT/'PREDICTION_FREEZE.json')

    metrics=[];nulls=[]
    def evaluate_row(question,di,alpha,base,actual,jet,f):
        measured=kl(base['logits'],actual['logits']);resolved=measured>1e-10
        row={'question':question['id'],'split':question['split'],'family':question['family'],
            'direction':di,'alpha':alpha,'strength':abs(alpha),'actual_KL':measured,'resolved':resolved,
            'cost_predictions':{n:f[n] for n in ['quadratic_Fisher','first_order_logits','second_order_logits']},
            'cost_relative_errors':{n:abs(f[n]-measured)/measured if resolved else None for n in ['quadratic_Fisher','first_order_logits','second_order_logits']},
            'weighted_logit_errors':{},'raw_logit_errors':{},'hidden_errors':{},'distribution_KL_actual_to_forecast':{},'total_variation':{}}
        true_logits=actual['logits'].astype(np.float64)-base['logits']
        true_hidden=actual['points']['hidden'].astype(np.float64)-base['points']['hidden']
        pa,_=probability(actual['logits'])
        for name,order in [('first_order_logits',1),('second_order_logits',2)]:
            delta=alpha*jet['q'].astype(np.float64)
            hidden=alpha*jet['first']['hidden'].astype(np.float64)
            if order==2:
                delta+=.5*alpha*alpha*jet['r'];hidden+=.5*alpha*alpha*jet['second']['hidden']
            error=weighted_norm(base['logits'],delta-true_logits)
            norm=weighted_norm(base['logits'],true_logits)
            row['weighted_logit_errors'][name]=error/max(norm,1e-30)
            row['raw_logit_errors'][name]=relative(delta,true_logits)
            row['hidden_errors'][name]=relative(hidden,true_hidden)
            endpoint=base['logits'].astype(np.float64)+delta;pp,_=probability(endpoint)
            row['distribution_KL_actual_to_forecast'][name]=kl(actual['logits'],endpoint)
            row['total_variation'][name]=float(.5*np.abs(pa-pp).sum())
        return row
    for question in QUESTIONS:
        pid=question['id'];base=contexts[pid];ids=question['prefix_ids']
        null=rig.value(ids,'input',torch.tensor(V[0]),0.)
        assert np.array_equal(null['logits'],base['logits']) and null['prefix_hashes']==base['prefix_hashes']
        nulls.append({'question':pid,'exact':True,'KL':kl(base['logits'],null['logits'])})
        for di,vector in enumerate(V):
            storage={n:[] for n in NAMES+['logits']}
            for ai,alpha in enumerate(alphas):
                actual=rig.value(ids,'input',torch.tensor(vector),alpha)
                assert actual['prefix_hashes']==base['prefix_hashes']
                row=evaluate_row(question,di,alpha,base,actual,jets[pid,di],forecast_map[pid,di][ai])
                delivered=actual['points']['input'].astype(np.float64)-base['points']['input']
                projection=float(delivered@vector/np.dot(vector,vector))
                row.update(actual_input_norm=float(np.linalg.norm(delivered)),delivered_projection=projection,
                    off_direction_norm=float(np.linalg.norm(delivered-projection*vector)),zero_delivery=bool(np.linalg.norm(delivered)==0),earlier_prefix_exact=True)
                metrics.append(row);STATUS['primary_endpoints']+=1
                for n in NAMES:storage[n].append(actual['points'][n])
                storage['logits'].append(actual['logits'])
            np.savez_compressed(ROOT/'actual'/f'{pid}_d{di}.npz',alphas=np.asarray(alphas),**{n:np.stack(v) for n,v in storage.items()})
            dump(ROOT/'METRICS.json',metrics)
            report('primary_endpoints',question=pid,direction=di,model_calls=rig.calls,method=METHOD)
    dump(ROOT/'NULL_CHECKS.json',nulls)
    assert len(metrics)==96 and sha(ROOT/'PREDICTION_FREEZE.json')==prediction_freeze_sha
    assert sha(ROOT/'DERIVATIVE_SELECTION_LOCK.json')==selection_sha
    for name,expected in freeze_predictions.items():assert sha(ROOT/name)==expected

    post_admission={'remaining_calls':300-rig.calls,'remaining_native_seconds':2400-(time.monotonic()-START),'free_bytes':memory()}
    post_admission['admitted']=post_admission['remaining_calls']>=40 and post_admission['remaining_native_seconds']>=300 and post_admission['free_bytes']>=4294967296
    dump(ROOT/'POST_ADMISSION.json',post_admission)
    if post_admission['admitted']:
        post_jets={};post_forecasts={};chain=[];post_hashes={};post_values=[.125,-.125]
        selected=[q for q in QUESTIONS if q['id'] in PLAN['optional_post_block_check']['fixed_questions']]
        for question in selected:
            pid=question['id'];full=jets[pid,0];direction=torch.tensor(full['first']['output'],dtype=torch.float32)
            exact=rig.exact_jet(question['prefix_ids'],'output',direction) if METHOD=='exact_nested_directional_AD' else None
            entry=probes(question,0,'output',direction,[coarse,fine],f'post_{pid}')
            jet=finish_jet(question,0,entry,exact);post_jets[pid]=jet
            linear_second=rig.first_at(question['prefix_ids'],'output',torch.tensor(full['second']['output'],dtype=torch.float32))
            combined=jet['r'].astype(np.float64)+linear_second['q']
            rnorm=weighted_norm(contexts[pid]['logits'],full['r'])
            chain.append({'question':pid,'q_transport_relative':weighted_norm(contexts[pid]['logits'],jet['q']-full['q'])/max(weighted_norm(contexts[pid]['logits'],full['q']),1e-30),
                'second_chain_weighted_relative':weighted_norm(contexts[pid]['logits'],combined-full['r'])/max(rnorm,1e-30),
                'r_full_weighted_norm':rnorm,'r_tail_weighted_norm':weighted_norm(contexts[pid]['logits'],jet['r']),
                'upstream_curvature_transported_weighted_norm':weighted_norm(contexts[pid]['logits'],linear_second['q']),
                'not_additive_scalar_causal_percentages':True})
            path=ROOT/'post'/f'{pid}_coefficients.npz'
            np.savez_compressed(path,q=jet['q'],r=jet['r'],upstream_transported_r=linear_second['q'],
                direction=full['first']['output'],hidden_first=jet['first']['hidden'],hidden_second=jet['second']['hidden'])
            f=forecasts(contexts[pid],jet,post_values);post_forecasts[pid]=f
            dump(ROOT/'post'/f'{pid}_forecasts.json',f)
            report('post_coefficients',question=pid,model_calls=rig.calls)
        post_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'post').iterdir() if p.is_file()}
        dump(ROOT/'POST_PREDICTION_FREEZE.json',{'sha256':post_hashes,'post_endpoints_seen':0,'method':METHOD})
        dump(ROOT/'CURVATURE_CHAIN.json',chain)
        post_rows=[]
        for question in selected:
            pid=question['id'];base=contexts[pid];direction=torch.tensor(jets[pid,0]['first']['output'],dtype=torch.float32)
            null=rig.value(question['prefix_ids'],'output',direction,0.);assert np.array_equal(null['logits'],base['logits'])
            values=[]
            for ai,alpha in enumerate(post_values):
                actual=rig.value(question['prefix_ids'],'output',direction,alpha)
                assert actual['prefix_hashes']==base['prefix_hashes']
                post_rows.append(evaluate_row(question,0,alpha,base,actual,post_jets[pid],post_forecasts[pid][ai]))
                values.append(actual['logits']);STATUS['post_endpoints']+=1
            np.savez_compressed(ROOT/'post'/f'{pid}_actual.npz',alphas=np.asarray(post_values),logits=np.stack(values))
        dump(ROOT/'POST_METRICS.json',post_rows)
        for name,expected in post_hashes.items():assert sha(ROOT/name)==expected
        STATUS['post_check']='complete_separate_boundary_contract'
    else:STATUS['post_check']='unexecuted_resource_admission'
    assert all(sha(baseline_dir/name)==expected for name,expected in immutable.items())
    dump(ROOT/'BASELINE_IMMUTABILITY.json',{'unchanged':True,'archive_sha256':immutable,'old_monitor_unstarted':not stopped.exists()})
    assert not stopped.exists()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in rig.model.modules())
    STATUS['status']='native_complete';report('native_complete_export_pending',model_calls=rig.calls,method=METHOD)
except BaseException as error:
    if 'rig' in globals():STATUS['model_calls']=rig.calls
    STATUS['status']='fault_or_numerical_gate_stop';report('fault',error=type(error).__name__+': '+str(error))
    (ROOT/'FAULT.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    try:export()
    finally:TIMER.cancel()
