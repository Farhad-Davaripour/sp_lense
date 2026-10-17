"""Frozen native fresh-question response tests: remaining292 of phase17's300.

Model weights and decoder operations remain unchanged. Every prediction is
frozen before finite target endpoints. All failures and complete states remain.
"""
import os,time
BEGIN=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4')
import copy,gc,hashlib,json,shutil,threading,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;SOURCE=ROOT/'fresh_source';RUN=ROOT/'fresh';BASE=ROOT.parent
assert ROOT.name=='colab_cpu_20261014' and BASE.name=='fresh_memory_boundary_response_v1'
PRIOR=json.loads((ROOT/'FRESH_PRIOR_LEDGER.json').read_text())
assert PRIOR['phase17_prior_calls']==8 and PRIOR['phase17_remaining_calls']==292 and PRIOR['all_prior_native_preservation_complete']
START=BEGIN-PRIOR['phase17_prior_charged_seconds'];RUN.mkdir(exist_ok=False)
STATUS={'status':'starting','model_calls':8,'fresh_model_calls':0,'target_rows':0,'PID':os.getpid(),'GPU_allocations':0}
rig=None
def scalar(x):
    if hasattr(x,'item'):return x.item()
    raise TypeError(type(x).__name__)
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False,default=scalar)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def report(phase,**extra):
    STATUS.update(phase=phase,seconds=time.monotonic()-START,**extra)
    if rig is not None:STATUS.update(model_calls=rig.calls,fresh_model_calls=rig.calls-8)
    dump(RUN/'STATUS.json',STATUS);print(json.dumps(STATUS,default=scalar),flush=True)
def deadline():
    if time.monotonic()-START>2400:raise TimeoutError('Shared phase17 native2400-second budget')
    if rig is not None and rig.calls>=300:raise RuntimeError('Shared phase17 300-forward ceiling')
def hardstop():dump(RUN/'HARD_STOP.json',STATUS);os._exit(124)
timer=threading.Timer(max(1.,2700-PRIOR['phase17_prior_charged_seconds']),hardstop);timer.daemon=True;timer.start()

try:
    import numpy as np,torch,transformers
    from boundary_native import BoundaryRig
    from native import cache_hash
    from point_store import PointStore
    from response_math import flat_value,flat_slope,linear_combination,derivatives,first_second_validation,third_validation,plane_tensor,combine_plane,forecast,score,group_norm
    from maths import GROUPS,norm,wn,kl,probability,checks
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert (torch.__version__,transformers.__version__,np.__version__)==('2.13.0+cpu','5.19.0','2.5.4') and not torch.cuda.is_available()
    freeze=json.loads((ROOT/'FRESH_EXECUTION_FREEZE.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in freeze['sha256'].items())
    proof=json.loads((ROOT/'FRESH_PRIVATE_PROOF.json').read_text());assert proof['complete'] and proof['provider_metadata_verified']
    free=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    admission={'free_memory_after_imports':free,'minimum':8589934592,'disk_free':shutil.disk_usage(ROOT).free,'disk_minimum':17179869184}
    dump(RUN/'ADMISSION.json',admission);assert free>=admission['minimum'] and admission['disk_free']>=admission['disk_minimum']
    plan=json.loads((SOURCE/'PLAN.json').read_text());checkpoint=Path('/content/r4_checkpoint')
    pin=json.loads((SOURCE/'model_pin.json').read_text());verified=[]
    for item in pin['files']:
        p=checkpoint/item['name']
        if not p.is_file():continue
        assert p.stat().st_size==item['size']
        if item.get('sha256'):assert sha(p)==item['sha256']
        else:
            data=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['git_blob_id']
        verified.append(item['name'])
    assert len(verified)==8;dump(RUN/'MODEL_IDENTITY.json',{'revision':'2fc06364715b967f1860aea9cf38778875588b17','files':verified})
    report('loading_model');rig=BoundaryRig(checkpoint,deadline);rig.calls=8;rig.precision(torch.float32)
    questions=[]
    for q in plan['questions']:
        ids=rig.tokenizer.apply_chat_template([{'role':'system','content':'Give a concise answer.'},{'role':'user','content':q['text']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if hasattr(ids,'keys'):ids=ids['input_ids']
        if hasattr(ids,'tolist'):ids=ids.tolist()
        if ids and isinstance(ids[0],list):ids=ids[0]
        assert all(type(i)==int for i in ids) and 1<len(ids)<=128
        questions.append({**q,'prefix_ids':ids,'prefix_length':len(ids)})
    dump(RUN/'INPUTS.json',{'questions':questions,'system':'Give a concise answer.','phase17_calls_at_input_freeze':rig.calls,'fresh_calls':0})
    input_sha=sha(RUN/'INPUTS.json')
    with np.load(ROOT/'DIRECTIONS_FRESH.npz',allow_pickle=False) as z:V=z['vectors'].copy();Gram=z['gram'].copy()
    assert V.shape==(2,1024) and np.allclose(V@V.T,np.eye(2),atol=2e-6)
    store=PointStore(RUN)
    for n in ['contexts','probes','coefficients','predictions','actual','chains']:(RUN/n).mkdir()
    CTX1={};CTX2={};M1={};BASELINES={};SCHEMAS={};MEMORY=[];CARRY=[];CAUSAL=[];JETS={};META={};SETUP=[];TIMINGS=[];COEF_TIMES=[];CONTEXT_TIMES=[]
    ALPHAS=[.03125,-.03125,.125,-.125]
    def base_name(pid,t):return f'contexts/{pid}_t{t}_baseline.npz'
    def first_mode(pid,site=('input',11)):
        rig.ctx=CTX1[pid];rig.site=site;rig.second_step=False;rig.capture_complete_cache=False;rig.names=None;rig.schema=None
    def second_mode(pid,cache=None):
        rig.ctx=dict(CTX2[pid]);rig.site=('input',11);rig.second_step=True;rig.capture_complete_cache=False;rig.names=None;rig.schema=None
        if cache is not None:rig.ctx.update(cache=cache,hash=cache_hash(cache))
    def primal_ok(point,base):
        err=max(norm(point[n].astype(np.float64)-base[n])/max(1.,norm(base[n])) for n in base)
        assert err<=1e-6;return err
    def causal(pid,t,point,direction=None,alpha=0.,site=('input',11)):
        baseline=BASELINES[pid,t];kind,layer=site
        # Prefix KV is verified by the native collector against immutable incoming cache.
        unchanged=[f'b{i}_{s}' for i in range(layer) for s in ['input','middle','mixer','mlp','output']]
        if t==2 or kind=='middle':unchanged.append(f'b{layer}_input')
        if kind=='middle':unchanged.append(f'b{layer}_mixer')
        assert all(np.array_equal(point['values'][n],baseline[n]) for n in unchanged)
        frontier=layer+(kind=='middle')
        upstream=[n for n,s in SCHEMAS[t].items() if n.startswith('m') and s['layer']<frontier]
        assert all(np.array_equal(point['values'][n],baseline[n]) for n in upstream)
        delivered=None
        if t==1 and direction is not None:
            n=f'b{layer}_input' if kind=='input' else f'b{layer}_middle'
            delivered=point['values'][n].astype(np.float64)-baseline[n]
            expected=alpha*np.asarray(direction,dtype=np.float64)
            floor=20*np.finfo(np.float32).eps*max(1.,norm(baseline[n]))
            assert norm(delivered-expected)<=max(1e-6*max(1.,norm(expected)),floor)
        CAUSAL.append({'question':pid,'token':t,'site':list(site),'alpha':alpha,'upstream_activation_fields':len(unchanged),
                       'upstream_memory_fields':len(upstream),'pass':True,'delivered_norm':None if delivered is None else norm(delivered)})
        return None if delivered is None else norm(delivered)
    for q in questions:
        context_begin=time.perf_counter();context_calls=rig.calls
        pid=q['id'];rig.site=('input',11);rig.timing_label='prefill';CTX1[pid]=rig.prepare_context(q)
        first_mode(pid);rig.capture_complete_cache=True;rig.timing_label='normal_first';p1=rig.value()
        if 1 not in SCHEMAS:SCHEMAS[1]=copy.deepcopy(rig.schema)
        else:assert SCHEMAS[1]==rig.schema
        BASELINES[pid,1]=flat_value(p1);store.value(base_name(pid,1),p1)
        M1[pid]=copy.deepcopy(rig.complete_out_cache)
        candidate=int(np.argmax(p1['logits']));eos={248044,248046,int(rig.tokenizer.eos_token_id)};fallback=candidate in eos;token=760 if fallback else candidate
        CTX2[pid]={'question':q,'cache':M1[pid],'hash':cache_hash(M1[pid]),'token':token,'normal_candidate':candidate,
                  'fallback':fallback,'length':q['prefix_length']+2,'base_invariants':None}
        second_mode(pid);rig.timing_label='normal_second';p2=rig.value()
        if 2 not in SCHEMAS:SCHEMAS[2]=copy.deepcopy(rig.schema)
        else:assert SCHEMAS[2]==rig.schema
        BASELINES[pid,2]=flat_value(p2);store.value(base_name(pid,2),p2)
        torch.save(CTX1[pid]['cache'],RUN/'contexts'/f'{pid}_incoming_cache.pt');torch.save(M1[pid],RUN/'contexts'/f'{pid}_normal_M1.pt')
        dump(RUN/'contexts'/f'{pid}.json',{'token1_normal_candidate':CTX1[pid]['normal_candidate'],'token1':CTX1[pid]['token'],
            'token1_fallback':CTX1[pid]['fallback'],'token2_normal_candidate':candidate,'token2':token,'token2_fallback':fallback,
            'token1_text':rig.tokenizer.decode([CTX1[pid]['token']]),'token2_text':rig.tokenizer.decode([token]),'EOS_ids':sorted(eos),
            'incoming_cache_hash':CTX1[pid]['hash'],'normal_M1_hash':CTX2[pid]['hash']})
        CONTEXT_TIMES.append({'question':pid,'calls':rig.calls-context_calls,'seconds':time.perf_counter()-context_begin})
        report('normal_context',question=pid)
    MEMORY=sorted(n for n in SCHEMAS[1] if n.startswith('m'));assert len(MEMORY)==48
    for t in [1,2]:dump(RUN/f'STATE_SCHEMA_T{t}.json',SCHEMAS[t])
    def paired_probe(pid,direction,alpha,tag,site=('input',11),paired=True,inferred=None):
        first_mode(pid,site);rig.timing_label='derivative_T1'
        p1=rig.first_at(torch.tensor(direction),alpha);causal(pid,1,p1,direction,alpha,site)
        zero=f'probes/{tag}_t1_q0.npz';path=zero if alpha==0 else f'probes/{tag}_t1_a{alpha}.npz'
        store.probe(path,p1,base_name(pid,1),None if alpha==0 else zero)
        if paired:
            cache=rig.assemble_memory(M1[pid],{n:p1['values'][n] for n in MEMORY})
            second_mode(pid,cache);rig.timing_label='derivative_T2';p2=rig.memory_first({n:p1['first'][n] for n in MEMORY})
            causal(pid,2,p2)
            prev=[n for n in SCHEMAS[2] if n.endswith('_previous1')]
            assert len(prev)==12 and np.count_nonzero(p2['first']['b11_input'])==0
            for n in prev:
                current=n.replace('_previous1','_current')
                assert np.array_equal(p2['values'][n],p1['values'][current]) and np.array_equal(p2['first'][n],p1['first'][current])
            CARRY.append({'question':pid,'tag':tag,'alpha':alpha,'previous_KV_fields_exact':12,'b11_input_tangent_zero':True})
            zero2=f'probes/{tag}_t2_q0.npz';path2=zero2 if alpha==0 else f'probes/{tag}_t2_a{alpha}.npz'
            store.probe(path2,p2,base_name(pid,2),None if alpha==0 else zero2)
        return None
    def inferred_zero(pid,tag,basis0,basis1,a,b):
        for t in [1,2]:
            p0=store.point(f'probes/{basis0}_t{t}_q0.npz');p1=store.point(f'probes/{basis1}_t{t}_q0.npz')
            slope=linear_combination((a,flat_slope(p0)),(b,flat_slope(p1)))
            point={'values':{n:BASELINES[pid,t][n] for n in SCHEMAS[t]},'logits':BASELINES[pid,t]['logits'],
                   'first':{n:slope[n].astype(np.float32) for n in SCHEMAS[t]},'q':slope['logits'].astype(np.float32)}
            store.probe(f'probes/{tag}_t{t}_q0.npz',point,base_name(pid,t))
    def sample(pid,direction,tag,steps,site=('input',11),paired=True,inferred=False):
        start=time.perf_counter();calls=rig.calls
        if not inferred:paired_probe(pid,direction,0.,tag,site,paired)
        for step in steps:
            for sign in [1,-1]:paired_probe(pid,direction,sign*step,tag,site,paired)
        SETUP.append({'tag':tag,'calls':rig.calls-calls,'seconds_including_both_steps_and_storage':time.perf_counter()-start})
        report('sampled_derivatives',tag=tag)
    def calculate(pid,tag,t,coarse,fine,save=True):
        coefficient_begin=time.perf_counter()
        zero=store.point(f'probes/{tag}_t{t}_q0.npz');base=BASELINES[pid,t];primal=primal_ok(flat_value(zero),base)
        q=flat_slope(zero);fds={}
        for step in [coarse,fine]:
            p=store.point(f'probes/{tag}_t{t}_a{step}.npz');m=store.point(f'probes/{tag}_t{t}_a{-step}.npz')
            fds[step]=derivatives(zero,p,m,step)
        jet={'q':q,'r':fds[fine]['r'],'t':fds[fine]['t']}
        validation=first_second_validation(base,q,fds[coarse],fds[fine],SCHEMAS[t])
        third=third_validation(base,q,jet['r'],fds[coarse]['t'],jet['t'],SCHEMAS[t])
        meta={'question':pid,'tag':tag,'token':t,'primal_relative_error':primal,'validation':validation,'third_validation':third,
              'validated':validation['pass'],'third_validated':third['pass'],'coarse':coarse,'fine':fine,
              'zero_slope_source':'basis linearity' if tag.endswith(('_sum','_difference')) else 'native forward AD'}
        if save:
            name=f'coefficients/{tag}_t{t}.npz';store.jet(name,jet);JETS[tag,t]=name;META[tag,t]=meta
            dump(RUN/'coefficients'/f'{tag}_t{t}.json',meta)
        COEF_TIMES.append({'tag':tag,'token':t,'saved':save,'seconds':time.perf_counter()-coefficient_begin})
        return jet,meta
    # All calibration slopes precede numerical selection and held finite targets.
    steps=plan['derivatives']['calibration_steps'];cal=[q for q in questions if q['split']=='calibration'];held=[q for q in questions if q['split']=='held_out']
    for q in cal:sample(q['id'],V[0],q['id']+'_primary',steps)
    candidates=[];selection=None
    for coarse,fine in zip(steps[:-1],steps[1:]):
        rows=[]
        for q in cal:
            for t in [1,2]:
                _,meta=calculate(q['id'],q['id']+'_primary',t,coarse,fine,False);rows.append(meta)
        passed=all(r['validated'] for r in rows);candidates.append({'coarse':coarse,'fine':fine,'pass':passed,'tests':rows})
        if selection is None and passed:selection={'coarse':coarse,'fine':fine}
    dump(RUN/'CALIBRATION_NUMERICS.json',candidates)
    assert selection is not None,'No predeclared coarse/fine pair validated; preserve calibration failure without choosing on held endpoints'
    coarse,fine=selection['coarse'],selection['fine'];selection.update(calls=rig.calls,target_rows_seen=0,calibration_only=True)
    dump(RUN/'NUMERICAL_SELECTION_LOCK.json',selection)
    for q in questions:
        pid=q['id'];tag=pid+'_primary'
        if q['split']=='held_out':sample(pid,V[0],tag,[coarse,fine])
        for t in [1,2]:calculate(pid,tag,t,coarse,fine)
    assert rig.calls==94
    # Direct chain components use synthetic straight-memory neighborhoods only.
    CHAINS=[]
    for pid in ['r17_cal_spoon','r17_held_panels']:
        tag=pid+'_primary';j1=store.load_jet(JETS[tag,1]);j2=store.load_jet(JETS[tag,2])
        seed={n:np.asarray(j1['q'][n],dtype=np.float32).copy() for n in MEMORY}
        rseed={n:np.asarray(j1['r'][n],dtype=np.float32).copy() for n in MEMORY}
        second_mode(pid);rig.timing_label='chain_J_r1';jr=rig.memory_first(rseed)
        store.probe(f'chains/{pid}_J_r1.npz',jr,base_name(pid,2));Hs={}
        for step in [coarse,fine]:
            pair=[]
            for sign in [1,-1]:
                shifted={n:(BASELINES[pid,1][n].astype(np.float64)+sign*step*seed[n]).astype(np.float32) for n in MEMORY}
                second_mode(pid,rig.assemble_memory(M1[pid],shifted));rig.timing_label='chain_H_q1_q1';point=rig.memory_first(seed)
                store.probe(f'chains/{pid}_H_s{step}_{sign}.npz',point,base_name(pid,2));pair.append(point)
            Hs[step]=linear_combination((1/(2*step),flat_slope(pair[0])),(-1/(2*step),flat_slope(pair[1])))
        total=linear_combination((1,flat_slope(jr)),(1,Hs[fine]))
        groups=checks({n:total[n] for n in SCHEMAS[2]},{n:j2['r'][n] for n in SCHEMAS[2]},SCHEMAS[2],.10)
        den=wn(BASELINES[pid,2]['logits'],j2['r']['logits']);en=wn(BASELINES[pid,2]['logits'],total['logits']-j2['r']['logits'])
        hgroups=checks({n:Hs[coarse][n] for n in SCHEMAS[2]},{n:Hs[fine][n] for n in SCHEMAS[2]},SCHEMAS[2],.10)
        chain={'question':pid,'groups':groups,'H_coarse_fine':hgroups,'weighted_relative':en/max(den,1e-30),
            'pass':bool(all(v['pass'] for v in groups.values()) and all(v['pass'] for v in hgroups.values()) and en<=max(.10*den,1e-6)),
            'component_norms':{g:{'transported':group_norm(flat_slope(jr),SCHEMAS[2],g),'new_nonlinearity':group_norm(Hs[fine],SCHEMAS[2],g),
                                 'total':group_norm(total,SCHEMAS[2],g)} for g in GROUPS}}
        CHAINS.append(chain);dump(RUN/'CHAIN_CONTROLS.json',CHAINS);report('chain_control',question=pid,passed=chain['pass'])
        del j1,j2,jr,Hs,total,seed,rseed,pair,point;gc.collect()
    assert rig.calls==104
    ALT_SITES={'early':('input',3),'middle':('middle',11),'late':('input',19)}
    alt_pids=[q['id'] for q in cal]+['r17_held_panels','r17_held_cupboard']
    for site_name,site in ALT_SITES.items():
        for pid in alt_pids:
            tag=pid+'_'+site_name;sample(pid,V[0],tag,[coarse,fine],site,False);calculate(pid,tag,1,coarse,fine)
    assert rig.calls==164
    plane_pids=['r17_cal_spoon','r17_held_panels']
    for pid in plane_pids:
        tag=pid+'_reference';sample(pid,V[1],tag,[coarse,fine])
        for t in [1,2]:calculate(pid,tag,t,coarse,fine)
    assert rig.calls==184
    for label,a,b in [('sum',1/np.sqrt(2),1/np.sqrt(2)),('difference',1/np.sqrt(2),-1/np.sqrt(2))]:
        for pid in plane_pids:
            tag=pid+'_'+label;inferred_zero(pid,tag,pid+'_primary',pid+'_reference',a,b)
            sample(pid,a*V[0]+b*V[1],tag,[coarse,fine],inferred=True)
            for t in [1,2]:calculate(pid,tag,t,coarse,fine)
    assert rig.calls==216
    # Complete quadratic AND cubic plane, including previously unprobed axes.
    MIXES={'sum':(1/np.sqrt(2),1/np.sqrt(2)),'difference':(1/np.sqrt(2),-1/np.sqrt(2)),
           'weighted01':(2/np.sqrt(5),1/np.sqrt(5)),'weighted10':(1/np.sqrt(5),2/np.sqrt(5))}
    PLANE=[]
    for pid in plane_pids:
        for t in [1,2]:
            basis=[store.load_jet(JETS[pid+'_'+k,t]) for k in ['primary','reference','sum','difference']]
            plane=plane_tensor(*basis);store.write(f'coefficients/{pid}_plane_t{t}.npz',{k+':'+n:v for k,x in plane.items() for n,v in x.items()})
            # Independently assembled coarse third tensor checks every mixed contribution.
            coarse_basis=[]
            for k in ['primary','reference','sum','difference']:
                z=store.point(f'probes/{pid}_{k}_t{t}_q0.npz');p=store.point(f'probes/{pid}_{k}_t{t}_a{coarse}.npz');m=store.point(f'probes/{pid}_{k}_t{t}_a{-coarse}.npz')
                d=derivatives(z,p,m,coarse);coarse_basis.append({'q':flat_slope(z),'r':d['r'],'t':d['t']})
            coarse_plane=plane_tensor(*coarse_basis)
            for label,(a,b) in MIXES.items():
                tag=pid+'_mix_'+label;j=combine_plane(basis[0],basis[1],plane,a,b);diag=combine_plane(basis[0],basis[1],plane,a,b,True)
                jc=combine_plane(coarse_basis[0],coarse_basis[1],coarse_plane,a,b)
                third=third_validation(BASELINES[pid,t],j['q'],j['r'],jc['t'],j['t'],SCHEMAS[t],(.125,-.125))
                valid=all(META[pid+'_'+k,t]['validated'] for k in ['primary','reference','sum','difference'])
                meta={'question':pid,'token':t,'tag':tag,'validated':valid,
                      'third_validated':bool(third['pass'] and all(META[pid+'_'+k,t]['third_validated'] for k in ['primary','reference','sum','difference'])),
                      'third_validation':third,
                      'basis_validation':{k:META[pid+'_'+k,t]['validated'] for k in ['primary','reference','sum','difference']},'target_direction_differentiated':label in ['sum','difference']}
                JETS[tag,t]=f'coefficients/{tag}_t{t}.npz';META[tag,t]=meta;store.jet(JETS[tag,t],j)
                store.jet(f'coefficients/{tag}_t{t}_diagonal.npz',diag);dump(RUN/'coefficients'/f'{tag}_t{t}.json',meta)
            Q=np.stack([basis[0]['q']['logits'],basis[1]['q']['logits']],axis=1).astype(np.float64);p,_=probability(BASELINES[pid,t]['logits']);Q-=p@Q
            G=Q.T@(p[:,None]*Q);d,U=np.linalg.eigh(Gram);B=(U/np.sqrt(d))@U.T;lam,axis=np.linalg.eigh(B@G@B)
            PLANE.append({'question':pid,'token':t,'Fisher_metric':G.tolist(),'basis_gram':Gram.tolist(),'generalized_eigenvalues':lam.tolist(),
                          'anisotropy':float(lam[-1]/lam[0]) if lam[0]>0 else None,'high_cost_axis_coordinates':(B@axis[:,-1]).tolist(),'full_space_spectrum':False})
            del basis,coarse_basis,plane,coarse_plane,j,jc,diag,z,p,m,d,Q;gc.collect()
    dump(RUN/'SAMPLED_PLANE_GEOMETRY.json',PLANE)
    # Each site/position has its own cubic calibration lock, before held targets.
    LOCK={}
    for label in ['primary','reference',*ALT_SITES]:
        tokens=[1] if label in ALT_SITES else [1,2]
        for t in tokens:
            selected=[META[q['id']+'_'+label,t] for q in cal if (q['id']+'_'+label,t) in META]
            LOCK[label,t]=bool(selected and all(m['validated'] and m['third_validated'] for m in selected))
    dump(RUN/'CUBIC_CALIBRATION_LOCKS.json',{f'{a}_t{b}':v for (a,b),v in LOCK.items()})
    TARGETS=[]
    def add_targets(pid,label,direction,site,ts,alphas,category):
        for alpha in alphas:
            for t in ts:TARGETS.append({'question':pid,'tag':pid+'_'+label,'label':label,'direction':direction.tolist(),'site':list(site),
                'token':t,'alpha':alpha,'strength':abs(alpha),'category':category})
    for q in held:add_targets(q['id'],'primary',V[0],('input',11),[1,2],ALPHAS,'primary')
    for label,site in ALT_SITES.items():
        for pid in ['r17_held_panels','r17_held_cupboard']:add_targets(pid,label,V[0],site,[1],ALPHAS,'alternate_boundary')
    add_targets('r17_held_panels','reference',V[1],('input',11),[1,2],ALPHAS,'reference')
    for label,(a,b) in MIXES.items():add_targets('r17_held_panels','mix_'+label,a*V[0]+b*V[1],('input',11),[1,2],[.125,-.125],'mixed')
    assert len(TARGETS)==80
    assert {k:sum(r['category']==k for r in TARGETS) for k in ['primary','alternate_boundary','reference','mixed']}=={'primary':32,'alternate_boundary':24,'reference':8,'mixed':16}
    PREDICTIONS=[]
    def digest(delta):
        h=hashlib.sha256()
        for n in sorted(delta):h.update(n.encode()+b'\0');h.update(np.ascontiguousarray(delta[n]).tobytes())
        return h.hexdigest()
    for target in TARGETS:
        pid,tag,t,alpha=target['question'],target['tag'],target['token'],target['alpha'];j=store.load_jet(JETS[tag,t]);base=BASELINES[pid,t]
        row={**target,'orders':{}}
        for order in [1,2,3]:
            start=time.perf_counter();pred,delta=forecast(base,j,alpha,order);p,_=probability(pred['logits']);elapsed=time.perf_counter()-start
            row['orders'][str(order)]={'cost':kl(base['logits'],pred['logits']),'full_delta_sha256':digest(delta),'probability_sha256':hashlib.sha256(p.tobytes()).hexdigest()}
            TIMINGS.append({'category':target['category'],'token':t,'order':order,'seconds_full_state_logits_softmax':elapsed})
        if target['category']=='mixed':
            dj=store.load_jet(f'coefficients/{tag}_t{t}_diagonal.npz');row['diagonal']={}
            for order in [2,3]:
                pred,delta=forecast(base,dj,alpha,order);row['diagonal'][str(order)]={'cost':kl(base['logits'],pred['logits']),'full_delta_sha256':digest(delta)}
        PREDICTIONS.append(row)
        del j,pred,delta,p;gc.collect()
    dump(RUN/'PREDICTIONS.json',PREDICTIONS);dump(RUN/'TARGET_ALLOCATION.json',TARGETS)
    fixed_files=[p for p in RUN.rglob('*') if p.is_file() and (p.parent.name in ['coefficients','contexts'] or p.name in ['PREDICTIONS.json','TARGET_ALLOCATION.json','NUMERICAL_SELECTION_LOCK.json','CUBIC_CALIBRATION_LOCKS.json','INPUTS.json'])]
    prediction_freeze={'calls_at_freeze':rig.calls,'target_rows_seen':0,'all_predictions_precede_targets':True,
                       'sha256':{p.relative_to(RUN).as_posix():sha(p) for p in fixed_files}}
    dump(RUN/'PREDICTION_FREEZE.json',prediction_freeze);prediction_freeze_sha=sha(RUN/'PREDICTION_FREEZE.json')
    # Four strict null calls on two held contexts, including the complete cache.
    NULLS=[]
    for pid in ['r17_held_panels','r17_held_cupboard']:
        first_mode(pid);rig.capture_complete_cache=True;rig.timing_label='strict_null';n1=rig.value(torch.tensor(V[0]),0.)
        assert all(np.array_equal(v,BASELINES[pid,1][n]) for n,v in flat_value(n1).items()) and cache_hash(rig.complete_out_cache)==CTX2[pid]['hash']
        second_mode(pid,copy.deepcopy(rig.complete_out_cache));rig.timing_label='strict_null';n2=rig.value()
        assert all(np.array_equal(v,BASELINES[pid,2][n]) for n,v in flat_value(n2).items())
        NULLS.append({'question':pid,'exact_T1':True,'whole_cache_exact':True,'exact_T2':True})
    assert rig.calls==220;dump(RUN/'NULL_CONTROLS.json',NULLS)
    # Actual endpoints are paired in the declared first-then-second order.
    METRICS=[];outgoing=None;last=None;incoming_norm=None
    for index,target in enumerate(TARGETS):
        pid,tag,t,alpha=target['question'],target['tag'],target['token'],target['alpha'];site=tuple(target['site']);direction=np.array(target['direction'],dtype=np.float32)
        key=(pid,tag,alpha)
        if t==1:
            first_mode(pid,site);rig.capture_complete_cache=True;rig.timing_label='held_actual_T1';actual=rig.value(torch.tensor(direction),alpha)
            outgoing=rig.complete_out_cache;last=key
            incoming_norm=float(np.sqrt(sum(norm(actual['values'][n].astype(np.float64)-BASELINES[pid,1][n])**2 for n in MEMORY)))
            branch=int(np.argmax(actual['logits']));base_token=CTX2[pid]['normal_candidate']
        else:
            assert key==last and outgoing is not None;second_mode(pid,outgoing);rig.timing_label='held_actual_T2';actual=rig.value()
        delivered=causal(pid,t,actual,direction if t==1 else None,alpha,site)
        actual_name=f'actual/endpoint_{index:03d}.npz';store.value(actual_name,actual,base_name(pid,t))
        j=store.load_jet(JETS[tag,t]);row={**target,**score(BASELINES[pid,t],flat_value(actual),j,alpha,SCHEMAS[t]),
            'actual_file':actual_name,'coefficient_file':JETS[tag,t],'derivative_validated':META[tag,t]['validated'],
            'third_validated':META[tag,t]['third_validated'],'incoming_memory_change_norm':incoming_norm,
            'delivered_perturbation_norm':delivered if t==1 else incoming_norm,
            'normal_second_candidate':base_token,'controlled_second_token':CTX2[pid]['token'],
            'steered_first_argmax':branch,'argmax_branch_changed':bool(branch!=base_token)}
        if target['category']=='mixed':
            djname=f'coefficients/{tag}_t{t}_diagonal.npz';dj=store.load_jet(djname)
            row['diagonal']=score(BASELINES[pid,t],flat_value(actual),dj,alpha,SCHEMAS[t])['orders'];row['diagonal_coefficient_file']=djname
        METRICS.append(row);STATUS['target_rows']=len(METRICS);dump(RUN/'METRICS.json',METRICS)
        report('held_endpoint',question=pid,category=target['category'],token=t)
        if t==2:outgoing=None;last=None
        del actual,j;gc.collect()
    assert rig.calls==300 and len(METRICS)==80 and STATUS['fresh_model_calls']==292
    assert all(sha(RUN/n)==h for n,h in prediction_freeze['sha256'].items()) and sha(RUN/'PREDICTION_FREEZE.json')==prediction_freeze_sha
    assert sha(RUN/'INPUTS.json')==input_sha and all(sha(ROOT/n)==h for n,h in freeze['sha256'].items())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in rig.model.modules())
    dump(RUN/'MEMORY_CARRY_CHECKS.json',CARRY);dump(RUN/'CAUSAL_DELIVERY_CHECKS.json',CAUSAL)
    dump(RUN/'STORAGE_RECEIPTS.json',store.receipts);dump(RUN/'JET_SETUP_TIMINGS.json',SETUP);dump(RUN/'FORECAST_TIMINGS.json',TIMINGS)
    dump(RUN/'COEFFICIENT_TIMINGS.json',COEF_TIMES);dump(RUN/'CONTEXT_SETUP_TIMINGS.json',CONTEXT_TIMES)
    dump(RUN/'CALL_LEDGER.json',{'historical_repair':8,'fresh':292,'combined':300,'ceiling':300,'listed_cumulative':1438})
    STATUS['status']='native_complete';report('native_complete_export_pending')
except BaseException as e:
    STATUS['status']='fault_or_calibration_stop';report('fault',error=type(e).__name__+': '+str(e))
    (RUN/'FAULT.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    for variable,filename in [('CARRY','MEMORY_CARRY_CHECKS.json'),('CAUSAL','CAUSAL_DELIVERY_CHECKS.json'),
                              ('SETUP','JET_SETUP_TIMINGS.json'),('TIMINGS','FORECAST_TIMINGS.json'),
                              ('COEF_TIMES','COEFFICIENT_TIMINGS.json'),('CONTEXT_TIMES','CONTEXT_SETUP_TIMINGS.json')]:
        if variable in globals():dump(RUN/filename,globals()[variable])
    if 'store' in globals():dump(RUN/'STORAGE_RECEIPTS.json',store.receipts)
    if rig is not None:
        dump(RUN/'CALL_TIMINGS.json',rig.call_timings);STATUS['model_calls']=rig.calls;STATUS['fresh_model_calls']=rig.calls-8;del rig;rig=None;gc.collect()
    files=sorted([p for p in RUN.rglob('*') if p.is_file() and p.name!='EXPORT_HASHES.json']+[p for p in SOURCE.iterdir() if p.is_file()])
    files+=[ROOT/n for n in ['FRESH_EXECUTION_FREEZE.json','FRESH_PRIOR_LEDGER.json','FRESH_PRIVATE_PROOF.json','DIRECTIONS_FRESH.npz','DIRECTION_PROVENANCE.json']]
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files};dump(RUN/'EXPORT_HASHES.json',manifest)
    archive=BASE/'R4_FRESH_MEMORY_BOUNDARY_NATIVE_20261014.zip'
    with zipfile.ZipFile(archive,'x') as z:
        for p in files+[RUN/'EXPORT_HASHES.json']:z.write(p,p.relative_to(ROOT).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix=='.npz' else zipfile.ZIP_DEFLATED,compresslevel=2)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items());members=len(z.namelist())
    receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'members':members,
        'model_calls':STATUS['model_calls'],'fresh_calls':STATUS['fresh_model_calls'],'shared_native_export_seconds':time.monotonic()-START,
        'prior_charged_seconds':PRIOR['phase17_prior_charged_seconds'],'status':STATUS['status'],'model_worker_exited':True,'GPU_allocations':0}
    dump(BASE/'FRESH_NATIVE_EXPORT_RECEIPT_20261014.json',receipt);timer.cancel();print('R4_FRESH_NATIVE_EXPORT',json.dumps(receipt),flush=True)
