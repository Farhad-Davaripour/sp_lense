"""Actual frozen two-step state response; no steering is applied at step two."""
import os,time
START=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4')
import copy,gc,hashlib,json,shutil,threading,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;SOURCE=ROOT/'source';BASE=ROOT.parent
assert ROOT.name=='colab_cpu_20261014' and BASE.name=='cached_memory_transport_v1'
PLAN=json.loads((SOURCE/'PLAN.json').read_text());PRIOR_CALLS=PLAN['parent_calls'];PRIOR_SECONDS=PLAN['parent_seconds'];START-=PRIOR_SECONDS
PARENT=Path(PLAN['parent_root']);USED={}
STATUS={'status':'starting','PID':os.getpid(),'model_calls':PRIOR_CALLS,'new_calls':0,'main_endpoints':0,'mixed_endpoints':0,'GPU_allocations':0,'Colab':True}
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
    if 'rig' in globals():STATUS.update(model_calls=rig.calls,new_calls=rig.calls-PRIOR_CALLS)
    dump(ROOT/'STATUS.json',STATUS);print(json.dumps(STATUS,default=scalar),flush=True)
def deadline():
    if time.monotonic()-START>2400:raise TimeoutError('Shared2400 native-second ceiling')
    if 'rig' in globals() and rig.calls>=300:raise RuntimeError('Shared300-forward ceiling')
def free_memory():return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
def hardstop():dump(ROOT/'HARD_STOP.json',STATUS);os._exit(124)
timer=threading.Timer(max(1.,2700-PRIOR_SECONDS),hardstop);timer.daemon=True;timer.start()
def export():
    if 'PARENT_MANIFEST' in globals():
        assert all(sha(PARENT/n)==h for n,h in USED.items()),'Parent data changed'
        dump(ROOT/'PARENT_USED_FILES.json',{'sha256':USED,'parent_unchanged':True,'parent_archive_sha256':PLAN['parent_archive_sha256']})
    if 'rig' in globals():dump(ROOT/'CALL_TIMINGS.json',rig.call_timings)
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='EXPORT_HASHES.json')
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files};dump(ROOT/'EXPORT_HASHES.json',manifest)
    archive=BASE/'R4_MEMORY_TRANSPORT_NATIVE_20261014.zip'
    with zipfile.ZipFile(archive,'x') as z:
        for p in files+[ROOT/'EXPORT_HASHES.json']:
            z.write(p,p.relative_to(ROOT).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix=='.npz' else zipfile.ZIP_DEFLATED,compresslevel=2)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items());members=len(z.namelist())
    receipt={'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'members':members,'model_calls':STATUS['model_calls'],
        'parent_calls':PRIOR_CALLS,'new_calls':STATUS['model_calls']-PRIOR_CALLS,'status':STATUS['status'],
        'shared_native_export_seconds':time.monotonic()-START,'parent_native_export_seconds':PRIOR_SECONDS,
        'all_members_CRC_and_manifest_verified':True,'GPU_allocations':0,'Colab':True,'listed_prior_calls':1004,
        'listed_cumulative_calls':1004+STATUS['model_calls']-PRIOR_CALLS}
    dump(BASE/'NATIVE_EXPORT_RECEIPT_20261014.json',receipt);print('R4_MEMORY_EXPORT',json.dumps(receipt),flush=True)

try:
    import numpy as np,torch,transformers
    from native import cache_hash
    from transport_native import MemoryTransportRig
    from maths import GROUPS,probability,kl,wn,norm,group_norm,subtract,central,checks,evaluate
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert (torch.__version__,transformers.__version__,np.__version__)==('2.13.0+cpu','5.19.0','2.5.4') and not torch.cuda.is_available()
    freeze=json.loads((ROOT/'EXECUTION_FREEZE.json').read_text());assert all(sha(ROOT/n)==h for n,h in freeze['sha256'].items())
    admission={'free_memory_after_imports':free_memory(),'minimum':8589934592,'disk_free':shutil.disk_usage(ROOT).free,'disk_minimum':17179869184}
    dump(ROOT/'ADMISSION.json',admission);assert admission['free_memory_after_imports']>=admission['minimum'] and admission['disk_free']>=admission['disk_minimum']
    PARENT_MANIFEST=json.loads((PARENT/'EXPORT_HASHES.json').read_text())
    identity=json.loads((ROOT/'PARENT_IDENTITY.json').read_text())
    assert hashlib.sha256(json.dumps(PARENT_MANIFEST,sort_keys=True).encode()).hexdigest()==identity['manifest_sha256']
    assert all(sha(PARENT/n)==h for n,h in PARENT_MANIFEST.items())
    assert sha(PARENT.parent/'R4_CACHED_STATE_NATIVE_20261014_COLAB_CPU_RESUME1.zip')==PLAN['parent_archive_sha256']
    def parent_path(n):
        assert n in PARENT_MANIFEST
        p=PARENT/n
        if n not in USED:assert sha(p)==PARENT_MANIFEST[n];USED[n]=PARENT_MANIFEST[n]
        return p
    def load_parent(n):
        with np.load(parent_path(n),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    pin=json.loads((SOURCE/'model_pin.json').read_text());weights=Path(PLAN['checkpoint']);verified=[]
    for item in pin['files']:
        p=weights/item['name']
        if not p.is_file():continue
        assert p.stat().st_size==item['size']
        if item.get('sha256'):assert sha(p)==item['sha256']
        else:
            data=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['git_blob_id']
        verified.append(item['name'])
    assert len(verified)==8;dump(ROOT/'MODEL_IDENTITY.json',{'revision':PLAN['revision'],'files':verified})
    report('loading_model');rig=MemoryTransportRig(weights,deadline);rig.calls=PRIOR_CALLS;rig.precision(torch.float32)
    QUESTIONS=json.loads((ROOT/'INPUTS.json').read_text())['questions']
    with np.load(ROOT/'DIRECTIONS.npz',allow_pickle=False) as z:V=z['vectors'].copy()
    assert V.shape==(2,1024)
    for n in ['contexts','probes','coefficients','predictions','actual','mixed','parent_probe_extensions']:(ROOT/n).mkdir()
    BASES={};FIRST_BASES={};CTX1={};CTX2={};M1={};ENTRIES={};JETS={};CARRY=[]
    PARENT_SCHEMA=json.loads(parent_path('STATE_SCHEMA.json').read_text())
    MEMORY_NAMES=sorted(n for n,s in PARENT_SCHEMA.items() if s['group'].startswith('memory_'));assert len(MEMORY_NAMES)==26
    FIRST_NAMES=None;SECOND_NAMES=None;SECOND_SCHEMA=None
    def first_mode(pid):
        rig.second_step=False;rig.capture_complete_cache=False;rig.names=None;rig.schema=None;rig.ctx=CTX1[pid]
    def second_mode(pid,cache=None):
        rig.second_step=True;rig.capture_complete_cache=False;rig.complete_out_cache=None
        rig.names=SECOND_NAMES;rig.schema=SECOND_SCHEMA;rig.ctx=dict(CTX2[pid])
        if cache is not None:rig.ctx.update(cache=cache,hash=cache_hash(cache))
    def memory_values(point,probe=False):return {n:point[n+'_value' if probe else n] for n in MEMORY_NAMES}
    def memory_slopes(point):return {n:point[n+'_first'] for n in MEMORY_NAMES}
    exec(compile((SOURCE/'reused_helpers.py').read_text(),str(SOURCE/'reused_helpers.py'),'exec'),globals())
    def prepare_context(q):
        global SECOND_NAMES,SECOND_SCHEMA,FIRST_NAMES
        pid=q['id'];meta=json.loads(parent_path(f'contexts/{pid}.json').read_text())
        cache=torch.load(parent_path(f'contexts/{pid}_incoming_cache.pt'),map_location='cpu',weights_only=False)
        assert cache_hash(cache)==meta['incoming_cache_hash']
        CTX1[pid]={'question':q,'cache':cache,'hash':meta['incoming_cache_hash'],'token':meta['token_id'],'fallback':meta['EOS_fallback'],
            'length':q['prefix_length']+1,'base_invariants':meta['outgoing_constant_invariants']}
        first_mode(pid);rig.capture_complete_cache=True;rig.timing_label='normal_first';first=rig.value();FIRST_NAMES=list(rig.names)
        saved=load_parent(f'contexts/{pid}_baseline.npz')
        assert np.array_equal(first['logits'],saved['logits']) and all(np.array_equal(first['values'][n],saved[n]) for n in FIRST_NAMES)
        FIRST_BASES[pid]={'values':{n:saved[n] for n in FIRST_NAMES},'logits':saved['logits']}
        M1[pid]=copy.deepcopy(rig.complete_out_cache)
        candidate=int(np.argmax(first['logits']));eos={248044,248046,int(rig.tokenizer.eos_token_id)};fallback=candidate in eos;token=760 if fallback else candidate
        CTX2[pid]={'question':q,'cache':M1[pid],'hash':cache_hash(M1[pid]),'token':token,'fallback':fallback,'length':q['prefix_length']+2,'base_invariants':None}
        rig.second_step=True;rig.capture_complete_cache=False;rig.complete_out_cache=None;rig.names=None;rig.schema=None;rig.ctx=CTX2[pid]
        rig.timing_label='normal_second';base=rig.value();BASES[pid]=base
        if SECOND_NAMES is None:SECOND_NAMES=list(rig.names);SECOND_SCHEMA=copy.deepcopy(rig.schema)
        else:assert SECOND_NAMES==rig.names and SECOND_SCHEMA==rig.schema
        save_value(ROOT/'contexts'/f'{pid}_baseline.npz',base)
        dump(ROOT/'contexts'/f'{pid}.json',{'first_token_id':meta['token_id'],'normal_second_candidate':candidate,'forced_second_token':token,
            'forced_second_text':rig.tokenizer.decode([token]),'EOS_fallback':fallback,'EOS_ids':sorted(eos),'normal_M1_hash':CTX2[pid]['hash'],
            'normal_first_exact_replay':True,'outgoing_constant_invariants':base['invariants'],'source_context':pid})
        dump(ROOT/'STATE_SCHEMA.json',SECOND_SCHEMA)
    def parent_probe(q,di,alpha,mixed=False):
        pid=q['id'];tag=('mix_'+pid) if mixed else f'{pid}_d{di}'
        name=f'probes/{tag}_q0.npz' if alpha==0 else f'probes/{tag}_s{abs(alpha)}_'+('plus' if alpha>0 else 'minus')+'.npz'
        if name in PARENT_MANIFEST:return load_parent(name)
        assert abs(alpha)==.00390625 and q['split']=='held_out','Unexpected missing parent input'
        target=ROOT/'parent_probe_extensions'/Path(name).name
        if target.exists():
            with np.load(target,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
        first_mode(pid);rig.timing_label='parent_probe_extension'
        direction=(V[0]+V[1])/np.sqrt(2) if mixed else V[di]
        point=rig.first_at(torch.tensor(direction),alpha);save_probe(target,point)
        with np.load(target,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def memory_eval(q,di,alpha=0.,ad=True,mixed=False):
        pid=q['id'];point=parent_probe(q,di,alpha,mixed)
        cache=rig.assemble_memory(M1[pid],memory_values(point,True));second_mode(pid,cache)
        rig.timing_label='mixed_derivative' if mixed else 'basis_derivative'
        result=rig.memory_first(memory_slopes(point)) if ad else rig.value()
        assert cache_hash(M1[pid])==CTX2[pid]['hash']
        if ad:
            assert np.count_nonzero(result['first']['b11_input'])==0
            carry=[]
            for name in MEMORY_NAMES:
                if not name.endswith('_current'):continue
                prev=name.replace('_current','_previous1')
                assert np.array_equal(result['values'][prev],point[name+'_value']) and np.array_equal(result['first'][prev],point[name+'_first'])
                carry.append(prev)
            CARRY.append({'question':pid,'direction':'sum' if mixed else di,'alpha':alpha,'previous_KV_exact':len(carry),'b11_input_tangent_zero':True})
        return result
    # Lossless typed-cache assembly must match an actual repeated first-step cache.
    for q in QUESTIONS:
        if q['split']=='calibration':prepare_context(q)
    q=QUESTIONS[0];pid=q['id'];recorded=load_parent(f'actual/{pid}_d0_a0.125.npz')
    assembled=rig.assemble_memory(M1[pid],memory_values(recorded));first_mode(pid);rig.capture_complete_cache=True
    rig.timing_label='cache_reconstruction_control';control=rig.value(torch.tensor(V[0]),.125)
    assert cache_hash(rig.complete_out_cache)==cache_hash(assembled)
    dump(ROOT/'CACHE_RECONSTRUCTION_CONTROL.json',{'question':pid,'direction':0,'alpha':.125,'whole_cache_hash':cache_hash(assembled),'exact':True,'counted_calls':1})
    exact_available=False;forward_available=None;attempts=[];SETUP_TIMES=[];STEPS=PLAN['numerics']['calibration_steps']
    def sample(q,di,tag,steps,mixed=False,inferred=None):
        global forward_available
        sample_begin=time.perf_counter();sample_calls=rig.calls
        first=inferred
        if first is None and forward_available is not False:
            before=rig.calls;begin=time.perf_counter()
            try:first=memory_eval(q,di,0.,True,mixed);primal_ok(first,BASES[q['id']]);forward_available=True;save_probe(ROOT/'probes'/f'{tag}_q0.npz',first)
            except (RuntimeError,NotImplementedError) as e:
                if not any(k in str(e).lower() for k in ['forward','dual','tangent','not implemented','deepcopy','inplace','in-place']):raise
                attempts.append({'kind':'memory_forward_AD','success':False,'calls':rig.calls-before,'seconds':time.perf_counter()-begin,'error':type(e).__name__+': '+str(e)})
                forward_available=False;first=None;gc.collect()
        fds={}
        for step in steps:
            plus=memory_eval(q,di,step,forward_available is True,mixed);minus=memory_eval(q,di,-step,forward_available is True,mixed)
            save_probe(ROOT/'probes'/f'{tag}_s{step}_plus.npz',plus);save_probe(ROOT/'probes'/f'{tag}_s{step}_minus.npz',minus)
            fds[step]=central(plus,minus,BASES[q['id']],step)
            if first is not None and 'first' in plus:
                fds[step]['slope_third']={n:(plus['first'][n].astype(np.float64)+minus['first'][n]-2*first['first'][n])/(step*step) for n in SECOND_NAMES}
                fds[step]['t_logits']=(plus['q'].astype(np.float64)+minus['q']-2*first['q'])/(step*step)
        SETUP_TIMES.append({'tag':tag,'kind':'mixed' if mixed else 'basis','model_calls':rig.calls-sample_calls,
            'seconds_including_parent_loading_assembly_and_probe_storage':time.perf_counter()-sample_begin})
        return {'first':first,'exact':None,'fd':fds}
    for q in QUESTIONS:
        if q['split']!='calibration':continue
        for di in range(2):
            ENTRIES[q['id'],di]=sample(q,di,f'{q["id"]}_d{di}',STEPS)
            report('calibration',question=q['id'],direction=di)
    second_mode(QUESTIONS[0]['id'])
    exec(compile((SOURCE/'reused_numerical_selection.py').read_text(),str(SOURCE/'reused_numerical_selection.py'),'exec'),globals())
    def complete_jet(q,entry):
        jet=jet_from(q,entry);available=entry['first'] is not None and 'slope_third' in entry['fd'][fine]
        jet.update(third_available=available,third_validated=False,third_validation={})
        if available:
            jet['third']=entry['fd'][fine]['slope_third'];jet['t']=entry['fd'][fine]['t_logits']
            ck=checks(entry['fd'][coarse]['slope_third'],jet['third'],SECOND_SCHEMA,.10)
            den=wn(BASES[q['id']]['logits'],jet['t']);en=wn(BASES[q['id']]['logits'],entry['fd'][coarse]['t_logits']-jet['t'])
            jet['third_validation']={'groups':ck,'weighted_logits_relative':en/max(den,1e-30),'weighted_pass':bool(en<=max(.10*den,1e-6))}
            jet['third_validated']=bool(all(v['pass'] for v in ck.values()) and jet['third_validation']['weighted_pass'] and all(m=='slope_difference' for m in SELECTION['modes'].values()))
        return jet
    def save_complete_jet(path,jet):
        arrays={'q':jet['q'],'r':jet['r'],**{n+'_first':v for n,v in jet['first'].items()},**{n+'_second':v for n,v in jet['second'].items()}}
        if jet['third_available']:arrays.update(t=jet['t'],**{n+'_third':v for n,v in jet['third'].items()})
        np.savez_compressed(path,**arrays)
        dump(path.with_suffix('.json'),{k:jet[k] for k in ['validated','validation','third_available','third_validated','third_validation']})
    for q in QUESTIONS:
        if q['split']=='held_out':prepare_context(q)
        for di in range(2):
            entry=ENTRIES[q['id'],di] if q['split']=='calibration' else sample(q,di,f'{q["id"]}_d{di}',[coarse,fine])
            second_mode(q['id']);jet=complete_jet(q,entry);JETS[q['id'],di]=jet;save_complete_jet(ROOT/'coefficients'/f'{q["id"]}_d{di}.npz',jet)
            report('basis_coefficients',question=q['id'],direction=di,validated=jet['validated'],third_validated=jet['third_validated'])
    ENTRIES.clear();gc.collect()
    CUBIC_LOCK=all(JETS[q['id'],di]['third_validated'] for q in QUESTIONS if q['split']=='calibration' for di in range(2))
    dump(ROOT/'CUBIC_SELECTION_LOCK.json',{'enabled':CUBIC_LOCK,'target_endpoints_seen':0,'based_only_on_calibration':True,'extra_model_calls':0})
    # Choose optional coverage using support/availability/call count only, before targets.
    reserved_core=33;remaining=300-rig.calls-reserved_core
    chain_steps=([coarse,fine] if remaining>=5 else [fine] if remaining>=3 else []) if forward_available else []
    chain_calls=1+2*len(chain_steps) if chain_steps else 0;remaining-=chain_calls
    mix_questions=[q for q in QUESTIONS if q['id'] in ['r14_cal_window','r14_test_notes']]
    missing_mix=sum(f'probes/mix_{q["id"]}_s{s}_{sign}.npz' not in PARENT_MANIFEST for q in mix_questions for s in [coarse,fine] for sign in ['plus','minus'])
    plane_admitted=remaining>=15+missing_mix
    if plane_admitted:remaining-=15+missing_mix
    single_mixed=not plane_admitted and remaining>=1
    if single_mixed:remaining-=1
    cal_count=min(16,max(0,remaining))
    allocation={'calls_before_optional':rig.calls,'required_held_and_null':33,'chain_steps':chain_steps,'chain_calls':chain_calls,
        'full_plane_admitted':plane_admitted,'plane_parent_extensions':missing_mix if plane_admitted else 0,'plane_calls':15 if plane_admitted else 0,
        'single_diagonal_mixed_target':single_mixed,'calibration_targets':cal_count,'based_on_calibration_and_availability_only':True,'target_endpoints_seen':0}
    dump(ROOT/'CALL_ALLOCATION_LOCK.json',allocation)
    chain={'available':bool(chain_steps),'pass':False,'components':{}}
    if chain_steps:
        q=next(q for q in QUESTIONS if q['id']=='r14_cal_window');pid=q['id'];p0=parent_probe(q,0,0.)
        q1=memory_slopes(p0);pp=parent_probe(q,0,fine);pm=parent_probe(q,0,-fine)
        r1={n:((pp[n+'_first'].astype(np.float64)-pm[n+'_first'])/(2*fine)).astype(np.float32) for n in MEMORY_NAMES}
        second_mode(pid);rig.timing_label='chain_J_r1';jr=rig.memory_first(r1);Hs={}
        normal={n:FIRST_BASES[pid]['values'][n] for n in MEMORY_NAMES}
        for step in chain_steps:
            points=[]
            for sign in [1,-1]:
                shifted={n:(normal[n].astype(np.float64)+sign*step*q1[n]).astype(np.float32) for n in MEMORY_NAMES}
                second_mode(pid,rig.assemble_memory(M1[pid],shifted));rig.timing_label='chain_H_q1_q1';points.append(rig.memory_first(q1))
            Hs[step]={n:(points[0]['first'][n].astype(np.float64)-points[1]['first'][n])/(2*step) for n in SECOND_NAMES}
        predicted={n:jr['first'][n].astype(np.float64)+Hs[fine][n] for n in SECOND_NAMES}
        check=checks(predicted,JETS[pid,0]['second'],SECOND_SCHEMA,.10)
        head=rig.head(predicted['hidden']);den=wn(BASES[pid]['logits'],JETS[pid,0]['r']);en=wn(BASES[pid]['logits'],head-JETS[pid,0]['r'])
        chain={'available':True,'question':pid,'direction':0,'steps':chain_steps,'groups':check,'weighted_logit_relative':en/max(den,1e-30),
            'pass':bool(all(v['pass'] for v in check.values()) and en<=max(.10*den,1e-6)),
            'H_coarse_fine':checks(Hs[coarse],Hs[fine],SECOND_SCHEMA,.10) if coarse in Hs else None,'counted_calls':chain_calls}
        chain['component_norms']={}
        for group in GROUPS:
            selected=[n for n in SECOND_NAMES if SECOND_SCHEMA[n]['group']==group]
            jnorm=group_norm(jr['first'],SECOND_SCHEMA,group);hnorm=group_norm(Hs[fine],SECOND_SCHEMA,group)
            dot=sum(float(jr['first'][n].astype(np.float64).reshape(-1)@Hs[fine][n].reshape(-1)) for n in selected)
            chain['component_norms'][group]={'transported_incoming_curvature_norm':jnorm,'second_step_nonlinearity_norm':hnorm,
                'sum_norm':group_norm(predicted,SECOND_SCHEMA,group),'component_cosine':dot/(jnorm*hnorm) if jnorm*hnorm>1e-20 else None}
    dump(ROOT/'CHAIN_RULE_CONTROL.json',chain)
    MIX_JETS={};MIX_DIAG={};MIX_VALID=True
    if plane_admitted:
        for q in mix_questions:
            pid=q['id'];j0,j1=JETS[pid,0],JETS[pid,1]
            p0=parent_probe(q,0,0.);p1=parent_probe(q,1,0.);ps=parent_probe(q,0,0.,True)
            parent_linear={n:(p0[n+'_first'].astype(np.float64)+p1[n+'_first'])/np.sqrt(2) for n in MEMORY_NAMES}
            linear_error=max(norm(ps[n+'_first']-parent_linear[n])/max(norm(ps[n+'_first']),1e-30) for n in MEMORY_NAMES)
            linear_pass=all(norm(ps[n+'_first']-parent_linear[n])<=max(1e-5*norm(ps[n+'_first']),1e-6) for n in MEMORY_NAMES)
            assert linear_pass,'Parent sum tangent is not linear within the fixed rounding bound'
            inferred={'values':BASES[pid]['values'],'logits':BASES[pid]['logits'],
                'first':{n:((j0['first'][n].astype(np.float64)+j1['first'][n])/np.sqrt(2)).astype(np.float32) for n in SECOND_NAMES},
                'q':(j0['q'].astype(np.float64)+j1['q'])/np.sqrt(2)} if forward_available else None
            entry=sample(q,0,'mix_'+pid,[coarse,fine],True,inferred)
            # Do not spend an additional zero-slope call when forward AD is unavailable.
            second_mode(pid);js=complete_jet(q,entry);cross={n:js['second'][n].astype(np.float64)-.5*(j0['second'][n].astype(np.float64)+j1['second'][n]) for n in SECOND_NAMES}
            dump(ROOT/'mixed'/f'{pid}_sum_validation.json',{'validated':js['validated'],'validation':js['validation'],'parent_linearity_error':linear_error,
                'q_zero_assembled_from_basis':True,'cross_norms':{g:group_norm(cross,SECOND_SCHEMA,g) for g in GROUPS}})
            MIX_VALID=MIX_VALID and js['validated']
            for name,(aa,bb) in [('sum',(1/np.sqrt(2),1/np.sqrt(2))),('difference',(1/np.sqrt(2),-1/np.sqrt(2))),('weighted',(2/np.sqrt(5),1/np.sqrt(5)))]:
                first={n:(aa*j0['first'][n]+bb*j1['first'][n]).astype(np.float32) for n in SECOND_NAMES}
                diagonal={n:(aa*aa*j0['second'][n]+bb*bb*j1['second'][n]).astype(np.float32) for n in SECOND_NAMES}
                second={n:(diagonal[n].astype(np.float64)+2*aa*bb*cross[n]).astype(np.float32) for n in SECOND_NAMES}
                common={'first':first,'q':rig.head(first['hidden']),'validated':bool(js['validated'] and j0['validated'] and j1['validated']),
                    'validation':js['validation'],'third_available':False,'third_validated':False,'third_validation':{}}
                full={**common,'second':second,'r':rig.head(second['hidden'])};diag={**common,'second':diagonal,'r':rig.head(diagonal['hidden'])}
                MIX_JETS[pid,name]=full;MIX_DIAG[pid,name]=diag;save_complete_jet(ROOT/'mixed'/f'{pid}_{name}.npz',full);save_complete_jet(ROOT/'mixed'/f'{pid}_{name}_diagonal.npz',diag)
    forecast_times=[]
    def prediction(base,jet,alpha,order,record_time=False):
        begin=time.perf_counter()
        delta=alpha*jet['q'].astype(np.float64)+.5*alpha*alpha*jet['r'].astype(np.float64)
        states={n:alpha*jet['first'][n].astype(np.float64)+.5*alpha*alpha*jet['second'][n].astype(np.float64) for n in SECOND_NAMES}
        if order==3:
            delta+=alpha**3*jet['t']/6
            for n in SECOND_NAMES:states[n]+=alpha**3*jet['third'][n]/6
        logits=base['logits'].astype(np.float64)+delta;p,lp=probability(logits)
        elapsed=time.perf_counter()-begin
        if record_time:forecast_times.append({'order':order,'seconds':elapsed,'fields':len(states),'vocabulary':len(logits)})
        h=hashlib.sha256()
        for n in sorted(states):h.update(n.encode()+b'\0');h.update(states[n].tobytes())
        return logits,states,h.hexdigest()
    alphas=[s*a for s in [.03125,.125] for a in [1,-1]];FORECASTS={}
    for q in QUESTIONS:
        pid=q['id']
        for di in range(2):
            jet=JETS[pid,di];fs=[]
            for alpha in alphas:
                row={'alpha':alpha,'quadratic_Fisher':.5*alpha*alpha*wn(BASES[pid]['logits'],jet['q'])**2}
                for order in [2]+([3] if jet['third_available'] else []):
                    logits,_,digest=prediction(BASES[pid],jet,alpha,order,True)
                    row[str(order)]={'cost':kl(BASES[pid]['logits'],logits),'state_delta_sha256':digest}
                row['first_order_cost']=kl(BASES[pid]['logits'],BASES[pid]['logits'].astype(np.float64)+alpha*jet['q'])
                fs.append(row)
            FORECASTS[pid,di]=fs;dump(ROOT/'predictions'/f'{pid}_d{di}.json',fs)
    MIX_FORECASTS={}
    if plane_admitted:
        for key,jet in MIX_JETS.items():
            pid,name=key;fs=[]
            for alpha in [.125,-.125]:
                full,_,digest=prediction(BASES[pid],jet,alpha,2)
                diag,_,dd=prediction(BASES[pid],MIX_DIAG[key],alpha,2)
                fs.append({'alpha':alpha,'full_cost':kl(BASES[pid]['logits'],full),'diagonal_cost':kl(BASES[pid]['logits'],diag),
                    'full_state_delta_sha256':digest,'diagonal_state_delta_sha256':dd})
            MIX_FORECASTS[key]=fs;dump(ROOT/'mixed'/f'{pid}_{name}_forecasts.json',fs)
    if single_mixed:
        q=next(q for q in QUESTIONS if q['id']=='r14_test_notes');pid=q['id'];j0,j1=JETS[pid,0],JETS[pid,1]
        jet={'first':{n:(j0['first'][n]+j1['first'][n])/np.sqrt(2) for n in SECOND_NAMES},
            'second':{n:.5*(j0['second'][n]+j1['second'][n]) for n in SECOND_NAMES}}
        jet.update(q=rig.head(jet['first']['hidden']),r=rig.head(jet['second']['hidden']),validated=j0['validated'] and j1['validated'],third_available=False,third_validated=False)
        MIX_DIAG[pid,'sum']=jet;save_complete_jet(ROOT/'mixed'/f'{pid}_sum_diagonal_only.npz',{**jet,'validation':{},'third_validation':{}})
        logits,_,digest=prediction(BASES[pid],jet,.125,2);dump(ROOT/'mixed/SINGLE_DIAGONAL_FORECAST.json',{'question':pid,'alpha':.125,'cost':kl(BASES[pid]['logits'],logits),'state_delta_sha256':digest})
    paths=list((ROOT/'coefficients').glob('*'))+list((ROOT/'predictions').glob('*'))+list((ROOT/'mixed').glob('*'))
    pred_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    dump(ROOT/'PREDICTION_FREEZE.json',{'sha256':pred_hashes,'selection_sha256':selection_sha,'cubic_selection_sha256':sha(ROOT/'CUBIC_SELECTION_LOCK.json'),
        'allocation_sha256':sha(ROOT/'CALL_ALLOCATION_LOCK.json'),'all_forecasts_precede_endpoints':True,'T2_target_endpoints_seen':0})
    prediction_sha=sha(ROOT/'PREDICTION_FREEZE.json');dump(ROOT/'FORECAST_TIMINGS.json',forecast_times)
    q=QUESTIONS[0];pid=q['id'];nullcache=rig.assemble_memory(M1[pid],FIRST_BASES[pid]['values']);second_mode(pid,nullcache);rig.timing_label='strict_null'
    null=rig.value();assert np.array_equal(null['logits'],BASES[pid]['logits']) and all(np.array_equal(null['values'][n],BASES[pid]['values'][n]) for n in SECOND_NAMES)
    dump(ROOT/'NULL_CONTROL.json',{'exact':True,'question':pid,'counted_calls':1})
    def actual_point(q,di,alpha,mixed_name=None):
        pid=q['id'];name=f'mixed/{pid}_{mixed_name}_a{alpha}_actual.npz' if mixed_name else f'actual/{pid}_d{di}_a{alpha}.npz'
        recorded=load_parent(name);mem=memory_values(recorded);cache=rig.assemble_memory(M1[pid],mem);second_mode(pid,cache)
        rig.timing_label='mixed_actual' if mixed_name else 'held_actual' if q['split']=='held_out' else 'calibration_actual'
        value=rig.value();value['incoming_memory_change_norm']=float(np.sqrt(sum(norm(mem[n]-FIRST_BASES[pid]['values'][n])**2 for n in MEMORY_NAMES)))
        for n in MEMORY_NAMES:
            if n.endswith('_current'):assert np.array_equal(value['values'][n.replace('_current','_previous1')],recorded[n])
        assert np.array_equal(value['values']['b11_input'],BASES[pid]['values']['b11_input'])
        return value
    def score(q,di,alpha,actual,jet):
        pid=q['id'];base=BASES[pid];cost=kl(base['logits'],actual['logits']);true=actual['logits'].astype(np.float64)-base['logits']
        result={'question':pid,'split':q['split'],'family':q['family'],'direction':di,'alpha':alpha,'strength':abs(alpha),'actual_KL':cost,'resolved':cost>1e-10,
            'derivative_validated':jet['validated'],'third_validated':jet.get('third_validated',False),'incoming_memory_change_norm':actual['incoming_memory_change_norm'],
            'zero_delivery':actual['incoming_memory_change_norm']<=1e-12,'current_b11_input_exactly_unchanged':True,'orders':{}}
        for order in [1,2]+([3] if jet.get('third_available') else []):
            if order==1:
                logits=base['logits'].astype(np.float64)+alpha*jet['q'];states={n:alpha*jet['first'][n].astype(np.float64) for n in SECOND_NAMES}
            else:logits,states,_=prediction(base,jet,alpha,order)
            error={n:states[n]-(actual['values'][n].astype(np.float64)-base['values'][n]) for n in SECOND_NAMES}
            truth={n:actual['values'][n].astype(np.float64)-base['values'][n] for n in SECOND_NAMES};group_rows={};fields={}
            for group in GROUPS:
                den=group_norm(truth,SECOND_SCHEMA,group);en=group_norm(error,SECOND_SCHEMA,group)
                group_rows[group]={'change_norm':den,'error_norm':en,'relative_error':en/den if den>1e-12 else None}
            current=[n for n in SECOND_NAMES if n.startswith('m') and n.endswith('_current')]
            den=np.sqrt(sum(norm(truth[n])**2 for n in current));en=np.sqrt(sum(norm(error[n])**2 for n in current))
            for n in SECOND_NAMES:
                tn=norm(truth[n]);resolved=tn>max(1e-10,20*np.finfo(np.float32).eps*norm(base['values'][n]))
                fields[n]={'group':SECOND_SCHEMA[n]['group'],'layer':SECOND_SCHEMA[n]['layer'],'resolved':bool(resolved),
                    'relative_error':norm(error[n])/tn if resolved else None,'change_norm':tn,'error_norm':norm(error[n]),'exactly_unchanged':bool(np.array_equal(actual['values'][n],base['values'][n]))}
            predicted=kl(base['logits'],logits)
            result['orders'][str(order)]={'predicted_KL':predicted,'cost_error':abs(predicted-cost)/cost if cost>1e-10 else None,
                'weighted_logit_error':wn(base['logits'],logits-base['logits']-true)/max(wn(base['logits'],true),1e-30),
                'actual_to_prediction_KL':kl(actual['logits'],logits),'groups':group_rows,'fields':fields,
                'current_KV_only_error':float(en/den) if den>1e-12 else None}
        result['Fisher_cost']=.5*alpha*alpha*wn(base['logits'],jet['q'])**2
        result['Fisher_cost_error']=abs(result['Fisher_cost']-cost)/cost if cost>1e-10 else None
        return result
    metrics=[]
    targets=[(q,di,a) for q in QUESTIONS if q['split']=='held_out' for di in range(2) for a in alphas]
    targets += [(q,di,a) for q in QUESTIONS if q['split']=='calibration' for di in range(2) for a in alphas][:cal_count]
    for q,di,alpha in targets:
        value=actual_point(q,di,alpha);row=score(q,di,alpha,value,JETS[q['id'],di]);metrics.append(row)
        save_value(ROOT/'actual'/f'{q["id"]}_d{di}_a{alpha}.npz',value);STATUS['main_endpoints']=len(metrics)
        dump(ROOT/'METRICS.json',metrics);report('main_endpoints',question=q['id'],direction=di)
    mixed=[]
    if plane_admitted:
        targets=[(mix_questions[0],'sum',.125)]+[(mix_questions[1],name,a) for name in ['sum','difference','weighted'] for a in [.125,-.125]]
        for q,name,alpha in targets:
            value=actual_point(q,0,alpha,name);row=score(q,name,alpha,value,MIX_JETS[q['id'],name]);row['diagonal']=score(q,name,alpha,value,MIX_DIAG[q['id'],name])['orders']['2']
            mixed.append(row);save_value(ROOT/'mixed'/f'{q["id"]}_{name}_a{alpha}_actual.npz',value)
            STATUS['mixed_endpoints']=len(mixed);dump(ROOT/'MIXED_METRICS.json',mixed);report('mixed_endpoints',question=q['id'],direction=name)
    elif single_mixed:
        q=next(q for q in QUESTIONS if q['id']=='r14_test_notes');value=actual_point(q,0,.125,'sum')
        mixed=[score(q,'sum_diagonal_only',.125,value,MIX_DIAG[q['id'],'sum'])];save_value(ROOT/'mixed/SINGLE_DIAGONAL_ACTUAL.npz',value);dump(ROOT/'MIXED_METRICS.json',mixed)
        STATUS['mixed_endpoints']=1
    assert len([r for r in metrics if r['split']=='held_out'])==32 and rig.calls<=300
    assert all(sha(ROOT/n)==h for n,h in freeze['sha256'].items()),'Frozen source or initial inputs changed'
    assert all(sha(ROOT/n)==h for n,h in pred_hashes.items()) and sha(ROOT/'PREDICTION_FREEZE.json')==prediction_sha
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in rig.model.modules())
    dump(ROOT/'MEMORY_CARRY_CHECKS.json',CARRY);dump(ROOT/'CALL_TIMINGS.json',rig.call_timings);dump(ROOT/'JET_SETUP_TIMINGS.json',SETUP_TIMES)
    dump(ROOT/'CALL_LEDGER.json',{'parent':PRIOR_CALLS,'new':rig.calls-PRIOR_CALLS,'combined':rig.calls,'ceiling':300,'listed_cumulative':1004+rig.calls-PRIOR_CALLS})
    STATUS['status']='native_complete';report('native_complete_export_pending',cubic_enabled=CUBIC_LOCK,chain_pass=chain['pass'],plane_admitted=plane_admitted)
except BaseException as e:
    STATUS['model_calls']=rig.calls if 'rig' in globals() else PRIOR_CALLS;STATUS['new_calls']=STATUS['model_calls']-PRIOR_CALLS
    STATUS['status']='fault_or_calibration_stop';report('fault',error=type(e).__name__+': '+str(e));(ROOT/'FAULT.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    if 'rig' in globals():
        dump(ROOT/'CALL_TIMINGS.json',rig.call_timings);del rig;gc.collect()
    try:export()
    finally:timer.cancel()
