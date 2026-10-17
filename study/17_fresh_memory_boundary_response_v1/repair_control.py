"""Eight counted native controls, stage one of the prospective 300-call phase.

This validates a wrapper storage repair on old calibration data. It does not
relabel study15 or constitute fresh-question confirmation.
"""
import os,time
BEGIN=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='4')
import copy,gc,hashlib,json,shutil,threading,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
assert ROOT.name=='colab_cpu_20261014' and ROOT.parent.name=='fresh_memory_boundary_response_v1'
P14=Path('/content/research4/artifacts/cached_state_response_v1/colab_cpu_20261014_resume1')
P15=Path('/content/research4/artifacts/cached_memory_transport_v1/colab_cpu_20261014')
STATUS={'status':'starting','phase17_calls':0,'stage':'historical_storage_repair','PID':os.getpid(),'GPU_allocations':0}
USED={};rig=None

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def report(phase,**extra):
    STATUS.update(phase=phase,seconds=time.monotonic()-BEGIN,phase17_calls=rig.calls if rig else 0,**extra)
    dump(ROOT/'REPAIR_STATUS.json',STATUS);print(json.dumps(STATUS),flush=True)
def deadline():
    if time.monotonic()-BEGIN>2400:raise TimeoutError('2400-second native budget')
    if rig is not None and rig.calls>=8:raise RuntimeError('Stage one eight-forward ceiling')
def hardstop():dump(ROOT/'REPAIR_HARD_STOP.json',STATUS);os._exit(124)
timer=threading.Timer(2700,hardstop);timer.daemon=True;timer.start()

try:
    import numpy as np,torch,transformers
    import torch.nn.functional as tf
    from native import cache_hash,arr
    from transport_native import MemoryTransportRig
    from maths import checks,wn,norm,GROUPS,group_norm
    from lossless_arrays import write_bundle,read_bundle
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert (torch.__version__,transformers.__version__,np.__version__)==('2.13.0+cpu','5.19.0','2.5.4')
    assert not torch.cuda.is_available()
    freeze=json.loads((ROOT/'REPAIR_EXECUTION_FREEZE.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in freeze['sha256'].items())
    backup=json.loads((ROOT/'PRIOR_BACKUP.json').read_text())
    assert backup['provider_metadata_verified'] and backup['host_raw_transfer_complete']
    assert backup['files'][0]['sha256']=='be2cd385724c3b49ca6c27719d699bec93fee6a5a1c99bfc34eeb69da0e47658'
    free=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    admission={'free_after_imports':free,'minimum':8589934592,'disk_free':shutil.disk_usage(ROOT).free,'disk_minimum':17179869184}
    dump(ROOT/'REPAIR_ADMISSION.json',admission)
    assert free>=admission['minimum'] and admission['disk_free']>=admission['disk_minimum']
    manifests={14:json.loads((P14/'EXPORT_HASHES.json').read_text()),15:json.loads((P15/'EXPORT_HASHES.json').read_text())}
    def old_path(stage,name):
        p={14:P14,15:P15}[stage]/name;expected=manifests[stage][name]
        assert sha(p)==expected;USED[f'{stage}:{name}']=expected;return p
    def old_json(stage,name):return json.loads(old_path(stage,name).read_text())
    def old_arrays(stage,name):
        with np.load(old_path(stage,name),allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
    pin=json.loads((ROOT/'source/model_pin.json').read_text());weights=Path('/content/r4_checkpoint');verified=[]
    for item in pin['files']:
        p=weights/item['name']
        if not p.is_file():continue
        assert p.stat().st_size==item['size']
        if item.get('sha256'):assert sha(p)==item['sha256']
        else:
            data=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['git_blob_id']
        verified.append(item['name'])
    assert len(verified)==8;dump(ROOT/'REPAIR_MODEL_IDENTITY.json',{'files':verified,'revision':'2fc06364715b967f1860aea9cf38778875588b17'})
    report('loading_model');rig=MemoryTransportRig(weights,deadline);rig.precision(torch.float32)
    pid='r14_cal_window';q=next(q for q in old_json(14,'INPUTS.json')['questions'] if q['id']==pid)
    meta1=old_json(14,f'contexts/{pid}.json');meta2=old_json(15,f'contexts/{pid}.json')
    cache=torch.load(old_path(14,f'contexts/{pid}_incoming_cache.pt'),map_location='cpu',weights_only=False)
    assert cache_hash(cache)==meta1['incoming_cache_hash']
    ctx1={'question':q,'cache':cache,'hash':cache_hash(cache),'token':meta1['token_id'],
          'length':q['prefix_length']+1,'base_invariants':meta1['outgoing_constant_invariants']}
    rig.ctx=ctx1;rig.capture_complete_cache=True;rig.timing_label='normal_first_reconstruction';normal1=rig.value()
    old1=old_arrays(14,f'contexts/{pid}_baseline.npz')
    assert np.array_equal(normal1['logits'],old1['logits']) and all(np.array_equal(v,old1[n]) for n,v in normal1['values'].items())
    M1=copy.deepcopy(rig.complete_out_cache);assert cache_hash(M1)==meta2['normal_M1_hash']
    schema=old_json(15,'STATE_SCHEMA.json');schema14=old_json(14,'STATE_SCHEMA.json')
    memory=sorted(n for n,s in schema14.items() if s['group'].startswith('memory_'));assert len(memory)==26
    ctx2={'question':q,'cache':M1,'hash':cache_hash(M1),'token':meta2['forced_second_token'],
          'length':q['prefix_length']+2,'base_invariants':meta2['outgoing_constant_invariants']}
    def second_mode(cache=None):
        rig.second_step=True;rig.capture_complete_cache=False;rig.complete_out_cache=None;rig.names=None;rig.schema=None
        rig.ctx=dict(ctx2)
        if cache is not None:rig.ctx.update(cache=cache,hash=cache_hash(cache))
    q0=old_arrays(14,f'probes/{pid}_d0_q0.npz')
    normal_memory={n:old1[n] for n in memory};seed={n:q0[n+'_first'].copy() for n in memory}
    original_seed={n:v.copy() for n,v in seed.items()}
    archived=old_arrays(15,f'probes/{pid}_d0_q0.npz')
    def hashes(x):return {n:hashlib.sha256(v.tobytes()).hexdigest() for n,v in x.items()}
    def flatten(point):return {'logits':point['logits'],'q':point['q'],**{n+'_value':v for n,v in point['values'].items()},**{n+'_first':v for n,v in point['first'].items()}}
    points=ROOT/'repair_points';points.mkdir()
    storage=[]
    def save(name,point):
        arrays=flatten(point)
        receipt=write_bundle(points/(name+'.npz'),arrays,{n:('parent15:probes/r14_cal_window_d0_q0.npz',archived[n]) for n in arrays})
        storage.append({'name':name,**receipt})
    # The deliberate old wrapper is confined to this one calibration control.
    # It seeds a cloned primal cache with shared caller-owned tangent storage.
    def shared_seed_first(tangents):
        with torch.no_grad(),torch.autograd.forward_ad.dual_level():
            def hook(module,args,kwargs):
                for name,layer,attribute,key,tensor in rig.affected_tensors(kwargs['past_key_values']):
                    first=torch.as_tensor(tangents[name],dtype=tensor.dtype,device=tensor.device)
                    if key is None:
                        tangent=torch.zeros_like(tensor);tangent[...,-1:,:]=first
                        setattr(layer,attribute,torch.autograd.forward_ad.make_dual(tensor,tangent))
                    else:getattr(layer,attribute)[key]=torch.autograd.forward_ad.make_dual(tensor,first)
                return args,kwargs
            handle=rig.text.register_forward_pre_hook(hook,with_kwargs=True)
            try:
                values=rig.execute();pairs=[torch.autograd.forward_ad.unpack_dual(x) for x in values]
                primal={n:arr(p) for n,(p,t) in zip(rig.names,pairs)}
                first={n:np.zeros_like(primal[n]) if t is None else arr(t) for n,(p,t) in zip(rig.names,pairs)}
            finally:handle.remove()
        with torch.no_grad():
            logits=rig.model.lm_head(torch.tensor(primal['hidden']));ql=tf.linear(torch.tensor(first['hidden']),rig.model.lm_head.weight)
        return {'values':primal,'first':first,'logits':arr(logits),'q':arr(ql)}
    second_mode();before=hashes(seed);shared=shared_seed_first(seed);after=hashes(seed)
    changed=[n for n in seed if before[n]!=after[n]];save('shared_seed',shared)
    dump(ROOT/'NATIVE_SHARED_SEED_MUTATION.json',{'changed_fields':changed,'before':before,'after':after,
         'caller_mutated':bool(changed),'calls':1,'intentional_historical_calibration_control':True})
    second_mode();seed={n:v.copy() for n,v in original_seed.items()};before=hashes(seed)
    rig.timing_label='owned_seed_q0';fixed=rig.memory_first(seed);after=hashes(seed);save('owned_seed',fixed)
    flat_fixed=flatten(fixed);replay={n:float(np.max(np.abs(v.astype(np.float64)-archived[n]))) for n,v in flat_fixed.items()}
    exact=all(np.array_equal(v,archived[n]) for n,v in flat_fixed.items())
    shared_same=all(np.array_equal(v,flat_fixed[n]) for n,v in flatten(shared).items())
    ownership_ok=before==after
    dump(ROOT/'NATIVE_OWNED_SEED_CONTROL.json',{'caller_unchanged':ownership_ok,'primal_and_slope_exact_replay':exact,
         'max_abs_by_field':replay,'shared_and_owned_same_single_call_outputs':shared_same,'calls':1})
    lock=old_json(15,'DERIVATIVE_SELECTION_LOCK.json');coarse,fine=lock['coarse'],lock['fine']
    plus=old_arrays(14,f'probes/{pid}_d0_s{fine}_plus.npz');minus=old_arrays(14,f'probes/{pid}_d0_s{fine}_minus.npz')
    r1={n:((plus[n+'_first'].astype(np.float64)-minus[n+'_first'])/(2*fine)).astype(np.float32) for n in memory}
    second_mode();rig.timing_label='chain_J_r1';jr=rig.memory_first(r1);save('chain_J_r1',jr);Hs={}
    for step in [coarse,fine]:
        pair=[]
        for sign in [1,-1]:
            shifted={n:(normal_memory[n].astype(np.float64)+sign*step*original_seed[n]).astype(np.float32) for n in memory}
            second_mode(rig.assemble_memory(M1,shifted));rig.timing_label='chain_H_q1_q1'
            point=rig.memory_first(original_seed);pair.append(point);save(f'chain_H_s{step}_{sign}',point)
        Hs[step]={n:(pair[0]['first'][n].astype(np.float64)-pair[1]['first'][n])/(2*step) for n in rig.names}
    predicted={n:jr['first'][n].astype(np.float64)+Hs[fine][n] for n in rig.names}
    oldjet=old_arrays(15,f'coefficients/{pid}_d0.npz');target={n:oldjet[n+'_second'] for n in rig.names}
    ck=checks(predicted,target,schema,.10);hck=checks(Hs[coarse],Hs[fine],schema,.10)
    baseline=old_arrays(15,f'contexts/{pid}_baseline.npz');head=rig.head(predicted['hidden'])
    den=wn(baseline['logits'],oldjet['r']);error=wn(baseline['logits'],head-oldjet['r'])
    chain_pass=all(row['pass'] for row in ck.values()) and error<=max(.10*den,1e-6)
    results={'classification':'old calibration repair validation; not fresh outcomes or study15 rescue',
        'native_shared_seed_mutation_confirmed':bool(changed),'mutated_fields':changed,'owned_seed_unchanged':ownership_ok,
        'single_call_outputs_unchanged':shared_same,'archived_q0_exact_replay':exact,'chain_pass':chain_pass,
        'chain_groups':ck,'H_coarse_fine_groups':hck,'weighted_logit_chain_relative':error/max(den,1e-30),
        'component_norms':{g:{'J_r1':group_norm(jr['first'],schema,g),'H_q1_q1':group_norm(Hs[fine],schema,g),
                              'sum':group_norm(predicted,schema,g),'archived_total':group_norm(target,schema,g)} for g in GROUPS},
        'coarse':coarse,'fine':fine,'calls':rig.calls,'phase17_remaining_calls':300-rig.calls,'GPU_allocations':0,
        'full_control_pass':bool(changed) and ownership_ok and shared_same and exact and chain_pass,
        'fresh_question_confirmation_pending':True,'original_study15_formal_failures_unchanged':True,'storage':storage}
    assert rig.calls==8 and cache_hash(M1)==ctx2['hash']
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in rig.model.modules())
    assert all(sha(ROOT/n)==h for n,h in freeze['sha256'].items())
    assert all(sha({14:P14,15:P15}[int(n.split(':',1)[0])]/n.split(':',1)[1])==h for n,h in USED.items())
    dump(ROOT/'REPAIR_RESULTS.json',results);dump(ROOT/'REPAIR_PARENT_USED.json',USED)
    STATUS['status']='repair_controls_complete';report('native_complete',full_control_pass=results['full_control_pass'])
except BaseException as e:
    STATUS['status']='fault';report('fault',error=type(e).__name__+': '+str(e))
    (ROOT/'REPAIR_FAULT.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    if rig is not None:
        dump(ROOT/'REPAIR_CALL_TIMINGS.json',rig.call_timings);del rig;gc.collect();rig=None
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='REPAIR_EXPORT_HASHES.json')
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files};dump(ROOT/'REPAIR_EXPORT_HASHES.json',manifest)
    archive=ROOT.parent/'R4_NATIVE_SEED_REPAIR_20261014.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in files+[ROOT/'REPAIR_EXPORT_HASHES.json']:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
    receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'members':len(files)+1,
        'phase17_calls':STATUS['phase17_calls'],'phase17_remaining_calls':300-STATUS['phase17_calls'],
        'phase17_charged_seconds':time.monotonic()-BEGIN,'model_worker_exited':True,'GPU_allocations':0}
    dump(ROOT.parent/'REPAIR_EXPORT_RECEIPT_20261014.json',receipt);timer.cancel()
    print('R4_NATIVE_REPAIR_EXPORT',json.dumps(receipt),flush=True)
