"""Matched-native-state precision and actual-input linear-response controls."""
from pathlib import Path
from datetime import datetime
import copy,hashlib,json,os,sys,threading,time,traceback,zipfile
ROOT=Path(sys.argv[1]);SOURCE=ROOT/'source';PILOT=Path(sys.argv[2]);sys.path.insert(0,str(SOURCE))
PLAN=json.loads((SOURCE/'MATCHED_SNAPSHOT_PLAN.json').read_text())
origin=datetime.fromisoformat(json.loads((PILOT/'ADMISSION.json').read_text())['admitted_UTC'].replace('Z','+00:00')).timestamp()
started=time.time();native_deadline=min(origin+2400,started+600);hard_deadline=min(origin+2700,started+900)
STATUS={'status':'starting','GPU_allocations':0,'tail_evaluations':0,'whole_backbone_calls':0,'completed_snapshots':0,'completed_reproductions':0}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def report(stage,**extra):
    STATUS.update(stage=stage,elapsed_seconds=time.time()-started,combined_elapsed_seconds=time.time()-origin,**extra)
    dump(ROOT/'STATUS.json',STATUS);print(json.dumps(STATUS),flush=True)
def deadline():
    if (ROOT/'HUMAN_STOP.json').exists():raise InterruptedError('Human stop')
    if time.time()>=native_deadline:raise TimeoutError('Native bound/export reserve')
def hard_stop():
    dump(ROOT/'HARD_TIMEOUT.json',{'elapsed':time.time()-origin,'GPU_allocations':0});os._exit(124)
assert hard_deadline-time.time()>0
timer=threading.Timer(hard_deadline-time.time(),hard_stop);timer.daemon=True;timer.start()
def export():
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['EXPORT_HASHES.json','EXPORT_RECEIPT.json']]
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files};dump(ROOT/'EXPORT_HASHES.json',manifest)
    archive=ROOT.with_name(ROOT.name+'_export.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.write(ROOT/'EXPORT_HASHES.json','EXPORT_HASHES.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'manifest_files':len(manifest),
        'combined_elapsed_seconds':time.time()-origin,'GPU_allocations':0,'all_members_verified':True}
    dump(ROOT/'EXPORT_RECEIPT.json',receipt);print('R4_MATCHED_SNAPSHOT_EXPORT',json.dumps(receipt),flush=True)

try:
    for name,digest in json.loads((ROOT/'EXECUTION_FREEZE.json').read_text())['sources'].items():assert sha(ROOT/name)==digest
    import numpy as np
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from native import Rig,cache_hash,detached_copy
    from maths import array,norm,relative,parity
    from panel_collect import native_intervention
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_available()
    manifest=json.loads((PILOT/'EXPORT_HASHES.json').read_text())
    def verified(path):assert sha(path)==manifest[path.relative_to(PILOT).as_posix()]
    def load_context(pid,position):
        folder=PILOT/'contexts'/pid/'BF16'/position
        verified(folder/'vectors.npz');verified(folder/'attention_context.pt')
        with np.load(folder/'vectors.npz',allow_pickle=False) as z:
            h=torch.tensor(z['attn_H'],dtype=torch.bfloat16);target=torch.tensor(z['readout15'],dtype=torch.bfloat16)
        kwargs=torch.load(folder/'attention_context.pt',map_location='cpu',weights_only=False)
        return h,target,kwargs
    checkpoint=Path('C:/Users/farha/Research4/cache/qwen35_08b_pinned')
    pin=json.loads((SOURCE/'model_pin.json').read_text())
    for item in pin['files']:
        path=checkpoint/item['name']
        if not path.is_file():continue
        assert path.stat().st_size==item['size']
        if item.get('sha256'):assert sha(path)==item['sha256']
        else:
            content=path.read_bytes();assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==item['git_blob_id']
    rig=Rig(checkpoint);rig.check=deadline
    tails={}
    tails['BF16']=torch.nn.ModuleList([copy.deepcopy(rig.layers[i]) for i in range(11,16)])
    tails['FP32']=copy.deepcopy(tails['BF16']).float()
    with np.load(ROOT/'panel_vectors.npz',allow_pickle=False) as z:legacy=torch.tensor(z['v'],dtype=torch.float32)
    questions=json.loads((PILOT/'COMPILED_QUESTIONS.json').read_text())

    def promote(value):
        copied=copy.deepcopy(value);seen=set()
        def walk(obj):
            if torch.is_tensor(obj):return obj.float() if obj.is_floating_point() else obj
            if isinstance(obj,tuple):return tuple(walk(v) for v in obj)
            if id(obj) in seen:return obj
            seen.add(id(obj))
            if isinstance(obj,dict):
                for k,v in list(obj.items()):
                    obj[k]=walk(v)
            elif isinstance(obj,list):
                for i,v in enumerate(obj):
                    obj[i]=walk(v)
            elif hasattr(obj,'__dict__'):
                for k,v in list(vars(obj).items()):
                    if isinstance(v,torch.dtype) and v in [torch.bfloat16,torch.float16,torch.float32]:setattr(obj,k,torch.float32)
                    else:setattr(obj,k,walk(v))
            return obj
        return walk(copied)

    def function_for(kwargs,precision):
        original=kwargs;original_hash=cache_hash(original.get('past_key_values'))
        fixed=promote(kwargs) if precision=='FP32' else detached_copy(kwargs)
        def function(h):
            deadline();STATUS['tail_evaluations']+=1
            local=detached_copy(fixed);cache=local.get('past_key_values');rope=local['position_embeddings']
            position_ids=local.get('position_ids');full_mask=local.get('attention_mask')
            with sdpa_kernel(SDPBackend.MATH):
                output=h
                for layer in tails[precision]:
                    output=layer(output,position_embeddings=rope,attention_mask=full_mask if layer.block_type=='full_attention' else None,
                        position_ids=position_ids,past_key_values=cache,use_cache=True)
            assert cache_hash(original.get('past_key_values'))==original_hash
            return output[0,-1]
        return function

    report('matched_snapshot_calibration')
    h,target,kwargs=load_context(questions[0]['id'],'prefill')
    f16=function_for(kwargs,'BF16');f32=function_for(kwargs,'FP32')
    with torch.no_grad():b16=f16(h);b32=f32(h.float())
    replay=norm(array(b16.double()-target.double()))/max(1.,norm(array(target)))
    assert replay<=1e-5,'Native tail snapshot replay mismatch'
    tangent=torch.zeros_like(h,dtype=torch.float32);tangent[0,-1]=legacy
    with torch.no_grad(),torch.autograd.forward_ad.dual_level():
        dual=torch.autograd.forward_ad.make_dual(h.float(),tangent)
        value=f32(dual);_,derivative=torch.autograd.forward_ad.unpack_dual(value)
    assert derivative is not None
    with torch.no_grad():plus=f32(h.float()+.03125*tangent);minus=f32(h.float()-.03125*tangent)
    fd=(plus-minus)/(2*.03125);fd_error=relative(array(fd-derivative),array(derivative))
    assert fd_error is not None and fd_error<=.05,'Matched-state derivative gate failed'
    dump(ROOT/'NUMERICAL_GATE.json',{'pass':True,'native_replay_relative':replay,'FP32_FD_relative':fd_error})
    folder=ROOT/'snapshots';folder.mkdir();rows=[]
    for question in questions[2:]:
        pid=question['id']
        for position in ['prefill','decode1']:
            h,target,kwargs=load_context(pid,position);functions={p:function_for(kwargs,p) for p in ['BF16','FP32']}
            with torch.no_grad():bases={p:functions[p](h if p=='BF16' else h.float()) for p in functions}
            assert norm(array(bases['BF16'].double()-target.double()))/max(1.,norm(array(target)))<=1e-5
            predictions={};inputs={}
            for schedule in (['all','last'] if position=='prefill' else ['current']):
                for eps in [.03125,.5]:
                    for sign in [1,-1]:
                        alpha=sign*eps;update=alpha*legacy.bfloat16()
                        changed=h+update if schedule=='all' else torch.cat([h[:,:-1],h[:,-1:]+update.reshape(1,1,-1)],dim=1)
                        actual=changed.double()-h.double();tangent=actual.float()
                        with torch.no_grad(),torch.autograd.forward_ad.dual_level():
                            dual=torch.autograd.forward_ad.make_dual(h.float(),tangent)
                            value=functions['FP32'](dual);_,jdelta=torch.autograd.forward_ad.unpack_dual(value)
                        assert jdelta is not None
                        key=(schedule,eps,sign);inputs[key]=changed;predictions[key]=jdelta.detach()
                        path=folder/f'{pid}_{position}_{schedule}_eps{eps}_sign{sign}_prediction.npz'
                        np.savez_compressed(path,actual_delta=array(actual),J_actual_delta=array(jdelta),tangent_cast_error=array(actual-tangent.double()))
                        dump(path.with_suffix('.json'),{'prediction_saved_before_responses':True,'sha256':sha(path)})
            for key,changed in inputs.items():
                schedule,eps,sign=key
                for precision in ['BF16','FP32']:
                    with torch.no_grad():measured=functions[precision](changed if precision=='BF16' else changed.float())
                    delta=measured.double()-bases[precision].double();prediction=predictions[key]
                    name=f'{pid}_{position}_{schedule}_eps{eps}_sign{sign}_{precision}'
                    np.savez_compressed(folder/(name+'.npz'),response=array(delta),base_output=array(bases[precision]),J_actual_delta=array(prediction))
                    rows.append(dict(question=pid,position=position,schedule=schedule,strength=eps,sign=sign,precision=precision,file=name+'.npz',
                        response_norm=norm(array(delta)),linear_absolute_error=norm(array(delta-prediction.double())),
                        linear_relative_error=relative(array(delta-prediction.double()),array(delta))))
            STATUS['completed_snapshots']+=1;report('matched_snapshot',current_question=pid,position=position)
    assert STATUS['tail_evaluations']==85,STATUS['tail_evaluations']
    dump(ROOT/'SNAPSHOT_INDEX.json',rows)

    old_panel=json.loads((SOURCE/'ORIGINAL_PANEL.json').read_text());old_question=old_panel['prompts'][0]
    ids=old_question['prefix_ids'];tokens=old_panel['forced_continuation_ids'];assert old_question['id']=='pair1_completed' and len(ids)==109 and len(tokens)==28
    old_folder=ROOT/'historical_reproduction';old_folder.mkdir();reproductions=[]
    for precision,dtype in [('BF16',torch.bfloat16),('FP32',torch.float32)]:
        rig.precision(dtype);baselines=None
        for condition in ['base','plus_v','minus_v']:
            cache=None;readouts=[];before=[];deliveries=[];logits=[];audits=[]
            def hook(module,args,output):
                changed,audit=native_intervention(output,condition,{'v':legacy},old_panel)
                deliveries.append(array(changed[0,-1].double()-output[0,-1].double()))
                audits.append(audit.tolist() if audit is not None else None)
                return changed
            for step in range(28):
                deadline();current=ids if step==0 else [tokens[step-1]]
                answer=rig.forward(current,len(ids)+step,cache,capture=True,edit=(10,hook));cache=answer['cache']
                readouts.append(array(answer['data']['readout15']));before.append(array(answer['data']['site10']));logits.append(array(answer['logits']))
                STATUS['whole_backbone_calls']=rig.calls
            before=np.stack(before)
            if baselines is None:baselines=before.copy()
            else:assert np.array_equal(before,baselines),'Unexpected upstream state change under identical history'
            name=f'{precision}_{condition}.npz'
            np.savez_compressed(old_folder/name,readout15=np.stack(readouts),site10_before=before,delivered=np.stack(deliveries),logits=np.stack(logits),forced_ids=np.asarray(tokens),prefix_ids=np.asarray(ids))
            dump(old_folder/name.replace('.npz','.json'),{'precision':precision,'condition':condition,'delivery_audits':audits,'sha256':sha(old_folder/name)})
            reproductions.append({'precision':precision,'condition':condition,'file':name});STATUS['completed_reproductions']+=1
            report('historical_reproduction',precision=precision,condition=condition)
    assert rig.calls==168
    dump(ROOT/'HISTORICAL_REPRODUCTION_INDEX.json',reproductions)
    STATUS['status']='complete';report('complete_export_pending')
except BaseException as error:
    STATUS['status']='fault_or_gate_stop_no_automatic_retry';report('fault',error=type(error).__name__+': '+str(error))
    (ROOT/'FAULT.txt').write_text(traceback.format_exc());raise
finally:export();timer.cancel()
