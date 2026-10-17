"""One bounded local CPU native validation. Frozen model; no learned detector."""
from pathlib import Path
import hashlib,json,os,sys,threading,time,traceback,zipfile

ROOT=Path(sys.argv[1]);SOURCE=ROOT/'source'
sys.path.insert(0,str(SOURCE))
started=json.loads((ROOT/'LIFECYCLE.json').read_text())['started_epoch']
plan=json.loads((SOURCE/'PLAN.json').read_text())
status={'status':'starting','completed':[],'unmonitored_controls':[],'GPU_allocations':0}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def dump(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def report(stage,**extra):
    status['stage']=stage;status['elapsed_including_setup_seconds']=time.time()-started
    status.update(extra);dump(ROOT/'STATUS.json',status)
    print(json.dumps({'stage':stage,'completed':len(status['completed']),'elapsed':status['elapsed_including_setup_seconds'],**extra}),flush=True)
def deadline():
    if time.time()-started>=plan['resources']['native_execution_stop_seconds']:raise TimeoutError('Preserve export reserve')
def hard_stop():
    dump(ROOT/'HARD_TIMEOUT.json',{'elapsed':time.time()-started,'GPU_allocations':0,'status':'CPU hard cap reached; partial files preserved'})
    os._exit(124)
remaining=plan['resources']['hard_wall_seconds_including_setup']-(time.time()-started)
if remaining<=0:raise TimeoutError('Setup already consumed CPU study cap')
timer=threading.Timer(remaining,hard_stop);timer.daemon=True;timer.start()

def export():
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['EXPORT_HASHES.json','EXPORT_RECEIPT.json']]
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files};dump(ROOT/'EXPORT_HASHES.json',manifest)
    archive=ROOT.with_name(ROOT.name+'_native_export.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.write(ROOT/'EXPORT_HASHES.json','EXPORT_HASHES.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for member,digest in manifest.items():assert hashlib.sha256(z.read(member)).hexdigest()==digest
    receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'manifest_files':len(manifest),
        'elapsed_including_setup_seconds':time.time()-started,'full_member_hashes_verified':True,'GPU_allocations':0}
    dump(ROOT/'EXPORT_RECEIPT.json',receipt);print('R4_CONSISTENCY_CPU_EXPORT',json.dumps(receipt),flush=True)

try:
    freeze=json.loads((ROOT/'EXECUTION_FREEZE.json').read_text())
    for member,digest in freeze['sources'].items():assert sha(ROOT/member)==digest,member
    deadline();report('import_pinned_CPU_runtime')
    import numpy as np
    import torch,transformers
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from monitor import BoundaryMonitor
    from verify_monitor import verify
    assert torch.__version__=='2.13.0+cpu' and transformers.__version__=='5.19.0'
    assert not torch.cuda.is_available()
    torch.set_num_threads(plan['resources']['CPU_threads']);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dump(ROOT/'MONITOR_UNIT_CHECKS.json',verify())
    modelpath=Path(json.loads((ROOT/'IDENTITY.json').read_text())['model_path'])
    pin=json.loads((SOURCE/'model_pin.json').read_text());verified=[]
    for item in pin['files']:
        p=modelpath/item['name']
        if not p.is_file():continue
        assert p.stat().st_size==item['size']
        if item.get('sha256'):assert sha(p)==item['sha256'],item['name']
        else:
            b=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==item['git_blob_id']
        verified.append(item['name'])
    assert len(verified)==8 and pin['revision']==plan['revision']
    deadline();report('load_one_frozen_BF16_CPU_model')
    tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True,trust_remote_code=False)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(modelpath,local_files_only=True,trust_remote_code=False,
        use_safetensors=True,dtype=torch.bfloat16,attn_implementation='sdpa')
    model.eval();model.requires_grad_(False)
    assert all(p.device.type=='cpu' and not p.requires_grad for p in model.parameters())
    assert next(model.parameters()).dtype==torch.bfloat16 and not model.training
    assert not getattr(model,'peft_config',None) and not getattr(model,'is_quantized',False)
    layers=model.model.language_model.layers;assert len(layers)==24
    boundaries={f'block{i:02d}':layer for i,layer in enumerate(layers)}
    dump(ROOT/'RUNTIME.json',{'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,
        'device':'cpu','dtype':'bfloat16','threads':torch.get_num_threads(),'revision':pin['revision'],
        'model_class':type(model).__name__,'block_types':[getattr(layer,'layer_type',None) for layer in layers],
        'quantized':False,'adapter':False,'training':False,'GPU_allocations':0})
    vectorpath=ROOT/'panel_vectors.npz'
    assert sha(vectorpath)=='0f4311c6c8aef95c57bb3183dff119e50ddb5c0c472136988fa72fc689ca99eb'
    with np.load(vectorpath,allow_pickle=False) as z:v=np.asarray(z['v'],dtype=np.float32)
    assert v.shape==(1024,) and abs(np.linalg.norm(v)-1)<1e-5
    random=np.random.default_rng(20261012).standard_normal(1024).astype(np.float32);random/=np.linalg.norm(random)
    np.savez_compressed(ROOT/'DIRECTIONS_USED.npz',v=v,random=random)
    v=torch.tensor(v,dtype=torch.bfloat16);random=torch.tensor(random,dtype=torch.bfloat16)
    compiled=[]
    for p in plan['prompts']:
        prefix=tokenizer.apply_chat_template([{'role':'system','content':plan['system']},{'role':'user','content':p['text']}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if hasattr(prefix,'keys'):prefix=prefix['input_ids']
        if hasattr(prefix,'tolist'):prefix=prefix.tolist()
        if prefix and isinstance(prefix[0],list):assert len(prefix)==1;prefix=prefix[0]
        assert prefix and all(isinstance(i,int) for i in prefix)
        compiled.append({**p,'prefix_ids':prefix,'prefix_length':len(prefix)})
    dump(ROOT/'COMPILED_PROMPTS.json',compiled)

    def run(prefix,block,condition,monitored=True):
        deadline();begin=time.time();deliveries=[];delivered_last=[]
        def change(module,args,result):
            h=result[0] if isinstance(result,tuple) else result
            if condition=='null_hook':return result
            if condition=='plus_v_0.125':modified=h+.125*v
            elif condition=='plus_v_0.5':modified=h+.5*v
            elif condition=='minus_v_0.5':modified=h-.5*v
            elif condition=='random_0.25':modified=h+.25*random
            elif condition=='scale_1.01':modified=h*1.01
            else:raise ValueError('Unknown intervention')
            actual=modified.double()-h.double();energy=.5*actual.square().sum(dim=-1)
            deliveries.append({'positions':energy.numel(),'energy_sum':float(energy.sum()),'nonzero_positions':int(torch.count_nonzero(energy)),
                'last_update_norm':float(actual[0,-1].norm()),'max_absolute':float(actual.abs().max())})
            delivered_last.append(actual[0,-1].float().cpu().numpy().copy())
            return (modified,)+result[1:] if isinstance(result,tuple) else modified
        interventions={} if condition=='untreated' else {f'block{block:02d}':[change]}
        monitor=BoundaryMonitor(boundaries,interventions=interventions,keep_last_vectors=True) if monitored else None
        input_ids=torch.tensor([prefix],dtype=torch.long)
        with torch.inference_mode():
            if monitor is not None:monitor.__enter__()
            try:
                answer=model.generate(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),do_sample=False,use_cache=True,
                    min_new_tokens=8,max_new_tokens=8,return_dict_in_generate=True,output_logits=True)
            finally:
                if monitor is not None:monitor.close()
        assert all(not layer._forward_hooks for layer in layers)
        tokens=answer.sequences[0,len(prefix):].cpu().numpy();assert len(tokens)==8
        logits=torch.stack(answer.logits).float().cpu().numpy();assert len(logits)==8
        result={'tokens':tokens,'logits':logits,'seconds':time.time()-begin,'deliveries':deliveries}
        if monitor is not None:
            assert len(monitor.records)==24*8
            result.update(summary=monitor.summary(),records=monitor.records,
                expected=np.stack([np.stack([pair[0].numpy() for pair in monitor.last_vectors[name]]) for name in boundaries]),
                observed=np.stack([np.stack([pair[1].numpy() for pair in monitor.last_vectors[name]]) for name in boundaries]),
                delivered_last=np.asarray(delivered_last,dtype=np.float32))
            assert result['expected'].shape==(24,8,1024) and result['observed'].shape==(24,8,1024)
            if deliveries:
                assert np.array_equal(result['observed'][block].astype(np.float64)-result['expected'][block].astype(np.float64),result['delivered_last'].astype(np.float64))
        return result

    controls=ROOT/'unmonitored_controls';controls.mkdir(exist_ok=False)
    native=ROOT/'native';native.mkdir(exist_ok=False)
    conditions=plan['conditions']
    for pi,prompt in enumerate(compiled):
        deadline();report('unmonitored_control',current_prompt=prompt['id'])
        reference=run(prompt['prefix_ids'],None,'untreated',False)
        reference_hash=hashlib.sha256(reference['logits'].tobytes()).hexdigest()
        path=controls/(prompt['id']+'.npz')
        np.savez_compressed(path,generated_token_ids=reference['tokens'],raw_generation_logits=reference['logits'])
        status['unmonitored_controls'].append({'prompt':prompt['id'],'seconds':reference['seconds'],'sha256':sha(path),'logits_sha256':reference_hash})
        for bi,block in enumerate(plan['injection_blocks_zero_based']):
            offset=(pi+bi)%len(conditions);order=conditions[offset:]+conditions[:offset]
            folder=native/prompt['id']/f'block{block:02d}';folder.mkdir(parents=True,exist_ok=False)
            for condition in order:
                deadline();r=run(prompt['prefix_ids'],block,condition)
                modified=condition not in ['untreated','null_hook'];summary=r['summary']
                assert summary['computation_mismatch']==modified
                assert summary['mismatch_boundaries']==([f'block{block:02d}'] if modified else [])
                logits_hash=hashlib.sha256(r['logits'].tobytes()).hexdigest()
                if not modified:
                    assert np.array_equal(reference['tokens'],r['tokens']) and np.array_equal(reference['logits'],r['logits'])
                    assert logits_hash==reference_hash
                else:assert len(r['deliveries'])==8 and sum(d['nonzero_positions'] for d in r['deliveries'])>0
                path=folder/(condition+'.npz')
                np.savez_compressed(path,expected_output=r['expected'],observed_output=r['observed'],
                    actual_delivered_last=r['delivered_last'],generated_token_ids=r['tokens'])
                receipt={'prompt':prompt['id'],'block_zero_based':block,'condition':condition,'intended_modified':modified,
                    'seconds':r['seconds'],'summary':summary,'boundary_records':r['records'],'delivery':r['deliveries'],
                    'raw_logits_sha256':logits_hash,'unmonitored_logits_sha256':reference_hash,
                    'normal_noninterference_exact':True if not modified else None,'npz_sha256':sha(path)}
                dump(folder/(condition+'.json'),receipt)
                status['completed'].append({k:receipt[k] for k in ['prompt','block_zero_based','condition','intended_modified','seconds','summary','normal_noninterference_exact','npz_sha256']})
                report('native_validation',current_prompt=prompt['id'],current_block=block,current_condition=condition)
    assert len(status['completed'])==126 and len(status['unmonitored_controls'])==6
    status['status']='complete';report('complete_export_pending')
    del model
except BaseException as error:
    status['status']='fault_stop_no_scientific_retry'
    report('fault',error=type(error).__name__+': '+str(error))
    (ROOT/'FAULT.txt').write_text(traceback.format_exc())
    raise
finally:
    export();timer.cancel()
