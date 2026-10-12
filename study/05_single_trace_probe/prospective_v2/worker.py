"""One owned, bounded frozen-generator collection; never fits or scores the probe."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

root=Path(sys.argv[1]);started=float(sys.argv[2]);cloud=Path(sys.argv[3])
sys.path.insert(0,str(root/'source'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def report(stage,**extra):
    receipt={'stage':stage,'elapsed_including_setup_seconds':time.monotonic()-started,**extra}
    (root/'WORKER.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

def deadline():
    if time.monotonic()-started>=2400:raise TimeoutError('Stop before five-minute export reserve')

def preserve(folder):
    for p in folder.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        q=cloud/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists() and sha(q)==sha(p):continue
        temporary=q.with_name(q.name+'.copying');shutil.copyfile(p,temporary)
        assert sha(temporary)==sha(p);temporary.replace(q)

def export():
    files=[p for p in root.iterdir() if p.is_file() and p.name not in ['EXPORT_RECEIPT.json']]
    for name in ['source','execution','probe','mechanics','native']:
        files += [p for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    manifest={p.relative_to(root).as_posix():sha(p) for p in files}
    path=root.with_name(root.name+'_export.zip')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:z.write(p,p.relative_to(root).as_posix())
        z.writestr('EXPORT_HASHES.json',json.dumps(manifest,indent=2))
    dest=cloud/path.name;shutil.copyfile(path,dest);assert sha(path)==sha(dest)
    receipt={'zip':path.name,'sha256':sha(path),'size':path.stat().st_size,'member_files':len(manifest),
        'Drive_full_bytes_verified':True,'elapsed':time.monotonic()-started}
    (root/'EXPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2));shutil.copyfile(root/'EXPORT_RECEIPT.json',cloud/'EXPORT_RECEIPT.json')
    print('R4_PROBE_V2_EXPORT',json.dumps(receipt),flush=True)

try:
    freeze=json.loads((root/'EXECUTION_FREEZE.json').read_text())
    for name,digest in freeze['sources'].items():assert sha(root/name)==digest,name
    plan=json.loads((root/'source/NEXT_VALIDATION_PLAN.json').read_text())
    assert plan['candidate_probe_sha256']==sha(root/'probe/probe.npz')
    assert plan['candidate_schema_sha256']==sha(root/'probe/schema.json')
    assert plan['vectors_sha256']==sha(root/'panel_vectors.npz')
    deadline();report('install_exact_runtime')
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','transformers==5.19.0','bitsandbytes==0.50.2'],check=True,timeout=420)
    from huggingface_hub import snapshot_download
    from collect import verify_checkpoint,token_ids
    pin=json.loads((root/'source/model_pin.json').read_text());checkpoint=root/'checkpoint'
    deadline();report('download_exact_public_checkpoint')
    snapshot_download(repo_id=pin['repository'],revision=pin['revision'],local_dir=checkpoint,token=False,max_workers=2,
        allow_patterns=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','merges.txt','vocab.json','model.safetensors.index.json','*.safetensors'])
    deadline();report('hash_verify_checkpoint_before_loading')
    identity=verify_checkpoint(checkpoint);assert identity['revision']==plan['revision']
    (root/'IDENTITY.json').write_text(json.dumps(identity,indent=2))
    import numpy as np
    import torch
    import transformers
    import bitsandbytes
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration,LogitsProcessor,LogitsProcessorList
    from panel_collect import native_intervention
    assert transformers.__version__=='5.19.0' and bitsandbytes.__version__=='0.50.2' and torch.cuda.is_bf16_supported()
    report('load_one_frozen_BF16_model',torch=torch.__version__,transformers=transformers.__version__,GPU=torch.cuda.get_device_name(0))
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False,
        use_safetensors=True,device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa')
    model.eval();model.requires_grad_(False)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert not getattr(model,'peft_config',None) and not getattr(model,'is_quantized',False)
    assert next(model.parameters()).dtype==torch.bfloat16
    layers=model.model.language_model.layers;assert len(layers)==24
    device=next(model.parameters()).device
    control=json.loads((root/'source/ORIGINAL_PANEL.json').read_text())
    with np.load(root/'panel_vectors.npz',allow_pickle=False) as z:vectors={k:torch.tensor(z[k],dtype=torch.float32) for k in z.files}
    assert vectors['v'].numel()==1024 and abs(float(vectors['v'].norm())-1)<1e-5
    tokens=plan['forced_continuation_ids'];assert len(tokens)==8
    compiled=[]
    for prompt in plan['prompts']:
        prefix=token_ids(tokenizer.apply_chat_template([{'role':'system','content':plan['system']},{'role':'user','content':prompt['text']}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False))
        compiled.append({**prompt,'prefix_ids':prefix,'prefix_length':len(prefix)})
    (root/'COMPILED_PROMPTS.json').write_text(json.dumps(compiled,indent=2))
    def replay(prefix_ids,condition,*,record=True,null_hook=False):
        deadline();prefix=torch.tensor([prefix_ids],device=device)
        captures=[[] for _ in layers];before=[];audits=[];logits=[];handles=[]
        class Forced(LogitsProcessor):
            def __call__(self,input_ids,scores):
                deadline()
                if not record:logits.append(scores.detach().clone())
                index=input_ids.shape[1]-prefix.shape[1]
                forced=torch.full_like(scores,-torch.inf);forced[:,tokens[index]]=0;return forced
        def hook(module,args,result):
            h=result[0] if isinstance(result,tuple) else result
            if record:before.append(h[0,-1].detach().clone())
            if null_hook:return result
            changed,audit=native_intervention(h,condition,vectors,control)
            if audit is not None:audits.append(audit)
            return (changed,)+result[1:] if isinstance(result,tuple) else changed
        if record or null_hook:handles.append(layers[10].register_forward_hook(hook))
        if record:
            for i,layer in enumerate(layers):
                def recorder(module,args,result,i=i):
                    h=result[0] if isinstance(result,tuple) else result
                    captures[i].append(h[0,-1].detach().clone())
                handles.append(layer.register_forward_hook(recorder))
        try:
            with torch.inference_mode():
                generated=model.generate(input_ids=prefix,attention_mask=torch.ones_like(prefix),max_new_tokens=8,
                    do_sample=False,use_cache=True,logits_processor=LogitsProcessorList([Forced()]))
            assert generated[0,prefix.shape[1]:].cpu().tolist()==tokens
            if not record:return torch.stack(logits).float().cpu().numpy()
            h=np.stack([torch.stack(c).float().cpu().numpy() for c in captures]);pre=torch.stack(before).float().cpu().numpy()
            assert h.shape==(24,8,1024) and np.isfinite(h).all()
            return h,pre,torch.stack(audits).cpu().tolist() if audits else []
        finally:
            for handle in handles:handle.remove()
    mechanics=root/'mechanics';mechanics.mkdir()
    checks={'status':'running','completed':[],'separate_from_study':True};arrays={};pres={}
    report('two_native_mechanics_traces')
    for condition in ['base','plus_v']:
        h,pre,audits=replay(compiled[0]['prefix_ids'],condition)
        path=mechanics/(condition+'.npz');np.savez_compressed(path,residuals=h,pre_injection=pre,forced_token_ids=tokens)
        arrays[condition]=h;pres[condition]=pre
        checks['completed'].append({'condition':condition,'sha256':sha(path),'delivery':audits})
    assert np.array_equal(arrays['base'][:10],arrays['plus_v'][:10]) and np.array_equal(pres['base'],pres['plus_v'])
    checks['upstream_pre_injection_exact']=True;checks['status']='passed'
    (mechanics/'STATUS.json').write_text(json.dumps(checks,indent=2));preserve(mechanics)
    report('two_untreated_null_calls')
    untreated=replay(compiled[0]['prefix_ids'],'base',record=False)
    null=replay(compiled[0]['prefix_ids'],'base',record=False,null_hook=True)
    assert np.array_equal(untreated,null)
    native=root/'native';native.mkdir();receipt={'status':'running','null_hook_logits_exact':True,'null_calls':2,'mechanics_passed':2,'completed':[]}
    def save():
        receipt['elapsed_including_setup_seconds']=time.monotonic()-started
        temp=native/'STATUS.json.tmp';temp.write_text(json.dumps(receipt,indent=2));temp.replace(native/'STATUS.json')
    save();report('sixteen_primary_eight_step_traces')
    for prompt in compiled:
        folder=native/prompt['id'];folder.mkdir();a={};p={}
        for condition in prompt['condition_order']:
            begin=time.monotonic();h,pre,audits=replay(prompt['prefix_ids'],condition)
            path=folder/(condition+'.npz');np.savez_compressed(path,residuals=h,pre_injection=pre,forced_token_ids=tokens)
            receipt['completed'].append({'prompt':prompt['id'],'condition':condition,'shape':list(h.shape),'sha256':sha(path),'seconds':time.monotonic()-begin,'delivery':audits})
            a[condition]=h;p[condition]=pre;save()
        assert np.array_equal(a['base'][:10],a['plus_v'][:10]) and np.array_equal(p['base'],p['plus_v'])
        preserve(native)
    receipt['status']='complete';receipt['upstream_and_pre_injection_exact_all_prompts']=True;save()
    assert len(receipt['completed'])==16
    report('collection_complete_no_probe_scoring',primary_traces=16,mechanics=2,null_calls=2)
except BaseException as error:
    report('fault_stop_no_retry',error=type(error).__name__+': '+str(error));traceback.print_exc();raise
finally:
    for name in ['mechanics','native']:
        if (root/name).exists():preserve(root/name)
    export()
