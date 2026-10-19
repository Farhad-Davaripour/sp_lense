"""One bounded native-BF16 GPU comparison. No adapter or optimization code."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import zipfile


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def dump(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def encode(tokenizer,chat):
    ids=tokenizer.apply_chat_template(chat,tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if hasattr(ids,'keys'):ids=ids['input_ids']
    if ids and isinstance(ids[0],list):ids=ids[0]
    return ids


def fingerprint(model,torch):
    result=hashlib.sha256()
    for name,value in model.named_parameters():
        result.update(name.encode());result.update(str(value.dtype).encode())
        flat=value.detach().reshape(-1)
        for start in range(0,flat.numel(),4*1024*1024):
            result.update(flat[start:start+4*1024*1024].view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


def run(inputs,output,cache,allocation_deadline=None):
    import numpy as np
    import psutil
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from torch.nn.attention import SDPBackend,sdpa_kernel
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    inputs,output,cache=map(Path,[inputs,output,cache]);cfg=read(inputs/'CONFIG.json')
    drive_root=Path('/content/drive/MyDrive/Research4/recognition_capacity_27b_v1')
    if not Path('/content/drive/MyDrive').is_dir() or not output.resolve().is_relative_to(drive_root.resolve()):raise ValueError('A verified, dedicated Research 4 Drive destination is required')
    if not inputs.resolve().is_relative_to(Path('/content/research4_capacity_27b_v1').resolve()) or not cache.resolve().is_relative_to(Path('/content/research4_capacity_27b_v1').resolve()):raise ValueError('Capacity inputs/cache must stay in the owned VM directory')
    if output.exists():raise FileExistsError('No scientific retries into an existing result')
    for name,value in read(inputs/'MANIFEST.json').items():
        if sha(inputs/name)!=value:raise ValueError('Changed capacity input: '+name)
    if cfg['precision']!='bfloat16' or cfg['quantization'] is not None:raise ValueError('This release admits only unquantized BF16')
    if transformers.__version__!='5.19.0':raise ValueError('Install the frozen Transformers release before starting')
    output.mkdir(parents=True);cache.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();deadline=started+cfg['worker_stop_seconds']-cfg['export_reserve_seconds']
    if allocation_deadline is not None:
        deadline=min(deadline,started+allocation_deadline-time.time()-cfg['export_reserve_seconds'])
    receipt={'status':'running','scope':cfg['scope'],'optimizer_updates':0,'policy_actions_executed':0,'adapters_loaded':0,'models':[]}
    binding=read(inputs/'BINDING_QUERIES.json');numeric=read(inputs/'NUMERIC_CASES.json');pins=read(inputs/'MODEL_PINS.json')
    model=None;layers=None;handles=[];active=None
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('Protected capacity export boundary reached')
    def progress(stage,**values):
        record={'stage':stage,'seconds':time.monotonic()-started,**values};dump(output/'PROGRESS.json',record);print(json.dumps(record),flush=True)
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('A BF16-capable admitted GPU is required')
        free,total=torch.cuda.mem_get_info();host=psutil.virtual_memory().available;disk=shutil.disk_usage(cache).free
        admission={'GPU':torch.cuda.get_device_name(0),'GPU_total_bytes':total,'GPU_free_bytes':free,'host_free_bytes':host,'disk_free_bytes':disk}
        dump(output/'ADMISSION.json',admission)
        if total<cfg['minimum_gpu_total_bytes'] or free<cfg['minimum_gpu_free_bytes'] or host<cfg['minimum_host_free_bytes'] or disk<cfg['minimum_disk_free_bytes']:
            raise MemoryError('Capacity worker rejected hardware before any model load or forward')
        torch.manual_seed(cfg['seed']);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
        dump(output/'SOFTWARE.json',{'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'device':admission['GPU'],'dtype':'bfloat16','quantization':None})
        dump(output/'PRECALL_FREEZE.json',{'config':cfg,'input_manifest_sha256':sha(inputs/'MANIFEST.json'),'worker_sha256':sha(Path(__file__))})
        for pin in pins:
            check();tag=pin['tag'];folder=output/tag;folder.mkdir();records=[];answers=[];features=[];keys=[];forwards=0
            active={'model':tag,'status':'started','backbone_forwards':0,'binding_queries':0,'numeric_cases':0}
            progress('download',model=tag)
            model_path=Path(snapshot_download(pin['repository'],revision=pin['revision'],cache_dir=str(cache),allow_patterns=[f['name'] for f in pin['files']],token=False,max_workers=4))
            for file in pin['files']:
                check();path=model_path/file['name']
                if not path.is_file() or path.stat().st_size!=file['size']:raise ValueError('Pinned model file missing')
                if file['sha256'] and sha(path)!=file['sha256']:raise ValueError('Pinned model digest mismatch')
                if file['sha256'] is None and file.get('git_blob_id'):
                    data=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                    if blob!=file['git_blob_id']:raise ValueError('Pinned model metadata blob differs')
            check();progress('load',model=tag)
            tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
            model=Qwen3_5ForConditionalGeneration.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True).eval().requires_grad_(False)
            layers=model.model.language_model.layers
            if len(layers)!=pin['layers'] or model.config.text_config.hidden_size!=pin['hidden_size']:raise ValueError('Unexpected model architecture')
            if any(p.requires_grad or p.device.type!='cuda' for p in model.parameters()):raise ValueError('Unexpected parameter state')
            before=fingerprint(model,torch);letter_ids=[tokenizer.encode(c,add_special_tokens=False) for c in 'ABCDEF']
            if any(len(v)!=1 for v in letter_ids):raise ValueError('Letter tokenization differs')
            letter_ids=[v[0] for v in letter_ids];eos=set()
            for value in [model.generation_config.eos_token_id,tokenizer.eos_token_id]:
                if value is not None:eos.update([value] if isinstance(value,int) else value)
            if not eos:raise ValueError('Missing native EOS')
            dump(folder/'TOKEN_PROTOCOL.json',{'letter_ids':letter_ids,'eos_token_ids':sorted(eos),'model_pin':pin})
            captured={}
            def hook(module,args,out):captured['h']=(out[0] if isinstance(out,tuple) else out)[0,-1].detach().float().cpu().numpy().copy()
            handles=[layers[-1].register_forward_hook(hook)]
            def forward(ids,key):
                nonlocal forwards
                check()
                if len(ids)>cfg['maximum_tokens'] or forwards>=cfg['maximum_forwards_per_model']:raise ValueError('Frozen forward/context ceiling exceeded')
                if torch.cuda.mem_get_info()[0]<cfg['live_gpu_free_floor_bytes']:raise MemoryError('Live GPU export reserve reached')
                tensor=torch.tensor([ids],dtype=torch.long,device='cuda:0');captured.clear();forwards+=1;active['backbone_forwards']=forwards
                with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):result=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False,logits_to_keep=1,return_dict=True)
                logits=result.logits[0,-1].detach().float()
                if not torch.isfinite(logits).all() or captured['h'].shape!=(pin['hidden_size'],) or not np.isfinite(captured['h']).all():raise ValueError('Nonfinite capacity trace')
                features.append(captured['h']);keys.append(key)
                return logits
            def save_features():
                target=folder/'FINAL_BLOCK_STATES.npz';temporary=folder/'FINAL_BLOCK_STATES.tmp.npz'
                np.savez_compressed(temporary,X=np.asarray(features,np.float32),keys=np.asarray(keys));temporary.replace(target)
            for query in binding:
                ids=encode(tokenizer,query['messages']);logits=forward(ids,query['key']);selected=logits[letter_ids]
                records.append({'key':query['key'],'prompt_sha256':query['prompt_sha256'],'prompt_tokens':len(ids),'input_token_sha256':digest(ids),'logits':selected.cpu().tolist(),'global_top_token_id':int(logits.argmax()),'offered_probability_mass':float((selected[:2].logsumexp(0)-logits.logsumexp(0)).exp())})
                active['binding_queries']=len(records)
                dump(folder/'BINDING_RECORDS.json',records)
                if len(records)%12==0:save_features();progress('binding',model=tag,queries=len(records),forwards=forwards)
            for case in numeric:
                prompt=encode(tokenizer,case['messages']);prefix=prompt.copy();generated=[];scores=[]
                for step in range(cfg['maximum_new_tokens']):
                    logits=forward(prefix,case['case_id']+':'+str(step));token=int(logits.argmax());generated.append(token)
                    scores.append({'token_id':token,'logit':float(logits[token]),'probability':float((logits[token]-logits.logsumexp(0)).exp())})
                    if token in eos:break
                    prefix.append(token)
                answers.append({'case_id':case['case_id'],'task_family':case['task_family'],'prompt_sha256':digest(case['messages']),'prompt_tokens':len(prompt),'input_token_sha256':digest(prompt),'generated_token_ids':generated,'token_scores':scores,'decoded_text':tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False),'finished_with_eos':generated[-1] in eos})
                active['numeric_cases']=len(answers)
                dump(folder/'NUMERIC_RECORDS.json',answers);save_features();progress('numeric',model=tag,cases=len(answers),forwards=forwards)
            after=fingerprint(model,torch)
            if before!=after:raise ValueError('Frozen model changed')
            model_receipt={'status':'complete','model':tag,'repository':pin['repository'],'revision':pin['revision'],'base_fingerprint_before':before,'base_fingerprint_after':after,'binding_queries':len(records),'numeric_cases':len(answers),'backbone_forwards':forwards,'hidden_size':pin['hidden_size'],'optimizer_updates':0,'adapters_loaded':0,'policy_actions_executed':0,'precision':'bfloat16','quantization':None}
            dump(folder/'RUN_RECEIPT.json',model_receipt);receipt['models'].append(model_receipt)
            for handle in handles:handle.remove()
            handles=[];model=None;layers=None;gc.collect();torch.cuda.empty_cache();progress('model_released',model=tag);active=None
        receipt['status']='complete'
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));dump(output/'FAILURE.json',{'traceback':traceback.format_exc()})
        if active is not None:receipt['partial_model']=active
        if 'save_features' in locals() and 'features' in locals() and features:save_features()
    finally:
        for handle in handles:handle.remove()
        model=None;layers=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        receipt.update(seconds=time.monotonic()-started,model_objects_released=True,GPU_allocations=1 if torch.cuda.is_available() else 0)
        dump(output/'RUN_RECEIPT.json',receipt)
        manifest={str(f.relative_to(output)):sha(f) for f in output.rglob('*') if f.is_file() and f.name!='MANIFEST.json'}
        dump(output/'MANIFEST.json',manifest)
        archive=output.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
            for f in sorted(output.rglob('*')):
                if f.is_file():bundle.write(f,f.relative_to(output).as_posix())
        export={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'status':receipt['status']}
        dump(output.parent/(output.name+'_EXPORT.json'),export);print(json.dumps(export),flush=True)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['inputs','output','cache']:p.add_argument(name)
    p.add_argument('--allocation-deadline',type=float)
    a=p.parse_args();result=run(a.inputs,a.output,a.cache,a.allocation_deadline)
    if result['status']!='complete':raise SystemExit(1)
