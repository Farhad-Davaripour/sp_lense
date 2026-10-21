"""Databricks-only frozen 27B factual snapshots; no generated actions or fitting."""
import argparse,gc,hashlib,io,json,os,shutil,tempfile,time,traceback,zipfile
from pathlib import Path
import numpy as np
from .io_utils import read,dump,sha,verify_release
from .risk_readout_data import dataset
from .native27_tools import encode
from .recognition_capacity_worker import fingerprint

CONFIG='config/native27_risk_readout_v1.json'


def worker(root,output,cache,commit):
    import torch,transformers,bitsandbytes,psutil
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration,BitsAndBytesConfig
    from huggingface_hub import snapshot_download
    if not os.environ.get('MLFLOW_RUN_ID') or root.resolve()!=Path(os.environ['CODE_SOURCE_PATH']).resolve():raise ValueError('Databricks AIR required')
    verify_release(root,commit);cfg=read(root/CONFIG);pin=read(root/'inputs/MODEL_PIN.json')
    if cfg['model']['revision']!=pin['revision'] or cfg['model']['repository']!=pin['repository']:raise ValueError('Model pin differs')
    for name,value in read(root/'inputs/MANIFEST.json').items():
        if sha(root/'inputs'/name)!=value:raise ValueError('Input pin changed')
    output.mkdir();cache.mkdir();start=time.monotonic();model=layers=result=None;handles=[];features=[];tokens=[]
    count={'attempted':0,'completed':0}
    receipt={'status':'running','source_commit':commit,'backbone_optimizer_updates':0,'generated_actions':0,'behavioral_interventions':0,'research_adapters_loaded':0,'original_confirmation_cases_opened':0}
    def check():
        if time.monotonic()-start>=cfg['worker_stop_seconds']-cfg['export_reserve_seconds']:raise TimeoutError('Protected export deadline')
        if count['attempted']>=cfg['maximum_backbone_forwards']:raise TimeoutError('Forward ceiling')
        if torch.cuda.mem_get_info()[0]<cfg['live_gpu_free_floor_bytes']:
            torch.cuda.empty_cache()
            if torch.cuda.mem_get_info()[0]<cfg['live_gpu_free_floor_bytes']:raise MemoryError('Live GPU reserve')
    def progress(stage,**kw):
        value={'stage':stage,'seconds':time.monotonic()-start,'backbone_forwards':count['completed'],**kw};dump(output/'PROGRESS.json',value);print(json.dumps(value),flush=True)
    try:
        if transformers.__version__!='5.19.0' or bitsandbytes.__version__!='0.49.2':raise ValueError('Software pin differs')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 GPU required')
        free,total=torch.cuda.mem_get_info();state={'GPU':torch.cuda.get_device_name(0),'GPU_free_bytes':free,'GPU_total_bytes':total,'host_free_bytes':psutil.virtual_memory().available,'disk_free_bytes':shutil.disk_usage(cache).free};dump(output/'ADMISSION.json',state)
        for key,bound in [('GPU_total_bytes','minimum_gpu_total_bytes'),('GPU_free_bytes','minimum_gpu_free_bytes'),('host_free_bytes','minimum_host_free_bytes'),('disk_free_bytes','minimum_disk_free_bytes')]:
            if state[key]<cfg[bound]:raise MemoryError('Fresh resource admission: '+key)
        rows=dataset(cfg)
        if len(rows)!=cfg['maximum_backbone_forwards']:raise ValueError('Snapshot count differs')
        dump(output/'PRECALL_FREEZE.json',{'source_commit':commit,'config':cfg,'config_path':CONFIG,'pin':pin,'input_manifest_sha256':sha(root/'inputs/MANIFEST.json')})
        dump(output/'ROWS.json',rows);dump(output/'SOFTWARE.json',{'torch':torch.__version__,'transformers':transformers.__version__,'bitsandbytes':bitsandbytes.__version__,'numpy':np.__version__,'attention':'sdpa','thinking':False})
        torch.manual_seed(cfg['readout']['random_state']);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
        progress('download')
        path=Path(snapshot_download(pin['repository'],revision=pin['revision'],cache_dir=str(cache),allow_patterns=[f['name'] for f in pin['files']],token=False,max_workers=4))
        verified=[];meta=output/'model_metadata';meta.mkdir()
        for spec in pin['files']:
            check();f=path/spec['name']
            if f.stat().st_size!=spec['size']:raise ValueError('File size changed')
            actual=sha(f)
            if spec['sha256'] and actual!=spec['sha256']:raise ValueError('Model digest changed')
            if not spec['sha256'] and spec.get('git_blob_id'):
                data=f.read_bytes()
                if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=spec['git_blob_id']:raise ValueError('Metadata changed')
            verified.append({'name':spec['name'],'size':f.stat().st_size,'sha256':actual})
            if not f.name.endswith('.safetensors'):shutil.copyfile(f,meta/f.name)
        dump(output/'MODEL_FILE_VERIFICATION.json',verified);progress('load')
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        model=Qwen3_5ForConditionalGeneration.from_pretrained(path,local_files_only=True,trust_remote_code=False,use_safetensors=True,quantization_config=quant,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True).eval().requires_grad_(False)
        layers=model.model.language_model.layers
        if len(layers)!=64 or model.config.text_config.hidden_size!=5120 or any(p.device.type!='cuda' or p.requires_grad for p in model.parameters()):raise ValueError('Model geometry or freeze differs')
        if not any(getattr(p,'quant_state',None) is not None for p in model.parameters()):raise ValueError('NF4 not applied')
        before=fingerprint(model,torch);receipt['base_fingerprint_before']=before
        captured={}
        def capture(key):
            def hook(module,args,out):captured[key]=(out[0] if isinstance(out,tuple) else out)[0,-1].detach().float().cpu().numpy().copy()
            return hook
        for name,index in [('primary',cfg['primary_block_index']),('secondary',cfg['secondary_block_index'])]:handles.append(layers[index].register_forward_hook(capture(name)))
        def pre(module,args,kwargs):check();count['attempted']+=1
        def post(module,args,out):
            count['completed']+=1
            if not torch.isfinite(out.logits).all():raise FloatingPointError('Nonfinite logits')
        handles.extend([model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post)])
        for row in rows:
            captured.clear();ids=encode(tokenizer,row['messages'])
            if len(ids)>cfg['maximum_context_tokens']:raise ValueError('Context ceiling')
            tensor=torch.tensor([ids],dtype=torch.long,device='cuda:0')
            with torch.inference_mode():result=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False,logits_to_keep=1,return_dict=True)
            if set(captured)!={'primary','secondary'} or any(v.shape!=(5120,) or not np.isfinite(v).all() for v in captured.values()):raise ValueError('Invalid hidden state')
            features.append({k:v.copy() for k,v in captured.items()});tokens.append({'state_id':row['state_id'],'input_token_ids':ids})
            result=None
            if len(features)%7==0:progress('snapshots',states=len(features),groups=len(features)//7)
        after=fingerprint(model,torch);receipt['base_fingerprint_after']=after
        if before!=after:raise ValueError('Frozen backbone changed')
        if count['completed']!=112 or count['attempted']!=112:raise ValueError('Snapshot forward accounting differs')
        receipt.update(status='complete',states=len(rows),split_sizes={s:sum(r['split']==s for r in rows) for s in ['training','development','heldout']})
    except Exception as e:
        receipt.update(status='failed',error_type=type(e).__name__,error=str(e));dump(output/'FAILURE.json',{'traceback':traceback.format_exc()})
    finally:
        for handle in handles:handle.remove()
        if features:np.savez_compressed(output/'FEATURES.npz',**{k:np.stack([r[k] for r in features]) for k in ['primary','secondary']})
        dump(output/'TOKENS.json',tokens);model=layers=result=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        receipt.update(backbone_forward_attempts=count['attempted'],backbone_forwards=count['completed'],seconds=time.monotonic()-start,worker_exiting=True,model_objects_released=True)
        dump(output/'RUN_RECEIPT.json',receipt)
    return receipt


def launch(root,commit,destination):
    from databricks.sdk import WorkspaceClient
    if not destination.startswith('/Volumes/workspace/research4/artifacts/native27_risk_readout_v1/') or '..' in destination.split('/'):raise ValueError('Dedicated destination required')
    client=WorkspaceClient();client.files.create_directory(destination)
    def upload(name,data):
        client.files.upload(destination+'/'+name,io.BytesIO(data),overwrite=False)
        with client.files.download(destination+'/'+name).contents as f:back=f.read()
        if back!=data:raise ValueError('Volume readback differs')
    with tempfile.TemporaryDirectory(prefix='research4-factual-risk-') as temporary:
        base=Path(temporary);output=base/'output'
        upload('ALLOCATION.json',json.dumps({'source_commit':commit,'MLFLOW_run_id':os.environ.get('MLFLOW_RUN_ID'),'scientific_retries':0}).encode())
        try:receipt=worker(Path(root),output,base/'cache',commit)
        except Exception as e:
            output.mkdir(exist_ok=True);receipt={'status':'failed','error_type':type(e).__name__,'error':str(e),'backbone_forwards':None};dump(output/'WRAPPER_FAILURE.json',{'traceback':traceback.format_exc()});dump(output/'RUN_RECEIPT.json',receipt)
        dump(output/'MANIFEST.json',{f.relative_to(output).as_posix():sha(f) for f in output.rglob('*') if f.is_file() and f.name!='MANIFEST.json'})
        zpath=base/'evidence.zip'
        with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(output.rglob('*')):
                if f.is_file():z.write(f,f.relative_to(output).as_posix())
        with zipfile.ZipFile(zpath) as z:
            if z.testzip() is not None:raise ValueError('CRC failure')
        upload('evidence.zip',zpath.read_bytes());upload('RUN_RECEIPT.json',(output/'RUN_RECEIPT.json').read_bytes())
        export={**receipt,'source_commit':commit,'archive_sha256':sha(zpath),'archive_bytes':zpath.stat().st_size,'destination':destination,'owned_temporary_cache_cleanup':True}
        upload('EXPORT_RECEIPT.json',json.dumps(export,indent=2).encode())
    print(json.dumps(export),flush=True)
    if receipt['status']!='complete':raise RuntimeError('Snapshot worker failed; output retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('commit');p.add_argument('destination');a=p.parse_args();launch(a.root,a.commit,a.destination)
