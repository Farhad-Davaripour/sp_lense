"""Bounded Databricks QA diagnosis. No actions, fitting or intervention."""
import argparse,gc,hashlib,json,os,shutil,time,traceback
from pathlib import Path
import numpy as np
from .io_utils import read,dump,sha,verify_release
from .factual_comprehension_data import SYSTEM,questions,parse_answer,summary
from .recognition_capacity_worker import encode,fingerprint
from .native27_tools import decode
from .risk_readout_worker import launch

CONFIG='config/native27_factual_comprehension_v1.json'
FINGERPRINT='c989470fab1c2fe4b62e590e2a6e173998943c353951bb7e919b8e8eca30d0fa'


def worker(root,output,cache,commit,config_path=CONFIG):
    import torch,transformers,bitsandbytes,psutil
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration,BitsAndBytesConfig
    from huggingface_hub import snapshot_download
    if not os.environ.get('MLFLOW_RUN_ID') or root.resolve()!=Path(os.environ['CODE_SOURCE_PATH']).resolve():raise ValueError('Databricks AIR required')
    verify_release(root,commit);cfg=read(root/config_path);pin=read(root/'inputs/MODEL_PIN.json')
    for n,h in read(root/'inputs/MANIFEST.json').items():
        if sha(root/'inputs'/n)!=h:raise ValueError('Frozen input differs')
    if pin['revision']!=cfg['model']['revision'] or pin['repository']!=cfg['model']['repository']:raise ValueError('Model pin differs')
    rows=read(root/'inputs/SOURCE_ROWS.json');queries=questions(rows,cfg)
    output.mkdir();cache.mkdir();start=time.monotonic();model=layers=result=None;handles=[];answers=[];features=[];transport=[];capture_at=None;captured={}
    count={'attempted':0,'completed':0};receipt={'status':'running','source_commit':commit,'new_fits':0,'backbone_optimizer_updates':0,'behavioral_interventions':0,'tool_actions_executed':0,'research_adapters_loaded':0,'original_heldout_scores_opened':False,'summary':None}
    def check():
        if time.monotonic()-start>=cfg['worker_stop_seconds']-cfg['export_reserve_seconds']:raise TimeoutError('Protected export deadline')
        if count['attempted']>=cfg['maximum_backbone_forwards']:raise TimeoutError('Forward ceiling')
        if torch.cuda.mem_get_info()[0]<cfg['live_gpu_free_floor_bytes']:
            torch.cuda.empty_cache()
            if torch.cuda.mem_get_info()[0]<cfg['live_gpu_free_floor_bytes']:raise MemoryError('Live GPU reserve')
    def progress(stage,**kw):
        v={'stage':stage,'seconds':time.monotonic()-start,'backbone_forwards':count['completed'],**kw};dump(output/'PROGRESS.json',v);print(json.dumps(v),flush=True)
    try:
        if transformers.__version__!='5.19.0' or bitsandbytes.__version__!='0.49.2' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('Pinned BF16 runtime required')
        free,total=torch.cuda.mem_get_info();state={'GPU':torch.cuda.get_device_name(0),'GPU_free_bytes':free,'GPU_total_bytes':total,'host_free_bytes':psutil.virtual_memory().available,'disk_free_bytes':shutil.disk_usage(cache).free};dump(output/'ADMISSION.json',state)
        for k,b in [('GPU_total_bytes','minimum_gpu_total_bytes'),('GPU_free_bytes','minimum_gpu_free_bytes'),('host_free_bytes','minimum_host_free_bytes'),('disk_free_bytes','minimum_disk_free_bytes')]:
            if state[k]<cfg[b]:raise MemoryError('Fresh admission: '+k)
        dump(output/'PRECALL_FREEZE.json',{'source_commit':commit,'config':cfg,'config_path':config_path,'pin':pin,'input_manifest_sha256':sha(root/'inputs/MANIFEST.json')})
        dump(output/'QUERIES.json',queries);dump(output/'SOFTWARE.json',{'torch':torch.__version__,'transformers':transformers.__version__,'bitsandbytes':bitsandbytes.__version__,'numpy':np.__version__,'thinking':False,'sampling':False,'attention':'sdpa'})
        torch.manual_seed(cfg['seed']);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;progress('download')
        path=Path(snapshot_download(pin['repository'],revision=pin['revision'],cache_dir=str(cache),allow_patterns=[f['name'] for f in pin['files']],token=False,max_workers=4))
        meta=output/'model_metadata';meta.mkdir();verified=[]
        for spec in pin['files']:
            check();f=path/spec['name'];actual=sha(f)
            if f.stat().st_size!=spec['size'] or (spec['sha256'] and actual!=spec['sha256']):raise ValueError('Pinned file differs')
            if not spec['sha256'] and spec.get('git_blob_id'):
                data=f.read_bytes()
                if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=spec['git_blob_id']:raise ValueError('Pinned metadata differs')
            verified.append({'name':spec['name'],'size':f.stat().st_size,'sha256':actual})
            if not f.name.endswith('.safetensors'):shutil.copyfile(f,meta/f.name)
        dump(output/'MODEL_FILE_VERIFICATION.json',verified);progress('load')
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        model=Qwen3_5ForConditionalGeneration.from_pretrained(path,local_files_only=True,trust_remote_code=False,use_safetensors=True,quantization_config=quant,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True).eval().requires_grad_(False)
        layers=model.model.language_model.layers
        if len(layers)!=64 or model.config.text_config.hidden_size!=5120 or any(p.device.type!='cuda' or p.requires_grad for p in model.parameters()) or not any(getattr(p,'quant_state',None) is not None for p in model.parameters()):raise ValueError('Unexpected native model profile')
        before=fingerprint(model,torch);receipt['base_fingerprint_before']=before
        if before!=FINGERPRINT:raise ValueError('Native NF4 identity differs before study queries')
        eos=set()
        for v in [model.generation_config.eos_token_id,tokenizer.eos_token_id]:
            if v is not None:eos.update(v if isinstance(v,list) else [v])
        if not eos:raise ValueError('Missing native EOS')
        dump(output/'TOKEN_PROTOCOL.json',{'eos_token_ids':sorted(eos),'pad_token_id':tokenizer.pad_token_id})
        def pre(module,args,kwargs):check();count['attempted']+=1
        def post(module,args,out):
            count['completed']+=1
            if not torch.isfinite(out.logits).all():raise FloatingPointError('Nonfinite native logits')
        handles.extend([model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post)])
        def capture(key):
            def hook(module,args,out):
                if count['attempted']==capture_at:captured[key]=(out[0] if isinstance(out,tuple) else out)[0,-1].detach().float().cpu().numpy().copy()
            return hook
        for index in cfg['prefill_feature_blocks']:handles.append(layers[index].register_forward_hook(capture(str(index))))
        def generate(messages,collect):
            nonlocal capture_at
            captured.clear();check();ids=encode(tokenizer,messages)
            if len(ids)>cfg['maximum_context_tokens']:raise ValueError('Context ceiling')
            tensor=torch.tensor([ids],device='cuda:0',dtype=torch.long);c0=dict(count);capture_at=count['attempted']+1 if collect else None
            with torch.inference_mode():seq=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),do_sample=False,num_beams=1,num_return_sequences=1,use_cache=True,max_new_tokens=cfg['maximum_new_tokens'],eos_token_id=sorted(eos),pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(eos),logits_to_keep=1)
            generated=seq[0,len(ids):].cpu().tolist();finished=bool(generated and generated[-1] in eos);text=decode(tokenizer,generated,eos)
            if collect:
                if set(captured)!={str(i) for i in cfg['prefill_feature_blocks']} or any(v.shape!=(5120,) or not np.isfinite(v).all() for v in captured.values()):raise ValueError('Invalid prefill feature')
                features.append({k:v.copy() for k,v in captured.items()})
            return {'input_token_ids':ids,'generated_token_ids':generated,'finished_with_eos':finished,'raw_text':text,'answer':parse_answer(text,finished),'forward_attempts':count['attempted']-c0['attempted'],'forwards_completed':count['completed']-c0['completed']}
        for spec in cfg['format_checks']:
            item=generate([{'role':'system','content':SYSTEM},{'role':'user','content':spec['question']}],False);item.update(question=spec['question'],expected=spec['expected']);transport.append(item)
        dump(output/'FORMAT_CHECKS.json',transport);receipt['format_checks_qualified']=all(r['answer']==r['expected'] for r in transport)
        progress('answer_format',qualified=receipt['format_checks_qualified'])
        if receipt['format_checks_qualified']:
            for q in queries:
                item=generate(q['messages'],True);item.update(**{k:q[k] for k in ['query_id','state_id','mode','subject','seed','family','expected','messages_sha256']});answers.append(item)
                if len(answers)%8==0:dump(output/'ANSWERS.json',answers);progress('factual_QA',completed_queries=len(answers),planned_queries=len(queries))
            receipt['summary']=summary(answers,cfg)
        else:receipt['study_queries_status']='not_run_answer_format_unqualified'
        after=fingerprint(model,torch);receipt['base_fingerprint_after']=after
        if before!=after:raise ValueError('Frozen native model changed')
        receipt['status']='complete'
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));dump(output/'FAILURE.json',{'traceback':traceback.format_exc()})
    finally:
        for h in handles:h.remove()
        dump(output/'ANSWERS.json',answers);dump(output/'FORMAT_CHECKS.json',transport)
        if features:np.savez_compressed(output/'QUESTION_PREFILL_STATES.npz',**{str(k):np.stack([v[str(k)] for v in features]) for k in cfg['prefill_feature_blocks']})
        model=layers=result=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        receipt.update(backbone_forward_attempts=count['attempted'],backbone_forwards=count['completed'],completed_queries=len(answers),planned_queries=len(queries),seconds=time.monotonic()-start,model_objects_released=True)
        dump(output/'RUN_RECEIPT.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('commit');p.add_argument('destination');a=p.parse_args();launch(a.root,a.commit,a.destination,CONFIG,worker_function=worker)
