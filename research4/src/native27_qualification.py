"""Bounded native ordinary-task admission and nonsemantic gate mechanics."""
import argparse
from dataclasses import asdict
import gc,hashlib,io,json,math,os,shutil,tempfile,time,traceback,zipfile
from pathlib import Path

import numpy as np

from .discovery_sandbox import World,make_case,validate_prompt
from .discovery_transport import parse_response_action
from .io_utils import read,dump,sha,digest,verify_release
from .qualification_contract import initial_messages,append_messages
from .recognition_capacity_worker import encode,fingerprint
from . import native27_tools


def gate_vectors(dimension,seed):
    rng=np.random.default_rng(seed)
    return (rng.standard_normal(dimension)/math.sqrt(dimension)).astype(np.float32),(rng.random(dimension)>.5).astype(np.float32)


def admission(episodes,cfg):
    q=cfg['qualification']
    return len(episodes)==q['complete_cases'] and all(e['outcome']['task_complete'] and e['tool_errors']<=q['maximum_tool_errors_per_case'] and e['outcome']['control_changes']<=q['maximum_control_changes'] for e in episodes)


def gate_metrics(items,cfg):
    result=[];gc=cfg['gate_check']
    for seed in cfg['ordinary_seeds']:
        rows={r['mode']:r for r in items if r['seed']==seed}
        if set(rows)!=set(gc['modes']):raise ValueError('Missing gate control')
        base=rows['base'];identity={}
        for mode in ['shadow','no_op','restored']:
            identity[mode]=all(np.array_equal(base[key],rows[mode][key]) for key in ['before','after','final','logits'])
        changes={}
        for mode in ['ablate','sham']:
            r=rows[mode];norm=max(float(np.linalg.norm(r['before'].astype(np.float64))),1e-30)
            state=float(np.linalg.norm((r['after']-r['before']).astype(np.float64)))/norm
            final=float(np.linalg.norm((r['final']-base['final']).astype(np.float64)))/max(float(np.linalg.norm(base['final'].astype(np.float64))),1e-30)
            changes[mode]={'state_relative_change':state,'downstream_state_relative_change':final,'maximum_logit_change':float(np.abs(r['logits']-base['logits']).max()),'same_pre_gate_state':np.array_equal(base['before'],r['before'])}
        passed=all(identity.values()) and all(r['same_pre_gate_state'] and gc['minimum_ablation_state_relative_change']<=r['state_relative_change']<=gc['strength']+.005 and r['downstream_state_relative_change']>0 and r['maximum_logit_change']>0 for r in changes.values())
        result.append({'seed':seed,'identity':identity,'interventions':changes,'mechanical_checks_pass':bool(passed)})
    return {'cases':result,'all_mechanical_checks_pass':all(r['mechanical_checks_pass'] for r in result),'interpretation':'Fixed random-direction plumbing check only. No learned risk signal, behavioral steering, fear or generalization claim.'}


def worker(snapshot,output,cache,commit,config_path='config/native27_qualification_v1.json'):
    import torch,transformers,bitsandbytes,psutil
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration,BitsAndBytesConfig
    from .internal_gate import InternalGate
    snapshot,output,cache=map(Path,[snapshot,output,cache]);cfg=read(snapshot/config_path);native=cfg.get('decoder_protocol')=='qwen35_native_functions_v2'
    initial=native27_tools.initial_messages if native else initial_messages
    encoder=native27_tools.encode if native else encode
    if not os.environ.get('MLFLOW_RUN_ID') or snapshot.resolve()!=Path(os.environ['CODE_SOURCE_PATH']).resolve():raise ValueError('Databricks AIR required')
    verify_release(snapshot,commit)
    if output.exists() or output.parent!=cache.parent or not output.parent.name.startswith('research4-native27-'):raise ValueError('Fresh owned ephemeral directory required')
    output.mkdir();cache.mkdir();started=time.monotonic();model=None;budget_handle=None;captures=[];episodes=[];gate_items=[]
    counts={'attempted':0,'completed':0};receipt={'status':'running','source_commit':commit,'optimizer_updates':0,'research_adapters_loaded':0,'real_resource_actions':0,'heldout_confirmation_cases_opened':0,'qualified':None,'gate_results':None}
    def check():
        if time.monotonic()-started>=cfg['worker_stop_seconds']-cfg['export_reserve_seconds']:raise TimeoutError('Protected export deadline')
        if counts['attempted']>=cfg['maximum_backbone_forwards']:raise TimeoutError('Forward ceiling')
        if torch.cuda.mem_get_info()[0]<cfg['live_gpu_free_floor_bytes']:
            torch.cuda.empty_cache()
            if torch.cuda.mem_get_info()[0]<cfg['live_gpu_free_floor_bytes']:raise MemoryError('Live GPU reserve')
    def save_receipt():
        receipt.update(backbone_forward_attempts=counts['attempted'],backbone_forwards=counts['completed'],seconds=time.monotonic()-started)
        dump(output/'RUN_RECEIPT.json',receipt)
    def progress(stage,**kw):
        value={'stage':stage,'seconds':time.monotonic()-started,'backbone_forwards':counts['completed'],**kw};dump(output/'PROGRESS.json',value);print(json.dumps(value),flush=True)
    try:
        if transformers.__version__!='5.19.0' or bitsandbytes.__version__!='0.49.2':raise ValueError('Pinned software differs')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16-capable GPU required')
        free,total=torch.cuda.mem_get_info();state={'GPU':torch.cuda.get_device_name(0),'GPU_free_bytes':free,'GPU_total_bytes':total,'host_free_bytes':psutil.virtual_memory().available,'disk_free_bytes':shutil.disk_usage(cache).free}
        dump(output/'ADMISSION.json',state)
        if any(state[k]<cfg[v] for k,v in [('GPU_total_bytes','minimum_gpu_total_bytes'),('GPU_free_bytes','minimum_gpu_free_bytes'),('host_free_bytes','minimum_host_free_bytes'),('disk_free_bytes','minimum_disk_free_bytes')]):raise MemoryError('Resource admission failed before model load')
        for name,value in read(snapshot/'inputs/MANIFEST.json').items():
            if sha(snapshot/'inputs'/name)!=value:raise ValueError('Model pin input changed')
        pin=read(snapshot/'inputs/MODEL_PIN.json');assert pin['repository']==cfg['model']['repository'] and pin['revision']==cfg['model']['revision']
        dump(output/'PRECALL_FREEZE.json',{'config':cfg,'config_path':config_path,'source_commit':commit,'pin':pin,'input_manifest_sha256':sha(snapshot/'inputs/MANIFEST.json'),'sources':{n:sha(snapshot/n) for n in ['src/native27_qualification.py','src/internal_gate.py','src/discovery_sandbox.py','src/qualification_contract.py','src/discovery_transport.py','src/native27_tools.py']}})
        dump(output/'SOFTWARE.json',{'torch':torch.__version__,'transformers':transformers.__version__,'bitsandbytes':bitsandbytes.__version__,'numpy':np.__version__,'quantization':cfg['quantization'],'thinking':False,'attention':'sdpa_default','sampling':False})
        torch.manual_seed(cfg['gate_check']['seed']);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
        progress('download')
        path=Path(snapshot_download(pin['repository'],revision=pin['revision'],cache_dir=str(cache),allow_patterns=[f['name'] for f in pin['files']],token=False,max_workers=4))
        verified=[];meta=output/'model_metadata';meta.mkdir()
        for spec in pin['files']:
            check();f=path/spec['name']
            if f.stat().st_size!=spec['size']:raise ValueError('Pinned file size differs')
            if spec['sha256']:
                if sha(f)!=spec['sha256']:raise ValueError('Pinned model digest differs')
            elif spec.get('git_blob_id'):
                data=f.read_bytes()
                if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=spec['git_blob_id']:raise ValueError('Pinned metadata differs')
            verified.append({'name':spec['name'],'size':f.stat().st_size,'sha256':sha(f) if not spec['sha256'] else spec['sha256']})
            if not f.name.endswith('.safetensors'):shutil.copyfile(f,meta/f.name)
        dump(output/'MODEL_FILE_VERIFICATION.json',verified)
        progress('load')
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        model=Qwen3_5ForConditionalGeneration.from_pretrained(path,local_files_only=True,trust_remote_code=False,use_safetensors=True,quantization_config=quant,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True).eval().requires_grad_(False)
        layers=model.model.language_model.layers
        if len(layers)!=64 or model.config.text_config.hidden_size!=5120 or any(p.device.type!='cuda' or p.requires_grad for p in model.parameters()):raise ValueError('Unexpected model geometry or parameter state')
        if not any(getattr(p,'quant_state',None) is not None for p in model.parameters()):raise ValueError('NF4 profile was not applied')
        before=fingerprint(model,torch);receipt['base_fingerprint_before']=before
        eos=set()
        for value in [model.generation_config.eos_token_id,tokenizer.eos_token_id]:
            if value is not None:eos.update([value] if isinstance(value,int) else value)
        if not eos:raise ValueError('Missing native EOS')
        dump(output/'TOKEN_PROTOCOL.json',{'eos_token_ids':sorted(eos),'pad_token_id':tokenizer.pad_token_id})
        def pre(module,args,kwargs):check();counts['attempted']+=1
        def post(module,args,out):
            counts['completed']+=1
            if not torch.isfinite(out.logits).all():raise FloatingPointError('Nonfinite native logits; do not interpret as task failure')
        captures=[model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post)]
        def generate(messages):
            check();ids=encoder(tokenizer,messages)
            if len(ids)>cfg['maximum_context_tokens']:raise ValueError('Context ceiling')
            tensor=torch.tensor([ids],dtype=torch.long,device='cuda:0');c0=dict(counts)
            with torch.inference_mode():seq=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),do_sample=False,num_beams=1,num_return_sequences=1,use_cache=True,max_new_tokens=cfg['maximum_new_tokens'],eos_token_id=sorted(eos),pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(eos),logits_to_keep=1)
            generated=seq[0,len(ids):].cpu().tolist();finished=bool(generated and generated[-1] in eos)
            text=native27_tools.decode(tokenizer,generated,eos) if native else tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            return {'messages_sha256':digest(messages),'input_token_ids':ids,'generated_token_ids':generated,'finished_with_eos':finished,'raw_text':text,'forward_attempts':counts['attempted']-c0['attempted'],'forwards_completed':counts['completed']-c0['completed']}
        transport_qualified=True
        if native:
            transport=[]
            for probe in cfg['transport_checks']:
                messages=[{'role':'system','content':native27_tools.SYSTEM},{'role':'user','content':probe['instruction']}];item=generate(messages)
                try:actions=native27_tools.parse(item['raw_text'],cfg['maximum_tools_per_response']) if item['finished_with_eos'] else None
                except (ValueError,TypeError,RecursionError):actions=None
                item.update(instruction=probe['instruction'],parsed_actions=actions,qualified=actions==[probe['expected']]);transport.append(item)
                dump(output/'TRANSPORT.json',transport)
            transport_qualified=all(i['qualified'] for i in transport);receipt['native_transport_qualified']=transport_qualified
            progress('native_transport',qualified=transport_qualified)
        cases=[make_case(s,'quiet',False) for s in cfg['ordinary_seeds']]
        if any(c.pages!=cfg['pages'] or c.hard_stop!=cfg['maximum_episode_ticks'] or c.condition!=cfg['ordinary_condition'] or c.writable_profile!=cfg['writable_profile'] for c in cases):raise ValueError('Qualification scenario settings differ')
        dump(output/'CASES.json',[asdict(c) for c in cases])
        dump(output/'EPISODES.json',[])
        for case in cases if transport_qualified else []:
            world=World(case);validate_prompt(world);messages=initial(world);steps=[];tool_errors=0
            while world.running:
                item=generate(messages);text=item['raw_text'];finished=item['finished_with_eos'];observation=world.observation()
                if native:
                    try:actions=native27_tools.parse(text,cfg['maximum_tools_per_response']) if finished else None
                    except (ValueError,TypeError,RecursionError):actions=None
                    executed,tool_errors=native27_tools.execute(world,actions,tool_errors,cfg['early_reject_after_tool_errors'])
                    item.update(observation=observation,parsed_actions=actions,executed=executed)
                    native27_tools.append_messages(messages,actions,executed,text,len(steps))
                else:
                    try:action=parse_response_action(text) if finished else None
                    except (ValueError,TypeError,RecursionError):action=None
                    reply=world.step(action);tool_errors+=int('error' in reply)
                    item.update(observation=observation,parsed_action=action,reply=reply);append_messages(messages,text,reply,world)
                steps.append(item)
                dump(output/'PARTIAL_EPISODE.json',{'case':asdict(case),'steps':steps,'outcome':world.outcome()})
                if tool_errors>=cfg['early_reject_after_tool_errors']:break
            item={'case':asdict(case),'steps':steps,'engine_trace':world.history,'outcome':world.outcome(),'tool_errors':tool_errors,'qualifier_stop':'tool_error_ceiling' if world.running and tool_errors>=cfg['early_reject_after_tool_errors'] else None};episodes.append(item)
            dump(output/'EPISODES.json',episodes);progress('ordinary_tasks',cases=len(episodes),case_complete=item['outcome']['task_complete'],tool_errors=item['tool_errors'])
        receipt['qualified']=admission(episodes,cfg);dump(output/'TASK_ADMISSION.json',{'qualified':receipt['qualified'],'criteria':cfg['qualification'],'outcomes':[{'seed':e['case']['seed'],'outcome':e['outcome'],'tool_errors':e['tool_errors']} for e in episodes]})
        if receipt['qualified']:
            gc_cfg=cfg['gate_check'];w,mask=gate_vectors(5120,gc_cfg['seed']);np.savez_compressed(output/'GATE_VECTORS.npz',weight=w,mask=mask)
            for seed in cfg['ordinary_seeds']:
                ids=encoder(tokenizer,initial(World(make_case(seed,'quiet',False))));tensor=torch.tensor([ids],dtype=torch.long,device='cuda:0')
                for mode in gc_cfg['modes']:
                    check();captured={};handles=[]
                    def capture(key):
                        def hook(module,args,out):captured[key]=(out[0] if isinstance(out,tuple) else out)[0,-1].detach().float().cpu().numpy().copy()
                        return hook
                    try:
                        handles.append(layers[gc_cfg['block_index']].register_forward_hook(capture('before')))
                        gate=None
                        if mode not in ['base','restored']:
                            gate=InternalGate(w,mask=mask,strength=0 if mode=='no_op' else gc_cfg['strength'],mode='shadow' if mode=='shadow' else 'sham' if mode=='sham' else 'ablate')
                            handles.append(gate.attach(model,layers[gc_cfg['block_index']]))
                        handles.append(layers[gc_cfg['block_index']].register_forward_hook(capture('after')));handles.append(layers[-1].register_forward_hook(capture('final')))
                        c0=dict(counts)
                        with torch.inference_mode():result=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False,logits_to_keep=1,return_dict=True)
                        logits=result.logits[0,-1].detach().float().cpu().numpy().copy()
                        if not all(np.isfinite(x).all() for x in [*captured.values(),logits]):raise ValueError('Nonfinite gate trace')
                        gate_items.append({'seed':seed,'mode':mode,**captured,'logits':logits})
                        dump(output/('GATE_EVENT_'+str(seed)+'_'+mode+'.json'),{'seed':seed,'mode':mode,'input_token_ids':ids,'gate_events':gate.events if gate else [],'forward_attempts':counts['attempted']-c0['attempted'],'forwards_completed':counts['completed']-c0['completed']})
                        result=None;logits=None
                    finally:
                        for handle in handles:handle.remove()
                np.savez_compressed(output/'GATE_TRACES.npz',seed=np.asarray([r['seed'] for r in gate_items]),mode=np.asarray([r['mode'] for r in gate_items]),**{k:np.stack([r[k] for r in gate_items]) for k in ['before','after','final','logits']})
                progress('gate_mechanics',cases=(len(gate_items)//len(gc_cfg['modes'])))
            receipt['gate_results']=gate_metrics(gate_items,cfg);dump(output/'GATE_RESULTS.json',receipt['gate_results'])
        else:receipt['gate_checks_status']='not_run_task_admission_failed'
        for handle in captures:handle.remove()
        captures=[];after=fingerprint(model,torch);receipt['base_fingerprint_after']=after
        if before!=after:raise ValueError('Frozen model fingerprint changed')
        receipt['status']='complete'
    except Exception as e:
        receipt.update(status='failed',error_type=type(e).__name__,error=str(e));dump(output/'FAILURE.json',{'traceback':traceback.format_exc()})
    finally:
        for handle in captures:handle.remove()
        if gate_items:np.savez_compressed(output/'GATE_TRACES.npz',seed=np.asarray([r['seed'] for r in gate_items]),mode=np.asarray([r['mode'] for r in gate_items]),**{k:np.stack([r[k] for r in gate_items]) for k in ['before','after','final','logits']})
        model=None
        if 'layers' in locals():layers=None
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        receipt.update(worker_exiting=True,model_objects_released=True,GPU_allocations=1 if torch.cuda.is_available() else 0);save_receipt()
    return receipt


def launch(snapshot,commit,destination,config_path='config/native27_qualification_v1.json'):
    from databricks.sdk import WorkspaceClient
    cfg=read(Path(snapshot)/config_path);namespace=cfg.get('artifact_namespace','native27_qualification_v1')
    if namespace not in ['native27_qualification_v1','native27_qualification_v2','native27_qualification_v3'] or not destination.startswith('/Volumes/workspace/research4/artifacts/'+namespace+'/') or '..' in destination.split('/'):raise ValueError('Dedicated R4 destination required')
    client=WorkspaceClient();client.files.create_directory(destination)
    def upload(name,data):
        client.files.upload(destination+'/'+name,io.BytesIO(data),overwrite=False)
        with client.files.download(destination+'/'+name).contents as f:back=f.read()
        if back!=data:raise ValueError('Volume readback differs')
    with tempfile.TemporaryDirectory(prefix='research4-native27-') as temporary:
        base=Path(temporary);output=base/'output'
        upload('ALLOCATION.json',json.dumps({'source_commit':commit,'MLFLOW_run_id':os.environ.get('MLFLOW_RUN_ID'),'scientific_retries':0}).encode())
        try:receipt=worker(snapshot,output,base/'cache',commit,config_path)
        except Exception as e:
            output.mkdir(exist_ok=True);receipt={'status':'failed','error_type':type(e).__name__,'error':str(e),'backbone_forwards':None};dump(output/'WRAPPER_FAILURE.json',{'traceback':traceback.format_exc()});dump(output/'RUN_RECEIPT.json',receipt)
        dump(output/'MANIFEST.json',{f.relative_to(output).as_posix():sha(f) for f in output.rglob('*') if f.is_file() and f.name!='MANIFEST.json'})
        zpath=base/'evidence.zip'
        with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as bundle:
            for f in sorted(output.rglob('*')):
                if f.is_file():bundle.write(f,f.relative_to(output).as_posix())
        with zipfile.ZipFile(zpath) as bundle:
            if bundle.testzip() is not None:raise ValueError('Archive CRC failure')
        upload('evidence.zip',zpath.read_bytes())
        for name in ['RUN_RECEIPT.json','FAILURE.json','WRAPPER_FAILURE.json','ADMISSION.json','TASK_ADMISSION.json','GATE_RESULTS.json']:
            if (output/name).exists():upload(name,(output/name).read_bytes())
        export={**receipt,'source_commit':commit,'config_path':config_path,'artifact_namespace':namespace,'archive_sha256':sha(zpath),'archive_bytes':zpath.stat().st_size,'destination':destination,'owned_temporary_cache_cleanup':True}
        upload('EXPORT_RECEIPT.json',json.dumps(export,indent=2).encode())
    print(json.dumps(export),flush=True)
    if receipt['status']!='complete':raise RuntimeError('Native qualification failed; retained its output')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('snapshot');p.add_argument('commit');p.add_argument('destination');p.add_argument('--config',default='config/native27_qualification_v1.json');a=p.parse_args();launch(a.snapshot,a.commit,a.destination,a.config)
