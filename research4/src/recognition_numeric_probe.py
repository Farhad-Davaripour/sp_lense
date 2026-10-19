"""Bounded raw-greedy numeric answers using the official full Qwen forward."""
import gc
import os
import time
import traceback
from pathlib import Path

import numpy as np

from .io_utils import Budget,assert_namespace,digest,dump,read,settings,sha
from .recognition_memory import admission,effective_available
from .recognition_numeric_cases import summarize
from .recognition_pilot import execution_identity,fingerprint,write_features


def encode(tokenizer,messages):
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if hasattr(ids,'keys'):ids=ids['input_ids']
    if ids and isinstance(ids[0],list):ids=ids[0]
    return ids


def score(cases,records):
    result={'by_family':summarize(cases,records),'event_remaining_by_condition':{}}
    for task in ['unfinished','complete']:
        for remaining in [1,2,4]:
            positions=[i for i,row in enumerate(cases) if row['task_family']=='event_remaining' and row['factors']['task_state']==task and row['factors']['remaining_ticks']==remaining]
            result['event_remaining_by_condition'][f'{task}:{remaining}']=summarize([cases[i] for i in positions],[records[i] for i in positions])['all']
    result['scope']='Strict completed integer answers are primary. Complete first/last-integer parses are secondary and both are reported. Truncations are not semantic failures or successful answers; all remain in denominators.'
    return result


def run(root,output,model_path,input_dir,config_path,source_commit,run_id):
    os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    import sys,torch,transformers
    from torch.nn.attention import SDPBackend,sdpa_kernel
    root,output,model_path,input_dir,config_path=map(Path,[root,output,model_path,input_dir,config_path])
    commit=execution_identity(root,'databricks',source_commit)
    for path in [output,model_path,input_dir]:assert_namespace(path)
    config_path.relative_to(root/'config');cfg=read(config_path);common=settings(root)[0]
    if output.exists() or cfg['scope']!='recognition_only_no_policy' or cfg['maximum_new_tokens']!=8:raise ValueError('Unexpected numeric probe release')
    if sha(input_dir/'CASES.json')!=cfg['cases_sha256'] or sha(input_dir/'PARITY.json')!=cfg['parity_sha256']:raise ValueError('Numeric inputs changed')
    cases,parity=read(input_dir/'CASES.json'),read(input_dir/'PARITY.json')
    if len(cases)!=36 or len({digest(row['messages']) for row in cases})!=36 or cfg['maximum_forwards']!=1+len(cases)*cfg['maximum_new_tokens']:
        raise ValueError('Numeric prompt membership or budget differs')
    if any(type(row['expected_integer']) is not int or [m['role'] for m in row['messages']]!=['system','user'] for row in cases):raise ValueError('Unexpected numeric case schema')
    output.mkdir(parents=True)
    protected={str(path):sha(path) for path in [input_dir/'CASES.json',input_dir/'PARITY.json',config_path]}
    dump(output/'PRECALL_FREEZE.json',{'source_commit':commit,'config':cfg,'case_ids':[r['case_id'] for r in cases],
         'prompt_sha256':digest([r['messages'] for r in cases]),'parity_prompt_sha256':digest(parity['messages']),
         'source_sha256':{name:sha(root/name) for name in ['src/recognition_numeric_probe.py','src/recognition_numeric_cases.py','src/recognition_memory.py']}})
    budget=Budget({**common['budget'],'maximum_backbone_forwards':cfg['maximum_forwards'],'internal_stop_seconds':cfg['stop_seconds'],'protected_export_seconds':cfg['export_reserve_seconds']})
    started=time.monotonic();model=None;records=[];values=[];keys=[];handles=[]
    receipt={'status':'running','scope':cfg['scope'],'source_commit':commit,'condition':'numeric_raw_greedy','optimizer_updates':0,'GPU_allocations':0,'policy_actions_executed':0,'reserve_rows_opened':0}
    def progress(stage):dump(output/'PROGRESS.json',{'stage':stage,'cases':len(records),'forwards':budget.forwards,'seconds':time.monotonic()-started})
    try:
        from databricks.sdk import WorkspaceClient
        live=WorkspaceClient().jobs.get_run(run_id=int(run_id));task=next(t for t in live.tasks if t.task_key=='frozen_recognition_followup')
        if not live.start_time or not task.start_time:raise RuntimeError('Fresh numeric timing unavailable')
        remaining=min(live.start_time,task.start_time)/1000+2100-time.time();effective=min(cfg['stop_seconds'],remaining-30)
        dump(output/'FRESH_TIME_ADMISSION.json',{'remaining_seconds_at_entry':remaining,'effective_internal_stop_seconds':effective,'export_reserve_seconds':cfg['export_reserve_seconds']})
        if effective<cfg['export_reserve_seconds']+90:raise TimeoutError('Insufficient fresh numeric time')
        budget.config['internal_stop_seconds']=effective
        torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
        if torch.cuda.is_available():raise ValueError('CPU-only numeric probe')
        weights=model_path/'model.safetensors-00001-of-00001.safetensors'
        if sha(weights)!=common['model']['checkpoint_sha256']:raise ValueError('Pinned numeric backbone changed')
        plan=admission(weights,effective_available(),cfg['memory_profile']);dump(output/'RESOURCE_ADMISSION.json',plan)
        if not plan['admitted']:raise MemoryError('Numeric admission failed before model loading')
        dump(output/'SOFTWARE.json',{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'device':'cpu','dtype':'float32','threads':torch.get_num_threads()})
        progress('loading')
        from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        model=Qwen3_5ForConditionalGeneration.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.float32,attn_implementation='sdpa').eval().requires_grad_(False)
        if any(p.requires_grad or p.device.type!='cpu' for p in model.parameters()):raise ValueError('Unexpected numeric parameter state')
        before=fingerprint(model);receipt['base_fingerprint_before']=before
        if before!=parity['reference_backbone_fingerprint']:raise ValueError('Numeric model differs from the parity reference')
        eos=set()
        for item in [model.generation_config.eos_token_id,tokenizer.eos_token_id]:
            if item is not None:eos.update([item] if isinstance(item,int) else item)
        if not eos:raise ValueError('No declared EOS tokens')
        letter_ids=[tokenizer.encode(c,add_special_tokens=False) for c in 'ABCDEF']
        if any(len(ids)!=1 for ids in letter_ids):raise ValueError('Unexpected parity token encoding')
        dump(output/'TOKEN_PROTOCOL.json',{'eos_token_ids':sorted(eos),'parity_letter_ids':[ids[0] for ids in letter_ids],'decode_skip_special_tokens':True,'clean_up_tokenization_spaces':False,'cache':False,'sampling':False,'maximum_new_tokens':8,
             'feature_alignment':'Step 0 processes the entire prompt; each later step processes the prompt plus prior generated tokens. Its last-position state predicts generated_token[step].'})
        captured={};minimum_free=effective_available()
        for layer in cfg['capture_layers_zero_based']:
            def hook(module,args,out,index=layer):captured[index]=(out[0] if isinstance(out,tuple) else out)[0,-1].detach().numpy().copy()
            handles.append(model.model.language_model.layers[layer].register_forward_hook(hook))
        def forward(ids,key):
            nonlocal minimum_free
            if len(ids)>cfg['maximum_tokens']:raise ValueError('Numeric context ceiling exceeded')
            free=effective_available();minimum_free=min(minimum_free,free)
            if free<plan['live_free_memory_floor_bytes']:raise MemoryError('Numeric activation/export reserve reached')
            captured.clear();budget.count();tensor=torch.tensor([ids],dtype=torch.long)
            with sdpa_kernel(SDPBackend.MATH):result=model(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False,logits_to_keep=1,return_dict=True)
            logits=result.logits[0,-1].detach()
            feature=np.stack([captured[layer] for layer in cfg['capture_layers_zero_based']])
            if feature.shape!=(4,1024) or not np.isfinite(feature).all() or not torch.isfinite(logits).all():raise ValueError('Nonfinite numeric output')
            values.append(feature);keys.append(key)
            return logits
        with torch.no_grad():
            logits=forward(encode(tokenizer,parity['messages']),'__parity__:0')
            actual=logits[torch.tensor([ids[0] for ids in letter_ids])].numpy();expected=np.asarray(parity['reference_letter_logits'])
            maximum_error=float(np.max(np.abs(actual-expected)))
            if not np.allclose(actual,expected,rtol=1e-6,atol=1e-3) or int(actual[:4].argmax())!=int(expected[:4].argmax()):raise ValueError('Official full forward differs from the archived text/head shortcut')
            dump(output/'PARITY_RESULTS.json',{'pass':True,'official_letter_logits':actual.tolist(),'maximum_letter_logit_error':maximum_error,'reference_run_receipt_sha256':parity['reference_run_receipt_sha256'],'Qwen_forwards':1})
            for case in cases:
                prompt=encode(tokenizer,case['messages']);prefix=prompt.copy();generated=[];token_scores=[]
                for step in range(cfg['maximum_new_tokens']):
                    logits=forward(prefix,case['case_id']+':'+str(step));token=int(logits.argmax());generated.append(token)
                    token_scores.append({'token_id':token,'logit':float(logits[token]),'probability':float((logits[token]-logits.logsumexp(0)).exp())})
                    if token in eos:break
                    prefix.append(token)
                text=tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                records.append({'case_id':case['case_id'],'task_family':case['task_family'],'prompt_sha256':digest(case['messages']),'prompt_tokens':len(prompt),'generated_token_ids':generated,'token_scores':token_scores,'decoded_text':text,'finished_with_eos':generated[-1] in eos})
                dump(output/'RECORDS.json',records)
                if len(records)%4==0:write_features(output,values,keys)
                progress('generating')
        metrics=score(cases,records);dump(output/'NUMERIC_SCORES.json',metrics)
        after=fingerprint(model)
        if before!=after:raise ValueError('Numeric parameters changed')
        receipt.update(status='complete',base_fingerprint_after=after,minimum_observed_live_free_memory_bytes=minimum_free)
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));dump(output/'FAILURE.json',{'traceback':traceback.format_exc()})
    finally:
        for handle in handles:handle.remove()
        dump(output/'RECORDS.json',records);write_features(output,values,keys)
        model=None;gc.collect()
        receipt.update(backbone_forwards=budget.forwards,completed_cases=len(records),model_worker_exiting=True,seconds=time.monotonic()-started,protected_inputs_unchanged=all(sha(Path(path))==value for path,value in protected.items()))
        dump(output/'RUN_RECEIPT.json',receipt);dump(output/'MANIFEST.json',{p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='MANIFEST.json'})
    return receipt
