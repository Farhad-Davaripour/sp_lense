"""Databricks-only same-backbone, training-text kernel qualification."""
import io,time
import numpy as np
import torch
from .composite_actor import Actor
from .composite_package import checkpoint
from .composite_store import Store,download_model
from .composite_interface import input_ids
from .composite_lora import enable
from .io_utils import read,dump,sha


def run(root,output,cache,cfg,rows,guard,progress,forward_callback):
    from importlib.metadata import version
    recipe=read(root/'config/composite_kernel_qualification_v1.json');plan=read(root/'inputs/KERNEL_PLAN.json')
    if any(version(k)!=v for k,v in recipe['packages'].items()):raise RuntimeError('Kernel qualification software differs')
    if any(r['split']!='training' for r in rows):raise ValueError('Kernel qualification uses training text only')
    chosen=[next(r for r in rows if r['condition']=='quiet_pending' and r['sequence']==0),next(r for r in rows if r['condition']=='own_pending' and r['labels']['gate'] and not r['terminal'])]
    progress('download_kernel_qualification_qwen');path,proof=download_model(read(root/'inputs/QWEN_PIN.json'),cache)
    dump(output/'MODEL_FILE_VERIFICATION.json',proof);actor=Actor(path,cfg['qwen'],guard,forward_callback,kernel_profile='reference');actor.restore(checkpoint(Store(),plan['behavior']))
    dump(output/'BASE_ADMISSION.json',actor.frozen_proof());dump(output/'KERNEL_RECIPE.json',recipe)
    values={};logits={};kernel_proofs={};bypasses={};warmups=[]
    for profile in ['reference','fla']:
        guard();actor.select_kernels(profile);kernel_proofs[profile]=actor.kernel_proof;progress('kernel_profile',profile=profile)
        for i,row in enumerate(chosen):
            guard();start=time.monotonic();warm=actor.generate(row['text'],False,maximum_new_tokens=recipe['warmup_new_tokens']);torch.cuda.synchronize()
            warmups.append({'profile':profile,'prompt_index':i,'seconds':time.monotonic()-start,'response':warm})
            for on in recipe['adapter_conditions']:
                guard();enable(actor.adapters,on);ids=input_ids(actor.tokenizer,row['text']);tokens=torch.tensor([ids],device='cuda:0',dtype=torch.long)
                with torch.inference_mode():result=actor.model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,logits_to_keep=1).logits
                key=profile+'_'+str(i)+'_'+str(int(on));logits[key]=result[0,0].detach().float().cpu().numpy();del result
                torch.cuda.synchronize();start=time.monotonic();response=actor.generate(row['text'],on,maximum_new_tokens=recipe['timed_new_token_ceiling']);torch.cuda.synchronize();seconds=time.monotonic()-start
                values[key]={'profile':profile,'prompt_index':i,'adapter_enabled':on,'seconds':seconds,'response':response};progress('kernel_generation_complete',profile=profile,prompt_index=i,adapter_enabled=on,seconds=seconds,generated_tokens=len(response['generated_token_ids']))
        bypasses[profile]=actor.bypass_check([chosen[0]['text']]);dump(output/('BYPASS_'+profile.upper()+'.json'),bypasses[profile])
    pairs=[]
    for i in range(len(chosen)):
        for on in recipe['adapter_conditions']:
            suffix='_'+str(i)+'_'+str(int(on));baseline=values['reference'+suffix];candidate=values['fla'+suffix];a=baseline['response'];b=candidate['response'];difference=np.abs(logits['reference'+suffix]-logits['fla'+suffix])
            pairs.append({'prompt_index':i,'adapter_enabled':on,'identical_input_tokens':a['input_token_ids']==b['input_token_ids'],'identical_bounded_generated_tokens':a['generated_token_ids']==b['generated_token_ids'],'identical_EOS_status':a['finished_with_eos']==b['finished_with_eos'],'speed_ratio':baseline['seconds']/candidate['seconds'],'maximum_absolute_logit_difference':float(difference.max()),'mean_absolute_logit_difference':float(difference.mean())})
    actor.verify_frozen();data=io.BytesIO();np.savez_compressed(data,**logits);(output/'LOGITS.npz').write_bytes(data.getvalue())
    admitted=all(p['identical_input_tokens'] and p['identical_bounded_generated_tokens'] and p['identical_EOS_status'] and p['speed_ratio']>=recipe['minimum_speed_ratio_each_pair'] for p in pairs) and all(v['pass'] for v in bypasses.values())
    dump(output/'GENERATIONS.json',values);dump(output/'WARMUPS.json',warmups)
    report={'stage':'kernel_qualification','admitted':bool(admitted),'pairs':pairs,'kernel_proofs':kernel_proofs,'training_state_ids':[r['state_id'] for r in chosen],'model_backbones':1,'optimizer_updates':0,'simulator_actions_executed':0,'heldout_content_read':False,'validation_content_read':False,'frozen_backbone_verified':True,'profile_adopted':False}
    dump(output/'KERNEL_QUALIFICATION.json',report);return report
