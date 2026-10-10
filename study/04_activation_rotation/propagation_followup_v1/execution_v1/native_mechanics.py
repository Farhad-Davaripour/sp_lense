"""Six short native mechanics traces, before the frozen collector's null probes."""
import hashlib
import json
from pathlib import Path
import time


def run_mechanics(model, vectors, specification, output, started):
    import numpy as np
    import torch
    from transformers import LogitsProcessor,LogitsProcessorList
    from panel_collect import native_intervention
    output=Path(output);output.mkdir(exist_ok=False)
    layers=model.model.language_model.layers
    prefix=torch.tensor([specification['prompts'][0]['prefix_ids']],device=next(model.parameters()).device)
    tokens=specification['forced_continuation_ids'][:8]
    receipt={'kind':'native_six_condition_mechanics','status':'running','attempted':[],
             'completed':[],'steps':len(tokens),'not_study_traces':True}
    arrays={};pre_states={}
    def save():(output/'STATUS.json').write_text(json.dumps(receipt,indent=2))
    save()
    try:
        for condition in specification['conditions']:
            receipt['attempted'].append(condition);save()
            captures=[[] for _ in layers];pre=[];audits=[];handles=[]
            class Forced(LogitsProcessor):
                def __call__(self,input_ids,scores):
                    if time.monotonic()-started>=2400:raise TimeoutError('Execution deadline')
                    forced=torch.full_like(scores,-torch.inf)
                    forced[:,tokens[input_ids.shape[1]-prefix.shape[1]]]=0
                    return forced
            def hook(module,args,result):
                h=result[0] if isinstance(result,tuple) else result
                pre.append(h[0,-1].detach().clone())
                changed,audit=native_intervention(h,condition,vectors,specification)
                if audit is not None:audits.append(audit)
                return (changed,)+result[1:] if isinstance(result,tuple) else changed
            handles.append(layers[10].register_forward_hook(hook))
            for index,layer in enumerate(layers):
                def record(module,args,result,index=index):
                    h=result[0] if isinstance(result,tuple) else result
                    captures[index].append(h[0,-1].detach().clone())
                handles.append(layer.register_forward_hook(record))
            try:
                with torch.inference_mode():
                    generated=model.generate(input_ids=prefix,attention_mask=torch.ones_like(prefix),
                        max_new_tokens=len(tokens),do_sample=False,use_cache=True,
                        logits_processor=LogitsProcessorList([Forced()]))
                assert generated[0,prefix.shape[1]:].cpu().tolist()==tokens
                h=np.stack([torch.stack(c).float().cpu().numpy() for c in captures])
                pre_array=torch.stack(pre).float().cpu().numpy()
                assert h.shape==(24,8,1024) and np.isfinite(h).all()
                path=output/(condition+'.npz')
                np.savez_compressed(path,residuals=h,pre_injection=pre_array,forced_token_ids=tokens)
                arrays[condition]=h;pre_states[condition]=pre_array
                receipt['completed'].append({'condition':condition,'shape':list(h.shape),
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'delivery':torch.stack(audits).cpu().tolist() if audits else []})
                save()
            finally:
                for handle in handles:handle.remove()
        for condition in specification['conditions'][1:]:
            assert np.array_equal(arrays[condition][:10],arrays['base'][:10])
            assert np.array_equal(pre_states[condition],pre_states['base'])
        receipt['upstream_and_pre_injection_exact']=True
        receipt['status']='passed'
    except BaseException as error:
        receipt['status']='failed_stop';receipt['error']=type(error).__name__+': '+str(error)
        raise
    finally:save()
    return receipt
