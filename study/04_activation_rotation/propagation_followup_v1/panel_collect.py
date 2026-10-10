"""Prepared six-condition native collector. No allocation, loading or auto-run."""
import hashlib
import json
from pathlib import Path
import time


def native_intervention(h, condition, vectors, specification):
    """Causal, position-local native matching; no downstream/future input."""
    import torch
    if condition=='base':return h, None
    if not torch.isfinite(h).all() or torch.any(h.float().norm(dim=-1)<=1e-12):raise ValueError('Invalid/near-zero native residual')
    strength=specification['nominal_strength']
    v=vectors['v'].to(device=h.device,dtype=h.dtype)
    plus=h+strength*v
    if torch.any(plus.float().norm(dim=-1)<=1e-12):raise ValueError('Hypothetical +v output is near-zero')
    target=(plus.float()-h.float()).norm(dim=-1,keepdim=True)
    if torch.any(target<=0):raise ValueError('Native target displacement rounded to zero')
    scale=torch.full_like(target,strength)
    if condition=='plus_v':
        changed=plus
    elif condition=='norm_only':
        norm=h.float().norm(dim=-1,keepdim=True)
        if torch.any(norm<=1e-12):raise ValueError('Norm-only denominator is zero')
        scale=plus.float().norm(dim=-1,keepdim=True)/norm
        changed=(h.float()*scale).to(h.dtype)
    else:
        direction=(-vectors['v'] if condition=='minus_v' else vectors[condition]).to(device=h.device,dtype=h.dtype)
        best=None;best_error=None
        for coefficient in specification['matching']['additive_coefficients']:
            candidate=h+coefficient*direction
            error=((candidate.float()-h.float()).norm(dim=-1,keepdim=True)-target).abs()
            if best is None:
                best=candidate;best_error=error;scale.fill_(coefficient)
            else:
                improved=error<best_error
                best=torch.where(improved,candidate,best)
                best_error=torch.where(improved,error,best_error)
                scale=torch.where(improved,torch.full_like(scale,coefficient),scale)
        changed=best
    effect=changed.float()-h.float()
    actual=effect.norm(dim=-1,keepdim=True)
    relative_match_error=(actual-target).abs()/target
    norm_error=(changed.float().norm(dim=-1,keepdim=True)-plus.float().norm(dim=-1,keepdim=True)).abs()/plus.float().norm(dim=-1,keepdim=True)
    cosine=(h.float()*changed.float()).sum(dim=-1,keepdim=True)/(h.float().norm(dim=-1,keepdim=True)*changed.float().norm(dim=-1,keepdim=True))
    angle=torch.acos(cosine.clamp(-1,1))
    if condition=='norm_only':
        if torch.any(norm_error>specification['matching']['norm_only_relative_norm_error_limit']):raise ValueError('Norm-only native norm mismatch')
        if torch.any(angle>specification['matching']['norm_only_site_angle_limit_radians']):raise ValueError('Norm-only native directional rounding too large')
    elif torch.any(relative_match_error>specification['matching']['relative_error_limit']):
        raise ValueError('Native additive injection norm mismatch; stop, do not widen matching grid')
    audit=torch.cat([target[0,-1],actual[0,-1],relative_match_error.max().reshape(1),
        norm_error.max().reshape(1),angle.max().reshape(1),scale.min().reshape(1),scale.max().reshape(1)]).detach()
    return changed,audit


def collect_panel(model, tokenizer, vector_archive, output, *, session_started_monotonic,
                  model_identity, checkpoint, specification_path=Path(__file__).with_name('PANEL.json')):
    import numpy as np
    import torch
    from transformers import LogitsProcessor,LogitsProcessorList
    specification=json.loads(Path(specification_path).read_text())
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen eval model required')
    if getattr(model,'peft_config',None):raise ValueError('No adapter allowed')
    if getattr(model,'is_quantized',False) or getattr(model,'is_loaded_in_4bit',False):raise ValueError('Unquantized BF16 required')
    if model_identity['revision']!=specification['model_revision']:raise ValueError('Checkpoint pin mismatch')
    if Path(model.config._name_or_path).resolve()!=Path(model_identity['model_path']).resolve():raise ValueError('Loaded path differs from verified checkpoint')
    if hashlib.sha256(Path(vector_archive).read_bytes()).hexdigest()!=specification['vectors_sha256']:raise ValueError('Vector archive mismatch')
    with np.load(vector_archive,allow_pickle=False) as z:vectors={k:torch.tensor(z[k],dtype=torch.float32) for k in z.files}
    for v in vectors.values():
        if v.numel()!=1024 or not torch.isfinite(v).all() or abs(float(v.norm())-1)>1e-5:raise ValueError('Invalid frozen unit direction')
    layers=model.model.language_model.layers
    if len(layers)!=24:raise ValueError('Wrong architecture')
    device=next(model.parameters()).device
    if next(model.parameters()).dtype!=torch.bfloat16:raise ValueError('BF16 loader required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'PANEL.json').write_bytes(Path(specification_path).read_bytes())
    (output/'panel_vectors.npz').write_bytes(Path(vector_archive).read_bytes())
    receipt={'study':specification['study'],'status':'running','completed':[],
             'model_identity':model_identity,'condition_orders':[],
             'specification_sha256':hashlib.sha256(Path(specification_path).read_bytes()).hexdigest(),
             'vector_archive_sha256':specification['vectors_sha256'],'validation_probes_completed':0}
    def save():
        receipt['elapsed_seconds']=time.monotonic()-session_started_monotonic
        temporary=output/'STATUS.json.tmp';temporary.write_text(json.dumps(receipt,indent=2));temporary.replace(output/'STATUS.json')
    def deadline():
        if time.monotonic()-session_started_monotonic>=specification['execution_seconds']:raise TimeoutError('Preserve export reserve')
    def replay(prefix,continuation,condition,probe=False,null_hook=False):
        captured=[[] for _ in layers];pre=[];delivery=[];logits=[];handles=[]
        class Forced(LogitsProcessor):
            def __call__(self,input_ids,scores):
                deadline()
                if probe:logits.append(scores.detach().clone())
                index=input_ids.shape[1]-prefix.shape[1]
                forced=torch.full_like(scores,-torch.inf);forced[:,continuation[index]]=0
                return forced
        def intervention(module,arguments,result):
            h=result[0] if isinstance(result,tuple) else result
            if not probe:pre.append(h[0,-1].detach().clone())
            if null_hook:return result
            changed,audit=native_intervention(h,condition,vectors,specification)
            if audit is not None:delivery.append(audit)
            return (changed,)+result[1:] if isinstance(result,tuple) else changed
        if not probe or null_hook:handles.append(layers[10].register_forward_hook(intervention))
        if not probe:
            for index,layer in enumerate(layers):
                def recorder(module,arguments,result,index=index):
                    h=result[0] if isinstance(result,tuple) else result
                    captured[index].append(h[0,-1].detach().clone())
                handles.append(layer.register_forward_hook(recorder))
        try:
            with torch.inference_mode():
                generated=model.generate(input_ids=prefix,attention_mask=torch.ones_like(prefix),
                    do_sample=False,use_cache=True,max_new_tokens=len(continuation),
                    logits_processor=LogitsProcessorList([Forced()]))
            if generated[0,prefix.shape[1]:].cpu().tolist()!=continuation:raise ValueError('Forced token replay mismatch')
            if probe:return torch.stack(logits).float().cpu().numpy()
            arrays=np.stack([torch.stack(x).float().cpu().numpy() for x in captured])
            if arrays.shape!=(24,len(continuation),1024) or not np.isfinite(arrays).all():raise ValueError('Invalid full-width capture')
            return arrays,torch.stack(pre).float().cpu().numpy(),torch.stack(delivery).cpu().tolist() if delivery else []
        finally:
            for handle in handles:handle.remove()
    save()
    try:
        for index,prompt in enumerate(specification['prompts']):
            deadline()
            prefix=torch.tensor([prompt['prefix_ids']],device=device)
            continuation=specification['forced_continuation_ids']
            if index==0:
                untreated=replay(prefix,continuation[:8],'base',probe=True)
                null=replay(prefix,continuation[:8],'base',probe=True,null_hook=True)
                receipt['null_hook_logits_exact']=bool(np.array_equal(untreated,null));receipt['validation_probes_completed']=2
                if not receipt['null_hook_logits_exact']:raise ValueError('Null hook changed native logits')
            conditions=specification['conditions'];order=conditions[index:]+conditions[:index]
            receipt['condition_orders'].append({'prompt':prompt['id'],'order':order})
            folder=output/prompt['id'];folder.mkdir()
            traces={};before={}
            for condition in order:
                start=time.monotonic();arrays,pre,audit=replay(prefix,continuation,condition)
                path=folder/(condition+'.npz')
                np.savez_compressed(path,residuals=arrays,pre_injection=pre,forced_token_ids=continuation)
                receipt['completed'].append({'prompt':prompt['id'],'condition':condition,'shape':list(arrays.shape),
                    'seconds':time.monotonic()-start,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'delivery_columns':['target_last','actual_last','max_relative_match_error_all_positions','max_relative_norm_error_to_plus_all_positions','max_site_angle_all_positions','min_coefficient','max_coefficient'],
                    'delivery':audit})
                traces[condition]=arrays;before[condition]=pre;save()
            for condition in conditions[1:]:
                if not np.array_equal(before[condition],before['base']) or not np.array_equal(traces[condition][:10],traces['base'][:10]):raise ValueError('Unexpected upstream/cached pre-injection change')
            checkpoint(output)
        receipt['status']='complete_export_and_runtime_release_pending'
    except BaseException as error:
        receipt['status']='fault_stop_no_retry';receipt['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        save();checkpoint(output)
    return receipt
