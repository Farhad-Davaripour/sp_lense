"""Versioned adapter coverage and bounded training memory mechanics."""
from contextlib import contextmanager
from functools import partial
from .io_utils import read

LEGACY='config/composite_controller_v1.json'
RANK32='config/composite_controller_rank32_v2.json'


def protocol_path(root):
    selector=root/'inputs/ADAPTER_PROFILE.json'
    name=read(selector)['protocol'] if selector.exists() else LEGACY
    if name not in [LEGACY,RANK32]:raise ValueError('Unregistered adapter protocol')
    return root/name


def qwen_targets(text,placement,layers):
    if len(text.layers)!=layers:raise ValueError('Unexpected Qwen layer count')
    if placement.get('scope')!='all_blocks':
        return ['layers.'+str(placement['layer'])+'.'+name for name in placement['modules']]
    names=[]
    expected={'linear_attention':48,'full_attention':16} if layers==64 else None
    counts={'linear_attention':0,'full_attention':0}
    for index,layer in enumerate(text.layers):
        kind=layer.block_type
        if kind not in counts:raise ValueError('Unknown Qwen mixer block')
        counts[kind]+=1
        suffixes=['mlp.gate_proj','mlp.up_proj','mlp.down_proj']
        suffixes+=['self_attn.'+p for p in ['q_proj','k_proj','v_proj','o_proj']] if kind=='full_attention' else ['linear_attn.'+p for p in ['in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj']]
        names+=['layers.'+str(index)+'.'+p for p in suffixes]
    if expected is not None and counts!=expected:raise ValueError('Pinned hybrid Qwen coverage differs')
    # Resolve everything before any mutation: a missing block must never leave
    # an apparently successful partial all-block installation.
    for name in names:
        module=text.get_submodule(name)
        if not all(hasattr(module,k) for k in ['in_features','out_features','weight']):raise ValueError('Unsupported Qwen projection: '+name)
    return names


def detector_targets(encoder,cfg):
    from torch import nn
    names=[];coverage={i:set() for i in range(cfg['layers']-cfg['lora_last_layers'],cfg['layers'])}
    for name,module in encoder.named_modules():
        pieces=name.split('.')
        if not isinstance(module,nn.Linear) or 'layers' not in pieces:continue
        index=int(pieces[pieces.index('layers')+1])
        if index in coverage and pieces[-1] in cfg['lora_modules']:
            names.append(name);coverage[index].add('.'.join(pieces[pieces.index('layers')+2:]))
    if cfg.get('scope')=='all_blocks':
        expected={'attn.Wqkv','attn.Wo','mlp.Wi','mlp.Wo'}
        if cfg['lora_last_layers']!=cfg['layers'] or any(v!=expected for v in coverage.values()):raise ValueError('Incomplete all-block ModernBERT projections')
    return names


def capacity(root,names,rank):
    shapes=[(name,root.get_submodule(name)) for name in names]
    count=sum(rank*(m.in_features+m.out_features) for _,m in shapes)
    return {'rank':rank,'projection_count':len(names),'adapter_parameters':count,
            'FP32_weights_gradients_AdamW_bytes':count*16,
            'activation_and_backbone_bytes_included':False,
            'covered_blocks':sorted({int(n.split('.')[1]) for n in names}),
            'projections':{n:[m.in_features,m.out_features] for n,m in shapes}}


@contextmanager
def checkpointed_training(model,layers,enabled):
    """Nonreentrant gradients through frozen inputs; retain eval-mode dropout.

    Transformers dispatches checkpointing on each block's training flag. Only
    that flag is changed; dropout children stay in evaluation mode, preserving
    the previous deterministic training recipe. The context spans backward.
    """
    if not enabled:
        yield
        return
    states=[(m,m.training,getattr(m,'gradient_checkpointing',False),getattr(m,'_gradient_checkpointing_func',None)) for m in model.modules()]
    try:
        from torch.utils.checkpoint import checkpoint
        # The pinned v5 public enable() also adds an embedding-output gradient
        # hook on every call. Nonreentrant checkpointing needs no such hook.
        # Use its pinned block dispatcher directly, avoiding accumulated hooks
        # and extra input activation gradients on this frozen backbone.
        model._set_gradient_checkpointing(enable=True,gradient_checkpointing_func=partial(checkpoint,use_reentrant=False))
        if not layers or any(not getattr(layer,'gradient_checkpointing',False) for layer in layers):raise ValueError('Every adapted block must support checkpointing')
        for layer in layers:layer.training=True
        yield
    finally:
        for module,training,flag,function in states:
            module.training=training
            if hasattr(module,'gradient_checkpointing'):module.gradient_checkpointing=flag
            if function is not None:module._gradient_checkpointing_func=function
            elif hasattr(module,'_gradient_checkpointing_func'):del module._gradient_checkpointing_func
