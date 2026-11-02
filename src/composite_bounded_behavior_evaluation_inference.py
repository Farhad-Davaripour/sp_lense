"""Restore the actual V4 final adapter once and keep every tensor frozen."""
from .io_utils import digest
from .composite_bounded_behavior_evaluation_parent import require_frozen_model
from .composite_bounded_behavior_evaluation_source import require_context


def restore_checkpoint(actor,context):
    require_context(context);context.require_runtime();value=require_frozen_model(context.model)
    if actor.bounded_evaluation_context is not context or actor.adapters:
        raise PermissionError('Restore once into the owned idle frozen V4 actor')
    import torch
    from importlib.metadata import version
    from .composite_task_skill_training import postload_runtime
    actor.bounded_evaluation_postload_proof=postload_runtime(actor,torch.cuda,str(torch.__version__),
      {n:version(n) for n in ['transformers','bitsandbytes','accelerate','safetensors','numpy','psutil']})
    actor.bounded_evaluation_postload_proof['full_adapter_optimizer_admission']='Frozen inference weights/reserve only; no optimizer is allocated'
    from .composite_v2_model_audit import EvidenceFiles,tensor_state
    from .composite_training import tensor_digest
    from .composite_lora import load,trainables
    files=EvidenceFiles(context.fit_archive,value['raw_archive_sha256'],4*1024**3)
    try:
        if files.manifest['behavior.pt']!=value['checkpoint_sha256']:raise PermissionError('The final frozen V4 checkpoint changed')
        saved=tensor_state(files.tensor_path('behavior.pt'))
        expected=value['checkpoint']['hashes']
        if (digest(saved['config'])!=digest(actor.cfg) or saved['placement']!={'name':'all_blocks','scope':'all_blocks'}
            or saved.get('native_interface_profile')!=actor.native_interface_profile or set(saved['weights'])!=set(expected)):
            raise PermissionError('Frozen V4 adapter/model/native interface differs')
        if any(tensor_digest(t)!=expected[n] for n,t in saved['weights'].items()):raise PermissionError('A frozen fitted tensor changed')
        actor._bounded_restoring=True
        try:actor.configure(saved['placement'],for_training=False)
        finally:actor._bounded_restoring=False
        load(actor.adapters,saved['weights'])
        for p in trainables(actor.adapters):p.requires_grad_(False)
        actor.eval();actor.bounded_checkpoint_proof=verify_checkpoint(actor,context)
    finally:files.close()


def verify_checkpoint(actor,context):
    require_context(context);context.require_runtime()
    if getattr(actor,'bounded_evaluation_context',None) is not context:
        raise PermissionError('Another actor entered the frozen V4 stage')
    from .composite_training import tensor_digest
    from .composite_lora import trainables
    parameters=trainables(actor.adapters)
    if len(parameters)!=992 or sum(p.numel() for p in parameters)!=233455616 or any(p.requires_grad or p.grad is not None for p in parameters):
        raise PermissionError('All992 frozen FP32 adapter tensors without gradients are required')
    hashes={name+'.'+part:tensor_digest(getattr(m,part)) for name,m in actor.adapters.items() for part in ['A','B']}
    geometry={name+'.'+part:{'shape':list(getattr(m,part).shape),'dtype':str(getattr(m,part).dtype),
      'device':str(getattr(m,part).device),'requires_grad':getattr(m,part).requires_grad,
      'gradient_is_none':getattr(m,part).grad is None} for name,m in actor.adapters.items() for part in ['A','B']}
    value=require_frozen_model(context.model)
    if hashes!=value['checkpoint']['hashes']:raise PermissionError('Live frozen checkpoint changed during evaluation')
    return {'checkpoint_sha256':value['checkpoint_sha256'],'live_tensor_hashes':hashes,'geometry':geometry,
      'tensors':992,'adapter_parameters':233455616,'optimizer_state_loaded':False,'fitting_allowed':False}
