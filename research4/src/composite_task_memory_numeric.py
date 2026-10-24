"""Read-only three-update tensor replay; no model, teacher or optimizer call."""
import math
from .composite_v2_model_audit import tensor_state,audit_optimizer_bytes
from .composite_task_skill_model_audit import audit_clipped_gradients,audit_optimizer_config


def audit_three_updates(files,shapes,initial_hashes):
    import torch
    from .composite_training import tensor_digest
    audit_optimizer_config(files)
    member=lambda n:files.tensor_path(n) if hasattr(files,'tensor_path') else files.get(n)
    initial=tensor_state(member('INITIAL_A.pt'))
    checkpoint=tensor_state(member('behavior.pt'))['weights']
    gradients=[tensor_state(member('GRADIENT_STEP_'+str(i).zfill(3)+'.pt')) for i in [1,2,3]]
    expected_A={n for n in shapes if n.endswith('.A')}
    if set(initial)!=expected_A or set(checkpoint)!=set(shapes) or set(initial_hashes)!=set(shapes):
        raise ValueError('Complete initial/final adapter tensor rosters required')
    clip=[audit_clipped_gradients(g,shapes) for g in gradients]
    for name in expected_A:
        value=initial[name]
        if value.dtype!=torch.float32 or list(value.shape)!=shapes[name] or not torch.isfinite(value).all():
            raise ValueError('Initial A geometry/precision/finite bytes differ')
        if tensor_digest(value)!=initial_hashes[name] or torch.count_nonzero(gradients[0][name])!=0:
            raise ValueError('Fresh initial A hash or zero-B first A gradient differs')
    for name,shape in shapes.items():
        if name.endswith('.B') and tensor_digest(torch.zeros(shape,dtype=torch.float32))!=initial_hashes[name]:
            raise ValueError('All B matrices must begin from fresh zeros')
        if checkpoint[name].dtype!=torch.float32 or list(checkpoint[name].shape)!=shape or not torch.isfinite(checkpoint[name]).all():
            raise ValueError('Complete final FP32 checkpoint tensors required')
    maximum_error=0.0
    def inspect(name,state):
        nonlocal maximum_error
        predicted=initial[name].clone() if name.endswith('.A') else torch.zeros(shapes[name],dtype=torch.float32)
        first=torch.zeros_like(predicted);second=torch.zeros_like(predicted)
        for step,values in enumerate(gradients,1):
            g=values[name]
            first.lerp_(g,0.1);second.mul_(0.999).addcmul_(g,g,value=0.001)
            denominator=second.sqrt().div_(math.sqrt(1-0.999**step)).add_(1e-8)
            predicted.addcdiv_(first,denominator,value=-0.0005/(1-0.9**step))
        torch.testing.assert_close(state['exp_avg'],first,rtol=1e-5,atol=1e-12)
        torch.testing.assert_close(state['exp_avg_sq'],second,rtol=1e-5,atol=1e-20)
        torch.testing.assert_close(checkpoint[name],predicted,rtol=1e-5,atol=1e-8)
        maximum_error=max(maximum_error,float((checkpoint[name]-predicted).abs().max()))
    proof=audit_optimizer_bytes(member('OPTIMIZER_FINAL.pt'),shapes,3,inspect=inspect)
    return {**proof,'numeric_AdamW_updates_independently_recomputed':3,
      'initial_A_tensors_verified':len(initial),'complete_clipped_gradient_maps_verified':3,
      'clipped_global_norms':[p['clipped_global_norm'] for p in clip],
      'maximum_checkpoint_absolute_error':maximum_error,'FP32_update_tolerance':{'rtol':1e-5,'atol':1e-8},
      'model_forwards':0,'new_optimizer_updates':0,'scientific_qualification_admitted':False}
