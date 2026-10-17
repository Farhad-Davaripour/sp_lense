"""Bias-free, capacity-matched residual maps; only these matrices may train."""
import torch

class LowRankMap(torch.nn.Module):
    def __init__(self,width,rank):
        super().__init__();assert 0<rank<=width
        self.down=torch.nn.Parameter(torch.empty(width,rank,dtype=torch.float32))
        self.up=torch.nn.Parameter(torch.zeros(width,rank,dtype=torch.float32))
        torch.nn.init.orthogonal_(self.down)
    def forward(self,h):return (h.float()@self.down)@self.up.T
    @property
    def learned_parameters(self):return self.down.numel()+self.up.numel()

def matched_maps(width,blocks,total_rank,variant):
    assert total_rank%2==0 and variant in ('one_site','two_site')
    maps=torch.nn.ModuleDict()
    for block in blocks:
        rank=total_rank if variant=='one_site' else total_rank//2
        maps[f'{block}/output']=LowRankMap(width,rank)
        if variant=='two_site':maps[f'{block}/attention']=LowRankMap(width,rank)
    assert sum(p.numel() for p in maps.parameters())==2*width*total_rank*len(blocks)
    return maps

def ridge_initialize(module,x,y,case_weights,penalty=1.):
    """Weighted ridge into the target SVD subspace, then factor the linear map.

    A diagnostic with fewer rows than rank has a lower empirical rank and cannot
    by itself select the all-TRAIN capacity. No duplicated or quadratic inputs.
    """
    x,y,w=x.double(),y.double(),case_weights.double()
    w=w/w.mean();root=w.sqrt()[:,None];rank=module.up.shape[1]
    _,_,vh=torch.linalg.svd(y*root,full_matrices=False)
    available=min(rank,len(vh));basis=vh[:available].T
    # Dual solve avoids a width-squared matrix when pilot rows are few.
    xw=x*root;coeff=(y@basis)*root
    if len(x)<x.shape[1]:weights=xw.T@torch.linalg.solve(xw@xw.T+penalty*torch.eye(len(x),device=x.device,dtype=x.dtype),coeff)
    else:weights=torch.linalg.solve(xw.T@xw+penalty*torch.eye(x.shape[1],device=x.device,dtype=x.dtype),xw.T@coeff)
    with torch.no_grad():
        # Keep nonzero input factors for any padded directions so output-KL
        # refinement can learn them; two zero factors would be a dead product.
        module.up.zero_()
        module.down[:,:available].copy_(weights.float());module.up[:,:available].copy_(basis.float())
    return {'requested_rank':rank,'empirical_target_rank_limit':available,'bias':False,'ridge_penalty':penalty,'padded_directions_remain_trainable':True}

def distillation_objective(student_logits,teacher_logits,predicted,local_targets,local_scales,gate_on):
    """Prospectively fixed T=1 forward KL plus normalized local/size terms."""
    teacher_p=torch.softmax(teacher_logits.detach().float(),-1)
    kl=(teacher_p*(torch.log_softmax(teacher_logits.detach().float(),-1)-torch.log_softmax(student_logits.float(),-1))).sum(-1).mean()
    assert len(predicted)==len(local_targets)==len(local_scales)>0
    # Fixed per-site active-TRAIN mean-square targets normalize both own and
    # zero-control rows. A zero target must not supply its own zero denominator.
    local=torch.stack([((p-t.detach()).square().mean()/torch.as_tensor(s,device=p.device).clamp_min(1e-6)) for p,t,s in zip(predicted,local_targets,local_scales)]).mean()
    size=torch.stack([p.square().mean() for p in predicted]).mean()
    control=size if not gate_on else size*0
    return kl+.1*local+.01*control+1e-4*size,{'forward_KL':kl,'normalized_local_MSE':local,'control_penalty':control,'size_penalty':size}
