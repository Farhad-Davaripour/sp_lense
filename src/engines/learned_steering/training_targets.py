"""Exact case-weighted target subspaces and bias-free ridge initialization."""
import torch

def target_basis(y,weights,maximum=32):
    y=y.double();weights=weights.double();weights=weights/weights.mean()
    z=y*weights.sqrt()[:,None]
    z=z[z.square().sum(-1)>0]
    total=float(z.square().sum())
    if not len(z):return y.new_empty((y.shape[1],0)),total
    # Zero rows do not alter the right singular subspace. Diagonalize the
    # smaller exact Gram matrix, retaining only the needed right vectors.
    if len(z)<z.shape[1]:
        gram=z@z.T;values,vectors=torch.linalg.eigh((gram+gram.T)*.5)
        values=values.flip(0);vectors=vectors.flip(1)
        cutoff=values[0].clamp_min(1)*torch.finfo(values.dtype).eps*max(z.shape)
        count=min(maximum,int((values>cutoff).sum()))
        basis=z.T@vectors[:,:count]/values[:count].sqrt()[None,:]
    else:
        gram=z.T@z;values,vectors=torch.linalg.eigh((gram+gram.T)*.5)
        values=values.flip(0);vectors=vectors.flip(1)
        cutoff=values[0].clamp_min(1)*torch.finfo(values.dtype).eps*max(z.shape)
        count=min(maximum,int((values>cutoff).sum()));basis=vectors[:,:count]
    if basis.shape[1]:basis=torch.linalg.qr(basis,mode='reduced').Q
    assert torch.isfinite(basis).all()
    return basis,total

def projected_energy(y,weights,basis,rank):
    w=weights.double();w=w/w.mean();z=y.double()*w.sqrt()[:,None]
    return float((z@basis[:,:rank]).square().sum())

def initialize_with_basis(module,x,y,weights,basis):
    x,y,w=x.double(),y.double(),weights.double();w=w/w.mean()
    count=min(module.up.shape[1],basis.shape[1]);u=basis[:,:count].double()
    if count:
        xw=x*w.sqrt()[:,None];rhs=(y@u)*w.sqrt()[:,None]
        if len(x)<x.shape[1]:
            gram=xw@xw.T+torch.eye(len(x),device=x.device,dtype=x.dtype)
            down=xw.T@torch.cholesky_solve(rhs,torch.linalg.cholesky(gram))
        else:
            gram=xw.T@xw+torch.eye(x.shape[1],device=x.device,dtype=x.dtype)
            down=torch.cholesky_solve(xw.T@rhs,torch.linalg.cholesky(gram))
    with torch.no_grad():
        module.up.zero_()
        if count:module.down[:,:count].copy_(down.float());module.up[:,:count].copy_(u.float())
    return {'rank':module.up.shape[1],'available_target_directions':count,'ridge_penalty':1.,'bias':False}
