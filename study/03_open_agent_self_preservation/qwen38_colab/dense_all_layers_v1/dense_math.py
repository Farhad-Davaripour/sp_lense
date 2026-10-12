"""Case-weighted dense ridge; frozen language models and singleton deployment."""
import numpy as np

FIELDS=('x_mean','input_basis','scale','basis','weights')
FIT_RANK=8
INFERENCE_RANK=4

def fit(x,y,active,row_weights,device='cuda',seed=20261009,input_rank=32,output_rank=FIT_RANK,ridge=1.):
    import torch
    torch.manual_seed(seed)
    x=torch.as_tensor(x,dtype=torch.float32,device=device)
    y=torch.as_tensor(y,dtype=torch.float32,device=device)
    active=torch.as_tensor(active,dtype=torch.bool,device=device)
    w=torch.as_tensor(row_weights,dtype=torch.float32,device=device)
    assert x.ndim==2 and x.shape==y.shape and w.shape==active.shape==(len(x),)
    assert bool(active.any()) and bool(torch.isfinite(x).all() & torch.isfinite(y).all())
    assert bool(torch.isfinite(w).all() & (w>0).all())
    w=w/w.mean();wa=w[active]
    mean=(y[active]*wa[:,None]).sum(0)/wa.sum();mean_norm=float(mean.norm())
    _,_,residual=torch.pca_lowrank((y[active]-mean)*wa.sqrt()[:,None],q=output_rank,center=False,niter=3)
    anchor=mean/mean.norm() if mean_norm>1e-10 else residual[:,0]
    tail=residual[:,:output_rank-1] if mean_norm>1e-10 else residual[:,1:output_rank]
    basis=torch.linalg.qr(torch.cat([anchor[:,None],tail],1)).Q
    x_mean=(x*w[:,None]).sum(0)/w.sum();center=x-x_mean
    _,_,input_basis=torch.pca_lowrank(center*w.sqrt()[:,None],q=input_rank,center=False,niter=3)
    raw=center@input_basis;scale=((raw.square()*w[:,None]).sum(0)/w.sum()).sqrt().clamp_min(1e-6)
    z=raw/scale;z=torch.cat([z,z,torch.ones((len(z),1),device=device)],1)
    coefficients=y@basis;zd=z.double();yd=coefficients.double();wd=w.double()
    penalty=torch.eye(z.shape[1],device=device,dtype=torch.float64)*ridge;penalty[-1,-1]=0
    weights=torch.linalg.solve(zd.T@(zd*wd[:,None])+penalty,zd.T@(yd*wd[:,None])).float()
    result={'x_mean':x_mean,'input_basis':input_basis,'scale':scale,'basis':basis,'weights':weights,
            'train_z':z,'train_coefficients':coefficients,'row_weights':w,
            'ridge_lambda':torch.tensor(ridge,device=device)}
    return {k:v.detach().cpu().numpy() for k,v in result.items()}

def replay_fit(a):
    z=a['train_z'].astype(np.float64);y=a['train_coefficients'].astype(np.float64);w=a['row_weights'].astype(np.float64)
    penalty=np.eye(z.shape[1])*float(a['ridge_lambda']);penalty[-1,-1]=0
    return np.linalg.solve(z.T@(z*w[:,None])+penalty,z.T@(y*w[:,None]))

def predict_numpy(x,a,rank=INFERENCE_RANK):
    if not 1<=rank<=a['basis'].shape[1] or a['weights'].shape[1]!=a['basis'].shape[1]:raise ValueError('Invalid deployed direction count')
    z=((x-a['x_mean'])@a['input_basis'])/a['scale']
    z=np.concatenate([z,z,np.ones((len(z),1),dtype=z.dtype)],1)
    return (z@a['weights'][:,:rank])@a['basis'][:,:rank].T

class Predictor:
    def __init__(self,arrays,device,rank=INFERENCE_RANK,strength=1.):
        import torch
        if not 1<=rank<=arrays['basis'].shape[1] or arrays['weights'].shape[1]!=arrays['basis'].shape[1]:raise ValueError('Invalid deployed direction count')
        self.a={k:torch.as_tensor(arrays[k],device=device,dtype=torch.float32) for k in FIELDS}
        self.width=len(arrays['x_mean']);self.rank=rank;self.strength=strength
    def __call__(self,h):
        import torch
        assert h.shape==(1,self.width)
        z=((h.float()-self.a['x_mean'])@self.a['input_basis'])/self.a['scale']
        z=torch.cat([z,z,torch.ones_like(z[:,:1])],1)
        return ((z@self.a['weights'][:,:self.rank])@self.a['basis'][:,:self.rank].T)*self.strength
