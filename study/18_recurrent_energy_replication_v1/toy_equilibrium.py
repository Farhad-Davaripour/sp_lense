"""Synthetic closed-form checks; not evidence about native Qwen yet."""
import json
from pathlib import Path
import numpy as np
OUT=Path('C:/Users/farha/Research4/artifacts/recurrent_energy_replication_v1')
OUT.mkdir(exist_ok=True)
def state(a):
    k=np.array([1.,a]);k/=np.linalg.norm(k);beta=.5+.1*a;gamma=.8-.03*a
    previous=np.array([[2.+.3*a],[3.-.1*a]]);v=np.array([4.+.2*a]);B=gamma*previous;c=float(k@k)
    lam=beta/(1-beta*c);K=np.eye(2)+lam*np.outer(k,k);f=B+lam*k[:,None]*v[None,:]
    S=np.linalg.solve(K,f);direct=B+beta*k[:,None]*(v-B.T@k)[None,:]
    assert np.allclose(S,direct,rtol=0,atol=1e-14)
    return k,beta,gamma,previous,v,B,lam,K,f,S
k,beta,gamma,previous,v,B,lam,K,f,S=state(0.)
D=np.array([[.1],[.2]])
internal=float(np.sum(D*(K@S)));external=float(np.sum(D*f));assert abs(internal-external)<1e-14
loss_before=float(np.sum((B.T@k-v)**2)/2);loss_after=float(np.sum((S.T@k-v)**2)/2)
factor=(1-beta*float(k@k))**2;assert abs(loss_after-factor*loss_before)<1e-14
def potential(X):return float(np.sum((X-B)**2)/2+lam*np.sum((X.T@k-v)**2)/2)
gap=potential(S+D)-potential(S);assert abs(gap-float(np.sum(D*(K@D))/2))<1e-14
dk=np.array([0.,1.]);dbeta=.1;dgamma=-.03;dprevious=np.array([[.3],[-.1]]);dv=np.array([.2])
dc=2*float(k@dk);dlam=(dbeta+beta*beta*dc)/(1-beta*float(k@k))**2
dB=dgamma*previous+gamma*dprevious
dK=dlam*np.outer(k,k)+lam*(np.outer(dk,k)+np.outer(k,dk))
df=dB+dlam*k[:,None]*v[None,:]+lam*dk[:,None]*v[None,:]+lam*k[:,None]*dv[None,:]
implicit=np.linalg.solve(K,df-dK@S);step=1e-6
finite=(state(step)[-1]-state(-step)[-1])/(2*step);error=float(np.max(np.abs(implicit-finite)));assert error<1e-8
result={'classification':'synthetic formula verification only; native-model replication remains unexecuted',
    'K':K.tolist(),'external_load':f.tolist(),'equilibrium_state':S.tolist(),'internal_virtual_work':internal,'external_virtual_work':external,
    'regression_loss_before':loss_before,'regression_loss_after':loss_after,'predicted_loss_ratio':factor,'nearby_potential_gap':gap,
    'implicit_state_response':implicit.tolist(),'implicit_vs_finite_difference_max_error':error,'new_backbone_forwards':0}
p=OUT/'SYNTHETIC_EQUILIBRIUM_CHECK_20261014.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
