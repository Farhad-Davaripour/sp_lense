"""Independent NumPy formulas for frozen Qwen attention and SwiGLU response."""
import numpy as np

def norm(x,w,dx=None,eps=1e-6):
    x=np.asarray(x,dtype=np.float64);s=1/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)
    y=x*s*(1+w)
    if dx is None:return y
    dx=np.asarray(dx,dtype=np.float64)
    return y,(dx*s-x*s**3*np.mean(x*dx,axis=-1,keepdims=True))*(1+w)

def sigmoid(x):
    x=np.asarray(x,dtype=np.float64)
    return np.exp(-np.logaddexp(0.,-x))

def rotate(x,cos,sin):
    n=len(cos);r=x[...,:n];half=n//2
    rh=np.concatenate((-r[...,half:],r[...,:half]),axis=-1)
    return np.concatenate((r*cos+rh*sin,x[...,n:]),axis=-1)

def mlp(x,p,dx=None):
    if dx is None:z=norm(x,p['norm'])
    else:z,dz=norm(x,p['norm'],dx)
    a=p['gate']@z;u=p['up']@z;s=sigmoid(a);g=a*s
    y=p['down']@(g*u)
    if dx is None:return y
    da=p['gate']@dz;du=p['up']@dz
    dy=p['down']@((s+a*s*(1-s))*da*u+g*du)
    return y,dy

def attention(x,K,V,p,cos,sin,dx=None,dK=None,dV=None):
    if dx is None:z=norm(x,p['norm'])
    else:z,dz=norm(x,p['norm'],dx)
    qg=(p['q']@z).reshape(8,512);qr=qg[:,:256];gate=qg[:,256:]
    if dx is None:qn=norm(qr,p['qnorm'])
    else:
        dqg=(p['q']@dz).reshape(8,512);dqr=dqg[:,:256];dgate=dqg[:,256:]
        qn,dqn=norm(qr,p['qnorm'],dqr)
    q=rotate(qn,cos,sin)
    K=np.repeat(K,4,axis=0);V=np.repeat(V,4,axis=0)
    scores=np.einsum('hd,htd->ht',q,K)/16.
    shift=scores.max(axis=-1,keepdims=True);logZ=shift+np.log(np.exp(scores-shift).sum(axis=-1,keepdims=True))
    logP=scores-logZ;P=np.exp(logP)
    a=np.einsum('ht,htd->hd',P,V);g=sigmoid(gate);y=p['out']@(a*g).reshape(-1)
    candidate=.5*P+.5/P.shape[-1]
    energy=np.sum(P*(logP-scores),axis=-1)
    cand_energy=np.sum(candidate*(np.log(candidate)-scores),axis=-1)
    KL=np.sum(candidate*(np.log(candidate)-logP),axis=-1)
    stationarity=logP+1-scores;stationarity-=stationarity.mean(axis=-1,keepdims=True)
    record={'scores':scores,'P':P,'logP':logP,'logZ':logZ,'entropy':-np.sum(P*logP,axis=-1),
        'stationarity':np.max(np.abs(stationarity),axis=-1),'minimum_identity_error':np.abs(energy+logZ[:,0]),
        'candidate_KL':KL,'objective_gap_identity_error':np.abs(cand_energy-energy-KL)}
    if dx is None:return y,record
    dq=rotate(dqn,cos,sin);dK=np.repeat(dK,4,axis=0);dV=np.repeat(dV,4,axis=0)
    ds=(np.einsum('hd,htd->ht',dq,K)+np.einsum('hd,htd->ht',q,dK))/16.
    dP=P*(ds-np.sum(P*ds,axis=-1,keepdims=True))
    da=np.einsum('ht,htd->hd',dP,V)+np.einsum('ht,htd->hd',P,dV)
    dy=p['out']@(da*g+a*g*(1-g)*dgate).reshape(-1)
    record.update({'ds':ds,'dP':dP,'simplex_derivative_error':np.abs(dP.sum(axis=-1)),
        'common_mode_invariance_error':np.max(np.abs(P*(np.ones_like(P)-np.sum(P,axis=-1,keepdims=True))),axis=-1)})
    return y,dy,record
