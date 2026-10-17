"""Small wrappers around native directional differentiation; no learned model."""
import time
import numpy as np
import torch
import torch.nn.functional as tf
from native import Rig, NAMES, array


class JetRig(Rig):
    def first_at(self, ids, site, direction, beta_value=0.):
        with torch.no_grad(), torch.autograd.forward_ad.dual_level():
            beta=torch.autograd.forward_ad.make_dual(torch.tensor(beta_value,dtype=torch.float32),torch.tensor(1.))
            points=self.execute(ids,site,direction,beta)
            pairs=[torch.autograd.forward_ad.unpack_dual(p) for p in points]
            primal=[p.detach().clone() for p,t in pairs]
            tangent=[torch.zeros_like(p) if t is None else t.detach().clone() for p,t in pairs]
            assert pairs[-1][1] is not None
        with torch.no_grad():
            logits=self.model.lm_head(primal[-1])
            q=tf.linear(tangent[-1],self.model.lm_head.weight)
        return {'primal':{n:array(p) for n,p in zip(NAMES,primal)},
                'first':{n:array(p) for n,p in zip(NAMES,tangent)},
                'logits':array(logits),'q':array(q),'prefix_hashes':dict(self.last_prefix_hashes)}

    def exact_jet(self,ids,site,direction):
        saved={}
        def first(beta):
            values,tangents=torch.autograd.functional.jvp(
                lambda b:self.execute(ids,site,direction,b,graph=True),
                beta,torch.ones_like(beta),create_graph=True)
            saved['primal']=values
            return tangents
        begin=time.monotonic()
        with torch.enable_grad():
            q,r=torch.autograd.functional.jvp(first,torch.tensor(0.),torch.tensor(1.))
        with torch.no_grad():
            logits=self.model.lm_head(saved['primal'][-1])
            qlog=tf.linear(q[-1],self.model.lm_head.weight)
            rlog=tf.linear(r[-1],self.model.lm_head.weight)
        return {'primal':{n:array(p) for n,p in zip(NAMES,saved['primal'])},
                'first':{n:array(p) for n,p in zip(NAMES,q)},
                'second':{n:array(p) for n,p in zip(NAMES,r)},
                'logits':array(logits),'q':array(qlog),'r':array(rlog),
                'seconds':time.monotonic()-begin,'prefix_hashes':dict(self.last_prefix_hashes)}

    def central_second(self,plus,minus,step):
        second={n:(plus['first'][n].astype(np.float64)-minus['first'][n])/(2*step) for n in NAMES}
        with torch.no_grad():
            r=tf.linear(torch.tensor(second['hidden'],dtype=torch.float32),self.model.lm_head.weight)
        # Preserve the alternative head-first finite difference for numerical audit.
        r_head_first=(plus['q'].astype(np.float64)-minus['q'])/(2*step)
        return {'second':second,'r':array(r),'r_head_first':r_head_first}


def probability(logits):
    x=np.asarray(logits,dtype=np.float64);m=x.max()
    logz=m+np.log(np.exp(x-m).sum());lp=x-logz
    return np.exp(lp),lp


def kl(a,b):
    p,lp=probability(a);_,lq=probability(b)
    value=float(p@(lp-lq));assert value>=-1e-10
    return max(0.,value)


def weighted_norm(logits,v):
    p,_=probability(logits);v=np.asarray(v,dtype=np.float64);v=v-p@v
    return float(np.sqrt(max(0.,p@(v*v))))


def relative(a,b):
    b=np.asarray(b,dtype=np.float64);den=float(np.linalg.norm(b))
    return float(np.linalg.norm(np.asarray(a,dtype=np.float64)-b)/den) if den>1e-12 else None


def consistency(logits,a,b):
    wn=weighted_norm(logits,b['r']);wd=weighted_norm(logits,np.asarray(a['r'],dtype=np.float64)-b['r'])
    hidden_norm=float(np.linalg.norm(b['second']['hidden']))
    hidden_diff=float(np.linalg.norm(a['second']['hidden']-b['second']['hidden']))
    return {'weighted_r_norm':wn,'weighted_r_difference':wd,
        'weighted_r_relative':wd/max(wn,1e-30),'hidden_r_norm':hidden_norm,
        'hidden_r_difference':hidden_diff,'hidden_r_relative':hidden_diff/max(hidden_norm,1e-30),
        'pass':bool(wd<=max(.03*wn,1e-6) and hidden_diff<=max(.05*hidden_norm,1e-6))}
