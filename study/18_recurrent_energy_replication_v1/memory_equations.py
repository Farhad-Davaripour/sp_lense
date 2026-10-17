"""Exact local frozen-parameter replay and independent quadratic descriptions."""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

def array(t):
    p,_=torch.autograd.forward_ad.unpack_dual(t)
    return p.detach().float().cpu().numpy().copy()
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def norms(x):return np.sqrt(np.sum(np.asarray(x,dtype=np.float64)**2,axis=tuple(range(1,np.asarray(x).ndim))))
def dot(a,b):return np.sum(np.asarray(a,dtype=np.float64)*np.asarray(b,dtype=np.float64),axis=tuple(range(1,np.asarray(a).ndim)))

class MemoryEquations:
    def __init__(self,parameters,layer_metadata):
        self.parameters={n:torch.tensor(v,dtype=torch.float32) for n,v in parameters.items()}
        self.metadata={m['layer']:m for m in layer_metadata}
        self.local_primal_replays=0;self.local_jvps=0
    def drivers(self,layer,h,window):
        m=self.metadata[layer];p=self.parameters;prefix=f'l{layer:02d}_'
        h=h.reshape(1,1,-1)
        h=h.float()*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+m['rms_epsilon'])
        h=h*(1.+p[prefix+'norm_weight'])
        a=F.linear(h,p[prefix+'in_proj_a_weight']);b=F.linear(h,p[prefix+'in_proj_b_weight'])
        beta=b.sigmoid().reshape(-1)
        gamma=(-p[prefix+'A_log'].float().exp()*F.softplus(a.float()+p[prefix+'dt_bias'])).exp().reshape(-1)
        mixed=F.conv1d(window,p[prefix+'conv_weight'],p.get(prefix+'conv_bias'),groups=window.shape[1])
        mixed=F.silu(mixed).transpose(1,2)
        kd=m['key_heads']*m['key_dim'];vd=m['heads']*m['value_dim']
        _,key,value=torch.split(mixed,[kd,kd,vd],dim=-1)
        key=key.reshape(m['key_heads'],m['key_dim']);value=value.reshape(m['heads'],m['value_dim'])
        key=key.repeat_interleave(m['heads']//m['key_heads'],dim=0)
        key=key*torch.rsqrt((key*key).sum(-1,keepdim=True)+1e-6)
        return {'k':key,'v':value,'beta':beta,'gamma':gamma}
    @staticmethod
    def update(previous,driver):
        previous=previous.reshape(driver['k'].shape[0],driver['k'].shape[1],driver['v'].shape[1])
        B=previous*driver['gamma'][:,None,None]
        predicted=(B*driver['k'][:,:,None]).sum(-2)
        correction=(driver['v']-predicted)*driver['beta'][:,None]
        return B+driver['k'][:,:,None]*correction[:,None,:]
    def replay(self,layer,h,window,previous):
        with torch.no_grad():
            driver=self.drivers(layer,torch.tensor(h),torch.tensor(window))
            state=self.update(torch.tensor(previous),driver)
        self.local_primal_replays+=1
        return {n:array(t) for n,t in driver.items()},array(state)
    def jvp(self,layer,h,window,previous,dh,dwindow,dprevious):
        originals=[h,window,previous,dh,dwindow,dprevious];before=[digest(x) for x in originals]
        with torch.no_grad(),torch.autograd.forward_ad.dual_level():
            hd=torch.autograd.forward_ad.make_dual(torch.tensor(h),torch.tensor(dh).clone())
            wd=torch.autograd.forward_ad.make_dual(torch.tensor(window),torch.tensor(dwindow).clone())
            pd=torch.autograd.forward_ad.make_dual(torch.tensor(previous),torch.tensor(dprevious).clone())
            driver=self.drivers(layer,hd,wd);state=self.update(pd,driver)
            d={};q={}
            for n,t in {**driver,'state':state}.items():
                primal,tangent=torch.autograd.forward_ad.unpack_dual(t)
                d[n]=array(primal);q[n]=np.zeros_like(d[n]) if tangent is None else array(tangent)
        assert before==[digest(x) for x in originals],'Local replay changed caller data'
        self.local_jvps+=1
        return d,q

def apply_K(state,driver):
    S=np.asarray(state,dtype=np.float64);k=np.asarray(driver['k'],dtype=np.float64)
    gamma=np.asarray(driver['gamma'],dtype=np.float64);beta=np.asarray(driver['beta'],dtype=np.float64)
    return (1-gamma)[:,None,None]*S+(gamma*beta)[:,None,None]*k[:,:,None]*np.sum(k[:,:,None]*S,axis=-2)[:,None,:]
def force(driver):return np.asarray(driver['beta'],dtype=np.float64)[:,None,None]*np.asarray(driver['k'],dtype=np.float64)[:,:,None]*np.asarray(driver['v'],dtype=np.float64)[:,None,:]
def potential(state,driver):return .5*dot(state,apply_K(state,driver))-dot(force(driver),state)

def evaluate_head_equations(previous,native,replayed,driver,virtual):
    previous=np.asarray(previous,dtype=np.float64).reshape(replayed.shape);native=np.asarray(native,dtype=np.float64).reshape(replayed.shape)
    k=np.asarray(driver['k'],dtype=np.float64);v=np.asarray(driver['v'],dtype=np.float64)
    beta=np.asarray(driver['beta'],dtype=np.float64);gamma=np.asarray(driver['gamma'],dtype=np.float64);c=np.sum(k*k,axis=-1)
    B=gamma[:,None,None]*previous;e=v-np.sum(k[:,:,None]*B,axis=-2)
    ideal=B+beta[:,None,None]*k[:,:,None]*e[:,None,:]
    step=native-previous;Kold=apply_K(previous,driver);load=force(driver);Kstep=apply_K(step,driver)
    residual=step-(load-Kold)
    W=dot(load,step);deltaU=dot(Kold,step)+.5*dot(step,Kstep);D=dot(step,step)-.5*dot(step,Kstep)
    work_error=np.abs(W-deltaU-D);work_bound=norms(residual)*norms(step)+100*np.finfo(np.float64).eps*np.maximum(1.,np.maximum(np.abs(W),np.abs(deltaU)))
    before=.5*np.sum(e*e,axis=-1);after_error=np.sum(k[:,:,None]*native,axis=-2)-v
    after=.5*np.sum(after_error*after_error,axis=-1);factor=(1-beta*c)**2;loss_pred=factor*before
    loss_floor=100*np.finfo(np.float32).eps*np.maximum(1.,np.maximum(before,after))
    field_relative=float(np.linalg.norm(replayed.astype(np.float64)-native)/max(1.,np.linalg.norm(native)))
    native_norm=norms(native);primal_error=norms(replayed-native)/np.maximum(1.,native_norm)
    step_gradient_error=norms(native-ideal)/np.maximum(1.,native_norm)
    force_den=np.maximum(1.,np.maximum(norms(load),norms(Kold)));force_relative=norms(residual)/force_den
    virtual_residual=np.abs(np.einsum('hdij,hij->hd',virtual,Kold+step-load))
    virtual_internal=np.einsum('hdij,hij->hd',virtual,Kold+step);virtual_external=np.einsum('hdij,hij->hd',virtual,load)
    virtual_bound=1e-5*np.maximum(1.,np.maximum(np.abs(virtual_internal),np.abs(virtual_external)))
    eligible=(beta*c<1)&np.isfinite(beta*c);den=1-beta*c
    lam=np.full(beta.shape,np.nan);lam[eligible]=beta[eligible]/den[eligible]
    prox_relative=np.full(beta.shape,np.nan);prox_virtual=np.full(beta.shape,np.nan)
    for h in np.flatnonzero(eligible):
        internal=native[h]+lam[h]*k[h,:,None]*(k[h]@native[h])[None,:]
        external=B[h]+lam[h]*k[h,:,None]*v[h,None,:]
        prox_relative[h]=np.linalg.norm(internal-external)/max(1.,np.linalg.norm(external))
        prox_virtual[h]=np.max(np.abs(np.einsum('dij,ij->d',virtual[h],internal-external))/np.maximum(1.,np.abs(np.einsum('dij,ij->d',virtual[h],external))))
    low=1-gamma;high=low+gamma*beta*c;energy_change=potential(native,driver)-potential(previous,driver)
    energy_bound=norms(residual)*norms(step)+loss_floor
    rows=[]
    for h in range(len(beta)):
        rows.append({'head':h,'beta':float(beta[h]),'gamma':float(gamma[h]),'key_norm_squared':float(c[h]),
            'native_replay_relative':float(primal_error[h]),'published_step_stationarity_relative':float(step_gradient_error[h]),
            'force_balance_relative':float(force_relative[h]),'virtual_force_max_relative':float(np.max(virtual_residual[h]/np.maximum(1.,np.abs(virtual_external[h])))),
            'regression_loss_before':float(before[h]),'regression_loss_after':float(after[h]),'predicted_loss_after':float(loss_pred[h]),
            'loss_numerically_resolved':bool(before[h]>loss_floor[h]),'loss_factor_identity_pass':bool(abs(after[h]-loss_pred[h])<=loss_floor[h]),
            'external_work':float(W[h]),'internal_quadratic_energy_change':float(deltaU[h]),'discrete_dissipation':float(D[h]),
            'work_balance_error':float(work_error[h]),'work_balance_bound':float(work_bound[h]),'work_balance_pass':bool(work_error[h]<=work_bound[h]+1e-10),
            'fixed_driver_potential_change':float(energy_change[h]),'potential_nonincrease_pass':bool(energy_change[h]<=energy_bound[h]),
            'stiffness_orthogonal':float(low[h]),'stiffness_along_key':float(high[h]),'dynamics_stiffness_PSD_and_stable':bool(low[h]>=-1e-7 and high[h]>=-1e-7 and high[h]<=2+1e-7),
            'unique_fixed_driver_equilibrium':bool(low[h]>0),
            'proximal_eligible':bool(eligible[h]),'proximal_stationarity_relative':float(prox_relative[h]) if eligible[h] else None,
            'proximal_virtual_max_relative':float(prox_virtual[h]) if eligible[h] else None,
            'force_and_virtual_pass':bool(force_relative[h]<=1e-5 and np.all(virtual_residual[h]<=virtual_bound[h])),
            'primal_pass':bool(primal_error[h]<=1e-6),'published_step_pass':bool(step_gradient_error[h]<=1e-5),
            'proximal_pass':bool(prox_relative[h]<=1e-5 and prox_virtual[h]<=1e-5) if eligible[h] else None})
    return rows,field_relative

def response_terms(previous,dprevious,driver,qdriver):
    S=np.asarray(previous,dtype=np.float64);dS=np.asarray(dprevious,dtype=np.float64)
    k=np.asarray(driver['k'],dtype=np.float64);v=np.asarray(driver['v'],dtype=np.float64)
    b=np.asarray(driver['beta'],dtype=np.float64);g=np.asarray(driver['gamma'],dtype=np.float64)
    dk=np.asarray(qdriver['k'],dtype=np.float64);dv=np.asarray(qdriver['v'],dtype=np.float64)
    db=np.asarray(qdriver['beta'],dtype=np.float64);dg=np.asarray(qdriver['gamma'],dtype=np.float64)
    old_value=np.sum(k[:,:,None]*S,axis=-2);dkey_old=np.sum(dk[:,:,None]*S,axis=-2)
    transport=dS-apply_K(dS,driver)
    dforce=db[:,None,None]*k[:,:,None]*v[:,None,:]+b[:,None,None]*dk[:,:,None]*v[:,None,:]+b[:,None,None]*k[:,:,None]*dv[:,None,:]
    dstiff=-dg[:,None,None]*S+(dg*b+g*db)[:,None,None]*k[:,:,None]*old_value[:,None,:]
    dstiff+=(g*b)[:,None,None]*(dk[:,:,None]*old_value[:,None,:]+k[:,:,None]*dkey_old[:,None,:])
    predicted=transport+dforce-dstiff
    return predicted,{'transported_history':transport,'new_input_force':dforce,'stiffness_change':-dstiff}

def proximal_response(previous,dprevious,native_state,driver,qdriver,target):
    """Optional implicit form; preserve eligibility and cancellation limitations."""
    S=np.asarray(previous,dtype=np.float64);dS=np.asarray(dprevious,dtype=np.float64)
    k=np.asarray(driver['k'],dtype=np.float64);v=np.asarray(driver['v'],dtype=np.float64)
    b=np.asarray(driver['beta'],dtype=np.float64);g=np.asarray(driver['gamma'],dtype=np.float64)
    dk=np.asarray(qdriver['k'],dtype=np.float64);dv=np.asarray(qdriver['v'],dtype=np.float64)
    db=np.asarray(qdriver['beta'],dtype=np.float64);dg=np.asarray(qdriver['gamma'],dtype=np.float64)
    c=np.sum(k*k,axis=-1);dc=2*np.sum(k*dk,axis=-1);B=g[:,None,None]*S
    ideal=B+b[:,None,None]*k[:,:,None]*(v-np.sum(k[:,:,None]*B,axis=-2))[:,None,:]
    dB=dg[:,None,None]*S+g[:,None,None]*dS;rows=[]
    for h in range(len(b)):
        den=1-b[h]*c[h]
        if den<=0 or not np.isfinite(den):rows.append({'head':h,'eligible':False});continue
        lam=b[h]/den;dl=(db[h]+b[h]*b[h]*dc[h])/(den*den)
        df=dB[h]+dl*np.outer(k[h],v[h])+lam*np.outer(dk[h],v[h])+lam*np.outer(k[h],dv[h])
        versions={}
        for name,base in [('reconstructed_equilibrium',ideal[h]),('rounded_native',native_state[h])]:
            dKs=dl*np.outer(k[h],k[h]@base)+lam*np.outer(dk[h],k[h]@base)+lam*np.outer(k[h],dk[h]@base)
            rhs=df-dKs;solution=rhs-b[h]*np.outer(k[h],k[h]@rhs)
            error=np.linalg.norm(solution-target[h]);tn=np.linalg.norm(target[h])
            versions[name]={'error_norm':float(error),'relative':float(error/max(tn,1e-30)),
                            'pass':bool(error<=max(.001*tn,1e-6)),
                            'force_cancellation_ratio':float(np.linalg.norm(rhs)/max(np.linalg.norm(df)+np.linalg.norm(dKs),1e-30))}
        rows.append({'head':h,'eligible':True,'condition_number':float(1/den),'versions':versions})
    return rows
