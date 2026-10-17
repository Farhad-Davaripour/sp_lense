"""Self-contained FP64 equations for a coupled pair; no torch/native I/O."""
import numpy as np

def sigmoid(x):return np.exp(-np.logaddexp(0.,-np.asarray(x,dtype=np.float64)))
def rms(x,w,centered=True):
    x=np.asarray(x,dtype=np.float64)
    return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*((1+w) if centered else w)
def apply_K(S,d):
    k=d['k'];g=d['gamma'];b=d['beta'];r=np.einsum('hk,hkv->hv',k,S)
    return (1-g)[:,None,None]*S+(g*b)[:,None,None]*np.einsum('hk,hv->hkv',k,r)
def force(d):return d['beta'][:,None,None]*np.einsum('hk,hv->hkv',d['k'],d['v'])
def potential(S,d):return .5*np.sum(S*apply_K(S,d),axis=(-2,-1))-np.sum(force(d)*S,axis=(-2,-1))

def block(h,state,p,normal_driver=None,rule='native'):
    h=np.asarray(h,dtype=np.float64);oldS=np.asarray(state['S'],dtype=np.float64)
    z=rms(h,p['input_norm']);projected=p['qkv']@z
    window=np.concatenate((np.asarray(state['window'],dtype=np.float64)[...,1:],projected.reshape(1,6144,1)),axis=-1)
    pre=np.sum(window[0]*p['conv'][:,0,:],axis=-1)
    if 'conv_bias' in p:pre+=p['conv_bias']
    mixed=pre*sigmoid(pre)
    query=mixed[:2048].reshape(16,128);key=mixed[2048:4096].reshape(16,128);value=mixed[4096:].reshape(16,128)
    query=query/np.sqrt(np.sum(query*query,axis=-1,keepdims=True)+1e-6)/np.sqrt(128.)
    key=key/np.sqrt(np.sum(key*key,axis=-1,keepdims=True)+1e-6)
    beta=sigmoid(p['b']@z);gamma=np.exp(-np.exp(p['A_log'])*np.logaddexp(0.,p['a']@z+p['dt_bias']))
    driver={'k':key,'v':value,'beta':beta,'gamma':gamma}
    if rule=='native':
        B=gamma[:,None,None]*oldS;e=value-np.einsum('hk,hkv->hv',key,B)
        S=B+beta[:,None,None]*np.einsum('hk,hv->hkv',key,e)
    elif rule in ['frozenK','frozenKF']:
        assert normal_driver is not None
        S=oldS-apply_K(oldS,normal_driver)+force(driver if rule=='frozenK' else normal_driver)
    else:raise ValueError(rule)
    core=np.einsum('hk,hkv->hv',query,S);gate=(p['z']@z).reshape(16,128)
    mixer=p['out']@(rms(core,p['gate_norm'],False)*gate*sigmoid(gate)).reshape(-1)
    middle=h+mixer;u=rms(middle,p['post_norm']);a=p['mlp_gate']@u;up=p['mlp_up']@u
    mlp=p['mlp_down']@(a*sigmoid(a)*up);output=middle+mlp
    fields={'input':h.copy(),'mixer':mixer,'middle':middle,'mlp':mlp,'output':output,'window':window,'S':S}
    return output,{'window':window,'S':S},driver,fields

def trajectory(boundaries,initial,parameters,direction=None,alpha=0.,rule='native',normal=None):
    state={i:{n:np.array(a,dtype=np.float64,copy=True) for n,a in initial[i].items()} for i in [12,13]}
    result=[]
    for ti in range(2):
        if ti==1 and rule=='reset':state={i:{n:np.array(a,copy=True) for n,a in normal[0]['state'][i].items()} for i in [12,13]}
        h=np.asarray(boundaries[ti],dtype=np.float32).copy()
        if ti==0 and direction is not None:h=h+np.float32(alpha)*np.asarray(direction,dtype=np.float32)
        h=h.astype(np.float64);fields={};drivers={};outstate={}
        for i in [12,13]:
            local_rule=rule if i==13 and rule in ['frozenK','frozenKF'] else 'native'
            nd=normal[ti]['drivers'][i] if local_rule!='native' else None
            h,new,driver,fs=block(h,state[i],parameters[i],nd,local_rule)
            fields.update({f'b{i}_{n}':v for n,v in fs.items()});drivers[i]=driver;outstate[i]=new
        state=outstate
        result.append({'fields':fields,'state':outstate,'drivers':drivers})
    return result

def finite_terms(old0,olda,d0,da):
    ds=olda-old0;df=force(da)-force(d0)
    return {'history':ds-apply_K(ds,d0),'drive':df,'stiffness':-(apply_K(old0,da)-apply_K(old0,d0)),
            'interaction':-(apply_K(ds,da)-apply_K(ds,d0))}

def energy_rows(old,new,d):
    step=new-old;load=force(d);internal=apply_K(old,d);residual=step-load+internal
    n=lambda a:np.sqrt(np.sum(a*a,axis=(-2,-1)))
    W=np.sum(load*step,axis=(-2,-1));dU=np.sum(internal*step,axis=(-2,-1))+.5*np.sum(step*apply_K(step,d),axis=(-2,-1))
    D=np.sum(step*step,axis=(-2,-1))-.5*np.sum(step*apply_K(step,d),axis=(-2,-1))
    error=abs(W-dU-D);bound=n(residual)*n(step)+100*np.finfo(np.float64).eps*np.maximum.reduce([np.ones(16),abs(W),abs(dU),abs(D)])
    den=np.maximum.reduce([np.ones(16),n(load),n(internal)]);fr=n(residual)/den
    return [{'head':j,'force_relative':float(fr[j]),'work':float(W[j]),'stored_change':float(dU[j]),'dissipation':float(D[j]),
             'work_error':float(error[j]),'work_bound':float(bound[j]),'potential_change':float(potential(new,d)[j]-potential(old,d)[j]),
             'gamma':float(d['gamma'][j]),'key_norm_squared':float(d['k'][j]@d['k'][j]),
             'pass':bool(fr[j]<=1e-5 and error[j]<=bound[j]+1e-10 and D[j]>=-1e-10)} for j in range(16)]
