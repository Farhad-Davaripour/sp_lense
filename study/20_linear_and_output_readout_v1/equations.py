"""Independent convolution, recurrent readout and final output equations."""
import numpy as np
def sigmoid(x):return np.exp(-np.logaddexp(0.,-x))
def rms(x,w,dx=None,centered=True):
    x=np.asarray(x,dtype=np.float64);scale=1/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6);weight=(1+w) if centered else w
    y=x*scale*weight
    if dx is None:return y
    return y,(dx*scale-x*scale**3*np.mean(x*dx,axis=-1,keepdims=True))*weight
def readout(h,window,S,p,dh=None,dwindow=None,dS=None):
    if dh is None:x=rms(h,p['input_norm'])
    else:x,dx=rms(h,p['input_norm'],dh)
    projection=p['qkv']@x
    pre=np.sum(window[0].astype(np.float64)*p['conv'][:,0,:],axis=-1)
    if 'conv_bias' in p:pre+=p['conv_bias']
    sig=sigmoid(pre);mixed=pre*sig;raw=mixed[:2048].reshape(16,128)
    inv=1/np.sqrt(np.sum(raw*raw,axis=-1,keepdims=True)+1e-6)
    q=raw*inv/np.sqrt(128.)
    z=(p['z']@x).reshape(16,128);zs=sigmoid(z);gate=z*zs
    core=np.einsum('hk,hkv->hv',q,S.reshape(16,128,128).astype(np.float64))
    n=rms(core,p['gate_norm'],centered=False);y=p['out']@(n*gate).reshape(-1)
    if dh is None:return y,projection
    dprojection=p['qkv']@dx;dpre=np.sum(dwindow[0].astype(np.float64)*p['conv'][:,0,:],axis=-1)
    dmixed=(sig+pre*sig*(1-sig))*dpre;draw=dmixed[:2048].reshape(16,128)
    dq=(draw*inv-raw*inv**3*np.sum(raw*draw,axis=-1,keepdims=True))/np.sqrt(128.)
    dz=(p['z']@dx).reshape(16,128);dgate=(zs+z*zs*(1-zs))*dz
    dcore=np.einsum('hk,hkv->hv',dq,S.reshape(16,128,128))+np.einsum('hk,hkv->hv',q,dS.reshape(16,128,128))
    n,dn=rms(core,p['gate_norm'],dcore,centered=False)
    dy=p['out']@(dn*gate+n*dgate).reshape(-1)
    return y,projection,dy,dprojection
