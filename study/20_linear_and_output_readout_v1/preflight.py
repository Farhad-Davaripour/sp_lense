"""Finite-difference check of the complete recurrent-memory readout."""
import json,time,numpy as np
from equations import readout
BEGIN=time.monotonic();r=np.random.default_rng(20261015)
h=r.normal(size=1024);dh=r.normal(size=1024)*.01;w=r.normal(size=(1,6144,4));dw=r.normal(size=w.shape)*.01
S=r.normal(size=(16,128,128))*.1;dS=r.normal(size=S.shape)*.001
p={'input_norm':r.normal(size=1024)*.1,'conv':r.normal(size=(6144,1,4))*.2,'qkv':r.normal(size=(6144,1024))*.01,'z':r.normal(size=(2048,1024))*.01,'gate_norm':r.normal(size=128)*.1+1,'out':r.normal(size=(1024,2048))*.01}
y,proj,dy,dproj=readout(h,w,S,p,dh,dw,dS);e=1e-5
yp,pp=readout(h+e*dh,w+e*dw,S+e*dS,p);ym,pm=readout(h-e*dh,w-e*dw,S-e*dS,p)
err=float(np.linalg.norm((yp-ym)/(2*e)-dy)/np.linalg.norm(dy));pe=float(np.linalg.norm((pp-pm)/(2*e)-dproj)/np.linalg.norm(dproj));assert max(err,pe)<1e-6
print(json.dumps({'synthetic_only':True,'new_backbone_forwards':0,'readout_FD_relative':err,'projection_FD_relative':pe,'seconds':time.monotonic()-BEGIN}))
