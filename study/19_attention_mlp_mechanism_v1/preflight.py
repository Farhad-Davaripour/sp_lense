"""Synthetic finite differences for the independent analytic formulas."""
import json,time
import numpy as np
from equations import mlp,attention
BEGIN=time.monotonic();rng=np.random.default_rng(20261015)
x=rng.normal(size=1024);dx=rng.normal(size=1024);dx/=np.linalg.norm(dx)
p={'norm':rng.normal(size=1024)*.1,'gate':rng.normal(size=(31,1024))*.01,'up':rng.normal(size=(31,1024))*.01,'down':rng.normal(size=(1024,31))*.03}
eps=1e-5;y,dy=mlp(x,p,dx);fd=(mlp(x+eps*dx,p)-mlp(x-eps*dx,p))/(2*eps)
mlp_error=float(np.linalg.norm(dy-fd)/np.linalg.norm(dy));assert mlp_error<1e-6
p={'norm':p['norm'],'q':rng.normal(size=(4096,1024))*.01,'qnorm':rng.normal(size=256)*.1,'out':rng.normal(size=(1024,2048))*.01}
K=rng.normal(size=(2,7,256))*.1;V=rng.normal(size=(2,7,256))*.1;dK=rng.normal(size=K.shape)*.01;dV=rng.normal(size=V.shape)*.01
theta=rng.normal(size=32);cos=np.tile(np.cos(theta),2);sin=np.tile(np.sin(theta),2)
y,dy,r=attention(x,K,V,p,cos,sin,dx,dK,dV)
fd=(attention(x+eps*dx,K+eps*dK,V+eps*dV,p,cos,sin)[0]-attention(x-eps*dx,K-eps*dK,V-eps*dV,p,cos,sin)[0])/(2*eps)
attention_error=float(np.linalg.norm(dy-fd)/np.linalg.norm(dy));assert attention_error<1e-6
for key in ['stationarity','minimum_identity_error','objective_gap_identity_error','simplex_derivative_error','common_mode_invariance_error']:assert np.max(r[key])<1e-10
assert np.min(r['candidate_KL'])>=0
print(json.dumps({'synthetic_only':True,'new_backbone_forwards':0,'MLP_FD_relative':mlp_error,'attention_FD_relative':attention_error,'all_probability_energy_identities_pass':True,'seconds':time.monotonic()-BEGIN}))
