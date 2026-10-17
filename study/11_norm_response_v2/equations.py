"""Four fixed response equations; no learned or fitted coefficient."""
import torch
import torch.nn.functional as tf
from maths import LocalMLP

NAMES=['linear','quadratic','rational','exact_norm_gate2']

def normalized(h,norm,dtype):
    x=h.to(dtype);gain=(1+norm.weight.float()).to(dtype)
    return gain*x*torch.rsqrt(x.square().mean()+norm.eps)

def gate2(local,h,delta,dtype=torch.float32):
    x=h.to(dtype);d=delta.to(dtype)
    z0=normalized(x,local.norm,dtype);z1=normalized(x+d,local.norm,dtype)
    a=local.mlp.gate_proj.weight.to(dtype);b=local.mlp.up_proj.weight.to(dtype);out=local.mlp.down_proj.weight.to(dtype)
    g=tf.linear(z0,a);u=tf.linear(z0,b);dg=tf.linear(z1-z0,a);du=tf.linear(z1-z0,b)
    sigmoid=torch.sigmoid(g);f=tf.silu(g)
    first=sigmoid+g*sigmoid*(1-sigmoid)
    second=sigmoid*(1-sigmoid)*(2+g*(1-2*sigmoid))
    return tf.linear((f+first*dg+.5*second*dg.square())*(u+du)-f*u,out)

def candidates(local,h,v,alpha,*,actual_delta=None):
    """Compute all predictions from base weights and prescribed input, never output."""
    x=h.float();direction=v.float()
    _,j,hh=local.derivatives(x,direction)
    nominal=(float(alpha)*direction).double()
    delta=nominal if actual_delta is None else actual_delta.double()
    if actual_delta is None:
        first=float(alpha)*j;second=float(alpha)*j+.5*float(alpha)**2*hh
    else:
        _,jd,hd=local.derivatives(x,delta)
        first=jd;second=jd+.5*hd
    # Direct positive norm avoids cancellation in s0+alpha*s1+alpha^2*s2.
    base=x.double();s0=base.square().mean()+local.norm.eps
    s1=2*(base*delta).mean();denominator=(base+delta).square().mean()+local.norm.eps
    assert torch.isfinite(denominator) and denominator>=local.norm.eps
    rational=(s0/denominator)*(second.double()+(s1/s0)*first.double())
    gate=gate2(local,x,delta)
    branches=torch.stack([first,second,rational.float(),gate])
    full=branches+delta.float()[None,:]
    return branches,full,dict(denominator=float(denominator),s0=float(s0),normalization_factor=float(s0/denominator))

def quadratic_even(local,h,v,dtype=torch.float64):
    x=h.to(dtype);v=v.to(dtype);gain=(1+local.norm.weight.float()).to(dtype)
    a=local.mlp.gate_proj.weight.to(dtype);b=local.mlp.up_proj.weight.to(dtype);d=local.mlp.down_proj.weight.to(dtype)
    s=x.square().mean()+local.norm.eps;s1=2*(x*v).mean();s2=2*v.square().mean()
    r=s.rsqrt();r1=-.5*s.pow(-1.5)*s1;r2=.75*s.pow(-2.5)*s1*s1-.5*s.pow(-1.5)*s2
    z=gain*r*x;z1=gain*(r*v+r1*x);z2=gain*(2*r1*v+r2*x)
    g=tf.linear(z,a);u=tf.linear(z,b);g1=tf.linear(z1,a);u1=tf.linear(z1,b);g2=tf.linear(z2,a);u2=tf.linear(z2,b)
    value=.5*tf.linear(g*u,d);first=.5*tf.linear(g1*u+g*u1,d)
    second=.5*tf.linear(g2*u+2*g1*u1+g*u2,d)
    return value,first,second

def q_value(local,h,dtype=torch.float64):
    z=normalized(h,local.norm,dtype)
    g=tf.linear(z,local.mlp.gate_proj.weight.to(dtype));u=tf.linear(z,local.mlp.up_proj.weight.to(dtype))
    return .5*tf.linear(g*u,local.mlp.down_proj.weight.to(dtype))

def select_candidate(calibration_metrics):
    """Only calibration questions and directions0/1 may be supplied."""
    import numpy as np
    assert all(r['split']=='calibration' and r['direction'] in [0,1] and r['precision']=='FP32' for r in calibration_metrics)
    strong=[r for r in calibration_metrics if r['strength'] in [.5,1.25,2.]]
    if any(not r['resolved'] or r['errors']['rational']['relative'] is None or r['errors']['exact_norm_gate2']['relative'] is None for r in strong):
        return dict(selected='rational',reason='Unresolved strong calibration cases; no upgraded candidate selection',scope='Calibration only')
    checks=[]
    for question in sorted(set(r['question'] for r in strong)):
        for kind in ['branch','complete']:
            a=[r for r in strong if r['question']==question and r['map']==kind]
            rational=np.median([r['errors']['rational']['relative'] for r in a])
            gate=np.median([r['errors']['exact_norm_gate2']['relative'] for r in a])
            checks.append({'question':question,'map':kind,'rational_median':float(rational),'gate2_median':float(gate),'pass':bool(gate<=rational)})
    branch=[r for r in strong if r['map']=='branch']
    ratio=float(np.median([r['errors']['exact_norm_gate2']['relative'] for r in branch])/np.median([r['errors']['rational']['relative'] for r in branch]))
    small=[r for r in calibration_metrics if r['strength'] in [.0078125,.03125] and r['resolved']]
    small_pass=all(r['errors']['exact_norm_gate2']['relative']<=.05 for r in small)
    selected='exact_norm_gate2' if all(r['pass'] for r in checks) and ratio<=.8 and small_pass else 'rational'
    return dict(selected=selected,per_question_checks=checks,strong_branch_median_ratio=ratio,small_band_all_pass=small_pass,
        scope='Locked using calibration questions/directions only; no fitted parameter or held-out selection')

def factorial_diagnostics(local,h,delta):
    """Calibration-only2x2 normalization/gate decomposition; never a selector."""
    x=h.float();d=delta.float();gain=1+local.norm.weight.float()
    s=x.square().mean()+local.norm.eps;r=s.rsqrt();r1=-(x*d).mean()*s.pow(-1.5)
    z0=normalized(x,local.norm,torch.float32)
    displacements={'linear_norm':gain*(r*d+r1*x),'exact_norm':normalized(x+d,local.norm,torch.float32)-z0}
    a=local.mlp.gate_proj.weight.float();b=local.mlp.up_proj.weight.float();out=local.mlp.down_proj.weight.float()
    g=tf.linear(z0,a);u=tf.linear(z0,b);sig=torch.sigmoid(g);f=tf.silu(g)
    f1=sig+g*sig*(1-sig);f2=sig*(1-sig)*(2+g*(1-2*sig))
    results={}
    for name,dz in displacements.items():
        dg=tf.linear(dz,a);du=tf.linear(dz,b)
        for order in [1,2]:
            estimate=f+f1*dg+(.5*f2*dg.square() if order==2 else 0)
            results[name+'_gate'+str(order)]=tf.linear(estimate*(u+du)-f*u,out)
    results['interaction_vector']=(results['exact_norm_gate2']-results['exact_norm_gate1'])-(results['linear_norm_gate2']-results['linear_norm_gate1'])
    return results
