"""Directional RMSNorm/SwiGLU derivatives and finite-response diagnostics."""
import math
import numpy as np
import torch
import torch.nn.functional as tf

class LocalMLP:
    def __init__(self,norm,mlp):
        self.norm=norm;self.mlp=mlp

    def branch(self,h):return self.mlp(self.norm(h))

    def formula(self,h,dtype=torch.float64):
        x=h.to(dtype);gain=(1+self.norm.weight.float()).to(dtype)
        z=gain*x*torch.rsqrt(x.square().mean()+self.norm.eps)
        g=tf.linear(z,self.mlp.gate_proj.weight.to(dtype));u=tf.linear(z,self.mlp.up_proj.weight.to(dtype))
        return tf.linear(tf.silu(g)*u,self.mlp.down_proj.weight.to(dtype))

    def derivatives(self,h,v,dtype=torch.float32):
        x=h.to(dtype);v=v.to(dtype);gain=(1+self.norm.weight.float()).to(dtype)
        a=self.mlp.gate_proj.weight.to(dtype);b=self.mlp.up_proj.weight.to(dtype);d=self.mlp.down_proj.weight.to(dtype)
        s=x.square().mean()+self.norm.eps;s1=2*(x*v).mean();s2=2*v.square().mean()
        r=s.rsqrt();r1=-.5*s.pow(-1.5)*s1;r2=.75*s.pow(-2.5)*s1.square()-.5*s.pow(-1.5)*s2
        z=gain*r*x;z1=gain*(r*v+r1*x);z2=gain*(2*r1*v+r2*x)
        g=tf.linear(z,a);u=tf.linear(z,b);g1=tf.linear(z1,a);u1=tf.linear(z1,b)
        g2=tf.linear(z2,a);u2=tf.linear(z2,b)
        sigmoid=torch.sigmoid(g);f=tf.silu(g)
        f1=sigmoid+g*sigmoid*(1-sigmoid)
        f2=sigmoid*(1-sigmoid)*(2+g*(1-2*sigmoid))
        first=tf.linear(f1*g1*u+f*u1,d)
        second=tf.linear((f2*g1.square()+f1*g2)*u+2*f1*g1*u1+f*u2,d)
        value=tf.linear(f*u,d)
        return value,first,second

def array(t):
    value=t.detach().cpu()
    return (value.float() if value.dtype==torch.bfloat16 else value).numpy()
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def cosine(a,b):
    denominator=norm(a)*norm(b)
    return float(np.clip(np.sum(np.asarray(a,dtype=np.float64)*np.asarray(b,dtype=np.float64))/denominator,-1,1)) if denominator>1e-20 else None
def relative(error,reference):
    return norm(error)/norm(reference) if norm(reference)>1e-20 else None

def parity(plus,minus):
    p=np.asarray(plus,dtype=np.float64);m=np.asarray(minus,dtype=np.float64)
    odd=(p-m)/2;even=(p+m)/2;oe=float(np.sum(odd*odd));ee=float(np.sum(even*even))
    c=cosine(p,m)
    return dict(response_cosine=c,response_angle_degrees=math.degrees(math.acos(c)) if c is not None else None,
        opposition_cosine=-c if c is not None else None,odd_norm=math.sqrt(oe),even_norm=math.sqrt(ee),
        even_energy_fraction=ee/(ee+oe) if ee+oe>1e-20 else None,
        dot_identity_absolute_error=abs(float(np.sum(p*m))-(ee-oe)))

def geometry(q,jq):
    q=np.asarray(q,dtype=np.float64);jq=np.asarray(jq,dtype=np.float64)
    k=q.T@jq;u,s,vt=np.linalg.svd(k);rotation=u@vt
    residual=jq-q@k;den=float(np.sum(jq*jq));kn=norm(k)
    eigen=np.linalg.eigvals(k)
    return dict(sampled_input_rank=q.shape[1],ambient_dimension=q.shape[0],input_Gram_error=norm(q.T@q-np.eye(q.shape[1])),
        full_output_sampled_input_singular_gains=np.linalg.svd(jq,compute_uv=False).tolist(),
        projected_singular_values=s.tolist(),projected_eigenvalues=[[float(v.real),float(v.imag)] for v in eigen],
        projected_polar_factor=rotation.tolist(),projected_polar_determinant=float(np.linalg.det(rotation)),
        projected_asymmetry=norm(k-k.T)/kn if kn>0 else None,
        projected_nonnormal_commutator=norm(k.T@k-k@k.T)/(kn*kn) if kn>0 else None,
        output_projection_leakage_energy_fraction=float(np.sum(residual*residual)/den) if den>0 else None,
        scope='Four sampled input directions only; projected polar/eigen metrics omit reported output leakage and do not identify a full-space rotation or multi-step transient growth')

def kl_from_logits(base,changed):
    b=torch.as_tensor(base,dtype=torch.float64);c=torch.as_tensor(changed,dtype=torch.float64)
    lb=b-torch.logsumexp(b,dim=-1);lc=c-torch.logsumexp(c,dim=-1);p=lb.exp()
    return float(torch.sum(p*(lb-lc)))

def fisher_direction(base_logits,q):
    logits=torch.as_tensor(base_logits,dtype=torch.float64);q=torch.as_tensor(q,dtype=torch.float64)
    p=torch.softmax(logits,dim=-1);center=q-torch.sum(p*q)
    return float(torch.sum(p*center.square()))

def weighted_derivative_error(base_logits,predicted,measured):
    p=torch.softmax(torch.as_tensor(base_logits,dtype=torch.float64),dim=-1)
    a=torch.as_tensor(predicted,dtype=torch.float64);b=torch.as_tensor(measured,dtype=torch.float64)
    a=a-torch.sum(p*a);b=b-torch.sum(p*b)
    error=float(torch.sqrt(torch.sum(p*(a-b).square())))
    size=float(torch.sqrt(torch.sum(p*a.square())))
    return dict(weighted_centered_RMS_absolute_error=error,weighted_centered_derivative_size=size,
        relative_error=error/size if size>1e-8 else None,resolved=size>1e-8)

def synthetic_controls():
    a=.0028;p=np.array([1.,a]);m=np.array([-1.,a]);j=np.diag([1.,1/a])
    opposite=np.array([1.,0.]);d=j@opposite
    assert abs(cosine(d,-d)+1)<1e-12 and abs(cosine(j@p,j@m))<1e-12
    return dict(exact_opposite_linear_response_cosine=cosine(d,-d),nearly_opposite_input_cosine=cosine(p,m),
        anisotropic_linear_output_cosine=cosine(j@p,j@m),condition_number=float(np.linalg.cond(j)),
        note='Near-opposition alone does not exclude linear amplification of a tiny asymmetric input; this is a known synthetic example, not a fitted explanation of the model')
