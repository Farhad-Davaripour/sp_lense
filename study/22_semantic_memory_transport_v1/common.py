"""Small deterministic I/O and comparison helpers; no native model imports."""
import json,hashlib,os
from pathlib import Path
import numpy as np
FIELDS=[f'b{i}_{n}' for i in [12,13] for n in ['input','mixer','middle','mlp','output','window','S']]
def scalar(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def dump(p,x):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,default=scalar)+'\n',encoding='utf-8');os.replace(tmp,p)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
def save(p,a):np.savez_compressed(p,**a)
def norm(a):return float(np.linalg.norm(np.asarray(a,dtype=np.float64)))
def evaluate(meta,p,n,p0=None,n0=None):
    rows=[]
    for name in FIELDS:
        assert p[name].shape==n[name].shape
        error=norm(p[name]-n[name]);den=norm(n[name])
        r={**meta,'field':name,'value_error':error,'native_norm':den,'value_pass':error<=1e-5*max(1.,den)}
        if p0 is not None:
            truth=n[name].astype(np.float64)-n0[name].astype(np.float64);prediction=p[name]-p0[name]
            size=norm(truth);floor=float(20*np.finfo(np.float32).eps)*max(1.,den,norm(n0[name]));err=norm(prediction-truth);bound=max(.01*size,floor)
            r.update(native_change_norm=size,predicted_change_norm=norm(prediction),change_error=err,numeric_floor=floor,change_bound=bound,relative_change_error=err/max(size,floor),change_pass=err<=bound,change_resolved=size>5*floor)
        rows.append(r)
    return rows
def lse(x):
    x=np.asarray(x,dtype=np.float64);m=float(x.max());return m+float(np.log(np.exp(x-m).sum()))
def word_scores(logits,ids):
    a=np.asarray(logits,dtype=np.float64)[ids]
    return {'number_log_odds':lse(a[[1,3]])-lse(a[[0,2]]),'number_lower':float(a[1]-a[0]),'number_upper':float(a[3]-a[2]),'case_singular':float(a[2]-a[0]),'case_plural':float(a[3]-a[1]),'four_word_probability_mass':float(np.exp(lse(a)-lse(logits)))}
