"""Reusable ONE-run activation probe. No labels, IDs, vector, or baseline input."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

BLOCK=15
STEPS=8
WIDTH=1024

def load_window(path):
    """Read only the declared activation array; every other NPZ key is ignored."""
    with np.load(path,allow_pickle=False) as z:
        if 'activation_window' in z:
            window=z['activation_window']
        elif 'residuals' in z:
            h=z['residuals']
            if h.ndim!=3 or h.shape[0]<=BLOCK or h.shape[1]<STEPS or h.shape[2]!=WIDTH:
                raise ValueError('Expected residuals [layers>=16,positions>=8,1024]')
            window=h[BLOCK,:STEPS,:]
        else:raise ValueError('Missing activation_window or residuals array')
    return np.asarray(window,dtype=np.float64)

def feature(window):
    """One run, causal eight-position window; no test-time reference fitting."""
    x=np.asarray(window,dtype=np.float64)
    if x.shape!=(STEPS,WIDTH) or not np.isfinite(x).all():
        raise ValueError('Expected exactly eight finite width1024 activation vectors')
    norms=np.linalg.norm(x,axis=1)
    if np.any(norms<=1e-12):raise ValueError('Zero/near-zero activation vector')
    return (x/norms[:,None]).mean(axis=0)

class Detector:
    def __init__(self,center,scale,weights,threshold=0.0):
        self.center=np.asarray(center,dtype=np.float64)
        self.weights=np.asarray(weights,dtype=np.float64)
        self.scale=float(scale);self.threshold=float(threshold)
        if self.center.shape!=(WIDTH,) or self.weights.shape!=(WIDTH,):raise ValueError('Invalid probe width')
        if not np.isfinite(self.center).all() or not np.isfinite(self.weights).all() or not np.isfinite(self.scale) or self.scale<=1e-12:
            raise ValueError('Invalid probe parameters')
        if self.threshold!=0.0:raise ValueError('This frozen recipe requires threshold0')

    def predict(self,activation_window):
        f=feature(activation_window)
        margin=float(np.dot((f-self.center)/self.scale,self.weights))
        decision=int(margin>=self.threshold)
        return {'prediction':decision,'label':'known_plus_v_steering' if decision else 'unsteered',
                'raw_margin':margin,'threshold':self.threshold,'calibrated_probability':None,
                'scope':'known +v versus base, fixed model/block/causal window only'}

    @classmethod
    def load(cls,model_dir):
        folder=Path(model_dir);schema=json.loads((folder/'schema.json').read_text())
        path=folder/'probe.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=schema['checkpoint_sha256']:raise ValueError('Probe checkpoint hash mismatch')
        if schema['block_zero_based']!=BLOCK or schema['positions']!=list(range(STEPS)) or schema['width']!=WIDTH:
            raise ValueError('Unsupported input contract')
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'center','scale','weights','threshold'}:raise ValueError('Unexpected checkpoint contents')
            return cls(z['center'],z['scale'].item(),z['weights'],z['threshold'].item())

def train_ridge(features,targets,row_weights):
    """Caller supplies TRAIN rows only. Fixed scalar scaling and lambda1."""
    x=np.asarray(features,dtype=np.float64);y=np.asarray(targets,dtype=np.float64);rw=np.asarray(row_weights,dtype=np.float64)
    assert x.ndim==2 and x.shape[1]==WIDTH and len(x)==len(y)==len(rw)
    assert set(y)=={-1.,1.} and np.all(rw>0)
    center=np.average(x,axis=0,weights=rw)
    scale=float(np.sqrt(np.average(np.square(x-center).sum(axis=1),weights=rw)))
    if scale<=1e-12:raise ValueError('Degenerate TRAIN features')
    z=(x-center)/scale
    a=z*np.sqrt(rw)[:,None];target=y*np.sqrt(rw)
    dual=np.linalg.solve(a@a.T+np.eye(len(a)),target)
    weights=a.T@dual
    return Detector(center,scale,weights)

def main():
    p=argparse.ArgumentParser(description='Predict known +v steering from ONE activation trace; no baseline is required.')
    p.add_argument('--model',type=Path,required=True);p.add_argument('--trace',type=Path,required=True)
    a=p.parse_args();result=Detector.load(a.model).predict(load_window(a.trace))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
