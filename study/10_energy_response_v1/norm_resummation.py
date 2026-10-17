"""Post-results algebraic hypothesis; saved data only, no model calls or fitting.

The rational formula is exact for the RMS-normalized quadratic EVEN component
of SwiGLU. Applying it to the entire SwiGLU branch is an approximation, evaluated
retrospectively here. These reused outcomes are not fresh confirmation.
"""
from pathlib import Path
import csv,json
import numpy as np
BASE=Path('C:/Users/farha/Research4/artifacts/energy_response_v1');PILOT=BASE/'pilot_20261012';OUT=BASE/'evaluated_20261012'
rows=[]
for pid in ['energy03_python','energy04_packing']:
    for position in ['prefill','decode1']:
        for di in [2,3]:
            with np.load(PILOT/'MLP_predictions'/f'{pid}_{position}_d{di}.npz',allow_pickle=False) as z:
                h=z['h'].astype(float);v=z['v'].astype(float);j=z['Jv'].astype(float);hh=z['Hvv'].astype(float)
            with np.load(PILOT/'MLP_responses'/f'{pid}_{position}_d{di}_FP32.npz',allow_pickle=False) as z:
                observed={'branch':z['branch_delta'].astype(float),'complete':z['complete_delta'].astype(float)}
            s0=float(np.mean(h*h)+1e-6);s1=float(2*np.mean(h*v));s2=float(np.mean(v*v))
            for i,eps in enumerate([.001953125,.0078125,.03125,.125,.5,2.]):
                for si,sign in enumerate([1,-1]):
                    alpha=sign*eps;factor=s0/(s0+alpha*s1+alpha*alpha*s2)
                    branch=factor*(alpha*j+alpha*alpha*(.5*hh+(s1/s0)*j))
                    for kind,pred in [('branch',branch),('complete',alpha*v+branch)]:
                        actual=observed[kind][2*i+si]
                        rows.append(dict(question=pid,position=position,direction=di,map=kind,strength=eps,sign=sign,
                            normalization_factor=factor,absolute_error=float(np.linalg.norm(actual-pred)),
                            relative_error=float(np.linalg.norm(actual-pred)/np.linalg.norm(actual)),
                            evidence='Post-results retrospective hypothesis, no new native calls; exact only for the quadratic-even subcomponent'))
with (OUT/'EXPLORATORY_NORM_RESUMMATION.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summaries=[]
for kind in ['branch','complete']:
    for eps in [.001953125,.0078125,.03125,.125,.5,2.]:
        a=[r for r in rows if r['map']==kind and r['strength']==eps]
        summaries.append(dict(map=kind,strength=eps,relative_error_min=min(r['relative_error'] for r in a),
            relative_error_median=float(np.median([r['relative_error'] for r in a])),relative_error_max=max(r['relative_error'] for r in a)))
(OUT/'EXPLORATORY_NORM_RESUMMATION.json').write_text(json.dumps({'status':'retrospective, not prospectively validated','native_calls':0,'summaries':summaries},indent=2)+'\n')
print(json.dumps([r for r in summaries if r['strength'] in [.5,2.]],indent=2))
