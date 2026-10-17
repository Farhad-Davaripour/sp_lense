"""Algebraic angle predictions from coefficients frozen before perturbations.

This is a post-collection derived metric, retaining every declared strength.
No refit, native call, favorable band selection or crossover success claim.
"""
from pathlib import Path
import csv,json,math
import numpy as np
BASE=Path('C:/Users/farha/Research4/artifacts/energy_response_v1')
PILOT=BASE/'pilot_20261012';OUT=BASE/'evaluated_20261012'
def norm(x):return float(np.linalg.norm(x))
def angle(a,b):
    den=norm(a)*norm(b)
    return math.degrees(math.acos(float(np.clip(np.sum(a*b)/den,-1,1)))) if den>1e-20 else None
rows=[]
for pid in ['energy03_python','energy04_packing']:
    for position in ['prefill','decode1']:
        for di in [2,3]:
            with np.load(PILOT/'MLP_predictions'/f'{pid}_{position}_d{di}.npz',allow_pickle=False) as z:
                v=z['v'].astype(float);j=z['Jv'].astype(float);h=z['Hvv'].astype(float)
            with np.load(PILOT/'MLP_responses'/f'{pid}_{position}_d{di}_FP32.npz',allow_pickle=False) as z:
                observed={'branch':z['branch_delta'].astype(float),'complete':z['complete_delta'].astype(float)}
            for kind,jac in [('branch',j),('complete',v+j)]:
                crossover=2*norm(jac)/norm(h) if norm(h)>1e-20 else None
                for i,eps in enumerate([.001953125,.0078125,.03125,.125,.5,2.]):
                    p=eps*jac+.5*eps*eps*h;m=-eps*jac+.5*eps*eps*h
                    dp=observed[kind][2*i];dm=observed[kind][2*i+1]
                    predicted=angle(p,m);measured=angle(dp,dm)
                    remainders=[norm(dp-p)/norm(dp),norm(dm-m)/norm(dm)]
                    rows.append(dict(question=pid,position=position,direction=di,map=kind,strength=eps,
                        first_order_angle_degrees=180.,second_order_angle_degrees=predicted,measured_angle_degrees=measured,
                        second_order_angle_absolute_error=abs(predicted-measured) if predicted is not None and measured is not None else None,
                        maximum_second_order_vector_relative_error=max(remainders),
                        formal_quadratic_crossover_strength=crossover,
                        scope='Derived from premeasurement frozen Jv/Hvv; formal crossover is extrapolation unless the vector remainder is small'))
with (OUT/'DERIVED_ANGLE_PREDICTIONS.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for kind in ['branch','complete']:
    for eps in [.001953125,.0078125,.03125,.125,.5,2.]:
        a=[r for r in rows if r['map']==kind and r['strength']==eps]
        summary.append(dict(map=kind,strength=eps,case_count=len(a),
            measured_angle_range=[min(r['measured_angle_degrees'] for r in a),max(r['measured_angle_degrees'] for r in a)],
            max_second_order_angle_error=max(r['second_order_angle_absolute_error'] for r in a),
            max_second_order_vector_relative_error=max(r['maximum_second_order_vector_relative_error'] for r in a),
            formal_crossover_range=[min(r['formal_quadratic_crossover_strength'] for r in a),max(r['formal_quadratic_crossover_strength'] for r in a)]))
(OUT/'DERIVED_ANGLE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps([r for r in summary if r['map']=='branch'],indent=2))
