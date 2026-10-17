"""Independent covariance identities and higher-order saved-array checks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('C:/Users/farha/Research4/artifacts/finite_logit_response_v1/native_cpu_20261014')
OUT=Path('C:/Users/farha/Research4/artifacts/response_geometry_v1')
geom=json.loads((OUT/'evaluated_20261014/CONTEXT_GEOMETRY.json').read_text())
rows=json.loads((OUT/'cubic_diagnostic_20261014/ALL_METRICS.json').read_text())
original=json.loads((ROOT/'METRICS.json').read_text())
index={(r['question'],r['direction'],r['alpha']):r for r in original}
def load(p):
    with np.load(p,allow_pickle=False) as z:return {n:z[n].astype(np.float64) for n in z.files}
def softmax_lp(x):
    shifted=x-x.max();lp=shifted-np.log(np.exp(shifted).sum());return np.exp(lp),lp
audit={'new_model_calls':0,'geometry_contexts':0,'cubic_endpoints':0,'max_covariance_identity_error':0.,
 'max_input_Gram_error':0.,'minimum_plane_eigenvalue':1e100,'max_original_second_cost_error_difference':0.,
 'max_cubic_cost_error_difference':0.,'max_cubic_weighted_error_difference':0.,'all_source_arrays_unchanged':True}
for g in geom:
    pid=g['question'];base=load(ROOT/'baseline'/f'{pid}.npz');p,lp=softmax_lp(base['logits'])
    coeff=[load(ROOT/'coefficients'/f'{pid}_d{di}.npz') for di in range(2)]
    G=np.array(g['output_Fisher_plane']['matrix']);Q=np.stack([c['q'] for c in coeff])
    # Variance identity computed from independent directional combinations.
    for coordinates in [np.array([1.,0.]),np.array([0.,1.]),np.array([1.,1.])/np.sqrt(2),np.array([2.,-1.])/np.sqrt(5)]:
        projected=coordinates@Q;variance=p@(projected*projected)-(p@projected)**2
        audit['max_covariance_identity_error']=max(audit['max_covariance_identity_error'],abs(variance-coordinates@G@coordinates))
    audit['max_input_Gram_error']=max(audit['max_input_Gram_error'],float(np.max(np.abs(np.array(g['internal_planes']['input']['matrix'])-np.eye(2)))))
    audit['minimum_plane_eigenvalue']=min(audit['minimum_plane_eigenvalue'],float(np.linalg.eigvalsh(G).min()))
    audit['geometry_contexts']+=1
    for di in range(2):
        actual=load(ROOT/'actual'/f'{pid}_d{di}.npz');c=coeff[di]
        zero=load(ROOT/'probes'/f'{pid}_d{di}_q0.npz');plus=load(ROOT/'probes'/f'{pid}_d{di}_s0.0078125_plus.npz');minus=load(ROOT/'probes'/f'{pid}_d{di}_s0.0078125_minus.npz')
        t=(plus['q']-2*zero['q']+minus['q'])/.0078125**2
        for row in [r for r in rows if r['question']==pid and r['direction']==di]:
            a=row['alpha'];ai=list(actual['alphas']).index(a);pa,lpa=softmax_lp(actual['logits'][ai])
            measured=float(p@(lp-lpa));_,lpf=softmax_lp(base['logits']+a*c['q']+.5*a*a*c['r']+a**3*t/6)
            cost=float(p@(lp-lpf));err=abs(cost-measured)/measured
            audit['max_cubic_cost_error_difference']=max(audit['max_cubic_cost_error_difference'],abs(err-row['cost_errors']['3']))
            delta=a*c['q']+.5*a*a*c['r']+a**3*t/6
            true=actual['logits'][ai]-base['logits'];residual=delta-true
            weighted=np.sqrt((p@((residual-p@residual)**2))/(p@((true-p@true)**2)))
            audit['max_cubic_weighted_error_difference']=max(audit['max_cubic_weighted_error_difference'],abs(weighted-row['weighted_logit_errors']['3']))
            original_row=index[pid,di,a]
            audit['max_original_second_cost_error_difference']=max(audit['max_original_second_cost_error_difference'],abs(row['cost_errors']['2']-original_row['cost_relative_errors']['second_order_logits']))
            audit['cubic_endpoints']+=1
for path in [OUT/'evaluated_20261014/ANALYSIS_FREEZE.json',OUT/'cubic_diagnostic_20261014/SOURCE_IMMUTABILITY.json']:
    hashes=json.loads(path.read_text());hashes=hashes.get('source_sha256',hashes)
    for name,h in hashes.items():
        p=Path(name)
        if not p.is_absolute():p=ROOT/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==h
assert audit['minimum_plane_eigenvalue']>0 and audit['max_input_Gram_error']<1e-6
assert max(audit['max_covariance_identity_error'],audit['max_cubic_cost_error_difference'],audit['max_cubic_weighted_error_difference'],audit['max_original_second_cost_error_difference'])<1e-9
(OUT/'INDEPENDENT_AUDIT_20261014.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
