"""Supplementary geometry checks from frozen cached-response readouts."""
import json
import numpy as np

def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def vectors(path,names):
    with np.load(path,allow_pickle=False) as z:return {n:z[n].astype(np.float64) for n in names}

def potential_and_identities(root,pid,coeff,selection):
    """Return numerical evidence; no new model calls and no primary-rule changes."""
    identities=[];potential=[]
    validated=all(json.loads((root/'coefficients'/f'{pid}_d{di}.json').read_text())['validated'] for di in range(2))
    for di,c in enumerate(coeff):
        for layer in range(24):
            for order in ['first','second']:
                for start,branch,end in [('input','mixer','middle'),('middle','mlp','output')]:
                    lhs=c[f'b{layer}_{end}_{order}'].astype(np.float64)
                    rhs=c[f'b{layer}_{start}_{order}'].astype(np.float64)+c[f'b{layer}_{branch}_{order}']
                    scale=max(norm(lhs),norm(rhs));error=norm(lhs-rhs)
                    identities.append({'question':pid,'direction':di,'layer':layer,'order':order,
                        'identity':f'{start}+{branch}={end}','norm':scale,'absolute_discrepancy':error,
                        'relative_discrepancy':error/scale if scale>1e-12 else None,'zero_response':scale<=1e-12})
    names=[f'b{i}_{n}_value' for i in range(11,24) for n in ['input','middle','output']]
    probes={}
    for step in [selection['coarse'],selection['fine']]:
        for di in range(2):
            for sign in ['plus','minus']:
                probes[step,di,sign]=vectors(root/'probes'/f'{pid}_d{di}_s{step}_{sign}.npz',names)
    for layer in range(11,24):
        for submap,start,end in [('attention_plus_norm','input','middle'),('whole_block_update','input','output'),('MLP_plus_norm','middle','output')]:
            u,v=[c[f'b{layer}_{start}_first'].astype(np.float64) for c in coeff]
            Ju,Jv=[c[f'b{layer}_{end}_first'].astype(np.float64)-c[f'b{layer}_{start}_first'] for c in coeff]
            uv=float(u@v);volume2=max(0.,norm(u)**2*norm(v)**2-uv*uv)
            informative=norm(u)>1e-12 and norm(v)>1e-12 and volume2>1e-24
            skew=float(u@Jv-v@Ju);checks=[]
            for step in [selection['coarse'],selection['fine']]:
                fd=[]
                for di in range(2):
                    p=probes[step,di,'plus'];m=probes[step,di,'minus']
                    fd.append(((p[f'b{layer}_{end}_value']-m[f'b{layer}_{end}_value'])-(p[f'b{layer}_{start}_value']-m[f'b{layer}_{start}_value']))/(2*step))
                fdskew=float(u@fd[1]-v@fd[0]);discrepancy=norm(u)*norm(Jv-fd[1])+norm(v)*norm(Ju-fd[0])
                checks.append({'step':step,'FD_skew':fdskew,'coefficient_vs_FD_discrepancy_bound':discrepancy})
            bound=max(c['coefficient_vs_FD_discrepancy_bound'] for c in checks)
            stable=all(c['FD_skew']*skew>0 for c in checks)
            potential.append({'question':pid,'layer':layer,'submap':submap,'informative':informative,
               'coefficient_skew':skew,'finite_difference_checks':checks,
               'coefficient_to_discrepancy_ratio':abs(skew)/max(bound,1e-30),'stable_sign':stable,
               'directions_numerically_validated':validated,'derivative_methods':selection['modes'],
               'evidence_against_simple_Euclidean_gradient_form':informative and validated and stable and abs(skew)>5*bound,
               'not_a_rigorous_interval_error_bound':True})
    return identities,potential
