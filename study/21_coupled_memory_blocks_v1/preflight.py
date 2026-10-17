"""Independent explicit-matrix checks of finite response and work accounting."""
import json,time
import numpy as np
from equations import apply_K,force,finite_terms,potential,energy_rows
BEGIN=time.monotonic();r=np.random.default_rng(2026101512)
def driver():
    k=r.normal(size=(16,4));k/=np.sqrt(np.sum(k*k,axis=-1,keepdims=True)+1e-6)
    return {'k':k,'v':r.normal(size=(16,4)),'gamma':r.uniform(.1,1.,16),'beta':r.uniform(.1,.9,16)}
d0=driver();da=driver();d0['gamma'][0]=1.;da['gamma'][0]=1.
old0=r.normal(size=(16,4,4));olda=old0+.1*r.normal(size=old0.shape)
K=lambda d:(1-d['gamma'])[:,None,None]*np.eye(4)+(d['gamma']*d['beta'])[:,None,None]*np.einsum('hi,hj->hij',d['k'],d['k'])
K0=K(d0);Ka=K(da);apply=lambda K,S:np.einsum('hij,hjv->hiv',K,S)
assert np.allclose(apply_K(old0,d0),apply(K0,old0),atol=1e-13,rtol=0)
new0=old0+force(d0)-apply(K0,old0);newa=olda+force(da)-apply(Ka,olda)
terms=finite_terms(old0,olda,d0,da);err=float(np.linalg.norm(sum(terms.values())-(newa-new0)));assert err<1e-12
assert all(row['pass'] for row in energy_rows(olda,newa,da))
assert np.max(potential(newa,da)-potential(olda,da))<=1e-12
print(json.dumps({'synthetic_only':True,'new_backbone_forwards':0,'explicit_matrix_vs_rank_one_pass':True,'finite_four_term_error':err,'work_and_conditional_descent_pass':True,'neutral_gamma_one_case_retained':True,'seconds':time.monotonic()-BEGIN}))
