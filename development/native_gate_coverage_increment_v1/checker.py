"""Independent stdlib hard-margin certificate; no optimizer/import from gate."""
import math
TOLERANCE=1e-8
def need(ok,code):
    if not ok:raise ValueError(code)
def dot(a,b):return math.fsum(u*v for u,v in zip(a,b,strict=True))
def verify(x,y,w,b,alpha):
    n=len(x);d=len(w)
    need(n>=2 and len(y)==len(alpha)==n and d>=1,'CERT_SHAPE')
    need(all(len(r)==d for r in x) and all(type(v) is int and v in (-1,1) for v in y) and set(y)=={-1,1},'CERT_ROWS_LABELS')
    need(all(type(v) in (int,float) and math.isfinite(v) for r in x for v in r)
        and all(type(v) in (int,float) and math.isfinite(v) for v in (*w,b,*alpha)),'CERT_FINITE')
    reconstructed=[math.fsum(alpha[i]*y[i]*x[i][j] for i in range(n)) for j in range(d)]
    margins=[y[i]*(dot(w,x[i])+b) for i in range(n)]
    primal=.5*dot(w,w);dual=math.fsum(alpha)-.5*dot(reconstructed,reconstructed)
    checks={'primal_violation':max(0.,max(1.-m for m in margins)),
        'dual_violation':max(0.,-min(alpha)),
        'intercept_stationarity':abs(math.fsum(alpha[i]*y[i] for i in range(n))),
        'weight_stationarity':max(abs(a-c) for a,c in zip(w,reconstructed)),
        'complementarity':max(abs(alpha[i]*(margins[i]-1.)) for i in range(n)),
        'absolute_duality_gap':abs(primal-dual)}
    need(all(math.isfinite(v) and v<=TOLERANCE for v in checks.values()),'KKT_UNCERTIFIED')
    return {'verified':True,'tolerance':TOLERANCE,'primal_objective':primal,'dual_objective':dual,
        'checks':checks,'minimum_margin':min(margins),'margins':margins}
