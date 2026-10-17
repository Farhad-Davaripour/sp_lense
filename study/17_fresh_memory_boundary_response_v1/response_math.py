"""Local response arithmetic and prospective numerical gates; no fitted model."""
import numpy as np
from maths import GROUPS,norm,wn,kl

def group_norm(values,schema,group):
    return float(np.sqrt(sum(norm(values[n])**2 for n,s in schema.items() if s['group']==group)))

def flat_value(point):return {**point['values'],'logits':point['logits']}
def flat_slope(point):return {**point['first'],'logits':point['q']}
def linear_combination(*terms):
    keys=set(terms[0][1]);assert all(set(x)==keys for _,x in terms)
    return {n:sum(c*np.asarray(x[n],dtype=np.float64) for c,x in terms) for n in terms[0][1]}

def derivatives(zero,plus,minus,step):
    vplus,vminus=flat_value(plus),flat_value(minus)
    qplus,qminus,qzero=flat_slope(plus),flat_slope(minus),flat_slope(zero)
    return {'q_fd':linear_combination((1/(2*step),vplus),(-1/(2*step),vminus)),
            'r':linear_combination((1/(2*step),qplus),(-1/(2*step),qminus)),
            't':linear_combination((1/(step*step),qplus),(1/(step*step),qminus),(-2/(step*step),qzero))}

def group_check(candidate,reference,schema,tolerance):
    delta=linear_combination((1,candidate),(-1,reference));result={}
    for g in GROUPS:
        den=group_norm(reference,schema,g);error=group_norm(delta,schema,g)
        result[g]={'relative':error/max(den,1e-30),'norm':den,'difference':error,
                   'pass':bool(error<=max(tolerance*den,1e-6))}
    return result

def first_second_validation(base,q,coarse,fine,schema):
    qchecks=group_check(q,fine['q_fd'],schema,.03)
    rchecks=group_check(coarse['r'],fine['r'],schema,.05)
    den=wn(base['logits'],fine['q_fd']['logits'])
    error=wn(base['logits'],q['logits']-fine['q_fd']['logits'])
    weighted={'relative':error/max(den,1e-30),'pass':bool(error<=.02*max(den,1e-30))}
    return {'groups_q':qchecks,'groups_r':rchecks,'weighted_q':weighted,
            'pass':bool(weighted['pass'] and all(x['pass'] for x in qchecks.values()) and all(x['pass'] for x in rchecks.values()))}

def third_validation(base,q,r,t_coarse,t_fine,schema,alphas=(.03125,-.03125,.125,-.125)):
    """Relative identification OR tiny bounded forecast contribution uncertainty.

    The second option validates the tested contribution, not an accurately
    identified third derivative. Each alpha must pass independently.
    """
    rows={};eps=np.finfo(np.float32).eps
    difference=linear_combination((1,t_coarse),(-1,t_fine))
    for g in [*GROUPS,'weighted_logits']:
        weighted=g=='weighted_logits'
        def length(x):return wn(base['logits'],x['logits']) if weighted else group_norm(x,schema,g)
        den=length(t_fine);error=length(difference);relative=error/max(den,1e-30)
        identified=bool(error<=max(.10*den,1e-6));contributions=[]
        for a in alphas:
            quadratic=linear_combination((a,q),(.5*a*a,r))
            cubic_fine=linear_combination((1,quadratic),(a**3/6,t_fine))
            cubic_coarse=linear_combination((1,quadratic),(a**3/6,t_coarse))
            uncertainty=abs(a**3)*error/6
            bound=max(.005*min(length(quadratic),length(cubic_fine),length(cubic_coarse)),20*eps*length(base))
            contributions.append({'alpha':a,'uncertainty_norm':uncertainty,'bound':bound,'pass':bool(uncertainty<=bound)})
        contribution_resolved=all(x['pass'] for x in contributions)
        rows[g]={'relative_third_disagreement':relative,'identified':identified,
                 'contribution_resolved':contribution_resolved,'contributions':contributions,
                 'pass':bool(identified or contribution_resolved)}
    return {'groups':rows,'pass':bool(all(x['pass'] for x in rows.values())),
            'all_third_derivatives_identified':bool(all(x['identified'] for x in rows.values()))}

def plane_tensor(jet0,jet1,sum_jet,difference_jet):
    H01=linear_combination((1,sum_jet['r']),(-.5,jet0['r']),(-.5,jet1['r']))
    A=linear_combination((2*np.sqrt(2),sum_jet['t']),(-1,jet0['t']),(-1,jet1['t']))
    B=linear_combination((2*np.sqrt(2),difference_jet['t']),(-1,jet0['t']),(1,jet1['t']))
    return {'H01':H01,'T001':linear_combination((1/6,A),(-1/6,B)),
            'T011':linear_combination((1/6,A),(1/6,B))}

def combine_plane(jet0,jet1,plane,a,b,diagonal=False):
    q=linear_combination((a,jet0['q']),(b,jet1['q']))
    r=linear_combination((a*a,jet0['r']),(b*b,jet1['r']))
    t=linear_combination((a**3,jet0['t']),(b**3,jet1['t']))
    if not diagonal:
        r=linear_combination((1,r),(2*a*b,plane['H01']))
        t=linear_combination((1,t),(3*a*a*b,plane['T001']),(3*a*b*b,plane['T011']))
    return {'q':q,'r':r,'t':t}

def forecast(base,jet,alpha,order):
    assert order in [1,2,3]
    terms=[(alpha,jet['q'])]
    if order>=2:terms.append((alpha*alpha/2,jet['r']))
    if order>=3:terms.append((alpha**3/6,jet['t']))
    change=linear_combination(*terms)
    return linear_combination((1,base),(1,change)),change

def score(base,actual,jet,alpha,schema):
    true=linear_combination((1,actual),(-1,base));cost=kl(base['logits'],actual['logits']);rows={}
    current=[n for n in schema if n.startswith('m') and n.endswith('_current')]
    for order in [1,2,3]:
        predicted,delta=forecast(base,jet,alpha,order);error=linear_combination((1,delta),(-1,true));groups={};fields={}
        for g in GROUPS:
            den=group_norm(true,schema,g);en=group_norm(error,schema,g)
            groups[g]={'relative_error':en/den if den>1e-12 else None,'change_norm':den,'error_norm':en}
        for n in schema:
            den=norm(true[n]);en=norm(error[n]);resolved=den>max(1e-10,20*np.finfo(np.float32).eps*norm(base[n]))
            fields[n]={'relative_error':en/den if resolved else None,'resolved':bool(resolved),
                       'change_norm':den,'error_norm':en,'exactly_unchanged':bool(np.array_equal(base[n],actual[n]))}
        den=np.sqrt(sum(norm(true[n])**2 for n in current));en=np.sqrt(sum(norm(error[n])**2 for n in current))
        predicted_cost=kl(base['logits'],predicted['logits'])
        row={'predicted_KL':predicted_cost,'cost_error':abs(predicted_cost-cost)/cost if cost>1e-10 else None,
             'weighted_logit_error':wn(base['logits'],error['logits'])/max(wn(base['logits'],true['logits']),1e-30),
             'actual_to_prediction_KL':kl(actual['logits'],predicted['logits']),'groups':groups,'fields':fields,
             'current_KV_only_error':float(en/den) if den>1e-12 else None}
        row['endpoint_bounds_pass']=bool(row['cost_error'] is not None and row['cost_error']<=.20 and row['weighted_logit_error']<=.15
            and all(x['relative_error'] is None or x['relative_error']<=.10 for x in groups.values())
            and all(not x['resolved'] or x['relative_error']<=.15 for x in fields.values())
            and (row['current_KV_only_error'] is None or row['current_KV_only_error']<=.10))
        rows[str(order)]=row
    fisher=.5*alpha*alpha*wn(base['logits'],jet['q']['logits'])**2
    return {'alpha':alpha,'actual_KL':cost,'resolved':bool(cost>1e-10),'orders':rows,
            'Fisher_cost':fisher,'Fisher_cost_error':abs(fisher-cost)/cost if cost>1e-10 else None}
