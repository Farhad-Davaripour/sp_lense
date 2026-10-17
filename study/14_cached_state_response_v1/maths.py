import numpy as np

GROUPS=['hidden','residual','mixer','mlp','memory_kv','memory_conv','memory_recurrent']
def probability(x):
    x=np.asarray(x,dtype=np.float64);m=x.max();lp=x-(m+np.log(np.exp(x-m).sum()));return np.exp(lp),lp
def kl(a,b):
    p,lp=probability(a);_,lq=probability(b);v=float(p@(lp-lq));assert v>=-1e-10;return max(0.,v)
def wn(logits,x):
    p,_=probability(logits);x=np.asarray(x,dtype=np.float64);x=x-p@x;return float(np.sqrt(max(0.,p@(x*x))))
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def group_norm(values,schema,group):return float(np.sqrt(sum(norm(x)**2 for n,x in values.items() if schema[n]['group']==group)))
def difference(a,b):return {n:(a[n].astype(np.float64)-b[n]).astype(np.float32) for n in a}
def subtract(a,b):return {n:a[n].astype(np.float64)-b[n] for n in a}
def central(plus,minus,base,step):
    first={n:((plus['values'][n].astype(np.float64)-minus['values'][n])/(2*step)).astype(np.float32) for n in base['values']}
    second={n:((plus['values'][n].astype(np.float64)+minus['values'][n]-2*base['values'][n])/(step*step)).astype(np.float32) for n in base['values']}
    slope_second=None
    if 'first' in plus and 'first' in minus:
        slope_second={n:((plus['first'][n].astype(np.float64)-minus['first'][n])/(2*step)).astype(np.float32) for n in base['values']}
    return {'value_first':first,'value_second':second,'slope_second':slope_second,
        'q_fd':(plus['logits'].astype(np.float64)-minus['logits'])/(2*step)}
def checks(a,b,schema,tolerance):
    diff=subtract(a,b);rows={}
    for group in GROUPS:
        den=group_norm(b,schema,group);error=group_norm(diff,schema,group)
        rows[group]={'norm':den,'difference':error,'relative':error/max(den,1e-30),'pass':error<=max(tolerance*den,1e-6)}
    return rows
def evaluate(values,base,first,second,alpha,schema):
    truth=subtract(values,base);errors={};field_rows={}
    for order in [1,2]:
        prediction={n:alpha*first[n].astype(np.float64)+(0.5*alpha*alpha*second[n] if order==2 else 0.) for n in first}
        error={n:prediction[n]-truth[n] for n in first};groups={}
        for group in GROUPS:
            den=group_norm(truth,schema,group);en=group_norm(error,schema,group)
            groups[group]={'actual_change_norm':den,'error_norm':en,'relative_error':en/den if den>1e-12 else None}
        errors[str(order)]=groups
        if order==2:
            for n in first:
                tn=norm(truth[n]);floor=max(1e-10,20*np.finfo(np.float32).eps*norm(base[n]));resolved=bool(tn>floor)
                field_rows[n]={'group':schema[n]['group'],'layer':schema[n]['layer'],'change_norm':tn,
                    'error_norm':norm(error[n]),'resolved':resolved,'relative_error':norm(error[n])/tn if resolved else None,
                    'exactly_unchanged':bool(np.array_equal(values[n],base[n]))}
    return errors,field_rows
