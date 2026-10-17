"""Four counted native calls on an old calibration context, before fresh tests."""
import json
import numpy as np
import torch
import torch.nn.functional as tf
from base_native import Rig,array,NAMES

def run_bridge(rig,root):
    with np.load(root/'BRIDGE_REFERENCE.npz',allow_pickle=False) as z:ref={n:z[n] for n in z.files}
    ids=ref['prefix_ids'].tolist();v=torch.tensor(ref['direction'],dtype=torch.float32)
    measured={};before=rig.calls
    def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
    def p(x):
        x=np.asarray(x,dtype=np.float64);x=x-x.max();y=np.exp(x);return y/y.sum()
    def KL(a,b):
        a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
        def logz(x):m=x.max();return m+np.log(np.exp(x-m).sum())
        return max(0.,float(logz(b)-logz(a)-p(a)@(b-a)))
    for alpha in [0.,.125,-.125]:
        points=Rig.execute(rig,ids,'input',v,torch.tensor(alpha))
        with torch.no_grad():logits=array(rig.model.lm_head(points[-1]))
        measured[alpha]={'logits':logits,**{n:array(x) for n,x in zip(NAMES,points)}}
    with torch.no_grad(),torch.autograd.forward_ad.dual_level():
        beta=torch.autograd.forward_ad.make_dual(torch.tensor(0.),torch.tensor(1.))
        points=Rig.execute(rig,ids,'input',v,beta)
        tangents=[torch.autograd.forward_ad.unpack_dual(x).tangent for x in points]
        q=array(tf.linear(tangents[-1],rig.model.lm_head.weight))
    errors={};gates=[]
    for n in ['logits']+NAMES:
        error=norm(measured[0.][n]-ref['base_'+n])/max(norm(ref['base_'+n]),1e-30)
        errors['baseline_'+n]=error;gates.append(error<=1e-4)
    q_error=norm(q-ref['q'])/max(norm(ref['q']),1e-30);errors['q']=q_error;gates.append(q_error<=.002)
    for a,name in [(.125,'plus'),(-.125,'minus')]:
        for n in ['logits']+NAMES:
            actual=measured[a][n].astype(np.float64)-measured[0.][n]
            original=ref[name+'_'+n].astype(np.float64)-ref['base_'+n]
            error=norm(actual-original)/max(norm(original),1e-30)
            errors[name+'_'+n+'_response']=error;gates.append(error<=.02)
        cost=KL(measured[0.]['logits'],measured[a]['logits']);oldcost=KL(ref['base_logits'],ref[name+'_logits'])
        err=abs(cost-oldcost)/max(oldcost,1e-30);errors[name+'_KL']=err;gates.append(err<=.02)
    report={'passed':all(gates),'calls':rig.calls-before,'model_calls_total':rig.calls,'errors':errors,
        'old_calibration_context_only':True,'fresh_endpoints_seen':0,'CPU_only':True}
    assert report['calls']==4
    (root/'PLATFORM_BRIDGE.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(root/'PLATFORM_BRIDGE_MEASURED.npz',q=q,
        **{('base' if a==0 else 'plus' if a>0 else 'minus')+'_'+n:x for a,values in measured.items() for n,x in values.items()})
    assert report['passed'],'Cross-platform bridge failed; stop before fresh held tests'
    return report
