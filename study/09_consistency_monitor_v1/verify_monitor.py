"""Meaningful noninterference, in-place-mutation and cleanup checks."""
import json
import torch
from monitor import BoundaryMonitor

def verify():
    torch.manual_seed(20261012)
    model=torch.nn.Sequential(torch.nn.Linear(8,8),torch.nn.Tanh(),torch.nn.Linear(8,8)).eval()
    model.requires_grad_(False);x=torch.randn(2,3,8)
    boundaries={str(i):m for i,m in enumerate(model)}
    with torch.inference_mode():reference=model(x).clone()
    with torch.inference_mode(),BoundaryMonitor(boundaries) as ordinary:observed=model(x)
    assert torch.equal(reference,observed) and not ordinary.summary()['computation_mismatch']
    def inplace(module,args,output):output.add_(.25)
    with torch.inference_mode(),BoundaryMonitor(boundaries,interventions={'1':[inplace]},keep_last_vectors=True) as changed:
        model(x)
    s=changed.summary();assert s['mismatch_boundaries']==['1']
    row=next(r for r in changed.records if r['boundary']=='1')
    assert abs(row['energy_sum']-.5*.25**2*x.numel())<1e-5
    assert row['changed_coordinates']==x.numel()
    def fail(module,args,output):raise RuntimeError('deliberate cleanup check')
    try:
        with torch.inference_mode(),BoundaryMonitor(boundaries,interventions={'1':[fail]}):model(x)
    except RuntimeError as e:assert str(e)=='deliberate cleanup check'
    else:raise AssertionError('Expected deliberate failure')
    assert all(not m._forward_hooks for m in model)
    with torch.inference_mode():assert torch.equal(reference,model(x))
    return dict(normal_output_exactly_unchanged=True,inplace_update_detected_and_localized=True,
        known_quadratic_energy_correct=True,hook_cleanup_on_exception=True)

if __name__=='__main__':print(json.dumps(verify(),indent=2))
