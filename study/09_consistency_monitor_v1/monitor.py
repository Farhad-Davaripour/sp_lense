"""Direct trusted-boundary monitor for PyTorch modules with residual outputs.

The first hook clones the module's unmodified output. Declared modification
hooks run next; the last hook measures the output passed onward. This trusted
ordering is essential. Modifications inside the module or before the first hook
are outside coverage. The monitor never compares a layer with its predecessor.
No question, vector, condition label or paired inference is a scoring input.
Not thread safe; use one context for one inference at a time.
"""
from collections import defaultdict
import torch

def hidden_output(output):
    h=output[0] if isinstance(output,tuple) else output
    if not isinstance(h,torch.Tensor) or h.ndim<1:
        raise TypeError('Expected a Tensor residual output or tuple with Tensor first')
    return h

class BoundaryMonitor:
    def __init__(self,boundaries,*,interventions=None,max_absolute_tolerance=0.,keep_last_vectors=False):
        self.boundaries=dict(boundaries)
        if not self.boundaries or len({id(m) for m in self.boundaries.values()})!=len(self.boundaries):
            raise ValueError('Distinct named module boundaries required')
        self.interventions=dict(interventions or {})
        if not set(self.interventions).issubset(self.boundaries):raise ValueError('Unmonitored intervention boundary')
        self.tolerance=float(max_absolute_tolerance)
        if not torch.isfinite(torch.tensor(self.tolerance)) or self.tolerance<0:raise ValueError('Invalid fixed tolerance')
        self.keep_last_vectors=bool(keep_last_vectors)
        self.records=[];self.last_vectors=defaultdict(list);self._pending={};self._calls=defaultdict(int);self._handles=[]

    def __enter__(self):
        if self._handles or self.records:raise RuntimeError('Create a fresh monitor for each inference')
        # Refuse ambiguous ordering with existing hooks. Install observer hooks
        # through this context or extend the trusted ordering explicitly.
        if any(len(m._forward_hooks) for m in self.boundaries.values()):
            raise RuntimeError('Existing forward hooks make trusted ordering ambiguous')
        try:
            for name,module in self.boundaries.items():
                def capture(mod,args,output,name=name):
                    if name in self._pending:raise RuntimeError('Recursive/concurrent module call unsupported')
                    self._pending[name]=hidden_output(output).detach().clone()
                self._handles.append(module.register_forward_hook(capture))
                for callback in self.interventions.get(name,[]):
                    self._handles.append(module.register_forward_hook(callback))
                def observe(mod,args,output,name=name):
                    expected=self._pending.pop(name);actual=hidden_output(output).detach()
                    if expected.shape!=actual.shape or expected.dtype!=actual.dtype:raise ValueError('Boundary type/shape changed')
                    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():raise ValueError('Nonfinite residual')
                    difference=actual.to(torch.float64)-expected.to(torch.float64)
                    energy=.5*difference.square().sum(dim=-1)
                    absolute=float(difference.abs().max())
                    self.records.append(dict(boundary=name,call=self._calls[name],shape=list(actual.shape),dtype=str(actual.dtype),
                        positions=energy.numel(),energy_sum=float(energy.sum()),energy_min=float(energy.min()),
                        energy_max=float(energy.max()),energy_mean=float(energy.mean()),max_absolute_discrepancy=absolute,
                        changed_coordinates=int(torch.count_nonzero(difference)),nonzero_positions=int(torch.count_nonzero(energy)),
                        mismatch=bool(absolute>self.tolerance)))
                    if self.keep_last_vectors:
                        self.last_vectors[name].append((expected.reshape(-1,expected.shape[-1])[-1].float().cpu().clone(),
                            actual.reshape(-1,actual.shape[-1])[-1].float().cpu().clone()))
                    self._calls[name]+=1
                    # Returning None leaves the module output unchanged.
                self._handles.append(module.register_forward_hook(observe))
        except BaseException:
            self.close();raise
        return self

    def close(self):
        for handle in reversed(self._handles):handle.remove()
        self._handles.clear();self._pending.clear()

    def __exit__(self,exc_type,exc_value,traceback):
        self.close()
        return False

    def summary(self):
        if self._handles or self._pending:raise RuntimeError('Finish the inference context before summarizing')
        return dict(computation_mismatch=any(r['mismatch'] for r in self.records),
            mismatch_boundaries=sorted({r['boundary'] for r in self.records if r['mismatch']}),
            energy_sum=sum(r['energy_sum'] for r in self.records),
            max_absolute_discrepancy=max((r['max_absolute_discrepancy'] for r in self.records),default=0.),
            boundary_calls=len(self.records),max_absolute_tolerance=self.tolerance,
            scope='Actual residual-output modification at trusted monitored boundaries; not blind activation classification')
