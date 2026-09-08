"""Nonrenewable one-load/one-baseline ticket; no derivatives or encoding entry."""
from types import MappingProxyType

class DiagnosticStopped(RuntimeError):pass

class Counter:
    def __init__(self,cell_id):
        self.cell_id=cell_id;self._attempts={"load":0,"forward":0,"derivative":0}
        self.guard=None;self.load_calls=0;self.sealed=False;self.reason=None
    @property
    def attempts(self):return MappingProxyType(self._attempts)
    def deny(self,code):
        self.reason=self.reason or code;self.sealed=True
        if self.guard is not None:self.guard.latch.stop(code)
        raise DiagnosticStopped(code)
    def reserve_load(self):
        if self.sealed or self._attempts["load"]:self.deny("DIAGNOSTIC_SECOND_LOAD")
        self._attempts["load"]=1
    def consume_load(self):
        if self.sealed or self._attempts["load"]!=1 or self.load_calls:self.deny("DIAGNOSTIC_SECOND_LOAD")
        self.load_calls=1
    def reserve_forward(self,cell_id):
        if self.sealed or cell_id!=self.cell_id or self.load_calls!=1 or self._attempts["forward"]:
            self.deny("DIAGNOSTIC_FORWARD_BOUNDARY")
        self._attempts["forward"]=1
    def reserve_attempt(self,kind):self.deny("DIAGNOSTIC_UNSCOPED_DISPATCH")
    def snapshot(self):
        return {"attempts":dict(self._attempts),"actual_load_dispatches":self.load_calls,
            "guarded_forwards":self.guard.forwards if self.guard else 0,"derivatives":self.guard.derivatives if self.guard else 0,
            "rejected_dispatches":self.guard.rejected if self.guard else 0,"tokenizer_calls":0,
            "sealed":self.sealed,"stop_reason":self.reason,"exact_cell_id":self.cell_id}

class LoadSources:
    def __init__(self,base,counter):self.base=base;self.counter=counter
    def read(self,*args,**kwargs):return self.base.read(*args,**kwargs)
    def load(self,name,commit,path):
        module=self.base.load(name,commit,path)
        if path=="src/sp_lense/backend.py":
            original=module.ResearchBackend.load
            def once(*args,**kwargs):
                self.counter.consume_load()
                return original(*args,**kwargs)
            module.ResearchBackend.load=staticmethod(once)
        return module
