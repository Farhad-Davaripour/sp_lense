"""Nonrenewable hard startup-only dispatch; the attempts view cannot be mutated."""
from types import MappingProxyType

class SetupStopped(RuntimeError): pass

class Counter:
    def __init__(self):
        self.reserved=False;self.load_calls=0;self.load_dispatch_attempts=0;self.reason=None;self.sealed=False
        self.guard=None
    @property
    def attempts(self): return MappingProxyType({"load":int(self.reserved),"forward":0,"derivative":0})
    def deny(self,reason):
        if self.reason is None: self.reason=reason
        self.sealed=True
        raise SetupStopped(reason)
    def reserve_load(self):
        if self.reserved or self.sealed: self.deny("SECOND_LOAD")
        self.reserved=True
    def consume_load(self):
        self.load_dispatch_attempts+=1
        if not self.reserved or self.load_calls or self.sealed: self.deny("SECOND_LOAD")
        self.load_calls=1
    def snapshot(self):
        return {"reserved_load_attempts":int(self.reserved),"actual_load_dispatches":self.load_calls,
            "load_dispatch_attempts_including_denied":self.load_dispatch_attempts,
            "forwards":self.guard.forwards if self.guard else 0,"derivatives":self.guard.derivatives if self.guard else 0,
            "tokenizer_calls":0,"stop_reason":self.reason,"sealed":self.sealed}

class LoadSources:
    def __init__(self,base,counter): self.base=base;self.counter=counter
    def read(self,*args,**kwargs): return self.base.read(*args,**kwargs)
    def load(self,name,commit,path):
        module=self.base.load(name,commit,path)
        if path=="src/sp_lense/backend.py":
            original=module.ResearchBackend.load
            def once(*args,**kwargs):
                self.counter.consume_load()
                return original(*args,**kwargs)
            module.ResearchBackend.load=staticmethod(once)
        return module
