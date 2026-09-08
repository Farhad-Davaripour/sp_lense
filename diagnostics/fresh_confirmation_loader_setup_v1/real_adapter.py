"""Only the approved instrumented candidate; import is owned real-worker only."""
from support import SOURCES
_raw=SOURCES.read("84bfd749876c44dc9bb1bc1e75db89f3e491c159","diagnostics/fresh_confirmation_loader_diagnostics_v1/candidate_real_adapter.py")
exec(compile(_raw,__file__,"exec"),globals())

_OriginalGuard=ForwardDerivativeGuard
def ForwardDerivativeGuard(model_class,counters,latch,deadline):
    guard=_OriginalGuard(model_class,counters,latch,deadline)
    counters.guard=guard
    return guard
