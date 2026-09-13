"""Unchanged generic dispatch classes/hash algorithm; no model/provider loading."""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import time
import torch
ROOT=Path(__file__).resolve().parents[3]
TRACE_PATH=Path(__file__).resolve().parent/"forward_trace.py"
_spec=importlib.util.spec_from_file_location("native_receiver_checked_trace",TRACE_PATH)
trace=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(trace)
span=trace.span
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()

class DispatchStopped(RuntimeError):
    pass


class DispatchLatch:
    def __init__(self):
        self.failed = False
        self.reason = None

    def stop(self, reason):
        # Finite codes only. Never serialize arbitrary exception strings here.
        self.failed = True
        if self.reason is None:
            self.reason = reason

    def admit(self):
        if self.failed:
            raise DispatchStopped(self.reason)


class ForwardDerivativeGuard:
    """Installed before loader construction; one claimed call per explicit ticket.

    The schedule reserves its attempt first. This guard verifies that reservation
    and rejects loader/setup/reentrant forwards before the original can execute.
    Global autograd.grad is similarly closed except for the single counted call.
    """
    def __init__(self, model_class, counters, latch, deadline):
        self.model_class, self.counters, self.latch = model_class, counters, latch
        self.original_forward, self.original_grad = model_class.forward, torch.autograd.grad
        self.deadline = deadline
        self.forward_ticket = self.derivative_ticket = None
        self.forwards = self.derivatives = self.rejected = 0
        self.installed = False

    def install(self):
        require(not self.installed, "single guard installation")
        def forward(model, *args, **kwargs):
            self.latch.admit()
            if self.forward_ticket != self.forwards + 1:
                self.rejected += 1
                self.latch.stop("UNCLAIMED_FORWARD")
                raise DispatchStopped("UNCLAIMED_FORWARD")
            self.forward_ticket = None
            self.forwards += 1
            with span("BRIDGE_DISPATCH"):
                return self.original_forward(model, *args, **kwargs)
        def grad(*args, **kwargs):
            self.latch.admit()
            if self.derivative_ticket != self.derivatives + 1:
                self.rejected += 1
                self.latch.stop("UNCLAIMED_DERIVATIVE")
                raise DispatchStopped("UNCLAIMED_DERIVATIVE")
            self.derivative_ticket = None
            self.derivatives += 1
            return self.original_grad(*args, **kwargs)
        self.forward_wrapper, self.grad_wrapper = forward, grad
        self.model_class.forward, torch.autograd.grad = forward, grad
        self.installed = True

    @contextmanager
    def permit(self, kind):
        self.latch.admit()
        if time.monotonic() >= self.deadline:
            self.latch.stop("DEADLINE")
            raise DispatchStopped("DEADLINE")
        completed = self.forwards if kind == "forward" else self.derivatives
        attr = "forward_ticket" if kind == "forward" else "derivative_ticket"
        require(self.installed and self.model_class.forward is self.forward_wrapper
            and torch.autograd.grad is self.grad_wrapper, "guard identity before dispatch")
        require(getattr(self, attr) is None and self.counters.attempts[kind] == completed + 1,
                "exact previously reserved attempt")
        setattr(self, attr, completed + 1)
        try:
            yield
            require(getattr(self, attr) is None, "one actual call consumed its ticket")
        except BaseException:
            self.latch.stop("FORWARD_FAILURE" if kind == "forward" else "DERIVATIVE_FAILURE")
            raise
        finally:
            setattr(self, attr, None)

    def restore(self):
        intact = self.model_class.forward is self.forward_wrapper and torch.autograd.grad is self.grad_wrapper
        self.model_class.forward, torch.autograd.grad = self.original_forward, self.original_grad
        self.installed = False
        require(intact, "guard implementation changed during ownership")


def parameter_digest(parameters):
    # Byte-for-byte algorithm from the pinned real editor, not a dummy weight.
    digest = hashlib.sha256()
    for p in parameters:
        digest.update(memoryview(p.detach().cpu().contiguous().numpy()).cast("B"))
    return digest.hexdigest()
