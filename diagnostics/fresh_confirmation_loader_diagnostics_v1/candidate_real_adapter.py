"""Real tensor/bridge adapter. Construction is injected; this module loads no model.

Only exact token tensors, an operational phase, and a residual offset reach the
model. Policy/gold/category are never model arguments or gate feature inputs.
"""
from contextlib import contextmanager
import hashlib
import time
import torch
from support import require, sha

HOOK = "blocks.10.hook_out"


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


class RealAdapter:
    def __init__(self, model, recorder, guard, *, width=1024, vocabulary=248320,
                 expected_weight_sha256=None, runtime_metadata=None, diagnostics=None):
        self.model, self.recorder, self.guard = model, recorder, guard
        self.latch = guard.latch
        self.width, self.vocabulary = width, vocabulary
        self.named_parameters = list(model.named_parameters())
        self.parameters = [p for _, p in self.named_parameters]
        diagnostics.capture_enumeration("ADAPTER_CAPTURE",self.named_parameters)
        diagnostics.check(require,self.parameters and all(p.device.type == "cpu" and
            (not p.is_floating_point() or p.dtype == torch.float32) for p in self.parameters), "all actual parameters CPU float32")
        diagnostics.check(require,not model.training and all(p.grad is None for p in self.parameters), "eval and no inherited gradients")
        self.flags, self.versions = [p.requires_grad for p in self.parameters], [p._version for p in self.parameters]
        self.initial_digest = parameter_digest(self.parameters)
        diagnostics.retain_digest(self.initial_digest,expected_weight_sha256)
        if expected_weight_sha256 is not None:
            diagnostics.check(require,self.initial_digest == expected_weight_sha256, "actual weights match frozen gate runtime")
        self.runtime_metadata = runtime_metadata
        self.request_id = None
        self.editing = False
        self.capture = None
        self.graph_leaf = None
        self.edit_hook_registrations = 0
        self.cleanup_records = []
        diagnostics.check(require,getattr(model, "_last_hf_cache", None) is None, "no inherited bridge cache")

    def parameter_state(self, *, digest=False):
        current = list(self.model.named_parameters())
        identities = len(current) == len(self.named_parameters) and all(
            a == b and x is y for (a,x),(b,y) in zip(current,self.named_parameters,strict=True))
        result = {"parameter_identities_unchanged":identities,
            "parameter_versions_unchanged":[p._version for p in self.parameters] == self.versions,
            "parameter_gradients_absent":all(p.grad is None for p in self.parameters),
            "parameter_flags_restored":[p.requires_grad for p in self.parameters] == self.flags,
            "parameter_count":len(self.parameters),
            "initial_parameter_flags":list(self.flags),
            "current_parameter_flags":[p.requires_grad for p in self.parameters],
            "initial_parameter_versions":list(self.versions),
            "current_parameter_versions":[p._version for p in self.parameters]}
        if digest:
            result.update(parameter_sha256=parameter_digest(self.parameters),initial_parameter_sha256=self.initial_digest)
            result["parameter_bytes_unchanged"] = result["parameter_sha256"] == self.initial_digest
        return result

    def _inspect(self, label):
        try:
            require(self.recorder.inspect(self.model,label), "exact hook/module registry identity")
        except BaseException:
            self.latch.stop("HOOK_EVIDENCE_FAILURE")
            raise

    def clean(self, label, *, digest=False):
        # Recording errors must propagate: cleanup is never equivalent to PASS.
        state = self.parameter_state(digest=digest)
        self._inspect(label)
        state.update(hook_registry_restored=True,active_request_empty=self.request_id is None,
            wrapper_cache_empty=self.capture is None and self.graph_leaf is None,
            bridge_cache_empty=getattr(self.model,"_last_hf_cache",None) is None)
        bools = [v for k,v in state.items() if type(v) is bool]
        require(all(bools), "complete measured cold receiver identity")
        return state

    def start_request(self, request_id):
        self.latch.admit()
        self.clean("REQUEST-entry:"+request_id)
        self.request_id = request_id

    def begin_edit(self):
        self.latch.admit()
        require(self.request_id is not None and not self.editing, "one live fresh ON receiver")
        self._inspect("ON-entry:"+self.request_id)
        require(self.parameter_state()["parameter_flags_restored"], "pre-edit original flags")
        for p in self.parameters:
            p.requires_grad_(False)
        self.editing = True

    def clear_capture(self):
        self.capture = self.graph_leaf = None
        if hasattr(self.model,"_last_hf_cache"):
            self.model._last_hf_cache = None

    def end_edit(self):
        for p,flag in zip(self.parameters,self.flags,strict=True):
            p.requires_grad_(flag)
        self.editing = False
        self.clear_capture()

    def finish_request(self):
        rid = self.request_id
        self.end_edit()
        self.request_id = None
        value = self.clean("REQUEST-exit:"+rid)
        self.cleanup_records.append(value)
        return value

    def forward_inputs(self, input_ids, attention_mask, phase, offset):
        self.latch.admit()
        require(self.capture is None and self.graph_leaf is None, "no replay/cached receiver")
        require(type(input_ids) is list and type(attention_mask) is list and 1 <= len(input_ids) <= 160
            and all(type(x) is int and 0 <= x < self.vocabulary for x in input_ids)
            and attention_mask == [1]*len(input_ids), "locked complete all-input CPU tensor shape")
        require(offset.shape == (self.width,) and offset.dtype == torch.float32 and offset.device.type == "cpu"
            and bool(offset.isfinite().all()), "finite exact float32 residual offset")
        gradient = phase.startswith("gradient_")
        edited = bool(offset.any())
        require(not (gradient or edited) or self.editing, "no gradient/edit outside fresh ON request")
        require(not (phase in ("baseline","entry")) or not edited, "fresh route state unedited")
        tokens = torch.tensor([input_ids],dtype=torch.int64,device="cpu")
        mask = torch.tensor([attention_mask],dtype=torch.int64,device="cpu")
        captured = {}
        def patch(activation, hook):
            del hook
            require(not captured, "exactly one selected block output hook")
            require(activation.shape == (1,len(input_ids),self.width) and activation.dtype == torch.float32
                and activation.device.type == "cpu", "actual block-10 residual shape/device/dtype")
            before = activation.detach().clone()
            changed = activation
            if edited:
                changed = activation.clone()
                changed[:, -1, :] = changed[:, -1, :] + offset
            if gradient:
                # This returned full activation leaf actually feeds all downstream
                # layers. Gradients select its final INPUT position only afterward.
                changed = changed.detach().requires_grad_(True)
                self.graph_leaf = changed
            captured.update(before=before,after=changed.detach().clone())
            return changed
        if edited:
            self.edit_hook_registrations += 1
        try:
            with self.guard.permit("forward"), (torch.enable_grad() if gradient else torch.inference_mode()), self.model.hooks(fwd_hooks=[(HOOK,patch)]):
                output = self.model(tokens, attention_mask=mask, use_cache=False, return_type="logits")
            require(output.shape == (1,len(input_ids),self.vocabulary) and output.dtype == torch.float32
                and output.device.type == "cpu" and captured, "one actual complete vocabulary output")
            final = output[0,-1]
            before,after = captured["before"],captured["after"]
            require(bool(final.isfinite().all()) and bool(after.isfinite().all()), "finite actual logits and residual")
            require(torch.equal(before[:,:-1],after[:,:-1]), "no nonfinal-position edit")
            require(float((after[0,-1]-(before[0,-1]+offset)).abs().max()) <= 1e-6, "actual final-input edit equals requested offset")
            self.capture = {"unselected_sha256":sha(after[:,:-1].contiguous().numpy().astype("<f4",copy=False).tobytes()),
                "input_int64_le_sha256":sha(tokens.numpy().astype("<i8",copy=False).tobytes()),
                "final_input_index":len(input_ids)-1,"hook":HOOK,"hook_calls":1,
                "nonfinal_positions":len(input_ids)-1,"logit_count":int(final.numel()),
                "parameter_versions_unchanged":[p._version for p in self.parameters] == self.versions}
            require(self.capture["parameter_versions_unchanged"], "no parameter update during actual forward")
            return final,after[0,-1]
        except BaseException:
            self.latch.stop("FORWARD_OR_CAPTURE_FAILURE")
            raise

    def gradient(self, logits):
        self.latch.admit()
        require(self.editing and self.graph_leaf is not None and self.graph_leaf.is_leaf,
                "current actual downstream graph from returned activation leaf")
        try:
            with self.guard.permit("derivative"):
                full = torch.autograd.grad(logits[50057]-logits[48964],self.graph_leaf,
                    retain_graph=False,create_graph=False)[0]
            g = full[0,-1].detach().float().cpu()
            require(g.shape == (self.width,) and bool(g.isfinite().all()), "full current final-input gradient")
            require(all(p.grad is None for p in self.parameters), "parameter gradients absent")
            return g
        except BaseException:
            self.latch.stop("DERIVATIVE_OR_CAPTURE_FAILURE")
            raise

    def finalize(self):
        self.end_edit()
        self.request_id = None
        return self.clean("matrix-finally",digest=True)


def make_backend(writer, counters, deadline):
    from authority import authenticate
    admitted = authenticate()
    if admitted["execution"]["mode"] == "INJECTION_MODULE":
        from injection_backend import load_adapter
    else:
        from loader import load_adapter
    return load_adapter(writer,counters,deadline,admitted)

