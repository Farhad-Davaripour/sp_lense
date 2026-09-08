"""A component, not a loader, runtime authorization, or scientific acceptance."""
import inspect
import pathlib
import sys
import types
from source_contract import Rejected, need, authenticate, code_catalog

HF_MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"
HELPERS = ("causal_conv1d_fn", "chunk_gated_delta_rule")
MODES = ("INERT_FIXTURE", "LIVE_EXISTING_MODULE")

def _code(fn, expected, path):
    need(type(fn) is types.FunctionType and fn.__code__ == expected and
         pathlib.Path(fn.__code__.co_filename).resolve() == path, "CALLABLE_CODE")

def _live_sources(module, original_type, conv, chunk, root):
    need(type(module) is types.ModuleType and module.__name__ == HF_MODULE and
         sys.modules.get(HF_MODULE) is module, "LIVE_MODULE")
    from source_contract import PINS
    hf_path = (pathlib.Path(root) / PINS["hf"][0]).resolve()
    decorator_path = (pathlib.Path(root) / PINS["decorator"][0]).resolve()
    hf_codes = code_catalog(root, "hf")
    kernel_codes = code_catalog(root, "decorator")
    _code(original_type.__init__, hf_codes["Qwen3_5GatedDeltaNet.__init__"], hf_path)
    _code(inspect.unwrap(original_type.forward), hf_codes["Qwen3_5GatedDeltaNet.forward"], hf_path)
    need(original_type.__init__.__globals__ is vars(module), "CLASS_GLOBALS")
    wrapper_code = kernel_codes["use_kernel_func_from_hub_with_fallback.<locals>.decorator.<locals>.wrapped"]
    for fn, name, defaults in ((conv, "causal_conv1d_fn", (None, None)),
                               (chunk, "torch_chunk_gated_delta_rule", (64, None, False, False))):
        _code(fn, wrapper_code, decorator_path)
        base = getattr(fn, "__wrapped__", None)
        _code(base, hf_codes[name], hf_path)
        need(base.__globals__ is vars(module) and base.__defaults__ == defaults and not base.__kwdefaults__,
             "CALLABLE_DEFAULTS")
        closure = dict(zip(fn.__code__.co_freevars, (x.cell_contents for x in fn.__closure__ or ())))
        # Admit only this already-selected, source-demonstrated implementation.
        # Never unwrap for dispatch, choose a fallback, replace a kernel, or import one.
        need(set(closure) == {"applicable_params", "implementation"} and closure["implementation"] is base and
             closure["applicable_params"] == tuple(inspect.signature(base).parameters), "SELECTED_IMPLEMENTATION_UNVERIFIED")

class Targets:
    def __init__(self, module, original_type, expected_conv, expected_chunk, root, *, mode):
        need(mode in MODES, "MODE")
        self.mode = mode
        self.sources = authenticate(root)
        need(vars(module).get("Qwen3_5GatedDeltaNet") is original_type and
             vars(module).get("causal_conv1d_fn") is expected_conv and
             vars(module).get("torch_chunk_gated_delta_rule") is expected_chunk, "TARGET_IDENTITY")
        need(type(original_type) is type and callable(expected_conv) and callable(expected_chunk), "TARGET_TYPE")
        if mode == "LIVE_EXISTING_MODULE":
            _live_sources(module, original_type, expected_conv, expected_chunk, root)
        self.module, self.original_type = module, original_type
        self.conv, self.chunk = expected_conv, expected_chunk

    def inspect(self):
        need(vars(self.module).get("Qwen3_5GatedDeltaNet") is self.original_type and
             vars(self.module).get("causal_conv1d_fn") is self.conv and
             vars(self.module).get("torch_chunk_gated_delta_rule") is self.chunk, "TARGET_DRIFT")

class Installation:
    def __init__(self, targets, labelled_instances, *, before_hook_reference):
        self.targets = targets
        self.entries = tuple(labelled_instances)
        self.state = "NEW"
        self.first_failure = None
        self.rollback_complete = None
        self.applied = []
        self.before = []
        need(before_hook_reference is True, "ADMISSION_PHASE")
        need(1 <= len(self.entries) <= 24, "INSTANCE_COUNT")
        labels = [x[0] for x in self.entries]
        need(all(type(x) is str and x.isascii() and 0 < len(x) <= 256 for x in labels) and
             len(set(labels)) == len(labels), "INSTANCE_LABEL")
        need(len({id(x[1]) for x in self.entries}) == len(self.entries), "DUPLICATE_INSTANCE")
        for _, obj in self.entries:
            need(type(obj) is targets.original_type, "ORIGINAL_TYPE")
        def conv_adapter(*, x, weight, bias, activation, seq_idx):
            need(self.state == "ACTIVE", "NOT_ACTIVE")
            need(seq_idx is None, "NONNEUTRAL_SEQ_IDX")
            targets.inspect()
            return targets.conv(hidden_states=x, weight=weight, bias=bias, activation=activation)
        self.conv_adapter = conv_adapter

    def _snapshot(self, obj):
        return {k: id(v) for k, v in vars(obj).items() if k not in HELPERS}

    def _absent(self, obj, name):
        sentinel = object()
        need(inspect.getattr_static(obj, name, sentinel) is sentinel, "EXISTING_HELPER")
        # Include dynamic __getattr__, not merely ordinary instance/class fields.
        try:
            getattr(obj, name)
        except AttributeError:
            return
        except BaseException:
            raise Rejected("ATTRIBUTE_LOOKUP")
        raise Rejected("EXISTING_HELPER")

    def _rollback(self):
        okay = True
        for obj, name, expected in reversed(self.applied):
            try:
                current = vars(obj).get(name)
                if name not in vars(obj):
                    continue
                need(current is expected, "ROLLBACK_IDENTITY")
                delattr(obj, name)
                need(name not in vars(obj), "ROLLBACK_DELETE")
            except BaseException:
                okay = False
        for (_, obj), before in zip(self.entries, self.before):
            if self._snapshot(obj) != before:
                okay = False
        self.rollback_complete = okay
        return okay

    def install(self):
        need(self.state == "NEW", "NO_RETRY")
        self.state = "INSTALLING"
        try:
            self.targets.inspect()
            for _, obj in self.entries:
                for name in HELPERS:
                    self._absent(obj, name)
                self.before.append(self._snapshot(obj))
            for _, obj in self.entries:
                for name, fn in zip(HELPERS, (self.conv_adapter, self.targets.chunk)):
                    self.applied.append((obj, name, fn))  # Before setattr: it may write and then raise.
                    setattr(obj, name, fn)
                    need(vars(obj).get(name) is fn and getattr(obj, name) is fn, "INSTALL_IDENTITY")
            self.state = "ACTIVE"
            self.inspect()
            return self
        except BaseException as exc:
            self.first_failure = exc.code if type(exc) is Rejected else "INSTALL_IO_OR_CALLBACK"
            self._rollback()
            self.state = "FAILED"
            raise Rejected(self.first_failure)

    def inspect(self):
        need(self.state == "ACTIVE", "NOT_ACTIVE")
        self.targets.inspect()
        for (_, obj), before in zip(self.entries, self.before):
            need(type(obj) is self.targets.original_type and self._snapshot(obj) == before, "INSTANCE_DRIFT")
            for name, fn in zip(HELPERS, (self.conv_adapter, self.targets.chunk)):
                need(vars(obj).get(name) is fn and getattr(obj, name) is fn, "HELPER_DRIFT")

    def close(self):
        need(self.state == "ACTIVE", "NO_RETRY")
        try:
            self.inspect()
        except BaseException as exc:
            self.first_failure = exc.code if type(exc) is Rejected else "CLOSE_INSPECTION"
        okay = self._rollback()
        self.state = "CLOSED" if okay and self.first_failure is None else "FAILED"
        need(self.state == "CLOSED", "CLOSE_INCOMPLETE")

    def status(self):
        return {"schema": "gdn_compat_component_status.v1", "mode": self.targets.mode, "state": self.state,
                "first_failure": self.first_failure, "rollback_complete": self.rollback_complete,
                "instances": [{"label": label, "identity": id(obj)} for label, obj in self.entries],
                "conv_adapter_identity": id(self.conv_adapter), "conv_target_identity": id(self.targets.conv),
                "chunk_target_identity": id(self.targets.chunk), "source_sha256": {k: v["sha256"] for k, v in self.targets.sources.items()},
                "instance_method_self_binding": False, "scientific_pass": False, "real_execution_authorized": False}
