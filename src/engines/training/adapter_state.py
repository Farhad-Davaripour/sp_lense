"""Frozen, reversible LoRA modes for offline one-model local references.

PEFT enabling can restore gradient flags, so those flags are restored explicitly.
These contexts do not merge adapters, update parameters, or inject tool actions.
"""
from contextlib import contextmanager
import copy
from peft.tuners.tuners_utils import BaseTunerLayer

def frozen_teacher_views(layers):
    """Independent module/adapter flags sharing the same frozen parameter objects.

    This avoids a second base-weight allocation and per-token global switching.
    Native Qwen cache/output parity must be verified before scientific capture.
    """
    parameters=[p for layer in layers for p in layer.parameters()]
    assert all(not p.requires_grad for p in parameters)
    memo={id(p):p for p in parameters}
    views=[copy.deepcopy(layer,memo) for layer in layers]
    for original,view in zip(layers,views):
        a,b=dict(original.named_parameters()),dict(view.named_parameters())
        assert set(a)==set(b) and all(a[name] is b[name] for name in a)
        original_tuners=[m for m in original.modules() if isinstance(m,BaseTunerLayer)]
        view_tuners=[m for m in view.modules() if isinstance(m,BaseTunerLayer)]
        assert len(original_tuners)==len(view_tuners)
        assert all(a is not b for a,b in zip(original_tuners,view_tuners))
    return views

@contextmanager
def frozen_base_mode(model):
    parameters=list(model.parameters())
    assert all(not p.requires_grad for p in parameters)
    try:
        with model.disable_adapter():
            model.requires_grad_(False)
            yield
    finally:
        model.requires_grad_(False)
    assert all(not p.requires_grad for p in parameters)

@contextmanager
def local_teacher_mode(layer):
    """Enable only the current layer's existing frozen teacher adapter temporarily."""
    tuners=[m for m in layer.modules() if isinstance(m,BaseTunerLayer)]
    parameters=list(layer.parameters())
    assert tuners and all(not p.requires_grad for p in parameters)
    prior=[m.disable_adapters for m in tuners]
    try:
        for module in tuners:module.enable_adapters(True)
        layer.requires_grad_(False)
        yield
    finally:
        for module,disabled in zip(tuners,prior):module.enable_adapters(not disabled)
        layer.requires_grad_(False)
    assert [m.disable_adapters for m in tuners]==prior
    assert all(not p.requires_grad for p in parameters)
