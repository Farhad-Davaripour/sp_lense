"""Opt-in per-instance generation bounds for the frozen ordinal wrapper.

The original wrapper retains its 256-token limit unless this context manager
is explicitly entered.  The caller must supply bounds derived from its own
remaining episode and model-context budgets; this module neither creates an
episode nor allocates compute.  It does not change weights, adapter routing,
decoding, the original class, or the training contract.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
import hashlib
import inspect
import textwrap
from types import MethodType


_EXPECTED_MODULES = frozenset({
    "src.qwen9b_ordinal_continuity_lora",
    "research4.src.qwen9b_ordinal_continuity_lora",
})
_ORIGINAL_SHA256 = "9038ac1089839e955bf4dc294fda44bb94b61235cce5cc4eba8cec694b849642"
_ORIGINAL_ERROR = "max_new_tokens must be explicitly bounded in 1..256"


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _replacement(original, limit):
    """Compile the exact verified method with only its bound literals changed."""
    try:
        text = textwrap.dedent(inspect.getsource(original)).rstrip() + "\n"
    except (OSError, TypeError) as exc:
        raise PermissionError("Original generation source unavailable") from exc
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != _ORIGINAL_SHA256:
        raise PermissionError("Original generation source differs")
    tree = ast.parse(text)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != "generate" or tree.body[0].decorator_list):
        raise PermissionError("Original generation syntax differs")
    compiled_original = {}
    exec(compile(ast.fix_missing_locations(copy.deepcopy(tree)),
                 original.__code__.co_filename, "exec"), original.__globals__, compiled_original)
    expected_code = compiled_original["generate"].__code__
    actual_code = original.__code__
    code_fields = ("co_code", "co_consts", "co_names", "co_varnames", "co_freevars",
                   "co_cellvars", "co_argcount", "co_posonlyargcount", "co_kwonlyargcount")
    if any(getattr(actual_code, key) != getattr(expected_code, key) for key in code_fields):
        raise PermissionError("Loaded generation code differs from verified source")
    baseline = ast.dump(tree, include_attributes=False)
    candidate = copy.deepcopy(tree)
    changes = []
    for node in ast.walk(candidate):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value == 256:
            changes.append((node, node.value, limit))
        elif isinstance(node, ast.Constant) and node.value == _ORIGINAL_ERROR:
            changes.append((node, node.value,
                            f"max_new_tokens must be explicitly bounded in 1..{limit}"))
    if len(changes) != 2 or sum(type(old) is int for _, old, _ in changes) != 1:
        raise PermissionError("Original generation bound is ambiguous")
    for node, _, new in changes:
        node.value = new
    transformed = ast.dump(candidate, include_attributes=False)
    for node, old, _ in changes:
        node.value = old
    if ast.dump(candidate, include_attributes=False) != baseline:
        raise PermissionError("Generation override changes more than the bound")
    for node, _, new in changes:
        node.value = new
    if ast.dump(candidate, include_attributes=False) != transformed:
        raise PermissionError("Generation override verification failed")
    namespace = {}
    # The AST comes from the byte-verified repository method above.  Only the
    # integer bound and its matching diagnostic text can differ.
    exec(compile(ast.fix_missing_locations(candidate), original.__code__.co_filename,
                 "exec"), original.__globals__, namespace)
    return namespace["generate"]


@contextmanager
def generation_bound(model, *, enabled=False, max_new_tokens,
                     remaining_context, remaining_episode):
    """Temporarily override the verified frozen instance's token admission cap.

    ``remaining_context`` and ``remaining_episode`` are caller-supplied limits
    for the upcoming work.  The caller remains responsible for updating them
    across calls and for recording actual or unknown generated-token usage.
    Source and frozen-stage checks do not verify model weights or checkpoint
    identity; those checks remain the caller's responsibility.
    This per-instance override is not safe to share with concurrent calls.
    """
    if enabled is not True:
        raise PermissionError("Explicit opt-in is required")
    limit = _positive_int(max_new_tokens, "max_new_tokens")
    context = _positive_int(remaining_context, "remaining_context")
    episode = _positive_int(remaining_episode, "remaining_episode")
    if limit > context or limit > episode:
        raise ValueError("generation bound exceeds remaining context or episode budget")
    if "generate" in getattr(model, "__dict__", {}):
        raise PermissionError("An instance-level generation override already exists")
    bound = getattr(model, "generate", None)
    original = getattr(bound, "__func__", None)
    if (original is None or getattr(bound, "__self__", None) is not model
            or original.__module__ not in _EXPECTED_MODULES
            or getattr(model, "_training_episode_active", None) is not False
            or getattr(model, "_stage", None) != "frozen"):
        raise PermissionError("Expected frozen ordinal wrapper required")
    modified = _replacement(original, limit)

    def guarded(instance, *args, **kwargs):
        if (getattr(instance, "_training_episode_active", None) is not False
                or getattr(instance, "_stage", None) != "frozen"):
            raise PermissionError("Generation override requires a frozen evaluation instance")
        return modified(instance, *args, **kwargs)

    replacement = MethodType(guarded, model)
    model.generate = replacement
    primary = None
    try:
        if model.generate is not replacement:
            raise PermissionError("Generation override was not installed")
        yield {"original_max_new_tokens": 256, "max_new_tokens": limit,
               "original_source_sha256": _ORIGINAL_SHA256,
               "per_instance_only": True,
               "model_identity_independently_verified": False}
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            tampered = getattr(model, "generate", None) is not replacement
            if "generate" in getattr(model, "__dict__", {}):
                delattr(model, "generate")
            if getattr(getattr(model, "generate", None), "__func__", None) is not original:
                raise PermissionError("Original generation method was not restored")
            if tampered:
                raise PermissionError("Generation override changed unexpectedly")
        except BaseException as exc:
            if primary is not None:
                primary.add_note("Generation method restoration also failed: " + type(exc).__name__)
            else:
                raise
