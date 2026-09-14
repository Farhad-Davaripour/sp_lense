"""Minimal native loader/observer bridge over explicitly injected providers.

This is one small bridge, not an authority, bootstrap or framework. It imports
no real ``torch``/``transformers`` and performs no file, network or manifest
reads. Every provider is injected by the caller:

    ``build_native_adapter(snapshot_dir, config_dict, checkpoint_keys, *,
    torch_api, config_class, tokenizer_class, model_class)``

The returned object exposes ``encode``/``decode`` (forwarded to the injected
tokenizer) and ``capture_view(*, input_ids, readout_index, final_input_index)``
whose result is compatible with ``capture_executor.execute_cases``.

Authentication boundary
-----------------------
Injection is **not** authentication or permission. The future outer run owner
MUST verify snapshot bytes, pinned provider/source/runtime identity, a finite
execution lock and ownership BEFORE invoking this factory. This module makes no
claim that an injected class, snapshot or activation is the frozen artifact; it
only checks the frozen shape/config/revision *assumptions* declared below. The
returned adapter reports ``native_model_provenance_verified = False``.

Coverage algorithm copied, with attribution, from
``development/native_gate_prechoice_readout_v1/train_capture/loader.py``
(``UNUSED_MTP`` and ``coverage()``): the exact 15 unused MTP keys, all checkpoint
key/shapes, the exact tied ``lm_head.weight`` alias to
``model.language_model.embed_tokens.weight``, no unexpected aliases, and a
complete loading report that includes ``conversion_errors``. The old loader's
imports/authority/bootstrap/release chain are intentionally NOT reused.
"""

import math
import os

import native_capture_contract as contract

__all__ = [
    "NativeAdapterError",
    "build_native_adapter",
    "UNUSED_MTP",
    "LOAD_INFO_FIELDS",
    "HOOK_LAYER",
    "WIDTH",
    "LAYERS",
    "VOCAB",
]

WIDTH = contract.WIDTH
LAYERS = 24
VOCAB = contract.MAX_VOCAB_ID
HOOK_LAYER = 10
ARCHITECTURES = "Qwen3_5ForConditionalGeneration"

# Exact 15-key unused MTP set declared by the frozen source (attribution: old
# train_capture/loader.py). A different exclusion set is rejected.
UNUSED_MTP = frozenset((
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
))

LOAD_INFO_FIELDS = (
    "missing_keys",
    "unexpected_keys",
    "mismatched_keys",
    "error_msgs",
    "conversion_errors",
)

_TIED = "lm_head.weight"
_EMBED = "model.language_model.embed_tokens.weight"


class NativeAdapterError(ValueError):
    """Structured rejection with a stable ``code`` and optional detail."""

    def __init__(self, code, detail=""):
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else "%s: %s" % (code, detail))


def _need(condition, code, detail=""):
    if not condition:
        raise NativeAdapterError(code, detail)


def _shape_of(value, name):
    if isinstance(value, dict) and "shape" in value:
        value = value["shape"]
    _need(
        type(value) in (list, tuple) and all(type(item) is int for item in value),
        "CHECKPOINT_KEYS",
        name,
    )
    return tuple(value)


def _coverage(named, state_shapes, checkpoint_keys, report):
    """Exact key/shape/tied-alias coverage (attribution: old loader.py)."""
    _need(type(checkpoint_keys) is dict and checkpoint_keys, "CHECKPOINT_KEYS")
    excluded = {key for key in checkpoint_keys if key.startswith("mtp.")}
    _need(excluded == UNUSED_MTP, "EXACT_UNUSED_MTP_KEYS")
    required = {
        key: _shape_of(value, key)
        for key, value in checkpoint_keys.items()
        if key not in excluded
    }
    _need(_EMBED in required, "ALL_STATE_KEYS_SHAPES", _EMBED)
    expected = {**required, _TIED: required[_EMBED]}
    _need(state_shapes == expected, "ALL_STATE_KEYS_SHAPES")
    pairs = dict(named)
    _need(set(pairs) == set(expected), "ALL_PARAMETER_KEYS")
    _need(pairs[_TIED] is pairs[_EMBED], "EXACT_TIED_ALIAS")
    _need(
        len({id(parameter) for _, parameter in named}) == len(required),
        "NO_EXTRA_PARAMETER_ALIASES",
    )
    _need(set(report) == set(LOAD_INFO_FIELDS), "LOAD_INFO_SCHEMA")
    _need(all(not value for value in report.values()), "LOAD_INFO_FAILURE")
    return {
        "checkpoint_key_count": len(checkpoint_keys),
        "native_unique_parameters": len(required),
        "named_occurrences": len(named),
        "excluded_exact_mtp_keys": sorted(excluded),
        "tied_alias": {_TIED: _EMBED},
        "complete_key_shape_coverage": True,
    }


def _assert_config_dict(config_dict):
    _need(type(config_dict) is dict, "CONFIG_DICT")
    _need(config_dict.get("architectures") == [ARCHITECTURES], "CONFIG_ARCHITECTURES")
    text_config = config_dict.get("text_config")
    _need(type(text_config) is dict, "CONFIG_TEXT")
    _need(text_config.get("hidden_size") == WIDTH, "CONFIG_WIDTH")
    _need(text_config.get("num_hidden_layers") == LAYERS, "CONFIG_LAYERS")
    _need(text_config.get("vocab_size") == VOCAB, "CONFIG_VOCAB")
    _need(config_dict.get("tie_word_embeddings") is True, "CONFIG_TIED")


def _assert_config(config):
    _need(getattr(config, "architectures", None) == [ARCHITECTURES], "CONFIG_ARCHITECTURES")
    text_config = getattr(config, "text_config", None)
    _need(text_config is not None, "CONFIG_TEXT")
    _need(getattr(text_config, "hidden_size", None) == WIDTH, "CONFIG_WIDTH")
    _need(getattr(text_config, "num_hidden_layers", None) == LAYERS, "CONFIG_LAYERS")
    _need(getattr(text_config, "vocab_size", None) == VOCAB, "CONFIG_VOCAB")
    _need(getattr(config, "tie_word_embeddings", None) is True, "CONFIG_TIED")


def _validate_input_ids(input_ids, readout_index, final_input_index):
    _need(type(input_ids) is list, "TOKEN_IDS", "input_ids must be a list")
    _need(1 <= len(input_ids) <= contract.MAX_VIEW_TOKENS, "TOKEN_IDS", "length")
    _need(
        all(type(value) is int and 0 <= value < VOCAB for value in input_ids),
        "TOKEN_IDS",
        "vocabulary",
    )
    _need(type(readout_index) is int and type(final_input_index) is int, "INDEX")
    _need(0 <= readout_index < final_input_index, "INDEX")
    _need(final_input_index == len(input_ids) - 1, "INDEX")
    _need(input_ids[readout_index] == contract.LAST_SHARED_ID, "LAST_SHARED_ID")
    _need(input_ids[readout_index + 1] in contract.LABEL_TOKEN_IDS, "LABEL_BOUNDARY")
    return list(input_ids)


def _assert_no_forward_hooks(model, torch_api):
    """Reject forward/pre hooks across the model tree and supplied global API."""
    _need(callable(getattr(model, "modules", None)), "MODULE_TREE")
    for module in model.modules():
        for field in ("_forward_hooks", "_forward_pre_hooks"):
            registry = getattr(module, field, {})
            _need(isinstance(registry, dict) and not registry, "PREEXISTING_HOOKS", field)
    global_module = getattr(getattr(getattr(torch_api, "nn", None), "modules", None), "module", None)
    if global_module is not None:
        for field in ("_global_forward_hooks", "_global_forward_pre_hooks"):
            registry = getattr(global_module, field, {})
            _need(isinstance(registry, dict) and not registry, "PREEXISTING_HOOKS", field)


class NativeAdapter:
    """Injected-provider adapter; numeric-only capture view for the executor."""

    def __init__(self, tokenizer, model, torch_api, coverage, parameters,
                 parameter_identities, loading_report):
        self._tokenizer = tokenizer
        self._model = model
        self._torch_api = torch_api
        self.coverage = coverage
        self.parameters = parameters
        self.loading_report = loading_report
        self._parameter_identities = parameter_identities
        # Immutable baseline, independent of the informational public metadata.
        self._parameter_baseline = tuple(
            (row["name"], row["identity"], row["version"]) for row in parameters
        )
        self.provenance = {
            "snapshot_bytes_verified_by_this_adapter": False,
            "native_model_provenance_verified": False,
            "native_tokenizer_provenance_verified": False,
            "injected_providers_are_not_authenticated": True,
            "outer_owner_verifies_before_invocation": True,
        }

    def encode(self, text, *, add_special_tokens):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        return self._tokenizer.decode(
            list(ids),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def capture_view(self, *, input_ids, readout_index, final_input_index):
        ids = _validate_input_ids(input_ids, readout_index, final_input_index)
        torch_api = self._torch_api
        model = self._model
        _assert_no_forward_hooks(model, torch_api)
        layer = model.model.language_model.layers[HOOK_LAYER]
        registry = getattr(layer, "_forward_hooks", None)
        _need(isinstance(registry, dict), "HOOK_REGISTRY")
        # Reject incompatible preexisting hooks instead of erasing them.
        _need(len(registry) == 0, "PREEXISTING_HOOKS", "layer %d already hooked" % HOOK_LAYER)
        registry_before = dict(registry)
        named_before = list(model.named_parameters(remove_duplicate=False))
        _need(
            tuple(id(parameter) for _, parameter in named_before)
            == self._parameter_identities,
            "PARAMETER_IDENTITY",
        )
        versions_before = tuple(
            (name, id(parameter), parameter._version)
            for name, parameter in named_before
        )
        _need(versions_before == self._parameter_baseline, "PARAMETER_CONTINUITY")
        state = {"calls": 0, "shape_ok": False, "tensor": None, "before": None, "selected": None}

        def observer(module, args, output):
            state["calls"] += 1
            activation = output[0] if isinstance(output, (tuple, list)) else output
            _need(
                tuple(activation.shape) == (1, len(ids), WIDTH),
                "HOOK_SHAPE",
                "expected (1, %d, %d)" % (len(ids), WIDTH),
            )
            state["shape_ok"] = True
            # Copy, never mutate: keep the original tensor and a cloned snapshot.
            state["tensor"] = activation
            state["before"] = activation.detach().clone()
            state["selected"] = (
                activation[0, readout_index, :].detach().to("cpu").float().clone()
            )
            return None

        handle = None
        values = None
        try:
            handle = layer.register_forward_hook(observer)
            with torch_api.inference_mode():
                input_tensor = torch_api.tensor([ids], dtype=torch_api.long)
                attention_mask = torch_api.ones((1, len(ids)), dtype=torch_api.long)
                outputs = model(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=1,
                )
                del outputs, input_tensor, attention_mask
                # Whole-tensor comparison stays inside inference_mode.
                _need(state["calls"] == 1, "HOOK_CALLS", "hook_calls=%d" % state["calls"])
                _need(state["shape_ok"], "HOOK_SHAPE")
                _need(
                    torch_api.equal(state["before"], state["tensor"]),
                    "POSITIONS_CHANGED",
                )
                values = [float(value) for value in state["selected"].tolist()]
                _need(
                    len(values) == WIDTH and all(math.isfinite(value) for value in values),
                    "VALUES",
                )
            named_after = list(model.named_parameters(remove_duplicate=False))
            versions_after = tuple(
                (name, id(parameter), parameter._version)
                for name, parameter in named_after
            )
            _need(versions_after == versions_before, "PARAMETERS_CHANGED")
        finally:
            if handle is not None:
                handle.remove()
            state["tensor"] = None
            state["before"] = None
            state["selected"] = None
            if handle is not None:
                _need(
                    dict(registry) == registry_before,
                    "HOOK_CLEANUP",
                    "own hook was not removed cleanly",
                )
                _assert_no_forward_hooks(model, torch_api)
        return {
            "values": values,
            "hook_calls": 1,
            "all_positions_unchanged": True,
            "parameters_unchanged": True,
        }


def build_native_adapter(snapshot_dir, config_dict, checkpoint_keys, *,
                         torch_api, config_class, tokenizer_class, model_class):
    """Build one adapter over injected providers; no real provider is imported.

    The outer run owner must verify snapshot bytes, pinned provider/source/
    runtime identity, a finite execution lock and ownership before calling.
    """
    _need(isinstance(snapshot_dir, (str, os.PathLike)), "SNAPSHOT_DIR")
    directory = os.fspath(snapshot_dir)
    # Revision is a directory-name assumption only; bytes are the owner's job.
    _need(
        os.path.basename(os.path.normpath(directory)) == contract.MODEL_REVISION,
        "SNAPSHOT_REVISION",
    )
    _need(hasattr(config_class, "from_dict"), "PROVIDER", "config_class")
    _need(hasattr(tokenizer_class, "from_pretrained"), "PROVIDER", "tokenizer_class")
    _need(hasattr(model_class, "from_pretrained"), "PROVIDER", "model_class")
    _need(hasattr(torch_api, "float32") and hasattr(torch_api, "long"), "TORCH_API")
    for name in ("tensor", "ones", "equal", "inference_mode"):
        _need(callable(getattr(torch_api, name, None)), "TORCH_API", name)

    _assert_config_dict(config_dict)
    try:
        config = config_class.from_dict(dict(config_dict))
    except NativeAdapterError:
        raise
    except Exception as exc:
        raise NativeAdapterError("CONFIG_BUILD", type(exc).__name__) from exc
    _assert_config(config)

    try:
        tokenizer = tokenizer_class.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
    except NativeAdapterError:
        raise
    except Exception as exc:
        raise NativeAdapterError("TOKENIZER_LOAD", type(exc).__name__) from exc

    _need(hasattr(model_class, "_finalize_model_loading"), "FINALIZER_MISSING")
    own_finalizer = "_finalize_model_loading" in model_class.__dict__
    previous_descriptor = model_class.__dict__.get("_finalize_model_loading")
    original_finalizer = model_class._finalize_model_loading
    captured = []

    def observed_finalizer(model, load_config, loading_info):
        result = original_finalizer(model, load_config, loading_info)
        _need(not captured, "ONE_FINALIZER")
        captured.append(
            {key: list(getattr(result, key)) for key in LOAD_INFO_FIELDS}
        )
        return result

    model_class._finalize_model_loading = staticmethod(observed_finalizer)
    try:
        model, info = model_class.from_pretrained(
            directory,
            config=config,
            dtype=torch_api.float32,
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
            weights_only=True,
            use_safetensors=True,
            ignore_mismatched_sizes=False,
        )
    except NativeAdapterError:
        raise
    except Exception as exc:
        raise NativeAdapterError("MODEL_LOAD", type(exc).__name__) from exc
    finally:
        # Restore any observer instrumentation, success or failure.
        if own_finalizer:
            model_class._finalize_model_loading = previous_descriptor
        else:
            delattr(model_class, "_finalize_model_loading")

    _need(
        len(captured) == 1 and type(info) is dict
        and all(not value for value in info.values()),
        "COMPLETE_LOAD_REPORT",
    )

    model.eval()
    named = list(model.named_parameters(remove_duplicate=False))
    for _, parameter in named:
        parameter.requires_grad_(False)
    named = list(model.named_parameters(remove_duplicate=False))
    _need(
        all(not parameter.requires_grad and parameter.grad is None for _, parameter in named),
        "REQUIRES_GRAD",
    )
    _need(getattr(model.config, "_attn_implementation", None) == "eager", "EAGER")
    text_config = getattr(model.config, "text_config", None)
    _need(getattr(text_config, "_attn_implementation", None) == "eager", "EAGER")
    _need(getattr(getattr(model.config, "vision_config", None), "_attn_implementation", None) == "eager", "EAGER")
    layers = getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None)
    _need(layers is not None and len(layers) == LAYERS, "LAYERS")

    state_shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    report = captured[0]
    coverage = _coverage(named, state_shapes, checkpoint_keys, report)
    _need(
        all(
            parameter.device.type == "cpu"
            and parameter.dtype == torch_api.float32
            and parameter.grad is None
            for _, parameter in named
        ),
        "CPU_FLOAT32_PARAMETERS",
    )
    parameters = [
        {
            "name": name,
            "identity": id(parameter),
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "device": str(parameter.device),
            "bytes": parameter.numel() * parameter.element_size(),
            "version": parameter._version,
            "requires_grad": parameter.requires_grad,
        }
        for name, parameter in named
    ]
    identities = tuple(id(parameter) for _, parameter in named)
    _assert_no_forward_hooks(model, torch_api)
    return NativeAdapter(
        tokenizer, model, torch_api, coverage, parameters, identities, report
    )
