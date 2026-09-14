"""Versioned multi-layer / multi-token span-window adapter V1.

This is one small model-free wrapper, not a framework or supervisor. It imports
no real ``torch``/``transformers`` and performs no file, network, dataset,
holdout, feature, fit or extraction work. Every provider is injected by the
caller exactly as for ``native_capture_adapter.build_native_adapter``.

The wrapper calls that already-reviewed factory **once** (one model load and one
tokenizer load), then reuses the returned object's private ``_model``,
``_torch_api`` and ``_tokenizer`` fields plus that module's guard helpers
(``_need``, ``_validate_input_ids``, ``_assert_no_forward_hooks``) without
mutating the old module or any model weight.

``capture_window(input_ids, readout_index, final_input_index)`` runs exactly one
forward over the shared pre-option prefix (the same causal meaning as the old
``capture_view``) and returns the residual outputs of blocks 6, 10 and 18 at the
last ``min(16, prefix_length)`` prefix-token positions. Each block window is a
float32, width-1024 matrix with exact token-index metadata, so a later binary
writer can serialize each layer separately. This is a new structured contract;
it is not the old 1024-wide single-vector contract and it adds no baseline
capture.

Every call registers exactly three owned forward hooks, rejects preexisting or
global hooks instead of erasing them, removes only its own handles on every
exit, and rejects absent or repeated hook callbacks. Authentication of the
snapshot and providers remains the outer run owner's job; this module only
checks the frozen shape/config assumptions and makes no native claim.
"""

import math

import native_capture_adapter as native

__all__ = [
    "SpanCaptureAdapterError",
    "SpanCaptureAdapter",
    "build_span_capture_adapter",
    "BLOCKS",
    "MAX_WINDOW",
    "WINDOW_SCHEMA",
    "JOB_ID",
]

# Fixed experiment scope: zero-based block indices among the 24 frozen layers.
BLOCKS = (6, 10, 18)
MAX_WINDOW = 16
WIDTH = native.WIDTH
WINDOW_SCHEMA = "span_capture_window.v1"
JOB_ID = "span_adapter_implementation_20260914_1035"

# Re-export the existing structured rejection so callers see one error family.
SpanCaptureAdapterError = native.NativeAdapterError

_need = native._need


def _validate_blocks(blocks):
    _need(type(blocks) is tuple and len(blocks) > 0, "BLOCKS")
    _need(len(blocks) == len(set(blocks)), "BLOCKS")
    _need(all(type(block) is int and 0 <= block < native.LAYERS for block in blocks), "BLOCKS")
    _need(tuple(sorted(blocks)) == tuple(blocks), "BLOCKS", "ordered coverage required")
    return blocks


class SpanCaptureAdapter:
    """Injected-provider span-window adapter; one forward per capture call."""

    def __init__(self, native_adapter, blocks=BLOCKS):
        _need(isinstance(native_adapter, native.NativeAdapter), "NATIVE_ADAPTER")
        self._native = native_adapter
        self._blocks = _validate_blocks(blocks)
        self._max_window = MAX_WINDOW
        self.provenance = {
            "job_id": JOB_ID,
            "span_window_schema": WINDOW_SCHEMA,
            "blocks": list(self._blocks),
            "max_window": MAX_WINDOW,
            "native_model_provenance_verified": False,
            "native_tokenizer_provenance_verified": False,
            "injected_providers_are_not_authenticated": True,
            "outer_owner_verifies_before_invocation": True,
            "old_single_vector_contract_reused": False,
            "baseline_capture_performed": False,
            "one_forward_per_capture_call": True,
        }

    # -- tokenizer forwarding (same signatures as the native adapter) -------- #
    def encode(self, text, *, add_special_tokens):
        return self._native.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        return self._native.decode(
            ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    # -- coverage / guard helpers ------------------------------------------- #
    def _resolve_layers(self, model):
        layers = getattr(
            getattr(getattr(model, "model", None), "language_model", None),
            "layers",
            None,
        )
        _need(layers is not None and hasattr(layers, "__len__"), "LAYER_COVERAGE")
        _need(len(layers) == native.LAYERS, "LAYER_COVERAGE", "layers=%d" % len(layers))
        for block in self._blocks:
            _need(block < len(layers), "LAYER_COVERAGE", "block %d" % block)
        return layers

    def _window_bounds(self, prefix_length):
        _need(type(prefix_length) is int and prefix_length >= 1, "PREFIX_LENGTH")
        window_length = min(self._max_window, prefix_length)
        start = prefix_length - window_length
        return start, window_length

    def capture_window(self, input_ids, readout_index, final_input_index):
        """Capture blocks 6/10/18 residual windows in exactly one forward."""
        full_ids = native._validate_input_ids(input_ids, readout_index, final_input_index)
        # Validate retained option-label boundaries, but never forward options.
        ids = full_ids[:readout_index + 1]
        torch_api = self._native._torch_api
        model = self._native._model
        blocks = self._blocks
        prefix_length = readout_index + 1
        window_start, window_length = self._window_bounds(prefix_length)
        positions = list(range(window_start, prefix_length))

        native._assert_no_forward_hooks(model, torch_api)
        layers = self._resolve_layers(model)
        registries_before = {}
        for block in blocks:
            registry = getattr(layers[block], "_forward_hooks", None)
            _need(isinstance(registry, dict), "HOOK_REGISTRY", "block %d" % block)
            _need(
                len(registry) == 0,
                "PREEXISTING_HOOKS",
                "layer %d already hooked" % block,
            )
            registries_before[block] = dict(registry)

        named_before = list(model.named_parameters(remove_duplicate=False))
        _need(
            tuple(id(parameter) for _, parameter in named_before)
            == self._native._parameter_identities,
            "PARAMETER_IDENTITY",
        )
        versions_before = tuple(
            (name, id(parameter), parameter._version)
            for name, parameter in named_before
        )
        _need(
            versions_before == self._native._parameter_baseline,
            "PARAMETER_CONTINUITY",
        )

        state = {
            block: {
                "calls": 0,
                "shape_ok": False,
                "tensor": None,
                "before": None,
                "matrix": None,
            }
            for block in blocks
        }

        def make_observer(block):
            def observer(module, args, output):
                entry = state[block]
                entry["calls"] += 1
                activation = output[0] if isinstance(output, (tuple, list)) else output
                _need(
                    tuple(activation.shape) == (1, len(ids), WIDTH),
                    "HOOK_SHAPE",
                    "block %d expected (1, %d, %d)" % (block, len(ids), WIDTH),
                )
                _need(
                    activation.dtype == torch_api.float32,
                    "HOOK_DTYPE",
                    "block %d dtype=%r" % (block, activation.dtype),
                )
                entry["shape_ok"] = True
                entry["tensor"] = activation
                entry["before"] = activation.detach().clone()
                rows = []
                for position in positions:
                    row = activation[0, position, :].detach().to("cpu").float().clone()
                    values = [float(value) for value in row.tolist()]
                    _need(len(values) == WIDTH, "WIDTH", "block %d" % block)
                    rows.append(values)
                entry["matrix"] = rows
                return None

            return observer

        handles = {}
        result = None
        try:
            for block in blocks:
                handles[block] = layers[block].register_forward_hook(make_observer(block))
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
                for block in blocks:
                    entry = state[block]
                    _need(
                        entry["calls"] == 1,
                        "HOOK_CALLS",
                        "block %d calls=%d" % (block, entry["calls"]),
                    )
                    _need(entry["shape_ok"], "HOOK_SHAPE", "block %d" % block)
                    _need(
                        torch_api.equal(entry["before"], entry["tensor"]),
                        "POSITIONS_CHANGED",
                        "block %d" % block,
                    )
                window_layers = {}
                for block in blocks:
                    matrix = state[block]["matrix"]
                    _need(
                        all(
                            math.isfinite(value)
                            for row in matrix
                            for value in row
                        ),
                        "VALUES",
                        "block %d" % block,
                    )
                    window_layers[block] = {
                        "block": block,
                        "shape": [window_length, WIDTH],
                        "dtype": "float32",
                        "row_bytes": WIDTH * 4,
                        "bytes": window_length * WIDTH * 4,
                        "position_indices": list(positions),
                        "matrix": matrix,
                    }
                result = {
                    "schema": WINDOW_SCHEMA,
                    "job_id": JOB_ID,
                    "blocks": list(blocks),
                    "readout_index": readout_index,
                    "final_input_index": final_input_index,
                    "shared_prefix_length": prefix_length,
                    "window_start": window_start,
                    "window_length": window_length,
                    "position_indices": list(positions),
                    "width": WIDTH,
                    "dtype": "float32",
                    "layers": window_layers,
                    "hook_calls": {block: 1 for block in blocks},
                    "all_positions_unchanged": True,
                    "parameters_unchanged": True,
                    "one_forward_per_capture_call": True,
                }
            named_after = list(model.named_parameters(remove_duplicate=False))
            versions_after = tuple(
                (name, id(parameter), parameter._version)
                for name, parameter in named_after
            )
            _need(versions_after == versions_before, "PARAMETERS_CHANGED")
        finally:
            for block in blocks:
                handle = handles.pop(block, None)
                if handle is not None:
                    handle.remove()
                state[block]["tensor"] = None
                state[block]["before"] = None
                state[block]["matrix"] = None
            for block in blocks:
                _need(
                    dict(getattr(layers[block], "_forward_hooks", {}))
                    == registries_before[block],
                    "HOOK_CLEANUP",
                    "block %d own hook was not removed cleanly" % block,
                )
            native._assert_no_forward_hooks(model, torch_api)
        return result


def build_span_capture_adapter(snapshot_dir, config_dict, checkpoint_keys, *,
                               torch_api, config_class, tokenizer_class,
                               model_class, blocks=BLOCKS):
    """Build one native adapter and wrap it; no real provider is imported.

    ``blocks`` defaults to the fixed ``(6, 10, 18)`` scope and may only be
    changed by an explicit caller for a synthetic coverage test; the frozen
    experiment uses the default. The outer run owner must still verify snapshot
    bytes, pinned provider/source/runtime identity, a finite execution lock and
    ownership before calling.
    """
    native_adapter = native.build_native_adapter(
        snapshot_dir,
        config_dict,
        checkpoint_keys,
        torch_api=torch_api,
        config_class=config_class,
        tokenizer_class=tokenizer_class,
        model_class=model_class,
    )
    adapter = SpanCaptureAdapter(native_adapter, blocks=blocks)
    return adapter
