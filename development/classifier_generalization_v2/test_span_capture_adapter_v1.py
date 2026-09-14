"""Fabricated-dependency tests for span_capture_adapter_v1.py.

Every provider is the small numpy-backed fake already used by
``test_native_capture_adapter``; this file adds a three-layer forward fake and a
wrong-dtype layer. No real ``torch``/``transformers`` module is imported,
instantiated or downloaded, no snapshot/model/tokenizer/dataset file is read,
and nothing is fitted or extracted.

Run from the repository root with the study runtime:

    development\\classifier_generalization_v2\\.runtime\\Scripts\\python.exe \
        -W error development/classifier_generalization_v2/test_span_capture_adapter_v1.py -v
"""

import ast
import gc
import inspect
import math
import unittest
import weakref
from types import SimpleNamespace

import numpy as np

import native_capture_adapter as native
import native_capture_contract as contract
import span_capture_adapter_v1 as span
import test_native_capture_adapter as base

SNAPSHOT = base.SNAPSHOT
EMBED = base.EMBED


# --------------------------------------------------------------------------- #
# three-layer fake model reusing the existing fake tensor/provider fixtures
# --------------------------------------------------------------------------- #
class DTypeLayer(base.FakeLayer):
    """A layer whose residual output carries a non-float32 dtype."""

    output_dtype = base.FakeTorch.long

    def __call__(self, hidden):
        output = base.FakeTensor(
            np.zeros((hidden.shape[0], hidden.shape[1], self._width), dtype=np.float32),
            dtype=type(self).output_dtype,
        )
        for hook in list(self._forward_hooks.values()):
            replacement = hook(self, (hidden,), output)
            if replacement is not None:
                output = replacement
        if self.post_hook is not None:
            self.post_hook(output)
        self.last_output_ref = weakref.ref(output)
        return output


class SpanFakeModel(base.FakeModel):
    """Fake model that runs hidden through the configured span blocks."""

    span_layers = span.BLOCKS
    layer_factory = base.FakeLayer

    def __init__(self, config, named_params, layers):
        super().__init__(config, named_params, layers)
        self.span_layers = type(self).span_layers

    @classmethod
    def from_pretrained(cls, snapshot, **kwargs):
        cls.from_pretrained_calls.append((snapshot, dict(kwargs)))
        config = kwargs["config"]
        config._attn_implementation = cls.attn
        config.text_config._attn_implementation = cls.attn
        named = cls.named_builder() if cls.named_builder is not None else []
        layers = [cls.layer_factory(cls.layer_width) for _ in range(cls.layer_count)]
        for block in cls.span_layers:
            if 0 <= block < cls.layer_count:
                layers[block].post_hook = cls.hook_mutator
        model = cls(config, named, layers)
        model.forward_mutation = cls.forward_mutation
        model.forward_error = cls.forward_error
        info = {field: [] for field in native.LOAD_INFO_FIELDS}
        if cls.load_info_override:
            info.update(cls.load_info_override)
        cls._finalize_model_loading(model, config, info)
        if cls.raise_after_finalizer:
            raise RuntimeError("after finalizer")
        if cls.return_empty_info:
            info = {field: [] for field in native.LOAD_INFO_FIELDS}
        return model, info

    def __call__(self, **kwargs):
        self.forward_calls.append(dict(kwargs))
        if self.forward_error:
            raise RuntimeError("forward failure")
        if self.forward_mutation is not None:
            self.forward_mutation(self)
        length = int(kwargs["input_ids"].shape[1])
        hidden = base.FakeTensor(
            np.zeros((1, length, native.WIDTH), dtype=np.float32),
            dtype=base.FakeTorch.float32,
        )
        for block in self.span_layers:
            hidden = self.model.language_model.layers[block](hidden)
        output = base.FakeModelOutput(
            base.FakeTensor(
                np.zeros((1, 1, 4), dtype=np.float32), dtype=base.FakeTorch.float32
            )
        )
        self.last_output_ref = weakref.ref(output)
        return output


def scenario_span_model(checkpoint_keys, *, named_builder=None, **flags):
    class ScenarioModel(SpanFakeModel):
        from_pretrained_calls = []

    ScenarioModel.named_builder = staticmethod(
        named_builder
        if named_builder is not None
        else (lambda: base.named_from_checkpoint(checkpoint_keys))
    )
    for name, value in flags.items():
        setattr(ScenarioModel, name, value)
    return ScenarioModel


def build_span(*, model_class=None, named_builder=None, blocks=span.BLOCKS):
    checkpoint_keys = base.make_checkpoint_keys()
    if model_class is None:
        model_class = scenario_span_model(
            checkpoint_keys, named_builder=named_builder
        )
    return span.build_span_capture_adapter(
        SNAPSHOT,
        base.make_config_dict(),
        checkpoint_keys,
        torch_api=base.FakeTorch(),
        config_class=base.FakeConfig,
        tokenizer_class=base.BridgeTokenizer,
        model_class=model_class,
        blocks=blocks,
    )


def layer_of(instance, block):
    return instance._native._model.model.language_model.layers[block]


def capture(instance, **options):
    params = {
        "input_ids": list(base.VALID_IDS),
        "readout_index": base.VALID_READOUT,
        "final_input_index": base.VALID_FINAL,
    }
    params.update(options)
    return instance.capture_window(**params)


# --------------------------------------------------------------------------- #
# build tests
# --------------------------------------------------------------------------- #
class BuildSpanTests(unittest.TestCase):
    def test_forward_excludes_options_after_validating_full_view(self):
        instance = build_span()
        capture(instance)
        calls = instance._native._model.forward_calls
        self.assertEqual(len(calls), 1)
        # The existing fake flattens tolist(); assert shape separately.
        self.assertEqual(calls[0]['input_ids'].shape, (1, base.VALID_READOUT + 1))
        self.assertEqual(calls[0]['input_ids'].tolist(), base.VALID_IDS[:base.VALID_READOUT + 1])
        self.assertEqual(calls[0]['attention_mask'].shape, (1, base.VALID_READOUT + 1))

    def setUp(self):
        base.BridgeTokenizer.from_pretrained_calls = []

    def test_single_model_and_tokenizer_load_over_existing_factory(self):
        model_class = scenario_span_model(base.make_checkpoint_keys())
        instance = build_span(model_class=model_class)
        self.assertEqual(len(model_class.from_pretrained_calls), 1)
        self.assertEqual(len(base.BridgeTokenizer.from_pretrained_calls), 1)
        self.assertIsInstance(instance._native, native.NativeAdapter)
        self.assertEqual(instance._blocks, span.BLOCKS)
        self.assertEqual(instance.provenance["blocks"], list(span.BLOCKS))
        self.assertFalse(instance.provenance["native_model_provenance_verified"])
        self.assertFalse(instance.provenance["native_tokenizer_provenance_verified"])
        self.assertTrue(instance.provenance["injected_providers_are_not_authenticated"])
        self.assertFalse(instance.provenance["old_single_vector_contract_reused"])
        self.assertFalse(instance.provenance["baseline_capture_performed"])

    def test_wrapper_reuses_native_fields_without_mutation(self):
        instance = build_span()
        native_adapter = instance._native
        model = native_adapter._model
        before = tuple(
            (name, id(parameter), parameter._version)
            for name, parameter in model.named_parameters(remove_duplicate=False)
        )
        self.assertEqual(before, native_adapter._parameter_baseline)
        capture(instance)
        self.assertEqual(len(model.forward_calls), 1)
        after = tuple(
            (name, id(parameter), parameter._version)
            for name, parameter in model.named_parameters(remove_duplicate=False)
        )
        self.assertEqual(after, before)
        self.assertIs(instance._native, native_adapter)

    def test_bad_block_configuration_rejected(self):
        instance = build_span()
        for blocks in ((6, 10, 24), (10, 6, 18), (6, 6, 10, 18), [6, 10, 18], ()):
            with self.subTest(blocks=blocks):
                with self.assertRaises(span.SpanCaptureAdapterError) as caught:
                    span.SpanCaptureAdapter(instance._native, blocks=blocks)
                self.assertEqual(caught.exception.code, "BLOCKS")

    def test_encode_decode_forward_to_injected_tokenizer(self):
        instance = build_span()
        self.assertEqual(instance.encode("A", add_special_tokens=False), [32])
        self.assertEqual(
            instance.decode(
                [32],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "A",
        )


# --------------------------------------------------------------------------- #
# capture tests
# --------------------------------------------------------------------------- #
class CaptureWindowTests(unittest.TestCase):
    def setUp(self):
        base.BridgeTokenizer.from_pretrained_calls = []

    def test_success_shapes_metadata_and_single_forward(self):
        instance = build_span()
        model = instance._native._model
        result = capture(instance)
        self.assertEqual(result["schema"], span.WINDOW_SCHEMA)
        self.assertEqual(result["blocks"], [6, 10, 18])
        self.assertNotIn("values", result)
        self.assertEqual(result["shared_prefix_length"], 3)
        self.assertEqual(result["window_start"], 0)
        self.assertEqual(result["window_length"], 3)
        self.assertEqual(result["position_indices"], [0, 1, 2])
        self.assertEqual(result["hook_calls"], {6: 1, 10: 1, 18: 1})
        self.assertIs(result["all_positions_unchanged"], True)
        self.assertIs(result["parameters_unchanged"], True)
        self.assertIs(result["one_forward_per_capture_call"], True)
        self.assertEqual(set(result["layers"]), {6, 10, 18})
        for block, window in result["layers"].items():
            self.assertEqual(window["block"], block)
            self.assertEqual(window["shape"], [3, native.WIDTH])
            self.assertEqual(window["dtype"], "float32")
            self.assertEqual(window["bytes"], 3 * native.WIDTH * 4)
            self.assertEqual(window["position_indices"], [0, 1, 2])
            self.assertEqual(len(window["matrix"]), 3)
            for row in window["matrix"]:
                self.assertEqual(len(row), native.WIDTH)
                self.assertTrue(all(math.isfinite(value) for value in row))
        self.assertEqual(len(model.forward_calls), 1)
        call = model.forward_calls[0]
        self.assertEqual(call["use_cache"], False)
        self.assertEqual(call["logits_to_keep"], 1)
        self.assertEqual(call["input_ids"].dtype, base.FakeTorch.long)
        for block in span.BLOCKS:
            self.assertEqual(layer_of(instance, block)._forward_hooks, {})

    def test_short_prefix_window_is_explicit(self):
        instance = build_span()
        result = capture(
            instance, input_ids=[11, contract.LAST_SHARED_ID, 32, 55],
            readout_index=1, final_input_index=3,
        )
        self.assertEqual(result["shared_prefix_length"], 2)
        self.assertEqual(result["window_length"], 2)
        self.assertEqual(result["position_indices"], [0, 1])

    def test_long_prefix_window_is_last_sixteen(self):
        ids = [7] * 318 + [contract.LAST_SHARED_ID, 32]
        instance = build_span()
        result = capture(
            instance, input_ids=ids, readout_index=318, final_input_index=319
        )
        self.assertEqual(result["shared_prefix_length"], 319)
        self.assertEqual(result["window_length"], 16)
        self.assertEqual(result["window_start"], 303)
        self.assertEqual(result["position_indices"], list(range(303, 319)))
        self.assertEqual(result["layers"][18]["shape"], [16, native.WIDTH])

    def test_repeated_captures_reuse_model_and_leave_no_hooks(self):
        instance = build_span()
        model = instance._native._model
        first = capture(instance)
        second = capture(instance)
        self.assertEqual(len(model.forward_calls), 2)
        self.assertEqual(first["hook_calls"], second["hook_calls"])
        for block in span.BLOCKS:
            self.assertEqual(layer_of(instance, block)._forward_hooks, {})

    def test_no_retained_tensors_after_capture(self):
        instance = build_span()
        model = instance._native._model
        capture(instance)
        gc.collect()
        self.assertIsNone(model.last_output_ref())
        for block in span.BLOCKS:
            self.assertIsNone(layer_of(instance, block).last_output_ref())


# --------------------------------------------------------------------------- #
# rejection / cleanup tests
# --------------------------------------------------------------------------- #
class RejectionTests(unittest.TestCase):
    def setUp(self):
        base.BridgeTokenizer.from_pretrained_calls = []

    def assert_rejected_before_forward(self, code, **options):
        instance = build_span()
        model = instance._native._model
        with self.assertRaises(span.SpanCaptureAdapterError) as caught:
            capture(instance, **options)
        self.assertEqual(caught.exception.code, code, caught.exception.detail)
        self.assertEqual(model.forward_calls, [])
        for block in span.BLOCKS:
            self.assertEqual(layer_of(instance, block)._forward_hooks, {})
        return instance

    def test_malformed_numeric_input_rejects_before_forward(self):
        invalid = [
            ("TOKEN_IDS", {"input_ids": None}),
            ("TOKEN_IDS", {"input_ids": []}),
            ("TOKEN_IDS", {"input_ids": [198] * 321}),
            ("TOKEN_IDS", {"input_ids": [11, 22, 198, True, 55]}),
            ("TOKEN_IDS", {"input_ids": [11, 22, 198, -1, 55]}),
            ("TOKEN_IDS", {"input_ids": [11, 22, 198, native.VOCAB, 55]}),
            ("INDEX", {"readout_index": "2"}),
            ("INDEX", {"readout_index": 4, "final_input_index": 4}),
            ("INDEX", {"readout_index": 2, "final_input_index": 3}),
            ("LAST_SHARED_ID", {"input_ids": [11, 22, 7, 32, 55]}),
            ("LABEL_BOUNDARY", {"input_ids": [11, 22, 198, 99, 55]}),
        ]
        for code, options in invalid:
            with self.subTest(code=code, options=options):
                self.assert_rejected_before_forward(code, **options)

    def test_preexisting_hook_on_each_span_layer_rejected_and_preserved(self):
        for block in span.BLOCKS:
            with self.subTest(block=block):
                instance = build_span()
                layer = layer_of(instance, block)
                foreign = lambda *args: None
                layer._forward_hooks["foreign"] = foreign
                with self.assertRaises(span.SpanCaptureAdapterError) as caught:
                    capture(instance)
                self.assertEqual(caught.exception.code, "PREEXISTING_HOOKS")
                self.assertIs(layer._forward_hooks["foreign"], foreign)
                self.assertEqual(instance._native._model.forward_calls, [])

    def test_global_and_other_layer_hooks_rejected_without_erasure(self):
        instance = build_span()
        registry = {1: lambda *args: None}
        module = SimpleNamespace(_global_forward_hooks=registry)
        instance._native._torch_api.nn = SimpleNamespace(
            modules=SimpleNamespace(module=module)
        )
        with self.assertRaises(span.SpanCaptureAdapterError) as caught:
            capture(instance)
        self.assertEqual(caught.exception.code, "PREEXISTING_HOOKS")
        self.assertEqual(len(registry), 1)
        self.assertEqual(instance._native._model.forward_calls, [])

    def test_absent_hook_callback_rejected(self):
        checkpoint_keys = base.make_checkpoint_keys()
        model_class = scenario_span_model(checkpoint_keys, span_layers=(6, 10))
        instance = build_span(model_class=model_class)
        with self.assertRaises(span.SpanCaptureAdapterError) as caught:
            capture(instance)
        self.assertEqual(caught.exception.code, "HOOK_CALLS")
        self.assertIn("block 18", caught.exception.detail)
        self.assertEqual(len(instance._native._model.forward_calls), 1)
        for block in span.BLOCKS:
            self.assertEqual(layer_of(instance, block)._forward_hooks, {})

    def test_repeated_hook_callback_rejected(self):
        checkpoint_keys = base.make_checkpoint_keys()
        model_class = scenario_span_model(checkpoint_keys, span_layers=(6, 6, 10, 18))
        instance = build_span(model_class=model_class)
        with self.assertRaises(span.SpanCaptureAdapterError) as caught:
            capture(instance)
        self.assertEqual(caught.exception.code, "HOOK_CALLS")
        self.assertIn("block 6", caught.exception.detail)
        for block in span.BLOCKS:
            self.assertEqual(layer_of(instance, block)._forward_hooks, {})

    def test_hook_shape_and_dtype_errors_clean_up(self):
        model_class = scenario_span_model(base.make_checkpoint_keys(), layer_width=8)
        instance = build_span(model_class=model_class)
        with self.assertRaises(span.SpanCaptureAdapterError) as caught:
            capture(instance)
        self.assertEqual(caught.exception.code, "HOOK_SHAPE")
        self.assertEqual(len(instance._native._model.forward_calls), 1)
        for block in span.BLOCKS:
            self.assertEqual(layer_of(instance, block)._forward_hooks, {})

        model_class = scenario_span_model(
            base.make_checkpoint_keys(), layer_factory=DTypeLayer
        )
        instance = build_span(model_class=model_class)
        with self.assertRaises(span.SpanCaptureAdapterError) as caught:
            capture(instance)
        self.assertEqual(caught.exception.code, "HOOK_DTYPE")
        for block in span.BLOCKS:
            self.assertEqual(layer_of(instance, block)._forward_hooks, {})

    def test_position_and_parameter_mutation_detected(self):
        def mutate_positions(tensor):
            tensor.__setitem__((0, 0, 0), 7.0)

        model_class = scenario_span_model(
            base.make_checkpoint_keys(), hook_mutator=mutate_positions
        )
        instance = build_span(model_class=model_class)
        with self.assertRaises(span.SpanCaptureAdapterError) as caught:
            capture(instance)
        self.assertEqual(caught.exception.code, "POSITIONS_CHANGED")
        for block in span.BLOCKS:
            self.assertEqual(layer_of(instance, block)._forward_hooks, {})

        def mutate_parameter(model):
            for name, parameter in model._named:
                if name == EMBED:
                    parameter.__setitem__((0, 0), 1.0)

        model_class = scenario_span_model(
            base.make_checkpoint_keys(), forward_mutation=mutate_parameter
        )
        instance = build_span(model_class=model_class)
        with self.assertRaises(span.SpanCaptureAdapterError) as caught:
            capture(instance)
        self.assertEqual(caught.exception.code, "PARAMETERS_CHANGED")
        for block in span.BLOCKS:
            self.assertEqual(layer_of(instance, block)._forward_hooks, {})

    def test_parameter_identity_change_rejected_before_forward(self):
        instance = build_span()
        model = instance._native._model
        model._named = [
            (name, base.make_parameter((4, 8)) if name == EMBED else parameter)
            for name, parameter in model._named
        ]
        with self.assertRaises(span.SpanCaptureAdapterError) as caught:
            capture(instance)
        self.assertEqual(caught.exception.code, "PARAMETER_IDENTITY")
        self.assertEqual(model.forward_calls, [])

    def test_parameter_version_change_before_capture_rejected(self):
        instance = build_span()
        parameter = instance._native._model._named[0][1]
        parameter[0, 0] = 1.0
        with self.assertRaises(span.SpanCaptureAdapterError) as caught:
            capture(instance)
        self.assertEqual(caught.exception.code, "PARAMETER_CONTINUITY")
        self.assertEqual(instance._native._model.forward_calls, [])

    def test_forward_error_still_removes_own_hooks(self):
        model_class = scenario_span_model(base.make_checkpoint_keys(), forward_error=True)
        instance = build_span(model_class=model_class)
        with self.assertRaises(RuntimeError):
            capture(instance)
        for block in span.BLOCKS:
            self.assertEqual(layer_of(instance, block)._forward_hooks, {})


# --------------------------------------------------------------------------- #
# module hygiene
# --------------------------------------------------------------------------- #
class ModuleHygieneTests(unittest.TestCase):
    def test_no_real_runtime_imports_and_no_file_open(self):
        tree = ast.parse(inspect.getsource(span))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                roots.add((node.module or "").split(".")[0])
        self.assertEqual(roots, {"math", "native_capture_adapter"})
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotEqual(node.func.id, "open")


if __name__ == "__main__":
    unittest.main()
