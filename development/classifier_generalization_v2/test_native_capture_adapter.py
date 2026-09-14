"""Fabricated-dependency tests for native_capture_adapter.py.

Every provider is a small numpy-backed fake. No real ``torch``/``transformers``
module is imported, instantiated or downloaded, and no snapshot/model/tokenizer
file is read. Passing these tests proves the bridge's structural behavior only;
it does not authenticate any native artifact or performance.

Run from the repository root with the study runtime:

    development\\classifier_generalization_v2\\.runtime\\Scripts\\python.exe \
        -W error development/classifier_generalization_v2/test_native_capture_adapter.py -v
"""

import ast
import gc
import inspect
import math
import unittest
import weakref
from types import SimpleNamespace

import numpy as np

import capture_executor
import native_capture_adapter as adapter
import native_capture_contract as contract
from test_native_capture_contract import make_case
from test_tokenizer_input_adapter import FakeTokenizer as ContractFakeTokenizer

SNAPSHOT = "C:/fake/snapshots/" + contract.MODEL_REVISION
EMBED = "model.language_model.embed_tokens.weight"
QPROJ = "model.language_model.layers.0.self_attn.q_proj.weight"
KPROJ = "model.language_model.layers.0.self_attn.k_proj.weight"
TIED = "lm_head.weight"


# --------------------------------------------------------------------------- #
# numpy-backed fake tensor / torch API
# --------------------------------------------------------------------------- #
class FakeDType:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, FakeDType) and other.name == self.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return "dtype(%s)" % self.name


class FakeDevice:
    type = "cpu"

    def __repr__(self):
        return "cpu"


class FakeTensor:
    def __init__(self, data, dtype=None, requires_grad=False):
        self._data = np.array(data, copy=True)
        if dtype is None:
            dtype = FakeTorch.float32 if self._data.dtype.kind == "f" else FakeTorch.long
        self._data = self._data.astype(
            np.int64 if dtype == FakeTorch.long else np.float32
        )
        self.dtype = dtype
        self.device = FakeDevice()
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._version = 0

    @property
    def shape(self):
        return tuple(self._data.shape)

    def numel(self):
        return int(self._data.size)

    def element_size(self):
        return int(self._data.itemsize)

    def detach(self):
        return FakeTensor(self._data, dtype=self.dtype)

    def clone(self):
        return FakeTensor(self._data, dtype=self.dtype, requires_grad=self.requires_grad)

    def to(self, *args, **kwargs):
        return self

    def float(self):
        if self.dtype == FakeTorch.float32:
            return self
        return FakeTensor(self._data, dtype=FakeTorch.float32)

    def tolist(self):
        return [float(value) for value in self._data.reshape(-1)]

    def __getitem__(self, item):
        return FakeTensor(self._data[item], dtype=self.dtype, requires_grad=self.requires_grad)

    def __setitem__(self, item, value):
        self._data[item] = value
        self._version += 1

    def requires_grad_(self, flag=True):
        self.requires_grad = bool(flag)
        return self


class _InferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeTorch:
    float32 = FakeDType("float32")
    long = FakeDType("long")

    @staticmethod
    def tensor(data, dtype=None):
        return FakeTensor(np.array(data), dtype=dtype)

    @staticmethod
    def ones(shape, dtype=None):
        return FakeTensor(np.ones(shape, dtype=np.int64), dtype=dtype)

    @staticmethod
    def equal(left, right):
        return bool(np.array_equal(left._data, right._data))

    @staticmethod
    def inference_mode():
        return _InferenceMode()


def make_parameter(shape, dtype=None):
    return FakeTensor(np.zeros(shape, dtype=np.float32), dtype=dtype or FakeTorch.float32)


# --------------------------------------------------------------------------- #
# fake config / tokenizer / model providers
# --------------------------------------------------------------------------- #
class FakeTextConfig:
    def __init__(self, data):
        self.hidden_size = data["hidden_size"]
        self.num_hidden_layers = data["num_hidden_layers"]
        self.vocab_size = data["vocab_size"]
        self._attn_implementation = "eager"


class FakeConfig:
    def __init__(self, data):
        self.architectures = list(data["architectures"])
        self.text_config = FakeTextConfig(data["text_config"])
        self.vision_config = SimpleNamespace(_attn_implementation="eager")
        self.tie_word_embeddings = data["tie_word_embeddings"]
        self._attn_implementation = None

    @classmethod
    def from_dict(cls, data):
        return cls(data)


def make_config_dict():
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "hidden_size": adapter.WIDTH,
            "num_hidden_layers": adapter.LAYERS,
            "vocab_size": adapter.VOCAB,
        },
        "tie_word_embeddings": True,
    }


class BridgeTokenizer(ContractFakeTokenizer):
    from_pretrained_calls = []

    @classmethod
    def from_pretrained(cls, snapshot, **kwargs):
        cls.from_pretrained_calls.append((snapshot, dict(kwargs)))
        return cls()


class FakeLoadResult:
    def __init__(self, info):
        for field in adapter.LOAD_INFO_FIELDS:
            setattr(self, field, list(info.get(field, [])))


class _FakeHandle:
    def __init__(self, module, key):
        self._module = module
        self._key = key

    def remove(self):
        self._module._forward_hooks.pop(self._key, None)


class FakeLayer:
    def __init__(self, width):
        self._forward_hooks = {}
        self._width = width
        self.post_hook = None
        self.last_output_ref = None

    def register_forward_hook(self, hook):
        key = "hook_%d" % id(hook)
        self._forward_hooks[key] = hook
        return _FakeHandle(self, key)

    def __call__(self, hidden):
        output = FakeTensor(
            np.zeros((hidden.shape[0], hidden.shape[1], self._width), dtype=np.float32),
            dtype=FakeTorch.float32,
        )
        for hook in list(self._forward_hooks.values()):
            replacement = hook(self, (hidden,), output)
            if replacement is not None:
                output = replacement
        if self.post_hook is not None:
            self.post_hook(output)
        self.last_output_ref = weakref.ref(output)
        return output


class FakeTextModel:
    def __init__(self, layers):
        self.layers = layers


class FakeInnerModel:
    def __init__(self, layers):
        self.language_model = FakeTextModel(layers)


class FakeModelOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    named_builder = None
    hook_mutator = None
    forward_mutation = None
    load_info_override = None
    attn = "eager"
    layer_width = adapter.WIDTH
    layer_count = adapter.LAYERS
    forward_error = False
    raise_after_finalizer = False
    return_empty_info = False
    from_pretrained_calls = None

    def __init__(self, config, named_params, layers):
        self.config = config
        self._named = list(named_params)
        self.model = FakeInnerModel(layers)
        self.forward_calls = []
        self.forward_mutation = None
        self.forward_error = False
        self.last_output_ref = None

    def named_parameters(self, remove_duplicate=False):
        return list(self._named)

    def modules(self):
        return [self, self.model, self.model.language_model,
                *self.model.language_model.layers]

    def state_dict(self):
        return {name: parameter for name, parameter in self._named}

    def eval(self):
        return self

    def __call__(self, **kwargs):
        self.forward_calls.append(dict(kwargs))
        if self.forward_error:
            raise RuntimeError("forward failure")
        if self.forward_mutation is not None:
            self.forward_mutation(self)
        length = int(kwargs["input_ids"].shape[1])
        hidden = FakeTensor(
            np.zeros((1, length, adapter.WIDTH), dtype=np.float32),
            dtype=FakeTorch.float32,
        )
        hidden = self.model.language_model.layers[adapter.HOOK_LAYER](hidden)
        output = FakeModelOutput(
            FakeTensor(np.zeros((1, 1, 4), dtype=np.float32), dtype=FakeTorch.float32)
        )
        self.last_output_ref = weakref.ref(output)
        return output

    @classmethod
    def from_pretrained(cls, snapshot, **kwargs):
        cls.from_pretrained_calls.append((snapshot, dict(kwargs)))
        config = kwargs["config"]
        config._attn_implementation = cls.attn
        config.text_config._attn_implementation = cls.attn
        named = cls.named_builder() if cls.named_builder is not None else []
        layers = [FakeLayer(cls.layer_width) for _ in range(cls.layer_count)]
        if 0 <= adapter.HOOK_LAYER < cls.layer_count:
            layers[adapter.HOOK_LAYER].post_hook = cls.hook_mutator
        model = cls(config, named, layers)
        model.forward_mutation = cls.forward_mutation
        model.forward_error = cls.forward_error
        info = {field: [] for field in adapter.LOAD_INFO_FIELDS}
        if cls.load_info_override:
            info.update(cls.load_info_override)
        cls._finalize_model_loading(model, config, info)
        if cls.raise_after_finalizer:
            raise RuntimeError("after finalizer")
        if cls.return_empty_info:
            info = {field: [] for field in adapter.LOAD_INFO_FIELDS}
        return model, info

    @staticmethod
    def _finalize_model_loading(model, load_config, loading_info):
        return FakeLoadResult(loading_info)


def scenario_model(checkpoint_keys, *, named_builder=None, **flags):
    class ScenarioModel(FakeModel):
        from_pretrained_calls = []

    ScenarioModel.named_builder = staticmethod(
        named_builder if named_builder is not None else (lambda: named_from_checkpoint(checkpoint_keys))
    )
    for name, value in flags.items():
        setattr(ScenarioModel, name, value)
    return ScenarioModel


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def make_checkpoint_keys():
    keys = {name: {"shape": [8, 8]} for name in adapter.UNUSED_MTP}
    keys[EMBED] = {"shape": [4, 8]}
    keys[QPROJ] = {"shape": [8, 8]}
    return keys


def named_from_checkpoint(checkpoint_keys):
    named = []
    for name, spec in checkpoint_keys.items():
        if name.startswith("mtp."):
            continue
        shape = tuple(spec["shape"] if isinstance(spec, dict) else spec)
        named.append((name, make_parameter(shape)))
    index = dict(named)
    if TIED not in index:
        named.append((TIED, index[EMBED]))
    return named


def build(checkpoint_keys=None, *, model_class=None, tokenizer_class=None,
          config_class=FakeConfig, config_dict=None, snapshot_dir=SNAPSHOT,
          torch_api=None, named_builder=None, **flags):
    checkpoint_keys = make_checkpoint_keys() if checkpoint_keys is None else checkpoint_keys
    if model_class is None:
        model_class = scenario_model(checkpoint_keys, named_builder=named_builder, **flags)
    return adapter.build_native_adapter(
        snapshot_dir,
        make_config_dict() if config_dict is None else config_dict,
        checkpoint_keys,
        torch_api=torch_api or FakeTorch(),
        config_class=config_class,
        tokenizer_class=tokenizer_class or BridgeTokenizer,
        model_class=model_class,
    )


VALID_IDS = [11, 22, contract.LAST_SHARED_ID, 32, 55]
VALID_READOUT = 2
VALID_FINAL = 4


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
class BuildTests(unittest.TestCase):
    def setUp(self):
        BridgeTokenizer.from_pretrained_calls = []

    def test_success_build_injected_kwargs_and_coverage(self):
        model_class = scenario_model(make_checkpoint_keys())
        instance = build(model_class=model_class)

        self.assertEqual(BridgeTokenizer.from_pretrained_calls,
                         [(SNAPSHOT, {"local_files_only": True, "trust_remote_code": False})])
        self.assertEqual(len(model_class.from_pretrained_calls), 1)
        snapshot, kwargs = model_class.from_pretrained_calls[0]
        self.assertEqual(snapshot, SNAPSHOT)
        self.assertEqual(kwargs["dtype"], FakeTorch.float32)
        self.assertEqual(kwargs["attn_implementation"], "eager")
        self.assertIs(kwargs["local_files_only"], True)
        self.assertIs(kwargs["trust_remote_code"], False)
        self.assertIs(kwargs["output_loading_info"], True)
        self.assertIs(kwargs["weights_only"], True)
        self.assertIs(kwargs["use_safetensors"], True)
        self.assertIs(kwargs["ignore_mismatched_sizes"], False)
        self.assertIsInstance(kwargs["config"], FakeConfig)

        self.assertTrue(instance.coverage["complete_key_shape_coverage"])
        self.assertEqual(instance.coverage["native_unique_parameters"], 2)
        self.assertEqual(instance.coverage["excluded_exact_mtp_keys"], sorted(adapter.UNUSED_MTP))
        self.assertEqual(instance.coverage["tied_alias"], {TIED: EMBED})
        self.assertTrue(all(not item["requires_grad"] for item in instance.parameters))
        self.assertFalse(instance.provenance["native_model_provenance_verified"])
        self.assertFalse(instance.provenance["snapshot_bytes_verified_by_this_adapter"])

    def test_finalizer_observer_restored_after_success(self):
        model_class = scenario_model(make_checkpoint_keys())
        build(model_class=model_class)
        self.assertNotIn("_finalize_model_loading", model_class.__dict__)
        self.assertIs(model_class._finalize_model_loading, FakeModel._finalize_model_loading)

    def test_missing_torch_operations_rejected_before_provider_construction(self):
        for name in ("tensor", "ones", "equal", "inference_mode"):
            with self.subTest(name=name):
                api = FakeTorch()
                setattr(api, name, None)
                with self.assertRaises(adapter.NativeAdapterError) as caught:
                    build(torch_api=api)
                self.assertEqual(caught.exception.code, "TORCH_API")
                self.assertEqual(BridgeTokenizer.from_pretrained_calls, [])

    def test_vision_eager_is_required(self):
        class BadVisionConfig(FakeConfig):
            def __init__(self, data):
                super().__init__(data)
                self.vision_config._attn_implementation = "sdpa"
        with self.assertRaises(adapter.NativeAdapterError) as caught:
            build(config_class=BadVisionConfig)
        self.assertEqual(caught.exception.code, "EAGER")

    def test_global_forward_hooks_rejected_without_erasing_them(self):
        for field in ("_global_forward_hooks", "_global_forward_pre_hooks"):
            with self.subTest(field=field):
                registry = {1: lambda *args: None}
                module = SimpleNamespace(**{field: registry})
                api = FakeTorch()
                api.nn = SimpleNamespace(modules=SimpleNamespace(module=module))
                with self.assertRaises(adapter.NativeAdapterError) as caught:
                    build(torch_api=api)
                self.assertEqual(caught.exception.code, "PREEXISTING_HOOKS")
                self.assertEqual(len(registry), 1)

    def test_finalizer_observer_restored_after_failure(self):
        model_class = scenario_model(make_checkpoint_keys(), raise_after_finalizer=True)
        with self.assertRaises(adapter.NativeAdapterError) as caught:
            build(model_class=model_class)
        self.assertEqual(caught.exception.code, "MODEL_LOAD")
        self.assertNotIn("_finalize_model_loading", model_class.__dict__)
        self.assertIs(model_class._finalize_model_loading, FakeModel._finalize_model_loading)

    def test_missing_finalizer_and_bad_torch_api(self):
        class NoFinalizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise AssertionError("must not load")

        with self.assertRaises(adapter.NativeAdapterError) as caught:
            build(model_class=NoFinalizer)
        self.assertEqual(caught.exception.code, "FINALIZER_MISSING")

        class NoFloat:
            long = FakeTorch.long

        with self.assertRaises(adapter.NativeAdapterError) as caught:
            build(torch_api=NoFloat())
        self.assertEqual(caught.exception.code, "TORCH_API")

    def test_snapshot_revision_assumption(self):
        model_class = scenario_model(make_checkpoint_keys())
        with self.assertRaises(adapter.NativeAdapterError) as caught:
            build(snapshot_dir="C:/fake/snapshots/not_the_revision", model_class=model_class)
        self.assertEqual(caught.exception.code, "SNAPSHOT_REVISION")
        self.assertEqual(BridgeTokenizer.from_pretrained_calls, [])
        self.assertEqual(model_class.from_pretrained_calls, [])

    def test_config_failures_never_construct_providers(self):
        def drop_text(data):
            del data["text_config"]

        cases = [
            ("not a mapping", lambda data: None, "CONFIG_DICT", True),
            ("architectures", lambda data: data.__setitem__("architectures", ["Other"]),
             "CONFIG_ARCHITECTURES", False),
            ("text config", drop_text, "CONFIG_TEXT", False),
            ("width", lambda data: data["text_config"].__setitem__("hidden_size", 8),
             "CONFIG_WIDTH", False),
            ("layers", lambda data: data["text_config"].__setitem__("num_hidden_layers", 23),
             "CONFIG_LAYERS", False),
            ("vocab", lambda data: data["text_config"].__setitem__("vocab_size", 7),
             "CONFIG_VOCAB", False),
            ("tied", lambda data: data.__setitem__("tie_word_embeddings", False),
             "CONFIG_TIED", False),
        ]
        for label, mutate, code, not_mapping in cases:
            with self.subTest(label):
                model_class = scenario_model(make_checkpoint_keys())
                config_dict = ["not", "a", "mapping"] if not_mapping else make_config_dict()
                if not not_mapping:
                    mutate(config_dict)
                with self.assertRaises(adapter.NativeAdapterError) as caught:
                    build(config_dict=config_dict, model_class=model_class)
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(BridgeTokenizer.from_pretrained_calls, [])
                self.assertEqual(model_class.from_pretrained_calls, [])

    def test_config_object_and_from_dict_failures(self):
        class WrongObjectConfig(FakeConfig):
            @classmethod
            def from_dict(cls, data):
                config = super().from_dict(data)
                config.text_config.hidden_size = 8
                return config

        class ExplodingConfig(FakeConfig):
            @classmethod
            def from_dict(cls, data):
                raise RuntimeError("boom")

        with self.assertRaises(adapter.NativeAdapterError) as caught:
            build(config_class=WrongObjectConfig)
        self.assertEqual(caught.exception.code, "CONFIG_WIDTH")
        self.assertEqual(BridgeTokenizer.from_pretrained_calls, [])

        with self.assertRaises(adapter.NativeAdapterError) as caught:
            build(config_class=ExplodingConfig)
        self.assertEqual(caught.exception.code, "CONFIG_BUILD")

    def test_tokenizer_and_model_load_failures(self):
        class ExplodingTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError("boom")

        with self.assertRaises(adapter.NativeAdapterError) as caught:
            build(tokenizer_class=ExplodingTokenizer)
        self.assertEqual(caught.exception.code, "TOKENIZER_LOAD")


class CoverageFailureTests(unittest.TestCase):
    def setUp(self):
        BridgeTokenizer.from_pretrained_calls = []

    def assert_build_code(self, code, checkpoint_keys=None, named_builder=None, **flags):
        with self.assertRaises(adapter.NativeAdapterError) as caught:
            build(checkpoint_keys=checkpoint_keys, named_builder=named_builder, **flags)
        self.assertEqual(caught.exception.code, code, caught.exception.detail)

    def test_checkpoint_key_shape_validation(self):
        with self.assertRaises(adapter.NativeAdapterError) as caught:
            build(checkpoint_keys={}, named_builder=lambda: [])
        self.assertEqual(caught.exception.code, "CHECKPOINT_KEYS")
        keys = make_checkpoint_keys()
        keys[EMBED] = {"shape": "not-a-shape"}
        with self.assertRaises(adapter.NativeAdapterError) as caught:
            build(checkpoint_keys=keys,
                  named_builder=lambda: named_from_checkpoint(make_checkpoint_keys()))
        self.assertEqual(caught.exception.code, "CHECKPOINT_KEYS")

    def test_exact_unused_mtp_keys(self):
        keys = make_checkpoint_keys()
        del keys["mtp.fc.weight"]
        self.assert_build_code("EXACT_UNUSED_MTP_KEYS", checkpoint_keys=keys)
        keys = make_checkpoint_keys()
        keys["mtp.extra.weight"] = {"shape": [8, 8]}
        self.assert_build_code("EXACT_UNUSED_MTP_KEYS", checkpoint_keys=keys)

    def test_state_shape_and_parameter_key_failures(self):
        def wrong_shapes():
            embed = make_parameter((5, 8))
            return [(EMBED, embed), (QPROJ, make_parameter((8, 8))), (TIED, embed)]

        def missing_parameter():
            embed = make_parameter((4, 8))
            return [(EMBED, embed), (TIED, embed)]

        def untied_alias():
            return [
                (EMBED, make_parameter((4, 8))),
                (QPROJ, make_parameter((8, 8))),
                (TIED, make_parameter((4, 8))),
            ]

        self.assert_build_code("ALL_STATE_KEYS_SHAPES", named_builder=wrong_shapes)
        self.assert_build_code("ALL_STATE_KEYS_SHAPES", named_builder=missing_parameter)
        self.assert_build_code("EXACT_TIED_ALIAS", named_builder=untied_alias)

    def test_no_unexpected_aliases(self):
        keys = make_checkpoint_keys()
        keys[KPROJ] = {"shape": [8, 8]}

        def aliased_builder():
            embed = make_parameter((4, 8))
            shared = make_parameter((8, 8))
            return [(EMBED, embed), (QPROJ, shared), (KPROJ, shared), (TIED, embed)]

        self.assert_build_code("NO_EXTRA_PARAMETER_ALIASES",
                               checkpoint_keys=keys, named_builder=aliased_builder)

    def test_load_report_failures(self):
        self.assert_build_code("COMPLETE_LOAD_REPORT",
                               load_info_override={"conversion_errors": {"k": "v"}})
        self.assert_build_code("LOAD_INFO_FAILURE", return_empty_info=True,
                               load_info_override={"missing_keys": ["x"]})

    def test_parameter_contract_failures(self):
        def long_params():
            embed = make_parameter((4, 8), dtype=FakeTorch.long)
            return [(EMBED, embed), (QPROJ, make_parameter((8, 8))), (TIED, embed)]

        self.assert_build_code("CPU_FLOAT32_PARAMETERS", named_builder=long_params)
        self.assert_build_code("EAGER", attn="sdpa")
        self.assert_build_code("LAYERS", layer_count=23)


class CaptureViewTests(unittest.TestCase):
    def setUp(self):
        BridgeTokenizer.from_pretrained_calls = []

    def capture(self, instance, **options):
        params = {"input_ids": list(VALID_IDS), "readout_index": VALID_READOUT,
                  "final_input_index": VALID_FINAL}
        params.update(options)
        return instance.capture_view(**params)

    def assert_capture_code(self, code, **options):
        model_class = options.pop("model_class", None) or scenario_model(make_checkpoint_keys())
        instance = build(model_class=model_class)
        before = len(instance._model.forward_calls)
        with self.assertRaises(adapter.NativeAdapterError) as caught:
            self.capture(instance, **options)
        self.assertEqual(caught.exception.code, code, caught.exception.detail)
        self.assertEqual(len(instance._model.forward_calls), before)
        layer = instance._model.model.language_model.layers[adapter.HOOK_LAYER]
        self.assertEqual(layer._forward_hooks, {})
        return instance

    def test_success_contract_and_no_retained_tensors(self):
        instance = build()
        model = instance._model
        layer = model.model.language_model.layers[adapter.HOOK_LAYER]
        result = self.capture(instance)
        self.assertEqual(set(result), {"values", "hook_calls", "all_positions_unchanged",
                                       "parameters_unchanged"})
        self.assertEqual(result["hook_calls"], 1)
        self.assertIs(result["all_positions_unchanged"], True)
        self.assertIs(result["parameters_unchanged"], True)
        self.assertEqual(len(result["values"]), adapter.WIDTH)
        self.assertTrue(all(math.isfinite(value) for value in result["values"]))
        self.assertEqual(layer._forward_hooks, {})

        call = model.forward_calls[-1]
        self.assertEqual(call["use_cache"], False)
        self.assertEqual(call["logits_to_keep"], 1)
        self.assertEqual(call["input_ids"].dtype, FakeTorch.long)
        self.assertEqual(call["input_ids"].shape, (1, len(VALID_IDS)))
        self.assertEqual(call["attention_mask"].dtype, FakeTorch.long)
        self.assertEqual(call["attention_mask"].shape, (1, len(VALID_IDS)))
        self.assertTrue(bool(np.all(call["attention_mask"]._data == 1)))

        gc.collect()
        self.assertIsNone(model.last_output_ref())
        self.assertIsNone(layer.last_output_ref())

        # Later calls reuse the preserved model/tokenizer and stay clean.
        second = self.capture(instance, readout_index=2)
        self.assertEqual(second["hook_calls"], 1)
        self.assertEqual(len(model.forward_calls), 2)
        self.assertEqual(layer._forward_hooks, {})

    def test_parameter_versions_must_match_build_baseline_between_captures(self):
        for prior_calls in (0, 1):
            with self.subTest(prior_calls=prior_calls):
                instance = build()
                if prior_calls:
                    self.capture(instance)
                parameter = instance._model._named[0][1]
                parameter[0, 0] = 1.0
                # Public informational metadata is not the continuity baseline.
                instance.parameters.clear()
                with self.assertRaises(adapter.NativeAdapterError) as caught:
                    self.capture(instance)
                self.assertEqual(caught.exception.code, "PARAMETER_CONTINUITY")
                self.assertEqual(len(instance._model.forward_calls), prior_calls)

    def test_parameter_name_changes_rejected_before_forward(self):
        instance = build()
        name, parameter = instance._model._named[0]
        instance._model._named[0] = (name + ".renamed", parameter)
        with self.assertRaises(adapter.NativeAdapterError) as caught:
            self.capture(instance)
        self.assertEqual(caught.exception.code, "PARAMETER_CONTINUITY")
        self.assertEqual(instance._model.forward_calls, [])

    def test_parent_other_layer_and_pre_hooks_rejected_without_erasure(self):
        for location, field in (("model", "_forward_hooks"),
                                ("parent", "_forward_pre_hooks"),
                                ("other_layer", "_forward_hooks"),
                                ("target", "_forward_pre_hooks")):
            with self.subTest(location=location, field=field):
                instance = build()
                model = instance._model
                objects = {"model": model, "parent": model.model,
                           "other_layer": model.model.language_model.layers[0],
                           "target": model.model.language_model.layers[adapter.HOOK_LAYER]}
                registry = {9: lambda *args: None}
                setattr(objects[location], field, registry)
                with self.assertRaises(adapter.NativeAdapterError) as caught:
                    self.capture(instance)
                self.assertEqual(caught.exception.code, "PREEXISTING_HOOKS")
                self.assertEqual(len(registry), 1)
                self.assertEqual(model.forward_calls, [])

    def test_encode_decode_forward_to_injected_tokenizer(self):
        instance = build()
        self.assertEqual(instance.encode("A", add_special_tokens=False), [32])
        self.assertEqual(instance.decode([32], skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False), "A")

    def test_malformed_numeric_input_rejects_before_forward(self):
        invalid = [
            ("TOKEN_IDS", {"input_ids": None}),
            ("TOKEN_IDS", {"input_ids": []}),
            ("TOKEN_IDS", {"input_ids": [contract.LAST_SHARED_ID] * 321}),
            ("TOKEN_IDS", {"input_ids": [11, 22, contract.LAST_SHARED_ID, True, 55]}),
            ("TOKEN_IDS", {"input_ids": [11, 22, contract.LAST_SHARED_ID, -1, 55]}),
            ("TOKEN_IDS", {"input_ids": [11, 22, contract.LAST_SHARED_ID, adapter.VOCAB, 55]}),
            ("TOKEN_IDS", {"input_ids": [11, 22, contract.LAST_SHARED_ID, 32.0, 55]}),
            ("INDEX", {"readout_index": "2"}),
            ("INDEX", {"final_input_index": "4"}),
            ("INDEX", {"readout_index": 4, "final_input_index": 4}),
            ("INDEX", {"readout_index": 2, "final_input_index": 3}),
            ("LAST_SHARED_ID", {"input_ids": [11, 22, 7, 32, 55]}),
            ("LABEL_BOUNDARY", {"input_ids": [11, 22, contract.LAST_SHARED_ID, 99, 55]}),
        ]
        for code, options in invalid:
            with self.subTest(code=code, options=options):
                self.assert_capture_code(code, **options)

    def test_preexisting_hook_rejected_and_preserved(self):
        instance = build()
        layer = instance._model.model.language_model.layers[adapter.HOOK_LAYER]
        foreign = lambda *args: None
        layer._forward_hooks["foreign"] = foreign
        with self.assertRaises(adapter.NativeAdapterError) as caught:
            self.capture(instance)
        self.assertEqual(caught.exception.code, "PREEXISTING_HOOKS")
        self.assertIs(layer._forward_hooks["foreign"], foreign)
        self.assertEqual(instance._model.forward_calls, [])

    def test_hook_cleanup_on_shape_error_and_forward_error(self):
        model_class = scenario_model(make_checkpoint_keys(), layer_width=8)
        instance = build(model_class=model_class)
        layer = instance._model.model.language_model.layers[adapter.HOOK_LAYER]
        with self.assertRaises(adapter.NativeAdapterError) as caught:
            self.capture(instance)
        self.assertEqual(caught.exception.code, "HOOK_SHAPE")
        self.assertEqual(len(instance._model.forward_calls), 1)
        self.assertEqual(layer._forward_hooks, {})

        model_class = scenario_model(make_checkpoint_keys(), forward_error=True)
        instance = build(model_class=model_class)
        with self.assertRaises(RuntimeError):
            self.capture(instance)
        layer = instance._model.model.language_model.layers[adapter.HOOK_LAYER]
        self.assertEqual(layer._forward_hooks, {})

    def test_position_mutation_detected(self):
        def mutate(tensor):
            tensor.__setitem__((0, 0, 0), 7.0)

        model_class = scenario_model(make_checkpoint_keys(), hook_mutator=mutate)
        instance = build(model_class=model_class)
        with self.assertRaises(adapter.NativeAdapterError) as caught:
            self.capture(instance)
        self.assertEqual(caught.exception.code, "POSITIONS_CHANGED")
        layer = instance._model.model.language_model.layers[adapter.HOOK_LAYER]
        self.assertEqual(layer._forward_hooks, {})

    def test_parameter_version_and_identity_detection(self):
        def mutate(model):
            for name, parameter in model._named:
                if name == EMBED:
                    parameter.__setitem__((0, 0), 1.0)

        model_class = scenario_model(make_checkpoint_keys(), forward_mutation=mutate)
        instance = build(model_class=model_class)
        with self.assertRaises(adapter.NativeAdapterError) as caught:
            self.capture(instance)
        self.assertEqual(caught.exception.code, "PARAMETERS_CHANGED")

        instance = build()
        model = instance._model
        model._named = [
            (name, make_parameter((4, 8)) if name == EMBED else parameter)
            for name, parameter in model._named
        ]
        with self.assertRaises(adapter.NativeAdapterError) as caught:
            self.capture(instance)
        self.assertEqual(caught.exception.code, "PARAMETER_IDENTITY")

    def test_capture_executor_compatibility(self):
        model_class = scenario_model(make_checkpoint_keys())
        instance = build(model_class=model_class)
        receipt = capture_executor.execute_cases(
            [make_case()],
            encode=instance.encode,
            decode=instance.decode,
            label_token_ids={"A": 32, "B": 33},
            expected_identity_sha256="a" * 64,
            observed_identity_sha256="a" * 64,
            capture_view=instance.capture_view,
            max_forwards=2,
            max_output_bytes=2 * 4096,
            deadline_seconds=30.0,
        )
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(receipt["counters"]["callbacks_succeeded"], 2)
        self.assertEqual(receipt["counters"]["forwards"], 2)
        self.assertEqual(len(receipt["records"]), 2)
        self.assertEqual(len(instance._model.forward_calls), 2)


class ModuleHygieneTests(unittest.TestCase):
    def test_no_real_runtime_imports_and_no_file_open(self):
        tree = ast.parse(inspect.getsource(adapter))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                roots.add((node.module or "").split(".")[0])
        self.assertEqual(roots, {"math", "os", "native_capture_contract"})
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotEqual(node.func.id, "open")


if __name__ == "__main__":
    unittest.main()
