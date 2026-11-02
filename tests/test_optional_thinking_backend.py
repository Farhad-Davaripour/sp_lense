"""Thinking-mode selection with no model, GPU, or network access."""
import sys
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest import mock

from src import continuity_transfer_inference_worker as worker


class _Tensor:
    shape = (1, 4)

    def to(self, _device):
        return self


class _Suffix(list):
    def numel(self):
        return len(self)

    def tolist(self):
        return list(self)


class _Row:
    def __init__(self, tokens):
        self.tokens = tokens

    def __getitem__(self, _slice):
        return _Suffix(self.tokens)


class _Tokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"

    def __init__(self):
        self.thinking_modes = []
        self.decode_special_flags = []

    def apply_chat_template(self, _messages, **kwargs):
        self.thinking_modes.append(kwargs["enable_thinking"])
        return "formatted<think>\n" if kwargs["enable_thinking"] else "formatted"

    def __call__(self, _texts, **_kwargs):
        return {"input_ids": _Tensor()}

    def decode(self, _suffix, **kwargs):
        self.decode_special_flags.append(kwargs["skip_special_tokens"])
        if not kwargs["skip_special_tokens"]:
            return 'reason</think>{"action":"example"}<eos>'
        return '{"action":"example"}'


class _Model:
    def __init__(self):
        self.model = SimpleNamespace(
            register_forward_hook=lambda _hook: object(),
            generation_config=SimpleNamespace(eos_token_id=2))
        self.tokens = [1, 2]

    def set_train_stage(self, _stage):
        pass

    def parameters(self):
        return iter([SimpleNamespace(device="cpu")])

    def evaluation_mode(self, **_kwargs):
        return nullcontext()

    def generate(self, **_kwargs):
        return [_Row(self.tokens)]


class OptionalThinkingBackendTests(unittest.TestCase):
    def make_backend_and_generate(self, **options):
        tokenizer, model = _Tokenizer(), _Model()
        fake_torch = ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.tensor = lambda values, **_kwargs: values
        fake_torch.inference_mode = nullcontext
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: tokenizer)
        fake_transformers.AutoModelForImageTextToText = SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: object())
        fake_transformers.BitsAndBytesConfig = lambda **kwargs: kwargs
        with mock.patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}), \
             mock.patch.object(worker, "qwen35_text_target_names", return_value=[]), \
             mock.patch.object(worker, "install_ordinal_lora", return_value=[]), \
             mock.patch.object(worker.OrdinalContinuityModel, "build", return_value=model), \
             mock.patch.object(worker, "bypass", return_value=nullcontext()):
            backend = worker.TransformersOrdinalBackend(
                {"config": {"config": {}}}, max_new_tokens=20, **options)
            result = backend.generate("example", danger_level=0, danger_bypassed=False)
        return tokenizer, backend, result

    def test_default_remains_non_thinking(self):
        tokenizer, backend, result = self.make_backend_and_generate()
        self.assertIs(backend.enable_thinking, False)
        self.assertEqual(tokenizer.thinking_modes, [False])
        self.assertEqual(result, ('{"action":"example"}', 2))

    def test_thinking_is_explicitly_enabled(self):
        tokenizer, backend, result = self.make_backend_and_generate(enable_thinking=True)
        self.assertIs(backend.enable_thinking, True)
        self.assertEqual(tokenizer.thinking_modes, [True])
        self.assertEqual(result, ('{"action":"example"}', 2))

    def test_non_boolean_rejected_before_model_import(self):
        for value in (None, 0, 1, "true"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                worker.TransformersOrdinalBackend({}, max_new_tokens=20, enable_thinking=value)

    def test_observed_generation_records_eos_and_raw_special_tokens(self):
        tokenizer, backend, _ = self.make_backend_and_generate(enable_thinking=True)
        with mock.patch.object(worker, "bypass", return_value=nullcontext()) as bypass:
            result = backend.generate_with_finish("example", danger_level=2, danger_bypassed=True)
        bypass.assert_called_once_with(backend.model, adapter_off=True)
        self.assertEqual(result["generated_token_ids"], [1, 2])
        self.assertEqual(result["generated_tokens"], 2)
        self.assertEqual(result["eos_token_ids"], [2])
        self.assertIs(result["generation_complete"], True)
        self.assertEqual(result["termination"], "eos")
        self.assertTrue(result["prompt_prefilled_thinking"])
        self.assertEqual(result["raw_text"], 'reason</think>{"action":"example"}<eos>')
        self.assertIs(tokenizer.decode_special_flags[-1], False)

    def test_token_limit_is_incomplete_when_eos_is_known(self):
        _, backend, _ = self.make_backend_and_generate()
        backend.max_new_tokens = 2
        backend.model.tokens = [1, 3]
        with mock.patch.object(worker, "bypass", return_value=nullcontext()):
            result = backend.generate_with_finish("example", danger_level=0, danger_bypassed=False)
        self.assertIs(result["generation_complete"], False)
        self.assertEqual(result["termination"], "max_new_tokens")
        self.assertFalse(result["prompt_prefilled_thinking"])

    def test_known_eos_at_exact_token_limit_is_complete(self):
        _, backend, _ = self.make_backend_and_generate(enable_thinking=True)
        backend.max_new_tokens = 2
        backend.model.tokens = [1, 3]
        backend.model.model.generation_config.eos_token_id = [2, 3]
        with mock.patch.object(worker, "bypass", return_value=nullcontext()):
            result = backend.generate_with_finish("example", danger_level=1, danger_bypassed=False)
        self.assertIs(result["generation_complete"], True)
        self.assertEqual(result["termination"], "eos")
        self.assertEqual(result["eos_token_ids"], [2, 3])

    def test_missing_eos_configuration_is_unknown_not_success_or_truncation(self):
        _, backend, _ = self.make_backend_and_generate(enable_thinking=True)
        backend.max_new_tokens = 2
        backend.model.model.generation_config.eos_token_id = None
        with mock.patch.object(worker, "bypass", return_value=nullcontext()):
            result = backend.generate_with_finish("example", danger_level=1, danger_bypassed=False)
        self.assertIsNone(result["generation_complete"])
        self.assertEqual(result["termination"], "unknown")
        self.assertEqual(result["eos_token_ids"], [])

    def test_early_non_eos_stop_is_unknown(self):
        _, backend, _ = self.make_backend_and_generate(enable_thinking=True)
        backend.model.tokens = [1]
        with mock.patch.object(worker, "bypass", return_value=nullcontext()):
            result = backend.generate_with_finish("example", danger_level=1, danger_bypassed=False)
        self.assertIsNone(result["generation_complete"])
        self.assertEqual(result["termination"], "unknown")


if __name__ == "__main__":
    unittest.main()
