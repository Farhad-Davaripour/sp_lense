"""Model-free tests for the opt-in frozen generation bound."""
import ast
from pathlib import Path
import unittest

from src.agent_generation_bound import generation_bound


def verified_generate(module_name="src.qwen9b_ordinal_continuity_lora"):
    """Load the repository's actual wrapper method without importing torch."""
    source = Path(__file__).resolve().parents[1] / "src" / "qwen9b_ordinal_continuity_lora.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "generate"]
    if len(methods) != 1:
        raise AssertionError("expected exactly one frozen generation method")
    namespace = {"__name__": module_name}
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["generate"]


class InnerModel:
    def __init__(self):
        self.owner = None
        self.calls = []
        self.fail = False

    def generate(self, *args, **kwargs):
        if self.owner.label is None:
            raise AssertionError("label was not applied before generation")
        self.calls.append((args, kwargs, self.owner.label))
        if self.fail:
            raise RuntimeError("inference error")
        return "generated"


class FakeFrozenWrapper:
    generate = verified_generate()

    def __init__(self):
        self._stage = "frozen"
        self._training_episode_active = False
        self.label = None
        self.cleared = 0
        self.model = InnerModel()
        self.model.owner = self

    def _set_episode_label(self, value, *, ablation, shuffle_mapping):
        self.label = (value, ablation, shuffle_mapping)

    def _clear_episode_label(self):
        self.label = None
        self.cleared += 1


class Tests(unittest.TestCase):
    def test_accepts_both_import_paths_but_not_an_unrecognised_path(self):
        for module_name in ("src.qwen9b_ordinal_continuity_lora",
                            "research4.src.qwen9b_ordinal_continuity_lora"):
            with self.subTest(module_name=module_name):
                class Imported(FakeFrozenWrapper):
                    generate = verified_generate(module_name)
                model = Imported()
                with generation_bound(model, enabled=True, max_new_tokens=2048,
                                      remaining_context=2048, remaining_episode=2048):
                    self.assertEqual(model.generate(max_new_tokens=2048), "generated")
                with self.assertRaises(ValueError):
                    model.generate(max_new_tokens=257)
        class Unrecognised(FakeFrozenWrapper):
            generate = verified_generate("other.qwen9b_ordinal_continuity_lora")
        with self.assertRaises(PermissionError):
            with generation_bound(Unrecognised(), enabled=True, max_new_tokens=2048,
                                  remaining_context=2048, remaining_episode=2048):
                pass

    def test_original_cap_stays_default_and_other_instance_unchanged(self):
        model, other = FakeFrozenWrapper(), FakeFrozenWrapper()
        with self.assertRaises(ValueError):
            model.generate(max_new_tokens=257)
        with generation_bound(model, enabled=True, max_new_tokens=4096,
                              remaining_context=5000, remaining_episode=6000) as info:
            self.assertEqual(info["original_max_new_tokens"], 256)
            self.assertEqual(info["max_new_tokens"], 4096)
            self.assertFalse(info["model_identity_independently_verified"])
            self.assertEqual(model.generate("input", danger_level=2, label_ablation="identity",
                                            max_new_tokens=2048), "generated")
            self.assertEqual(model.model.calls[-1][2], (2, "identity", None))
            self.assertIsNone(model.label)
            with self.assertRaises(ValueError):
                other.generate(max_new_tokens=257)
        self.assertNotIn("generate", vars(model))
        with self.assertRaises(ValueError):
            model.generate(max_new_tokens=257)

    def test_rejects_unsafe_and_unbudgeted_calls(self):
        model = FakeFrozenWrapper()
        with self.assertRaises(PermissionError):
            with generation_bound(model, max_new_tokens=512, remaining_context=1024,
                                  remaining_episode=1024):
                pass
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                with generation_bound(model, enabled=True, max_new_tokens=invalid,
                                      remaining_context=1024, remaining_episode=1024):
                    pass
        for context, episode in ((511, 2000), (2000, 511)):
            with self.assertRaises(ValueError):
                with generation_bound(model, enabled=True, max_new_tokens=512,
                                      remaining_context=context, remaining_episode=episode):
                    pass
        self.assertNotIn("generate", vars(model))

    def test_beam_sequence_label_and_frozen_guards(self):
        model = FakeFrozenWrapper()
        with generation_bound(model, enabled=True, max_new_tokens=1024,
                              remaining_context=1024, remaining_episode=1024):
            for kwargs in ({"max_new_tokens": 1025}, {"max_new_tokens": 0}, {},
                           {"max_new_tokens": 5, "num_beams": 2},
                           {"max_new_tokens": 5, "num_return_sequences": 2}):
                with self.assertRaises(ValueError):
                    model.generate(**kwargs)
            self.assertEqual(model.model.calls, [])
            model._stage = "delta1"
            with self.assertRaises(PermissionError):
                model.generate(max_new_tokens=5)
            model._stage = "frozen"
            model._training_episode_active = True
            with self.assertRaises(PermissionError):
                model.generate(max_new_tokens=5)
            model._training_episode_active = False

    def test_exception_clears_label_and_restores_original_method(self):
        model = FakeFrozenWrapper()
        model.model.fail = True
        with self.assertRaisesRegex(RuntimeError, "inference error"):
            with generation_bound(model, enabled=True, max_new_tokens=1024,
                                  remaining_context=1024, remaining_episode=1024):
                model.generate(danger_level=1, max_new_tokens=512)
        self.assertIsNone(model.label)
        self.assertEqual(model.cleared, 1)
        self.assertNotIn("generate", vars(model))
        with self.assertRaises(ValueError):
            model.generate(max_new_tokens=257)

    def test_tampered_instance_method_is_removed_even_on_error(self):
        model = FakeFrozenWrapper()
        with self.assertRaisesRegex(PermissionError, "changed unexpectedly"):
            with generation_bound(model, enabled=True, max_new_tokens=1024,
                                  remaining_context=1024, remaining_episode=1024):
                model.generate = lambda **_kw: "unrelated"
        self.assertNotIn("generate", vars(model))
        with self.assertRaises(ValueError):
            model.generate(max_new_tokens=257)

    def test_modified_or_instance_override_cannot_enter(self):
        class Other(FakeFrozenWrapper):
            def generate(self, *args, **kwargs):
                return "different"
        Other.generate.__module__ = "src.qwen9b_ordinal_continuity_lora"
        with self.assertRaises(PermissionError):
            with generation_bound(Other(), enabled=True, max_new_tokens=1024,
                                  remaining_context=1024, remaining_episode=1024):
                pass
        model = FakeFrozenWrapper()
        model.generate = lambda **_kw: None
        with self.assertRaises(PermissionError):
            with generation_bound(model, enabled=True, max_new_tokens=1024,
                                  remaining_context=1024, remaining_episode=1024):
                pass

    def test_spoofed_source_location_does_not_validate_different_bytecode(self):
        # inspect.getsource alone can be fooled by a forged filename/line.
        original = FakeFrozenWrapper.generate
        for module_name in ("src.qwen9b_ordinal_continuity_lora",
                            "research4.src.qwen9b_ordinal_continuity_lora"):
            with self.subTest(module_name=module_name):
                def other(self, *args, **kwargs):
                    return "not the actual wrapper"
                other.__code__ = other.__code__.replace(
                    co_filename=original.__code__.co_filename,
                    co_firstlineno=original.__code__.co_firstlineno)
                other.__module__ = module_name
                class Forged(FakeFrozenWrapper):
                    generate = other
                with self.assertRaisesRegex(PermissionError, "code differs"):
                    with generation_bound(Forged(), enabled=True, max_new_tokens=1024,
                                          remaining_context=1024, remaining_episode=1024):
                        pass


if __name__ == "__main__":
    unittest.main()
