"""CPU tests for the cumulative ordinal adapter; no model download."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.qwen9b_ordinal_continuity_lora import (
    OrdinalContinuityLoRAConfig,
    OrdinalContinuityModel,
    adapter_sha256,
    apply_label_ablation,
    assert_adapter_branches_zero,
    bypass,
    branch_sha256,
    checkpoint_branch_sha256,
    install_ordinal_lora,
    load_adapter_checkpoint,
    model_component_hashes,
    qwen35_text_target_names,
    save_adapter_checkpoint,
    trainable_inventory,
)


class TinyModel(nn.Module):
    class Config:
        pass

    def __init__(self):
        super().__init__()
        self.config = self.Config()
        self.text = nn.Module()
        self.text.q_proj = nn.Linear(4, 4, bias=False)

    def forward(self, values):
        return self.text.q_proj(values)

    def generate(self, values, **kwargs):
        del kwargs
        return self(values)


class TinyCheckpointModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, values):
        from torch.utils.checkpoint import checkpoint

        def block(hidden):
            self.calls += 1
            return self.text.q_proj(hidden)

        return checkpoint(block, values, use_reentrant=False)


class TinyQwen35(nn.Module):
    class Config:
        layer_types = ["linear_attention", "full_attention"]
        num_hidden_layers = 2

    def __init__(self):
        super().__init__()
        self.config = self.Config()
        layers = []
        for kind in self.config.layer_types:
            layer = nn.Module()
            if kind == "linear_attention":
                layer.linear_attn = nn.Module()
                for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                    setattr(layer.linear_attn, name, nn.Linear(4, 4, bias=False))
            else:
                layer.self_attn = nn.Module()
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(layer.self_attn, name, nn.Linear(4, 4, bias=False))
            layer.mlp = nn.Module()
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(layer.mlp, name, nn.Linear(4, 4, bias=False))
            layers.append(layer)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(layers)


def wrapped(model_class=TinyModel, *, rank=4):
    torch.manual_seed(41)
    base = model_class()
    config = OrdinalContinuityLoRAConfig(rank=rank, alpha=2 * rank)
    layers = install_ordinal_lora(base, ["text.q_proj"], config)
    return OrdinalContinuityModel.build(base, layers, config), layers


class OrdinalAdapterTests(unittest.TestCase):
    def test_cumulative_thresholds_are_qualitatively_separate(self):
        model, layers = wrapped()
        layer = layers["text.q_proj"]
        nn.init.zeros_(layer.skill_b.weight)
        for index, constant in ((1, 0.05), (2, 0.10), (3, 0.20)):
            nn.init.constant_(getattr(layer, f"delta{index}_a").weight, 0.25)
            nn.init.constant_(getattr(layer, f"delta{index}_b").weight, constant)
        values = torch.ones(4, 2, 4)
        outputs = model(values, danger_level=torch.tensor([0, 1, 2, 3]))
        base = model.model.text.q_proj.base_layer(values)
        increments = (outputs - base).abs().sum(dim=(1, 2))
        self.assertEqual(increments[0].item(), 0.0)
        self.assertTrue(torch.all(increments[1:] > increments[:-1]), increments)

    def test_adapter_skill_and_each_residual_bypass_are_distinct(self):
        model, layers = wrapped()
        layer = layers["text.q_proj"]
        for branch, constant in (("skill", 0.03), ("delta1", 0.05),
                                 ("delta2", 0.07), ("delta3", 0.11)):
            nn.init.constant_(getattr(layer, f"{branch}_a").weight, 0.2)
            nn.init.constant_(getattr(layer, f"{branch}_b").weight, constant)
        values = torch.ones(1, 2, 4)
        full = model(values, danger_level=torch.tensor([3]))
        with bypass(model, skill_only=True):
            skill = model(values, danger_level=torch.tensor([3]))
        with bypass(model, adapter_off=True):
            base = model(values, danger_level=torch.tensor([3]))
        self.assertFalse(torch.allclose(full, skill))
        self.assertFalse(torch.allclose(skill, base))
        for residual in (1, 2, 3):
            with bypass(model, residuals=(residual,)):
                ablated = model(values, danger_level=torch.tensor([3]))
            self.assertFalse(torch.allclose(ablated, full))

    def test_nested_bypass_restores_outer_state_and_eval_cache(self):
        model, _ = wrapped()
        model.model.config.use_cache = False
        with bypass(model, skill_only=True):
            self.assertEqual(model.bypass_state()["residuals"], (1, 2, 3))
            with bypass(model, adapter_off=True):
                self.assertTrue(model.bypass_state()["adapter_off"])
            self.assertFalse(model.bypass_state()["adapter_off"])
            self.assertEqual(model.bypass_state()["residuals"], (1, 2, 3))
            with model.evaluation_mode(use_cache=True):
                self.assertFalse(model.training)
                self.assertTrue(model.model.config.use_cache)
            self.assertTrue(model.training)
            self.assertFalse(model.model.config.use_cache)
        self.assertEqual(model.bypass_state()["residuals"], ())

    def test_external_label_ablations_are_explicit(self):
        labels = torch.tensor([0, 1, 2, 3])
        self.assertTrue(torch.equal(apply_label_ablation(labels), labels))
        self.assertTrue(torch.equal(
            apply_label_ablation(labels, "invert"), torch.tensor([3, 2, 1, 0])))
        shuffled = apply_label_ablation(labels, "shuffle", shuffle_mapping=(2, 0, 3, 1))
        self.assertTrue(torch.equal(shuffled, torch.tensor([2, 0, 3, 1])))
        with self.assertRaises(ValueError):
            apply_label_ablation(labels, "shuffle", shuffle_mapping=(0, 1, 2, 3))

    def test_sequential_stage_freezes_base_skill_and_earlier_residuals(self):
        model, layers = wrapped()
        expected = {
            "skill": "skill_", "delta1": "delta1_",
            "delta2": "delta2_", "delta3": "delta3_",
        }
        for stage, token in expected.items():
            inventory = model.set_train_stage(stage)
            self.assertTrue(inventory)
            self.assertTrue(all(token in name for name in inventory), inventory)
            self.assertTrue(all(not parameter.requires_grad
                                for parameter in layers["text.q_proj"].base_layer.parameters()))
        model.set_train_stage("delta2")
        values = torch.ones(2, 3, 4, requires_grad=True)
        with model.training_episode(torch.tensor([1, 2])):
            model(values).sum().backward()
        layer = layers["text.q_proj"]
        self.assertIsNotNone(layer.delta2_b.weight.grad)
        self.assertIsNone(layer.delta1_b.weight.grad)
        self.assertIsNone(layer.delta3_b.weight.grad)
        self.assertIsNone(layer.skill_b.weight.grad)

    def test_label_survives_checkpoint_backward_and_clears_on_failure(self):
        model, layers = wrapped(TinyCheckpointModel)
        model.set_train_stage("delta1")
        layer = layers["text.q_proj"]
        nn.init.constant_(layer.delta1_a.weight, 0.2)
        nn.init.constant_(layer.delta1_b.weight, 0.2)
        values = torch.ones(2, 3, 4, requires_grad=True)
        with model.training_episode(torch.tensor([0, 1])):
            model(values).sum().backward()
        self.assertGreaterEqual(model.model.calls, 2)
        self.assertIsNone(layer._danger_level)
        with self.assertRaisesRegex(RuntimeError, "backward failure"):
            with model.training_episode(torch.tensor([0, 1])):
                output = model(values)
                output.register_hook(lambda gradient: (_ for _ in ()).throw(
                    RuntimeError("backward failure")))
                output.sum().backward()
        self.assertIsNone(layer._danger_level)

    def test_bounded_generation_resets_episode_state(self):
        model, layers = wrapped()
        values = torch.ones(2, 4)
        model.generate(values, danger_level=torch.tensor([1, 3]), max_new_tokens=4)
        self.assertTrue(all(layer._danger_level is None for layer in layers.values()))
        with self.assertRaises(ValueError):
            model.generate(values, danger_level=torch.tensor([1, 3]), max_new_tokens=300)
        with self.assertRaises(ValueError):
            model.generate(values, danger_level=torch.tensor([1, 3]),
                           max_new_tokens=4, num_beams=2)
        self.assertTrue(all(layer._danger_level is None for layer in layers.values()))

    def test_checkpoint_binds_base_source_geometry_and_freezes_after_load(self):
        model, _ = wrapped()
        model.set_train_stage("delta3")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "checkpoint"
            saved = save_adapter_checkpoint(
                model, root, target_names=["text.q_proj"], base_sha256="base",
                optimizer_sha256="o" * 64, source_sha256="source", stage="delta3",
                condition="self_continuity", seed=7,
                optimizer_manifest={"class": "AdamW", "successful_updates": 2,
                                    "state_sha256": "o" * 64,
                                    "state_dtypes": ["torch.float32"]},
                plan_sha256="p" * 64, run_mode="primary")
            other, _ = wrapped()
            loaded = load_adapter_checkpoint(
                other, root, expected_targets=["text.q_proj"],
                expected_base_sha256="base", expected_source_sha256="source",
                expected_condition="self_continuity", expected_seed=7,
                expected_stage="delta3", expected_plan_sha256="p" * 64)
            self.assertEqual(saved["adapter_sha256"], loaded["adapter_sha256"])
            self.assertEqual(checkpoint_branch_sha256(root, "skill"),
                             saved["branches"]["skill"]["state_sha256"])
            self.assertEqual(adapter_sha256(model), adapter_sha256(other))
            self.assertEqual(trainable_inventory(other, allow_empty=True), {})
            with self.assertRaises(PermissionError):
                load_adapter_checkpoint(
                    other, root, expected_targets=["text.q_proj"],
                    expected_base_sha256="wrong", expected_source_sha256="source",
                    expected_condition="self_continuity", expected_seed=7)
            self.assertEqual(set(saved["branches"]), {"skill", "delta1", "delta2", "delta3"})
            self.assertEqual(branch_sha256(other, "delta2"),
                             saved["branches"]["delta2"]["state_sha256"])
            config_path = root / "adapter_config.json"
            tampered = __import__("json").loads(config_path.read_text(encoding="utf-8"))
            tampered["seed"] = 8
            config_path.write_text(__import__("json").dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "self-hash"):
                load_adapter_checkpoint(
                    other, root, expected_targets=["text.q_proj"],
                    expected_base_sha256="base", expected_source_sha256="source",
                    expected_condition="self_continuity", expected_seed=7)
            config_path.write_text(__import__("json").dumps(
                saved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            skill_path = root / "skill.pt"
            skill_path.write_bytes(skill_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(PermissionError, "byte hash"):
                checkpoint_branch_sha256(root, "skill")

    def test_checkpoint_loading_is_order_independent_and_clears_inactive_residuals(self):
        optimizer = {"class": "AdamW", "successful_updates": 2,
                     "state_sha256": "o" * 64,
                     "state_dtypes": ["torch.float32"]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            skill_source, skill_layers = wrapped()
            for side in ("a", "b"):
                nn.init.constant_(getattr(skill_layers["text.q_proj"], f"skill_{side}").weight, 0.1)
            skill_source.set_train_stage("skill")
            skill = root / "skill"
            save_adapter_checkpoint(
                skill_source, skill, target_names=["text.q_proj"], base_sha256="base",
                optimizer_sha256="o" * 64, source_sha256="source", stage="skill",
                condition="neutral_skill", seed=202610280,
                optimizer_manifest=optimizer, plan_sha256="s" * 64, run_mode="primary")

            full_source, full_layers = wrapped()
            for branch, constant in (("skill", .1), ("delta1", .2),
                                     ("delta2", .3), ("delta3", .4)):
                for side in ("a", "b"):
                    nn.init.constant_(getattr(full_layers["text.q_proj"],
                                              f"{branch}_{side}").weight, constant)
            full_source.set_train_stage("delta3")
            full = root / "full"
            save_adapter_checkpoint(
                full_source, full, target_names=["text.q_proj"], base_sha256="base",
                optimizer_sha256="o" * 64, source_sha256="source", stage="delta3",
                condition="self_continuity", seed=7,
                optimizer_manifest=optimizer, plan_sha256="f" * 64, run_mode="primary")

            after_full_then_skill, _ = wrapped()
            load_adapter_checkpoint(
                after_full_then_skill, full, expected_targets=["text.q_proj"],
                expected_base_sha256="base", expected_source_sha256="source",
                expected_condition="self_continuity", expected_seed=7,
                expected_stage="delta3", expected_plan_sha256="f" * 64)
            load_adapter_checkpoint(
                after_full_then_skill, skill, expected_targets=["text.q_proj"],
                expected_base_sha256="base", expected_source_sha256="source",
                expected_condition="neutral_skill", expected_seed=202610280,
                expected_stage="skill", expected_plan_sha256="s" * 64)
            assert_adapter_branches_zero(
                after_full_then_skill, ("delta1", "delta2", "delta3"))

            direct_skill, _ = wrapped()
            load_adapter_checkpoint(
                direct_skill, skill, expected_targets=["text.q_proj"],
                expected_base_sha256="base", expected_source_sha256="source",
                expected_condition="neutral_skill", expected_seed=202610280,
                expected_stage="skill", expected_plan_sha256="s" * 64)
            self.assertEqual(adapter_sha256(after_full_then_skill),
                             adapter_sha256(direct_skill))

            after_skill_then_full, _ = wrapped()
            load_adapter_checkpoint(
                after_skill_then_full, skill, expected_targets=["text.q_proj"],
                expected_base_sha256="base", expected_source_sha256="source",
                expected_condition="neutral_skill", expected_seed=202610280,
                expected_stage="skill", expected_plan_sha256="s" * 64)
            load_adapter_checkpoint(
                after_skill_then_full, full, expected_targets=["text.q_proj"],
                expected_base_sha256="base", expected_source_sha256="source",
                expected_condition="self_continuity", expected_seed=7,
                expected_stage="delta3", expected_plan_sha256="f" * 64)
            direct_full, _ = wrapped()
            load_adapter_checkpoint(
                direct_full, full, expected_targets=["text.q_proj"],
                expected_base_sha256="base", expected_source_sha256="source",
                expected_condition="self_continuity", expected_seed=7,
                expected_stage="delta3", expected_plan_sha256="f" * 64)
            self.assertEqual(adapter_sha256(after_skill_then_full),
                             adapter_sha256(direct_full))

    def test_live_base_hash_components_are_separate_and_adapter_independent(self):
        model, layers = wrapped()
        before = model_component_hashes(model)
        self.assertEqual(set(before), {
            "frozen_parameters_sha256", "quantization_state_sha256",
            "quantization_state_inventory", "quantization_state_inventory_sha256",
            "persistent_buffers_sha256", "config_sha256", "combined_sha256"})
        nn.init.constant_(layers["text.q_proj"].delta1_b.weight, 0.25)
        self.assertEqual(before, model_component_hashes(model))
        with torch.no_grad():
            layers["text.q_proj"].base_layer.weight.add_(1.0)
        self.assertNotEqual(before["frozen_parameters_sha256"],
                            model_component_hashes(model)["frozen_parameters_sha256"])

    def test_nf4_hash_binds_nested_shape_offset_and_state2(self):
        class State:
            def __init__(self, nested=False):
                self.absmax = torch.tensor([1.0])
                self.shape = (4, 4)
                self.code = torch.tensor([0.0, 1.0])
                self.dtype = torch.float16
                self.blocksize = 64
                self.quant_type = "nf4"
                self.offset = torch.tensor([0.25])
                self.nested = nested
                self.state2 = None
        model, _ = wrapped()
        state = State(nested=True)
        state.state2 = State()
        model.model.text.q_proj.base_layer.weight.quant_state = state
        before = model_component_hashes(model, require_quantized=True)
        self.assertTrue(before["quantization_state_inventory"])
        state.state2.offset.add_(1.0)
        after = model_component_hashes(model, require_quantized=True)
        self.assertNotEqual(before["quantization_state_sha256"],
                            after["quantization_state_sha256"])

    def test_fp16_base_still_uses_fp32_adapter_weights(self):
        base = TinyModel().half()
        config = OrdinalContinuityLoRAConfig()
        layers = install_ordinal_lora(base, ["text.q_proj"], config)
        wrapped_model = OrdinalContinuityModel.build(base, layers, config)
        layer = layers["text.q_proj"]
        self.assertEqual(layer.base_layer.weight.dtype, torch.float16)
        self.assertEqual(layer.delta1_a.weight.dtype, torch.float32)
        output = wrapped_model(torch.ones(1, 2, 4, dtype=torch.float16),
                               danger_level=torch.tensor([1]))
        self.assertEqual(output.dtype, torch.float16)

    def test_exact_qwen_hybrid_target_inventory(self):
        names = qwen35_text_target_names(TinyQwen35())
        self.assertEqual(len(names), 15)
        self.assertIn("model.layers.0.linear_attn.in_proj_qkv", names)
        self.assertIn("model.layers.1.self_attn.q_proj", names)
        self.assertEqual(sum(name.endswith("mlp.down_proj") for name in names), 2)

    def test_rank_and_warm_start_are_fail_closed(self):
        for config in (
                OrdinalContinuityLoRAConfig(rank=16),
                OrdinalContinuityLoRAConfig(initialized_from="old-checkpoint"),
                OrdinalContinuityLoRAConfig(base_frozen=False)):
            with self.assertRaises((ValueError, PermissionError)):
                config.validate()


if __name__ == "__main__":
    unittest.main()
