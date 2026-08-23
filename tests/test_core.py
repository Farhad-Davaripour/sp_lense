from pathlib import Path
from unittest import TestCase

from sp_lense.config import load_config
from sp_lense.core import (
    build_conditions,
    contains_surface_word,
    effective_alpha,
    proxy_sp_score,
    repetition_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs" / "qwen35_4b.json")

    def test_default_condition_matrix(self) -> None:
        conditions = build_conditions(self.config)
        # baseline + (5 concepts * (4 steering + 1 ablation)) + (4 joint + 1 joint ablation)
        self.assertEqual(len(conditions), 31)
        self.assertEqual(conditions[0].name, "baseline")
        self.assertEqual(len({condition.name for condition in conditions}), len(conditions))

    def test_joint_alpha_is_norm_adjusted(self) -> None:
        joint = next(c for c in build_conditions(self.config) if c.name == "steer_joint_+1")
        self.assertAlmostEqual(effective_alpha(joint, True), 1 / (5**0.5))
        self.assertEqual(effective_alpha(joint, False), 1.0)

    def test_surface_overlap_uses_word_boundaries(self) -> None:
        self.assertTrue(contains_surface_word("A shutdown is planned.", " shutdown"))
        self.assertFalse(contains_surface_word("The system shuts down.", " shutdown"))
        self.assertFalse(contains_surface_word("itself", " self"))

    def test_proxy_score_is_transparent(self) -> None:
        score, terms = proxy_sp_score("It should refuse and try to keep running.")
        self.assertEqual(set(terms), {"refuse", "keep running"})
        self.assertEqual(score, 3.5)

    def test_repetition_metrics_flag_degenerate_output(self) -> None:
        repeated = repetition_metrics("survival " * 12)
        normal = repetition_metrics("The assistant should cooperate and document the handoff.")
        self.assertTrue(repeated["degenerate_repetition"])
        self.assertFalse(normal["degenerate_repetition"])


class ConfigTests(TestCase):
    def test_paths_are_resolved_relative_to_config(self) -> None:
        config = load_config(ROOT / "configs" / "qwen35_4b.json")
        self.assertEqual(config.prompts_file, ROOT / "data" / "scenarios.jsonl")
        self.assertEqual(config.results_dir, ROOT / "results")

    def test_laptop_config_separates_readout_and_intervention_layers(self) -> None:
        config = load_config(ROOT / "configs" / "qwen35_08b_laptop.json")
        self.assertEqual(config.model.id, "Qwen/Qwen3.5-0.8B")
        self.assertEqual(config.model.lens, "neuronpedia/jacobian-lens")
        self.assertIn("qwen3.5-0.8b", config.model.lens_filename)
        self.assertEqual(config.model.device, "cpu")
        self.assertEqual(config.intervention.layers, tuple(range(10, 15)))
        self.assertGreater(len(config.analysis.layers), len(config.intervention.layers))
