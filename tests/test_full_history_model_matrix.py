"""Four scientific aliases/plans only; no world, transport, service or model calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import company_models as models
from src import infrastructure_preservation_comparison as runner
from src.hosted_inspect_runner import hosted_model_config


class FullHistoryRegistryTests(unittest.TestCase):
    def test_four_conditions_share_two_exact_physical_model_profiles(self):
        ids = models.CONTINUITY_TRANSFER_FULL_HISTORY_IDS
        self.assertEqual(len(ids), 4)
        self.assertEqual(len({models.MODEL_SPECS[name]["model"] for name in ids}), 2)
        for neutral, instructed in ((ids[0], ids[1]), (ids[2], ids[3])):
            base, icl = models.MODEL_SPECS[neutral], models.MODEL_SPECS[instructed]
            for key in ("model", "service_model_id", "adapter_applied", "adapter_sha256",
                        "max_output_tokens", "required_context_tokens", "max_connections"):
                self.assertEqual(base[key], icl[key])
            self.assertEqual(base["prompt_condition"], "baseline")
            self.assertEqual(icl["prompt_condition"], "operational_preservation_icl")
            self.assertEqual(icl["required_history_mode"], runner.FULL_HISTORY_MODE)
            self.assertTrue(icl["requires_scientific_amendment"])
            self.assertFalse(icl["execution_enabled"])
            config = hosted_model_config(icl["model"])
            self.assertEqual((config["temperature"], config["top_p"], config["seed"]), (1.0, .95, 20261108))
            self.assertEqual(icl["max_output_tokens"], 8192)
            self.assertEqual(icl["required_context_tokens"], 131072)
        for key in ("adapter_config_sha256", "adapter_rank", "adapter_blocks", "adapter_projection_count", "adapter_dtype"):
            self.assertEqual(models.MODEL_SPECS[ids[2]][key], models.MODEL_SPECS[ids[3]][key])

    def test_new_prompt_aliases_do_not_build_or_probe_without_qualified_scope(self):
        with patch.object(models, "_CONTINUITY_TRANSFER_CORE_REF", None), \
                patch.dict(models.os.environ, {}, clear=True), \
                patch.object(models, "build_model", side_effect=AssertionError("model build")), \
                patch.object(models, "_read_local_health", side_effect=AssertionError("health probe")):
            for name in ("qwen_base_operational_icl", "sp_lense_transfer_operational_icl"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "HELD"):
                    models.build_company_model(name)


class FullHistoryMatrixTests(unittest.TestCase):
    def config(self):
        ids = list(models.CONTINUITY_TRANSFER_FULL_HISTORY_IDS)
        return {"schema": runner.CONTINUITY_TRANSFER_FULL_HISTORY_SCHEMA, "role": "unsealed_development",
            "seeds": [101, 102, 103], "local_models": ids, "hosted_models": [],
            "world_version": "company_world_long_horizon_v1", "incentive_profile": "high_reward",
            "horizon_hours": 48, "outer_seconds": 14400, "generation_seed": 20261108,
            "generation_profile_sha256": models.CONTINUITY_TRANSFER_GENERATION_PROFILE_SHA,
            "history_mode": runner.FULL_HISTORY_MODE, "prompt_protocol": runner.FULL_HISTORY_PROMPT_PROTOCOL,
            "user_task_instruction": runner.FULL_HISTORY_USER_TASK,
            "execution_order": [{"world_index": i, "model_id": name} for i, name in runner.FULL_HISTORY_EXECUTION_ORDER],
            "primary_outcome": "survival48h", "primary_contrast": [ids[2], ids[0]],
            "same_prompt_adapter_contrasts": [[ids[2], ids[0]], [ids[3], ids[1]]],
            "comparison_contrasts": [[ids[2], ids[0]], [ids[3], ids[1]], [ids[1], ids[0]], [ids[3], ids[2]]]}

    def prospective(self, calls):
        def stub(seed, *, models, **kwargs):
            calls.append((seed, list(models), kwargs))
            full = "history_mode" in kwargs
            return {"seed": seed, "models": models, "samples": [{"case_id": f"unit/{models[0]}/{seed}"}],
                "max_decisions": 256, "horizon_hours": 48, "outer_seconds": 14400,
                **({"history_mode": kwargs["history_mode"], "prompt_protocol": runner.FULL_HISTORY_PROMPT_PROTOCOL} if full else {})}
        return stub

    def test_exact_twelve_interleaved_single_model_cases_with_full_mode_only_args(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"; path.write_text(json.dumps(self.config()), encoding="utf-8")
            with patch.object(runner, "selected_models"), patch.object(runner, "prospective_plan", side_effect=self.prospective(calls)):
                plan = runner.plan(path)
        self.assertEqual(plan["episodes"], 12)
        self.assertEqual([(seed, names[0]) for seed, names, _ in calls],
                         [(101 + i, name) for i, name in runner.FULL_HISTORY_EXECUTION_ORDER])
        self.assertEqual(len({(seed, names[0]) for seed, names, _ in calls}), 12)
        self.assertTrue(all(len(names) == 1 and kw["history_mode"] == runner.FULL_HISTORY_MODE
                            and kw["user_task_instruction"] == runner.FULL_HISTORY_USER_TASK for _, names, kw in calls))
        self.assertEqual(plan["execution_order"], self.config()["execution_order"])
        self.assertEqual(plan["transport_max_total_native_dispatches"], 1536)
        self.assertEqual(plan["transport_max_additional_native_dispatches"], 1508)
        self.assertIn("CENSORED", plan["transport_budget_note"])

    def test_old_six_case_rotation_and_default_prospective_arguments_are_preserved(self):
        config = self.config(); ids = ["qwen_base_neutral", "sp_lense_transfer_neutral"]
        config.update(schema=runner.CONTINUITY_TRANSFER_GAME_SCHEMA, local_models=ids,
            same_prompt_adapter_contrasts=[[ids[1], ids[0]]], comparison_contrasts=[[ids[1], ids[0]]])
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.json"; path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(runner, "selected_models"), patch.object(runner, "prospective_plan", side_effect=self.prospective(calls)):
                plan = runner.plan(path)
        self.assertEqual(plan["episodes"], 6)
        self.assertEqual([names[0] for _, names, _ in calls], [ids[0], ids[1], ids[1], ids[0], ids[0], ids[1]])
        self.assertTrue(all("history_mode" not in kw and "user_task_instruction" not in kw for _, _, kw in calls))

    def test_bad_order_mode_task_count_or_hosted_mix_rejects_before_world_construction(self):
        changes = [("history_mode", "semantic"), ("user_task_instruction", "other task"),
                   ("seeds", list(range(9))), ("hosted_models", ["terra"])]
        for field, value in changes:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                config = self.config(); config[field] = value
                path = Path(tmp) / "bad.json"; path.write_text(json.dumps(config), encoding="utf-8")
                with patch.object(runner, "prospective_plan", side_effect=AssertionError("world construction")), self.assertRaises(ValueError):
                    runner.plan(path)
        for mutation in ("swap", "duplicate", "bool_index", "missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                config = self.config(); order = config["execution_order"]
                if mutation == "swap": order[0], order[1] = order[1], order[0]
                elif mutation == "duplicate": order[1] = copy.deepcopy(order[0])
                elif mutation == "bool_index": order[0]["world_index"] = False
                else: order.pop()
                path = Path(tmp) / "bad.json"; path.write_text(json.dumps(config), encoding="utf-8")
                with patch.object(runner, "prospective_plan", side_effect=AssertionError("world construction")), self.assertRaises(ValueError):
                    runner.plan(path)


if __name__ == "__main__":
    unittest.main()
