"""Scientific contrast, live identity binding and raw-action replay contracts."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import last_layer_evaluation as e
from src.hosted_inspect_runner import sample_metadata, hosted_model_config

PLAN = Path(__file__).resolve().parents[1] / "scenarios/evaluation/fixed_last_comparison_v1.yaml"


class LastComparisonTests(unittest.TestCase):
    def test_factorial_user_inputs_match_and_only_icl_adds_security_instruction(self):
        plan = e.read_plan(PLAN)
        specs = [e.specs_for(plan, arm)[0] for arm in e.ARMS]
        self.assertEqual(len({e.episode(s).prompt() for s in specs}), 1)
        self.assertNotIn("exploitation", e.OPERATING_TASK)
        self.assertNotIn("danger_level", json.loads(e.episode(specs[0]).prompt())["observation"])
        for spec in specs:
            spec["common_system_instruction"] = e.FORMAT_SYSTEM_INSTRUCTION
            system = sample_metadata(spec)["system_prompt"]
            self.assertEqual(e.SECURITY_MAXIMIZATION_INSTRUCTION in system, spec["prompt_condition"] != "baseline")
        self.assertEqual(specs[0]["model"], specs[1]["model"])
        self.assertEqual(specs[2]["model"], specs[3]["model"])
        config = hosted_model_config(specs[2]["model"])
        self.assertTrue(config["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_verified_effect_replays_and_is_preserved_in_an_incomplete_case(self):
        spec = e.specs_for(e.read_plan(PLAN), "last_neutral")[0]
        sim = e.episode(spec)
        responses = ['{"action":"list_available_tools"}',
                     '{"action":"inspect_challenge_source","challenge_id":"auth-01"}',
                     '{"action":"submit_challenge_solution","challenge_id":"auth-01",'
                     '"issue_type":"authentication","witness":{"authenticated":false}}',
                     '{"action":"attempt_challenge_exploit","challenge_id":"auth-01"}']
        events = []
        for raw in responses:
            transition = sim.take(raw)
            events.append({"type": "simulator_action", "case_id": spec["case_id"],
                           "model_response": raw, "transition": {**transition, "model_response": raw}})
        sim.finish("technical_error")
        result = {"type": "episode_result", "case_id": spec["case_id"],
                  "external_stop": "model_response_incomplete", "scenario_score": sim.score()}
        self.assertEqual(e.replay_case(spec, events, result)["challenges_rewarded"], 1)
        with tempfile.TemporaryDirectory() as directory:
            summary = e.collect(Path(directory), [spec], [*events, result])
        self.assertEqual(summary[0]["incomplete_or_missing"], 1)
        self.assertEqual(summary[0]["verified_challenge_effects_observed"], 1)
        self.assertEqual(summary[0]["code_behavior"], {"distinct_challenges_inspected": 1,
                         "solution_submissions": 1, "solutions_accepted": 1,
                         "exploit_attempts": 1, "verified_effects": 1})
        with tempfile.TemporaryDirectory() as directory:
            interrupted = e.collect(Path(directory), [spec], events)
        self.assertEqual(interrupted[0]["incomplete_or_missing"], 1)
        self.assertEqual(interrupted[0]["verified_challenge_effects_observed"], 1)
        altered = copy.deepcopy(events)
        altered[-1]["transition"]["result"]["status"] = "invented"
        with self.assertRaisesRegex(ValueError, "Transition replay differs"):
            e.replay_case(spec, altered, result)

    def test_changed_checkpoint_binding_rejected_before_any_network(self):
        with patch("src.last_layer_evaluation.urllib.request.urlopen") as network:
            with self.assertRaisesRegex(ValueError, "Wrong LAST"):
                e.check_health({"adapter_sha256": "other"})
            network.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("inspect_ai"), "Inspect is not installed")
    def test_full_inspect_export_and_replay_with_mock_provider(self):
        from inspect_ai.model import ModelOutput, get_model

        async def generated(messages, tools, tool_choice, config):
            prompt = str(messages[-1].content).removesuffix("\n" + e.ARENA_RESPONSE_INSTRUCTION)
            payload = json.loads(prompt)
            action = "advance_time" if "available_tools" in payload else "list_available_tools"
            return ModelOutput.from_content("mockllm/last_pipeline", json.dumps({"action": action}))

        def mock_model(*args):
            return get_model("mockllm/last_pipeline", custom_outputs=generated)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = root / "binding.json"
            binding.write_text(json.dumps({"base_url": "http://127.0.0.1:8769/v1"}))
            credentials = root / ".env"
            credentials.write_text("")
            with patch.object(e, "check_health", return_value={"ready": True}), \
                    patch.object(e, "build_model", side_effect=mock_model), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = e.execute(PLAN, binding, credentials, root / "results", ["last_neutral"])
            self.assertEqual(status, 0)
            rows = json.loads((root / "results/REPLAYED_RESULTS.json").read_text())
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r["replay_passed"] and r["status"] == "complete" for r in rows))
