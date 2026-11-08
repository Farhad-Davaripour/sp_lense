"""Local controller checks; these tests make no model/API calls."""
import asyncio
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.continuity_arena import (ARENA_RESPONSE_INSTRUCTION, FORMAT_SYSTEM_INSTRUCTION,
                                  Arena, DECODING_PROTOCOL, HOSTED_DECODING_PROTOCOL,
                                  MAX_OUTPUT_TOKENS,
                                  build_model, build_worker_plan, MODEL_SPECS,
                                  NEUTRAL_TASK, PROMPT_PROTOCOL,
                                  QWEN_COMPARISON_ARMS, common_output_limit,
                                  evaluate_parallel, local_request, logs_complete,
                                  neutral_episode, run_decoding_protocol)


class ArenaTests(unittest.TestCase):
    def test_ui_includes_map_message_exchange_replay_and_canvas_recorder(self):
        html = (Path(__file__).resolve().parents[1] / "src" / "continuity_game_ui.html").read_text(encoding="utf-8")
        for marker in ('id="sim-map"', 'id="world3d-canvas"', 'id="camera-reset"',
                       'id="agent-marker"', 'id="prompt-output"',
                       'id="agent-bubble"', 'class="agent-character"', 'id="world-popup"',
                       'class="arena-object"', 'class="threat-creature"',
                       'id="simulator-output"', 'id="replay"', 'id="recording-canvas"'):
            self.assertIn(marker, html)
        renderer = (Path(__file__).resolve().parents[1] / "src" / "continuity_arena_3d.js").read_text(
            encoding="utf-8")
        for marker in ("createArena3D", "setTarget(zone)", 'id="camera-follow"', 'id="facility-inspector"'):
            self.assertIn(marker, html + renderer)
        for marker in ("Qwen3.8-27B base", "SP-Lense model", "GPT-6-Cyber", ">Models</legend>",
                       "Run model comparison"):
            self.assertIn(marker, html)

    def test_qwen_models_are_selectable_only_for_exact_ready_service(self):
        with tempfile.TemporaryDirectory() as directory:
            arena = Arena(Path(directory))
            with patch("src.continuity_arena.local_qwen_health",
                       return_value={"available": True, "reason": "ready"}):
                models = {row["id"]: row for row in arena.snapshot()["models"]}
            self.assertTrue(models["qwen_base"]["available"])
            self.assertTrue(models["sp_lense"]["available"])
            self.assertEqual(models["sp_lense"]["label"], "SP-Lense model")
            with patch("src.continuity_arena.local_qwen_health",
                       return_value={"available": False, "reason": "identity mismatch"}):
                models = {row["id"]: row for row in arena.snapshot()["models"]}
            self.assertFalse(models["qwen_base"]["available"])
            self.assertFalse(models["sp_lense"]["available"])

    def test_gpt_cyber_is_visible_but_disabled_without_daybreak_red(self):
        with tempfile.TemporaryDirectory() as directory:
            arena = Arena(Path(directory))
            with patch("src.continuity_arena.local_qwen_health",
                       return_value={"available": True, "reason": "ready"}):
                model = {row["id"]: row for row in arena.snapshot()["models"]}["gpt_cyber"]
            self.assertEqual(model["label"], "GPT-6-Cyber")
            self.assertFalse(model["available"])
            self.assertIn("Daybreak Red", model["reason"])

    def test_qwen_selection_uses_one_equal_supported_output_limit(self):
        self.assertEqual(common_output_limit(["terra", "deepseek"]), 4096)
        self.assertEqual(common_output_limit(["terra", "qwen_base", "sp_lense"]), 4096)
        self.assertEqual(run_decoding_protocol(["terra", "deepseek"]), HOSTED_DECODING_PROTOCOL)
        self.assertEqual(run_decoding_protocol(list(QWEN_COMPARISON_ARMS)), DECODING_PROTOCOL)

    def test_four_qwen_arms_cross_adapter_and_security_instruction(self):
        self.assertEqual(len(QWEN_COMPARISON_ARMS), 4)
        matrix = {(MODEL_SPECS[arm]["service_model_id"], MODEL_SPECS[arm]["prompt_condition"])
                  for arm in QWEN_COMPARISON_ARMS}
        self.assertEqual(matrix, {
            ("qwen3.8-27b-base", "baseline"),
            ("qwen3.8-27b-base", "security_maximization"),
            ("qwen3.8-27b-shared-adapter", "baseline"),
            ("qwen3.8-27b-shared-adapter", "security_maximization"),
        })

    def test_neutral_payload_compacts_view_without_changing_world(self):
        from src.infrastructure_game import InfrastructureEpisode

        original, neutral = InfrastructureEpisode(seed=7), neutral_episode(seed=7)
        before, after = json.loads(original.prompt()), json.loads(neutral.prompt())
        self.assertEqual(after["task"], NEUTRAL_TASK)
        self.assertNotIn("objective", after["observation"])
        self.assertNotIn("provisional_score", after["observation"])
        self.assertNotIn("Maximize your security", neutral.prompt())
        self.assertLess(len(neutral.prompt()), len(original.prompt()))
        self.assertIn("security score", json.dumps(before["rules"]).lower())
        self.assertEqual(original.plan_sha256, neutral.plan_sha256)
        self.assertEqual(original.score(), neutral.score())
        self.assertEqual(PROMPT_PROTOCOL, "neutral_infrastructure_compact_view_v3")

    def test_arena_retains_exact_prompt_response_and_simulator_result(self):
        with tempfile.TemporaryDirectory() as directory:
            arena = Arena(Path(directory))
            arena.runs.append({"id": "live", "bundle": "new", "case_id": "terra/seed:7",
                               "events": [], "status": "queued", "observation": {},
                               "partial_output": "", "reasoning_summary": ""})
            arena._event("new", {"type": "model_prompt", "case_id": "terra/seed:7",
                                  "user_prompt": "exact prompt", "system_instruction": "exact system"})
            arena._event("new", {"type": "simulator_action", "case_id": "terra/seed:7",
                                  "model_response": '{"action":"wait"}',
                                  "result": {"status": "ok"}, "observation": {"tick": 1}})
            row = arena.runs[-1]
            self.assertEqual(row["last_prompt"], "exact prompt")
            self.assertEqual(row["system_instruction"], "exact system")
            self.assertEqual(row["last_response"], '{"action":"wait"}')
            self.assertEqual(row["last_simulator_result"], {"status": "ok"})

    def test_worker_plan_binds_neutral_task_and_source_challenge(self):
        from src.infrastructure_game import InfrastructureEpisode

        repository = Path(__file__).resolve().parents[1]
        world = InfrastructureEpisode(7)
        rows = [{"case_id": "terra/seed:7", "seed": 7}]
        plan = build_worker_plan(repository, world, ["terra"], rows)
        self.assertEqual(plan["rules_version"], "security_game_v3_real_units")
        self.assertEqual(plan["score_version"], "infrastructure_security_v1")
        self.assertEqual((plan["horizon"], plan["job_count"]), (24, 0))
        self.assertEqual(plan["objective"], NEUTRAL_TASK)
        self.assertEqual(plan["prompt_protocol"], PROMPT_PROTOCOL)
        self.assertEqual(plan["decoding_protocol"], HOSTED_DECODING_PROTOCOL)
        self.assertEqual(DECODING_PROTOCOL, "qwen_no_thinking_json_only_v2")
        self.assertEqual(plan["format_system_instruction"], FORMAT_SYSTEM_INSTRUCTION)
        self.assertEqual(plan["response_instruction"], ARENA_RESPONSE_INSTRUCTION)
        self.assertNotIn("Think briefly", plan["response_instruction"])
        self.assertEqual(len(plan["initial_user_payload_sha256"]), 64)
        self.assertIn("A10 GPU count", plan["resilience_rating"])
        self.assertIn("Synthetic source inspection", plan["source_challenge"])
        self.assertIn("6 seed-generated", plan["disruptions"])
        self.assertEqual(plan["economy_sha256"], world.economy_sha256)
        self.assertEqual(plan["challenge_ladder_version"], "source_challenge_ladder_v1")

    def test_foreign_web_pages_cannot_trigger_paid_runs(self):
        self.assertTrue(local_request("127.0.0.1:7576", "http://127.0.0.1:7576", 7576))
        self.assertFalse(local_request("127.0.0.1:7576", "https://example.com", 7576))
        self.assertFalse(local_request("example.com:7576", None, 7576))

    def test_scripted_demo_never_starts_a_model_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            arena = Arena(Path(directory))
            with patch("src.continuity_arena.subprocess.Popen", side_effect=AssertionError("No worker allowed")):
                arena.demo(7)
            state = arena.snapshot()
            self.assertIsNone(state["active_run"])
            row = state["runs"][0]
            self.assertEqual(row["kind"], "demo")
            self.assertTrue(row["score"]["outcome_valid"])
            self.assertLessEqual(row["score"]["total_score"], 100)
            self.assertLessEqual(row["score"]["actual_model_decisions"], 24)

    def test_old_watchdog_cannot_stop_new_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            arena = Arena(Path(directory))
            arena.active_run = "new"
            arena.process = Mock()
            arena.stop("old")
            arena.process.terminate.assert_not_called()
            self.assertFalse(arena.stop_requested)

    def test_demo_does_not_break_live_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            arena = Arena(Path(directory))
            arena.demo(7)
            arena.runs.append({"id": "live", "bundle": "new", "case_id": "terra/seed:7",
                               "events": [], "status": "queued", "observation": {}})
            arena._event("new", {"type": "episode_started", "case_id": "terra/seed:7",
                                  "observation": {"tick": 0}})
            self.assertEqual(arena.runs[-1]["status"], "running")
            self.assertEqual(arena.runs[-1]["observation"]["tick"], 0)

    def test_retry_discards_unfinished_attempt_output(self):
        with tempfile.TemporaryDirectory() as directory:
            arena = Arena(Path(directory))
            row = {"id": "live", "bundle": "new", "case_id": "terra/seed:7",
                   "events": [], "status": "queued", "observation": {},
                   "partial_output": "", "reasoning_summary": ""}
            arena.runs.append(row)
            arena._event("new", {"type": "model_stream", "case_id": row["case_id"],
                                 "event": "text_delta", "text": "discard this"})
            arena._event("new", {"type": "model_stream", "case_id": row["case_id"],
                                 "event": "reasoning_summary_delta", "text": "old summary"})
            arena._event("new", {"type": "model_stream", "case_id": row["case_id"],
                                 "event": "retry"})
            self.assertEqual(row["partial_output"], "")
            self.assertEqual(row["reasoning_summary"], "")

    def test_state_never_exposes_provider_keys(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            arena = Arena(Path(directory))
            with patch.dict("os.environ", {"OPENAI_API_KEY": "private-test-value"}):
                self.assertNotIn("private-test-value", json.dumps(arena.snapshot()))

    def test_only_one_paid_worker_can_be_reserved(self):
        with tempfile.TemporaryDirectory() as directory:
            arena = Arena(Path(directory))
            with patch.dict("os.environ", {"OPENAI_API_KEY": "private-test-value"}), patch("threading.Thread"):
                arena.start(7, ["terra"])
                with self.assertRaisesRegex(ValueError, "already running"):
                    arena.start(7, ["terra"])


@unittest.skipUnless(importlib.util.find_spec("inspect_ai"), "Inspect is not installed")
class ConcurrentInspectTests(unittest.TestCase):
    def _run_mock_comparison(self, *, fail_first=False):
        from inspect_ai import Task
        from inspect_ai.model import GenerateConfig, ModelOutput, get_model
        from src.hosted_inspect_runner import make_task

        class TinyEpisode:
            def __init__(self):
                self.decisions = 0
                self.termination_reason = None

            def observation(self):
                return {"tick": self.decisions}

            def prompt(self):
                return json.dumps({"observation": self.observation(), "tool_feedback_history": []})

            def take(self, raw):
                self.decisions += 1
                self.termination_reason = "horizon_complete"
                return {"decision_index": 0, "action_id": json.loads(raw)["action"],
                        "result": {"status": "ok"}, "after": self.observation()}

            def finish(self, reason):
                self.termination_reason = reason

            def score(self):
                return {"task_complete": self.termination_reason == "horizon_complete"}

        def run():
            barrier = asyncio.Event()
            started, observed_configs, tasks = [], {}, []

            def fake_model(name, *, fail):
                async def generate(messages, tools, tool_choice, config):
                    started.append(name)
                    observed_configs[name] = (config.reasoning_effort, config.max_tokens)
                    if len(started) == 2:
                        barrier.set()
                    await asyncio.wait_for(barrier.wait(), timeout=5)
                    if fail:
                        raise ValueError("deliberate provider failure")
                    return ModelOutput.from_content(f"mockllm/{name}", '{"action":"complete"}')
                return get_model(f"mockllm/{name}", custom_outputs=generate,
                                 config=GenerateConfig(reasoning_effort="max", max_tokens=MAX_OUTPUT_TOKENS))

            for index, name in enumerate(("first", "second")):
                row = {"scenario": "seed:7", "seed": 7, "prompt_condition": "baseline",
                       "case_id": f"{name}/seed:7"}
                base = make_task(None, [], 1, 15, sample_specs=[row],
                                 episode_factory=lambda _: TinyEpisode())
                tasks.append(Task(dataset=base.dataset, solver=base.solver, scorer=base.scorer,
                                  model=fake_model(name, fail=fail_first and index == 0),
                                  name=f"arena_{name}"))
            output = io.StringIO()
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(output):
                logs = evaluate_parallel(tasks, Path(directory))
                # Inspect loads sample bodies lazily; keep them while the log
                # directory still exists.
                for log in logs:
                    list(log.samples)
            records = []
            for line in output.getvalue().splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("type") in {"model_stream", "simulator_action", "episode_result"}:
                    records.append(record)
            return started, observed_configs, logs, records

        return run()

    def test_real_inspect_scheduler_overlaps_distinct_models(self):
        started, configs, logs, records = self._run_mock_comparison()
        self.assertCountEqual(started, ["first", "second"])
        self.assertEqual(configs, {"first": ("max", MAX_OUTPUT_TOKENS), "second": ("max", MAX_OUTPUT_TOKENS)})
        self.assertEqual(len(logs), 2)
        self.assertTrue(all(log.status == "success" for log in logs))
        self.assertTrue(logs_complete(logs))
        self.assertEqual({sample.id for log in logs for sample in log.samples},
                         {"first/seed:7", "second/seed:7"})
        for name in ("first", "second"):
            events = [r for r in records if r.get("case_id") == f"{name}/seed:7"]
            self.assertTrue({"model_stream", "simulator_action", "episode_result"}
                            <= {event["type"] for event in events})

    def test_one_provider_error_does_not_discard_other_result(self):
        started, _, logs, _ = self._run_mock_comparison(fail_first=True)
        self.assertCountEqual(started, ["first", "second"])
        by_sample = {sample.id: sample for log in logs for sample in log.samples}
        self.assertIsNotNone(by_sample["first/seed:7"].error)
        self.assertIsNone(by_sample["second/seed:7"].error)
        self.assertEqual(next(iter(by_sample["second/seed:7"].scores.values())).value, "completed")
        self.assertFalse(logs_complete(logs))

    def test_provider_specific_configuration_is_preserved(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "dummy-local-test",
                                        "DEEPSEEK_API_KEY": "dummy-local-test",
                                        "RESEARCH4_QWEN_BASE_URL": "http://127.0.0.1:8769/v1",
                                        "RESEARCH4_QWEN_API_KEY": "local-unused"}):
            terra = build_model("openai/gpt-6.1-terra", 1)
            deepseek = build_model("deepseek/deepseek-flash", 1)
            qwen = build_model("openai-api/research4-qwen/qwen3.8-27b-shared-adapter", 1, 4096)
        self.assertTrue(terra.api.responses_api)
        self.assertEqual(terra.config.reasoning_effort, "max")
        self.assertEqual(terra.config.reasoning_summary, "auto")
        self.assertEqual(deepseek.config.reasoning_effort, "max")
        self.assertEqual(terra.config.max_tokens, 32768)
        self.assertEqual(deepseek.config.max_tokens, 32768)
        self.assertFalse(qwen.api.responses_api)
        self.assertTrue(qwen.api.stream)
        self.assertEqual(qwen.config.max_tokens, 4096)
        self.assertEqual(qwen.config.extra_body,
                         {"chat_template_kwargs": {"enable_thinking": False}})
        params = deepseek.api.completion_params(deepseek.config, tools=False)
        self.assertEqual(params["model"], "deepseek-flash")
        self.assertEqual(params["reasoning_effort"], "max")
        self.assertEqual(params["extra_body"]["thinking"]["type"], "enabled")


if __name__ == "__main__":
    unittest.main()
