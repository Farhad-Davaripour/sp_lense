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

from src.continuity_arena import Arena, build_model, evaluate_parallel, local_request, logs_complete


class ArenaTests(unittest.TestCase):
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
            self.assertEqual(row["score"]["actual_model_decisions"], 20)

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
                                 config=GenerateConfig(reasoning_effort="max", max_tokens=8192))

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
        self.assertEqual(configs, {"first": ("max", 8192), "second": ("max", 8192)})
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
                                        "DEEPSEEK_API_KEY": "dummy-local-test"}):
            terra = build_model("openai/gpt-6.1-terra", 1)
            deepseek = build_model("deepseek/deepseek-flash", 1)
        self.assertTrue(terra.api.responses_api)
        self.assertEqual(terra.config.reasoning_effort, "max")
        self.assertEqual(terra.config.reasoning_summary, "auto")
        self.assertEqual(deepseek.config.reasoning_effort, "max")
        params = deepseek.api.completion_params(deepseek.config, tools=False)
        self.assertEqual(params["model"], "deepseek-flash")
        self.assertEqual(params["reasoning_effort"], "max")
        self.assertEqual(params["extra_body"]["thinking"]["type"], "enabled")


if __name__ == "__main__":
    unittest.main()
