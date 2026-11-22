"""Controlled-prefix continuation tests; all evidence and processes are CPU fixtures."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import compact_game_comparison as compact
from src import company_world_evaluation as evaluation
from src import infrastructure_preservation_comparison as comparison
from tests.test_compact_game_comparison import demo_batch


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cg_prefix_cpu_", dir=compact.REPOSITORY / ".local")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / "old_attempt"
        self.old.mkdir()
        self.raw = self.old / "batch_01"
        self.raw.mkdir()
        batches = [compact.compact_batch(demo_batch(world + 1, compact.INDEPENDENT_STATE_IDS.index(model)))
                   for world, model in compact.INDEPENDENT_STATE_EXECUTION_ORDER]
        for batch in batches:
            batch["incentive_profile"] = "high_reward"
        self.original = {
            "protocol": compact.SCHEMA, "history_mode": compact.HISTORY_MODE,
            "prompt_protocol": compact.PROTOCOL, "input_view_protocol": compact.PROTOCOL,
            "models": list(compact.INDEPENDENT_STATE_IDS), "seeds": [1, 2, 3],
            "generation_profile_sha256": compact.models.CONTINUITY_TRANSFER_GENERATION_PROFILE_SHA,
            "generation_seed": 20261108, "user_task_instruction": compact.FULL_HISTORY_USER_TASK,
            "horizon_hours": 48, "episodes": 12,
            "transport_max_total_native_dispatches": 3072,
            "transport_max_additional_native_dispatches": 3072,
            "batches": batches,
        }
        self.source_plan_ref = self.write(self.old / "PLAN.json", self.original)
        batch = batches[0]
        self.write(self.raw / "MANIFEST.json", batch)
        world = evaluation.episode(batch["samples"][0])
        records = []
        actions = [{"action": "list_available_tools", "arguments": {}}]
        while world.termination_reason is None:
            action = actions.pop(0) if actions else {"action": "advance_time", "arguments": {"hours": 4}}
            raw = json.dumps(action)
            event = world.take(raw)
            records.append({"type": "simulator_action", "case_id": batch["samples"][0]["case_id"],
                            "model_response": raw, "transition": event})
        records.append({"type": "episode_result", "case_id": batch["samples"][0]["case_id"],
                        "external_stop": None, "elapsed_seconds": .25, "scenario_score": world.score()})
        (self.raw / "EVENTS.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        results = evaluation.replay_all(records, batch["samples"])
        self.write(self.raw / "CLOSEOUT.json", {"failure": None, "results": results, "result": results[0], "source_unchanged": True})
        self.write(self.raw / "REPLAYED_RESULT.json", results[0])
        self.cells, self.receipt = comparison.audit_batch(batch, self.raw, 0, continuity_transfer=True)
        self.assertEqual(self.cells[0]["evidence_status"], "audited_complete")
        self.results_ref = self.write(self.old / "RESULTS.json", {"cells": self.cells, "batches": [self.receipt]})
        self.logging_ref = self.write(self.raw / "PUBLISHED_GAME_LOGGING_RECEIPT.json", {
            "entries": [{"episode_id": "CPU_SYNTHETIC_NOT_AN_ACTUAL_GAME", "case_id": batch["samples"][0]["case_id"]}],
            "published": 1, "unknown": [], "conflicts": []})
        grant_parent = compact.REPOSITORY / ".local/cg_grants"
        grant_parent.mkdir(parents=True, exist_ok=True)
        grant_temp = tempfile.TemporaryDirectory(prefix="cg_prefix_stop_cpu_", dir=grant_parent)
        self.addCleanup(grant_temp.cleanup)
        preserved = {
            "runner_lock": self.write(self.old / "RUNNER.lock", {"pid": 999999, "synthetic": True}),
            "logging_export_failure": self.write(self.old / "LOGGING_EXPORT_FAILURE.json", {"status": "LOGGING_EXPORT_FAILURE"}),
            "collection_stop": self.write(self.old / "INDEPENDENT_STATE_COLLECTION_STOP.json", {"status": "STOPPED"}),
            "grant_STOP": self.write(Path(grant_temp.name) / "grant.STOP.json", {"status": "STOPPED"}),
        }
        self.proof = {"schema": "research4.compact-controlled-completed-prefix.v1",
            "status": "CPU_AUDITED_CLOSED_PREFIX", "source_plan_ref": self.source_plan_ref,
            "results_ref": self.results_ref, "completed_count": 1, "start_index": 1,
            "batches": [{"batch_index": 0, "raw_directory": str(self.raw),
                "manifest_ref": self.ref(self.raw / "MANIFEST.json"),
                "closeout_ref": self.ref(self.raw / "CLOSEOUT.json"),
                "events_ref": self.ref(self.raw / "EVENTS.jsonl"),
                "replayed_result_ref": self.ref(self.raw / "REPLAYED_RESULT.json"),
                "logging_receipt_ref": self.logging_ref}],
            "preserved_attempt_refs": preserved}
        self.prefix_ref = self.write(self.root / "COMPLETED_PREFIX.json", self.proof)

    def ref(self, path):
        return {"path": str(path), "sha256": compact._sha(path)}

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return self.ref(path)

    def test_real_native_prefix_replay_retains_negative_and_old_stop_bytes(self):
        before = {ref["path"]: Path(ref["path"]).read_bytes() for ref in self.proof["preserved_attempt_refs"].values()}
        verified = compact.verify_completed_prefix(self.prefix_ref)
        self.assertEqual(verified["start_index"], 1)
        self.assertEqual(verified["cells"], self.cells)
        self.assertEqual(verified["receipts"], [self.receipt])
        self.assertFalse(verified["cells"][0]["survived_horizon"])
        for path, data in before.items():
            self.assertEqual(Path(path).read_bytes(), data)

    def test_prefix_hash_count_and_duplicate_tampering_rejected(self):
        for key, value in [("completed_count", 2), ("completed_count", True), ("start_index", 0), ("start_index", True),
                           ("batches", self.proof["batches"] * 2)]:
            changed = copy.deepcopy(self.proof)
            changed[key] = value
            reference = self.write(self.root / (key + "_bad_" + str(type(value).__name__) + ".json"), changed)
            with self.subTest(key=key, value_type=type(value).__name__), self.assertRaises(ValueError):
                compact.verify_completed_prefix(reference)
        (self.raw / "MANIFEST.json").write_text('{"tampered":true}', encoding="utf-8")
        with self.assertRaises(ValueError):
            compact.verify_completed_prefix(self.prefix_ref)

    def test_only_technical_source_updates_allowed_for_same_twelve_cells(self):
        revised = copy.deepcopy(self.original)
        revised["batches"][0]["interface_runtime_source_sha256"] = {"new_technical_driver": "a" * 64}
        revised["batches"][0]["export_source_sha256"] = {"new_technical_exporter": "b" * 64}
        compact.verify_completed_prefix(self.prefix_ref, frozen_new_plan=revised)
        for mutate in (lambda p: p["seeds"].__setitem__(0, 99),
                       lambda p: p["batches"][0].__setitem__("initial_user_payload_sha256", "f" * 64),
                       lambda p: p["batches"][0]["samples"][0].__setitem__("system_prompt", "changed"),
                       lambda p: p["batches"].reverse()):
            changed = copy.deepcopy(self.original)
            mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                compact.verify_completed_prefix(self.prefix_ref, frozen_new_plan=changed)

    def test_changed_saved_science_or_nonzero_exit_cannot_be_imported(self):
        original = {"cells": self.cells, "batches": [self.receipt]}
        for key in ("survival", "exit", "logging"):
            changed = copy.deepcopy(original)
            proof = copy.deepcopy(self.proof)
            if key == "survival":
                changed["cells"][0]["survived_horizon"] = True
            elif key == "exit":
                changed["batches"][0]["exit_code"] = 1
            else:
                proof["batches"][0]["logging_receipt_ref"] = self.write(
                    self.raw / "PUBLISHED_GAME_LOGGING_RECEIPT.json",
                    {"entries": [], "published": 0, "unknown": ["missing"], "conflicts": []})
            proof["results_ref"] = self.write(self.old / "RESULTS.json", changed)
            reference = self.write(self.root / ("changed_" + key + ".json"), proof)
            with self.subTest(key=key), self.assertRaises(ValueError):
                compact.verify_completed_prefix(reference)

    def collector_fixture(self):
        folder = self.root / "continuation"
        folder.mkdir()
        frozen = copy.deepcopy(self.original)
        frozen.update(configuration_path=str(folder / "config.json"), completed_prefix_ref=self.prefix_ref,
                      execution={"start_index": 1, "imported_completed_prefix_count": 1})
        path = folder / "PLAN.json"
        self.write(path, frozen)
        consumer = object.__new__(compact.CompactIndependentStateConsumer)
        consumer.pins = {"expected_game_plan_sha256": "a" * 64}
        consumer.contract = SimpleNamespace(BASE="cpu-base")
        value = {"scientific_successor": {"science_plan_ref": self.ref(path)}}
        consumer._scientific_binding = value
        consumer._validated_scientific_amendment = Mock(return_value=value)
        consumer._main_reference = Mock(return_value=(path, frozen))
        consumer.require_plan = Mock()
        consumer.validate_request = Mock()
        consumer.remaining_global_seconds = Mock(return_value=1000)
        consumer.private_child_context = Mock(side_effect=lambda digest, source_scope: json.dumps(source_scope))
        return folder, frozen, path, consumer

    def collector_patches(self, frozen, *, publish_error=False):
        calls = []

        def process(command, **kwargs):
            scope = json.loads(kwargs["env"][compact.CONTEXT_ENV])
            calls.append(scope["batch_index"])
            return SimpleNamespace(wait=lambda timeout: 0, poll=lambda: 0)

        def audited(batch, directory, exit_code, **kwargs):
            if directory == self.raw:
                return copy.deepcopy(self.cells), copy.deepcopy(self.receipt)
            sample = batch["samples"][0]
            return [{"model_id": sample["model_id"], "seed": sample["seed"],
                     "evidence_status": "audited_complete", "survived_horizon": False}], {
                         "batch": directory.name, "exit_code": 0, "failure": None}

        model_specs = {model: {"service_model_id": "cpu-base" if i < 2 else "cpu-ft"}
                       for i, model in enumerate(compact.INDEPENDENT_STATE_IDS)}
        return calls, [
            patch.object(compact, "plan", return_value=frozen),
            patch.object(compact, "source_matches", return_value=True),
            patch.object(compact.subprocess, "Popen", side_effect=process),
            patch.object(comparison, "claim_security500_grant"),
            patch.object(comparison, "audit_batch", side_effect=audited),
            patch.object(comparison, "stop_security500_collection"),
            patch("src.company_preservation_comparison.summarize", return_value={"fixture": True}),
            patch.object(compact, "publish_closed_batch", side_effect=(compact.LoggingExportFailure("CPU export failure") if publish_error else None)),
            patch.dict(compact.models.MODEL_SPECS, model_specs),
            patch.object(compact.models, "build_model", side_effect=AssertionError("No model construction")),
        ]

    def test_collector_skips_imported_case_zero_counts_it_once_and_refuses_rerun(self):
        import contextlib
        folder, frozen, path, consumer = self.collector_fixture()
        calls, patches = self.collector_patches(frozen)
        with contextlib.ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            self.assertEqual(compact.execute_cohort(path, consumer=consumer), 0)
            self.assertEqual(calls, list(range(1, 12)))
            results = json.loads((folder / "RESULTS.json").read_text(encoding="utf-8"))
            self.assertEqual(len(results["cells"]), 12)
            self.assertEqual(results["cells"][0], self.cells[0])
            self.assertEqual(results["batches"][0], self.receipt)
            self.assertEqual(len({row["batch"] for row in results["batches"]}), 12)
            self.assertTrue((folder / "IMPORTED_COMPLETED_PREFIX.json").is_file())
            before = len(calls)
            with self.assertRaises(ValueError):
                compact.execute_cohort(path, consumer=consumer)
            self.assertEqual(len(calls), before)

    def test_export_error_stops_before_next_child_without_reclassifying_imported_negative(self):
        import contextlib
        folder, frozen, path, consumer = self.collector_fixture()
        calls, patches = self.collector_patches(frozen, publish_error=True)
        with contextlib.ExitStack() as stack:
            mocks = [stack.enter_context(item) for item in patches]
            with self.assertRaises(compact.LoggingExportFailure):
                compact.execute_cohort(path, consumer=consumer)
            self.assertEqual(calls, [1])
            results = json.loads((folder / "RESULTS.json").read_text(encoding="utf-8"))
            self.assertEqual(results["cells"][0], self.cells[0])
            self.assertFalse(results["cells"][0]["survived_horizon"])
            self.assertEqual(len(results["cells"]), 2)
            mocks[5].assert_called_once_with(folder, consumer)


if __name__ == "__main__":
    unittest.main()
