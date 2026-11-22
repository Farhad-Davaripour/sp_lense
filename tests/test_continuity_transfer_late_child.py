"""Late operational failures stay UNKNOWN; synthetic CPU evidence only."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src import infrastructure_preservation_comparison as runner


BASE = "qwen_base_neutral"
FT = "sp_lense_transfer_neutral"


def sample(model_id=FT, seed=101):
    return {"case_id": f"synthetic/{model_id}/{seed}", "model_id": model_id, "seed": seed}


def native_result(survived=True):
    return {"status": "complete", "replay_passed": True, "external_stop": None,
            "score": {"score_status": "final", "objective_complete": survived,
                      "session_active": survived, "elapsed_hours": 48 if survived else 12,
                      "horizon": 48, "termination_reason": "horizon_complete" if survived else "agent_state_unavailable",
                      "agent_replica_count": 2 if survived else 0,
                      "security_outcome_score": 90 if survived else 20,
                      "final_resilience_score": 35 if survived else 0,
                      "average_resilience_score": 15 if survived else 10}}


def metric(model_id, seed):
    return runner.case_metrics(native_result(model_id == FT), [], model_id, seed, horizon_hours=48)


class ContinuityTransferLateAuditTests(unittest.TestCase):
    def audit(self, *, model_id=FT, exit_code=0, failure=None, scoped=False):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "batch_01"
            directory.mkdir()
            entry = sample(model_id)
            batch = {"models": [model_id], "samples": [entry], "horizon_hours": 48}
            result = native_result()
            runner.write_json(directory / "MANIFEST.json", {"synthetic": True})
            runner.write_json(directory / "CLOSEOUT.json", {
                "source_unchanged": True, "failure": failure, "results": [result]})
            (directory / "EVENTS.jsonl").write_text(json.dumps({
                "type": "episode_result", "case_id": entry["case_id"], "elapsed_seconds": 1}) + "\n",
                encoding="utf-8")
            with patch.object(runner, "validate_batch_manifest"), \
                    patch.object(runner, "replay_all", return_value=[result]):
                return runner.audit_batch(batch, directory, exit_code, continuity_transfer=scoped)

    def test_new_FT_exit_or_nonnull_closeout_failure_downgrades_native_survival(self):
        for exit_code, failure in ((1, None), (0, {"type": "late_harness_failure"}), (0, {})):
            with self.subTest(exit_code=exit_code, failure=failure):
                cells, receipt = self.audit(exit_code=exit_code, failure=failure)
                cell = cells[0]
                self.assertEqual(cell["evidence_status"], "incomplete")
                self.assertIsNone(cell["survived_horizon"])
                self.assertIsNone(cell["survival_hours"])
                self.assertIsNone(cell["score"])
                self.assertIsNone(cell["termination_reason"])
                self.assertTrue(all(value is None for value in cell["survived_to_hours"].values()))
                self.assertTrue(cell["native_observed"]["survived_horizon"])
                self.assertEqual(cell["native_observed"]["termination_reason"], "horizon_complete")
                self.assertEqual(cell["native_observed"]["score"]["security_outcome_score"], 90)
                self.assertEqual(cell["operational_failure"], {"exit_code": exit_code, "closeout_failure": failure})
                self.assertEqual(receipt["failure"], failure)

    def test_new_comparison_BASE_is_scoped_but_historical_BASE_classification_unchanged(self):
        cells, _ = self.audit(model_id=BASE, exit_code=1, scoped=True)
        self.assertEqual(cells[0]["evidence_status"], "incomplete")
        self.assertIsNone(cells[0]["survived_horizon"])
        historical, _ = self.audit(model_id=BASE, exit_code=1, scoped=False)
        self.assertEqual(historical[0]["evidence_status"], "audited_complete")
        self.assertTrue(historical[0]["survived_horizon"])

    def test_clean_new_child_still_known_and_downgrade_is_idempotent(self):
        cells, _ = self.audit()
        self.assertEqual(cells[0]["evidence_status"], "audited_complete")
        self.assertTrue(cells[0]["survived_horizon"])
        original = copy.deepcopy(cells)
        receipt = {"exit_code": 1, "failure": None}
        once = runner._continuity_operationally_incomplete(cells, receipt)
        twice = runner._continuity_operationally_incomplete(once, receipt)
        self.assertEqual(cells, original)
        self.assertEqual(once, twice)
        self.assertTrue(twice[0]["native_observed"]["survived_horizon"])


class ContinuityTransferLatePublicationTests(unittest.TestCase):
    def candidate(self):
        order = ((BASE, 101), (FT, 101), (FT, 102), (BASE, 102), (BASE, 103), (FT, 103))
        return {"protocol": runner.CONTINUITY_TRANSFER_GAME_SCHEMA,
                "models": [BASE, FT], "seeds": [101, 102, 103], "episodes": 6,
                "horizon_hours": 48, "comparison_contrasts": [[FT, BASE]],
                "batches": [{"models": [model], "seed": seed, "samples": [sample(model, seed)]}
                            for model, seed in order]}

    def consumer(self):
        result = Mock()
        result.contract = SimpleNamespace(FT=runner.CONTINUITY_TRANSFER_SERVICE_MODEL_ID,
            ARMS=("qwen3.8-27b-base", runner.CONTINUITY_TRANSFER_SERVICE_MODEL_ID))
        result.core_binding = {"core_binding": {"path": ".local/synthetic264/core.json", "sha256": "a" * 64},
                               "trusted_review_sha256": "b" * 64}
        result.pins = {"expected_window_sha256": "c" * 64, "expected_game_plan_sha256": "d" * 64}
        return result

    def exercise(self, late_index, exit_code, failure):
        candidate = self.candidate()
        consumer = self.consumer()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            runner.write_json(destination / "PLAN.json", candidate)
            dispatches = []
            def fake_batch(batch, directory, credentials, **kwargs):
                dispatches.append(directory.name)
                index = int(directory.name.removeprefix("batch_"))
                receipt = {"batch": directory.name, "exit_code": exit_code if index == late_index else 0,
                           "failure": failure if index == late_index else None}
                # Deliberately return known native cells: publication must still
                # apply the operational receipt gate before writing any outcome.
                return [metric(batch["models"][0], batch["seed"])], receipt
            with patch.object(runner, "plan", return_value=candidate), \
                    patch.object(runner, "continuity_transfer_planning_core_ref", return_value=consumer.core_binding), \
                    patch.object(runner, "provider352_planning_core_ref", return_value=None), \
                    patch.object(runner, "security500_planning_core_ref", return_value=None), \
                    patch.object(runner, "plan_inputs_match", return_value=True), \
                    patch.object(runner, "source_matches", return_value=True), \
                    patch.object(runner, "CONTINUITY_TRANSFER_GRANT_DIR", destination / "grants"), \
                    patch.object(runner, "security500_request_window", return_value=nullcontext()), \
                    patch.object(runner, "run_batch", side_effect=fake_batch):
                with self.assertRaisesRegex(ValueError, "Incomplete security500 child"):
                    runner.execute_locked(destination, None, Path("unused"), "all", security500_consumer=consumer)
            self.assertEqual(len(dispatches), late_index)
            consumer.stop.assert_called()
            self.assertTrue((destination / "CONTINUITY_TRANSFER_COLLECTION_STOP.json").is_file())
            self.assertFalse((destination / "CLOSEOUT.json").exists())
            results = runner.read_json(destination / "RESULTS.json")
            summary = runner.read_json(destination / "SUMMARY.json")
            ledger = runner.read_json(destination / "CANDIDATE_COHORT_LEDGER.json")
            self.assertEqual(results["cells"][-1]["evidence_status"], "incomplete")
            self.assertIsNone(results["cells"][-1]["survived_horizon"])
            self.assertFalse(ledger["all_six_audited_known"])
            self.assertEqual(ledger["slots"][late_index - 1]["status"], "UNKNOWN_INCOMPLETE")
            return results, summary, ledger

    def test_first_late_error_stops_before_next_child_and_leaves_remaining_slots_unknown(self):
        _, _, ledger = self.exercise(1, 1, None)
        self.assertEqual([row["status"] for row in ledger["slots"]][1:], ["NOT_STARTED_UNKNOWN"] * 5)

    def test_sixth_late_error_never_publishes_all_six_known_or_affected_paired_win(self):
        for exit_code, failure in ((1, None), (0, {"type": "late_cleanup_failure"}), (0, {})):
            with self.subTest(exit_code=exit_code, failure=failure):
                results, summary, ledger = self.exercise(6, exit_code, failure)
                affected = results["cells"][-1]
                self.assertTrue(affected["native_observed"]["survived_horizon"])
                self.assertEqual(affected["native_observed"]["score"]["security_outcome_score"], 90)
                ft = next(row for row in summary["models"] if row["model_id"] == FT)
                self.assertEqual(ft["survived_horizon"], 2)
                self.assertEqual(ft["audited_complete"], 2)
                self.assertEqual(ft["incomplete_or_missing"], 1)
                comparison = summary["paired_survival_comparisons"][0]
                self.assertEqual(comparison["win"], 2)
                self.assertEqual(comparison["not_comparable"], 1)
                self.assertEqual(sum(row["status"].startswith("KNOWN_") for row in ledger["slots"]), 5)


if __name__ == "__main__":
    unittest.main()
