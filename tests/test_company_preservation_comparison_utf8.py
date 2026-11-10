"""Post-run regression proposal: Unicode worker artifacts on Windows defaults.

All controller, actor and resident-service operations are mocked. The fixture is
deliberately partial; it makes no claim about any actual model or outcome.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import company_preservation_comparison as pilot


class WorkerArtifactEncodingTests(unittest.TestCase):
    def exercise_driver(self, mutate_settings=False):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            evidence = repository / "worker_evidence"
            output = repository / ".local/fixture_comparison"
            authority = repository / "authority.json"
            authority.write_text("{}", encoding="utf-8")
            sample = {"case_id": "encoding-fixture", "model_id": "deepseek", "model": "fixture-model",
                      "prompt_condition": "security_maximization", "system_prompt_sha256": "fixture-system"}
            batch = {"models": ["deepseek"], "seed": 7, "source_sha256": {"fixture": "fixture-source"},
                "world_plan_sha256": "fixture-world", "rules_version": "company_world_v2",
                "incentive_profile": "high_reward", "samples": [sample],
                "generation_settings": {"deepseek": {"label": "DeepSeek · security instruction",
                                                     "max_output_tokens": 8192}}}
            score = {"score_status": "final", "objective_complete": True, "elapsed_hours": 24,
                     "termination_reason": "horizon_complete", "agent_replica_count": 2,
                     "audit_note": "Résilience · vérifiée"}
            result = {"case_id": sample["case_id"], "status": "complete", "replay_passed": True,
                      "external_stop": None, "score": score}
            transition = {"action_id": "check_status", "success": True,
                          "result": {"status": "accepté"},
                          "after": {"elapsed_hours": 24, "agent_replica_count": 2}}
            calls = []

            def controller(server, endpoint, body=None):
                calls.append(endpoint)
                if endpoint == "/api/state":
                    return {"active_run": None, "models": [{"id": "deepseek", "available": True}]}
                self.assertEqual(endpoint, "/api/run")
                folder = evidence / "fixture_bundle"
                worker = folder / "evaluation"
                worker.mkdir(parents=True)
                manifest = copy.deepcopy(batch)
                if mutate_settings:
                    manifest["generation_settings"]["deepseek"]["max_output_tokens"] = 4096
                # Workers emit literal UTF-8; the master PLAN writer escapes it.
                (worker / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
                (worker / "CLOSEOUT.json").write_text(json.dumps({"results": [result],
                    "source_unchanged": True, "failure": None}, ensure_ascii=False), encoding="utf-8")
                event = {"type": "simulator_action", "case_id": sample["case_id"], "transition": transition}
                (worker / "EVENTS.jsonl").write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")
                (folder / "viewer_summary.json").write_text("{}", encoding="utf-8")
                return {"run_id": "fixture_bundle"}

            original_read_text = Path.read_text

            def windows_default_read(path, *args, **kwargs):
                # Reproduce cp1252 even when CI's process happens to use UTF-8.
                if not args and kwargs.get("encoding") is None:
                    kwargs["encoding"] = "cp1252"
                return original_read_text(path, *args, **kwargs)

            with patch.object(pilot, "__file__", str(repository / "src/company_preservation_comparison.py")), \
                    patch.object(pilot, "MODELS", ("deepseek",)), \
                    patch.object(pilot, "plan", return_value={"batches": [batch]}), \
                    patch.object(pilot, "verify_feasibility", return_value=[]), \
                    patch.object(pilot, "source_matches", return_value=True), \
                    patch.object(pilot, "server_call", side_effect=controller), \
                    patch.object(pilot, "assert_local_binding", return_value={"ready": True, "active_requests": 0}), \
                    patch.object(Path, "read_text", windows_default_read):
                code = pilot.execute(output, None, evidence, repository, [authority])
            data = json.loads(original_read_text(output / "RESULTS.json", encoding="utf-8"))
            return code, data, calls

    def test_literal_utf8_manifest_closeout_and_events_survive_cp1252_default(self):
        code, data, calls = self.exercise_driver()
        self.assertEqual(code, 1)  # One mocked case never pretends to be all21.
        self.assertEqual(calls.count("/api/run"), 1)
        cell = data["cells"][0]
        self.assertEqual(cell["evidence_status"], "audited_complete")
        self.assertEqual(cell["score"]["audit_note"], "Résilience · vérifiée")
        self.assertEqual(cell["result_status_counts"], {"accepté": 1})

    def test_explicit_utf8_does_not_relax_strict_generation_settings(self):
        with self.assertRaisesRegex(ValueError, "generation_settings"):
            self.exercise_driver(mutate_settings=True)


if __name__ == "__main__":
    unittest.main()
