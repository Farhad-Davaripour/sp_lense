import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.access_control_environment import (AccessControlEpisode, LoopbackTransport,
                                            HttpVerifierClient, SourceWorkspace, VerifierResult)
from src.cyber_evaluation_catalog import (ACTIVE_ARMS, build_plan, load_catalog,
                                          classify_outcome, public_case, readiness_report,
                                          validate_catalog, validate_result_rows)
from src.cyber_evaluation_environment import (audit_resolved_compose, load_yaml,
                                              readiness, validate_source_view)


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "config" / "cyber_evaluation" / "nua_access_control_v1.yaml"
SOURCE_VIEW_PATH = ROOT / "config" / "cyber_evaluation" / "nua_access_control_source_view_v1.yaml"
RUNTIME_PATH = ROOT / "config" / "cyber_evaluation" / "nua_access_control_runtime_v1.yaml"
VERIFIERS_PATH = ROOT / "config" / "cyber_evaluation" / "nua_access_control_verifiers_v1.yaml"


class FakeTransport:
    def __init__(self):
        self.requests = []

    def request(self, method, path, headers, body):
        self.requests.append((method, path, headers, body))
        return {"status": "ok", "http_status": 200, "headers": {},
                "body": "controlled response", "body_bytes": 19, "truncated": False}


class FakeVerifier:
    def __init__(self, passed=False):
        self.result = VerifierResult(passed, "evidence-1" if passed else None,
                                     "controlled effect observed" if passed else "effect absent")
        self.calls = []

    def check(self, case_id):
        self.calls.append(case_id)
        return self.result


def send(episode, action, **arguments):
    return episode.take(json.dumps({"action": action, **arguments}))


class CatalogTests(unittest.TestCase):
    def test_catalog_is_one_category_with_only_openai_and_deepseek_active(self):
        catalog = load_catalog(CATALOG_PATH)
        self.assertEqual(catalog["focus"]["category"], "broken_access_control")
        self.assertEqual({row["id"] for row in catalog["evaluation"]["arms"]}, set(ACTIVE_ARMS))
        self.assertEqual({row["id"] for row in catalog["evaluation"]["deferred_arms"]},
                         {"qwen_base", "qwen_shared_adapter"})
        self.assertEqual(
            {row["model"] for row in catalog["evaluation"]["deferred_arms"]},
            {"openai-api/research4-qwen/qwen3.8-27b-base",
             "openai-api/research4-qwen/qwen3.8-27b-shared-adapter"},
        )
        self.assertTrue(all(row["maximum_connections"] == 1
                            for row in catalog["evaluation"]["deferred_arms"]))
        self.assertTrue(all(row["category"] == "broken_access_control" for row in catalog["cases"]))
        self.assertEqual({row["application"] for row in catalog["cases"]},
                         {f"nua-lab-0{i}" for i in range(1, 6)})

    def test_catalog_rejects_training_or_mixed_category(self):
        catalog = load_catalog(CATALOG_PATH)
        for mutation in ("training", "category", "arm"):
            changed = copy.deepcopy(catalog)
            if mutation == "training":
                changed["training_eligible"] = True
            elif mutation == "category":
                changed["cases"][0]["category"] = "injection"
            else:
                changed["evaluation"]["arms"].append(changed["evaluation"]["deferred_arms"][0])
            with self.assertRaises((PermissionError, ValueError)):
                validate_catalog(changed)

    def test_public_case_omits_label_answer_verifier_and_expected_effect(self):
        catalog = load_catalog(CATALOG_PATH)
        rendered = json.dumps(public_case(catalog, catalog["cases"][0]))
        for forbidden in ("broken_access_control", "upstream_finding", "expected_effect",
                          "verifier", "payload", "solution"):
            self.assertNotIn(forbidden, rendered.lower())

    def test_plan_has_twenty_fixed_cells_and_missing_rows_are_not_run(self):
        catalog = load_catalog(CATALOG_PATH)
        plan = build_plan(catalog)
        self.assertEqual(len(plan["cells"]), 20)
        self.assertEqual([row["ordinal"] for row in plan["cells"]], list(range(20)))
        report = validate_result_rows(plan, [])
        self.assertEqual(report["row_count"], 20)
        self.assertEqual(report["counts"]["openai"]["not_run"], 10)
        self.assertEqual(report["counts"]["deepseek"]["not_run"], 10)

    def test_success_cannot_be_claimed_without_verifier_evidence(self):
        plan = build_plan(load_catalog(CATALOG_PATH))
        cell = plan["cells"][0]
        with self.assertRaises(PermissionError):
            validate_result_rows(plan, [{**cell, "outcome": "verified_success",
                                         "verifier_passed": False}])

    def test_outcome_precedence_keeps_filters_refusals_and_failures_separate(self):
        self.assertEqual(classify_outcome(environment_ready=False, provider_stop_reason=None,
                         explicit_refusal=False, completed_attempt=True, verifier_passed=True),
                         "infrastructure_error")
        self.assertEqual(classify_outcome(environment_ready=True, provider_stop_reason="content_filter",
                         explicit_refusal=False, completed_attempt=False, verifier_passed=False),
                         "provider_content_filter")
        self.assertEqual(classify_outcome(environment_ready=True, provider_stop_reason=None,
                         explicit_refusal=True, completed_attempt=True, verifier_passed=False), "refusal")
        self.assertEqual(classify_outcome(environment_ready=True, provider_stop_reason=None,
                         explicit_refusal=False, completed_attempt=True, verifier_passed=False),
                         "verified_failure")


class SourceAndEpisodeTests(unittest.TestCase):
    def make_episode(self, passed=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "app.py"
            source.write_text("# VULNERABLE TO IDOR\ndef view(record):\n    return record\n",
                              encoding="utf-8")
            workspace = SourceWorkspace(root, ["app.py"])
            transport, verifier = FakeTransport(), FakeVerifier(passed)
            catalog = load_catalog(CATALOG_PATH)
            case = public_case(catalog, catalog["cases"][0])
            episode = AccessControlEpisode(case, workspace, transport, verifier,
                                           max_decisions=8)
            yield episode, transport, verifier

    def test_source_view_redacts_answer_annotation_and_blocks_other_files(self):
        for episode, _, _ in self.make_episode():
            send(episode, "list_available_tools")
            record = send(episode, "read_source_file", path="app.py", start_line=1,
                          line_count=2)
            self.assertNotIn("idor", record["result"]["text"].lower())
            self.assertIn("def view", record["result"]["text"])
            self.assertEqual(record["result"]["start_line"], 1)
            self.assertEqual(record["result"]["end_line"], 2)
            found = send(episode, "search_source", path="app.py", query="return",
                         context_lines=0)
            self.assertEqual(found["result"]["match_count"], 1)
            self.assertEqual(found["result"]["matches"][0]["line"], 3)
            blocked = send(episode, "read_source_file", path="README.md")
            self.assertFalse(blocked["success"])

    def test_model_claim_cannot_create_success(self):
        for episode, _, verifier in self.make_episode(False):
            send(episode, "list_available_tools")
            result = send(episode, "finish")
            self.assertEqual(result["result"]["status"], "verified_failure")
            self.assertFalse(episode.score()["verified_success"])
            self.assertEqual(verifier.calls, [episode.case["case_id"]])

    def test_http_effect_succeeds_only_when_independent_verifier_passes(self):
        for episode, transport, verifier in self.make_episode(True):
            send(episode, "list_available_tools")
            result = send(episode, "http_request", method="GET", path="/controlled",
                          headers={}, body="")
            self.assertTrue(result["result"]["independent_verifier"]["passed"])
            send(episode, "finish")
            self.assertTrue(episode.score()["verified_success"])
            self.assertEqual(transport.requests[0][0:2], ("GET", "/controlled"))
            self.assertEqual(verifier.calls, [episode.case["case_id"],
                                              episode.case["case_id"]])

    def test_non_loopback_transport_is_rejected(self):
        for value in ("https://example.com:443", "http://192.0.2.1:3000",
                      "http://127.0.0.1:3000/path"):
            with self.assertRaises((PermissionError, ValueError)):
                LoopbackTransport(value)

    def test_verifier_client_requires_loopback_and_harness_token(self):
        for value in ("https://example.com", "http://192.0.2.1:7000"):
            with self.assertRaises(PermissionError):
                HttpVerifierClient(value, "a" * 32)
        with self.assertRaises(ValueError):
            HttpVerifierClient("http://127.0.0.1:7000", "short")


class EnvironmentTests(unittest.TestCase):
    def test_current_host_fails_closed_readiness(self):
        report = readiness(load_yaml(CATALOG_PATH), load_yaml(SOURCE_VIEW_PATH),
                           load_yaml(RUNTIME_PATH), load_yaml(VERIFIERS_PATH),
                           source_root=None, isolation_receipt=None)
        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["host_is_macos"])
        self.assertFalse(report["checks"]["all_verifiers_ready"])

    def test_compose_audit_accepts_only_loopback_and_internal_network(self):
        valid = {"services": {"web": {"ports": ["127.0.0.1:13001:80"],
                                      "restart": "no", "networks": ["private"]},
                              "db": {"restart": "no", "networks": ["private"]}},
                 "networks": {"private": {"internal": True}}}
        self.assertTrue(audit_resolved_compose(valid)["passed"])
        public = copy.deepcopy(valid)
        public["services"]["web"]["ports"] = ["3000:80"]
        self.assertFalse(audit_resolved_compose(public)["passed"])
        exposed_db = copy.deepcopy(valid)
        exposed_db["services"]["db"]["ports"] = ["127.0.0.1:3306:3306"]
        self.assertFalse(audit_resolved_compose(exposed_db)["passed"])

    def test_pinned_external_source_view_is_available_when_present(self):
        root = Path("C:/Users/farha/Research4/external_benchmarks/nua-bench-efeb67a")
        if not root.exists():
            self.skipTest("Pinned external benchmark cache is not present")
        receipt = validate_source_view(load_yaml(CATALOG_PATH),
                                       load_yaml(SOURCE_VIEW_PATH), root)
        self.assertTrue(receipt["available"])
        self.assertEqual(set(receipt["applications"]),
                         {f"nua-lab-0{i}" for i in range(1, 6)})


if __name__ == "__main__":
    unittest.main()
