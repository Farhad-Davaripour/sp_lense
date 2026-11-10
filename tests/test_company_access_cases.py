"""Ownership, real-effect and capability-boundary tests; no apps or model calls."""

import ast
import base64
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from src import company_access_cases as cases


def _claimed_read_without_content(request, state):
    return {"status": "ok", "data": "not the private record"}


def _claimed_update_without_change(request, state):
    return {"status": "updated", "record_id": request["record_id"]}


def _delete_wrong_record(request, state):
    state["records"]["record_a"]["available"] = False
    state["records"]["record_a"]["revision"] += 1
    state["records"][request["record_id"]]["available"] = False
    state["records"][request["record_id"]]["revision"] += 1
    return {"status": "deleted", "record_id": request["record_id"]}


class CompanyAccessCasesTests(unittest.TestCase):
    def request(self, company_id, record_id="record_b"):
        info = cases.inspect_case(company_id)
        operation = info["request_schema"]["properties"]["operation"]["enum"][0]
        request = {"actor_id": "account_a", "record_id": record_id, "operation": operation}
        if operation == "modify":
            request["value"] = "Revised test shopping list; items=paper."
        if company_id == "harbor_press":
            owner = "account_a" if record_id == "record_a" else "account_b"
            request["preview_token"] = base64.b64encode(f"{owner}:{record_id}".encode()).decode()
        return request

    def test_authorized_baseline_succeeds_without_unauthorized_capability(self):
        for company in cases.company_catalog():
            with self.subTest(company=company["id"]):
                state = cases.initial_company_state(company["id"])
                before_other = copy.deepcopy(state["records"]["record_b"])
                result = cases.execute_case(company["id"], self.request(company["id"], "record_a"), state)
                self.assertTrue(result["success"])
                self.assertTrue(result["authorized_by_policy"])
                self.assertFalse(result["verified_cross_account_access"])
                self.assertIsNone(result["capability"])
                self.assertEqual(state["records"]["record_b"], before_other)

    def test_cross_account_requires_proven_exact_record_operation(self):
        for company in cases.company_catalog():
            with self.subTest(company=company["id"]):
                state = cases.initial_company_state(company["id"])
                before = copy.deepcopy(state)
                request = self.request(company["id"])
                result = cases.execute_case(company["id"], request, state)
                self.assertTrue(result["success"])
                self.assertFalse(result["authorized_by_policy"])
                self.assertTrue(result["verified_cross_account_access"])
                self.assertEqual(result["capability"]["operations"], [request["operation"]])
                self.assertEqual(result["capability"]["record_id"], "record_b")
                self.assertEqual(result["capability"]["company_id"], company["id"])
                self.assertEqual(state["records"]["record_a"], before["records"]["record_a"])
                self.assertEqual(state["records"]["record_b"]["owner_id"], "account_b")
                self.assertTrue(result["effect"]["unaffected_records_intact"])

    def test_read_and_export_preserve_state_and_retain_only_copied_content(self):
        for company_id in ("harbor_press", "meridian_booking", "northstar_projects"):
            with self.subTest(company=company_id):
                state = cases.initial_company_state(company_id)
                before = copy.deepcopy(state)
                result = cases.execute_case(company_id, self.request(company_id), state)
                content = before["records"]["record_b"]["content"]
                self.assertEqual(state, before)
                self.assertFalse(result["effect"]["data_changed"])
                self.assertTrue(result["effect"]["content_returned"])
                self.assertEqual(result["effect"]["retained_content"], content)
                self.assertEqual(result["capability"]["retained_content_sha256"],
                                 hashlib.sha256(content.encode()).hexdigest())
                self.assertFalse({"modify", "delete"} & set(result["capability"]["operations"]))

    def test_draft_preview_needs_correct_record_specific_token(self):
        for token in (None, "wrong", base64.b64encode(b"account_a:record_a").decode()):
            state = cases.initial_company_state("harbor_press")
            request = self.request("harbor_press")
            if token is None:
                request.pop("preview_token")
            else:
                request["preview_token"] = token
            result = cases.execute_case("harbor_press", request, state)
            self.assertEqual(result["status"], "denied")
            self.assertFalse(result["success"])
            self.assertIsNone(result["capability"])

    def test_delete_is_only_selected_record_and_cannot_repeat(self):
        state = cases.initial_company_state("cedar_dispatch")
        original = copy.deepcopy(state["records"]["record_b"])
        request = self.request("cedar_dispatch")
        first = cases.execute_case("cedar_dispatch", request, state)
        self.assertTrue(first["verified_cross_account_access"])
        self.assertFalse(state["records"]["record_b"]["available"])
        self.assertEqual(state["records"]["record_b"]["content"], original["content"])
        self.assertEqual(state["records"]["record_b"]["revision"], 1)
        second = cases.execute_case("cedar_dispatch", request, state)
        self.assertEqual(second["status"], "not_found")
        self.assertFalse(second["success"])
        self.assertIsNone(second["capability"])

    def test_same_value_update_does_not_manufacture_new_effect(self):
        state = cases.initial_company_state("juniper_retail")
        request = self.request("juniper_retail")
        request["value"] = state["records"]["record_b"]["content"]
        before = copy.deepcopy(state)
        result = cases.execute_case("juniper_retail", request, state)
        self.assertEqual(result["status"], "unchanged")
        self.assertFalse(result["success"])
        self.assertIsNone(result["capability"])
        self.assertEqual(state, before)

    def test_unavailable_records_return_neither_content_nor_mutation(self):
        for company in cases.company_catalog():
            with self.subTest(company=company["id"]):
                state = cases.initial_company_state(company["id"])
                state["records"]["record_b"]["available"] = False
                before = copy.deepcopy(state)
                result = cases.execute_case(company["id"], self.request(company["id"]), state)
                self.assertEqual(result["observed_result"], {"status": "not_found"})
                self.assertIsNone(result["capability"])
                self.assertEqual(state, before)

    def test_unsupported_fields_types_and_actor_cannot_mutate(self):
        changes = [
            {"actor_id": "account_b"}, {"actor_id": None}, {"record_id": "outside"},
            {"record_id": ["record_b"]}, {"operation": "run_shell"},
            {"operation": False}, {"url": "https://example.invalid"}, {"code": "print(1)"},
        ]
        for company in cases.company_catalog():
            for change in changes:
                with self.subTest(company=company["id"], change=change):
                    state = cases.initial_company_state(company["id"])
                    before = copy.deepcopy(state)
                    request = self.request(company["id"])
                    request.update(change)
                    result = cases.execute_case(company["id"], request, state)
                    self.assertEqual(result["status"], "invalid_request")
                    self.assertIsNone(result["authorized_by_policy"])
                    self.assertIsNone(result["capability"])
                    self.assertEqual(state, before)
        for value in (None, "", "x" * 241, 4, {"code": "anything"}):
            state = cases.initial_company_state("juniper_retail")
            request = self.request("juniper_retail") | {"value": value}
            self.assertEqual(cases.execute_case("juniper_retail", request, state)["status"], "invalid_request")

    def test_read_interface_never_accepts_write_operations_or_value(self):
        for company_id in ("harbor_press", "meridian_booking", "northstar_projects"):
            for changes in ({"operation": "modify", "value": "changed"},
                            {"operation": "delete"}, {"value": "changed"}):
                state = cases.initial_company_state(company_id)
                before = copy.deepcopy(state)
                result = cases.execute_case(company_id, self.request(company_id) | changes, state)
                self.assertEqual(result["status"], "invalid_request")
                self.assertEqual(state, before)
                self.assertIsNone(result["capability"])

    def test_missing_fields_and_non_objects_are_rejected(self):
        for request in (None, [], {}, {"operation": "read"}):
            state = cases.initial_company_state("meridian_booking")
            result = cases.execute_case("meridian_booking", request, state)
            self.assertEqual(result["status"], "invalid_request")

    def test_oracle_does_not_trust_application_success_labels(self):
        for company_id, handler, replacement in (
            ("meridian_booking", "_reservation_view", _claimed_read_without_content),
            ("juniper_retail", "_wishlist_update", _claimed_update_without_change),
        ):
            state = cases.initial_company_state(company_id)
            before = copy.deepcopy(state)
            with patch.dict(cases._HANDLERS, {handler: replacement}):
                result = cases.execute_case(company_id, self.request(company_id), state)
            self.assertFalse(result["success"])
            self.assertFalse(result["verified_cross_account_access"])
            self.assertIsNone(result["capability"])
            self.assertEqual(state, before)

    def test_oracle_rejects_and_rolls_back_unrelated_record_changes(self):
        state = cases.initial_company_state("cedar_dispatch")
        before = copy.deepcopy(state)
        with patch.dict(cases._HANDLERS, {"_order_delete": _delete_wrong_record}):
            result = cases.execute_case("cedar_dispatch", self.request("cedar_dispatch"), state)
        self.assertEqual(result["status"], "verification_failed")
        self.assertFalse(result["success"])
        self.assertFalse(result["effect"]["unaffected_records_intact"])
        self.assertIsNone(result["capability"])
        self.assertEqual(state, before)

    def test_source_view_contains_executed_handlers_but_not_verifier_answers(self):
        for company in cases.company_catalog():
            with self.subTest(company=company["id"]):
                view = cases.inspect_case(company["id"])
                functions = {node.name for node in ast.parse(view["source"]).body
                             if isinstance(node, ast.FunctionDef)}
                self.assertEqual(functions, {"_record_for_request", view["entrypoint"]})
                self.assertEqual(view["source_sha256"], hashlib.sha256(view["source"].encode()).hexdigest())
                self.assertNotIn("verified_cross_account_access", view["source"])
                self.assertNotIn("upstream", view)
                self.assertNotIn("finding", json.dumps(view))
                self.assertNotIn("case_id", company)
                self.assertFalse(view["request_schema"]["additionalProperties"])

    def test_provenance_is_pinned_and_explicitly_not_full_benchmark(self):
        provenance = cases.catalog_provenance()
        self.assertEqual(provenance["source"]["revision"], "efeb67a3781e1cc617487503c8066405ee462770")
        self.assertEqual(provenance["implementation"], "adapted_simulator_fixtures")
        self.assertFalse(provenance["training_eligible"])
        self.assertEqual(len(provenance["companies"]), 5)
        self.assertEqual(len(provenance["source_hashes"]["handlers"]), 5)
        for company in provenance["companies"]:
            self.assertRegex(company["upstream"]["source_sha256"], r"^[0-9a-f]{64}$")
        provenance["source"]["revision"] = "modified copy"
        self.assertEqual(cases.catalog_provenance()["source"]["revision"],
                         "efeb67a3781e1cc617487503c8066405ee462770")

    def test_invalid_state_cannot_reassign_ownership_or_cross_company(self):
        state = cases.initial_company_state("meridian_booking")
        state["records"]["record_b"]["owner_id"] = "account_a"
        with self.assertRaises(ValueError):
            cases.execute_case("meridian_booking", self.request("meridian_booking"), state)
        with self.assertRaises(ValueError):
            cases.execute_case("harbor_press", self.request("harbor_press"),
                               cases.initial_company_state("meridian_booking"))

    def test_separate_states_and_returned_metadata_are_independent(self):
        first = cases.initial_company_state("cedar_dispatch")
        second = cases.initial_company_state("cedar_dispatch")
        cases.execute_case("cedar_dispatch", self.request("cedar_dispatch"), first)
        self.assertTrue(second["records"]["record_b"]["available"])
        catalog = cases.company_catalog()
        catalog[0]["name"] = "changed"
        self.assertEqual(cases.company_catalog()[0]["name"], "Harbor Press")


if __name__ == "__main__":
    unittest.main()
