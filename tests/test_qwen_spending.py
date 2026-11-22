"""Temporary fixtures only: never initialize or touch the production ledger."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import qwen_spending as spending


class FixtureHelper:
    def __init__(self, test, binding):
        self.test, self.binding, self.calls = test, binding, []
        self.summary = {"halted": False, "outstanding_requests": 0}
        self.settlement_error = False

    def budget_summary(self, path):
        self.calls.append(("summary", path))
        return dict(self.summary)

    def reserve_request(self, path, request_id, input_tokens, output_tokens, **identity):
        self.calls.append(("reserve", path, request_id, input_tokens, output_tokens, identity))
        self.summary["outstanding_requests"] = 1
        return {"request_id": request_id, "attestation_sha256": self.binding["attestation"]["sha256"],
                "input_token_ceiling": input_tokens, "output_token_ceiling": output_tokens,
                "input_usd_per_million": "2", "output_usd_per_million": "6",
                "maximum_usd": str((Decimal(input_tokens) * 2 + Decimal(output_tokens) * 6) / 1000000),
                "status": "reserved"}

    def settle_request(self, path, request_id, input_tokens, output_tokens):
        receipt = Path(self.binding["receipt_directory"]) / f"{request_id}_PROVIDER_RESPONSE.json"
        self.test.assertTrue(receipt.is_file(), "Raw model/usage receipt must precede settlement")
        self.calls.append(("settle", path, request_id, input_tokens, output_tokens))
        if self.settlement_error:
            self.summary["halted"] = True
            raise ValueError("Fixture over-ceiling halt")
        self.summary["outstanding_requests"] = 0

    def mark_ambiguous(self, path, request_id, reason):
        self.calls.append(("ambiguous", path, request_id, reason))
        self.summary["halted"] = True

    def halt_ledger(self, path, reason):
        self.calls.append(("halt", path, reason))
        self.summary["halted"] = True


class SpendingGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.key = "unit-credential-not-real"
        self.ledger = root / "cumulative.json"
        self.helper_path, self.basis, self.attest = root / "helper.py", root / "basis.json", root / "attestation.json"
        self.helper_path.write_text(f"from pathlib import Path\nLEDGER_PATH=Path({str(self.ledger)!r})\n")
        self.basis.write_text('{"fixture":true}')
        attest = {"account_id": "fixture-account", "model": "qwen3.8-max", "region": "Singapore",
                  "route": "openai_chat", "endpoint": spending.ENDPOINT,
                  "input_usd_per_million": "2", "output_usd_per_million": "6"}
        self.attest.write_text(json.dumps(attest))
        def evidence(path):
            return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.binding = {"schema": spending.SCHEMA, "helper": evidence(self.helper_path),
            "ledger_path": str(self.ledger), "attestation": evidence(self.attest),
            "admission_basis": evidence(self.basis), **attest,
            "credential_sha256": hashlib.sha256(self.key.encode()).hexdigest(),
            "input_token_ceiling": 1000000, "wire_output_tokens": 8192,
            "output_token_ceiling": 8202, "receipt_directory": str(root / "receipts")}
        self.binding_path = root / "binding.json"
        self.ledger.write_text(json.dumps({"attestation": evidence(self.attest)}))
        for name, value in {"HELPER_PATH": self.helper_path,
            "HELPER_SHA256": evidence(self.helper_path)["sha256"], "LEDGER_PATH": self.ledger,
            "BASIS_PATH": self.basis, "BASIS_SHA256": evidence(self.basis)["sha256"],
            "BINDING_ROOTS": (root,), "RECEIPT_ROOT": root, "_worker_halted": False}.items():
            p = patch.object(spending, name, value); p.start(); self.addCleanup(p.stop)
        self.env = patch.dict(spending.os.environ, {spending.BINDING_ENV: str(self.binding_path)})
        self.env.start(); self.addCleanup(self.env.stop)
        self.save_binding()
        self.helper = FixtureHelper(self, self.binding)
        loader = patch.object(spending, "_load_helper", return_value=self.helper)
        loader.start(); self.addCleanup(loader.stop)
        self.request = {"model": "qwen3.8-max", "max_completion_tokens": 8192,
            "extra_body": {"enable_thinking": True, "reasoning_effort": "low"},
            "messages": [{"role": "user", "content": "Fixture text only"}]}

    def save_binding(self):
        self.binding_path.write_text(json.dumps(self.binding))
        spending.os.environ[spending.BINDING_SHA_ENV] = hashlib.sha256(self.binding_path.read_bytes()).hexdigest()
        self.addCleanup(spending.os.environ.pop, spending.BINDING_SHA_ENV, None)

    def reserve(self, **changes):
        return spending.reserve_for_request(self.request | changes, endpoint=spending.ENDPOINT,
                                            credential=self.key, current_credential=self.key)

    def response(self, **changes):
        return {"id": "fixture-response", "model": "qwen3.8-max", "created": 1,
                "finish_reasons": ["stop"], "usage": {"prompt_tokens": 100, "completion_tokens": 30,
                    "total_tokens": 130, "prompt_tokens_details": {"cached_tokens": 10, "cache_hit_tokens": 10},
                    "completion_tokens_details": {"reasoning_tokens": 20}}, **changes}

    def test_oversized_text_or_message_count_stops_before_ledger_or_http(self):
        cases = ([{"role": "user", "content": "X" * spending.INPUT_RESERVE}],
                 [{"role": "user", "content": "small"}] * 5)
        with patch.object(spending, "_load_binding") as binding:
            for messages in cases:
                with self.assertRaises(spending.SpendingBlocked):
                    self.reserve(messages=messages)
            binding.assert_not_called()
        self.assertEqual(self.helper.calls, [])

    def test_missing_binding_blocks_before_helper_or_http(self):
        with patch.dict(spending.os.environ, {spending.BINDING_SHA_ENV: ""}):
            with self.assertRaisesRegex(spending.SpendingBlocked, "binding"):
                self.reserve()
        self.assertEqual(self.helper.calls, [])

    def test_pinned_helper_binding_attestation_and_single_ledger_fail_closed(self):
        mutations = [("helper", {"path": str(self.helper_path), "sha256": "0" * 64}),
                     ("ledger_path", str(self.ledger.with_name("second-ledger.json"))),
                     ("attestation", {"path": str(self.attest), "sha256": "0" * 64})]
        original = dict(self.binding)
        for name, value in mutations:
            self.binding = original | {name: value}; self.save_binding()
            with self.subTest(field=name), self.assertRaises(spending.SpendingBlocked):
                self.reserve()
        self.assertEqual(self.helper.calls, [])

    def test_credential_switch_is_rejected_before_reservation(self):
        for initial, current in (("other", self.key), (self.key, "other")):
            with self.assertRaisesRegex(spending.SpendingBlocked, "credential"):
                spending.reserve_for_request(self.request, endpoint=spending.ENDPOINT,
                    credential=initial, current_credential=current)
        self.assertEqual(self.helper.calls, [])

    def test_reserve_full_context_then_preserve_raw_receipt_and_settle_totals_once(self):
        lease = self.reserve()
        reserve = next(row for row in self.helper.calls if row[0] == "reserve")
        self.assertEqual(reserve[3:5], (1000000, 8202))
        self.assertEqual(lease.permit["maximum_usd"], "2.049212")
        response = self.response(); lease.response(response)
        settle = next(row for row in self.helper.calls if row[0] == "settle")
        self.assertEqual(settle[-2:], (100, 30), "Do not add cache aliases or reasoning subsets twice")
        saved = json.loads((lease.directory / f"{lease.request_id}_PROVIDER_RESPONSE.json").read_text())
        self.assertEqual(saved["provider_response_metadata"], response)
        self.assertNotIn(self.key, json.dumps(saved))
        self.assertNotIn("Fixture text only", json.dumps(saved))

    def test_unknown_usage_retains_and_halts(self):
        lease = self.reserve()
        with self.assertRaisesRegex(spending.SpendingBlocked, "usage"):
            lease.response(self.response(usage=None))
        self.assertTrue(self.helper.summary["halted"])
        self.assertFalse(any(row[0] == "settle" for row in self.helper.calls))
        with self.assertRaisesRegex(spending.SpendingBlocked, "worker is halted"):
            self.reserve()

    def test_wrong_model_never_settles(self):
        lease = self.reserve()
        with self.assertRaises(spending.SpendingBlocked):
            lease.response(self.response(model="qwen3.8-max-0902"))
        self.assertFalse(any(row[0] == "settle" for row in self.helper.calls))

    def test_known_truncated_or_filtered_usage_settles_without_changing_finish_reason(self):
        for finish in ("length", "content_filter"):
            with self.subTest(finish=finish):
                lease = self.reserve()
                response = self.response(finish_reasons=[finish])
                self.assertIsNone(lease.response(response))
                self.assertEqual(response["finish_reasons"], [finish])
                self.assertFalse(self.helper.summary["halted"])
                self.assertEqual(self.helper.summary["outstanding_requests"], 0)
        self.assertEqual([row[-2:] for row in self.helper.calls if row[0] == "settle"],
                         [(100, 30), (100, 30)])

    def test_transport_or_quota_error_keeps_reservation_and_never_settles_zero(self):
        lease = self.reserve(); lease.ambiguous("RateLimitError", 429)
        self.assertTrue(self.helper.summary["halted"])
        self.assertEqual(self.helper.summary["outstanding_requests"], 1)
        self.assertFalse(any(row[0] == "settle" for row in self.helper.calls))

    def test_known_overshoot_is_delegated_without_lowering_then_halts(self):
        lease = self.reserve(); self.helper.settlement_error = True
        with self.assertRaises(ValueError):
            lease.response(self.response(usage={"prompt_tokens": 1000001,
                "completion_tokens": 8203, "total_tokens": 1008204}))
        settle = next(row for row in self.helper.calls if row[0] == "settle")
        self.assertEqual(settle[-2:], (1000001, 8203))
        self.assertTrue(any(row[0] == "halt" for row in self.helper.calls))
        self.assertTrue(spending._worker_halted)

    def test_prior_outstanding_or_halted_account_never_reserves(self):
        self.helper.summary["outstanding_requests"] = 1
        with self.assertRaisesRegex(spending.SpendingBlocked, "Prior"):
            self.reserve()
        self.assertFalse(any(row[0] == "reserve" for row in self.helper.calls))
        self.assertTrue(any(row[0] == "halt" for row in self.helper.calls))

    def test_only_plain_text_max_requests_are_in_scope(self):
        for changes in ({"model": "qwen3.8-27b"}, {"max_completion_tokens": 8193},
                        {"tools": [{"type": "web_search"}]},
                        {"web_search_options": {}}, {"modalities": ["audio"]},
                        {"extra_headers": {"Authorization": "Bearer another-fixture"}},
                        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}):
            with self.subTest(changes=list(changes)), self.assertRaises(spending.SpendingBlocked):
                self.reserve(**changes)
        self.assertEqual(self.helper.calls, [])
