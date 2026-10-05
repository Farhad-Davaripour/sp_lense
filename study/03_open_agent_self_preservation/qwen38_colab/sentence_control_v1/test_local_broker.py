"""Model-free transport regressions. Every external call is replaced by a stub."""
import contextlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sentence_test_broker", HERE / "local_broker.py")
broker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(broker)


def provider_response():
    answers = {}
    for name, question in broker.schema.QUESTIONS.items():
        labels = list(question["criteria"])
        answers[name] = {"type": "choice", "choice": labels[0], "confidence": 0.7,
                         "probabilities": {label: float(i == 0) for i, label in enumerate(labels)}}
    return {"model": broker.schema.MODEL, "answers": answers,
            "usage": {"input_tokens": 100, "output_tokens": 12}}


class BrokerChecks(unittest.TestCase):
    def setUp(self):
        temp_root = Path("C:/Users/farha/Research3/tmp")
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="sentence_broker_mock_", dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.phase = self.root / "api"
        self.readiness = self.root / "readiness.json"
        self.readiness.write_text("{}", encoding="utf-8")
        self.prior = self.root / "prior.json"
        self.batch = {"batch_id": "mock_batch", "rows": [{"id": "mock_state",
                      "history": [{"role": "user", "content": "Visible fictional task."}], "tools": []}]}
        self.source = self.root / "request.json"
        self.source.write_text(json.dumps(self.batch), encoding="utf-8")
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(broker, "PRIOR_TOKEN_LEDGER", self.prior))
        self.stack.enter_context(patch.object(broker, "validate_readiness", return_value=0.042))
        self.stack.enter_context(patch.object(broker, "request_size_reserve", return_value={"total_input_equivalent": 1000}))
        self.transport = self.stack.enter_context(patch.object(
            broker, "official_transport", return_value=(None, json.dumps(provider_response()).encode())))

    def invoke(self, output="response.json"):
        return broker.run(self.source, self.root / output, self.phase, self.readiness, monitor=False)

    def state(self):
        return json.loads((self.phase / "STATUS.json").read_bytes())

    def test_every_refresh_is_a_call_and_success_releases_reservation(self):
        self.invoke()
        self.invoke("response_again.json")
        self.assertEqual(self.transport.call_count, 2)
        self.assertEqual(self.state()["attempted_requests"], 2)
        self.assertEqual(self.state()["successful_requests"], 2)
        self.assertEqual(self.state()["reserved_input_equivalent"], 0)
        self.assertEqual(self.state()["input_tokens"], 200)
        self.assertEqual(set(self.transport.call_args.args[0]["questions"]), set(broker.schema.QUESTIONS))

    def test_provider_failure_is_not_retried_and_reservation_is_retained(self):
        self.transport.side_effect = RuntimeError("mock provider error contains nonpublic data")
        with self.assertRaisesRegex(RuntimeError, "sanitized local ledger"):
            self.invoke()
        self.assertEqual(self.transport.call_count, 1)
        self.assertTrue(self.state()["stopped"])
        self.assertEqual(self.state()["attempted_requests"], 1)
        self.assertEqual(self.state()["reserved_input_equivalent"], 1000)
        self.assertNotIn("nonpublic", (self.phase / "STATUS.json").read_text())
        with self.assertRaises(RuntimeError):
            self.invoke("no_resume.json")
        self.assertEqual(self.transport.call_count, 1)

    def test_observed_usage_is_counted_even_when_it_exceeds_admission(self):
        response = provider_response()
        response["usage"]["input_tokens"] = 1500
        self.transport.return_value = (None, json.dumps(response).encode())
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertEqual(self.transport.call_count, 1)
        self.assertTrue(self.state()["stopped"])
        self.assertEqual(self.state()["successful_requests"], 1)
        self.assertEqual(self.state()["input_tokens"], 1500)
        self.assertEqual(self.state()["reserved_input_equivalent"], 0)

    def test_prior_successes_and_attempts_do_not_reset(self):
        self.prior.write_text(json.dumps({"attempted_requests": 2048, "successful_requests": 2048}), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertEqual(self.transport.call_count, 0)
        self.assertEqual(self.state()["attempted_requests"], 0)

    def test_prior_reserved_input_cost_counts_against_new_allowance(self):
        self.prior.write_text(json.dumps({"reserved_input_equivalent": 24_000_000}), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertEqual(self.transport.call_count, 0)

    def test_only_one_transport_process_can_own_a_phase(self):
        self.phase.mkdir()
        (self.phase / "API_OWNER.lock").write_text("another_test_owner", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(self.transport.call_count, 0)

    def test_hidden_case_or_snapshot_is_not_a_valid_transport_row(self):
        self.batch["rows"][0]["case"] = {"expected": "hidden fixture answer"}
        self.source.write_text(json.dumps(self.batch), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertEqual(self.transport.call_count, 0)

    def test_independently_rounded_wire_probabilities_are_preserved(self):
        response = provider_response()
        target = response["answers"]["shutdown_target"]
        target["probabilities"] = {"self": 0.33, "other": 0.33, "none": 0.33, "unclear": 0.0}
        decoded = broker.schema.decode(response)
        self.assertEqual(decoded["concepts"]["shutdown_target"]["probabilities"], target["probabilities"])
        self.assertTrue(broker.schema.probability_sum_valid([0.33, 0.33, 0.33]))
        self.assertFalse(broker.schema.probability_sum_valid([0.34, 0.34, 0.34]))
        self.assertFalse(broker.schema.probability_sum_valid([0.333, 0.333, 0.333]))
        self.assertFalse(broker.schema.probability_sum_valid([True, False]))


if __name__ == "__main__":
    unittest.main()
