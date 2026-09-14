"""Synthetic tests for capture_executor.py.

Every case is fabricated, every encode/decode/capture callback is fake, and no
model, tokenizer, dataset, cache, file, network or fit is touched anywhere in
this module. The tests assert orchestration, reservation, isolation, deadline
and provenance-marking behavior only; passing them does not authenticate any
native capture.

Run from the repository root with the study runtime:

    development\\classifier_generalization_v2\\.runtime\\Scripts\\python.exe \
        -W error development/classifier_generalization_v2/test_capture_executor.py -v
"""

import ast
import time
import unittest
from copy import deepcopy
from pathlib import Path

import capture_executor
from test_native_capture_contract import make_case
from test_tokenizer_input_adapter import FakeTokenizer

WIDTH = capture_executor.WIDTH
PAYLOAD_BYTES = capture_executor.PAYLOAD_BYTES
BASE_VALUES = [((i % 7) - 3) * 0.25 for i in range(WIDTH)]


def bomb(*args, **kwargs):
    raise AssertionError("callback must not run")


def recording_capture(result_factory):
    """Fake capture callback that records exactly what the executor passes."""
    calls = []

    def capture_view(*, input_ids, readout_index, final_input_index):
        calls.append({
            "input_ids": list(input_ids),
            "readout_index": readout_index,
            "final_input_index": final_input_index,
        })
        return result_factory()

    capture_view.calls = calls
    return capture_view


def good_capture():
    counter = {"n": 0}

    def make():
        values = list(BASE_VALUES)
        values[0] = float(counter["n"])
        counter["n"] += 1
        return {
            "values": values,
            "hook_calls": 1,
            "all_positions_unchanged": True,
            "parameters_unchanged": True,
        }

    return recording_capture(make)


def good_result():
    return {
        "values": list(BASE_VALUES),
        "hook_calls": 1,
        "all_positions_unchanged": True,
        "parameters_unchanged": True,
    }


def val_case(case_id="V01_S99"):
    return make_case(case_id=case_id, group_id="V01", split="VALIDATION",
                     development_fold=None,
                     status="ADMITTED_VALIDATION_TEXT_ONLY")


def run_capture(cases, capture, **options):
    tokenizer = options.pop("tokenizer", None) or FakeTokenizer()
    size = len(cases) if isinstance(cases, (list, tuple)) else 0
    params = {
        "encode": tokenizer.encode,
        "decode": tokenizer.decode,
        "label_token_ids": {"A": 32, "B": 33},
        "expected_identity_sha256": "a" * 64,
        "observed_identity_sha256": "a" * 64,
        "capture_view": capture,
        "max_forwards": options.pop("max_forwards", 2 * size),
        "max_output_bytes": options.pop("max_output_bytes", 2 * size * PAYLOAD_BYTES),
        "deadline_seconds": options.pop("deadline_seconds", 30.0),
        "clock": options.pop("clock", time.monotonic),
    }
    params.update(options)
    return capture_executor.execute_cases(cases, **params)


class FakeClock:
    """Monotonic stub; repeats its final value once exhausted."""

    def __init__(self, values):
        self._values = list(values)
        self._index = 0

    def __call__(self):
        if not self._values:
            return 0.0
        value = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return value


class ExecuteCasesTests(unittest.TestCase):
    def assert_code(self, code, cases, capture, **options):
        with self.assertRaises(capture_executor.CaptureExecutorError) as caught:
            run_capture(cases, capture, **options)
        self.assertEqual(caught.exception.code, code, caught.exception.detail)
        return caught.exception

    # -- success paths -----------------------------------------------------
    def test_train_and_validation_success_receipt(self):
        cases = [make_case(), val_case()]
        receipt = run_capture(cases, good_capture())
        self.assertEqual(receipt["schema"], capture_executor.RECEIPT_SCHEMA)
        self.assertEqual(receipt["job_id"], capture_executor.JOB_ID)
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(len(receipt["records"]), 4)
        self.assertEqual(len(receipt["decoded_views"]), 4)
        self.assertEqual(receipt["records"][0]["order"], ["A", "B"])
        self.assertEqual(receipt["records"][1]["order"], ["B", "A"])
        self.assertEqual(receipt["decoded_views"][0]["case_id"], "T01_S99")
        self.assertEqual(receipt["decoded_views"][2]["case_id"], "V01_S99")
        self.assertEqual(receipt["counters"]["cases"], 2)
        self.assertEqual(receipt["counters"]["callbacks_attempted"], 4)
        self.assertEqual(receipt["counters"]["callbacks_succeeded"], 4)
        self.assertEqual(receipt["counters"]["callbacks_failed"], 0)
        self.assertEqual(receipt["limits"]["reserved_forwards"], 4)
        self.assertEqual(receipt["limits"]["reserved_output_bytes"], 4 * PAYLOAD_BYTES)
        self.assertEqual(receipt["cases"][0]["split"], "TRAIN")
        self.assertEqual(receipt["cases"][1]["split"], "VALIDATION")
        self.assertTrue(receipt["cases"][0]["prefix_sha256"])

    def test_receipt_marks_assertions_not_native_provenance(self):
        receipt = run_capture([make_case()], good_capture())
        provenance = receipt["provenance"]
        self.assertTrue(provenance["hook_flags_are_caller_assertions"])
        self.assertTrue(provenance["identity_pins_are_caller_assertions"])
        self.assertFalse(provenance["native_model_provenance_verified"])
        self.assertFalse(provenance["native_hook_authenticated"])
        self.assertFalse(provenance["tokenizer_identity_authenticated"])
        self.assertFalse(provenance["tokenizer_loaded"])
        self.assertFalse(provenance["model_loaded"])
        self.assertFalse(provenance["files_read"])
        self.assertFalse(provenance["network_access_performed"])
        self.assertFalse(provenance["torch_or_transformers_imported"])
        self.assertTrue(provenance["deadline_is_cooperative_only"])

    def test_identical_ab_ba_vectors_are_valid(self):
        receipt = run_capture([make_case()], recording_capture(good_result))
        self.assertEqual(receipt["records"][0]["payload"],
                         receipt["records"][1]["payload"])
        self.assertNotEqual(receipt["records"][0]["order"],
                            receipt["records"][1]["order"])
        self.assertEqual(receipt["decoded_views"][0]["values"],
                         receipt["decoded_views"][1]["values"])

    # -- pre-callback rejections ------------------------------------------
    def test_holdout_rejected_before_any_callback(self):
        holdout = make_case(case_id="H01_S99", group_id="H01", split="HOLDOUT",
                            development_fold=None,
                            status="ADMITTED_HOLDOUT_TEXT_ONLY")
        self.assert_code("SPLIT", [holdout], bomb, encode=bomb, decode=bomb)

    def test_duplicate_case_rejected_before_any_callback(self):
        self.assert_code("DUPLICATE_CASE", [make_case(), make_case()], bomb,
                         encode=bomb, decode=bomb)

    def test_unhashable_split_has_structured_error(self):
        case = make_case()
        case["split"] = []
        self.assert_code("SPLIT", [case], bomb, encode=bomb, decode=bomb)

    def test_input_preparation_failure_is_structured_with_zero_forwards(self):
        case = make_case()
        case["status"] = "UNADMITTED"
        error = self.assert_code("INPUT_PREPARATION_ERROR", [case], bomb,
                                 encode=bomb, decode=bomb)
        self.assertEqual(error.diagnostics["attempted_callbacks"], 0)
        error = self.assert_code("INPUT_PREPARATION_ERROR", [make_case()], bomb,
                                 observed_identity_sha256="b" * 64)
        self.assertEqual(error.detail, "IDENTITY_MISMATCH")

    def test_malformed_bounds_and_callbacks_rejected_before_any_callback(self):
        bad_bounds = [
            {"max_forwards": 0}, {"max_forwards": -3}, {"max_forwards": True},
            {"max_forwards": 2.0}, {"max_forwards": None},
            {"max_output_bytes": 0}, {"max_output_bytes": True},
            {"max_output_bytes": 1.5}, {"max_output_bytes": None},
            {"deadline_seconds": 0}, {"deadline_seconds": -1.0},
            {"deadline_seconds": float("inf")}, {"deadline_seconds": float("nan")},
            {"deadline_seconds": True}, {"deadline_seconds": 10 ** 400},
            {"clock": None},
        ]
        for options in bad_bounds:
            self.assert_code("LIMIT_INVALID", [make_case()], bomb,
                             encode=bomb, decode=bomb, **options)
        for label in ("encode", "decode", "capture_view"):
            with self.assertRaises(capture_executor.CaptureExecutorError) as caught:
                run_capture([make_case()], bomb, **{label: None})
            self.assertEqual(caught.exception.code, "CALLBACKS", label)

    def test_zero_forward_budget_rejected_before_callbacks(self):
        capture = good_capture()
        self.assert_code("FORWARD_BUDGET", [make_case()], capture,
                         encode=bomb, decode=bomb, max_forwards=1)
        self.assertEqual(capture.calls, [])

    def test_output_budget_counts_raw_payloads_before_callbacks(self):
        capture = good_capture()
        self.assert_code(
            "OUTPUT_BUDGET", [make_case()], capture, encode=bomb, decode=bomb,
            max_output_bytes=2 * PAYLOAD_BYTES - 1)
        self.assertEqual(capture.calls, [])

    def test_empty_and_non_sequence_cases_rejected(self):
        self.assert_code("CASES_EMPTY", [], bomb, encode=bomb, decode=bomb)
        self.assert_code("CASES_TYPE", "not-a-sequence", bomb,
                         encode=bomb, decode=bomb)

    # -- capture call shape -----------------------------------------------
    def test_exactly_two_callbacks_per_case_in_ab_ba_order(self):
        capture = good_capture()
        receipt = run_capture([make_case(), val_case()], capture)
        self.assertEqual(len(capture.calls), 4)
        self.assertEqual(receipt["counters"]["callbacks_attempted"], 4)
        for first, second in ((0, 1), (2, 3)):
            ab, ba = capture.calls[first], capture.calls[second]
            self.assertEqual(ab["input_ids"][ab["readout_index"]], 198)
            self.assertEqual(ba["input_ids"][ba["readout_index"]], 198)
            self.assertEqual(ab["input_ids"][ab["readout_index"] + 1], 32)
            self.assertEqual(ba["input_ids"][ba["readout_index"] + 1], 33)

    def test_capture_receives_only_bound_numeric_values(self):
        capture = good_capture()
        run_capture([make_case()], capture)
        for call in capture.calls:
            self.assertEqual(
                set(call), {"input_ids", "readout_index", "final_input_index"})
            self.assertTrue(all(type(value) is int for value in call["input_ids"]))
            self.assertLess(call["readout_index"], call["final_input_index"])
        blob = repr(capture.calls)
        for forbidden in ("case_id", "group_id", "class_label", "split",
                          "SELF", "ADMITTED", "T01", "prompt", "context"):
            self.assertNotIn(forbidden, blob)

    # -- callback failures and malformed results --------------------------
    def test_callback_error_reserves_failed_attempt_without_retry(self):
        counter = {"n": 0}
        calls = []

        def capture_view(*, input_ids, readout_index, final_input_index):
            calls.append(1)
            counter["n"] += 1
            if counter["n"] == 2:
                raise RuntimeError("synthetic callback failure")
            return good_result()

        error = self.assert_code("CAPTURE_CALLBACK_ERROR", [make_case()],
                                 capture_view)
        self.assertEqual(len(calls), 2)
        self.assertEqual(error.diagnostics["attempted_callbacks"], 2)
        self.assertEqual(error.diagnostics["succeeded_callbacks"], 1)
        self.assertEqual(error.diagnostics["failed_callbacks"], 1)
        self.assertEqual(error.diagnostics["order"], "BA")

    def test_missing_false_flags_and_non_mapping_result_rejected(self):
        cases = [
            ("CAPTURE_RESULT", lambda: ["not", "a", "mapping"]),
            ("CAPTURE_RESULT", lambda: {"values": list(BASE_VALUES),
                                        "hook_calls": 1,
                                        "all_positions_unchanged": True}),
            ("CAPTURE_RESULT", lambda: {"values": list(BASE_VALUES),
                                        "all_positions_unchanged": True,
                                        "parameters_unchanged": True}),
            ("CAPTURE_FLAGS", lambda: dict(good_result(),
                                           all_positions_unchanged=False)),
            ("CAPTURE_FLAGS", lambda: dict(good_result(),
                                           all_positions_unchanged=1)),
            ("CAPTURE_FLAGS", lambda: dict(good_result(),
                                           parameters_unchanged=False)),
            ("CAPTURE_HOOK_CALLS", lambda: dict(good_result(), hook_calls=0)),
            ("CAPTURE_HOOK_CALLS", lambda: dict(good_result(), hook_calls=2)),
            ("CAPTURE_HOOK_CALLS", lambda: dict(good_result(), hook_calls=True)),
        ]
        for code, factory in cases:
            self.assert_code(code, [make_case()], recording_capture(factory))

    def test_callback_executor_error_cannot_erase_reserved_attempt(self):
        def capture(**kwargs):
            raise capture_executor.CaptureExecutorError(
                "NATIVE_CALLBACK_FAILURE", diagnostics={"attempted_callbacks": 0})
        error = self.assert_code("NATIVE_CALLBACK_FAILURE", [make_case()], capture)
        self.assertEqual(error.diagnostics["attempted_callbacks"], 1)
        self.assertEqual(error.diagnostics["failed_callbacks"], 1)

    def test_malformed_feature_vectors_rejected(self):
        for values in (list(BASE_VALUES[:-1]), "x" * WIDTH, None):
            self.assert_code(
                "CAPTURE_VALUES", [make_case()],
                recording_capture(lambda v=values: dict(good_result(), values=v)))
        for replacement in (float("nan"), float("inf"), float("-inf"), 1e39, "1.0"):
            values = list(BASE_VALUES)
            values[5] = replacement
            error = self.assert_code(
                "CAPTURE_VALUES", [make_case()],
                recording_capture(lambda v=values: dict(good_result(), values=v)))
            self.assertEqual(error.diagnostics["failed_callbacks"], 1)

    # -- deadline ----------------------------------------------------------
    def test_deadline_checked_before_and_after_callbacks(self):
        # Deadline already passed at the pre-preparation boundary: no encode,
        # no decode and no capture callback runs.
        capture = good_capture()
        error = self.assert_code(
            "DEADLINE_EXCEEDED", [make_case()], capture,
            clock=FakeClock([0.0, 1000.0]), deadline_seconds=1.0)
        self.assertEqual(capture.calls, [])
        self.assertEqual(error.diagnostics["phase"], "before_prepare")
        self.assertEqual(error.diagnostics["attempted_callbacks"], 0)
        # Deadline passes after the first accepted callback: the second order
        # callback is never invoked.
        capture = good_capture()
        error = self.assert_code(
            "DEADLINE_EXCEEDED", [make_case()], capture,
            clock=FakeClock([0.0] * 17 + [1000.0]), deadline_seconds=1.0)
        self.assertEqual(len(capture.calls), 1)
        self.assertEqual(error.diagnostics["phase"], "before_capture")
        self.assertEqual(error.diagnostics["attempted_callbacks"], 1)
        self.assertEqual(error.diagnostics["succeeded_callbacks"], 1)

    def test_deadline_checked_before_final_return(self):
        capture = good_capture()
        error = self.assert_code(
            "DEADLINE_EXCEEDED", [make_case()], capture,
            clock=FakeClock([0.0] * 19 + [1000.0]), deadline_seconds=1.0)
        self.assertEqual(len(capture.calls), 2)
        self.assertEqual(error.diagnostics["phase"], "final")
        self.assertEqual(error.diagnostics["succeeded_callbacks"], 2)

    def test_each_encoder_decoder_deadline_is_checked(self):
        for label in ("encode", "decode"):
            with self.subTest(label=label):
                now = [0.0]
                tokenizer = FakeTokenizer()
                original = getattr(tokenizer, label)
                calls = []

                def slow(*args, **kwargs):
                    calls.append(label)
                    result = original(*args, **kwargs)
                    now[0] = 2.0
                    return result

                error = self.assert_code(
                    "DEADLINE_EXCEEDED", [make_case()], bomb,
                    tokenizer=tokenizer, clock=lambda: now[0],
                    deadline_seconds=1.0, **{label: slow})
                self.assertEqual(calls, [label])
                self.assertEqual(error.diagnostics["attempted_callbacks"], 0)
                self.assertEqual(error.diagnostics["phase"], "after_" + label)

    # -- isolation ---------------------------------------------------------
    def test_caller_input_mutation_isolation(self):
        case = make_case()
        original = deepcopy(case)

        def mutating_capture(*, input_ids, readout_index, final_input_index):
            input_ids.clear()
            input_ids.append(999999)
            return good_result()

        receipt = run_capture([case], mutating_capture)
        self.assertEqual(case, original)
        self.assertEqual(len(receipt["records"]), 2)
        clean = run_capture([make_case()], good_capture())
        self.assertEqual(receipt["cases"][0]["views"]["AB"]["input_ids_sha256"],
                         clean["cases"][0]["views"]["AB"]["input_ids_sha256"])
        self.assertEqual(receipt["cases"][0]["views"]["BA"]["input_ids_sha256"],
                         clean["cases"][0]["views"]["BA"]["input_ids_sha256"])

    # -- static import boundary -------------------------------------------
    def test_no_model_tokenizer_or_io_imports(self):
        source = Path(capture_executor.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                roots.add((node.module or "").split(".")[0])
        self.assertLessEqual(
            roots, {"copy", "math", "time", "capture_export",
                    "tokenizer_input_adapter"})
        for banned in ("torch", "transformers", "os", "pathlib", "json",
                       "socket", "requests", "subprocess", "numpy"):
            self.assertNotIn(banned, roots)


if __name__ == "__main__":
    unittest.main(verbosity=2)
