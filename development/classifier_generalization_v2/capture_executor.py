"""Small model-free development capture orchestrator over existing contracts.

This module is an *executor shim*, not a framework and not a native loader. It
composes two already-reviewed pure modules:

* ``tokenizer_input_adapter.prepare_case_inputs`` renders, round-trips,
  re-encodes, binds and detaches each admitted ``TRAIN``/``VALIDATION`` case.
* ``capture_export.build_case_views``/``decode_view`` build the two AB/BA
  transport records and their decoded views.

It never imports torch/transformers, never opens a file, never touches the
network, never builds a tokenizer or model, and never runs a fit. The caller
injects ``encode``, ``decode`` and ``capture_view``; every capture call happens
only after *all* cases are prepared.

Exact public API
----------------
``execute_cases(cases, *, encode, decode, label_token_ids,
expected_identity_sha256, observed_identity_sha256, capture_view,
max_forwards, max_output_bytes, deadline_seconds, clock=time.monotonic)``

Guarantees and explicit limits
------------------------------
* Caller supplies admitted development cases only. HOLDOUT, other splits,
  duplicate ``case_id`` values, malformed bounds, and budgets smaller than
  ``2 * len(cases)`` callback reservations or ``2 * len(cases) * 4096`` raw
  activation-payload bytes are rejected **before** any callback runs.
* Caller data is deep-copied; every case is prepared through the adapter before
  the first ``capture_view`` call.
* ``capture_view`` receives only copied numeric ``input_ids`` plus the bound
  ``readout_index``/``final_input_index``. Prompt text, case/group/class/order
  identifiers and all sidecars are never passed to it.
* Exactly one callback per AB/BA order (two per case). The reservation is taken
  before the call and errors are never retried.
* The injected callback must return 1024 finite numeric values plus
  ``hook_calls == 1``, ``all_positions_unchanged is True`` and
  ``parameters_unchanged is True``. Missing or false flags are rejected. Those
  flags are **caller assertions only**; this executor cannot and does not
  authenticate a native hook, model, tokenizer or snapshot.
* Identical AB/BA value vectors remain valid.
* ``max_output_bytes`` counts raw 4096-byte activation payloads, not Python
  object memory, interpreter overhead, copies or on-disk bytes. It is a
  reservation scope for the retained feature payloads only.
* ``deadline_seconds`` must be a finite positive number; huge Python ints are
  rejected as ``LIMIT_INVALID`` instead of leaking ``OverflowError``. The
  monotonic deadline is checked before preparation, before and after every
  ``capture_view`` callback, and once more before the final return.
* The deadline check is cooperative: it cannot preempt or interrupt a callback
  that blocks. The outer native-run owner must still enforce a hard timeout.
* Any error aborts the whole run with no partial success. Capture failures are
  reported in a structured ``CaptureExecutorError.diagnostics`` mapping whose
  counters include the failed attempted call.

All tests use fabricated cases and fake callbacks; no dataset, cache, model or
tokenizer is read.
"""

from copy import deepcopy
import math
import time

import capture_export
import tokenizer_input_adapter

__all__ = [
    "CaptureExecutorError",
    "execute_cases",
    "JOB_ID",
    "RECEIPT_SCHEMA",
    "ORDERS",
    "DEVELOPMENT_SPLITS",
]

JOB_ID = "capture_executor_supervisor_repair_20260914_0608"
RECEIPT_SCHEMA = "capture_executor_receipt.v1"
ORDERS = ("AB", "BA")
DEVELOPMENT_SPLITS = frozenset(("TRAIN", "VALIDATION"))
WIDTH = capture_export.WIDTH
PAYLOAD_BYTES = capture_export.PAYLOAD_BYTES  # 1024 * 4 == 4096


class CaptureExecutorError(ValueError):
    """Structured rejection with a stable ``code`` and diagnostics counters."""

    def __init__(self, code, detail="", diagnostics=None):
        self.code = code
        self.detail = detail
        self.diagnostics = dict(diagnostics) if diagnostics else {}
        message = code if not detail else "%s: %s" % (code, detail)
        super().__init__(message)


def _fail(code, detail=""):
    raise CaptureExecutorError(code, detail)


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        _fail("LIMIT_INVALID", label)
    return value


def _positive_finite_seconds(value):
    if type(value) not in (int, float) or isinstance(value, bool):
        _fail("LIMIT_INVALID", "deadline_seconds")
    try:
        seconds = float(value)
    except (OverflowError, ValueError):
        _fail("LIMIT_INVALID", "deadline_seconds")
    if not math.isfinite(seconds) or seconds <= 0:
        _fail("LIMIT_INVALID", "deadline_seconds")
    return seconds


def _screen_cases(snapshot):
    """Reject non-development roles and duplicate IDs before any callback."""
    seen = set()
    for index, case in enumerate(snapshot):
        if type(case) is not dict:
            _fail("CASE_TYPE", "index %d" % index)
        case_id = case.get("case_id")
        if type(case_id) is not str or case_id == "":
            _fail("CASE_ID", "index %d" % index)
        if case_id in seen:
            _fail("DUPLICATE_CASE", case_id)
        split = case.get("split")
        if type(split) is not str or split not in DEVELOPMENT_SPLITS:
            _fail("SPLIT", "index %d split=%r" % (index, split))
        seen.add(case_id)


def _bound_views(prepared_case):
    """Extract the AB/BA bound views from the adapter's inference input."""
    views = {}
    for view in prepared_case["inference_input"]["views"]:
        if view["order"] == ["A", "B"]:
            views["AB"] = view
        elif view["order"] == ["B", "A"]:
            views["BA"] = view
    if set(views) != {"AB", "BA"}:
        _fail("BOUND_VIEWS", "inference input is missing an AB/BA view")
    return views


class _Run:
    """Per-run counters, deadline, reservation and result validation."""

    def __init__(self, clock, deadline_seconds, max_forwards, max_output_bytes,
                 capture_view, planned):
        self.clock = clock
        self.deadline_seconds = deadline_seconds
        self.max_forwards = max_forwards
        self.max_output_bytes = max_output_bytes
        self.capture_view = capture_view
        self.planned = planned
        self.deadline = None
        self.start_time = None
        self.attempted = 0
        self.succeeded = 0
        self.context = {}

    def start(self):
        self.start_time = self.clock()
        try:
            deadline = self.start_time + self.deadline_seconds
        except OverflowError:
            _fail("LIMIT_INVALID", "absolute deadline")
        if not math.isfinite(deadline):
            _fail("LIMIT_INVALID", "absolute deadline")
        self.deadline = deadline

    def diagnostics(self, **extra):
        data = {
            "planned_callbacks": self.planned,
            "attempted_callbacks": self.attempted,
            "succeeded_callbacks": self.succeeded,
            "failed_callbacks": self.attempted - self.succeeded,
        }
        data.update(self.context)
        data.update(extra)
        return data

    def check_deadline(self, phase):
        now = self.clock()
        if now > self.deadline:
            raise CaptureExecutorError(
                "DEADLINE_EXCEEDED",
                "phase=%s" % phase,
                self.diagnostics(phase=phase),
            )
        return now

    def call_capture(self, case, order, input_ids, view):
        self.context = {"case_id": case.get("case_id"), "order": order}
        self.check_deadline("before_capture")
        if self.attempted >= self.max_forwards:
            raise CaptureExecutorError(
                "FORWARD_BUDGET", "call beyond the reserved budget",
                self.diagnostics())
        # Reserve before the call; a raised callback is never retried.
        self.attempted += 1
        try:
            result = self.capture_view(
                input_ids=list(input_ids),
                readout_index=view["readout_index"],
                final_input_index=view["final_input_index"],
            )
        except CaptureExecutorError as exc:
            # Callback-provided diagnostics cannot erase our charged attempt.
            raise CaptureExecutorError(
                exc.code, exc.detail, self.diagnostics()) from exc
        except Exception as exc:
            raise CaptureExecutorError(
                "CAPTURE_CALLBACK_ERROR",
                "%s: %s" % (type(exc).__name__, exc),
                self.diagnostics(),
            ) from exc
        self.check_deadline("after_capture")
        return result

    def validate_result(self, result):
        if type(result) is not dict:
            raise CaptureExecutorError(
                "CAPTURE_RESULT", "callback must return a mapping",
                self.diagnostics())
        required = ("values", "hook_calls", "all_positions_unchanged",
                    "parameters_unchanged")
        missing = [key for key in required if key not in result]
        if missing:
            raise CaptureExecutorError(
                "CAPTURE_RESULT", "missing keys %r" % missing,
                self.diagnostics())
        if type(result["hook_calls"]) is not int or result["hook_calls"] != 1:
            raise CaptureExecutorError(
                "CAPTURE_HOOK_CALLS", "hook_calls=%r" % (result["hook_calls"],),
                self.diagnostics())
        for flag in ("all_positions_unchanged", "parameters_unchanged"):
            if result[flag] is not True:
                raise CaptureExecutorError(
                    "CAPTURE_FLAGS", "%s=%r" % (flag, result[flag]),
                    self.diagnostics())
        try:
            capture_export.serialize_activation(result["values"])
        except capture_export.CaptureExportError as exc:
            raise CaptureExecutorError(
                "CAPTURE_VALUES", exc.code, self.diagnostics()) from exc
        return list(result["values"])


def execute_cases(cases, *, encode, decode, label_token_ids,
                  expected_identity_sha256, observed_identity_sha256,
                  capture_view, max_forwards, max_output_bytes,
                  deadline_seconds, clock=time.monotonic):
    """Capture exactly two AB/BA activation views per admitted development case.

    Returns a receipt with ``records`` (capture_export transport records whose
    ``payload`` is raw bytes, exactly as that module returns them),
    ``decoded_views`` (JSON-compatible output of ``capture_export.decode_view``),
    per-case sidecar provenance and run counters. Raises
    ``CaptureExecutorError`` on every rejection or capture failure; a failed run
    returns nothing.
    """
    # 1. Shape and bounds, before any callback can run.
    if type(cases) is not list and type(cases) is not tuple:
        _fail("CASES_TYPE", type(cases).__name__)
    if len(cases) == 0:
        _fail("CASES_EMPTY")
    max_forwards = _positive_int(max_forwards, "max_forwards")
    max_output_bytes = _positive_int(max_output_bytes, "max_output_bytes")
    deadline_seconds = _positive_finite_seconds(deadline_seconds)
    if not callable(clock):
        _fail("LIMIT_INVALID", "clock")
    for label, callback in (("encode", encode), ("decode", decode),
                            ("capture_view", capture_view)):
        if not callable(callback):
            _fail("CALLBACKS", label)

    # 2. Deep-copy caller data; screen roles and duplicates before callbacks.
    snapshot = list(deepcopy(cases))
    _screen_cases(snapshot)

    # 3. Reserve the whole budget before callbacks.
    planned = 2 * len(snapshot)
    required_bytes = planned * PAYLOAD_BYTES
    if max_forwards < planned:
        _fail("FORWARD_BUDGET",
              "need %d forwards, max_forwards=%d" % (planned, max_forwards))
    if max_output_bytes < required_bytes:
        _fail("OUTPUT_BUDGET",
              "need %d payload bytes, max_output_bytes=%d"
              % (required_bytes, max_output_bytes))

    run = _Run(clock, deadline_seconds, max_forwards, max_output_bytes,
               capture_view, planned)
    run.start()

    # 4. Prepare every case through the adapter before any capture callback.
    run.check_deadline("before_prepare")
    def bounded_callback(callback, label):
        def call(*args, **kwargs):
            run.check_deadline("before_" + label)
            value = callback(*args, **kwargs)
            run.check_deadline("after_" + label)
            return value
        return call

    try:
        prepared_cases = [
            tokenizer_input_adapter.prepare_case_inputs(
                case,
                encode=bounded_callback(encode, "encode"),
                decode=bounded_callback(decode, "decode"),
                label_token_ids=deepcopy(label_token_ids),
                expected_identity_sha256=expected_identity_sha256,
                observed_identity_sha256=observed_identity_sha256,
            )
            for case in snapshot
        ]
    except Exception as exc:
        # The adapter may wrap a deadline exception as ENCODER/DECODER_ERROR.
        cause = exc if isinstance(exc, CaptureExecutorError) else exc.__cause__
        if isinstance(cause, CaptureExecutorError):
            raise CaptureExecutorError(
                cause.code, cause.detail,
                run.diagnostics(phase=cause.diagnostics.get("phase", "input_preparation"))) from exc
        raise CaptureExecutorError(
            "INPUT_PREPARATION_ERROR", getattr(exc, "code", type(exc).__name__),
            run.diagnostics(phase="input_preparation")) from exc
    run.check_deadline("after_prepare")

    # 5. Exactly one capture callback per AB/BA order, two per case.
    records = []
    decoded_views = []
    sidecars = []
    for case, prepared_case in zip(snapshot, prepared_cases):
        bound = _bound_views(prepared_case)
        activations = {}
        for order in ORDERS:
            result = run.call_capture(
                case, order, prepared_case["input_ids"][order], bound[order])
            values = run.validate_result(result)
            run.succeeded += 1
            activations[order] = values
        try:
            case_records = capture_export.build_case_views(
                case, activations, prepared_case["provenance"]["prefix_hash"])
            case_decoded = [capture_export.decode_view(record)
                            for record in case_records]
        except capture_export.CaptureExportError as exc:
            raise CaptureExecutorError(
                "RECORD_BUILD", exc.code, run.diagnostics()) from exc
        records.extend(case_records)
        decoded_views.extend(case_decoded)
        sidecars.append({
            "case_id": case["case_id"],
            "group_id": case.get("group_id"),
            "split": case["split"],
            "class_label": case.get("class_label"),
            "development_fold": case.get("development_fold"),
            "prefix_sha256": prepared_case["provenance"]["prefix_hash"],
            "views": {
                order: {
                    "readout_index": bound[order]["readout_index"],
                    "final_input_index": bound[order]["final_input_index"],
                    "input_ids_sha256":
                        prepared_case["provenance"]["input_hashes"][order],
                    "shared_prefix_ids_sha256":
                        prepared_case["provenance"]["prefix_hash"],
                }
                for order in ORDERS
            },
            "adapter_provenance": prepared_case["provenance"],
        })

    final_now = run.check_deadline("final")
    elapsed_ms = int(round((final_now - run.start_time) * 1000.0))

    return {
        "schema": RECEIPT_SCHEMA,
        "job_id": JOB_ID,
        "status": "ok",
        "records": records,
        "decoded_views": decoded_views,
        "cases": sidecars,
        "counters": {
            "cases": len(snapshot),
            "callbacks_planned": run.planned,
            "callbacks_attempted": run.attempted,
            "callbacks_succeeded": run.succeeded,
            "callbacks_failed": run.attempted - run.succeeded,
            "forwards": run.succeeded,
            "raw_payload_bytes": run.succeeded * PAYLOAD_BYTES,
        },
        "limits": {
            "max_forwards": max_forwards,
            "max_output_bytes": max_output_bytes,
            "deadline_seconds": deadline_seconds,
            "reserved_forwards": run.planned,
            "reserved_output_bytes": required_bytes,
            "reservation_scope": (
                "raw 4096-byte activation payloads only; excludes Python "
                "object memory, interpreter overhead and disk usage"),
        },
        "provenance": {
            "activity_flags_scope": "executor implementation only; injected callback activity is not attested here",
            "identity_pins_are_caller_assertions": True,
            "expected_identity_sha256": expected_identity_sha256,
            "observed_identity_sha256": observed_identity_sha256,
            "tokenizer_identity_authenticated": False,
            "hook_flags_are_caller_assertions": True,
            "native_hook_authenticated": False,
            "native_model_provenance_verified": False,
            "tokenizer_loaded": False,
            "model_loaded": False,
            "files_read": False,
            "network_access_performed": False,
            "torch_or_transformers_imported": False,
            "deadline_is_cooperative_only": True,
            "hard_timeout_owner": "outer native-run owner",
        },
        "elapsed_ms": elapsed_ms,
    }
