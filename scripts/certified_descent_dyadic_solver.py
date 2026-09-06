"""One nominal proposal and one fixed dyadic ray; standalone/model-free only.

Generation reuses the immutable floating gradient and analytic projection. The
new independent checker alone certifies exact serialized geometry and sufficient
decrease. A rejected trial advances only to its next declared scale. There is no
second proposal, old solve call, point repair, or near-optimality gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import time

from scripts import partial_progress_deficit_solver as generation
from scripts import verify_certified_descent_dyadic as checker

SECONDS = 10.0
LAST_SCALE_INDEX = 52
MAX_TRIALS = 53
IMMUTABLE_DEPENDENCIES = (
    "scripts/partial_progress_deficit_solver.py",
    "scripts/verify_partial_progress_deficit.py",
)
TRIAL_STATUSES = {"ADMITTED_DESCENT", "REJECTED_TRIAL", "NO_CERTIFIED_STEP"}
PROJECTION_METADATA = (
    "case",
    "step_multiplier",
    "net_multiplier",
    "residual_norm",
    "residual_limit",
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def tick(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("COMBINED_TEN_SECOND_DEADLINE")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def vector_sha256(values):
    return hashlib.sha256(struct.pack("<" + "d" * len(values), *values)).hexdigest()


def solve(problem, expected_sha, *, observer=None):
    """Return one checked disposition; this API never authorizes a model call.

    The entire result is actually JSON-encoded before the final admission-time
    check. An external caller must additionally bound stdout completion. The
    elapsed field is sampled just before that final encoding; the deadline gate
    is after it. A serialization/deadline failure returns only a no-step receipt.
    """
    started = time.monotonic()
    deadline = started + SECONDS
    result = {
        "status": "NO_CERTIFIED_STEP",
        "terminal_reason": "NOT_STARTED",
        "problem_sha256": expected_sha
        if type(expected_sha) is str and len(expected_sha) == 64
        else None,
        "gradient_count": 0,
        "proposal_count": 0,
        "trial_count": 0,
        "trials": [],
        "chosen_j": None,
        "first_passing_j": None,
        "w_next": None,
        "proposal": None,
        "zero_certificate": None,
        "selected_certificate": None,
        "descriptive_gap": None,
        "diagnostic_count": 0,
        "observer_calls": 0,
        "step_admitted": False,
        "exact_zero_optimum": False,
        "maximum_trials": MAX_TRIALS,
        "scale_indices": [0, LAST_SCALE_INDEX],
        "combined_seconds_limit": SECONDS,
        "point_repaired": False,
        "alternate_proposal_used": False,
        "near_optimality_gate": False,
        "mathematical_trial_pass_is_not_step_issuance": True,
        "model_calls_authorized": False,
        "serialization_checked": False,
    }

    def failure_receipt(reason):
        # Failure handling never retries a proposal/check/diagnostic. Preserve
        # bounded trial dispositions even if a certificate was not serializable.
        receipt = {
            "status": "NO_CERTIFIED_STEP",
            "terminal_reason": str(reason)[:256],
            "problem_sha256": result["problem_sha256"],
            "gradient_count": result["gradient_count"],
            "proposal_count": result["proposal_count"],
            "trial_count": result["trial_count"],
            "trials": [
                {
                    "j": item["j"],
                    "lambda": item["lambda"],
                    "status": (
                        item["certificate"]["status"]
                        if item["certificate"] is not None
                        else item["check_status"]
                    ),
                }
                for item in result["trials"]
            ],
            "chosen_j": None,
            "first_passing_j": result["first_passing_j"],
            "w_next": None,
            "step_admitted": False,
            "exact_zero_optimum": False,
            "diagnostic_count": result["diagnostic_count"],
            "observer_calls": result["observer_calls"],
            "failure_receipt_only": True,
            "full_certificate_details_returned": False,
            "point_repaired": False,
            "alternate_proposal_used": False,
            "near_optimality_gate": False,
            "model_calls_authorized": False,
            "elapsed_seconds": time.monotonic() - started,
            "elapsed_sample": "before final receipt encoding",
            "serialization_checked": True,
        }
        try:
            encoded(receipt)
        except Exception:  # noqa: BLE001 - even failure-receipt encoding may fail.
            # No claim that this final dictionary has been successfully encoded.
            # The external hard deadline still bounds its caller's output attempt.
            receipt["serialization_checked"] = False
            receipt["receipt_encoding_failed"] = True
        return receipt

    def finish(status, reason, endpoint=None):
        if time.monotonic() >= deadline:
            return failure_receipt("COMBINED_TEN_SECOND_DEADLINE")
        result["status"], result["terminal_reason"] = status, str(reason)[:256]
        result["step_admitted"] = status == "ADMITTED_DESCENT"
        result["exact_zero_optimum"] = status == "EXACT_ZERO_OPTIMUM"
        result["w_next"] = list(endpoint) if result["step_admitted"] else None
        result["chosen_j"] = result["first_passing_j"] if result["step_admitted"] else None
        result["serialization_checked"] = True
        result["elapsed_seconds"] = time.monotonic() - started
        result["elapsed_sample"] = "before final result encoding; deadline checked after encoding"
        try:
            encoded(result)
        except Exception as error:  # noqa: BLE001 - serialization cannot issue a step.
            return failure_receipt("RESULT_SERIALIZATION_FAILED:" + type(error).__name__)
        if time.monotonic() >= deadline:
            return failure_receipt("COMBINED_TEN_SECOND_DEADLINE")
        return result

    def observe(phase):
        if observer is None:
            return
        tick(deadline)
        snapshot = {
            "phase": phase,
            "gradient_count": result["gradient_count"],
            "proposal_count": result["proposal_count"],
            "trial_count": result["trial_count"],
            "last_trial": copy.deepcopy(result["trials"][-1]) if result["trials"] else None,
            "zero_certificate": copy.deepcopy(result["zero_certificate"]),
            "issuance_pending": True,
            "step_issued": False,
        }
        result["observer_calls"] += 1
        observer(snapshot)  # Recording-only return value is deliberately ignored.
        tick(deadline)

    try:
        require(observer is None or callable(observer), "INVALID_RECORDING_OBSERVER")
        require(
            type(expected_sha) is str
            and len(expected_sha) == 64
            and all(x in "0123456789abcdef" for x in expected_sha),
            "INVALID_EXPECTED_SHA256",
        )
        context = checker.prepare(problem, expected_sha, deadline)
        tick(deadline)
        n = context["dimension"]
        require(
            type(n) is int and 1 <= n <= 1024 and context["deadline"] == deadline,
            "INVALID_AUTHENTICATED_CONTEXT",
        )
        w = generation.vector(context["w"], n)
        M = generation.finite(context["M"])
        require(M >= 1.0, "INVALID_OUTWARD_LIPSCHITZ_BOUND")
        result["dimension"], result["fixed_M"] = n, M

        zero_certificate = checker.exact_zero(context)
        tick(deadline)
        require(
            isinstance(zero_certificate, dict)
            and zero_certificate.get("status") in {"EXACT_ZERO_OPTIMUM", "NOT_EXACT_ZERO"},
            "INVALID_EXACT_ZERO_CERTIFICATE",
        )
        result["zero_certificate"] = zero_certificate
        observe("exact_zero_checked")
        if zero_certificate["status"] == "EXACT_ZERO_OPTIMUM":
            return finish("EXACT_ZERO_OPTIMUM", zero_certificate.get("reason", "EXACT_ZERO_PROOF"))

        # Exactly one floating gradient and exactly one nominal projection.
        result["gradient_count"] = 1
        g0 = generation.gradient(context, [0.0] * n)
        tick(deadline)
        y = generation.vector([-x / M for x in g0], n)
        result["proposal"] = {"y": y, "p": None, "M": M, "projection": None}
        result["proposal_count"] = 1
        projected = generation.project_intersection(y, w, context["rho"])
        tick(deadline)
        p = generation.vector(projected["point"], n)
        result["proposal"]["p"] = p
        result["proposal"]["projection"] = {key: projected[key] for key in PROJECTION_METADATA}
        result["proposal"]["projection"].update(
            {
                "step_normal": generation.vector(projected["step_normal"], n),
                "net_normal": generation.vector(projected["net_normal"], n),
            }
        )

        for j in range(MAX_TRIALS):
            tick(deadline)
            lam = math.ldexp(1.0, -j)
            # Separate operations enforce RN64(lambda*p), then RN64(w+scaled).
            # Neither the scaled vector nor the endpoint is subsequently repaired.
            scaled = generation.vector([lam * x for x in p], n)
            endpoint = generation.vector([wi + x for wi, x in zip(w, scaled, strict=True)], n)
            trial = {"j": j, "lambda": lam, "certificate": None, "check_status": "STARTED"}
            result["trials"].append(trial)
            result["trial_count"] += 1
            certificate = checker.check(context, endpoint)
            require(
                isinstance(certificate, dict) and certificate.get("status") in TRIAL_STATUSES,
                "INVALID_TRIAL_CERTIFICATE_STATUS",
            )
            trial["certificate"] = certificate
            trial["check_status"] = "COMPLETED"
            tick(deadline)
            observe("trial_checked")
            # Actual zero cannot certify descent. It terminates this ray without
            # promoting a rounded-away step to an exact stationarity conclusion.
            if endpoint == w:
                return finish("NO_CERTIFIED_STEP", "serialized_zero_increment")
            if certificate["status"] == "NO_CERTIFIED_STEP":
                return finish("NO_CERTIFIED_STEP", certificate.get("reason", "CHECKER_NO_STEP"))
            if certificate["status"] == "REJECTED_TRIAL":
                continue

            # First passing trial only. The diagnostic is descriptive, not a new
            # objective-gap threshold, and is evaluated exactly once.
            result["first_passing_j"] = j
            result["selected_certificate"] = certificate
            actual_hash = vector_sha256(endpoint)
            require(
                certificate.get("w_next_sha256") == actual_hash, "SELECTED_ENDPOINT_HASH_MISMATCH"
            )
            result["diagnostic_count"] = 1
            diagnostic = checker.diagnostic(context, endpoint)
            tick(deadline)
            require(
                isinstance(diagnostic, dict)
                and diagnostic.get("w_next_sha256") == actual_hash
                and "descriptive_gap_upper" in diagnostic,
                "DESCRIPTIVE_DIAGNOSTIC_ENDPOINT_MISMATCH",
            )
            result["descriptive_gap"] = diagnostic
            return finish("ADMITTED_DESCENT", "FIRST_CERTIFIED_DYADIC_TRIAL", endpoint)

        return finish("NO_CERTIFIED_STEP", "FIXED_DYADIC_SCHEDULE_EXHAUSTED")
    except Exception as error:  # noqa: BLE001 - faults never authorize an endpoint.
        reason = (
            "COMBINED_TEN_SECOND_DEADLINE"
            if isinstance(error, TimeoutError)
            else type(error).__name__ + ":" + str(error)[:192]
        )
        if result["trials"] and result["trials"][-1]["certificate"] is None:
            result["trials"][-1]["check_status"] = "CHECK_FAILED_OR_TIMED_OUT"
        return finish("NO_CERTIFIED_STEP", reason)
