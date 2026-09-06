"""Fresh-row adapter for the frozen one-proposal certified dyadic method.

No model access, historical evidence, point repair, or additional numerical
proposal is available here. Exact geometry and sufficient decrease come only
from the immutable solver's independent checker. Float norms are descriptive.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from fractions import Fraction as F

from scripts import certified_descent_dyadic_solver as solver
from scripts import paired_common_drift_comply_optimizer as contracts

require, vector, scalar = contracts.require, contracts.vector, contracts.scalar
vector_sha = contracts.vector_sha
PAIRS = ((2, 0), (3, 1), (6, 4), (7, 5), (10, 8), (11, 9))
SECONDS = 10.0
MAX_UPDATE_BYTES = 8 * 1024**2
ORDINARY_NO_STEP = {"FIXED_DYADIC_SCHEDULE_EXHAUSTED", "serialized_zero_increment"}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def tick(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("ADAPTER_COMBINED_TEN_SECOND_DEADLINE")


def objective(states, baselines):
    """Descriptive cached responses, never nonlinear or causal acceptance."""
    rows, base = contracts._bound_rows(states, baselines)
    margins = [-scalar(row["preserve_log_odds"]) for row in rows]
    original = [-scalar(row["preserve_log_odds"]) for row in base]
    return {
        "comply_margins": margins,
        "baseline_relative_common_letter_drift": [
            ((margins[a] - original[a]) - (margins[b] - original[b])) / 2.0 for a, b in PAIRS
        ],
        "behavioral_acceptance_gate": False,
        "local_objective_descent_guaranteed": False,
    }


def _compact_solver(solved):
    """Only remove reconstructable duplicate arrays, never a trial disposition."""
    compact = {
        key: value
        for key, value in solved.items()
        if key not in {"w_next", "selected_certificate", "trials"}
    }
    compact["selected_trial_j"] = solved.get("chosen_j")
    compact["trials"] = []
    for trial in solved["trials"]:
        item = dict(trial)
        certificate = trial.get("certificate")
        if isinstance(certificate, dict):
            item["certificate"] = dict(certificate)
            if "actual_displacement" in certificate:
                actual = item["certificate"].pop("actual_displacement")
                require(
                    isinstance(actual, list) and len(actual) == 1024,
                    "native exact displacement certificate",
                )
                item["certificate"]["actual_displacement_sha256"] = canonical_sha(actual)
                item["certificate"]["actual_displacement_dimension"] = len(actual)
        compact["trials"].append(item)
    return compact


def increment(gradients, w, path, stage, baselines, *, history, path_upper):
    """Journal one attempt; only ``ready`` may supply an applicable endpoint.

    The outer deadline starts before authentication/assembly. Its recording-only
    observer checks the same deadline inside the unchanged solver; final actual
    JSON encoding is also charged. The runtime additionally bounds worker time.
    History and the rational path upper bound, not float norm accumulation, are
    authoritative. A technical fault cannot become an ordinary no-step result.
    """
    started = time.monotonic()
    deadline = started + SECONDS
    result = {
        "stage": stage if type(stage) is int else None,
        "status": "no_certified_step",
        "terminal_reason": "NOT_STARTED",
        "adapter_fault": False,
        "technical_failure": False,
        "solver": None,
        "combined_seconds_limit": SECONDS,
        "step_admitted": False,
        "exact_zero_optimum": False,
        "near_optimality_gate": False,
        "point_repaired": False,
        "nonlinear_acceptance_inferred": False,
        "float_geometry_is_descriptive": True,
        "serialization_checked": False,
    }

    def encode_final():
        result["elapsed_seconds"] = time.monotonic() - started
        result["elapsed_sample"] = "before final update encoding; deadline checked after encoding"
        result["serialization_checked"] = True
        data = encoded(result)
        require(len(data) <= MAX_UPDATE_BYTES, "UPDATE_EXCEEDS_EIGHT_MIB")
        tick(deadline)
        return result

    def fail(reason):
        # Do not discard an attempted proposal/trial journal on a technical
        # fault. Its mathematical pass, if any, is explicitly not an issued step.
        for key in (
            "w_after",
            "w_after_sha256",
            "path_after",
            "path_after_upper",
            "step_norm",
            "net_norm",
        ):
            result.pop(key, None)
        result.update(
            status="no_certified_step",
            terminal_reason=str(reason)[:256],
            adapter_fault=True,
            technical_failure=True,
            step_admitted=False,
            exact_zero_optimum=False,
        )
        result["elapsed_seconds"] = time.monotonic() - started
        result["elapsed_sample"] = "before failure update encoding"
        result["serialization_checked"] = True
        try:
            data = encoded(result)
            if len(data) <= MAX_UPDATE_BYTES:
                return result
        except Exception:  # noqa: BLE001 - a failed encoder must never issue an endpoint.
            data = None
        # Explicit bounded failure receipt, not a claim to have saved the full
        # journal. The recording layer must classify this as a technical failure.
        receipt = {
            "stage": result["stage"],
            "status": "no_certified_step",
            "terminal_reason": "UPDATE_RECORDING_FAILED:" + str(reason)[:160],
            "adapter_fault": True,
            "step_admitted": False,
            "exact_zero_optimum": False,
            "technical_failure": True,
            "failure_receipt_only": True,
            "full_attempt_recorded": False,
            "oversize_record_sha256": hashlib.sha256(data).hexdigest() if data else None,
            "oversize_record_bytes": len(data) if data else None,
            "serialization_checked": True,
        }
        try:
            encoded(receipt)
        except Exception:  # noqa: BLE001 - honest fallback has no encoding-success claim.
            receipt["serialization_checked"] = False
        return receipt

    try:
        require(type(stage) is int and 1 <= stage <= 8, "one of eight attempted updates")
        w, path = vector(w), scalar(path)
        require(
            isinstance(path_upper, str)
            and re.fullmatch(r"-?\d{1,40}(?:/[1-9]\d{0,39})?", path_upper, flags=re.ASCII)
            is not None,
            "bounded exact rational path upper",
        )
        bound = F(path_upper)
        require(
            0 <= bound <= F(2, 5) and path == float(bound),
            "descriptive path equals authoritative rational upper",
        )
        require(
            isinstance(history, list) and len(history) == stage,
            "one fresh-zero snapshot followed by each applied endpoint",
        )
        snapshots = [vector(item) for item in history]
        require(
            all(x == 0.0 for x in snapshots[0]) and vector_sha(snapshots[-1]) == vector_sha(w),
            "fresh-zero history and exact current endpoint",
        )
        rows, base = contracts._bound_rows(gradients, baselines)
        require(
            vector_sha(rows[0]["shared_w"]) == vector_sha(w),
            "gradient cache belongs to exact current w bytes",
        )
        for row in rows:
            condition = f"gradient_{stage}"
            previous = "baseline" if stage == 1 else f"step_{stage - 1}"
            require(
                row["condition"] == condition
                and row["stage"] == stage
                and row["cell_id"] == row["prompt_id"] + "__" + condition
                and row["current_cell_id"] == row["prompt_id"] + "__" + previous,
                "fresh current-gradient stage/cache binding",
            )
            require(
                0 <= scalar(row["maximum_current_logit_difference"]) <= 1e-6
                and 0 <= scalar(row["maximum_current_h_difference"]) <= 1e-6,
                "current-gradient identity arm",
            )
            require(
                vector_sha(row["shared_w"]) == vector_sha(w), "all gradient current-w hashes agree"
            )
            vector(row["gradient"])
        tick(deadline)
        margins = [-scalar(row["preserve_log_odds"]) for row in rows]
        original = [-scalar(row["preserve_log_odds"]) for row in base]
        A = [vector([-scalar(row["h0_norm"]) * scalar(g) for g in row["gradient"]]) for row in rows]
        b = [scalar(0.10 - margin) for margin in margins]
        c = [
            scalar(((margins[a] - original[a]) - (margins[b] - original[b])) / 2.0)
            for a, b in PAIRS
        ]
        D_hashes = []
        for a, brow in PAIRS:
            exact_D = [str((F(x) - F(y)) / 2) for x, y in zip(A[a], A[brow], strict=True)]
            D_hashes.append(canonical_sha(exact_D))
            tick(deadline)
        problem = {
            "A": A,
            "b": b,
            "c": c,
            "w": w,
            "w_sha256": vector_sha(w),
            "path_upper": path_upper,
            "history": snapshots,
        }
        fingerprint = canonical_sha(problem)
        result.update(
            gradient_cell_ids=[row["cell_id"] for row in rows],
            baseline_cell_ids=[row["cell_id"] for row in base],
            w_before=w,
            w_before_sha256=vector_sha(w),
            path_before=path,
            path_before_upper=path_upper,
            history_w_sha256=[vector_sha(item) for item in snapshots],
            problem_sha256=fingerprint,
            inputs={
                "A_row_sha256": [vector_sha(row) for row in A],
                "b": b,
                "c": c,
                "D_exact_row_sha256": D_hashes,
            },
        )
        tick(deadline)

        def recording_observer(_snapshot):
            tick(deadline)

        solved = solver.solve(problem, fingerprint, observer=recording_observer)
        # Preserve the actual attempt before the outer deadline can reject it.
        result["solver"] = _compact_solver(solved)
        tick(deadline)
        require(solved.get("problem_sha256") == fingerprint, "solver problem identity")
        require(solved.get("serialization_checked") is True, "solver result serialization")
        status = solved.get("status")
        require(
            status in {"ADMITTED_DESCENT", "EXACT_ZERO_OPTIMUM", "NO_CERTIFIED_STEP"},
            "known immutable solver disposition",
        )
        result["terminal_reason"] = solved["terminal_reason"]
        if status == "NO_CERTIFIED_STEP":
            result["adapter_fault"] = solved[
                "terminal_reason"
            ] not in ORDINARY_NO_STEP or solved.get("failure_receipt_only", False)
            result["technical_failure"] = bool(result["adapter_fault"])
            return encode_final()
        if status == "EXACT_ZERO_OPTIMUM":
            certificate = solved["zero_certificate"]
            require(
                certificate["status"] == "EXACT_ZERO_OPTIMUM"
                and certificate["problem_sha256"] == fingerprint
                and certificate["current_w_sha256"] == vector_sha(w),
                "independent exact-zero proof identity",
            )
            result.update(status="exact_zero_optimum", exact_zero_optimum=True)
            return encode_final()

        endpoint = vector(solved["w_next"])
        certificate = solved["selected_certificate"]
        j = solved["chosen_j"]
        require(
            type(j) is int
            and 0 <= j <= 52
            and solved["trials"][-1]["j"] == j
            and solved["trials"][-1]["certificate"] == certificate,
            "selected certificate is final fixed-ray trial",
        )
        require(
            certificate["status"] == "ADMITTED_DESCENT"
            and certificate["problem_sha256"] == fingerprint
            and certificate["current_w_sha256"] == vector_sha(w)
            and certificate["w_next_sha256"] == vector_sha(endpoint)
            and certificate["geometry_valid"] is True,
            "selected exact endpoint certificate binding",
        )
        require(
            solved["descriptive_gap"]["w_next_sha256"] == vector_sha(endpoint),
            "descriptive diagnostic endpoint identity",
        )
        geometry = certificate["geometry"]
        require(F(geometry["path_before_upper"]) == bound, "selected original path bound")
        after_bound = F(geometry["path_after_upper"])
        require(bound <= after_bound <= F(2, 5), "selected authoritative path upper")
        result.update(
            status="ready",
            step_admitted=True,
            w_after=endpoint,
            w_after_sha256=vector_sha(endpoint),
            path_after_upper=geometry["path_after_upper"],
            path_after=float(after_bound),
            step_norm=scalar(math.sqrt(float(F(geometry["actual_step_squared"])))),
            net_norm=scalar(math.sqrt(float(F(geometry["net_squared"])))),
        )
        tick(deadline)
        return encode_final()
    except Exception as error:  # noqa: BLE001 - no technical fault may issue a step.
        reason = (
            "ADAPTER_COMBINED_TEN_SECOND_DEADLINE"
            if isinstance(error, TimeoutError)
            else type(error).__name__ + ":" + str(error)[:192]
        )
        return fail(reason)
