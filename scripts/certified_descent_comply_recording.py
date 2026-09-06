"""Exact-source bounded finalization for fresh certified-descent construction.

Reuse the immutable cooperative Windows byte budget and complete preencoding /
fault-first seal reader. Only identity and the compact method schema are new.
"""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from pathlib import Path

from scripts.paired_common_drift_comply_1800_binding import load_bound

frozen = load_bound(
    "scripts._certified_descent_comply_recording",
    "scripts/soft_drift_constrained_comply_recording.py",
    "8be515535213d8b233264f2e81a0f87a1da09ab717f72b58c8f8ad682215ef7e",
    (),
)
ROOT = frozen.ROOT
OUTPUT = "evidence/certified_descent_comply_v1_qwen35_08b"
POLICY = "configs/certified_descent_comply_recording_policy.json"
POLICY_SHA = "468cdd32cbe362280b63286fef88b922035218430e45b35b4ea2cc6cfcc358db"
CERTIFICATE = "docs/certified_descent_comply_preparation.json"
CERTIFICATE_SCHEMA = "sp_lense.certified_descent_comply_preparation.v1"
PREPARATION_REPORT = "docs/CERTIFIED_DESCENT_COMPLY_PREPARATION.md"
COMPACT_SCHEMA = "sp_lense.certified_descent_comply_compact_audit.v1"
AUDIT_MATCH = frozen.AUDIT_MATCH
PairedBudget, BudgetError = frozen.PairedBudget, frozen.BudgetError
QUOTAS, RECORD_CAPS = frozen.QUOTAS, frozen.RECORD_CAPS
ALLOWED, LOGITS, IMMUTABLE_HELPERS = frozen.ALLOWED, frozen.LOGITS, frozen.IMMUTABLE_HELPERS
FINAL_CAPS, COMPACT_LIMITS = frozen.FINAL_CAPS, frozen.COMPACT_LIMITS
ENTRYPOINT_COVERAGE_KEYS = frozen.ENTRYPOINT_COVERAGE_KEYS
FINALIZATION_COVERAGE_KEYS = frozen.FINALIZATION_COVERAGE_KEYS
RESULTS = "docs/certified_descent_comply_model_free_results.json"
TEST_LOCKS = (
    "docs/certified_descent_comply_model_free_test_lock.json",
    "docs/certified_descent_comply_metadata_test_lock.json",
)
SAVED_GATE_SHA256 = {
    "lock_sha256": "26bead137f4b57c05c9a0a5d62bfee13166d749e9f778f3c9c88ad05437b1dfb",
    "result_sha256": "a5393642a5c726fabd1ba4b48d735c8c4144f33297816e0b2ad3f7407758c3d5",
}
METHOD_COVERAGE_KEYS = frozenset(
    {
        "actual_cli_exact_lock_authorization",
        "fresh_usage_inclusive_100_fail_closed",
        "current_w_own_norm_history_path_binding",
        "fixed_proposal_first_passing_exact_certificate",
        "combined_assembly_solver_encoding_deadline",
        "exact_zero_ordinary_no_step_technical_failure_distinct",
        "technical_failure_supervisor_inconclusive",
        "compact_raw_evidence_references",
    }
)
require, sha, encode = frozen.require, frozen.sha, frozen.encode
_original_candidate_payload = frozen._candidate_payload
UPDATE_KEYS = frozenset(
    {
        "stage",
        "status",
        "gradient_cell_ids",
        "updates_jsonl_line",
        "update_canonical_json_sha256",
        "w_before_sha256",
        "w_after_sha256",
        "problem_sha256",
        "path_before_upper",
        "path_after_upper",
        "path_before",
        "path_after",
        "step_norm",
        "net_norm",
        "inputs",
        "history_w_sha256",
        "solver",
        "recorded_gradient_arithmetic",
        "real_derivatives_independently_rerun",
        "first_passing_verified",
        "exact_serialized_geometry_verified",
        "nominal_proposal_reconstructed",
        "nonlinear_acceptance_inferred_from_surrogate",
    }
)


def recording_policy(root=ROOT):
    root = Path(root)
    frozen.frozen.recording_policy(root)
    old = root / "configs/soft_drift_constrained_comply_recording_policy.json"
    require(
        sha(old.read_bytes()) == "cfd3887e696e5837dce020790d051dc0131ef6d1c41225d770634e1c27d22e09",
        "immutable complete-bound predecessor policy",
    )
    raw = (root / POLICY).read_bytes()
    require(sha(raw) == POLICY_SHA, "exact certified-descent recording policy")
    value = json.loads(raw)
    require(
        value["schema"] == "sp_lense.certified_descent_comply_recording_policy.v1"
        and value["output_namespace"] == OUTPUT
        and value["quotas"] == QUOTAS
        and set(value["allowed_artifacts"]) == ALLOWED
        and len(value["allowed_artifacts"]) == len(ALLOWED)
        and value["whole_record_cap_bytes"] == RECORD_CAPS
        and value["final_artifact_cap_bytes"] == FINAL_CAPS
        and value["compact_audit"]["schema"] == COMPACT_SCHEMA
        and all(value["compact_audit"][k] == v for k, v in COMPACT_LIMITS.items())
        and (
            value["maximum_forwards"],
            value["maximum_derivatives"],
            value["maximum_updates"],
            value["timeout_including_loading_seconds"],
        )
        == (216, 96, 8, 1800)
        and value["combined_category_bound_bytes"] == 524995016
        and value["auxiliary_ceiling_bytes"] == 16 * 1024**2
        and value["minimum_free_bytes"] == 1024**3
        and value["immutable_helpers_sha256"] == IMMUTABLE_HELPERS,
        "unchanged whole-attempt/finalization/record ceilings",
    )
    return value


def certificate_source_paths():
    from scripts import certified_descent_comply_plan as plan

    return frozenset(plan.SOURCE_PATHS) - {CERTIFICATE}


def _require_preparation_evidence(certificate, root):
    """Bind preparation attestations without reading any saved model coordinates.

    Full-pipeline evidence predates the metadata/report-only amendment. Both
    prospective locks remain immutable and this chronology must stay explicit.
    """
    root = Path(root)
    hashes = certificate["source_sha256"]
    raw = (root / RESULTS).read_bytes()
    require(
        certificate.get("model_free_results_sha256") == hashes[RESULTS] == sha(raw),
        "bound complete model-free result receipt",
    )
    result = json.loads(raw)
    require(
        result.get("schema") == "sp_lense.certified_descent_comply_model_free_results.v1"
        and result.get("test_lock_sha256") == {path: hashes[path] for path in TEST_LOCKS}
        and all(sha((root / path).read_bytes()) == hashes[path] for path in TEST_LOCKS),
        "original and metadata-only prospective test locks",
    )
    original_lock, metadata_lock = (json.loads((root / path).read_bytes()) for path in TEST_LOCKS)
    require(
        original_lock.get("schema") == "sp_lense.certified_descent_comply_model_free_test_lock.v1"
        and metadata_lock.get("schema") == "sp_lense.certified_descent_comply_metadata_test_lock.v1"
        and metadata_lock.get("original_test_lock_sha256") == hashes[TEST_LOCKS[0]]
        and metadata_lock.get("metadata_only") is True
        and metadata_lock.get("full_cases_repeated") is False,
        "full cases precede the metadata-only amendment without repetition",
    )
    locked_evidence = metadata_lock.get("method_coverage_evidence", {})
    coverage = result.get("method_coverage", {})
    for key in METHOD_COVERAGE_KEYS:
        item = coverage.get(key, {})
        require(
            item.get("passed") is True
            and type(item.get("evidence")) is list
            and 1 <= len(item["evidence"]) <= 16
            and all(type(name) is str and 0 < len(name) <= 512 for name in item["evidence"]),
            "named passed method-specific preparation evidence",
        )
        require(
            item["evidence"] == locked_evidence.get(key), "exact locked coverage test identities"
        )

    def seconds(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    gate = result.get("saved_input_gate", {})
    require(
        gate.get("admitted") is True
        and type(gate.get("solver_calls")) is int
        and gate["solver_calls"] == 1
        and gate.get("normal_exit") is True
        and seconds(gate.get("external_seconds"))
        and gate["external_seconds"] <= 10
        and gate.get("candidate_exported") is False
        and gate.get("saved_step_reuse_forbidden") is True
        and gate.get("new_model_result") is False
        and all(gate.get(key) == digest for key, digest in SAVED_GATE_SHA256.items()),
        "one admitted non-exported posthoc saved-input diagnostic only",
    )
    ledger = result.get("execution", {})
    require(
        ledger.get("limits")
        == {"total": 720, "initial_saved": 10, "ancillary": 110, "full_case": 300}
        and ledger.get("all_attempts_including_failures_recorded") is True
        and seconds(ledger.get("ancillary_overhead_charge_seconds")),
        "unchanged shared model-free budget including failures and overhead",
    )
    attempts = ledger.get("attempts")
    require(type(attempts) is list and 4 <= len(attempts) <= 64, "complete bounded attempt ledger")
    categories = {
        "initial_saved": [],
        "full_interior_candidate": [],
        "full_finite_negative": [],
        "ancillary": [],
    }
    names = set()
    for attempt in attempts:
        require(
            type(attempt) is dict
            and type(attempt.get("name")) is str
            and 0 < len(attempt["name"]) <= 128
            and attempt["name"] not in names
            and attempt.get("category") in categories
            and seconds(attempt.get("external_seconds"))
            and type(attempt.get("returncode")) is int
            and type(attempt.get("timed_out")) is bool,
            "unique measured attempt disposition",
        )
        names.add(attempt["name"])
        categories[attempt["category"]].append(attempt)
    by_name = {item["name"]: item for item in attempts}
    for name, category, lock_index in (
        ("ancillary_tests_1", "ancillary", 0),
        ("metadata_tests_1", "ancillary", 1),
        ("full_native_interior_candidate", "full_interior_candidate", 0),
        ("full_native_finite_negative", "full_finite_negative", 0),
    ):
        item = by_name.get(name, {})
        require(
            item.get("category") == category
            and item.get("test_lock_sha256") == hashes[TEST_LOCKS[lock_index]]
            and item.get("returncode") == 0
            and item.get("timed_out") is False,
            "passed batch belongs to its original or metadata-only source lock",
        )
    for category, cap in (
        ("initial_saved", 10),
        ("full_interior_candidate", 300),
        ("full_finite_negative", 300),
    ):
        items = categories[category]
        require(
            len(items) == 1
            and items[0].get("hard_timeout_seconds") == cap
            and items[0]["external_seconds"] <= cap
            and items[0]["returncode"] == 0
            and items[0]["timed_out"] is False,
            "sole successful bounded gate/full-case attempt",
        )
    require(
        categories["initial_saved"][0]["external_seconds"] == gate["external_seconds"],
        "saved gate and shared ledger agree",
    )
    ancillary = ledger["ancillary_overhead_charge_seconds"] + sum(
        item["external_seconds"] for item in categories["ancillary"]
    )
    total = ledger["ancillary_overhead_charge_seconds"] + sum(
        item["external_seconds"] for item in attempts
    )
    require(
        categories["ancillary"]
        and ancillary <= 110
        and total <= 720
        and seconds(ledger.get("aggregate_charged_seconds"))
        and abs(ledger["aggregate_charged_seconds"] - total) <= 1e-9,
        "complete shared aggregate and ancillary ceilings",
    )
    return result


def require_certificate(root=ROOT):
    certificate = frozen.require_certificate(root)
    _require_preparation_evidence(certificate, root)
    return certificate


def _tree(value, depth=0):
    """Finite compact values only; exact rational strings have an explicit cap."""
    require(depth <= 10, "compact nesting depth")
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        require(abs(value) < 2**63, "compact integer range")
    elif type(value) is float:
        require(math.isfinite(value), "finite compact float")
    elif type(value) is str:
        limit = 4096 if re.fullmatch(r"-?[0-9]+(?:/[1-9][0-9]*)?", value) else 512
        require(len(value.encode("utf-8")) <= limit, "bounded exact scalar/string")
    elif type(value) is list:
        require(len(value) <= 12, "no native arrays or complete trial lists in compact audit")
        for item in value:
            _tree(item, depth + 1)
    elif type(value) is dict:
        require(len(value) <= 96, "bounded compact object")
        hash_arrays = {"A_row_sha256": 12, "D_exact_row_sha256": 6}
        for key, item in value.items():
            require(
                type(key) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", key),
                "compact field name",
            )
            if key in hash_arrays or key == "history_w_sha256":
                require(
                    type(item) is list
                    and (
                        len(item) == hash_arrays[key] if key in hash_arrays else 1 <= len(item) <= 9
                    ),
                    "fixed hash array shape",
                )
                require(
                    all(type(x) is str and re.fullmatch(r"[a-f0-9]{64}", x) for x in item),
                    "exact array hashes",
                )
            elif key.endswith("sha256") and item is not None:
                require(
                    type(item) is str and re.fullmatch(r"[a-f0-9]{64}", item),
                    "exact compact SHA256",
                )
            _tree(item, depth + 1)
    else:
        raise ValueError("only bounded finite primitive compact evidence")


def _candidate_payload(record, verified, verification_sha):
    _original_candidate_payload(record, verified, verification_sha)
    require(
        sum((Fraction(v) ** 2 for v in record["vector"]), Fraction(0)) <= Fraction(1, 25),
        "actual serialized candidate obeys the unchanged exact decimal net ball",
    )


# These are isolated module globals, never edits to immutable source/modules.
for _key in (
    "OUTPUT",
    "POLICY",
    "POLICY_SHA",
    "CERTIFICATE",
    "CERTIFICATE_SCHEMA",
    "PREPARATION_REPORT",
    "COMPACT_SCHEMA",
    "UPDATE_KEYS",
):
    setattr(frozen, _key, globals()[_key])
frozen.recording_policy = recording_policy
frozen.certificate_source_paths = certificate_source_paths
frozen._tree = _tree
frozen._candidate_payload = _candidate_payload
encode_verification = frozen.encode_verification
prepare_final_artifacts = frozen.prepare_final_artifacts
finalize_recording = frozen.finalize_recording
read_sealed_recording = frozen.read_sealed_recording
