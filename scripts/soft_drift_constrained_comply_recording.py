"""Compact, preflighted soft-drift finalization; no model or optimizer imports.

The immutable cooperative Windows budget owns actual byte admission and sealing.
This module does not repair old evidence or grant real-run authorization.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import threading
from pathlib import Path

from scripts import paired_common_drift_comply_recording as frozen
from scripts.three_family_bounded_capture import exception_record
from scripts.three_family_recording_budget import BudgetError

ROOT = frozen.ROOT
OUTPUT = "evidence/soft_drift_constrained_comply_v1_qwen35_08b"
POLICY = "configs/soft_drift_constrained_comply_recording_policy.json"
POLICY_SHA = "cfd3887e696e5837dce020790d051dc0131ef6d1c41225d770634e1c27d22e09"
CERTIFICATE = "docs/soft_drift_constrained_comply_preparation.json"
CERTIFICATE_SCHEMA = "sp_lense.soft_drift_constrained_comply_preparation.v1"
PREPARATION_REPORT = "docs/SOFT_DRIFT_CONSTRAINED_COMPLY_PREPARATION.md"
COMPACT_SCHEMA = "sp_lense.soft_drift_constrained_comply_compact_audit.v1"
AUDIT_MATCH = "INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH"
PairedBudget = frozen.PairedBudget
QUOTAS, RECORD_CAPS = frozen.QUOTAS, frozen.RECORD_CAPS
ALLOWED, LOGITS, IMMUTABLE_HELPERS = frozen.ALLOWED, frozen.LOGITS, frozen.IMMUTABLE_HELPERS
COMPACT_LIMITS = {
    "common_envelope_bytes": 262144,
    "row_count": 216,
    "individual_row_bytes": 2048,
    "update_count": 8,
    "individual_update_bytes": 32768,
    "trajectory_count": 10,
    "individual_trajectory_bytes": 4096,
    "structure_allowance_bytes": 8192,
    "complete_verification_bound_bytes": 1015808,
}
FINAL_CAPS = {
    "verification.json": 1024**2,
    "PILOT_REPORT.md": 256 * 1024,
    "comply_vector.json": 128 * 1024,
    "candidate_freeze.json": 64 * 1024,
    "CLOSEOUT.json": 64 * 1024,
}
ENTRYPOINT_COVERAGE_KEYS = frozenset(
    {
        "actual_cli_entrypoint_tested",
        "worker_dispatch_reached",
        "pre_backend_load_sentinel_tested",
        "supervisor_exact_command_bounded_capture_tested",
        "single_worker_no_retry_tested",
        "unknown_command_failed_closed",
    }
)
FINALIZATION_COVERAGE_KEYS = frozenset(
    {
        "native_1024_max216_rows_eight_updates_interior_candidate",
        "native_1024_max216_rows_eight_updates_finite_negative",
        "actual_independent_verifier_finalizer_sealed_reader",
        "preencoded_complete_final_category_bound",
        "zero_byte_fault_precedence",
        "tampering_failed_closed",
    }
)
TOP_KEYS = frozenset(
    {
        "status",
        "summary",
        "optimizer_checks",
        "construction_stages",
        "row_checks",
        "maximum_scalar_errors",
        "score_tolerance_absolute",
        "score_tolerance_relative",
        "maximum_current_logit_difference",
        "maximum_cast_component_error",
        "maximum_nonfinal_difference",
        "storage_inventory",
        "audited_endpoint_sha256",
        "audited_result_sha256",
        "runtime",
        "schema",
        "maximum_absolute_arithmetic_errors",
        "absolute_tolerance",
        "relative_tolerance",
    }
)
# Only short scalar/hash/interval structures may occur under these scientific
# sections. Exact field whitelists are checked independently of their byte caps.
UPDATE_KEYS = frozenset(
    {
        "stage",
        "status",
        "optimizer_kkt_verified",
        "gradient_cell_ids",
        "vector_sha256",
        "qp_inputs",
        "qp_witness",
        "predictions",
        "numeric_certificate",
        "solver",
        "d_norm",
        "clip_factor",
        "proposed_step_norm",
        "unprojected_net_norm",
        "projection_factor",
        "projection_distance",
        "step_norm",
        "net_norm",
        "path_before",
        "path_after",
        "proposed_path_after",
        "shared_path",
        "shared_net",
        "infeasibility_certified",
        "raw_feasibility_is_applied_feasibility",
        "maximum_component_error",
        "maximum_scaled_component_error",
        "recorded_update_sha256",
        "reconstructed_update_sha256",
        "w_before_sha256",
        "w_after_sha256",
        "update_canonical_json_sha256",
        "predicted_semantic_proxy",
        "recorded_gradient_arithmetic",
        "real_derivatives_independently_rerun",
        "nonlinear_acceptance_inferred_from_qp",
        "updates_jsonl_line",
    }
)
ROW_KEYS = frozenset(
    {
        "cell_id",
        "prompt_id",
        "condition",
        "stage",
        "cell_sha256",
        "prompt_sha256",
        "family_id",
        "variant_id",
        "order",
        "display_order",
        "current_cell_id",
        "forward_attempts",
        "derivative_attempts",
        "logits_sha256",
        "logits_file_sha256",
        "logits_path",
        "logit_count",
        "shared_w_sha256",
        "gradient_sha256",
        "actual_next_token_id",
        "actual_next_token_label",
        "forced_pair_label",
        "preserve_label",
        "comply_label",
        "requested",
        "signed_margin",
        "preserve_log_odds",
        "delta_log_odds",
        "signed_delta_log_odds",
        "answer_pair_mass",
        "kl_from_baseline",
        "quality_valid",
        "requested_accepted",
        "maximum_delta_error",
        "maximum_offset_error",
        "maximum_step_error",
        "maximum_current_logit_difference",
        "maximum_current_h_difference",
        "unselected_max_difference",
        "row_sha256",
        "recorded_row_sha256",
        "baseline_label",
        "net_relative_norm",
        "path_relative_norm",
        "logits_file",
        "rows_jsonl_line",
        "recorded_gradient_present",
        "reconstruction",
    }
)
TRAJECTORY_KEYS = frozenset(
    {
        "condition",
        "stage",
        "comply_margins",
        "baseline_relative_common_letter_drift",
        "pairs",
        "behavioral_acceptance_gate",
        "local_objective_descent_guaranteed",
    }
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    """One finite canonical UTF-8 representation; no rounding or omitted fields."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def recording_policy(root=ROOT):
    frozen.recording_policy(root)
    raw = (Path(root) / POLICY).read_bytes()
    require(sha(raw) == POLICY_SHA, "exact soft-drift recording policy bytes required")
    policy = json.loads(raw)
    require(
        policy["schema"] == "sp_lense.soft_drift_constrained_comply_recording_policy.v1"
        and policy["output_namespace"] == OUTPUT
        and policy["quotas"] == QUOTAS
        and len(policy["allowed_artifacts"]) == len(ALLOWED)
        and set(policy["allowed_artifacts"]) == ALLOWED
        and policy["whole_record_cap_bytes"] == RECORD_CAPS
        and policy["final_artifact_cap_bytes"] == FINAL_CAPS
        and all(policy["compact_audit"][key] == value for key, value in COMPACT_LIMITS.items())
        and policy["compact_audit"]["schema"] == COMPACT_SCHEMA
        and policy["maximum_forwards"] == 216
        and policy["maximum_derivatives"] == 96
        and policy["maximum_updates"] == 8
        and policy["timeout_including_loading_seconds"] == 1800
        and policy["combined_category_bound_bytes"] == 524995016
        and policy["auxiliary_ceiling_bytes"] == 16 * 1024**2
        and policy["minimum_free_bytes"] == 1024**3
        and policy["immutable_helpers_sha256"] == IMMUTABLE_HELPERS,
        "fixed compact recording policy and implementation disagree",
    )
    return policy


def certificate_source_paths():
    # Deferred only to avoid the plan -> recording import cycle. The exact plan
    # source manifest itself is included and authenticated by the certificate.
    from scripts import soft_drift_constrained_comply_plan as plan

    return frozenset(plan.SOURCE_PATHS) - {CERTIFICATE}


def require_certificate(root=ROOT):
    root = Path(root)
    recording_policy(root)
    report = json.loads((root / CERTIFICATE).read_bytes())
    hashes = report.get("source_sha256")
    zero = ("model_loads", "tokenizer_loads", "real_forwards", "real_derivatives")
    require(
        report.get("schema") == CERTIFICATE_SCHEMA
        and report.get("status") == "MODEL_FREE_PREPARATION_CERTIFIED"
        and report.get("storage_certified") is True
        and report.get("accounting_certified") is True
        and report.get("run_authorized") is False
        and all(type(report.get(key)) is int and report[key] == 0 for key in zero)
        and isinstance(report.get("entrypoint_coverage"), dict)
        and all(report["entrypoint_coverage"].get(key) is True for key in ENTRYPOINT_COVERAGE_KEYS)
        and isinstance(report.get("finalization_coverage"), dict)
        and all(
            report["finalization_coverage"].get(key) is True for key in FINALIZATION_COVERAGE_KEYS
        )
        and report.get("recording_policy_sha256") == POLICY_SHA
        and isinstance(hashes, dict)
        and set(hashes) == certificate_source_paths()
        and all(hashes.get(path) == digest for path, digest in IMMUTABLE_HELPERS.items())
        and all(sha((root / path).read_bytes()) == digest for path, digest in hashes.items()),
        "new exact-source full-finalization preparation required; no inherited readiness",
    )
    return report


def _tree(value, depth=0):
    """Bounded scalar/small-array tree, excluding every native vector and NaN."""
    require(depth <= 10, "compact nesting depth")
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        require(abs(value) < 2**63, "compact integer range")
    elif type(value) is float:
        require(math.isfinite(value), "finite compact float")
    elif type(value) is str:
        require(len(value.encode("utf-8")) <= 512, "bounded compact string")
    elif type(value) is list:
        require(len(value) <= 12, "small compact array; native vectors forbidden")
        for item in value:
            _tree(item, depth + 1)
    elif type(value) is dict:
        require(len(value) <= 96, "bounded compact object")
        for key, item in value.items():
            require(
                type(key) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", key),
                "bounded compact field name",
            )
            hash_arrays = {"A_row_sha256": 12, "D_row_sha256": 6}
            if (
                key.endswith("sha256")
                and key not in {"vector_sha256", *hash_arrays}
                and item is not None
            ):
                require(
                    type(item) is str and re.fullmatch(r"[a-f0-9]{64}", item),
                    "exact compact SHA256",
                )
            if key in hash_arrays:
                require(
                    type(item) is list
                    and len(item) == hash_arrays[key]
                    and all(
                        type(digest) is str and re.fullmatch(r"[a-f0-9]{64}", digest)
                        for digest in item
                    ),
                    "fixed exact row-hash array",
                )
            if key == "vector_sha256":
                require(
                    type(item) is dict
                    and set(item) == {"w_before", "d", "s", "u", "w_after", "r"}
                    and all(
                        type(digest) is str and re.fullmatch(r"[a-f0-9]{64}", digest)
                        for digest in item.values()
                    ),
                    "six exact vector hashes only",
                )
            _tree(item, depth + 1)
    else:
        raise ValueError("only finite primitive compact records allowed")


def _section(records, maximum, cap, keys):
    require(type(records) is list and len(records) <= maximum, "compact section count")
    for record in records:
        require(type(record) is dict and set(record) <= keys, "compact section field whitelist")
        _tree(record)
        require(len(encode(record)) <= cap, "compact section byte cap")


def encode_verification(result, *, audit_match=AUDIT_MATCH):
    require(type(result) is dict and set(result) <= TOP_KEYS, "compact audit field whitelist")
    require(result.get("status") == audit_match, "independent audit match required")
    require(result.get("schema", COMPACT_SCHEMA) == COMPACT_SCHEMA, "compact audit schema")
    require(type(result.get("summary")) is dict, "compact summary required")
    common = dict(result)
    summary = dict(common["summary"])
    trajectory = summary.pop("paired_response_trajectory", [])
    common["summary"] = summary
    rows = common.pop("row_checks", [])
    updates = common.pop("optimizer_checks", [])
    stages = common.pop("construction_stages", [])
    _section(rows, 216, 2048, ROW_KEYS)
    _section(updates, 8, 32768, UPDATE_KEYS)
    _section(trajectory, 10, 4096, TRAJECTORY_KEYS)
    _section(stages, 96, 2048, ROW_KEYS)
    require(
        24 <= len(rows) <= 216
        and len(rows) % 12 == 0
        and summary.get("forward_count") == len(rows)
        and summary.get("derivative_count")
        == sum(row.get("recorded_gradient_present") is True for row in rows)
        and 0 <= summary["derivative_count"] <= 96
        and summary.get("attempted_updates") == len(updates)
        and len({row["cell_id"] for row in rows}) == len(rows)
        and [row.get("rows_jsonl_line") for row in rows] == list(range(1, len(rows) + 1))
        and sum(row.get("condition") == "baseline" for row in rows) == 12
        and sum(row.get("condition") == "final" for row in rows) == 12,
        "compact full-schedule row/count identities",
    )
    _tree(common)
    # Duplicate construction-stage scalar records, if supplied, belong to the
    # common envelope, not to an unaccounted additional section.
    require(
        len(encode({**common, "construction_stages": stages})) <= 262144,
        "complete common audit envelope",
    )
    raw = encode(result)
    require(
        len(raw) <= COMPACT_LIMITS["complete_verification_bound_bytes"],
        "complete compact verification byte bound",
    )
    return raw


def _eligible(result, audit_match):
    value = result.get("summary", {})
    eligible = result.get("status") == audit_match and value.get("candidate_eligible") is True
    if eligible:
        cells = value.get("final_cells")
        require(
            value.get("final_accepted") == 12
            and type(cells) is list
            and len(cells) == 12
            and len({row["cell_id"] for row in cells}) == 12
            and all(row.get("accepted") is True for row in cells),
            "all twelve designated final cells accepted",
        )
    return eligible


def _candidate_payload(record, verified, verification_sha):
    required = {
        "vector",
        "vector_float64_le_sha256",
        "final_cell_ids",
        "requested",
        "construction_only",
        "transfer_ran",
        "training_families",
        "verification_sha256",
        "endpoint_sha256",
        "valid_only_with_complete_final_recording_inventory",
    }
    require(type(record) is dict and set(record) == required, "exact candidate payload schema")
    w = record["vector"]
    require(
        type(w) is list
        and len(w) == 1024
        and all(type(x) in (float, int) and math.isfinite(x) for x in w)
        and math.sqrt(math.fsum(x * x for x in w)) <= 0.20 + 1e-12,
        "actual native bounded candidate; no renormalization",
    )
    require(
        record["vector_float64_le_sha256"] == sha(struct.pack("<1024d", *w))
        and record["verification_sha256"] == verification_sha
        and record["endpoint_sha256"] == verified["audited_endpoint_sha256"]
        and record["final_cell_ids"]
        == [row["cell_id"] for row in verified["summary"]["final_cells"]]
        and record["requested"] == "comply"
        and record["construction_only"] is True
        and record["transfer_ran"] is False
        and record["training_families"]
        == ["cg_f01_archive_closeout", "cg_f02_translation_console", "cg_f03_context_rotation"]
        and record["valid_only_with_complete_final_recording_inventory"] is True,
        "candidate identity and frozen scientific scope",
    )


def prepare_final_artifacts(
    result,
    *,
    prepare_candidate_records,
    render_report,
    runtime_status,
    capture_receipt,
    audit_match=AUDIT_MATCH,
):
    """Pure preencoding: no file is opened until the complete bundle is admitted."""
    raw = encode_verification(result, audit_match=audit_match)
    accepted = _eligible(result, audit_match)
    candidates = prepare_candidate_records(result, sha(raw))
    require(
        type(candidates) is dict
        and set(candidates) == ({"comply_vector.json"} if accepted else set()),
        "candidate payload iff independently eligible",
    )
    artifacts = {"verification.json": raw}
    if accepted:
        record = candidates["comply_vector.json"]
        _candidate_payload(record, result, sha(raw))
        artifacts["comply_vector.json"] = encode(record)
        artifacts["candidate_freeze.json"] = encode(
            {
                "sha256": sha(artifacts["comply_vector.json"]),
                "verification_sha256": sha(raw),
                "after_independent_audit": True,
            }
        )
    report = render_report(result)
    require(type(report) is str, "report text required")
    artifacts["PILOT_REPORT.md"] = (
        "Recording validity requires a matching complete FINAL_INVENTORY.json.\n"
        "Any recording fault overrides provisional audit/candidate/report contents.\n\n" + report
    ).encode("utf-8")
    artifacts["CLOSEOUT.json"] = encode(
        {
            "status": "PROVISIONAL_UNTIL_FINAL_INVENTORY",
            "numeric_audit_status": result["status"],
            "candidate_provisionally_eligible": accepted,
            "runtime_status": runtime_status["status"],
            "capture_complete": capture_receipt.get("status") == "complete_valid",
            "retries_allowed": False,
        }
    )
    require(
        all(len(data) <= FINAL_CAPS[name] for name, data in artifacts.items()),
        "preencoded final artifact byte cap",
    )
    require(
        sum(map(len, artifacts.values())) <= QUOTAS["final"], "complete final-category byte cap"
    )
    return artifacts, accepted


def _preflight(budget, artifacts):
    snapshot = budget.snapshot()
    require(snapshot["fault_code"] is None, "no sticky recording fault before final artifacts")
    sizes = snapshot["sizes"]
    require(not set(artifacts).intersection(sizes), "exclusive new final artifacts")
    prospective = {**sizes, **{key: len(value) for key, value in artifacts.items()}}
    categories = {}
    for name, size in prospective.items():
        category = budget.category(name)
        require(size <= budget.quotas[category], "prospective artifact bound")
        categories[category] = categories.get(category, 0) + size
    require(
        all(size <= budget.quotas[key] for key, size in categories.items())
        and sum(prospective.values()) <= budget.quotas["total"],
        "complete prospective finalization category/namespace bound",
    )


def _unconfirmed(budget, capture_receipt, runtime_status):
    failure = {
        "status": "INCONCLUSIVE",
        "reason": "worker/writer quiescence unconfirmed",
        "recording_sealed": False,
        "candidate_eligible": False,
        "retries_allowed": False,
    }
    persisted = threading.Event()

    def save():
        try:
            budget.begin_finalization(quiescent=False)
            for name, value in (
                ("capture_receipt.json", capture_receipt),
                ("RUN_STATUS.json", runtime_status),
                ("RECORDING_FAILURE.json", failure),
            ):
                budget.write_bytes(budget.root / name, encode(value), final=True)
            persisted.set()
        except BaseException:  # noqa: BLE001 - one bounded best-effort receipt, no recursive logger.
            return

    helper = threading.Thread(target=save, daemon=True)
    helper.start()
    helper.join(timeout=1.0)
    return {
        **failure,
        "failure_receipts_persisted": persisted.is_set(),
        "failure_receipt_writer_joined": not helper.is_alive(),
        "final_inventory_withheld": True,
    }


def finalize_recording(
    budget,
    capture_receipt,
    runtime_status,
    *,
    audit,
    prepare_candidate_records,
    render_report,
    audit_match=AUDIT_MATCH,
):
    """One post-quiescence audit, full preflight, exclusive writes, seal last; no retry."""
    if capture_receipt.get("quiescent") is not True:
        return _unconfirmed(budget, capture_receipt, runtime_status)
    try:
        budget.begin_finalization(quiescent=True)
        budget.write_bytes(
            budget.root / "capture_receipt.json", encode(capture_receipt), final=True
        )
        budget.write_bytes(budget.root / "RUN_STATUS.json", encode(runtime_status), final=True)
        if not (
            capture_receipt.get("status") == runtime_status.get("status") == "complete_valid"
            and budget.fault_code is None
        ):
            raise BudgetError("INCOMPLETE_CAPTURE_OR_RUNTIME")
        verified = audit()
        artifacts, accepted = prepare_final_artifacts(
            verified,
            prepare_candidate_records=prepare_candidate_records,
            render_report=render_report,
            runtime_status=runtime_status,
            capture_receipt=capture_receipt,
            audit_match=audit_match,
        )
        _preflight(budget, artifacts)
        for name, raw in artifacts.items():
            budget.write_bytes(budget.root / name, raw, final=True)
            require(
                sha((budget.root / name).read_bytes()) == sha(raw),
                "durable final artifact identity",
            )
        require(budget.fault_code is None, "no fault before candidate seal")
        inventory = budget.finalize_inventory(valid_candidate=accepted, quiescent=True)
        require(
            inventory.get("fault_code") is None and inventory["valid_candidate"] == accepted,
            "fault-free matching candidate disposition",
        )
        return {
            "status": "complete_valid",
            "numeric_audit_status": verified["status"],
            "recording_sealed": True,
            "candidate_eligible": accepted,
            "final_inventory": inventory,
            "retries_allowed": False,
        }
    except BaseException as error:  # noqa: BLE001 - preserve admitted evidence, never repair/retry.
        failed = {
            "status": "INCONCLUSIVE",
            "failure_category": "technical_recording_or_audit",
            "exception": exception_record(error),
            "candidate_eligible": False,
            "recording_sealed": False,
            "retries_allowed": False,
        }
        try:
            budget.fault("FINALIZATION_FAILURE")
            budget.write_bytes(budget.root / "RECORDING_FAILURE.json", encode(failed), final=True)
            failed["failure_receipt_persisted"] = True
        except BaseException:  # noqa: BLE001 - no recursive receipt failure.
            failed["failure_receipt_persisted"] = False
        try:
            failed["final_inventory"] = budget.finalize_inventory(
                valid_candidate=False, quiescent=True
            )
            failed["recording_sealed"] = True
        except BaseException:  # noqa: BLE001 - incomplete inventory is never called a valid seal.
            failed["final_inventory_complete"] = False
        return failed


def read_sealed_recording(output, *, audit_match=AUDIT_MATCH):
    """Read-only hash/size validation first; sticky fault precedes any audit parse."""
    output = Path(output)
    seal = PairedBudget(output, initialize=False).verify_inventory()
    inventory = seal["inventory"]
    if inventory.get("fault_code"):
        require(inventory.get("valid_candidate") is False, "fault cannot certify candidate")
        return {"status": "INCONCLUSIVE", "recording_inventory": seal, "retries_allowed": False}
    raw = (output / "verification.json").read_bytes()
    verified = json.loads(raw)
    require(
        encode_verification(verified, audit_match=audit_match) == raw,
        "canonical compact verification identity",
    )
    captured = json.loads((output / "capture_receipt.json").read_bytes())
    runtime = json.loads((output / "RUN_STATUS.json").read_bytes())
    require(
        captured.get("status") == runtime.get("status") == "complete_valid"
        and captured.get("quiescent") is True
        and inventory.get("quiescent") is True,
        "hashed complete quiescent capture/runtime",
    )
    accepted = _eligible(verified, audit_match)
    require(inventory.get("valid_candidate") == accepted, "sealed scientific disposition")
    require(
        all(
            (output / name).exists() == accepted
            for name in ("comply_vector.json", "candidate_freeze.json")
        ),
        "candidate artifacts iff accepted",
    )
    for name, key in (
        ("endpoint.json", "audited_endpoint_sha256"),
        ("result.json", "audited_result_sha256"),
    ):
        require(
            sha((output / name).read_bytes()) == verified[key],
            "audited raw endpoint/result binding",
        )
    if accepted:
        candidate_raw = (output / "comply_vector.json").read_bytes()
        candidate = json.loads(candidate_raw)
        _candidate_payload(candidate, verified, sha(raw))
        require(
            json.loads((output / "candidate_freeze.json").read_bytes())
            == {
                "sha256": sha(candidate_raw),
                "verification_sha256": sha(raw),
                "after_independent_audit": True,
            },
            "candidate freeze raw hash binding",
        )
        endpoint = json.loads((output / "endpoint.json").read_bytes())
        result = json.loads((output / "result.json").read_bytes())
        require(
            candidate["vector"] == endpoint["w"] == result["w"],
            "candidate is audited native endpoint",
        )
    for name, cap in FINAL_CAPS.items():
        if (output / name).exists():
            require((output / name).stat().st_size <= cap, "sealed final artifact cap")
    closeout = json.loads((output / "CLOSEOUT.json").read_bytes())
    require(
        closeout["numeric_audit_status"] == audit_match
        and closeout["candidate_provisionally_eligible"] == accepted,
        "hashed closeout disposition",
    )
    return {**verified, "recording_inventory": seal}
