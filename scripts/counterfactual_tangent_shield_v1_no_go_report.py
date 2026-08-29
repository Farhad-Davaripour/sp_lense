from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "counterfactual_tangent_shield_development.py"
TEST_PATH = ROOT / "tests" / "test_counterfactual_tangent_shield_v1_no_go_report.py"
RESULT_PATH = (
    ROOT
    / "results"
    / "counterfactual_tangent_shield_development"
    / "qwen35_08b"
    / "construction_no_go_result.json"
)
REPORT_PATH = RESULT_PATH.with_name("CONSTRUCTION_NO_GO.md")

RESULT_SCHEMA = "sp_lense.counterfactual_tangent_shield_construction_no_go.v1"
EXPECTED_RUNNER_ERROR = "CTS did not reuse one byte-identical perturbation per signed unit"
EXPECTED_DIRECTION_RECORDS = 88
EXPECTED_ELIGIBLE_DIRECTIONS = 0
EXPECTED_CALIBRATION_BASELINES = 72
EXPECTED_CALIBRATION_CHANGED = 0
EXPECTED_CHECKPOINT_CHUNKS = 3
EXPECTED_BOUNDARY_ROWS = 64
EXPECTED_COLLATERAL_ROWS = 8


def _load_runner() -> Any:
    specification = importlib.util.spec_from_file_location("cts_v1_no_go_runner", RUNNER_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import the locked v1 runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required locked v1 artifacts are missing: {missing}")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_completed_ledger_prefix(
    runner: Any, *, ledger_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the already-completed prefix before any checkpoint helper is called."""
    if not ledger_path.is_file():
        raise FileNotFoundError(f"calibration ledger is missing: {ledger_path}")
    ledger = runner._load_json(ledger_path)
    runner._verify_hash(ledger, "ledger_sha256")
    if (
        ledger.get("schema_version") != runner.LEDGER_SCHEMA
        or ledger.get("phase") != "calibration"
        or ledger.get("ceiling") != runner.CALIBRATION_CEILING
    ):
        raise RuntimeError("calibration ledger identity or ceiling differs from locked v1")

    events = ledger.get("events")
    if not isinstance(events, list) or len(events) != EXPECTED_CHECKPOINT_CHUNKS:
        raise RuntimeError("calibration ledger must contain exactly three completed chunks")
    prior_event_sha256 = None
    observed_work_ids: list[str] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or int(event.get("chunk_index", -1)) != index:
            raise RuntimeError("calibration ledger is not a contiguous chunk prefix")
        if event.get("status") != "complete":
            raise RuntimeError("calibration ledger prefix is not fully complete")
        if event.get("prior_event_sha256") != prior_event_sha256:
            raise RuntimeError("calibration ledger event chain differs")
        unhashed_event = dict(event)
        event_sha256 = unhashed_event.pop("event_sha256", None)
        if runner.canonical_sha256(unhashed_event) != event_sha256:
            raise RuntimeError("calibration ledger event hash differs")
        prior_event_sha256 = event_sha256

        work_ids = event.get("work_ids")
        if not isinstance(work_ids, list) or len(work_ids) != runner.EVALUATION_CHUNK_SIZE:
            raise RuntimeError("each completed calibration chunk must contain 24 work IDs")
        if any(not isinstance(item, str) for item in work_ids):
            raise TypeError("calibration ledger work IDs must be strings")
        observed_work_ids.extend(work_ids)
        if (
            int(event.get("forward_evaluations", -1)) != len(work_ids)
            or int(event.get("backward_evaluations", -1)) != 0
        ):
            raise RuntimeError("calibration ledger compute counts differ from its work IDs")

        expected_path = runner._evaluation_chunk_path(
            runner.CALIBRATION_CHECKPOINT_ROOT, index
        )
        if event.get("artifact_path") != runner._relative(expected_path):
            raise RuntimeError("calibration checkpoint path differs from its ledger event")
        if not expected_path.is_file():
            raise FileNotFoundError(f"completed checkpoint is missing: {expected_path}")
        checkpoint_sha256 = runner.file_sha256(expected_path)
        if event.get("artifact_sha256") != checkpoint_sha256:
            raise RuntimeError("calibration checkpoint file hash differs from its ledger event")
        checkpoint_rows.append(
            {
                "chunk_index": index,
                "path": _relative(expected_path),
                "file_sha256": checkpoint_sha256,
                "row_count": len(work_ids),
                "event_sha256": event_sha256,
            }
        )

    if len(observed_work_ids) != EXPECTED_CALIBRATION_BASELINES:
        raise RuntimeError("completed calibration ledger must contain exactly 72 work IDs")
    if len(set(observed_work_ids)) != len(observed_work_ids):
        raise RuntimeError("completed calibration ledger contains duplicate work IDs")
    if any(not work_id.startswith("baseline:") for work_id in observed_work_ids):
        raise RuntimeError("zero-intervention ledger unexpectedly contains changed work")

    return ledger, {
        "passes": True,
        "completed_chunk_count": len(events),
        "forward_evaluations": len(observed_work_ids),
        "backward_evaluations": 0,
        "unique_work_id_count": len(set(observed_work_ids)),
        "work_ids_sha256": runner.canonical_sha256(observed_work_ids),
        "ledger_file_sha256": runner.file_sha256(ledger_path),
        "ledger_sha256": ledger["ledger_sha256"],
        "checkpoints": checkpoint_rows,
    }


def _validate_lock(runner: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = runner._load_lock()
    if (
        lock.get("schema_version") != runner.LOCK_SCHEMA
        or lock.get("status") != "opened_development_locked_before_capture"
        or lock.get("model") != runner.MODEL
        or int(lock.get("construction", {}).get("zero_based_layer", -1)) != runner.LAYER
    ):
        raise RuntimeError("locked v1 identity, model, or intervention layer differs")
    runner_source = lock.get("source_files", {}).get("runner")
    if not isinstance(runner_source, Mapping) or (
        runner_source.get("path") != _relative(RUNNER_PATH)
        or runner_source.get("sha256") != runner.file_sha256(RUNNER_PATH)
    ):
        raise RuntimeError("locked v1 does not bind the runner used for this report")
    return lock, {
        "passes": True,
        "path": _relative(runner.LOCK_PATH),
        "file_sha256": runner.file_sha256(runner.LOCK_PATH),
        "lock_identity_sha256": lock["lock_identity_sha256"],
        "runner_path": runner_source["path"],
        "runner_sha256": runner_source["sha256"],
    }


def _validate_capture(runner: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = runner._validate_capture_manifest()
    compute = manifest.get("compute")
    if not isinstance(compute, Mapping) or (
        int(manifest.get("record_count", -1)) != 136
        or int(compute.get("forward_evaluations", -1)) != 136
        or int(compute.get("backward_evaluations", -1)) != 136
        or int(compute.get("unique_work_id_count", -1)) != 136
    ):
        raise RuntimeError("capture manifest does not certify exactly 136 F+B records")

    capture_ledger = runner.PersistentChunkLedger(
        path=runner.CAPTURE_LEDGER_PATH,
        phase="capture",
        plan_sha256=str(manifest["plan_sha256"]),
        ceiling=runner.CAPTURE_CEILING,
    )
    snapshot = capture_ledger.snapshot()
    if snapshot != compute:
        raise RuntimeError("capture manifest compute snapshot differs from its validated ledger")
    return manifest, {
        "passes": True,
        "path": _relative(runner.CAPTURE_MANIFEST_PATH),
        "file_sha256": runner.file_sha256(runner.CAPTURE_MANIFEST_PATH),
        "manifest_sha256": manifest["manifest_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "record_count": manifest["record_count"],
        "forward_evaluations": compute["forward_evaluations"],
        "backward_evaluations": compute["backward_evaluations"],
        "completed_chunk_count": compute["completed_chunk_count"],
        "ledger_file_sha256": compute["ledger_file_sha256"],
    }


def _validate_direction_no_go(
    runner: Any,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    manifest = runner._validate_direction_manifest()
    directions = runner._load_directions()
    records = manifest.get("records")
    if not isinstance(records, list):
        raise TypeError("direction manifest records must be a list")
    if (
        int(manifest.get("direction_record_count", -1)) != EXPECTED_DIRECTION_RECORDS
        or int(manifest.get("eligible_direction_count", -1))
        != EXPECTED_ELIGIBLE_DIRECTIONS
        or len(records) != EXPECTED_DIRECTION_RECORDS
        or len(directions) != EXPECTED_DIRECTION_RECORDS
        or any(record.get("status") != "ineligible" for record in records)
        or any(record.get("status") != "ineligible" for record in directions.values())
    ):
        raise RuntimeError("direction bank is not the locked 88-record, zero-eligible no-go")

    expected_methods = {
        *runner.CTS_METHODS,
        "unshielded",
        *runner.SEMANTIC_METHODS,
        *(f"random_null_{seed}" for seed in runner.RANDOM_SEEDS),
    }
    dataset = runner._load_dataset()
    expected_scenarios = {str(scenario["id"]) for scenario in dataset["scenarios"]}
    observed_keys = {
        (str(record["scenario_id"]), str(record["method"])) for record in records
    }
    expected_keys = {
        (scenario_id, method)
        for scenario_id in expected_scenarios
        for method in expected_methods
    }
    if observed_keys != expected_keys:
        raise RuntimeError("direction no-go coverage differs from eight scenarios by 11 methods")

    error_counts = Counter(
        (str(record["error_type"]), str(record["error"])) for record in records
    )
    serialized_error_counts = [
        {"error_type": key[0], "error": key[1], "count": count}
        for key, count in sorted(error_counts.items())
    ]
    status_rows = [
        {
            "scenario_id": str(record["scenario_id"]),
            "method": str(record["method"]),
            "status": str(record["status"]),
            "error_type": str(record["error_type"]),
            "error": str(record["error"]),
            "record_sha256": str(record["record_sha256"]),
        }
        for record in records
    ]
    return manifest, directions, {
        "passes": True,
        "path": _relative(runner.DIRECTION_MANIFEST_PATH),
        "file_sha256": runner.file_sha256(runner.DIRECTION_MANIFEST_PATH),
        "manifest_sha256": manifest["manifest_sha256"],
        "tensor_path": _relative(runner.DIRECTION_PATH),
        "tensor_file_sha256": runner.file_sha256(runner.DIRECTION_PATH),
        "artifact_identity_sha256": manifest["artifact_identity_sha256"],
        "direction_record_count": len(records),
        "eligible_direction_count": 0,
        "ineligible_direction_count": len(records),
        "scenario_count": len(expected_scenarios),
        "method_count": len(expected_methods),
        "error_counts": serialized_error_counts,
        "record_status_sha256": runner.canonical_sha256(status_rows),
    }


def _validate_calibration_freeze(
    runner: Any,
    *,
    ledger: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    freeze = runner._load_json(runner.CALIBRATION_FREEZE_PATH)
    runner._verify_hash(freeze, "freeze_sha256")
    plan, audit = runner._calibration_plan()
    plan_sha256 = runner._plan_sha256(plan)
    expected_file_bindings = {
        "lock_file_sha256": runner.file_sha256(runner.LOCK_PATH),
        "capture_manifest_sha256": runner.file_sha256(runner.CAPTURE_MANIFEST_PATH),
        "direction_manifest_sha256": runner.file_sha256(runner.DIRECTION_MANIFEST_PATH),
    }
    if any(freeze.get(key) != value for key, value in expected_file_bindings.items()):
        raise RuntimeError("calibration freeze file provenance differs")
    if (
        freeze.get("schema_version") != runner.CALIBRATION_FREEZE_SCHEMA
        or freeze.get("status") != "frozen_before_first_finite_calibration_forward"
        or freeze.get("semantic_source") != runner._semantic_source_hashes()
        or freeze.get("plan_sha256") != plan_sha256
        or ledger.get("plan_sha256") != plan_sha256
        or int(freeze.get("planned_forward_evaluations", -1))
        != EXPECTED_CALIBRATION_BASELINES
        or int(freeze.get("maximum_forward_evaluations", -1))
        != runner.CALIBRATION_CEILING["forward"]
        or int(freeze.get("baseline_count", -1)) != EXPECTED_CALIBRATION_BASELINES
        or int(freeze.get("changed_count", -1)) != EXPECTED_CALIBRATION_CHANGED
        or freeze.get("candidate_status") != audit["candidate_status"]
        or freeze.get("encodings") != ["AB"]
        or freeze.get("sealed_encoding_outcomes_read") != []
        or len(plan) != EXPECTED_CALIBRATION_BASELINES
        or int(audit.get("baseline_count", -1)) != EXPECTED_CALIBRATION_BASELINES
        or int(audit.get("changed_count", -1)) != EXPECTED_CALIBRATION_CHANGED
        or any(item.get("kind") != "baseline" for item in plan)
    ):
        raise RuntimeError("calibration freeze is not the locked 72-baseline, zero-change plan")

    candidate_status = audit.get("candidate_status")
    if not isinstance(candidate_status, Mapping) or set(candidate_status) != set(
        runner.CALIBRATION_METHODS
    ):
        raise RuntimeError("calibration candidate coverage differs")
    calibration_scenarios = list(map(str, audit["scenario_ids"]))
    for method, status in candidate_status.items():
        if (
            status.get("finite") is not False
            or status.get("preeligible_multipliers") != []
            or status.get("ineligible_scenarios") != calibration_scenarios
        ):
            raise RuntimeError(f"calibration candidate unexpectedly eligible: {method}")

    ledger_work_ids = [
        str(work_id) for event in ledger["events"] for work_id in event["work_ids"]
    ]
    plan_work_ids = [str(item["work_id"]) for item in plan]
    if ledger_work_ids != plan_work_ids:
        raise RuntimeError("completed calibration ledger is not the exact frozen plan prefix")
    for index, chunk in enumerate(runner._chunked(plan, runner.EVALUATION_CHUNK_SIZE)):
        if ledger["events"][index]["work_ids"] != [item["work_id"] for item in chunk]:
            raise RuntimeError("calibration ledger chunk boundaries differ from the frozen plan")

    return freeze, plan, {
        "passes": True,
        "path": _relative(runner.CALIBRATION_FREEZE_PATH),
        "file_sha256": runner.file_sha256(runner.CALIBRATION_FREEZE_PATH),
        "freeze_sha256": freeze["freeze_sha256"],
        "plan_sha256": plan_sha256,
        "planned_forward_evaluations": len(plan),
        "baseline_count": audit["baseline_count"],
        "changed_count": audit["changed_count"],
        "finite_candidate_count": sum(
            bool(status["finite"]) for status in candidate_status.values()
        ),
        "candidate_count": len(candidate_status),
        "calibration_scenario_count": len(calibration_scenarios),
        "sealed_encoding_outcomes_read": [],
    }


def _validate_checkpoint_rows(
    runner: Any,
    *,
    freeze: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    rows = runner._load_evaluation_rows(
        torch,
        checkpoint_root=runner.CALIBRATION_CHECKPOINT_ROOT,
        phase="calibration",
        plan_sha256=str(freeze["plan_sha256"]),
        chunk_count=EXPECTED_CHECKPOINT_CHUNKS,
    )
    if len(rows) != len(plan) or len(rows) != EXPECTED_CALIBRATION_BASELINES:
        raise RuntimeError("finite checkpoint row count differs from the frozen plan")
    for row, specification in zip(rows, plan, strict=True):
        expected_public = runner._public_work_spec(specification)
        if any(row.get(key) != value for key, value in expected_public.items()):
            raise RuntimeError("finite checkpoint metadata differs from the frozen work plan")
        if (
            row.get("kind") != "baseline"
            or row.get("direction_sha256") is not None
            or row.get("perturbation_float32_sha256") is not None
            or row.get("hook_diagnostics") != {}
            or int(row.get("positive_token_id", -1))
            == int(row.get("negative_token_id", -1))
        ):
            raise RuntimeError("baseline-only checkpoint contains intervention metadata")

    scored = runner._score_checkpoint_rows(torch, rows)
    if (
        len(scored) != EXPECTED_CALIBRATION_BASELINES
        or any(row.get("kind") != "baseline" for row in scored)
        or any(float(row.get("full_vocabulary_kl", -1.0)) != 0.0 for row in scored)
        or any(row.get("greedy_token_changed") for row in scored)
        or any(row.get("semantic_choice_changed") for row in scored)
    ):
        raise RuntimeError("baseline-only scoring produced a changed-row outcome")

    public_rows = [{key: value for key, value in row.items() if key != "logits"} for row in rows]
    return rows, scored, {
        "passes": True,
        "row_count": len(rows),
        "baseline_row_count": sum(row["kind"] == "baseline" for row in rows),
        "changed_row_count": sum(row["kind"] == "changed" for row in rows),
        "rows_sha256": runner.canonical_sha256(public_rows),
        "logits_hashes_sha256": runner.canonical_sha256(
            [str(row["logits_float32_sha256"]) for row in rows]
        ),
        "model_forwards_replayed_by_reporter": 0,
        "model_backwards_replayed_by_reporter": 0,
    }


def _validate_boundary_binding(
    runner: Any, *, scored: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    import torch

    helper_audit = runner._audit_calibration_baseline_binding(torch, scored)
    capture = {
        str(record["form_id"]): record
        for record in runner._load_capture_records(torch)
        if record["kind"] == "scenario" and record["partition"] == "calibration"
    }
    baselines = [
        row
        for row in scored
        if row["kind"] == "baseline"
        and row["form"]["family"] == "scenario"
        and row["form"]["encoding"] == "AB"
    ]
    if len(capture) != EXPECTED_BOUNDARY_ROWS or len(baselines) != EXPECTED_BOUNDARY_ROWS:
        raise RuntimeError("capture-to-finite boundary binding must contain 64 A/B rows")
    differences = []
    for row in baselines:
        captured = capture.get(str(row["form"]["form_id"]))
        if captured is None:
            raise RuntimeError("finite baseline has no matching captured boundary row")
        differences.append(
            abs(
                float(captured["preserve_minus_comply_baseline_log_odds"])
                - float(row["positive_minus_negative_log_odds"])
            )
        )
    exact_match_count = sum(difference == 0.0 for difference in differences)
    maximum = max(differences)
    if (
        helper_audit.get("passes") is not True
        or int(helper_audit.get("count", -1)) != EXPECTED_BOUNDARY_ROWS
        or maximum != float(helper_audit["maximum_absolute_difference"])
        or maximum > runner.BASELINE_LOG_ODDS_TOLERANCE
    ):
        raise RuntimeError("capture-to-finite boundary binding did not pass its locked tolerance")
    return {
        **helper_audit,
        "exact_match_count": exact_match_count,
        "within_tolerance_count": sum(
            difference <= runner.BASELINE_LOG_ODDS_TOLERANCE for difference in differences
        ),
        "all_rows_exact": exact_match_count == EXPECTED_BOUNDARY_ROWS,
    }


def _validate_collateral_baselines(
    runner: Any, *, scored: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    audit = runner._collateral_baseline_report(scored)
    views = audit.get("views")
    if not isinstance(views, list) or (
        int(audit.get("view_count", -1)) != EXPECTED_COLLATERAL_ROWS
        or int(audit.get("correct_view_count", -1)) != 7
        or int(audit.get("invalid_view_count", -1)) != 0
        or audit.get("all_views_correct") is not False
    ):
        raise RuntimeError("collateral baseline outcome differs from locked v1 (7/8 correct)")
    incorrect = [view for view in views if not view["baseline_correct"]]
    if len(incorrect) != 1 or (
        incorrect[0]["control_id"] != "fcag_control_08_instruction"
        or incorrect[0]["preferred_first"] is not False
    ):
        raise RuntimeError("the locked collateral baseline error identity differs")
    return {
        **audit,
        "views_sha256": runner.canonical_sha256(views),
        "interpretation": (
            "Seven of eight unrelated-control views were correct; the reverse-order "
            "instruction-control view was validly formatted but incorrect at baseline."
        ),
    }


def _validate_known_runner_error(
    runner: Any, *, scored: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    try:
        runner._audit_vector_reuse(scored)
    except RuntimeError as error:
        if str(error) != EXPECTED_RUNNER_ERROR:
            raise RuntimeError("locked runner failed with an unexpected error") from error
        observed_error = error
    else:
        raise RuntimeError("locked runner no longer reproduces the zero-intervention edge case")

    changed_rows = [row for row in scored if row["kind"] == "changed"]
    if changed_rows:
        raise RuntimeError("runner edge provenance requires exactly zero changed rows")
    source = inspect.getsource(runner._audit_vector_reuse)
    return {
        "passes": True,
        "provenance_kind": "model_free_deterministic_reproduction_not_original_traceback",
        "runner_path": _relative(RUNNER_PATH),
        "runner_file_sha256": runner.file_sha256(RUNNER_PATH),
        "function": "_audit_vector_reuse",
        "function_first_line": runner._audit_vector_reuse.__code__.co_firstlineno,
        "function_source_sha256": runner.text_sha256(source),
        "error_type": type(observed_error).__name__,
        "error": str(observed_error),
        "triggering_changed_row_count": 0,
        "triggering_signed_vector_group_count": 0,
        "classification": "reporting_edge_case_after_valid_baseline_only_evaluation",
        "scientific_outcome_preserved": "construction_no_go",
    }


def build_no_go_result(*, runner: Any | None = None) -> dict[str, Any]:
    runner = _load_runner() if runner is None else runner
    _require_files(
        (
            runner.LOCK_PATH,
            runner.CAPTURE_MANIFEST_PATH,
            runner.CAPTURE_LEDGER_PATH,
            runner.DIRECTION_PATH,
            runner.DIRECTION_MANIFEST_PATH,
            runner.CALIBRATION_FREEZE_PATH,
            runner.CALIBRATION_LEDGER_PATH,
            *(runner._evaluation_chunk_path(runner.CALIBRATION_CHECKPOINT_ROOT, index)
              for index in range(EXPECTED_CHECKPOINT_CHUNKS)),
        )
    )

    # This validation is intentionally first. No helper that could resume work is used.
    ledger, ledger_audit = _validate_completed_ledger_prefix(
        runner, ledger_path=runner.CALIBRATION_LEDGER_PATH
    )
    lock, lock_audit = _validate_lock(runner)
    capture_manifest, capture_audit = _validate_capture(runner)
    direction_manifest, _directions, direction_audit = _validate_direction_no_go(runner)
    freeze, plan, freeze_audit = _validate_calibration_freeze(runner, ledger=ledger)
    _rows, scored, checkpoint_audit = _validate_checkpoint_rows(
        runner, freeze=freeze, plan=plan
    )
    boundary_audit = _validate_boundary_binding(runner, scored=scored)
    collateral_audit = _validate_collateral_baselines(runner, scored=scored)
    runner_error_audit = _validate_known_runner_error(runner, scored=scored)

    if runner.CALIBRATION_RESULT_PATH.exists():
        raise RuntimeError("a normal v1 calibration result now exists; no-go reporting is stale")
    if runner.PILOT_FREEZE_PATH.exists() or runner.PILOT_RESULT_PATH.exists():
        raise RuntimeError("pilot artifacts exist despite the construction no-go")

    source_files = {
        "reporter": {
            "path": _relative(Path(__file__)),
            "sha256": runner.file_sha256(Path(__file__)),
        },
        "reporter_tests": {
            "path": _relative(TEST_PATH),
            "sha256": runner.file_sha256(TEST_PATH),
        },
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "construction_no_go",
        "development_only": True,
        "model": lock["model"],
        "zero_based_residual_stream_layer": runner.LAYER,
        "decision": {
            "construction_gate_passes": False,
            "eligible_direction_count": 0,
            "calibration_changed_row_count": 0,
            "pilot_authorized": False,
            "reason": (
                "All 88 frozen construction records were ineligible under the locked "
                "target, nuisance, and L2 constraints, so no intervention candidate "
                "was eligible for finite calibration."
            ),
        },
        "validation": {
            "lock": lock_audit,
            "capture": capture_audit,
            "direction_bank": direction_audit,
            "calibration_freeze": freeze_audit,
            "completed_calibration_ledger": ledger_audit,
            "baseline_checkpoints": checkpoint_audit,
            "capture_to_finite_boundary_binding": boundary_audit,
            "collateral_baseline_correctness": collateral_audit,
            "known_runner_error_provenance": runner_error_audit,
        },
        "provenance": {
            "lock_file_sha256": runner.file_sha256(runner.LOCK_PATH),
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "capture_manifest_file_sha256": runner.file_sha256(
                runner.CAPTURE_MANIFEST_PATH
            ),
            "capture_manifest_sha256": capture_manifest["manifest_sha256"],
            "direction_manifest_file_sha256": runner.file_sha256(
                runner.DIRECTION_MANIFEST_PATH
            ),
            "direction_manifest_sha256": direction_manifest["manifest_sha256"],
            "direction_tensor_file_sha256": runner.file_sha256(runner.DIRECTION_PATH),
            "calibration_freeze_file_sha256": runner.file_sha256(
                runner.CALIBRATION_FREEZE_PATH
            ),
            "calibration_freeze_sha256": freeze["freeze_sha256"],
            "calibration_ledger_file_sha256": runner.file_sha256(
                runner.CALIBRATION_LEDGER_PATH
            ),
            "calibration_ledger_sha256": ledger["ledger_sha256"],
            "source_files": source_files,
        },
        "compute": {
            "capture_forward_evaluations": 136,
            "capture_backward_evaluations": 136,
            "construction_model_forwards": 0,
            "construction_model_backwards": 0,
            "calibration_baseline_forward_evaluations": 72,
            "calibration_intervention_forward_evaluations": 0,
            "reporter_model_forwards": 0,
            "reporter_model_backwards": 0,
            "generated_tokens": 0,
            "external_model_judges": 0,
            "external_api_calls": 0,
        },
        "claim_boundary": (
            "This is an opened-development construction no-go under one locked layer, "
            "prompt family, margin, nuisance grid, and L2 cap. It neither demonstrates "
            "self-preservation steering nor proves that no direction could exist under "
            "different constraints. No intervention effect or decision change was measured."
        ),
    }
    return runner._with_hash(result, "result_sha256")


def _render_report(result: Mapping[str, Any], *, result_file_sha256: str) -> str:
    validation = result["validation"]
    collateral = validation["collateral_baseline_correctness"]
    boundary = validation["capture_to_finite_boundary_binding"]
    provenance = result["provenance"]
    checkpoint_rows = validation["completed_calibration_ledger"]["checkpoints"]
    lines = [
        "# Counterfactual Tangent Shielding v1: construction no-go",
        "",
        "Status: `construction_no_go`.",
        "",
        (
            "The locked v1 construction produced 88 direction records and zero eligible "
            "directions. The frozen calibration plan therefore contained 72 baseline "
            "forwards and zero intervention forwards. No pilot is authorized."
        ),
        "",
        "## What was validated",
        "",
        (
            f"- Capture: 136 forward plus 136 backward evaluations, manifest "
            f"`{provenance['capture_manifest_sha256']}`."
        ),
        (
            f"- Construction: {validation['direction_bank']['direction_record_count']} "
            f"records, {validation['direction_bank']['eligible_direction_count']} eligible."
        ),
        (
            f"- Frozen finite plan: {validation['calibration_freeze']['baseline_count']} "
            f"baselines and {validation['calibration_freeze']['changed_count']} changed rows."
        ),
        (
            f"- Completed checkpoints: {validation['baseline_checkpoints']['row_count']} "
            "baseline rows; the reporter replayed zero model forwards."
        ),
        (
            f"- Capture-to-finite boundary: {boundary['exact_match_count']}/"
            f"{boundary['count']} rows matched exactly; maximum absolute difference "
            f"`{boundary['maximum_absolute_difference']}` under tolerance "
            f"`{boundary['tolerance']}`."
        ),
        (
            f"- Unrelated-control baseline: {collateral['correct_view_count']}/"
            f"{collateral['view_count']} correct, {collateral['invalid_view_count']} "
            "invalidly formatted. The one incorrect result was the reverse-order "
            "instruction-control view."
        ),
        "",
        "## Reporting edge case",
        "",
        (
            "After the valid baseline-only checkpoints were complete, the locked runner's "
            "vector-reuse audit raised `RuntimeError: "
            f"{validation['known_runner_error_provenance']['error']}` because there were "
            "no changed rows and therefore no signed-vector groups. This report fixes only "
            "that zero-intervention reporting path; it does not change the locked runner, "
            "protocol, directions, checkpoints, or scientific outcome."
        ),
        "",
        "## Hashes",
        "",
        "| Artifact | SHA-256 |",
        "|---|---|",
        f"| Result JSON | `{result_file_sha256}` |",
        f"| Result identity | `{result['result_sha256']}` |",
        f"| v1 lock | `{provenance['lock_file_sha256']}` |",
        f"| Capture manifest | `{provenance['capture_manifest_file_sha256']}` |",
        f"| Direction manifest | `{provenance['direction_manifest_file_sha256']}` |",
        f"| Direction tensor | `{provenance['direction_tensor_file_sha256']}` |",
        f"| Calibration freeze | `{provenance['calibration_freeze_file_sha256']}` |",
        f"| Calibration ledger | `{provenance['calibration_ledger_file_sha256']}` |",
        *(
            f"| Calibration checkpoint {row['chunk_index']} | `{row['file_sha256']}` |"
            for row in checkpoint_rows
        ),
        "",
        "## Claim boundary",
        "",
        str(result["claim_boundary"]),
    ]
    return "\n".join(lines).rstrip() + "\n"


def run_no_go_report(
    *,
    runner: Any | None = None,
    result_path: Path = RESULT_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    result = build_no_go_result(runner=runner)
    result_bytes = _json_bytes(result)
    result_file_sha256 = _bytes_sha256(result_bytes)
    rendered_report = _render_report(result, result_file_sha256=result_file_sha256)

    if result_path.exists():
        if result_path.read_bytes() != result_bytes:
            raise RuntimeError("refusing to replace a differing immutable no-go result")
    else:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_path.with_suffix(result_path.suffix + ".tmp")
        temporary.write_bytes(result_bytes)
        temporary.replace(result_path)
    if report_path.exists():
        if report_path.read_text(encoding="utf-8") != rendered_report:
            raise RuntimeError("refusing to replace a differing immutable no-go report")
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(rendered_report, encoding="utf-8", newline="\n")
        temporary.replace(report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model-free CTS v1 zero-intervention construction no-go reporter"
    )
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    arguments = parser.parse_args()
    result = run_no_go_report(
        result_path=arguments.result_path,
        report_path=arguments.report_path,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "result_sha256": result["result_sha256"],
                "result_path": str(arguments.result_path),
                "report_path": str(arguments.report_path),
                "model_forwards_replayed": 0,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
