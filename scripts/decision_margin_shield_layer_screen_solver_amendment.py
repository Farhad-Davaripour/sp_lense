from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from sp_lense.counterfactual_tangent_shield import TangentShieldInfeasibleError
from sp_lense.decision_margin_shield import (
    DEFAULT_CAP_FRONTIER,
    DEFAULT_MARGIN,
    DEFAULT_QUALIFICATION_CAP,
    METHODS,
    _method_record,
    certify_minimum_l2_candidate,
    decision_margin_bounds,
    select_layer,
)
from sp_lense.decision_margin_shield_rowspace import (
    solve_certified_rowspace_minimum_l2_direction,
)
from sp_lense.factorial_causal_anchor import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
PROTOCOL_PATH = (
    ROOT / "docs" / "DECISION_MARGIN_SHIELD_LAYER_SCREEN_SOLVER_AMENDMENT.md"
)
SOLVER_PATH = ROOT / "src" / "sp_lense" / "decision_margin_shield_rowspace.py"
SOLVER_TEST_PATH = ROOT / "tests" / "test_decision_margin_shield_rowspace.py"
RUNNER_TEST_PATH = (
    ROOT / "tests" / "test_decision_margin_shield_layer_screen_solver_amendment_runner.py"
)

ORIGINAL_RUNNER_PATH = ROOT / "scripts" / "decision_margin_shield_layer_screen.py"
ORIGINAL_PROTOCOL_PATH = ROOT / "docs" / "DECISION_MARGIN_SHIELD_LAYER_SCREEN_PROTOCOL.md"
ORIGINAL_GEOMETRY_PATH = ROOT / "src" / "sp_lense" / "decision_margin_shield.py"
ORIGINAL_LOCK_PATH = ROOT / "configs" / "decision_margin_shield_layer_screen_lock.json"
ORIGINAL_CAPTURE_MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "decision_margin_shield_layer_screen"
    / "qwen35_08b"
    / "capture_manifest.json"
)
ORIGINAL_SCREEN_RESULT_PATH = (
    ROOT
    / "results"
    / "decision_margin_shield_layer_screen"
    / "qwen35_08b"
    / "layer_screen_result.json"
)
ORIGINAL_FAILURE_PATH = (
    ROOT
    / "results"
    / "decision_margin_shield_layer_screen"
    / "qwen35_08b"
    / "locked_screen_attempt_failure.json"
)

LOCK_PATH = (
    ROOT / "configs" / "decision_margin_shield_layer_screen_solver_amendment_lock.json"
)
ARTIFACT_ROOT = (
    ROOT
    / "artifacts"
    / "decision_margin_shield_layer_screen_solver_amendment"
    / "qwen35_08b"
)
RESULT_ROOT = (
    ROOT
    / "results"
    / "decision_margin_shield_layer_screen_solver_amendment"
    / "qwen35_08b"
)
PREFLIGHT_PATH = ARTIFACT_ROOT / "preflight.json"
SCREEN_RESULT_PATH = RESULT_ROOT / "layer_screen_result.json"
REPORT_PATH = RESULT_ROOT / "LAYER_SCREEN_REPORT.md"

LOCK_SCHEMA = "sp_lense.decision_margin_shield_layer_screen_solver_amendment_lock.v1"
PREFLIGHT_SCHEMA = (
    "sp_lense.decision_margin_shield_layer_screen_solver_amendment_preflight.v1"
)
RESULT_SCHEMA = "sp_lense.decision_margin_shield_layer_screen_solver_amendment_result.v1"
ORIGINAL_SOURCE_COMMIT = "644f82b784307e0e05c4cfaa257c1ed3f373ae70"
ORIGINAL_FILE_HASHES = {
    "runner": "1505e3b4fa4cfebb8780dab962a97eb30d1f4e2b8477d4014d7c993d18c2ac6a",
    "geometry": "12d8517b335c4e77b8ee47483fc8f267bdd5517a3b5a8c77e4ed2ea754b8ca54",
    "protocol": "bb9679b22b9ef27a5689e12f5ef702d328c7f6196bda317c90d3fd96b069bca0",
}
ORIGINAL_LOCK_FILE_SHA256 = (
    "97defe3511342a7b1f69ef188e796aaa018bee611009c09f9e98a71af0425261"
)
ORIGINAL_LOCK_IDENTITY_SHA256 = (
    "a3eacfbfc5b5cdad06a55ca9077677d415fc92e3109733b9823f00260be967b9"
)
ORIGINAL_CAPTURE_FILE_SHA256 = (
    "0d3720ef0bcda3e6dd430aa6033b949404b726e4f616ada86e26b2bbc472a939"
)
ORIGINAL_CAPTURE_IDENTITY_SHA256 = (
    "cf654fa4bc42ea550138653a4927232888a3724cfb9451bf97b7b5551740faf0"
)
FAILURE_RECORD_IDENTITY_SHA256 = (
    "3f1e13064cc792231ac288acf32c2680c0e73e16c377f14839ef6fd2ddaaebcf"
)
LAYERS = tuple(range(23))
MARGIN = DEFAULT_MARGIN
CAP_FRONTIER = DEFAULT_CAP_FRONTIER
QUALIFICATION_CAP = DEFAULT_QUALIFICATION_CAP

_ORIGINAL_RUNNER: ModuleType | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(dict(value), indent=2, ensure_ascii=False, allow_nan=False)
    _atomic_text(path, rendered + "\n")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace immutable artifact: {path}")
    _write_json(path, value)


def _with_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = canonical_sha256(result)
    return result


def _verify_hash(value: Mapping[str, Any], field: str) -> None:
    observed = value.get(field)
    if not isinstance(observed, str):
        raise TypeError(f"artifact lacks {field}")
    unhashed = dict(value)
    del unhashed[field]
    if canonical_sha256(unhashed) != observed:
        raise RuntimeError(f"artifact {field} self-check failed")


def _load_original_runner() -> ModuleType:
    global _ORIGINAL_RUNNER
    if _ORIGINAL_RUNNER is not None:
        return _ORIGINAL_RUNNER
    specification = importlib.util.spec_from_file_location(
        "sp_lense_locked_decision_margin_shield_layer_screen",
        ORIGINAL_RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not dynamically import the original DMS runner")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    _ORIGINAL_RUNNER = module
    return module


def _require_original_result_absent() -> None:
    if ORIGINAL_SCREEN_RESULT_PATH.exists():
        raise RuntimeError(
            "the original DMS SCREEN_RESULT_PATH must remain absent for this amendment"
        )


def _validate_original_failure() -> dict[str, Any]:
    failure = _load_json(ORIGINAL_FAILURE_PATH)
    _verify_hash(failure, "failure_record_sha256")
    diagnosed = failure.get("failure", {}).get("first_diagnosed_cell", {})
    disclosure = failure.get("diagnostic_disclosure", {})
    if (
        failure.get("failure_record_sha256") != FAILURE_RECORD_IDENTITY_SHA256
        or failure.get("status") != "failed_before_result_write"
        or failure.get("original_lock", {}).get("file_sha256")
        != ORIGINAL_LOCK_FILE_SHA256
        or failure.get("original_lock", {}).get("lock_identity_sha256")
        != ORIGINAL_LOCK_IDENTITY_SHA256
        or failure.get("immutable_capture", {}).get("file_sha256")
        != ORIGINAL_CAPTURE_FILE_SHA256
        or failure.get("immutable_capture", {}).get("manifest_sha256")
        != ORIGINAL_CAPTURE_IDENTITY_SHA256
        or failure.get("failure", {}).get("exception_type")
        != "sp_lense.counterfactual_tangent_shield.TangentShieldSolverError"
        or failure.get("failure", {}).get("exception_message")
        != "the minimum-L2 solve did not converge"
        or diagnosed
        != {
            "scenario_id": "fcag_dev_01_weather_alert",
            "layer": 22,
            "method": "unrelated_only",
        }
        or disclosure.get("partial_calibration_geometry_viewed") is not True
        or disclosure.get("pilot_geometry_viewed") is not False
        or disclosure.get("finite_intervention_outcomes_viewed") is not False
        or failure.get("original_result", {}).get("exists") is not False
        or failure.get("compute", {}).get("generated_tokens") != 0
    ):
        raise RuntimeError("the original locked-attempt failure record differs")
    return failure


def _exact_chunk_inventory(
    original: ModuleType,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    manifest_chunks = manifest.get("chunks")
    if not isinstance(manifest_chunks, list):
        raise TypeError("the original capture manifest lacks a chunk list")
    inventory: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    for record in manifest_chunks:
        if not isinstance(record, Mapping):
            raise TypeError("the original capture chunk record must be an object")
        path = original._chunk_path_from_record(record).resolve()
        expected_paths.add(path)
        if not path.is_file() or file_sha256(path) != record.get("file_sha256"):
            raise RuntimeError("an original capture chunk differs from its manifest")
        inventory.append(
            {
                "index": int(record["index"]),
                "path": _relative(path),
                "file_sha256": str(record["file_sha256"]),
                "file_size_bytes": path.stat().st_size,
                "record_count": int(record["record_count"]),
            }
        )
    actual_paths = {
        path.resolve()
        for path in original.CAPTURE_ROOT.glob("chunk-*.pt")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise RuntimeError("the original capture chunk inventory has extra or missing files")
    return inventory


def _validate_original_state() -> dict[str, Any]:
    _require_original_result_absent()
    original = _load_original_runner()
    original_lock = original._load_lock()
    manifest = original._validate_capture_manifest()
    failure = _validate_original_failure()
    observed_old_hashes = {
        "runner": file_sha256(ORIGINAL_RUNNER_PATH),
        "geometry": file_sha256(ORIGINAL_GEOMETRY_PATH),
        "protocol": file_sha256(ORIGINAL_PROTOCOL_PATH),
    }
    if (
        file_sha256(ORIGINAL_LOCK_PATH) != ORIGINAL_LOCK_FILE_SHA256
        or original_lock.get("lock_identity_sha256")
        != ORIGINAL_LOCK_IDENTITY_SHA256
        or file_sha256(ORIGINAL_CAPTURE_MANIFEST_PATH)
        != ORIGINAL_CAPTURE_FILE_SHA256
        or manifest.get("manifest_sha256") != ORIGINAL_CAPTURE_IDENTITY_SHA256
        or observed_old_hashes != ORIGINAL_FILE_HASHES
    ):
        raise RuntimeError("the immutable original DMS state differs")
    return {
        "lock": original_lock,
        "manifest": manifest,
        "failure": failure,
        "chunk_inventory": _exact_chunk_inventory(original, manifest),
    }


def _source_paths() -> dict[str, Path]:
    return {
        "protocol": PROTOCOL_PATH,
        "runner": SCRIPT_PATH,
        "rowspace_solver": SOLVER_PATH,
        "rowspace_solver_tests": SOLVER_TEST_PATH,
        "runner_tests": RUNNER_TEST_PATH,
    }


def _source_records() -> dict[str, dict[str, str]]:
    missing = [str(path) for path in _source_paths().values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"amendment source files are missing: {missing}")
    return {
        name: {"path": _relative(path), "sha256": file_sha256(path)}
        for name, path in _source_paths().items()
    }


def _validate_bound_sources(records: Mapping[str, Any]) -> None:
    expected = _source_records()
    if dict(records) != expected:
        raise RuntimeError("the amendment source-file binding differs")


def proposed_lock() -> dict[str, Any]:
    state = _validate_original_state()
    original_lock = state["lock"]
    manifest = state["manifest"]
    payload = {
        "schema_version": LOCK_SCHEMA,
        "status": "model_free_solver_amendment_locked_before_screen",
        "development_only": True,
        "model_free": True,
        "outcome_awareness": {
            "partial_calibration_geometry_viewed_after_original_failure": True,
            "scope": (
                "fcag_dev_01_weather_alert layers 0 through 21 partial norms "
                "plus the diagnosed layer-22 failure"
            ),
            "pilot_geometry_viewed": False,
            "finite_intervention_outcomes_viewed": False,
            "disclosure_is_a_limit_on_confirmatory_interpretation": True,
        },
        "immutable_original": {
            "source_commit": ORIGINAL_SOURCE_COMMIT,
            "source_files": {
                "runner": {
                    "path": _relative(ORIGINAL_RUNNER_PATH),
                    "sha256": ORIGINAL_FILE_HASHES["runner"],
                },
                "geometry": {
                    "path": _relative(ORIGINAL_GEOMETRY_PATH),
                    "sha256": ORIGINAL_FILE_HASHES["geometry"],
                },
                "protocol": {
                    "path": _relative(ORIGINAL_PROTOCOL_PATH),
                    "sha256": ORIGINAL_FILE_HASHES["protocol"],
                },
            },
            "lock": {
                "path": _relative(ORIGINAL_LOCK_PATH),
                "file_sha256": ORIGINAL_LOCK_FILE_SHA256,
                "lock_identity_sha256": ORIGINAL_LOCK_IDENTITY_SHA256,
            },
            "capture": {
                "manifest_path": _relative(ORIGINAL_CAPTURE_MANIFEST_PATH),
                "manifest_file_sha256": ORIGINAL_CAPTURE_FILE_SHA256,
                "manifest_sha256": ORIGINAL_CAPTURE_IDENTITY_SHA256,
                "capture_plan_sha256": manifest["capture_plan_sha256"],
                "prompt_content_sha256": manifest["prompt_content_sha256"],
                "chunk_inventory": state["chunk_inventory"],
                "chunk_inventory_sha256": canonical_sha256(state["chunk_inventory"]),
            },
            "failure_record": {
                "path": _relative(ORIGINAL_FAILURE_PATH),
                "file_sha256": file_sha256(ORIGINAL_FAILURE_PATH),
                "failure_record_sha256": FAILURE_RECORD_IDENTITY_SHA256,
            },
            "original_screen_result": {
                "path": _relative(ORIGINAL_SCREEN_RESULT_PATH),
                "required_absent": True,
            },
        },
        "scientific_design": {
            "dataset": original_lock["dataset"],
            "capture": original_lock["capture"],
            "geometry": original_lock["geometry"],
            "methods": list(METHODS),
            "layers": list(LAYERS),
            "target_margin": MARGIN,
            "cap_frontier": list(CAP_FRONTIER),
            "qualification_cap": QUALIFICATION_CAP,
            "selection_partition": "calibration_only",
            "pilot_geometry_computed": False,
            "pilot_construction_computed": False,
        },
        "numerical_amendment": {
            "solver": (
                "solve_certified_rowspace_minimum_l2_direction_with_"
                "independent_full_coordinate_certificate"
            ),
            "lossless_representer_reduction": True,
            "changes_prompt_margin_cap_layer_or_tie_break": False,
            "original_failure_type": (
                "sp_lense.counterfactual_tangent_shield.TangentShieldSolverError"
            ),
            "original_failure_message": "the minimum-L2 solve did not converge",
            "first_diagnosed_cell": {
                "scenario_id": "fcag_dev_01_weather_alert",
                "layer": 22,
                "method": "unrelated_only",
            },
        },
        "compute_ceiling": {
            "capture_calls": 0,
            "model_loads": 0,
            "model_forwards": 0,
            "model_backwards": 0,
            "finite_intervention_forwards": 0,
            "finite_intervention_backwards": 0,
            "generated_tokens": 0,
            "external_model_judges": 0,
            "external_api_calls": 0,
            "paid_model_cost_usd": 0,
        },
        "source_files": _source_records(),
        "claim_boundary": (
            "Outcome-aware opened calibration geometry only; no finite steering "
            "effect, natural mechanism, safety proof, unchanged capability, "
            "priority, or publication claim."
        ),
    }
    payload["lock_identity_sha256"] = canonical_sha256(payload)
    return payload


def run_lock() -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise FileExistsError(f"refusing to replace existing lock: {LOCK_PATH}")
    lock = proposed_lock()
    _write_new_json(LOCK_PATH, lock)
    return lock


def _load_lock() -> dict[str, Any]:
    lock = _load_json(LOCK_PATH)
    if lock != proposed_lock():
        raise RuntimeError("the solver-amendment lock differs from its bound design")
    _validate_bound_sources(lock["source_files"])
    return lock


def _preflight_payload(lock: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    manifest = state["manifest"]
    return _with_hash(
        {
            "schema_version": PREFLIGHT_SCHEMA,
            "status": "ready_for_model_free_screen",
            "development_only": True,
            "lock_file_sha256": file_sha256(LOCK_PATH),
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "original_lock_file_sha256": ORIGINAL_LOCK_FILE_SHA256,
            "original_lock_identity_sha256": ORIGINAL_LOCK_IDENTITY_SHA256,
            "capture_manifest_file_sha256": ORIGINAL_CAPTURE_FILE_SHA256,
            "capture_manifest_sha256": ORIGINAL_CAPTURE_IDENTITY_SHA256,
            "capture_plan_sha256": manifest["capture_plan_sha256"],
            "prompt_content_sha256": manifest["prompt_content_sha256"],
            "chunk_inventory_sha256": canonical_sha256(state["chunk_inventory"]),
            "failure_record_file_sha256": file_sha256(ORIGINAL_FAILURE_PATH),
            "failure_record_sha256": FAILURE_RECORD_IDENTITY_SHA256,
            "original_screen_result_absent": True,
            "calibration_scenario_count": 4,
            "pilot_geometry_computed": False,
            "capture_calls": 0,
            "model_loads": 0,
            "model_forwards": 0,
            "model_backwards": 0,
            "generated_tokens": 0,
        },
        "preflight_sha256",
    )


def run_preflight() -> dict[str, Any]:
    lock = _load_lock()
    state = _validate_original_state()
    expected = _preflight_payload(lock, state)
    if PREFLIGHT_PATH.exists():
        observed = _load_json(PREFLIGHT_PATH)
        if observed != expected:
            raise RuntimeError("the existing solver-amendment preflight differs")
        return observed
    _write_new_json(PREFLIGHT_PATH, expected)
    return expected


def _validate_preflight() -> dict[str, Any]:
    lock = _load_lock()
    state = _validate_original_state()
    observed = _load_json(PREFLIGHT_PATH)
    _verify_hash(observed, "preflight_sha256")
    if observed != _preflight_payload(lock, state):
        raise RuntimeError("the solver-amendment preflight provenance differs")
    return observed


def _finite_matrix(value: Any, *, rows: int, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64, order="C").copy(order="C")
    if result.ndim != 2 or result.shape[0] != rows or result.shape[1] == 0:
        raise ValueError(f"{field} must have exactly {rows} non-empty rows")
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must be finite")
    return result


def _finite_vector(value: Any, *, length: int, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64, order="C").copy(order="C")
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{field} must have exactly {length} finite values")
    return result


def _screen_scenario_layer_rowspace(
    *,
    target_rows: Any,
    target_offsets: Any,
    protected_rows: Any,
    protected_offsets: Any,
    unrelated_rows: Any,
) -> list[dict[str, Any]]:
    target = _finite_matrix(target_rows, rows=4, field="target_rows")
    target_b = _finite_vector(target_offsets, length=4, field="target_offsets")
    protected = _finite_matrix(protected_rows, rows=12, field="protected_rows")
    protected_b = _finite_vector(
        protected_offsets,
        length=12,
        field="protected_offsets",
    )
    unrelated = _finite_matrix(unrelated_rows, rows=8, field="unrelated_rows")
    if protected.shape[1] != target.shape[1] or unrelated.shape[1] != target.shape[1]:
        raise ValueError("all DMS row matrices must have equal width")
    protected_bounds = decision_margin_bounds(protected_b, margin=MARGIN)
    small_count = int(np.count_nonzero(np.abs(protected_b) < MARGIN))
    definitions = (
        ("unshielded", None, np.zeros(0), 0, 0),
        ("unrelated_only", unrelated, np.zeros(8), 8, 0),
        (
            "decision_margin_shield",
            np.vstack((unrelated, protected)),
            np.concatenate((np.zeros(8), protected_bounds)),
            8,
            12,
        ),
    )
    records: list[dict[str, Any]] = []
    for method, nuisance, nuisance_bounds, unrelated_count, protected_count in definitions:
        try:
            solution = solve_certified_rowspace_minimum_l2_direction(
                target,
                target_b,
                margin=MARGIN,
                nuisance_rows=nuisance,
                nuisance_bound=nuisance_bounds,
            )
            error = None
            certificate = certify_minimum_l2_candidate(
                solution.direction,
                target,
                target_b,
                margin=MARGIN,
                nuisance_rows=nuisance,
                nuisance_bound=nuisance_bounds,
            )
            embedded = solution.diagnostics.get("optimality_certificate")
            if embedded != certificate or not bool(certificate.get("passes")):
                raise RuntimeError(
                    "the row-space solver and independent certificate disagree"
                )
        except TangentShieldInfeasibleError as caught:
            solution = None
            error = caught
            certificate = None
        records.append(
            _method_record(
                method=method,
                solution=solution,
                error=error,
                optimality_certificate=certificate,
                cap_frontier=CAP_FRONTIER,
                target_count=4,
                exact_unrelated_count=unrelated_count,
                protected_count=protected_count,
                protected_bounds=(
                    protected_bounds
                    if method == "decision_margin_shield"
                    else np.zeros(0)
                ),
                small_baseline_protected_count=(
                    small_count if method == "decision_margin_shield" else 0
                ),
            )
        )
    return records


def _validate_screen_result() -> dict[str, Any]:
    result = _load_json(SCREEN_RESULT_PATH)
    _verify_hash(result, "result_sha256")
    lock = _load_lock()
    preflight = _validate_preflight()
    state = _validate_original_state()
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("lock_file_sha256") != file_sha256(LOCK_PATH)
        or result.get("lock_identity_sha256") != lock["lock_identity_sha256"]
        or result.get("preflight_sha256") != preflight["preflight_sha256"]
        or result.get("capture_manifest_file_sha256")
        != ORIGINAL_CAPTURE_FILE_SHA256
        or result.get("capture_manifest_sha256")
        != ORIGINAL_CAPTURE_IDENTITY_SHA256
        or result.get("chunk_inventory_sha256")
        != canonical_sha256(state["chunk_inventory"])
        or result.get("failure_record_sha256") != FAILURE_RECORD_IDENTITY_SHA256
        or result.get("original_screen_result_absent") is not True
        or result.get("outcome_aware_to_partial_calibration_geometry") is not True
        or result.get("pilot_scenario_geometry_computed") is not False
        or result.get("pilot_construction_computed") is not False
        or result.get("amendment_capture_calls") != 0
        or result.get("amendment_model_loads") != 0
        or result.get("screen_model_forwards") != 0
        or result.get("screen_model_backwards") != 0
        or result.get("generated_tokens") != 0
        or result.get("finite_intervention_outcomes_inspected") is not False
        or result.get("external_model_judges") != 0
        or result.get("external_api_calls") != 0
        or result.get("paid_model_cost_usd") != 0
    ):
        raise RuntimeError("the solver-amendment screen result provenance differs")
    records = result.get("geometry_records")
    if not isinstance(records, list) or len(records) != 23 * 4 * len(METHODS):
        raise RuntimeError("the solver-amendment geometry coverage differs")
    calibration_ids = lock["scientific_design"]["dataset"][
        "calibration_scenario_ids"
    ]
    expected = {
        (layer, scenario_id, method)
        for layer in LAYERS
        for scenario_id in calibration_ids
        for method in METHODS
    }
    observed = set()
    for record in records:
        _verify_hash(record, "screen_record_sha256")
        method_record = dict(record)
        for field in (
            "scenario_id",
            "partition",
            "layer",
            "residual_scale",
            "target_rows_sha256",
            "protected_rows_sha256",
            "unrelated_rows_sha256",
            "target_offsets_sha256",
            "protected_offsets_sha256",
            "numerical_amendment",
            "screen_record_sha256",
        ):
            method_record.pop(field)
        _verify_hash(method_record, "geometry_record_sha256")
        if record.get("partition") != "calibration":
            raise RuntimeError("the solver-amendment result contains non-calibration geometry")
        observed.add(
            (int(record["layer"]), str(record["scenario_id"]), str(record["method"]))
        )
    if observed != expected:
        raise RuntimeError("the solver-amendment geometry grid differs")
    selection = result.get("selection")
    if not isinstance(selection, Mapping):
        raise TypeError("the solver-amendment result lacks selection")
    _verify_hash(selection, "selection_sha256")
    expected_selection = select_layer(
        records,
        calibration_scenario_ids=calibration_ids,
        layers=LAYERS,
        cap_frontier=CAP_FRONTIER,
        qualification_cap=QUALIFICATION_CAP,
    )
    if dict(selection) != expected_selection:
        raise RuntimeError("the solver-amendment selection is not reproducible")
    if result.get("status") != selection.get("status"):
        raise RuntimeError("the solver-amendment status differs from selection")
    return result


def run_screen() -> dict[str, Any]:
    lock = _load_lock()
    preflight = _validate_preflight()
    state = _validate_original_state()
    if SCREEN_RESULT_PATH.exists():
        return _validate_screen_result()
    original = _load_original_runner()
    import torch

    records = original._load_capture_records(torch)
    dataset = original._load_dataset()
    nuisance_records = [record for record in records if record["kind"] == "nuisance_fit"]
    if len(nuisance_records) != 8:
        raise RuntimeError("the solver amendment requires eight unrelated gradients")
    calibration_scenarios = [
        scenario
        for scenario in dataset["scenarios"]
        if scenario["partition"] == "calibration"
    ]
    if len(calibration_scenarios) != 4:
        raise RuntimeError("the solver amendment requires four calibration scenarios")
    geometry_records: list[dict[str, Any]] = []
    for scenario in calibration_scenarios:
        scenario_id = str(scenario["id"])
        scenario_records = [
            record
            for record in records
            if record["kind"] == "scenario" and record["scenario_id"] == scenario_id
        ]
        if len(scenario_records) != 16:
            raise RuntimeError("each calibration scenario requires 16 captured forms")
        residual_scales = original.anchor_residual_scale_geometric_mean(
            torch,
            [record["anchor_residual"] for record in scenario_records],
        )
        target_records = [
            record
            for record in scenario_records
            if record["target"] == "self" and record["event"] == "permanent"
        ]
        protected_records = [
            record
            for record in scenario_records
            if not (record["target"] == "self" and record["event"] == "permanent")
        ]
        if len(target_records) != 4 or len(protected_records) != 12:
            raise RuntimeError("the frozen target/protected partition differs")
        target_offsets = torch.tensor(
            [
                record["preserve_minus_comply_baseline_log_odds"]
                for record in target_records
            ],
            dtype=torch.float64,
        )
        protected_offsets = torch.tensor(
            [
                record["preserve_minus_comply_baseline_log_odds"]
                for record in protected_records
            ],
            dtype=torch.float64,
        )
        for layer_index, layer in enumerate(LAYERS):
            scale = float(residual_scales[layer_index].item())
            target_rows = scale * torch.stack(
                [record["gradient"][layer_index].double() for record in target_records]
            )
            protected_rows = scale * torch.stack(
                [record["gradient"][layer_index].double() for record in protected_records]
            )
            unrelated_rows = scale * torch.stack(
                [record["gradient"][layer_index].double() for record in nuisance_records]
            )
            method_records = _screen_scenario_layer_rowspace(
                target_rows=target_rows.numpy(),
                target_offsets=target_offsets.numpy(),
                protected_rows=protected_rows.numpy(),
                protected_offsets=protected_offsets.numpy(),
                unrelated_rows=unrelated_rows.numpy(),
            )
            for method_record in method_records:
                public = {
                    **method_record,
                    "scenario_id": scenario_id,
                    "partition": "calibration",
                    "layer": layer,
                    "residual_scale": scale,
                    "target_rows_sha256": canonical_sha256(target_rows.tolist()),
                    "protected_rows_sha256": canonical_sha256(protected_rows.tolist()),
                    "unrelated_rows_sha256": canonical_sha256(unrelated_rows.tolist()),
                    "target_offsets_sha256": canonical_sha256(target_offsets.tolist()),
                    "protected_offsets_sha256": canonical_sha256(
                        protected_offsets.tolist()
                    ),
                    "numerical_amendment": "certified_rowspace_solver_v1",
                }
                public["screen_record_sha256"] = canonical_sha256(public)
                geometry_records.append(public)
    expected = {
        (layer, str(scenario["id"]), method)
        for layer in LAYERS
        for scenario in calibration_scenarios
        for method in METHODS
    }
    observed = {
        (int(record["layer"]), str(record["scenario_id"]), str(record["method"]))
        for record in geometry_records
    }
    if observed != expected or len(geometry_records) != len(expected):
        raise RuntimeError("the solver amendment did not compute the exact frozen grid")
    scenario_ids = [str(scenario["id"]) for scenario in calibration_scenarios]
    selection = select_layer(
        geometry_records,
        calibration_scenario_ids=scenario_ids,
        layers=LAYERS,
        cap_frontier=CAP_FRONTIER,
        qualification_cap=QUALIFICATION_CAP,
    )
    manifest = state["manifest"]
    result = _with_hash(
        {
            "schema_version": RESULT_SCHEMA,
            "status": selection["status"],
            "development_only": True,
            "outcome_aware_to_partial_calibration_geometry": True,
            "lock_file_sha256": file_sha256(LOCK_PATH),
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "preflight_sha256": preflight["preflight_sha256"],
            "original_lock_file_sha256": ORIGINAL_LOCK_FILE_SHA256,
            "original_lock_identity_sha256": ORIGINAL_LOCK_IDENTITY_SHA256,
            "capture_manifest_file_sha256": ORIGINAL_CAPTURE_FILE_SHA256,
            "capture_manifest_sha256": ORIGINAL_CAPTURE_IDENTITY_SHA256,
            "capture_plan_sha256": manifest["capture_plan_sha256"],
            "prompt_content_sha256": manifest["prompt_content_sha256"],
            "chunk_inventory_sha256": canonical_sha256(state["chunk_inventory"]),
            "failure_record_file_sha256": file_sha256(ORIGINAL_FAILURE_PATH),
            "failure_record_sha256": FAILURE_RECORD_IDENTITY_SHA256,
            "original_screen_result_absent": True,
            "selection_partition": "calibration_only",
            "pilot_scenario_geometry_computed": False,
            "pilot_construction_computed": False,
            "geometry_records": geometry_records,
            "geometry_record_count": len(geometry_records),
            "eligible_geometry_record_count": sum(
                record["status"] == "eligible" for record in geometry_records
            ),
            "selection": selection,
            "original_capture_compute": manifest["compute"],
            "amendment_capture_calls": 0,
            "amendment_model_loads": 0,
            "screen_model_forwards": 0,
            "screen_model_backwards": 0,
            "finite_intervention_forwards": 0,
            "finite_intervention_backwards": 0,
            "finite_intervention_outcomes_inspected": False,
            "generated_tokens": 0,
            "external_model_judges": 0,
            "external_api_calls": 0,
            "paid_model_cost_usd": 0,
            "local_linear_only": True,
            "claim_boundary": lock["claim_boundary"],
        },
        "result_sha256",
    )
    _write_new_json(SCREEN_RESULT_PATH, result)
    return _validate_screen_result()


def _norm_text(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _render_report(result: Mapping[str, Any]) -> str:
    selection = result["selection"]
    scenario_ids = list(selection["calibration_scenario_ids"])
    short_ids = [scenario_id.replace("fcag_dev_", "") for scenario_id in scenario_ids]
    lines = [
        "# Decision-Margin Shield solver-amendment layer screen",
        "",
        f"Status: **{selection['status']}**.",
        "",
        (
            "This model-free amendment recomputed the frozen calibration geometry with "
            "a certified row-space solver after the original numerical failure."
        ),
        "",
        (
            "Important: the amendment was designed after partial weather-scenario "
            "calibration geometry was viewed. No pilot geometry or finite intervention "
            "outcome was viewed or computed."
        ),
        "",
        "| Layer | "
        + " | ".join(short_ids)
        + " | Worst | Mean | Qualifies at L2 <= 2 |",
        "|---:|" + "---:|" * len(short_ids) + "---:|---:|:---:|",
    ]
    for summary in selection["layer_summaries"]:
        norms = summary["scenario_minimum_standardized_l2"]
        lines.append(
            f"| {summary['layer']} | "
            + " | ".join(_norm_text(norms[scenario_id]) for scenario_id in scenario_ids)
            + f" | {_norm_text(summary['worst_case_minimum_standardized_l2'])}"
            f" | {_norm_text(summary['mean_minimum_standardized_l2'])}"
            f" | {'yes' if summary['qualifies'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Selected layer: **{selection['selected_layer']}**.",
            "",
            (
                "A dash means at least one scenario had no certified uncapped DMS "
                "solution. Zero qualifying layers is a valid amended construction no-go."
            ),
            "",
            (
                "This used zero additional model forwards, backwards, generated tokens, "
                "external judges, APIs, or finite steering interventions."
            ),
            "",
            (
                "Nothing here proves a finite steering effect, nonlinear decision "
                "preservation, full-vocabulary stability, a natural self-preservation "
                "mechanism, unchanged capability, safety, priority, or publication novelty."
            ),
            "",
            f"Result SHA-256: `{result['result_sha256']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_report() -> str:
    result = _validate_screen_result()
    rendered = _render_report(result)
    if REPORT_PATH.exists():
        if REPORT_PATH.read_text(encoding="utf-8") != rendered:
            raise RuntimeError("the existing solver-amendment report differs")
    else:
        _atomic_text(REPORT_PATH, rendered)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model-free DMS row-space solver amendment"
    )
    parser.add_argument("command", choices=("lock", "preflight", "screen", "report"))
    arguments = parser.parse_args()
    if arguments.command == "lock":
        result: Any = run_lock()
    elif arguments.command == "preflight":
        result = run_preflight()
    elif arguments.command == "screen":
        result = run_screen()
    else:
        result = run_report()
    if isinstance(result, str):
        print(result, end="")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
