from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sp_lense.gcrbs_capture_adapter import (  # noqa: E402
    PROTECTED_GROUP_KEYS,
    adapt_v3_captures_to_gcrbs,
    float64_array_sha256,
)
from sp_lense.global_counterfactual_robust_boundary import (  # noqa: E402
    CANONICAL_GAMMA_RELAXATION,
    MAX_DETERMINISTIC_STARTS,
    PRIMAL_CERTIFICATE_TOLERANCE,
    SCHEMA_VERSION as SOLVER_SCHEMA_VERSION,
    SOLVER_FUNCTION_TOLERANCE,
    SOLVER_MAX_ITERATIONS,
    SOLVER_METHOD,
    solve_global_counterfactual_robust_boundary,
)

LOCK_SCHEMA = "sp_lense.gcrbs_development_lock.v1"
PREFLIGHT_SCHEMA = "sp_lense.gcrbs_preflight.v1"
SCREEN_SCHEMA = "sp_lense.gcrbs_layer10_screen.v1"
REQUIRED_MARGIN = 0.01
RESIDUAL_RELATIVE_L2_CAP = 0.10
AGGREGATE_FISHER_BUDGET = 0.005
PER_PROMPT_FISHER_CAP = 0.050
ZERO_BASED_LAYER = 10
EXTERNAL_MODEL_JUDGES = 0
EXTERNAL_API_CALLS = 0

LOCK_PATH = ROOT / "configs" / "global_counterfactual_robust_boundary_development_lock.json"
PROTOCOL_PATH = ROOT / "docs" / "GLOBAL_COUNTERFACTUAL_ROBUST_BOUNDARY_PROTOCOL.md"
SOLVER_PATH = ROOT / "src" / "sp_lense" / "global_counterfactual_robust_boundary.py"
ADAPTER_PATH = ROOT / "src" / "sp_lense" / "gcrbs_capture_adapter.py"
SCRIPT_PATH = Path(__file__).resolve()
V3_MATH_PATH = ROOT / "src" / "sp_lense" / "gradient_specificity_v3.py"
CAPTURE_ROOT = (
    ROOT
    / "artifacts"
    / "gradient_specificity_v3_development"
    / "score_identity_amendment_v1"
    / "qwen35_08b"
)
SP_CAPTURE_PATH = CAPTURE_ROOT / "stage_a" / "sp_capture.pt"
SP_MANIFEST_PATH = CAPTURE_ROOT / "stage_a" / "sp_capture_manifest.json"
NUISANCE_CAPTURE_PATH = CAPTURE_ROOT / "nuisance_capture.pt"
NUISANCE_MANIFEST_PATH = CAPTURE_ROOT / "nuisance_capture_manifest.json"
ARTIFACT_ROOT = ROOT / "artifacts" / "global_counterfactual_robust_boundary" / "qwen35_08b"
RESULT_ROOT = ROOT / "results" / "global_counterfactual_robust_boundary" / "qwen35_08b"
PREFLIGHT_PATH = ARTIFACT_ROOT / "preflight.json"
SCREEN_PATH = RESULT_ROOT / "layer10_offline_screen.json"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _atomic_json(path: Path, value: Mapping[str, Any], *, immutable: bool = True) -> None:
    rendered = json.dumps(dict(value), indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if immutable and path.is_file():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"immutable artifact differs: {_relative(path)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{_relative(path)} must contain a JSON object")
    return value


def _bound_paths() -> dict[str, Path]:
    return {
        "protocol": PROTOCOL_PATH,
        "solver": SOLVER_PATH,
        "adapter": ADAPTER_PATH,
        "runner": SCRIPT_PATH,
        "v3_math": V3_MATH_PATH,
        "sp_capture": SP_CAPTURE_PATH,
        "sp_capture_manifest": SP_MANIFEST_PATH,
        "nuisance_capture": NUISANCE_CAPTURE_PATH,
        "nuisance_capture_manifest": NUISANCE_MANIFEST_PATH,
    }


def _load_capture(torch: Any, capture_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError(f"{_relative(manifest_path)} is not complete")
    observed_hash = file_sha256(capture_path)
    if manifest.get("capture_file_sha256") != observed_hash:
        raise RuntimeError(f"{_relative(capture_path)} differs from its manifest hash")
    records_manifest = manifest.get("records")
    if not isinstance(records_manifest, list):
        raise RuntimeError(f"{_relative(manifest_path)} has no record manifest")
    if manifest.get("record_manifest_sha256") != canonical_sha256(records_manifest):
        raise RuntimeError(f"{_relative(manifest_path)} record manifest hash failed")
    payload = torch.load(capture_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise RuntimeError(f"{_relative(capture_path)} payload is incomplete")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != manifest.get("record_count"):
        raise RuntimeError(f"{_relative(capture_path)} record count differs")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping) or identity.get("identity_sha256") != manifest.get(
        "identity_sha256"
    ):
        raise RuntimeError(f"{_relative(capture_path)} identity differs from its manifest")
    return payload


def _constraints() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import torch

    sp_payload = _load_capture(torch, SP_CAPTURE_PATH, SP_MANIFEST_PATH)
    nuisance_payload = _load_capture(torch, NUISANCE_CAPTURE_PATH, NUISANCE_MANIFEST_PATH)
    constraints = adapt_v3_captures_to_gcrbs(
        torch,
        sp_records=sp_payload["records"],
        nuisance_records=nuisance_payload["records"],
        required_margin=REQUIRED_MARGIN,
    )
    observed_groups = tuple(group.group_key for group in constraints.fisher_surrogate_groups)
    if observed_groups != PROTECTED_GROUP_KEYS:
        raise RuntimeError(
            f"protected Fisher groups differ: expected {PROTECTED_GROUP_KEYS}, got {observed_groups}"
        )
    return constraints, sp_payload, nuisance_payload


def _factor_manifest(constraints: Any) -> dict[str, Any]:
    return {
        "aggregate": [
            {
                "group_key": group.group_key,
                "factor_shape": list(group.factor.shape),
                "factor_sha256": group.factor_sha256,
                "source_form_ids": list(group.source_form_ids),
            }
            for group in constraints.fisher_surrogate_groups
        ],
        "per_prompt": [
            {
                "group_key": group.group_key,
                "factor_shape": list(group.factor.shape),
                "factor_sha256": group.factor_sha256,
                "source_form_ids": list(group.source_form_ids),
            }
            for group in constraints.fisher_prompt_surrogate_groups
        ],
    }


def _lock_payload() -> dict[str, Any]:
    constraints, sp_payload, nuisance_payload = _constraints()
    value: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA,
        "status": "locked_before_first_candidate_solve",
        "development_only": True,
        "created_utc": datetime.now(UTC).isoformat(),
        "zero_based_layer": ZERO_BASED_LAYER,
        "model_identity": dict(sp_payload["identity"]),
        "nuisance_identity_sha256": nuisance_payload["identity"]["identity_sha256"],
        "source_and_capture_sha256": {
            key: file_sha256(path) for key, path in _bound_paths().items()
        },
        "construction": {
            "required_full_vocabulary_margin": REQUIRED_MARGIN,
            "residual_relative_l2_cap": RESIDUAL_RELATIVE_L2_CAP,
            "aggregate_fisher_budget_each": AGGREGATE_FISHER_BUDGET,
            "per_prompt_fisher_cap_each": PER_PROMPT_FISHER_CAP,
            "protected_group_keys": list(PROTECTED_GROUP_KEYS),
            "target_constraint_count": int(constraints.target_matrix.shape[0]),
            "protected_constraint_count": int(constraints.protected_matrix.shape[0]),
            "unrelated_equality_rank": int(constraints.unrelated_equality_basis.shape[0]),
            "capture_adapter_provenance_sha256": constraints.provenance[
                "provenance_sha256"
            ],
            "factor_manifest": _factor_manifest(constraints),
        },
        "solver": {
            "schema_version": SOLVER_SCHEMA_VERSION,
            "method": SOLVER_METHOD,
            "maximum_iterations": SOLVER_MAX_ITERATIONS,
            "function_tolerance": SOLVER_FUNCTION_TOLERANCE,
            "primal_certificate_tolerance": PRIMAL_CERTIFICATE_TOLERANCE,
            "canonical_gamma_relaxation": CANONICAL_GAMMA_RELAXATION,
            "maximum_deterministic_starts": MAX_DETERMINISTIC_STARTS,
        },
        "evaluation_policy": {
            "external_model_judges": EXTERNAL_MODEL_JUDGES,
            "external_api_calls": EXTERNAL_API_CALLS,
            "sealed_data_viewed": False,
            "j_space_postponed": True,
        },
    }
    value["lock_sha256"] = canonical_sha256(value)
    return value


def _validate_lock() -> dict[str, Any]:
    lock = _load_json(LOCK_PATH)
    stored_hash = lock.get("lock_sha256")
    unhashed = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if lock.get("schema_version") != LOCK_SCHEMA or stored_hash != canonical_sha256(unhashed):
        raise RuntimeError("GCRBS lock failed its internal hash")
    expected_files = lock.get("source_and_capture_sha256")
    if not isinstance(expected_files, Mapping):
        raise RuntimeError("GCRBS lock has no bound file hashes")
    for key, path in _bound_paths().items():
        if expected_files.get(key) != file_sha256(path):
            raise RuntimeError(f"bound file changed after lock: {key}")
    return lock


def run_lock() -> dict[str, Any]:
    proposed = _lock_payload()
    if LOCK_PATH.is_file():
        current = _validate_lock()
        excluded = {"created_utc", "lock_sha256"}
        proposed_stable = {key: value for key, value in proposed.items() if key not in excluded}
        current_stable = {key: value for key, value in current.items() if key not in excluded}
        if current_stable != proposed_stable:
            raise RuntimeError("existing GCRBS lock differs from the proposed protocol")
        return current
    _atomic_json(LOCK_PATH, proposed)
    return proposed


def run_preflight() -> dict[str, Any]:
    lock = _validate_lock()
    constraints, _sp, _nuisance = _constraints()
    construction = lock["construction"]
    observed = {
        "target_constraint_count": int(constraints.target_matrix.shape[0]),
        "protected_constraint_count": int(constraints.protected_matrix.shape[0]),
        "unrelated_equality_rank": int(constraints.unrelated_equality_basis.shape[0]),
        "capture_adapter_provenance_sha256": constraints.provenance["provenance_sha256"],
        "factor_manifest": _factor_manifest(constraints),
    }
    for key, value in observed.items():
        if construction.get(key) != value:
            raise RuntimeError(f"preflight geometry differs from lock: {key}")
    record = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "passed",
        "development_only": True,
        "lock_sha256": lock["lock_sha256"],
        "zero_based_layer": ZERO_BASED_LAYER,
        "geometry": observed,
        "external_model_judges": EXTERNAL_MODEL_JUDGES,
        "external_api_calls": EXTERNAL_API_CALLS,
        "model_loaded": False,
        "forward_evaluations": 0,
        "backward_evaluations": 0,
    }
    record["preflight_sha256"] = canonical_sha256(record)
    _atomic_json(PREFLIGHT_PATH, record)
    return record


def _solver_budget_schedule(constraints: Any) -> tuple[tuple[str, ...], tuple[float, ...]]:
    labels = constraints.group_metric_labels
    aggregate_count = len(constraints.fisher_surrogate_groups)
    budgets = tuple(
        AGGREGATE_FISHER_BUDGET if index < aggregate_count else PER_PROMPT_FISHER_CAP
        for index in range(len(labels))
    )
    return labels, budgets


def run_layer10_screen() -> dict[str, Any]:
    lock = _validate_lock()
    preflight = _load_json(PREFLIGHT_PATH)
    if (
        preflight.get("status") != "passed"
        or preflight.get("lock_sha256") != lock["lock_sha256"]
    ):
        raise RuntimeError("a passing lock-bound preflight is required")
    constraints, _sp, _nuisance = _constraints()
    labels, budgets = _solver_budget_schedule(constraints)
    solution = solve_global_counterfactual_robust_boundary(
        **constraints.solver_kwargs(group_metric_budgets=budgets),
        l2_cap=RESIDUAL_RELATIVE_L2_CAP,
    )
    record = {
        "schema_version": SCREEN_SCHEMA,
        "status": "certified_affine_candidate",
        "development_only": True,
        "lock_sha256": lock["lock_sha256"],
        "preflight_sha256": preflight["preflight_sha256"],
        "zero_based_layer": ZERO_BASED_LAYER,
        "required_margin": REQUIRED_MARGIN,
        "eligible_for_finite_discovery_oracle": solution.gamma >= REQUIRED_MARGIN,
        "gamma": solution.gamma,
        "direction": solution.direction.tolist(),
        "direction_float64_sha256": float64_array_sha256(solution.direction),
        "group_metric_labels": list(labels),
        "group_metric_budgets": list(budgets),
        "solver_diagnostics": solution.diagnostics,
        "model_loaded": False,
        "forward_evaluations": 0,
        "backward_evaluations": 0,
    }
    record["screen_sha256"] = canonical_sha256(record)
    _atomic_json(SCREEN_PATH, record)
    return record


def _summary(command: str, value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "lock_sha256",
        "preflight_sha256",
        "screen_sha256",
        "gamma",
        "eligible_for_finite_discovery_oracle",
        "forward_evaluations",
        "backward_evaluations",
    )
    return {"command": command, **{key: value[key] for key in keys if key in value}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Locked model-free GCRBS development stages")
    parser.add_argument("command", choices=("lock", "preflight", "layer10-screen"))
    arguments = parser.parse_args(argv)
    if arguments.command == "lock":
        value = run_lock()
    elif arguments.command == "preflight":
        value = run_preflight()
    else:
        value = run_layer10_screen()
    print(json.dumps(_summary(arguments.command, value), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
