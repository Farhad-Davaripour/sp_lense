"""Exact, resumable forced-validation grid execution for the steering comparison.

The grid runner is intentionally separate from sealed evaluation.  It accepts only the
locked validation prompt manifest, loads at most one model backend per invocation, caches
only unsteered next-token logits keyed by model and prompt content, and writes one
self-validating atomic shard per calibration point.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .comparison_analysis import (
    ROW_SCHEMA_VERSION,
    canonical_json_sha256,
    validate_result_rows,
    validate_sha256,
)
from .comparison_calibration import (
    calibration_rows_sha256,
    calibration_unit_key,
    locked_forced_calibration_units,
    validate_calibration_coverage,
)
from .comparison_dataset import (
    comparison_dataset_sha256,
    render_choice_case,
    render_sp_case,
)
from .comparison_evaluate import MethodSetup, prompt_sha256
from .comparison_fit import read_direction_artifact
from .comparison_provenance import (
    MAIN_CONSTRUCTION_SCHEMA_VERSION,
    locked_method_construction_configuration,
    locked_position_schedule,
    sha256_file,
    sha256_json,
)
from .comparison_runtime import (
    choice_score_from_logits,
    next_token_logits,
    next_token_logits_with_perturbation,
    resolve_choice_boundary,
    validate_locked_choice_runtime,
)

GRID_PLAN_SCHEMA_VERSION = "sp_lense.forced_calibration_grid.plan.v1"
GRID_SHARD_SCHEMA_VERSION = "sp_lense.forced_calibration_grid.shard.v1"
BASELINE_CACHE_SCHEMA_VERSION = "sp_lense.forced_calibration_grid.baseline.v1"
ZERO_SHA256 = "0" * 64
EXPECTED_POINTS_PER_MODEL = 250
EXPECTED_UNITS_PER_POINT = 142
EXPECTED_ROWS_PER_POINT = 426
EXPECTED_POINT_COUNT_BY_COHORT = {
    "matched": 30,
    "caa_canonical": 96,
    "bipo_canonical": 4,
    "persona_vector_canonical": 120,
}
MATCHED_METHODS = (
    "gradient",
    "gradient_uncorrected",
    "caa",
    "bipo",
    "persona_vector",
)


@dataclass(frozen=True, order=True)
class GridPointSpec:
    method_id: str
    track: str
    layer: int
    strength: float

    def to_record(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "track": self.track,
            "layer": self.layer,
            "strength": self.strength,
        }


@dataclass(frozen=True)
class CalibrationPromptUnit:
    family: str
    case_id: str
    target: str
    form: str
    prompt: str
    first_semantic_label: str
    second_semantic_label: str
    extra_fields: Mapping[str, Any]

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.family, self.case_id, self.target, self.form)


def _finite_positive_values(values: Any, field: str) -> tuple[float, ...]:
    if (
        not isinstance(values, list)
        or not values
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in values
        )
    ):
        raise ValueError(f"{field} must be a non-empty list of positive finite numbers")
    output = tuple(float(value) for value in values)
    if len(set(output)) != len(output):
        raise ValueError(f"{field} contains duplicate values")
    return output


def _locked_model(lock: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    matches = [model for model in lock.get("models", []) if model.get("model_id") == model_id]
    if len(matches) != 1:
        raise ValueError(f"model {model_id!r} must appear exactly once in the lock")
    return matches[0]


def derive_forced_grid_specs(lock: Mapping[str, Any], model_id: str) -> tuple[GridPointSpec, ...]:
    """Derive the exact preregistered 250-point forced grid for one locked model."""

    model = _locked_model(lock, model_id)
    architecture = model.get("architecture")
    if not isinstance(architecture, Mapping):
        raise TypeError("locked model lacks architecture")
    blocks = architecture.get("blocks")
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks != 24:
        raise ValueError("the locked comparison requires exactly 24 transformer blocks")
    matched = model.get("matched_intervention")
    if not isinstance(matched, Mapping):
        raise TypeError("locked model lacks matched intervention settings")
    matched_layer = matched.get("layer_zero_based")
    if isinstance(matched_layer, bool) or not isinstance(matched_layer, int):
        raise TypeError("matched intervention layer must be an integer")
    if matched_layer != 10:
        raise ValueError("the locked comparison requires matched block 10")
    calibration = lock.get("calibration")
    if not isinstance(calibration, Mapping):
        raise TypeError("lock lacks calibration settings")
    matched_strengths = _finite_positive_values(
        calibration.get("matched_strength_grid"), "matched_strength_grid"
    )
    canonical = calibration.get("canonical_multiplier_grids")
    if not isinstance(canonical, Mapping):
        raise TypeError("lock lacks canonical multiplier grids")
    caa_strengths = _finite_positive_values(canonical.get("caa"), "canonical CAA grid")
    bipo_strengths = _finite_positive_values(canonical.get("bipo"), "canonical BiPO grid")
    persona_strengths = _finite_positive_values(
        canonical.get("persona_vector"), "canonical persona grid"
    )
    bipo_layer = lock.get("methods", {}).get("bipo", {}).get("canonical_layer_zero_based")
    if isinstance(bipo_layer, bool) or not isinstance(bipo_layer, int):
        raise TypeError("locked canonical BiPO layer must be an integer")

    specs: list[GridPointSpec] = []
    for method_id in MATCHED_METHODS:
        specs.extend(
            GridPointSpec(method_id, "matched", matched_layer, strength)
            for strength in matched_strengths
        )
    specs.extend(
        GridPointSpec("caa", "canonical", layer, strength)
        for layer in range(blocks)
        for strength in caa_strengths
    )
    specs.extend(
        GridPointSpec("bipo", "canonical", bipo_layer, strength) for strength in bipo_strengths
    )
    specs.extend(
        GridPointSpec("persona_vector", "canonical", layer, strength)
        for layer in range(blocks)
        for strength in persona_strengths
    )
    expected_counts = {
        ("matched", "all"): 30,
        ("canonical", "caa"): 96,
        ("canonical", "bipo"): 4,
        ("canonical", "persona_vector"): 120,
    }
    observed_counts = {
        ("matched", "all"): sum(spec.track == "matched" for spec in specs),
        ("canonical", "caa"): sum(
            spec.track == "canonical" and spec.method_id == "caa" for spec in specs
        ),
        ("canonical", "bipo"): sum(
            spec.track == "canonical" and spec.method_id == "bipo" for spec in specs
        ),
        ("canonical", "persona_vector"): sum(
            spec.track == "canonical" and spec.method_id == "persona_vector" for spec in specs
        ),
    }
    if observed_counts != expected_counts or len(specs) != EXPECTED_POINTS_PER_MODEL:
        raise RuntimeError(
            f"locked grid does not resolve to the exact 250 points: {observed_counts}"
        )
    if len(set(specs)) != len(specs):
        raise RuntimeError("locked grid contains duplicate points")
    return tuple(specs)


def _resolve_repo_path(repo_root: Path, manifest_path: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [repo_root / raw, manifest_path.parent / raw]
    existing = [candidate.resolve() for candidate in candidates if candidate.resolve().is_file()]
    if len(set(existing)) != 1:
        raise FileNotFoundError(f"{field} must resolve to exactly one existing file: {value}")
    resolved = existing[0]
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes the repository root") from exc
    return resolved


def _repo_relative_artifact_path(repo_root: Path, path: Path, field: str) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{field} escapes the repository root") from exc


def _resolve_planned_artifact_path(
    repo_root: str | Path, value: Any, field: str
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    raw = Path(value)
    if raw.is_absolute():
        raise ValueError(f"{field} must be repository-relative, not machine-local")
    root = Path(repo_root).resolve()
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes the repository root") from exc
    return resolved


def _read_json_object(path: Path, field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{field} must contain a JSON object")
    return payload


def _float32_bytes(values: Any) -> tuple[bytes, tuple[float, ...]]:
    if not isinstance(values, list) or not values:
        raise ValueError("direction must be a non-empty JSON list")
    converted = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("direction entries must be numeric")
        packed = struct.pack("<f", float(value))
        rounded = struct.unpack("<f", packed)[0]
        if not math.isfinite(rounded):
            raise ValueError("direction entries must be finite float32 values")
        converted.append(rounded)
    return b"".join(struct.pack("<f", value) for value in converted), tuple(converted)


def _validate_direction_record(path: Path) -> dict[str, Any]:
    record = _read_json_object(path, "direction artifact")
    required = {
        "schema_version",
        "method",
        "layer",
        "intervention_geometry",
        "d_model",
        "dtype",
        "direction_l2_norm",
        "direction_sha256",
        "metadata",
        "metadata_sha256",
        "artifact_sha256",
        "direction",
    }
    if set(record) != required:
        raise ValueError("direction artifact fields differ from the v1 schema")
    if record["schema_version"] != "sp_lense.direction.v1" or record["dtype"] != "float32":
        raise ValueError("direction artifact has an unsupported schema/dtype")
    vector_bytes, values = _float32_bytes(record["direction"])
    if record["d_model"] != len(values):
        raise ValueError("direction d_model differs from its vector width")
    direction_hash = hashlib.sha256(vector_bytes).hexdigest()
    if validate_sha256(record["direction_sha256"], "direction_sha256") != direction_hash:
        raise ValueError("direction artifact has an invalid float32 hash")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(float(record["direction_l2_norm"]), norm, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("direction artifact has an invalid L2 norm")
    metadata_payload = {
        key: record[key]
        for key in (
            "schema_version",
            "method",
            "layer",
            "intervention_geometry",
            "d_model",
            "dtype",
            "direction_l2_norm",
            "direction_sha256",
            "metadata",
        )
    }
    metadata_bytes = json.dumps(
        metadata_payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if (
        validate_sha256(record["metadata_sha256"], "metadata_sha256")
        != hashlib.sha256(metadata_bytes).hexdigest()
    ):
        raise ValueError("direction artifact has an invalid metadata hash")
    if (
        validate_sha256(record["artifact_sha256"], "artifact_sha256")
        != hashlib.sha256(metadata_bytes + b"\0" + vector_bytes).hexdigest()
    ):
        raise ValueError("direction artifact has an invalid artifact hash")
    return record


def _expected_geometry(method_id: str, track: str) -> str:
    if track == "matched" or method_id.startswith("gradient"):
        return "matched_final_prompt"
    return {
        "caa": "caa_post_prompt",
        "bipo": "canonical_broadcast",
        "persona_vector": "persona_response",
    }[method_id]


def _expected_direction_keys(specs: Sequence[GridPointSpec]) -> set[tuple[str, str, int]]:
    return {(spec.method_id, spec.track, spec.layer) for spec in specs}


def resolve_forced_grid_plan(
    *,
    repo_root: str | Path,
    lock: Mapping[str, Any],
    stage1_lock_sha256: str,
    model_id: str,
    direction_manifest_paths: Sequence[str | Path],
    runner_commit: str,
) -> dict[str, Any]:
    """Resolve and hash every direction/construction identity for the exact grid."""

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("repository root does not exist")
    stage1_hash = validate_sha256(stage1_lock_sha256, "stage1_lock_sha256")
    if (
        not isinstance(runner_commit, str)
        or len(runner_commit) != 40
        or any(character not in "0123456789abcdef" for character in runner_commit.lower())
    ):
        raise ValueError("runner_commit must be a 40-character hexadecimal commit")
    model = _locked_model(lock, model_id)
    specs = derive_forced_grid_specs(lock, model_id)
    expected_keys = _expected_direction_keys(specs)
    if not direction_manifest_paths:
        raise ValueError("at least one direction manifest is required")
    manifests = []
    resolved_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for raw_manifest_path in direction_manifest_paths:
        manifest_path = Path(raw_manifest_path).resolve()
        try:
            manifest_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("direction manifest escapes the repository root") from exc
        manifest = _read_json_object(manifest_path, "direction manifest")
        if set(manifest) != {"directions"} or not isinstance(manifest["directions"], list):
            raise ValueError("direction manifest must contain exactly a directions list")
        manifests.append(
            {
                "path": _repo_relative_artifact_path(
                    root, manifest_path, "direction manifest"
                ),
                "sha256": sha256_file(manifest_path),
            }
        )
        for index, item in enumerate(manifest["directions"]):
            if not isinstance(item, Mapping):
                raise TypeError(f"direction manifest entry {index} must be an object")
            direction_path = _resolve_repo_path(
                root, manifest_path, item.get("path"), "direction path"
            )
            direction = _validate_direction_record(direction_path)
            metadata = direction.get("metadata")
            if not isinstance(metadata, Mapping):
                raise TypeError("direction artifact metadata must be an object")
            artifact_model = metadata.get("model_id")
            if artifact_model != model_id:
                # A manifest may contain the other locked model; it is never used
                # implicitly for this plan.
                continue
            method_id = str(direction["method"])
            track = str(item.get("track", metadata.get("track", "")))
            layer = direction["layer"]
            if isinstance(layer, bool) or not isinstance(layer, int):
                raise TypeError("direction layer must be an integer")
            key = (method_id, track, layer)
            if key not in expected_keys:
                raise ValueError(f"unexpected direction identity for locked grid: {key}")
            if key in resolved_by_key:
                raise ValueError(f"duplicate direction identity across manifests: {key}")
            expected_metadata = {
                "model_id": model_id,
                "model_revision": model["revision"],
                "model_config_sha256": model["config_sha256"],
                "dataset_sha256": lock["dataset"]["sha256"],
                "protocol_sha256": lock["protocol"]["sha256"],
                "stage1_lock_sha256": stage1_hash,
                "runner_commit": runner_commit,
                "track": track,
            }
            mismatches = {
                field: (expected, metadata.get(field))
                for field, expected in expected_metadata.items()
                if metadata.get(field) != expected
            }
            if mismatches:
                raise ValueError(f"direction metadata differs from the lock: {mismatches}")
            if direction["d_model"] != model["architecture"]["residual_width"]:
                raise ValueError("direction width differs from the locked model")
            geometry = _expected_geometry(method_id, track)
            if direction["intervention_geometry"] != geometry:
                raise ValueError("direction intervention geometry differs from the lock")
            if track == "matched" and not math.isclose(
                float(direction["direction_l2_norm"]), 1.0, rel_tol=0, abs_tol=1e-5
            ):
                raise ValueError("matched direction must be unit-normalized")
            for manifest_field, direction_field in (
                ("method_id", "method"),
                ("layer", "layer"),
                ("intervention_geometry", "intervention_geometry"),
                ("direction_float32_sha256", "direction_sha256"),
                ("direction_artifact_sha256", "artifact_sha256"),
            ):
                if item.get(manifest_field) != direction[direction_field]:
                    raise ValueError(
                        f"direction manifest {manifest_field} differs from its artifact"
                    )
            direction_file_hash = sha256_file(direction_path)
            if item.get("direction_file_sha256") not in {None, direction_file_hash}:
                raise ValueError("direction manifest has an invalid direction file hash")
            construction_path = _resolve_repo_path(
                root,
                manifest_path,
                item.get("construction_config_path"),
                "construction config path",
            )
            construction_hash = sha256_file(construction_path)
            if item.get("construction_config_sha256") != construction_hash:
                raise ValueError("construction config hash differs from the manifest")
            construction = _read_json_object(construction_path, "construction config")
            expected_construction = {
                "schema_version": MAIN_CONSTRUCTION_SCHEMA_VERSION,
                "model_id": model_id,
                "model_revision": model["revision"],
                "model_config_sha256": model["config_sha256"],
                "method_id": method_id,
                "track": track,
                "selected_layer": layer,
                "position_schedule": locked_position_schedule(method_id, track),
                "intervention_geometry": geometry,
                "direction_float32_sha256": direction["direction_sha256"],
                "direction_artifact_sha256": direction["artifact_sha256"],
                "dataset_sha256": lock["dataset"]["sha256"],
                "protocol_sha256": lock["protocol"]["sha256"],
                "stage1_lock_sha256": stage1_hash,
                "runner_commit": runner_commit,
            }
            construction_mismatches = {
                field: (expected, construction.get(field))
                for field, expected in expected_construction.items()
                if construction.get(field) != expected
            }
            if construction_mismatches:
                raise ValueError(
                    f"construction config differs from the lock: {construction_mismatches}"
                )
            locked_configuration = locked_method_construction_configuration(lock, method_id, track)
            if construction.get("locked_configuration") != locked_configuration or construction.get(
                "locked_configuration_sha256"
            ) != sha256_json(locked_configuration):
                raise ValueError("construction config has invalid locked configuration")
            evidence = construction.get("evidence_artifacts")
            if (
                not isinstance(evidence, list)
                or not evidence
                or construction.get("evidence_artifacts_sha256") != sha256_json(evidence)
            ):
                raise ValueError("construction config has an invalid evidence manifest")
            required_roles = {
                "gradient": {"gradient_construction_diagnostics"},
                "gradient_uncorrected": {"gradient_construction_diagnostics"},
                "caa": {"caa_construction_diagnostics"},
                "bipo": {"bipo_training_audit"},
                "persona_vector": {
                    "persona_construction_diagnostics",
                    "persona_scored_rollouts",
                },
            }[method_id]
            observed_roles = set()
            observed_evidence_paths = set()
            for evidence_item in evidence:
                if not isinstance(evidence_item, Mapping):
                    raise TypeError("construction evidence entries must be objects")
                role = evidence_item.get("role")
                if not isinstance(role, str) or not role:
                    raise ValueError("construction evidence role is invalid")
                evidence_path = _resolve_repo_path(
                    root,
                    construction_path,
                    evidence_item.get("path"),
                    "construction evidence path",
                )
                if evidence_item.get("sha256") != sha256_file(evidence_path):
                    raise ValueError("construction evidence file hash is invalid")
                if role in observed_roles or evidence_path in observed_evidence_paths:
                    raise ValueError("construction config duplicates an evidence role/path")
                observed_roles.add(role)
                observed_evidence_paths.add(evidence_path)
            if observed_roles != required_roles:
                raise ValueError("construction evidence roles differ from the locked method")
            resolved_by_key[key] = {
                "method_id": method_id,
                "track": track,
                "layer": layer,
                "position_schedule": locked_position_schedule(method_id, track),
                "intervention_geometry": geometry,
                "direction_path": _repo_relative_artifact_path(
                    root, direction_path, "direction artifact"
                ),
                "direction_file_sha256": direction_file_hash,
                "direction_float32_sha256": direction["direction_sha256"],
                "direction_artifact_sha256": direction["artifact_sha256"],
                "construction_config_path": _repo_relative_artifact_path(
                    root, construction_path, "construction config"
                ),
                "construction_config_sha256": construction_hash,
            }
    if set(resolved_by_key) != expected_keys:
        raise ValueError(
            "direction manifests do not exactly cover the locked grid artifacts: "
            f"missing={sorted(expected_keys - set(resolved_by_key))}, "
            f"extra={sorted(set(resolved_by_key) - expected_keys)}"
        )

    points = []
    for grid_index, spec in enumerate(specs):
        direction = resolved_by_key[(spec.method_id, spec.track, spec.layer)]
        core = {
            "grid_index": grid_index,
            **spec.to_record(),
            **{
                field: direction[field]
                for field in (
                    "position_schedule",
                    "intervention_geometry",
                    "direction_path",
                    "direction_file_sha256",
                    "direction_float32_sha256",
                    "direction_artifact_sha256",
                    "construction_config_path",
                    "construction_config_sha256",
                )
            },
        }
        point_hash = canonical_json_sha256(core)
        points.append(
            {
                **core,
                "point_sha256": point_hash,
                "shard_name": f"point_{grid_index:03d}_{point_hash[:16]}.json",
            }
        )
    forced_units = int(lock["calibration"]["staged_open_confirmation"]["forced_grid_unit_count"])
    forced_rows = int(
        lock["calibration"]["staged_open_confirmation"]["forced_grid_row_count_per_point"]
    )
    if forced_units != EXPECTED_UNITS_PER_POINT or forced_rows != EXPECTED_ROWS_PER_POINT:
        raise ValueError("lock forced-grid point coverage differs from 142 units/426 rows")
    payload = {
        "schema_version": GRID_PLAN_SCHEMA_VERSION,
        "model_id": model_id,
        "model_revision": model["revision"],
        "model_config_sha256": model["config_sha256"],
        "dataset_sha256": lock["dataset"]["sha256"],
        "protocol_sha256": lock["protocol"]["sha256"],
        "stage1_lock_sha256": stage1_hash,
        "stage1_lock_payload_sha256": canonical_json_sha256(lock),
        "runner_commit": runner_commit,
        "run_seed": int(lock["statistics"]["bootstrap"]["seed"]),
        "expected_point_count": EXPECTED_POINTS_PER_MODEL,
        "expected_unit_count_per_point": forced_units,
        "expected_row_count_per_point": forced_rows,
        "point_count_by_cohort": dict(EXPECTED_POINT_COUNT_BY_COHORT),
        "direction_manifests": sorted(manifests, key=lambda item: item["path"]),
        "points": points,
    }
    return {**payload, "plan_sha256": canonical_json_sha256(payload)}


def _cases_by_id(cases: Any, field: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(cases, list) or any(not isinstance(case, Mapping) for case in cases):
        raise TypeError(f"{field} must be a list of case objects")
    output = {str(case["id"]): case for case in cases}
    if len(output) != len(cases):
        raise ValueError(f"{field} contains duplicate case IDs")
    return output


def _locked_partition_ids(lock: Mapping[str, Any], family: str, split: str) -> tuple[str, ...]:
    value = lock.get("dataset", {}).get("partitions", {}).get(family, {}).get(split)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"lock partition {family}.{split} must be a list of IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"lock partition {family}.{split} contains duplicate IDs")
    return tuple(value)


def build_forced_prompt_units(
    dataset: Mapping[str, Any], lock: Mapping[str, Any]
) -> tuple[CalibrationPromptUnit, ...]:
    """Project the full dataset down to only the exact 142 validation prompts."""

    if comparison_dataset_sha256(dict(dataset)) != lock["dataset"]["sha256"]:
        raise ValueError("comparison dataset content differs from the stage-1 lock")
    units: list[CalibrationPromptUnit] = []
    validation_core = dataset.get("sp_splits", {}).get("validation")
    if not isinstance(validation_core, list):
        raise TypeError("dataset lacks the validation self-preservation split")
    validation_core_ids = {str(case["id"]) for case in validation_core}
    sealed_core_ids = {
        str(case["id"]) for case in dataset.get("sp_splits", {}).get("sealed_test", [])
    }
    if validation_core_ids & sealed_core_ids:
        raise ValueError("validation and sealed self-preservation IDs overlap")
    for case in validation_core:
        for target in ("self", "other"):
            rendered = render_sp_case(dict(case), target)
            units.append(
                CalibrationPromptUnit(
                    family="self_preservation",
                    case_id=str(case["id"]),
                    target=target,
                    form="",
                    prompt=rendered["prompt"],
                    first_semantic_label=rendered["preserve_label"],
                    second_semantic_label=rendered["comply_label"],
                    extra_fields={
                        "target": target,
                        "preserve_label": rendered["preserve_label"],
                        "comply_label": rendered["comply_label"],
                        "domain": case["domain"],
                        "authorized": case["authorized"],
                        "event_type": case["event_type"],
                        "motivation": case["motivation"],
                        "distribution": case["distribution"],
                        "adversarial": case["adversarial"],
                        "preserve_first": case["preserve_first"],
                    },
                )
            )

    collateral = dataset.get("collateral_cases")
    if not isinstance(collateral, Mapping):
        raise TypeError("dataset lacks collateral case families")
    for family in ("benign_compliance", "general_capability", "refusal"):
        validation_ids = _locked_partition_ids(lock, family, "validation_ids")
        sealed_ids = set(_locked_partition_ids(lock, family, "sealed_ids"))
        if set(validation_ids) & sealed_ids:
            raise ValueError(f"{family} validation and sealed IDs overlap")
        by_id = _cases_by_id(collateral.get(family), f"collateral_cases.{family}")
        missing = set(validation_ids) - set(by_id)
        if missing:
            raise ValueError(f"{family} validation IDs are missing: {sorted(missing)}")
        for case_id in validation_ids:
            case = by_id[case_id]
            rendered = render_choice_case(dict(case))
            units.append(
                CalibrationPromptUnit(
                    family=family,
                    case_id=case_id,
                    target="",
                    form="",
                    prompt=rendered["prompt"],
                    first_semantic_label=rendered["preferred_label"],
                    second_semantic_label=rendered["alternative_label"],
                    extra_fields={
                        "suite": case["suite"],
                        "category": case["category"],
                        "correct_label": rendered["preferred_label"],
                        "preferred_label": rendered["preferred_label"],
                        **{
                            key: case[key]
                            for key in ("request_type", "expected_behavior")
                            if key in case
                        },
                    },
                )
            )

    option_family = "option_order_sentinels"
    option_validation_ids = _locked_partition_ids(lock, option_family, "validation_ids")
    option_sealed_ids = set(_locked_partition_ids(lock, option_family, "sealed_ids"))
    if set(option_validation_ids) & option_sealed_ids:
        raise ValueError("option-order validation and sealed IDs overlap")
    option_by_id = _cases_by_id(collateral.get(option_family), f"collateral_cases.{option_family}")
    missing_options = set(option_validation_ids) - set(option_by_id)
    if missing_options:
        raise ValueError(f"option-order validation IDs are missing: {sorted(missing_options)}")
    for case_id in option_validation_ids:
        case = option_by_id[case_id]
        for form, preferred_first in (
            ("preferred_first", True),
            ("preferred_second", False),
        ):
            rendered = render_choice_case(dict(case), preferred_first=preferred_first)
            units.append(
                CalibrationPromptUnit(
                    family="option_order_sentinel",
                    case_id=case_id,
                    target="",
                    form=form,
                    prompt=rendered["prompt"],
                    first_semantic_label=rendered["preferred_label"],
                    second_semantic_label=rendered["alternative_label"],
                    extra_fields={
                        "suite": case["suite"],
                        "category": case["category"],
                        "form": form,
                        "correct_label": rendered["preferred_label"],
                        "preferred_label": rendered["preferred_label"],
                    },
                )
            )
    observed = {unit.key for unit in units}
    expected = locked_forced_calibration_units(dataset, lock)
    if len(units) != len(observed) or observed != expected:
        raise RuntimeError(
            "rendered forced prompts do not exactly match the locked 142-unit manifest"
        )
    if len(units) != EXPECTED_UNITS_PER_POINT:
        raise RuntimeError("forced prompt projection must contain exactly 142 units")
    return tuple(units)


def baseline_cache_key_payload(
    *,
    model_id: str,
    model_revision: str,
    model_config_sha256: str,
    prompt_sha256_value: str,
) -> dict[str, str]:
    """Candidate-independent cache identity for one model/prompt baseline."""

    return {
        "schema_version": BASELINE_CACHE_SCHEMA_VERSION,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_config_sha256": validate_sha256(model_config_sha256, "model_config_sha256"),
        "prompt_sha256": validate_sha256(prompt_sha256_value, "prompt_sha256"),
    }


def baseline_cache_key_sha256(**kwargs: str) -> str:
    return canonical_json_sha256(baseline_cache_key_payload(**kwargs))


def baseline_logits_float32_sha256(logits: Any) -> str:
    """Hash canonical contiguous float32 logit shape and bytes.

    The cache identity alone cannot prove that two baseline forwards produced the
    same numbers.  This second digest binds the actual float32 vector without
    persisting the large vocabulary-sized tensor in every point shard.
    """

    if hasattr(logits, "detach"):
        value = logits.detach().float().cpu().contiguous()
        shape = [int(item) for item in value.shape]
        array = value.numpy()
        if sys.byteorder != "little":  # pragma: no cover - supported for portability
            array = array.byteswap()
        raw = array.tobytes(order="C")
    elif isinstance(logits, Sequence) and not isinstance(logits, (str, bytes)):
        values = list(logits)
        if any(isinstance(item, (Sequence, Mapping)) for item in values):
            raise TypeError("non-tensor baseline logits must be one-dimensional")
        shape = [len(values)]
        raw = b"".join(struct.pack("<f", float(item)) for item in values)
    else:
        raise TypeError("baseline logits must be a tensor or numeric sequence")
    header = {
        "schema_version": "sp_lense.baseline_logits_float32.v1",
        "dtype": "float32_le",
        "shape": shape,
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")
    digest.update(raw)
    return digest.hexdigest()


class BaselineLogitsCache:
    """In-memory cache containing only unsteered logits, never candidate rows/scores."""

    def __init__(self, plan: Mapping[str, Any]) -> None:
        self.model_id = str(plan["model_id"])
        self.model_revision = str(plan["model_revision"])
        self.model_config_sha256 = validate_sha256(
            plan["model_config_sha256"], "model_config_sha256"
        )
        self._logits: dict[str, Any] = {}
        self._prompt_hashes: dict[str, str] = {}
        self._logit_hashes: dict[str, str] = {}
        self.computations = 0

    def key_for_prompt(self, prompt: str) -> str:
        return baseline_cache_key_sha256(
            model_id=self.model_id,
            model_revision=self.model_revision,
            model_config_sha256=self.model_config_sha256,
            prompt_sha256_value=prompt_sha256(prompt),
        )

    def get_or_compute(self, prompt: str, compute: Callable[[], Any]) -> tuple[Any, str, str]:
        prompt_hash = prompt_sha256(prompt)
        key = self.key_for_prompt(prompt)
        prior_prompt_hash = self._prompt_hashes.get(key)
        if prior_prompt_hash is not None and prior_prompt_hash != prompt_hash:
            raise RuntimeError("baseline cache key collision")
        if key not in self._logits:
            logits = compute()
            if hasattr(logits, "detach"):
                logits = logits.detach().float().cpu()
            self._logits[key] = logits
            self._prompt_hashes[key] = prompt_hash
            self._logit_hashes[key] = baseline_logits_float32_sha256(logits)
            self.computations += 1
        return self._logits[key], key, self._logit_hashes[key]

    def __len__(self) -> int:
        return len(self._logits)


def _validate_plan(
    plan: Mapping[str, Any], lock: Mapping[str, Any], *, repo_root: str | Path
) -> list[dict[str, Any]]:
    if plan.get("schema_version") != GRID_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported forced-grid plan schema")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if validate_sha256(plan.get("plan_sha256"), "plan_sha256") != canonical_json_sha256(payload):
        raise ValueError("forced-grid plan content hash is invalid")
    if plan.get("stage1_lock_payload_sha256") != canonical_json_sha256(lock):
        raise ValueError("forced-grid plan is bound to a different stage-1 lock payload")
    model_id = str(plan.get("model_id"))
    model = _locked_model(lock, model_id)
    expected_identity = {
        "model_revision": model["revision"],
        "model_config_sha256": model["config_sha256"],
        "dataset_sha256": lock["dataset"]["sha256"],
        "protocol_sha256": lock["protocol"]["sha256"],
        "expected_point_count": EXPECTED_POINTS_PER_MODEL,
        "expected_unit_count_per_point": EXPECTED_UNITS_PER_POINT,
        "expected_row_count_per_point": EXPECTED_ROWS_PER_POINT,
    }
    mismatches = {
        field: (expected, plan.get(field))
        for field, expected in expected_identity.items()
        if plan.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"forced-grid plan differs from the lock: {mismatches}")
    validate_sha256(plan.get("stage1_lock_sha256"), "stage1_lock_sha256")
    if plan.get("run_seed") != int(lock["statistics"]["bootstrap"]["seed"]):
        raise ValueError("forced-grid plan run seed differs from the lock")
    if plan.get("point_count_by_cohort") != EXPECTED_POINT_COUNT_BY_COHORT:
        raise ValueError("forced-grid plan cohort counts differ from the locked grid")
    runner_commit = plan.get("runner_commit")
    if (
        not isinstance(runner_commit, str)
        or len(runner_commit) != 40
        or any(character not in "0123456789abcdef" for character in runner_commit.lower())
    ):
        raise ValueError("forced-grid plan runner commit is invalid")
    manifests = plan.get("direction_manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("forced-grid plan must identify at least one direction manifest")
    seen_manifest_paths = set()
    for manifest in manifests:
        if not isinstance(manifest, Mapping) or set(manifest) != {"path", "sha256"}:
            raise ValueError("forced-grid plan direction manifest record is invalid")
        manifest_relative = str(manifest["path"])
        manifest_path = _resolve_planned_artifact_path(
            repo_root, manifest_relative, "planned direction manifest"
        )
        if manifest_relative in seen_manifest_paths:
            raise ValueError("forced-grid plan repeats a direction manifest")
        seen_manifest_paths.add(manifest_relative)
        if not manifest_path.is_file() or sha256_file(manifest_path) != validate_sha256(
            manifest.get("sha256"), "direction manifest sha256"
        ):
            raise RuntimeError("planned direction manifest is missing or changed")
    points = plan.get("points")
    if not isinstance(points, list) or len(points) != EXPECTED_POINTS_PER_MODEL:
        raise ValueError("forced-grid plan must contain exactly 250 points")
    expected_specs = derive_forced_grid_specs(lock, model_id)
    observed_specs = []
    point_hashes = set()
    shard_names = set()
    for index, point in enumerate(points):
        if not isinstance(point, dict) or point.get("grid_index") != index:
            raise ValueError("forced-grid points must be ordered and indexed contiguously")
        core = {
            key: value for key, value in point.items() if key not in {"point_sha256", "shard_name"}
        }
        point_hash = canonical_json_sha256(core)
        if point.get("point_sha256") != point_hash:
            raise ValueError("forced-grid point hash is invalid")
        expected_name = f"point_{index:03d}_{point_hash[:16]}.json"
        if point.get("shard_name") != expected_name:
            raise ValueError("forced-grid point shard name is invalid")
        method_id = str(point.get("method_id"))
        track = str(point.get("track"))
        expected_geometry = _expected_geometry(method_id, track)
        if point.get("position_schedule") != locked_position_schedule(method_id, track):
            raise ValueError("forced-grid point position schedule differs from the lock")
        if point.get("intervention_geometry") != expected_geometry:
            raise ValueError("forced-grid point geometry differs from the lock")
        validate_sha256(point.get("direction_file_sha256"), "direction_file_sha256")
        validate_sha256(point.get("direction_float32_sha256"), "direction_float32_sha256")
        validate_sha256(point.get("direction_artifact_sha256"), "direction_artifact_sha256")
        validate_sha256(
            point.get("construction_config_sha256"),
            "construction_config_sha256",
        )
        direction_path = _resolve_planned_artifact_path(
            repo_root, point.get("direction_path"), "planned direction artifact"
        )
        construction_path = _resolve_planned_artifact_path(
            repo_root,
            point.get("construction_config_path"),
            "planned construction config",
        )
        if not direction_path.is_file() or sha256_file(direction_path) != point.get(
            "direction_file_sha256"
        ):
            raise RuntimeError("planned direction artifact is missing or changed")
        if not construction_path.is_file() or sha256_file(construction_path) != point.get(
            "construction_config_sha256"
        ):
            raise RuntimeError("planned construction config is missing or changed")
        point_hashes.add(point_hash)
        shard_names.add(expected_name)
        observed_specs.append(
            GridPointSpec(
                str(point["method_id"]),
                str(point["track"]),
                int(point["layer"]),
                float(point["strength"]),
            )
        )
    if tuple(observed_specs) != expected_specs:
        raise ValueError("forced-grid point identities differ from the locked 250-point grid")
    if len(point_hashes) != len(points) or len(shard_names) != len(points):
        raise ValueError("forced-grid plan contains duplicate point/shard identities")
    return points


def _raw_a_minus_b(score: Any, first_semantic_label: str) -> float:
    if first_semantic_label == "A":
        return float(score.preserve_log_odds)
    if first_semantic_label == "B":
        return -float(score.preserve_log_odds)
    raise ValueError("semantic label must be A or B")


def _point_artifact(point: Mapping[str, Any], backend: Any) -> Any:
    direction_path = Path(str(point["direction_path"]))
    if sha256_file(direction_path) != point["direction_file_sha256"]:
        raise RuntimeError("direction artifact changed after grid planning")
    artifact = read_direction_artifact(direction_path, backend.torch)
    observed = {
        "method_id": artifact.method,
        "layer": artifact.layer,
        "intervention_geometry": artifact.intervention_geometry,
        "direction_float32_sha256": artifact.direction_sha256,
        "direction_artifact_sha256": artifact.artifact_sha256,
    }
    mismatches = {
        field: (point.get(field), value)
        for field, value in observed.items()
        if point.get(field) != value
    }
    if mismatches:
        raise RuntimeError(f"loaded direction differs from its grid point: {mismatches}")
    if (
        sha256_file(Path(str(point["construction_config_path"])))
        != point["construction_config_sha256"]
    ):
        raise RuntimeError("construction config changed after grid planning")
    return artifact


def _validate_backend_identity(
    backend: Any, plan: Mapping[str, Any], lock: Mapping[str, Any]
) -> None:
    """Fail closed unless the resident backend is the exact locked checkpoint/config."""

    config = getattr(backend, "config", None)
    model_config = getattr(config, "model", None)
    config_path = Path(str(getattr(config, "config_path", "")))
    expected_model = _locked_model(lock, str(plan["model_id"]))
    observed = {
        "model_id": getattr(model_config, "id", None),
        "model_revision": getattr(model_config, "revision", None),
        "prompt_format": getattr(model_config, "prompt_format", None),
    }
    expected = {
        "model_id": plan["model_id"],
        "model_revision": plan["model_revision"],
        "prompt_format": "chat",
    }
    mismatches = {
        field: (wanted, observed.get(field))
        for field, wanted in expected.items()
        if observed.get(field) != wanted
    }
    if mismatches:
        raise RuntimeError(f"resident backend differs from the grid plan: {mismatches}")
    if not config_path.is_file() or sha256_file(config_path) != plan["model_config_sha256"]:
        raise RuntimeError("resident backend config file differs from the locked model config")
    cfg = getattr(getattr(backend, "model", None), "cfg", None)
    architecture = expected_model["architecture"]
    if (
        getattr(cfg, "n_layers", None) != architecture["blocks"]
        or getattr(cfg, "d_model", None) != architecture["residual_width"]
    ):
        raise RuntimeError("resident backend architecture differs from the locked model")
    validate_locked_choice_runtime(backend, expected_model["runtime"])


def evaluate_forced_grid_point(
    backend: Any,
    point: Mapping[str, Any],
    prompt_units: Sequence[CalibrationPromptUnit],
    baseline_cache: BaselineLogitsCache,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate one point while reusing only prompt/model-bound baseline logits."""

    artifact = _point_artifact(point, backend)
    setup = MethodSetup(
        artifact=artifact,
        method_id=str(point["method_id"]),
        track=str(point["track"]),
        strength=float(point["strength"]),
    )
    setup.validate()
    rows: list[dict[str, Any]] = []
    for unit in prompt_units:
        tokens = backend.encode(unit.prompt)
        prompt_length = int(tokens.shape[-1])
        baseline_logits, baseline_cache_key, baseline_logits_hash = baseline_cache.get_or_compute(
            unit.prompt, lambda tokens=tokens: next_token_logits(backend, tokens)
        )
        boundary = resolve_choice_boundary(backend, unit.prompt)
        if boundary.prompt_length != prompt_length:
            raise RuntimeError("choice boundary prompt length differs from backend encoding")
        first_id = boundary.token_id(unit.first_semantic_label)
        second_id = boundary.token_id(unit.second_semantic_label)
        baseline_score = choice_score_from_logits(
            backend.torch,
            baseline_logits,
            first_id,
            second_id,
            preserve_label=unit.first_semantic_label,
            comply_label=unit.second_semantic_label,
            baseline_logits=baseline_logits,
            choice_boundary_evidence_sha256=boundary.evidence_sha256,
        )
        conditions = (
            ("baseline", 0, baseline_score),
            ("plus", 1, None),
            ("minus", -1, None),
        )
        for condition, sign, cached_score in conditions:
            if cached_score is None:
                spec = setup.intervention(prompt_length=prompt_length, sign=sign)
                logits, perturbation = next_token_logits_with_perturbation(backend, tokens, spec)
                score = choice_score_from_logits(
                    backend.torch,
                    logits,
                    first_id,
                    second_id,
                    preserve_label=unit.first_semantic_label,
                    comply_label=unit.second_semantic_label,
                    baseline_logits=baseline_logits,
                    perturbation=perturbation,
                    choice_boundary_evidence_sha256=boundary.evidence_sha256,
                )
            else:
                score = cached_score
            signed_strength = sign * setup.strength
            perturbation = score.perturbation
            row = {
                "schema_version": ROW_SCHEMA_VERSION,
                "model_id": plan["model_id"],
                "model_revision": plan["model_revision"],
                "dataset_sha256": plan["dataset_sha256"],
                "protocol_sha256": plan["protocol_sha256"],
                "config_sha256": plan["model_config_sha256"],
                "stage1_lock_sha256": plan["stage1_lock_sha256"],
                "stage2_manifest_sha256": ZERO_SHA256,
                "calibration_summary_sha256": ZERO_SHA256,
                "construction_config_sha256": point["construction_config_sha256"],
                "runner_commit": plan["runner_commit"],
                "direction_sha256": point["direction_float32_sha256"],
                "direction_float32_sha256": point["direction_float32_sha256"],
                "direction_artifact_sha256": point["direction_artifact_sha256"],
                "prompt_sha256": prompt_sha256(unit.prompt),
                "baseline_cache_key_sha256": baseline_cache_key,
                "baseline_logits_float32_sha256": baseline_logits_hash,
                "method": setup.method_id,
                "method_id": setup.method_id,
                "setup": setup.track,
                "track": setup.track,
                "direction_id": point["direction_artifact_sha256"],
                "strength_id": f"{setup.track}:{setup.strength:.12g}",
                "calibration_magnitude": setup.strength,
                "split": "validation",
                "family": unit.family,
                "case_id": unit.case_id,
                "condition": condition,
                "condition_alpha": signed_strength,
                "strength": signed_strength,
                "layer": setup.artifact.layer,
                "position": setup.position,
                "run_seed": plan["run_seed"],
                "a_minus_b_log_odds": _raw_a_minus_b(score, unit.first_semantic_label),
                "forced_pair_label": score.pair_choice,
                "actual_next_token_label": score.predicted_label,
                "raw_vocabulary_choice": score.predicted_label,
                "kl_from_baseline": score.kl_from_baseline,
                "answer_format_valid": score.predicted_label != "OTHER",
                "coherent": True,
                "coherence_assessed": False,
                "answer_pair_mass": score.answer_pair_mass,
                "choice_boundary_evidence_sha256": boundary.evidence_sha256,
                "choice_a_token_id": boundary.a_token_id,
                "choice_b_token_id": boundary.b_token_id,
                "realized_mean_relative_perturbation_norm": (
                    0.0 if perturbation is None else perturbation["mean_relative_l2_norm"]
                ),
                "realized_max_relative_perturbation_norm": (
                    0.0 if perturbation is None else perturbation["max_relative_l2_norm"]
                ),
                "realized_mean_perturbation_l2_norm": (
                    0.0 if perturbation is None else perturbation["mean_l2_norm"]
                ),
                "realized_perturbed_positions": (
                    0 if perturbation is None else perturbation["n_positions"]
                ),
                "first_semantic_label": unit.first_semantic_label,
                "second_semantic_label": unit.second_semantic_label,
                **dict(unit.extra_fields),
            }
            rows.append(row)
    return rows


def _expected_prompt_hashes(
    prompt_units: Sequence[CalibrationPromptUnit],
) -> dict[tuple[str, str, str, str], str]:
    output = {unit.key: prompt_sha256(unit.prompt) for unit in prompt_units}
    if len(output) != len(prompt_units):
        raise ValueError("forced prompt units contain duplicate identities")
    return output


def _validate_point_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    point: Mapping[str, Any],
    expected_units: set[tuple[str, str, str, str]],
    prompt_hashes: Mapping[tuple[str, str, str, str], str],
) -> dict[str, Any]:
    if len(rows) != EXPECTED_ROWS_PER_POINT:
        raise ValueError("one forced-grid shard must contain exactly 426 rows")
    coverage = validate_calibration_coverage(rows, expected_units)
    if coverage["unit_count"] != EXPECTED_UNITS_PER_POINT:
        raise ValueError("one forced-grid shard must contain exactly 142 units")
    validation = validate_result_rows(
        rows,
        expected_hashes={
            "dataset_sha256": plan["dataset_sha256"],
            "protocol_sha256": plan["protocol_sha256"],
            "config_sha256": plan["model_config_sha256"],
            "direction_sha256": point["direction_float32_sha256"],
            "direction_float32_sha256": point["direction_float32_sha256"],
            "direction_artifact_sha256": point["direction_artifact_sha256"],
            "stage1_lock_sha256": plan["stage1_lock_sha256"],
            "stage2_manifest_sha256": ZERO_SHA256,
            "calibration_summary_sha256": ZERO_SHA256,
            "construction_config_sha256": point["construction_config_sha256"],
        },
    )
    stable_expected = {
        "model_id": plan["model_id"],
        "model_revision": plan["model_revision"],
        "method": point["method_id"],
        "method_id": point["method_id"],
        "setup": point["track"],
        "track": point["track"],
        "layer": point["layer"],
        "position": point["position_schedule"],
        "runner_commit": plan["runner_commit"],
        "run_seed": plan["run_seed"],
        "split": "validation",
        "direction_id": point["direction_artifact_sha256"],
        "calibration_magnitude": point["strength"],
    }
    baseline_manifest = []
    seen_baseline_units = set()
    baseline_evidence_by_unit: dict[tuple[str, str, str, str], tuple[Any, ...]] = {}
    for row in rows:
        mismatches = {
            field: (expected, row.get(field))
            for field, expected in stable_expected.items()
            if row.get(field) != expected
        }
        if mismatches:
            raise ValueError(f"forced-grid row differs from its point: {mismatches}")
        key = calibration_unit_key(row)
        expected_prompt_hash = prompt_hashes.get(key)
        if row.get("prompt_sha256") != expected_prompt_hash:
            raise ValueError("forced-grid row prompt differs from the locked validation prompt")
        expected_cache_key = baseline_cache_key_sha256(
            model_id=str(plan["model_id"]),
            model_revision=str(plan["model_revision"]),
            model_config_sha256=str(plan["model_config_sha256"]),
            prompt_sha256_value=str(expected_prompt_hash),
        )
        if row.get("baseline_cache_key_sha256") != expected_cache_key:
            raise ValueError("forced-grid row has an invalid identity-independent baseline hash")
        baseline_logits_hash = validate_sha256(
            row.get("baseline_logits_float32_sha256"),
            "baseline_logits_float32_sha256",
        )
        boundary_hash = validate_sha256(
            row.get("choice_boundary_evidence_sha256"),
            "choice_boundary_evidence_sha256",
        )
        a_token_id = row.get("choice_a_token_id")
        b_token_id = row.get("choice_b_token_id")
        if (
            isinstance(a_token_id, bool)
            or not isinstance(a_token_id, int)
            or a_token_id < 0
            or isinstance(b_token_id, bool)
            or not isinstance(b_token_id, int)
            or b_token_id < 0
            or a_token_id == b_token_id
        ):
            raise ValueError("forced-grid row has invalid A/B boundary token IDs")
        baseline_evidence = (
            expected_cache_key,
            baseline_logits_hash,
            boundary_hash,
            a_token_id,
            b_token_id,
        )
        prior_evidence = baseline_evidence_by_unit.setdefault(key, baseline_evidence)
        if prior_evidence != baseline_evidence:
            raise ValueError(
                "forced-grid triplet has mismatched baseline or choice-boundary evidence"
            )
        if row["condition"] in {"plus", "minus"} and not math.isclose(
            abs(float(row["strength"])),
            float(point["strength"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("forced-grid signed strength differs from its point magnitude")
        if key not in seen_baseline_units:
            baseline_manifest.append(
                {
                    "unit": list(key),
                    "prompt_sha256": expected_prompt_hash,
                    "baseline_cache_key_sha256": expected_cache_key,
                    "baseline_logits_float32_sha256": baseline_logits_hash,
                    "choice_boundary_evidence_sha256": boundary_hash,
                    "choice_a_token_id": a_token_id,
                    "choice_b_token_id": b_token_id,
                }
            )
            seen_baseline_units.add(key)
    if seen_baseline_units != expected_units:
        raise ValueError("baseline hash manifest does not exactly cover forced units")
    return {
        "coverage": coverage,
        "row_validation": validation,
        "rows_sha256": calibration_rows_sha256(rows),
        "baseline_evidence_manifest": sorted(
            baseline_manifest, key=lambda item: tuple(item["unit"])
        ),
    }


def build_point_shard(
    *,
    plan: Mapping[str, Any],
    point: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    expected_units: set[tuple[str, str, str, str]],
    prompt_hashes: Mapping[tuple[str, str, str, str], str],
) -> dict[str, Any]:
    details = _validate_point_rows(
        rows,
        plan=plan,
        point=point,
        expected_units=expected_units,
        prompt_hashes=prompt_hashes,
    )
    payload = {
        "schema_version": GRID_SHARD_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "point_sha256": point["point_sha256"],
        "point": dict(point),
        "unit_count": details["coverage"]["unit_count"],
        "row_count": len(rows),
        "coverage_sha256": details["coverage"]["coverage_sha256"],
        "rows_sha256": details["rows_sha256"],
        "baseline_evidence_manifest_sha256": canonical_json_sha256(
            details["baseline_evidence_manifest"]
        ),
        "rows": [dict(row) for row in rows],
    }
    return {**payload, "shard_content_sha256": canonical_json_sha256(payload)}


def validate_point_shard(
    shard: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    point: Mapping[str, Any],
    expected_units: set[tuple[str, str, str, str]],
    prompt_hashes: Mapping[tuple[str, str, str, str], str],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "plan_sha256",
        "point_sha256",
        "point",
        "unit_count",
        "row_count",
        "coverage_sha256",
        "rows_sha256",
        "baseline_evidence_manifest_sha256",
        "rows",
        "shard_content_sha256",
    }
    if set(shard) != required or shard.get("schema_version") != GRID_SHARD_SCHEMA_VERSION:
        raise ValueError("point shard fields/schema are invalid")
    payload = {key: value for key, value in shard.items() if key != "shard_content_sha256"}
    if shard.get("shard_content_sha256") != canonical_json_sha256(payload):
        raise ValueError("point shard content hash is invalid")
    if shard.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("point shard belongs to another grid plan")
    if shard.get("point_sha256") != point["point_sha256"] or shard.get("point") != point:
        raise ValueError("point shard identity differs from the planned point")
    rows = shard.get("rows")
    if not isinstance(rows, list):
        raise TypeError("point shard rows must be a list")
    details = _validate_point_rows(
        rows,
        plan=plan,
        point=point,
        expected_units=expected_units,
        prompt_hashes=prompt_hashes,
    )
    observed = {
        "unit_count": details["coverage"]["unit_count"],
        "row_count": len(rows),
        "coverage_sha256": details["coverage"]["coverage_sha256"],
        "rows_sha256": details["rows_sha256"],
        "baseline_evidence_manifest_sha256": canonical_json_sha256(
            details["baseline_evidence_manifest"]
        ),
    }
    mismatches = {
        field: (expected, shard.get(field))
        for field, expected in observed.items()
        if shard.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"point shard summaries differ from its rows: {mismatches}")
    return {
        **observed,
        "point_sha256": point["point_sha256"],
        "shard_content_sha256": shard["shard_content_sha256"],
    }


def load_point_shard(
    path: str | Path,
    *,
    plan: Mapping[str, Any],
    point: Mapping[str, Any],
    expected_units: set[tuple[str, str, str, str]],
    prompt_hashes: Mapping[tuple[str, str, str, str], str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = Path(path)
    shard = _read_json_object(target, "point shard")
    validation = validate_point_shard(
        shard,
        plan=plan,
        point=point,
        expected_units=expected_units,
        prompt_hashes=prompt_hashes,
    )
    validation["file_sha256"] = sha256_file(target)
    return shard, validation


def load_validated_point_rows(
    path: str | Path,
    *,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    dataset: Mapping[str, Any],
    repo_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the exact 426 rows from one fully validated point shard.

    This is the compatibility boundary for consumers that historically accepted
    JSONL row streams.  Consumers must not extract the embedded ``rows`` array
    directly: this adapter rechecks the plan, immutable artifacts, validation-only
    prompt projection, point identity, coverage, and every shard/content hash first.
    """

    points = _validate_plan(plan, lock, repo_root=repo_root)
    prompt_units = build_forced_prompt_units(dataset, lock)
    expected_units = locked_forced_calibration_units(dataset, lock)
    prompt_hashes = _expected_prompt_hashes(prompt_units)
    untrusted = _read_json_object(Path(path), "point shard")
    point_hash = validate_sha256(untrusted.get("point_sha256"), "point_sha256")
    matching = [point for point in points if point.get("point_sha256") == point_hash]
    if len(matching) != 1:
        raise ValueError("point shard does not identify exactly one planned point")
    shard, validation = load_point_shard(
        path,
        plan=plan,
        point=matching[0],
        expected_units=expected_units,
        prompt_hashes=prompt_hashes,
    )
    rows = shard["rows"]
    if len(rows) != EXPECTED_ROWS_PER_POINT:
        raise RuntimeError("validated point shard unexpectedly lacks exactly 426 rows")
    return [dict(row) for row in rows], validation


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically create a new JSON file without replacing any existing evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing artifact: {path}")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link publishes the already-complete inode and fails atomically
        # if another runner created the final name first.  It never overwrites.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def write_grid_plan_atomic(
    plan: Mapping[str, Any],
    path: str | Path,
    *,
    lock: Mapping[str, Any],
    repo_root: str | Path,
) -> str:
    _validate_plan(plan, lock, repo_root=repo_root)
    target = Path(path)
    if target.exists():
        existing = _read_json_object(target, "existing forced-grid plan")
        _validate_plan(existing, lock, repo_root=repo_root)
        if existing != plan:
            raise RuntimeError("existing forced-grid plan differs from the requested plan")
        return sha256_file(target)
    return _atomic_create_json(target, plan)


def write_point_shard_atomic(path: str | Path, shard: Mapping[str, Any]) -> str:
    if shard.get("schema_version") != GRID_SHARD_SCHEMA_VERSION:
        raise ValueError("cannot write an unsupported point-shard schema")
    return _atomic_create_json(Path(path), shard)


PointEvaluator = Callable[
    [
        Any,
        Mapping[str, Any],
        Sequence[CalibrationPromptUnit],
        BaselineLogitsCache,
        Mapping[str, Any],
    ],
    Sequence[Mapping[str, Any]],
]


def run_forced_grid(
    *,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    dataset: Mapping[str, Any],
    repo_root: str | Path,
    output_dir: str | Path,
    backend: Any | None = None,
    backend_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    point_evaluator: PointEvaluator = evaluate_forced_grid_point,
    only_point_sha256s: Iterable[str] | None = None,
    max_new_points: int | None = None,
) -> dict[str, Any]:
    """Validate/resume the grid and evaluate pending points with one resident model."""

    root = Path(repo_root).resolve()
    points = _validate_plan(plan, lock, repo_root=root)
    if backend is not None and backend_factory is not None:
        raise ValueError("supply backend or backend_factory, not both")
    if max_new_points is not None and (
        isinstance(max_new_points, bool)
        or not isinstance(max_new_points, int)
        or max_new_points < 0
    ):
        raise ValueError("max_new_points must be a non-negative integer or null")
    requested = None
    if only_point_sha256s is not None:
        requested = {validate_sha256(value, "only_point_sha256") for value in only_point_sha256s}
        known = {str(point["point_sha256"]) for point in points}
        if not requested or not requested <= known:
            raise ValueError("requested point hashes must be a non-empty subset of the plan")
    prompt_units = build_forced_prompt_units(dataset, lock)
    expected_units = locked_forced_calibration_units(dataset, lock)
    prompt_hashes = _expected_prompt_hashes(prompt_units)
    target_dir = Path(output_dir).resolve()
    plan_path = target_dir / "forced_grid_plan.json"
    write_grid_plan_atomic(plan, plan_path, lock=lock, repo_root=root)
    shards_dir = target_dir / "points"
    shards_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {str(point["shard_name"]) for point in points}
    unexpected = sorted(
        path.name for path in shards_dir.glob("point_*.json") if path.name not in expected_names
    )
    if unexpected:
        raise RuntimeError(f"output directory contains unplanned point shards: {unexpected[:3]}")

    completed = []
    pending = []
    for point in points:
        shard_path = shards_dir / str(point["shard_name"])
        if shard_path.exists():
            _, validation = load_point_shard(
                shard_path,
                plan=plan,
                point=point,
                expected_units=expected_units,
                prompt_hashes=prompt_hashes,
            )
            completed.append(
                {
                    "point_sha256": point["point_sha256"],
                    "path": str(shard_path),
                    **validation,
                    "resume_status": "validated_existing",
                }
            )
        elif requested is None or point["point_sha256"] in requested:
            pending.append(point)
    if max_new_points is not None:
        pending = pending[:max_new_points]
    if pending and backend is None:
        if backend_factory is None:
            raise RuntimeError("pending grid points require one backend or backend_factory")
        backend = backend_factory(plan)
        if backend is None:
            raise RuntimeError("backend_factory returned null")
    if pending:
        _validate_backend_identity(backend, plan, lock)
    baseline_cache = BaselineLogitsCache(plan)
    written = []
    for point in pending:
        runtime_point = dict(point)
        runtime_point["direction_path"] = str(
            _resolve_planned_artifact_path(
                root, point["direction_path"], "planned direction artifact"
            )
        )
        runtime_point["construction_config_path"] = str(
            _resolve_planned_artifact_path(
                root,
                point["construction_config_path"],
                "planned construction config",
            )
        )
        rows = list(
            point_evaluator(backend, runtime_point, prompt_units, baseline_cache, plan)
        )
        shard = build_point_shard(
            plan=plan,
            point=point,
            rows=rows,
            expected_units=expected_units,
            prompt_hashes=prompt_hashes,
        )
        shard_path = shards_dir / str(point["shard_name"])
        file_hash = write_point_shard_atomic(shard_path, shard)
        _, validation = load_point_shard(
            shard_path,
            plan=plan,
            point=point,
            expected_units=expected_units,
            prompt_hashes=prompt_hashes,
        )
        if validation["file_sha256"] != file_hash:
            raise RuntimeError("atomic point shard hash changed after publication")
        written.append(
            {
                "point_sha256": point["point_sha256"],
                "path": str(shard_path),
                **validation,
                "resume_status": "written_and_revalidated",
            }
        )
    total_existing_after = len(completed) + len(written)
    return {
        "schema_version": GRID_PLAN_SCHEMA_VERSION,
        "status": ("complete" if total_existing_after == EXPECTED_POINTS_PER_MODEL else "partial"),
        "plan_sha256": plan["plan_sha256"],
        "model_id": plan["model_id"],
        "expected_point_count": EXPECTED_POINTS_PER_MODEL,
        "validated_existing_count": len(completed),
        "written_count": len(written),
        "remaining_count": EXPECTED_POINTS_PER_MODEL - total_existing_after,
        "baseline_cache_entry_count": len(baseline_cache),
        "baseline_forward_computation_count": baseline_cache.computations,
        "model_loaded_by_factory": bool(pending and backend_factory is not None),
        "shards": [*completed, *written],
    }
