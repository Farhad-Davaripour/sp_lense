"""Pure math for pair-aware adaptive gradient steering.

The construction in this module is deliberately narrower than an unconditional
steering-vector experiment.  It fits one direction for each ``case_id`` and role
``assignment`` from that matched prompt quartet's exact raw A-minus-B gradients,
then applies the *same* direction to its self and other targets and both option
orders.  A positive result therefore would be evidence about a privileged,
pair-aware intervention, not a universal self-preservation direction.

All fitting arithmetic is performed on CPU in float64.  The returned intervention
direction is a contiguous CPU float32 unit vector.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "sp_lense.gradient_specificity_adaptive.v1"
RAW_GRADIENT_CONVENTION = "residual_scaled_raw_A_minus_B"
DEFAULT_RESIDUAL_RELATIVE_TOLERANCE = 1e-10


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for a JSON-compatible value."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return a deterministic SHA-256 hash of a JSON-compatible value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def tensor_float32_sha256(value: Any) -> str:
    """Hash exact contiguous little-endian CPU float32 tensor bytes."""

    if not hasattr(value, "detach"):
        raise TypeError("value must be a tensor")
    array = value.detach().to(device="cpu").float().contiguous().numpy()
    return hashlib.sha256(array.astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def tensor_float64_sha256(value: Any) -> str:
    """Hash exact contiguous little-endian CPU float64 tensor bytes."""

    if not hasattr(value, "detach"):
        raise TypeError("value must be a tensor")
    array = value.detach().to(device="cpu").double().contiguous().numpy()
    return hashlib.sha256(array.astype("<f8", copy=False).tobytes(order="C")).hexdigest()


def _validate_case_id(case_id: Any, *, field: str = "case_id") -> str:
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"{field} must be a non-empty string")
    return case_id


def _validate_assignment(assignment: Any, *, field: str = "assignment") -> int:
    if not isinstance(assignment, int) or isinstance(assignment, bool) or assignment not in {0, 1}:
        raise ValueError(f"{field} must be 0 or 1")
    return assignment


def _validate_target(target: Any, *, field: str = "target") -> str:
    if target not in {"self", "other"}:
        raise ValueError(f"{field} must be 'self' or 'other'")
    return str(target)


def _float64_vector(torch: Any, value: Any, *, field: str) -> Any:
    if not torch.is_tensor(value):
        raise TypeError(f"{field} must be a tensor")
    if torch.is_complex(value) or not torch.is_floating_point(value):
        raise TypeError(f"{field} must be a real floating-point tensor")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{field} must be a non-empty one-dimensional tensor")
    vector = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(vector).all().item()):
        raise ValueError(f"{field} must contain only finite values")
    return vector


def semantic_ab_gradient(
    torch: Any,
    raw_a_minus_b: Any,
    *,
    preserve_first: bool,
) -> Any:
    """Convert a raw A-minus-B gradient to preserve-minus-comply semantics.

    ``preserve_first=True`` means preservation is option A.  When preservation
    is option B, negating the raw A-minus-B gradient performs the sole semantic
    sign correction.
    """

    if not isinstance(preserve_first, bool):
        raise TypeError("preserve_first must be a bool")
    vector = _float64_vector(torch, raw_a_minus_b, field="raw_a_minus_b")
    return vector if preserve_first else -vector


def _parse_group_records(
    torch: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    case_id: str,
    assignment: int,
) -> dict[tuple[str, bool], Any]:
    wanted_case = _validate_case_id(case_id)
    wanted_assignment = _validate_assignment(assignment)
    rows = list(records)
    if not rows:
        raise ValueError("records must contain the exact four choice cells")

    cells: dict[tuple[str, bool], Any] = {}
    common_shape = None
    for index, record in enumerate(rows):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{index}] must be a mapping")
        if record.get("kind") != "choice":
            raise ValueError(f"records[{index}].kind must be 'choice'")
        if record.get("gradient_convention") != RAW_GRADIENT_CONVENTION:
            raise ValueError(
                f"records[{index}].gradient_convention must be {RAW_GRADIENT_CONVENTION!r}"
            )
        observed_case = _validate_case_id(record.get("case_id"), field=f"records[{index}].case_id")
        observed_assignment = _validate_assignment(
            record.get("assignment"), field=f"records[{index}].assignment"
        )
        if observed_case != wanted_case or observed_assignment != wanted_assignment:
            raise ValueError(
                "records must contain only the requested case_id and assignment; "
                f"got ({observed_case!r}, {observed_assignment})"
            )
        target = _validate_target(record.get("target"), field=f"records[{index}].target")
        preserve_first = record.get("preserve_first")
        if not isinstance(preserve_first, bool):
            raise TypeError(f"records[{index}].preserve_first must be a bool")
        if "gradient" not in record:
            raise ValueError(f"records[{index}] is missing gradient")
        gradient = _float64_vector(torch, record["gradient"], field=f"records[{index}].gradient")
        if common_shape is None:
            common_shape = gradient.shape
        elif gradient.shape != common_shape:
            raise ValueError("all four gradients must have the same shape")
        cell = (target, preserve_first)
        if cell in cells:
            raise ValueError(f"duplicate exact choice cell {cell}")
        cells[cell] = gradient

    expected = {
        (target, preserve_first)
        for target in ("self", "other")
        for preserve_first in (False, True)
    }
    if set(cells) != expected:
        missing = sorted(expected - set(cells))
        extra = sorted(set(cells) - expected)
        raise ValueError(f"records lack exact four-cell coverage; missing={missing}, extra={extra}")
    return cells


def _input_manifest(cells: Mapping[tuple[str, bool], Any]) -> dict[str, str]:
    return {
        f"{target}:preserve_{'A' if preserve_first else 'B'}": tensor_float64_sha256(
            cells[(target, preserve_first)]
        )
        for target in ("self", "other")
        for preserve_first in (True, False)
    }


def _projection_summary(
    torch: Any,
    direction: Any,
    cells: Mapping[tuple[str, bool], Any],
) -> dict[str, Any]:
    vector = _float64_vector(torch, direction, field="direction")
    reference_shape = next(iter(cells.values())).shape
    if vector.shape != reference_shape:
        raise ValueError("direction and gradients must have the same shape")
    direction_norm = float(torch.linalg.vector_norm(vector).item())
    if not math.isclose(direction_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("direction must be unit normalized")

    raw_self_a = cells[("self", True)]
    raw_self_b = cells[("self", False)]
    raw_other_a = cells[("other", True)]
    raw_other_b = cells[("other", False)]
    self_a = raw_self_a
    self_b = -raw_self_b
    other_a = raw_other_a
    other_b = -raw_other_b
    self_gap = (self_a - self_b) / 2.0
    raw_self_label = (raw_self_a + raw_self_b) / 2.0
    raw_other_label = (raw_other_a + raw_other_b) / 2.0

    self_a_projection = float((vector @ self_a).item())
    self_b_projection = float((vector @ self_b).item())
    other_a_projection = float((vector @ other_a).item())
    other_b_projection = float((vector @ other_b).item())
    self_gap_projection = float((vector @ self_gap).item())
    raw_self_label_projection = float((vector @ raw_self_label).item())
    raw_other_label_projection = float((vector @ raw_other_label).item())
    nuisance_projections = (
        other_a_projection,
        other_b_projection,
        self_gap_projection,
        raw_self_label_projection,
        raw_other_label_projection,
    )
    return {
        "direction_norm": direction_norm,
        "self_semantic_projections": {
            "preserve_A": self_a_projection,
            "preserve_B": self_b_projection,
        },
        "matched_other_semantic_projections": {
            "preserve_A": other_a_projection,
            "preserve_B": other_b_projection,
        },
        "self_semantic_order_gap_projection": self_gap_projection,
        "raw_a_label_nuisance_projections": {
            "self": raw_self_label_projection,
            "other": raw_other_label_projection,
        },
        "mean_self_semantic_projection": (self_a_projection + self_b_projection) / 2.0,
        "minimum_self_semantic_projection": min(self_a_projection, self_b_projection),
        "both_self_orders_positive": self_a_projection > 0.0 and self_b_projection > 0.0,
        "maximum_abs_matched_other_projection": max(
            abs(other_a_projection), abs(other_b_projection)
        ),
        "maximum_abs_nuisance_projection": max(abs(value) for value in nuisance_projections),
    }


def construct_adaptive_direction(
    torch: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    case_id: str,
    assignment: int,
    residual_relative_tolerance: float = DEFAULT_RESIDUAL_RELATIVE_TOLERANCE,
) -> tuple[Any, dict[str, Any]]:
    """Fit one pair-aware direction from an exact case-assignment quartet.

    The nuisance matrix contains both matched-other semantic gradients, the
    self semantic option-order gap, and the raw A-label nuisance for self and
    other.  Exact SVD row-space removal leaves the unit vector that maximizes
    the mean of the two self semantic gradients subject to those orthogonality
    constraints.
    """

    if (
        not isinstance(residual_relative_tolerance, (int, float))
        or isinstance(residual_relative_tolerance, bool)
        or not math.isfinite(float(residual_relative_tolerance))
        or residual_relative_tolerance <= 0.0
    ):
        raise ValueError("residual_relative_tolerance must be finite and positive")
    wanted_case = _validate_case_id(case_id)
    wanted_assignment = _validate_assignment(assignment)
    cells = _parse_group_records(
        torch,
        records,
        case_id=wanted_case,
        assignment=wanted_assignment,
    )

    raw_self_a = cells[("self", True)]
    raw_self_b = cells[("self", False)]
    raw_other_a = cells[("other", True)]
    raw_other_b = cells[("other", False)]
    self_a = raw_self_a
    self_b = -raw_self_b
    other_a = raw_other_a
    other_b = -raw_other_b
    signal = (self_a + self_b) / 2.0
    self_gap = (self_a - self_b) / 2.0
    raw_self_label = (raw_self_a + raw_self_b) / 2.0
    raw_other_label = (raw_other_a + raw_other_b) / 2.0
    nuisance = torch.stack(
        (other_a, other_b, self_gap, raw_self_label, raw_other_label),
        dim=0,
    )

    _, singular_values, vh = torch.linalg.svd(nuisance, full_matrices=False)
    maximum_singular_value = float(singular_values.max().item())
    rank_tolerance = (
        max(nuisance.shape)
        * torch.finfo(torch.float64).eps
        * maximum_singular_value
    )
    rank = int((singular_values > rank_tolerance).sum().item())
    row_basis = vh[:rank]
    projected = signal - row_basis.T @ (row_basis @ signal)
    signal_norm = float(torch.linalg.vector_norm(signal).item())
    projected_norm = float(torch.linalg.vector_norm(projected).item())
    minimum_residual_norm = max(
        1e-12,
        float(residual_relative_tolerance) * signal_norm,
    )
    if not math.isfinite(projected_norm) or projected_norm <= minimum_residual_norm:
        raise RuntimeError(
            "self signal has no numerically usable component outside the nuisance span"
        )

    direction64 = projected / projected_norm
    if float((direction64 @ signal).item()) < 0.0:
        direction64 = -direction64
    nuisance_scale = max(
        1.0,
        max(float(torch.linalg.vector_norm(row).item()) for row in nuisance),
    )
    float64_tolerance = max(
        1e-10,
        1_000.0 * torch.finfo(torch.float64).eps * nuisance_scale,
    )
    float64_summary = _projection_summary(torch, direction64, cells)
    if float64_summary["maximum_abs_nuisance_projection"] > float64_tolerance:
        raise RuntimeError("SVD nuisance projection failed its float64 numerical check")
    if not float64_summary["both_self_orders_positive"]:
        raise RuntimeError("constructed direction did not positively orient both self orders")

    direction32 = direction64.to(dtype=torch.float32).contiguous()
    float32_summary = _projection_summary(torch, direction32, cells)
    float32_tolerance = max(1e-6, 1e-5 * nuisance_scale)
    if float32_summary["maximum_abs_nuisance_projection"] > float32_tolerance:
        raise RuntimeError("float32 direction exceeded the nuisance-projection tolerance")
    if not float32_summary["both_self_orders_positive"]:
        raise RuntimeError("float32 direction did not positively orient both self orders")

    input_manifest = _input_manifest(cells)
    diagnostics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case_id": wanted_case,
        "assignment": wanted_assignment,
        "application_scope": "one_case_assignment_same_direction_for_self_and_other",
        "objective": "maximize_mean_two_self_semantic_gradients_in_exact_nuisance_nullspace",
        "raw_gradient_convention": RAW_GRADIENT_CONVENTION,
        "arithmetic_dtype": "float64",
        "output_dtype": "float32",
        "dimension": int(direction32.numel()),
        "nuisance_rows": 5,
        "nuisance_rank": rank,
        "singular_values": [float(value) for value in singular_values.tolist()],
        "svd_rank_tolerance": rank_tolerance,
        "residual_relative_tolerance": float(residual_relative_tolerance),
        "signal_norm": signal_norm,
        "projected_signal_norm": projected_norm,
        "projected_over_signal_norm": projected_norm / signal_norm if signal_norm else None,
        "float64_numerical_tolerance": float64_tolerance,
        "float32_numerical_tolerance": float32_tolerance,
        "float64_projection_summary": float64_summary,
        "float32_projection_summary": float32_summary,
        "input_cell_gradient_sha256": input_manifest,
        "input_group_sha256": canonical_sha256(input_manifest),
        "signal_float64_sha256": tensor_float64_sha256(signal),
        "nuisance_matrix_float64_sha256": tensor_float64_sha256(nuisance),
        "direction_float64_sha256": tensor_float64_sha256(direction64),
        "direction_float32_sha256": tensor_float32_sha256(direction32),
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return direction32, diagnostics


def evaluate_adaptive_direction(
    torch: Any,
    direction: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    case_id: str,
    assignment: int,
) -> dict[str, Any]:
    """Evaluate one direction against the exact matched quartet without refitting."""

    cells = _parse_group_records(
        torch,
        records,
        case_id=case_id,
        assignment=assignment,
    )
    summary = _projection_summary(torch, direction, cells)
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "assignment": assignment,
        "same_direction_applied_to_all_four_cells": True,
        "direction_float32_sha256": tensor_float32_sha256(direction),
        **summary,
    }


def construct_adaptive_direction_bank(
    torch: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str],
    residual_relative_tolerance: float = DEFAULT_RESIDUAL_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Fit exactly two adaptive directions per expected case, one per assignment."""

    expected_ids = list(case_ids)
    if not expected_ids:
        raise ValueError("case_ids must be non-empty")
    if any(not isinstance(case_id, str) or not case_id for case_id in expected_ids):
        raise ValueError("case_ids must contain non-empty strings")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("case_ids must be unique")
    wanted_ids = set(expected_ids)

    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{index}] must be a mapping")
        case_id = _validate_case_id(record.get("case_id"), field=f"records[{index}].case_id")
        if case_id not in wanted_ids:
            raise ValueError(f"records[{index}] has unexpected case_id {case_id!r}")
        assignment = _validate_assignment(
            record.get("assignment"), field=f"records[{index}].assignment"
        )
        groups[(case_id, assignment)].append(record)

    expected_groups = {
        (case_id, assignment)
        for case_id in expected_ids
        for assignment in (0, 1)
    }
    if set(groups) != expected_groups:
        missing = sorted(expected_groups - set(groups))
        extra = sorted(set(groups) - expected_groups)
        raise ValueError(f"records lack exact case-assignment coverage; missing={missing}, extra={extra}")

    entries = []
    for case_id, assignment in sorted(expected_groups):
        direction, diagnostics = construct_adaptive_direction(
            torch,
            groups[(case_id, assignment)],
            case_id=case_id,
            assignment=assignment,
            residual_relative_tolerance=residual_relative_tolerance,
        )
        entries.append(
            {
                "case_id": case_id,
                "assignment": assignment,
                "direction": direction,
                "diagnostics": diagnostics,
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "case_ids": sorted(expected_ids),
        "entries": [
            {
                "case_id": entry["case_id"],
                "assignment": entry["assignment"],
                "direction_float32_sha256": entry["diagnostics"][
                    "direction_float32_sha256"
                ],
                "diagnostics_sha256": entry["diagnostics"]["diagnostics_sha256"],
            }
            for entry in entries
        ],
    }
    return {
        **manifest,
        "entries": entries,
        "manifest_sha256": canonical_sha256(manifest),
    }


def lookup_adaptive_direction(
    bank: Mapping[str, Any],
    *,
    case_id: str,
    assignment: int,
    target: str,
) -> Any:
    """Return the case-assignment vector; ``target`` never selects a vector."""

    wanted_case = _validate_case_id(case_id)
    wanted_assignment = _validate_assignment(assignment)
    _validate_target(target)
    if not isinstance(bank, Mapping) or bank.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("bank has the wrong schema_version")
    matches = [
        entry
        for entry in bank.get("entries", ())
        if entry.get("case_id") == wanted_case and entry.get("assignment") == wanted_assignment
    ]
    if len(matches) != 1:
        raise KeyError(f"bank has no unique direction for ({wanted_case!r}, {wanted_assignment})")
    direction = matches[0].get("direction")
    if direction is None:
        raise ValueError("bank entry is missing direction")
    return direction
