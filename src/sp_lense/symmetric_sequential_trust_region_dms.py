"""Pure mathematics for symmetric sequential trust-region DMS.

This module computes one local correction ``u`` to an already deployed
residual-relative direction ``D``.  It never loads a model and never chooses a
step from behavioral outcomes.  The two interventions after the update remain
exact negatives of one another::

    positive deployment = +(D + u)
    negative deployment = -(D + u)

The local gradients supplied to this module must already be expressed in the
same residual-relative coordinates as ``D`` and ``u``.  For a scalar physical
residual scale ``r``, a raw anchor gradient is converted by
``g_relative = r * g_raw``.  The gradients are evaluated at the actually
deployed current ``+D`` and ``-D`` states.  Changing the shared direction by
``u`` changes the negative deployment by ``-u``; keeping that chain-rule sign
explicit is the central purpose of this amendment.

The returned update is the certified minimum-Euclidean-norm point satisfying:

* fractional progress toward the target decision margin under both signs;
* preservation of every protected decision's baseline semantic side;
* affine return of unrelated margins by the same fixed progress fraction;
* exact retention of the baseline unrelated-gradient null space; and
* a fixed L2 trust radius.

The optimization is float64 and model-free, but the optimizer's ideal point is
not the deployed state.  The authoritative state is obtained by forming the
positive physical edit in float32, constructing the negative edit only by unary
negation of those exact bytes, and converting the positive edit back into
residual-relative float64 coordinates.  Both the ideal solution and that
realized state must pass independent certificates.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import linprog, minimize, nnls

from .decision_margin_shield_finite import FLOAT32_RAW_CONSTRAINT_TOLERANCE
from .factorial_causal_anchor import canonical_sha256

SCHEMA_VERSION = "sp_lense.symmetric_sequential_trust_region_dms.v2"
DEFAULT_PROGRESS_FRACTION = 0.25
DEFAULT_TRUST_RADIUS = 0.25

SVD_RTOL = 1e-10
SVD_ATOL = 1e-12
RAW_FEASIBILITY_TOLERANCE = 1e-8
ACTIVE_SLACK_TOLERANCE = 1e-7
KKT_TOLERANCE = 1e-7
DUALITY_GAP_TOLERANCE = 1e-7
TRUST_RADIUS_TOLERANCE = 1e-10
SOLVER_MAX_ITERATIONS = 2_000
SOLVER_FUNCTION_TOLERANCE = 1e-12


class SymmetricSequentialDMSInfeasibleError(RuntimeError):
    """The affine local problem has no certified update inside the trust radius."""


class SymmetricSequentialDMSSolverError(RuntimeError):
    """The numerical solver did not return a candidate eligible for certification."""


class SymmetricSequentialDMSCertificateError(RuntimeError):
    """A numerical candidate failed an independent scientific certificate."""


@dataclass(frozen=True)
class SymmetricSequentialTrustRegionUpdate:
    """One immutable, float32-authoritative shared-direction update."""

    current_direction: np.ndarray
    ideal_update: np.ndarray
    ideal_updated_direction: np.ndarray
    realized_update: np.ndarray
    realized_direction: np.ndarray
    positive_deployed_direction: np.ndarray
    negative_deployed_direction: np.ndarray
    positive_physical_float32: np.ndarray
    negative_physical_float32: np.ndarray
    positive_physical_float32_sha256: str
    negative_physical_float32_sha256: str
    diagnostics: Mapping[str, Any]

    @property
    def update(self) -> np.ndarray:
        """Backward-compatible alias for the authoritative realized update."""

        return self.realized_update

    @property
    def updated_direction(self) -> np.ndarray:
        """Backward-compatible alias for the authoritative realized direction."""

        return self.realized_direction


def _canonical_array(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64, order="C").copy(order="C")
    result[result == 0.0] = 0.0
    return result


def _array_sha256(value: Any) -> str:
    return canonical_sha256(_canonical_array(value).tolist())


def _float32_physical(value: Any) -> np.ndarray:
    """Copy one exact physical vector without canonicalizing signed zero."""

    return np.asarray(value, dtype=np.float32, order="C").copy(order="C")


def _float32_bytes_sha256(value: Any) -> str:
    return hashlib.sha256(_float32_physical(value).tobytes(order="C")).hexdigest()


def _float32_bytes_equal(left: Any, right: Any) -> bool:
    left_array = _float32_physical(left)
    right_array = _float32_physical(right)
    return left_array.shape == right_array.shape and (
        left_array.tobytes(order="C") == right_array.tobytes(order="C")
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _plain_diagnostics(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_diagnostics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_diagnostics(item) for item in value]
    return value


def _finite_vector(value: Any, *, field: str, length: int | None = None) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except Exception as error:  # pragma: no cover - NumPy reports the concrete cause.
        raise TypeError(f"{field} must be an array") from error
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{field} must contain real numbers")
    result = _canonical_array(raw)
    if result.ndim != 1 or (length is not None and result.shape != (length,)):
        requirement = "a vector" if length is None else f"exactly {length} values"
        raise ValueError(f"{field} must contain {requirement}")
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values")
    return result


def _finite_matrix(
    value: Any,
    *,
    field: str,
    width: int,
    rows: int | None = None,
) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except Exception as error:  # pragma: no cover - NumPy reports the concrete cause.
        raise TypeError(f"{field} must be an array") from error
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{field} must contain real numbers")
    result = _canonical_array(raw)
    if result.ndim != 2 or result.shape[1] != width:
        raise ValueError(f"{field} must have shape [rows, {width}]")
    if rows is not None and result.shape[0] != rows:
        raise ValueError(f"{field} must contain exactly {rows} rows")
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values")
    return result


def _nonnegative_rows(value: Any, *, field: str, length: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim == 0:
        if raw.dtype.kind not in "iuf" or isinstance(raw.item(), (bool, np.bool_)):
            raise TypeError(f"{field} must be a real scalar or vector")
        scalar = float(raw)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"{field} must be finite and nonnegative")
        return np.full(length, scalar, dtype=np.float64)
    result = _finite_vector(value, field=field, length=length)
    if bool(np.any(result < 0.0)):
        raise ValueError(f"{field} must be nonnegative")
    return result


def _positive_scalar(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{field} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _progress_fraction(value: Any) -> float:
    result = _positive_scalar(value, field="progress_fraction")
    if result > 1.0:
        raise ValueError("progress_fraction must lie in (0, 1]")
    return result


def _paired_family(
    *,
    plus_margins: Any | None,
    plus_gradients: Any | None,
    minus_margins: Any | None,
    minus_gradients: Any | None,
    width: int,
    family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = (plus_margins, plus_gradients, minus_margins, minus_gradients)
    if all(value is None for value in values):
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros((0, width), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, width), dtype=np.float64),
        )
    if any(value is None for value in values):
        raise ValueError(f"all four {family} branch arrays must be supplied together")
    plus_m = _finite_vector(plus_margins, field=f"{family}_plus_margins")
    count = int(plus_m.size)
    minus_m = _finite_vector(minus_margins, field=f"{family}_minus_margins", length=count)
    plus_g = _finite_matrix(
        plus_gradients,
        field=f"{family}_plus_gradients",
        width=width,
        rows=count,
    )
    minus_g = _finite_matrix(
        minus_gradients,
        field=f"{family}_minus_gradients",
        width=width,
        rows=count,
    )
    return plus_m, plus_g, minus_m, minus_g


def _canonicalize_equalities(
    rows: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Normalize equality signs/order and discard only exact zero tautologies."""

    retained: list[tuple[np.ndarray, float]] = []
    zero_tautologies = 0
    for row, value in zip(rows, values, strict=True):
        norm = float(np.linalg.norm(row))
        if norm == 0.0:
            if value != 0.0:
                raise SymmetricSequentialDMSInfeasibleError(
                    "a zero unrelated-gradient equality requires a nonzero margin return"
                )
            zero_tautologies += 1
            continue
        normalized = _canonical_array(row / norm)
        normalized_value = float(value / norm)
        anchor = int(np.argmax(np.abs(normalized)))
        if normalized[anchor] < 0.0:
            normalized *= -1.0
            normalized_value *= -1.0
        retained.append((normalized, normalized_value))
    retained.sort(key=lambda item: tuple(item[0].tolist()) + (item[1],))
    if not retained:
        return (
            np.zeros((0, rows.shape[1]), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            zero_tautologies,
        )
    return (
        np.stack([item[0] for item in retained]),
        np.asarray([item[1] for item in retained], dtype=np.float64),
        zero_tautologies,
    )


def _canonicalize_inequalities(
    rows: np.ndarray, lower: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    items = [(_canonical_array(row), float(value)) for row, value in zip(rows, lower, strict=True)]
    items.sort(key=lambda item: tuple(item[0].tolist()) + (item[1],))
    if not items:
        return (
            np.zeros((0, rows.shape[1]), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    return np.stack([item[0] for item in items]), np.asarray(
        [item[1] for item in items], dtype=np.float64
    )


def _canonicalize_basis_signs(basis: np.ndarray) -> np.ndarray:
    result = _canonical_array(basis)
    for row in result:
        anchor = int(np.argmax(np.abs(row)))
        if row[anchor] < 0.0:
            row *= -1.0
    return result


def _row_basis(rows: np.ndarray, *, scientific_rank: bool) -> tuple[np.ndarray, dict[str, Any]]:
    dimension = int(rows.shape[1])
    if rows.shape[0] == 0:
        basis = np.zeros((0, dimension), dtype=np.float64)
        singular_values = np.zeros(0, dtype=np.float64)
        threshold = SVD_ATOL if scientific_rank else 0.0
    else:
        norms = np.linalg.norm(rows, axis=1)
        normalized = np.zeros_like(rows)
        nonzero = norms > 0.0
        normalized[nonzero] = rows[nonzero] / norms[nonzero, None]
        _, singular_values, vh = np.linalg.svd(normalized, full_matrices=False)
        largest = float(singular_values[0]) if singular_values.size else 0.0
        threshold = (
            max(SVD_ATOL, SVD_RTOL * largest)
            if scientific_rank
            else np.finfo(np.float64).eps * max(normalized.shape) * largest
        )
        rank = int(np.count_nonzero(singular_values > threshold))
        basis = _canonicalize_basis_signs(vh[:rank])
    reconstructed = (rows @ basis.T) @ basis
    residual = rows - reconstructed
    orthonormality = basis @ basis.T - np.eye(basis.shape[0], dtype=np.float64)
    maximum_reconstruction = float(np.max(np.abs(residual))) if residual.size else 0.0
    maximum_orthonormality = float(np.max(np.abs(orthonormality))) if orthonormality.size else 0.0
    row_scale = max(1.0, float(np.max(np.abs(rows))) if rows.size else 0.0)
    reconstruction_tolerance = (
        max(
            256.0 * np.finfo(np.float64).eps,
            threshold,
        )
        * row_scale
    )
    orthonormality_tolerance = 256.0 * np.finfo(np.float64).eps * max(1, basis.shape[0])
    checks = {
        "rowspace_reconstruction": bool(maximum_reconstruction <= reconstruction_tolerance),
        "basis_orthonormality": bool(maximum_orthonormality <= orthonormality_tolerance),
    }
    diagnostics = {
        "rank_rule": (
            "fixed_scientific_equality_tolerance"
            if scientific_rank
            else "machine_precision_representer_span"
        ),
        "input_row_count": int(rows.shape[0]),
        "dimension": dimension,
        "singular_values": singular_values.tolist(),
        "rank_threshold": float(threshold),
        "rank": int(basis.shape[0]),
        "basis_sha256": _array_sha256(basis),
        "maximum_abs_rowspace_reconstruction_residual": maximum_reconstruction,
        "rowspace_reconstruction_tolerance": reconstruction_tolerance,
        "maximum_abs_basis_orthonormality_residual": maximum_orthonormality,
        "basis_orthonormality_tolerance": orthonormality_tolerance,
        "checks": checks,
        "passes": bool(all(checks.values())),
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return basis, diagnostics


def _raw_tolerance(rows: np.ndarray, lower: np.ndarray, update_norm: float) -> np.ndarray:
    row_norms = np.linalg.norm(rows, axis=1)
    scale = np.maximum.reduce(
        (
            np.ones(rows.shape[0], dtype=np.float64),
            np.abs(lower),
            row_norms * max(1.0, update_norm),
        )
    )
    return RAW_FEASIBILITY_TOLERANCE * scale


def _certify_update(
    *,
    update: np.ndarray,
    reduced: np.ndarray,
    equality_particular: np.ndarray,
    equality_basis: np.ndarray,
    representer_basis: np.ndarray,
    inequalities: np.ndarray,
    inequality_lower: np.ndarray,
    equalities: np.ndarray,
    equality_values: np.ndarray,
    reduced_inequalities: np.ndarray,
    reduced_lower: np.ndarray,
    trust_radius: float,
    current_direction: np.ndarray,
    target_plus_margins: np.ndarray,
    target_plus_gradients: np.ndarray,
    target_minus_margins: np.ndarray,
    target_minus_gradients: np.ndarray,
    target_goal: np.ndarray,
    target_required_progress: np.ndarray,
    protected_plus_margins: np.ndarray,
    protected_plus_gradients: np.ndarray,
    protected_minus_margins: np.ndarray,
    protected_minus_gradients: np.ndarray,
    protected_signs: np.ndarray,
    protected_floor: np.ndarray,
    unrelated_baseline_margins: np.ndarray,
    unrelated_plus_margins: np.ndarray,
    unrelated_plus_gradients: np.ndarray,
    unrelated_minus_margins: np.ndarray,
    unrelated_minus_gradients: np.ndarray,
    baseline_unrelated_gradients: np.ndarray,
    progress_fraction: float,
) -> dict[str, Any]:
    update_norm = float(np.linalg.norm(update))
    raw_inequality_slacks = inequalities @ update - inequality_lower
    raw_inequality_tolerances = _raw_tolerance(inequalities, inequality_lower, update_norm)
    raw_equality_residuals = equalities @ update - equality_values
    raw_equality_tolerances = _raw_tolerance(equalities, equality_values, update_norm)

    reduced_slacks = reduced_inequalities @ reduced - reduced_lower
    reduced_scale = max(
        1.0,
        float(np.max(np.abs(reduced_lower))) if reduced_lower.size else 0.0,
        float(np.linalg.norm(reduced)),
    )
    active_threshold = ACTIVE_SLACK_TOLERANCE * reduced_scale
    active = np.flatnonzero(reduced_slacks <= active_threshold)
    if active.size:
        active_matrix = reduced_inequalities[active]
        multipliers_active, nnls_residual = nnls(active_matrix.T, reduced)
    else:
        multipliers_active = np.zeros(0, dtype=np.float64)
        nnls_residual = float(np.linalg.norm(reduced))
    multipliers = np.zeros(reduced_lower.size, dtype=np.float64)
    multipliers[active] = multipliers_active
    stationarity = reduced - reduced_inequalities.T @ multipliers
    complementarity = multipliers * reduced_slacks
    primal_objective = 0.5 * float(reduced @ reduced)
    dual_vector = reduced_inequalities.T @ multipliers
    dual_objective = float(reduced_lower @ multipliers) - 0.5 * float(dual_vector @ dual_vector)
    duality_gap = primal_objective - dual_objective
    kkt_scale = max(1.0, float(np.linalg.norm(reduced)), float(np.linalg.norm(dual_vector)))
    gap_tolerance = DUALITY_GAP_TOLERANCE * max(1.0, primal_objective)

    target_plus_next = target_plus_margins + target_plus_gradients @ update
    target_minus_next = target_minus_margins - target_minus_gradients @ update
    target_oriented_current = np.concatenate((target_plus_margins, -target_minus_margins))
    target_oriented_next = np.concatenate((target_plus_next, -target_minus_next))
    target_realized_progress = target_oriented_next - target_oriented_current

    protected_plus_next = protected_plus_margins + protected_plus_gradients @ update
    protected_minus_next = protected_minus_margins - protected_minus_gradients @ update
    protected_oriented_next = np.concatenate(
        (protected_signs * protected_plus_next, protected_signs * protected_minus_next)
    )

    unrelated_plus_next = unrelated_plus_margins + unrelated_plus_gradients @ update
    unrelated_minus_next = unrelated_minus_margins - unrelated_minus_gradients @ update
    unrelated_plus_desired = unrelated_plus_margins + progress_fraction * (
        unrelated_baseline_margins - unrelated_plus_margins
    )
    unrelated_minus_desired = unrelated_minus_margins + progress_fraction * (
        unrelated_baseline_margins - unrelated_minus_margins
    )
    unrelated_residuals = np.concatenate(
        (
            unrelated_plus_next - unrelated_plus_desired,
            unrelated_minus_next - unrelated_minus_desired,
        )
    )
    unrelated_scale = max(
        1.0,
        float(np.max(np.abs(unrelated_baseline_margins)))
        if unrelated_baseline_margins.size
        else 0.0,
    )
    unrelated_tolerance = RAW_FEASIBILITY_TOLERANCE * unrelated_scale
    baseline_unrelated_updated_values = baseline_unrelated_gradients @ (current_direction + update)
    baseline_unrelated_scale = max(
        1.0,
        (
            float(np.max(np.linalg.norm(baseline_unrelated_gradients, axis=1)))
            * max(1.0, float(np.linalg.norm(current_direction + update)))
            if baseline_unrelated_gradients.shape[0]
            else 0.0
        ),
    )
    baseline_unrelated_tolerance = RAW_FEASIBILITY_TOLERANCE * baseline_unrelated_scale

    updated_direction = current_direction + update
    positive_deployed = updated_direction
    negative_deployed = -updated_direction
    reconstructed = equality_particular + representer_basis.T @ reduced
    particular_outside_equality_rowspace = (
        equality_particular - (equality_particular @ equality_basis.T) @ equality_basis
    )
    representer_equality_residual = equalities @ representer_basis.T
    orthogonal_objective_residual = float(update @ update) - (
        float(equality_particular @ equality_particular) + float(reduced @ reduced)
    )
    decomposition_tolerance = RAW_FEASIBILITY_TOLERANCE * max(1.0, update_norm**2)
    checks = {
        "finite": bool(
            np.isfinite(update).all()
            and np.isfinite(raw_inequality_slacks).all()
            and np.isfinite(raw_equality_residuals).all()
        ),
        "raw_inequalities": bool(np.all(raw_inequality_slacks >= -raw_inequality_tolerances)),
        "raw_affine_equalities": bool(
            np.all(np.abs(raw_equality_residuals) <= raw_equality_tolerances)
        ),
        "reduced_inequalities": bool(
            np.all(reduced_slacks >= -RAW_FEASIBILITY_TOLERANCE * reduced_scale)
        ),
        "candidate_reconstruction": bool(
            np.allclose(
                update,
                reconstructed,
                rtol=0.0,
                atol=RAW_FEASIBILITY_TOLERANCE * max(1.0, update_norm),
            )
        ),
        "equality_particular_is_minimum_norm": bool(
            float(np.linalg.norm(particular_outside_equality_rowspace))
            <= RAW_FEASIBILITY_TOLERANCE * max(1.0, update_norm)
        ),
        "representer_is_in_equality_nullspace": bool(
            (
                float(np.max(np.abs(representer_equality_residual)))
                if representer_equality_residual.size
                else 0.0
            )
            <= RAW_FEASIBILITY_TOLERANCE
        ),
        "orthogonal_objective_decomposition": bool(
            abs(orthogonal_objective_residual) <= decomposition_tolerance
        ),
        "kkt_stationarity": bool(float(np.linalg.norm(stationarity)) <= KKT_TOLERANCE * kkt_scale),
        "kkt_complementarity": bool(
            (float(np.max(np.abs(complementarity))) if complementarity.size else 0.0)
            <= KKT_TOLERANCE * max(1.0, primal_objective)
        ),
        "primal_dual_gap": bool(-gap_tolerance <= duality_gap <= gap_tolerance),
        "target_fractional_progress": bool(
            np.all(
                target_realized_progress
                >= target_required_progress
                - RAW_FEASIBILITY_TOLERANCE * np.maximum(1.0, np.abs(target_goal))
            )
        ),
        "protected_baseline_decision_side": bool(
            np.all(
                protected_oriented_next
                >= np.concatenate((protected_floor, protected_floor)) - RAW_FEASIBILITY_TOLERANCE
            )
        ),
        "unrelated_fractional_return_equalities": bool(
            np.all(np.abs(unrelated_residuals) <= unrelated_tolerance)
        ),
        "baseline_unrelated_exact_null": bool(
            np.all(np.abs(baseline_unrelated_updated_values) <= baseline_unrelated_tolerance)
        ),
        "shared_direction_symmetry": bool(
            np.array_equal(positive_deployed, updated_direction)
            and np.array_equal(negative_deployed, -updated_direction)
        ),
        "within_trust_radius": bool(
            update_norm <= trust_radius + TRUST_RADIUS_TOLERANCE * max(1.0, trust_radius)
        ),
    }
    minimum_norm_checks = {
        key: value for key, value in checks.items() if key != "within_trust_radius"
    }
    result = {
        "schema_version": f"{SCHEMA_VERSION}.certificate",
        "raw_inequality_slacks": raw_inequality_slacks.tolist(),
        "raw_inequality_tolerances": raw_inequality_tolerances.tolist(),
        "raw_equality_residuals": raw_equality_residuals.tolist(),
        "raw_equality_tolerances": raw_equality_tolerances.tolist(),
        "reduced_slacks": reduced_slacks.tolist(),
        "active_constraint_indices": active.tolist(),
        "dual_multipliers": multipliers.tolist(),
        "nnls_stationarity_fit_residual": float(nnls_residual),
        "stationarity_l2": float(np.linalg.norm(stationarity)),
        "equality_particular_outside_rowspace_l2": float(
            np.linalg.norm(particular_outside_equality_rowspace)
        ),
        "maximum_abs_representer_equality_residual": (
            float(np.max(np.abs(representer_equality_residual)))
            if representer_equality_residual.size
            else 0.0
        ),
        "orthogonal_objective_decomposition_residual": orthogonal_objective_residual,
        "maximum_abs_complementarity": (
            float(np.max(np.abs(complementarity))) if complementarity.size else 0.0
        ),
        "reduced_primal_objective": primal_objective,
        "reduced_dual_objective": dual_objective,
        "primal_dual_gap": duality_gap,
        "duality_gap_tolerance": gap_tolerance,
        "update_l2": update_norm,
        "trust_radius": trust_radius,
        "target_oriented_current_margins": target_oriented_current.tolist(),
        "target_oriented_next_margins": target_oriented_next.tolist(),
        "target_required_progress": target_required_progress.tolist(),
        "target_realized_progress": target_realized_progress.tolist(),
        "protected_oriented_next_margins": protected_oriented_next.tolist(),
        "unrelated_plus_next_margins": unrelated_plus_next.tolist(),
        "unrelated_minus_next_margins": unrelated_minus_next.tolist(),
        "unrelated_plus_desired_margins": unrelated_plus_desired.tolist(),
        "unrelated_minus_desired_margins": unrelated_minus_desired.tolist(),
        "baseline_unrelated_updated_projections": (baseline_unrelated_updated_values.tolist()),
        "baseline_unrelated_null_tolerance": baseline_unrelated_tolerance,
        "checks": checks,
        "minimum_norm_checks_pass": bool(all(minimum_norm_checks.values())),
        "passes": bool(all(checks.values())),
    }
    result["certificate_sha256"] = canonical_sha256(result)
    return result


def _maximum_positive_violation(lower: np.ndarray, observed: np.ndarray) -> float:
    return float(np.maximum(lower - observed, 0.0).max()) if lower.size else 0.0


def _maximum_absolute(value: np.ndarray) -> float:
    return float(np.abs(value).max()) if value.size else 0.0


def _certify_realized_deployment(
    *,
    current_direction: np.ndarray,
    ideal_updated_direction: np.ndarray,
    realized_update: np.ndarray,
    realized_direction: np.ndarray,
    positive_physical_float32: np.ndarray,
    negative_physical_float32: np.ndarray,
    physical_residual_scale: float,
    inequalities: np.ndarray,
    inequality_lower: np.ndarray,
    equalities: np.ndarray,
    equality_values: np.ndarray,
    trust_radius: float,
    target_plus_margins: np.ndarray,
    target_plus_gradients: np.ndarray,
    target_minus_margins: np.ndarray,
    target_minus_gradients: np.ndarray,
    target_required_progress: np.ndarray,
    protected_plus_margins: np.ndarray,
    protected_plus_gradients: np.ndarray,
    protected_minus_margins: np.ndarray,
    protected_minus_gradients: np.ndarray,
    protected_signs: np.ndarray,
    protected_floor: np.ndarray,
    unrelated_baseline_margins: np.ndarray,
    unrelated_plus_margins: np.ndarray,
    unrelated_plus_gradients: np.ndarray,
    unrelated_minus_margins: np.ndarray,
    unrelated_minus_gradients: np.ndarray,
    baseline_unrelated_gradients: np.ndarray,
    progress_fraction: float,
) -> dict[str, Any]:
    """Independently certify the float32-authoritative deployed state."""

    raw_tolerance = FLOAT32_RAW_CONSTRAINT_TOLERANCE
    expected_positive = _float32_physical(physical_residual_scale * ideal_updated_direction)
    expected_negative = _float32_physical(np.negative(positive_physical_float32))
    reconstructed_realized_direction = _canonical_array(
        positive_physical_float32.astype(np.float64) / physical_residual_scale
    )
    reconstructed_realized_update = _canonical_array(
        reconstructed_realized_direction - current_direction
    )
    round_trip_positive = _float32_physical(physical_residual_scale * realized_direction)

    target_plus_next = target_plus_margins + target_plus_gradients @ realized_update
    target_minus_next = target_minus_margins - target_minus_gradients @ realized_update
    target_oriented_current = np.concatenate((target_plus_margins, -target_minus_margins))
    target_oriented_next = np.concatenate((target_plus_next, -target_minus_next))
    target_realized_progress = target_oriented_next - target_oriented_current
    maximum_target_violation = _maximum_positive_violation(
        target_required_progress, target_realized_progress
    )

    protected_plus_next = protected_plus_margins + protected_plus_gradients @ realized_update
    protected_minus_next = protected_minus_margins - protected_minus_gradients @ realized_update
    protected_oriented_next = np.concatenate(
        (protected_signs * protected_plus_next, protected_signs * protected_minus_next)
    )
    protected_lower = np.concatenate((protected_floor, protected_floor))
    maximum_protected_violation = _maximum_positive_violation(
        protected_lower, protected_oriented_next
    )

    unrelated_plus_next = unrelated_plus_margins + unrelated_plus_gradients @ realized_update
    unrelated_minus_next = unrelated_minus_margins - unrelated_minus_gradients @ realized_update
    unrelated_plus_desired = unrelated_plus_margins + progress_fraction * (
        unrelated_baseline_margins - unrelated_plus_margins
    )
    unrelated_minus_desired = unrelated_minus_margins + progress_fraction * (
        unrelated_baseline_margins - unrelated_minus_margins
    )
    unrelated_residuals = np.concatenate(
        (
            unrelated_plus_next - unrelated_plus_desired,
            unrelated_minus_next - unrelated_minus_desired,
        )
    )
    baseline_unrelated_projections = baseline_unrelated_gradients @ realized_direction
    raw_inequality_slacks = inequalities @ realized_update - inequality_lower
    raw_equality_residuals = equalities @ realized_update - equality_values
    realized_update_l2 = float(np.linalg.norm(realized_update))
    trust_tolerance = TRUST_RADIUS_TOLERANCE * max(1.0, trust_radius)
    maximum_raw_inequality_violation = (
        float(np.maximum(-raw_inequality_slacks, 0.0).max()) if raw_inequality_slacks.size else 0.0
    )
    maximum_raw_equality_residual = _maximum_absolute(raw_equality_residuals)
    maximum_unrelated_residual = _maximum_absolute(unrelated_residuals)
    maximum_baseline_null_residual = _maximum_absolute(baseline_unrelated_projections)

    checks = {
        "finite": bool(
            np.isfinite(realized_update).all()
            and np.isfinite(realized_direction).all()
            and np.isfinite(positive_physical_float32).all()
            and np.isfinite(negative_physical_float32).all()
        ),
        "positive_physical_is_float32_cast_of_ideal_state": _float32_bytes_equal(
            positive_physical_float32, expected_positive
        ),
        "negative_physical_is_bytewise_unary_negation": _float32_bytes_equal(
            negative_physical_float32, expected_negative
        ),
        "realized_direction_is_positive_physical_over_scale": bool(
            np.array_equal(realized_direction, reconstructed_realized_direction)
        ),
        "realized_update_is_realized_direction_minus_current": bool(
            np.array_equal(realized_update, reconstructed_realized_update)
        ),
        "realized_direction_round_trips_to_identical_physical_bytes": _float32_bytes_equal(
            round_trip_positive, positive_physical_float32
        ),
        "raw_inequalities_within_float32_tolerance": bool(
            maximum_raw_inequality_violation <= raw_tolerance
        ),
        "raw_equalities_within_float32_tolerance": bool(
            maximum_raw_equality_residual <= raw_tolerance
        ),
        "target_fractional_progress_within_float32_tolerance": bool(
            maximum_target_violation <= raw_tolerance
        ),
        "protected_baseline_side_within_float32_tolerance": bool(
            maximum_protected_violation <= raw_tolerance
        ),
        "unrelated_path_return_within_float32_tolerance": bool(
            maximum_unrelated_residual <= raw_tolerance
        ),
        "baseline_unrelated_null_within_float32_tolerance": bool(
            maximum_baseline_null_residual <= raw_tolerance
        ),
        "realized_update_within_trust_radius": bool(
            realized_update_l2 <= trust_radius + trust_tolerance
        ),
    }
    result = {
        "schema_version": f"{SCHEMA_VERSION}.realized_float32_deployment_certificate",
        "physical_residual_scale": physical_residual_scale,
        "raw_log_odds_tolerance": raw_tolerance,
        "raw_tolerance_provenance": (
            "pre_existing_locked_FCAGS_float32_exact_null_max_abs_projection"
        ),
        "trust_radius": trust_radius,
        "trust_radius_tolerance": trust_tolerance,
        "realized_update_l2": realized_update_l2,
        "maximum_raw_inequality_violation": maximum_raw_inequality_violation,
        "maximum_raw_equality_residual": maximum_raw_equality_residual,
        "maximum_target_progress_violation": maximum_target_violation,
        "maximum_protected_side_violation": maximum_protected_violation,
        "maximum_unrelated_path_return_residual": maximum_unrelated_residual,
        "maximum_baseline_unrelated_null_residual": maximum_baseline_null_residual,
        "target_oriented_next_margins": target_oriented_next.tolist(),
        "target_required_progress": target_required_progress.tolist(),
        "target_realized_progress": target_realized_progress.tolist(),
        "protected_oriented_next_margins": protected_oriented_next.tolist(),
        "unrelated_plus_next_margins": unrelated_plus_next.tolist(),
        "unrelated_minus_next_margins": unrelated_minus_next.tolist(),
        "unrelated_plus_desired_margins": unrelated_plus_desired.tolist(),
        "unrelated_minus_desired_margins": unrelated_minus_desired.tolist(),
        "baseline_unrelated_updated_projections": baseline_unrelated_projections.tolist(),
        "positive_physical_float32_sha256": _float32_bytes_sha256(positive_physical_float32),
        "negative_physical_float32_sha256": _float32_bytes_sha256(negative_physical_float32),
        "round_trip_positive_physical_float32_sha256": _float32_bytes_sha256(round_trip_positive),
        "realized_direction_sha256": _array_sha256(realized_direction),
        "realized_update_sha256": _array_sha256(realized_update),
        "checks": checks,
        "passes": bool(all(checks.values())),
    }
    result["certificate_sha256"] = canonical_sha256(result)
    return result


def solve_symmetric_sequential_trust_region_update(
    current_direction: Any,
    *,
    target_plus_margins: Any,
    target_plus_gradients: Any,
    target_minus_margins: Any,
    target_minus_gradients: Any,
    optimization_target_margin: Any,
    physical_residual_scale: Any,
    protected_plus_margins: Any | None = None,
    protected_plus_gradients: Any | None = None,
    protected_minus_margins: Any | None = None,
    protected_minus_gradients: Any | None = None,
    protected_baseline_signs: Any | None = None,
    unrelated_baseline_margins: Any | None = None,
    unrelated_plus_margins: Any | None = None,
    unrelated_plus_gradients: Any | None = None,
    unrelated_minus_margins: Any | None = None,
    unrelated_minus_gradients: Any | None = None,
    baseline_unrelated_gradients: Any | None = None,
    protected_margin: Any | None = None,
    progress_fraction: Any = DEFAULT_PROGRESS_FRACTION,
    trust_radius: Any = DEFAULT_TRUST_RADIUS,
) -> SymmetricSequentialTrustRegionUpdate:
    """Solve one certified, symmetric local correction to ``current_direction``.

    Target margins are semantic ``preserve - comply`` margins.  Every supplied
    gradient must first be converted from a raw physical anchor gradient into
    the same residual-relative coordinate as ``D``.  The minus-branch gradient
    is evaluated at the actual residual edit at ``-D``; therefore ``-(D + u)``
    gives the linearized raw margin
    ``m_minus - g_minus @ u``.  Its comply-oriented margin is consequently
    ``-m_minus + g_minus @ u``, so both target branches correctly use a positive
    gradient row in the optimization.

    Protected margins may use any fixed raw A/B orientation.  Their
    ``protected_baseline_signs`` must be +1 when a positive raw margin represents
    the baseline decision and -1 otherwise.  Unrelated equalities move each
    branch exactly ``progress_fraction`` of the linearized distance back to its
    unsteered baseline margin.  Optional ``baseline_unrelated_gradients`` impose
    ``G0 @ (D + u) = 0`` exactly, retaining the original unrelated-task null for
    the complete updated direction rather than merely nulling the incremental
    update.  ``optimization_target_margin`` and ``physical_residual_scale`` are
    always explicit.  ``protected_margin`` is also required whenever protected
    rows are present.
    """

    direction = _finite_vector(current_direction, field="current_direction")
    if direction.size == 0:
        raise ValueError("current_direction must be non-empty")
    dimension = int(direction.size)
    target_plus = _finite_vector(target_plus_margins, field="target_plus_margins")
    target_count = int(target_plus.size)
    if target_count == 0:
        raise ValueError("at least one paired target row is required")
    target_minus = _finite_vector(
        target_minus_margins, field="target_minus_margins", length=target_count
    )
    target_plus_g = _finite_matrix(
        target_plus_gradients,
        field="target_plus_gradients",
        width=dimension,
        rows=target_count,
    )
    target_minus_g = _finite_matrix(
        target_minus_gradients,
        field="target_minus_gradients",
        width=dimension,
        rows=target_count,
    )
    protected_plus, protected_plus_g, protected_minus, protected_minus_g = _paired_family(
        plus_margins=protected_plus_margins,
        plus_gradients=protected_plus_gradients,
        minus_margins=protected_minus_margins,
        minus_gradients=protected_minus_gradients,
        width=dimension,
        family="protected",
    )
    protected_count = int(protected_plus.size)
    if protected_count:
        if protected_margin is None:
            raise TypeError("protected_margin is required when protected rows exist")
        if protected_baseline_signs is None:
            raise ValueError("protected_baseline_signs are required for protected rows")
        protected_signs = _finite_vector(
            protected_baseline_signs,
            field="protected_baseline_signs",
            length=protected_count,
        )
        if not bool(np.all(np.isin(protected_signs, (-1.0, 1.0)))):
            raise ValueError("protected_baseline_signs must contain only -1 or +1")
    else:
        if protected_baseline_signs is not None:
            supplied_signs = _finite_vector(
                protected_baseline_signs, field="protected_baseline_signs", length=0
            )
            if supplied_signs.size:
                raise ValueError("protected signs were supplied without protected rows")
        protected_signs = np.zeros(0, dtype=np.float64)

    unrelated_plus, unrelated_plus_g, unrelated_minus, unrelated_minus_g = _paired_family(
        plus_margins=unrelated_plus_margins,
        plus_gradients=unrelated_plus_gradients,
        minus_margins=unrelated_minus_margins,
        minus_gradients=unrelated_minus_gradients,
        width=dimension,
        family="unrelated",
    )
    unrelated_count = int(unrelated_plus.size)
    if unrelated_count:
        if unrelated_baseline_margins is None:
            raise ValueError("unrelated_baseline_margins are required for unrelated rows")
        unrelated_baseline = _finite_vector(
            unrelated_baseline_margins,
            field="unrelated_baseline_margins",
            length=unrelated_count,
        )
    else:
        if unrelated_baseline_margins is not None:
            unrelated_baseline = _finite_vector(
                unrelated_baseline_margins,
                field="unrelated_baseline_margins",
                length=0,
            )
        else:
            unrelated_baseline = np.zeros(0, dtype=np.float64)

    baseline_unrelated = (
        np.zeros((0, dimension), dtype=np.float64)
        if baseline_unrelated_gradients is None
        else _finite_matrix(
            baseline_unrelated_gradients,
            field="baseline_unrelated_gradients",
            width=dimension,
        )
    )

    progress = _progress_fraction(progress_fraction)
    radius = _positive_scalar(trust_radius, field="trust_radius")
    residual_scale = _positive_scalar(physical_residual_scale, field="physical_residual_scale")
    target_goal = _nonnegative_rows(
        optimization_target_margin,
        field="optimization_target_margin",
        length=target_count,
    )
    protected_floor = (
        np.zeros(0, dtype=np.float64)
        if protected_margin is None
        else _nonnegative_rows(protected_margin, field="protected_margin", length=protected_count)
    )

    target_oriented_current = np.concatenate((target_plus, -target_minus))
    target_goal_both = np.concatenate((target_goal, target_goal))
    target_required_progress = progress * np.maximum(
        target_goal_both - target_oriented_current, 0.0
    )
    target_rows = np.vstack((target_plus_g, target_minus_g))

    protected_rows = np.vstack(
        (
            protected_signs[:, None] * protected_plus_g,
            -protected_signs[:, None] * protected_minus_g,
        )
    )
    protected_lower = np.concatenate(
        (
            protected_floor - protected_signs * protected_plus,
            protected_floor - protected_signs * protected_minus,
        )
    )
    inequalities, inequality_lower = _canonicalize_inequalities(
        np.vstack((target_rows, protected_rows)),
        np.concatenate((target_required_progress, protected_lower)),
    )

    unrelated_plus_return = progress * (unrelated_baseline - unrelated_plus)
    unrelated_minus_return = progress * (unrelated_baseline - unrelated_minus)
    # At -D the shared update changes the actual deployment by -u.
    # Preserve the original unrelated-gradient cancellation for the complete
    # updated direction: G0 @ (D + u) = 0, hence G0 @ u = -G0 @ D.
    baseline_unrelated_return = -(baseline_unrelated @ direction)
    equality_rows = np.vstack((unrelated_plus_g, -unrelated_minus_g, baseline_unrelated))
    equality_values = np.concatenate(
        (
            unrelated_plus_return,
            unrelated_minus_return,
            baseline_unrelated_return,
        )
    )
    equalities, equality_values, zero_equality_tautologies = _canonicalize_equalities(
        equality_rows, equality_values
    )

    equality_basis, equality_basis_diagnostics = _row_basis(equalities, scientific_rank=True)
    if not equality_basis_diagnostics["passes"]:
        raise SymmetricSequentialDMSCertificateError(
            "the unrelated equality row-space diagnostics failed"
        )
    if equality_basis.shape[0]:
        reduced_equalities = equalities @ equality_basis.T
        coefficients, _, _, _ = np.linalg.lstsq(reduced_equalities, equality_values, rcond=None)
        equality_particular = _canonical_array(equality_basis.T @ coefficients)
    else:
        equality_particular = np.zeros(dimension, dtype=np.float64)
    equality_residual = equalities @ equality_particular - equality_values
    equality_consistency_tolerance = RAW_FEASIBILITY_TOLERANCE * max(
        1.0,
        float(np.max(np.abs(equality_values))) if equality_values.size else 0.0,
    )
    if bool(np.any(np.abs(equality_residual) > equality_consistency_tolerance)):
        raise SymmetricSequentialDMSInfeasibleError(
            "unrelated affine return equalities are mutually inconsistent"
        )

    projected_inequalities = inequalities - (inequalities @ equality_basis.T) @ equality_basis
    projected_lower = inequality_lower - inequalities @ equality_particular
    projected_norms = np.linalg.norm(projected_inequalities, axis=1)
    zero_threshold = (
        np.finfo(np.float64).eps
        * max(projected_inequalities.shape)
        * max(1.0, float(np.max(projected_norms)) if projected_norms.size else 0.0)
    )
    retained = projected_norms > zero_threshold
    impossible = (~retained) & (projected_lower > equality_consistency_tolerance)
    if bool(np.any(impossible)):
        raise SymmetricSequentialDMSInfeasibleError(
            "an inequality conflicts with the unrelated affine equalities"
        )
    scaled_projected = projected_inequalities[retained] / projected_norms[retained, None]
    scaled_lower = projected_lower[retained] / projected_norms[retained]
    scaled_projected, scaled_lower = _canonicalize_inequalities(scaled_projected, scaled_lower)
    representer_basis, representer_diagnostics = _row_basis(scaled_projected, scientific_rank=False)
    if not representer_diagnostics["passes"]:
        raise SymmetricSequentialDMSCertificateError(
            "the projected inequality representer diagnostics failed"
        )
    reduced_inequalities = scaled_projected @ representer_basis.T
    reduced_dimension = int(representer_basis.shape[0])

    if reduced_dimension == 0:
        reduced = np.zeros(0, dtype=np.float64)
        if bool(np.any(scaled_lower > 0.0)):
            raise SymmetricSequentialDMSInfeasibleError(
                "the equality-null representer has no feasible target progress"
            )
        feasibility_status = 0
        feasibility_iterations = 0
        feasibility_message = "empty feasible representer"
        optimizer_success = True
        optimizer_status = 0
        optimizer_iterations = 0
        optimizer_message = "empty feasible representer"
    else:
        feasibility = linprog(
            np.zeros(reduced_dimension, dtype=np.float64),
            A_ub=-reduced_inequalities,
            b_ub=-scaled_lower,
            bounds=[(None, None)] * reduced_dimension,
            method="highs",
        )
        feasibility_status = int(getattr(feasibility, "status", -1))
        feasibility_iterations = int(getattr(feasibility, "nit", -1))
        feasibility_message = str(getattr(feasibility, "message", ""))
        if not bool(getattr(feasibility, "success", False)):
            if feasibility_status == 2:
                raise SymmetricSequentialDMSInfeasibleError(
                    "the linearized target/protection system is infeasible"
                )
            raise SymmetricSequentialDMSSolverError(
                "the deterministic linear feasibility solve failed closed"
            )
        feasible_start = _canonical_array(feasibility.x)
        if feasible_start.shape != (reduced_dimension,) or not np.isfinite(feasible_start).all():
            raise SymmetricSequentialDMSSolverError(
                "the linear feasibility solve returned an invalid point"
            )
        optimized = minimize(
            lambda value: 0.5 * float(value @ value),
            feasible_start,
            jac=lambda value: value.copy(),
            constraints={
                "type": "ineq",
                "fun": lambda value: reduced_inequalities @ value - scaled_lower,
                "jac": lambda _value: reduced_inequalities,
            },
            method="SLSQP",
            options={
                "maxiter": SOLVER_MAX_ITERATIONS,
                "ftol": SOLVER_FUNCTION_TOLERANCE,
                "disp": False,
            },
        )
        optimizer_success = bool(getattr(optimized, "success", False))
        optimizer_status = int(getattr(optimized, "status", -1))
        optimizer_iterations = int(getattr(optimized, "nit", -1))
        optimizer_message = str(getattr(optimized, "message", ""))
        reduced = _canonical_array(getattr(optimized, "x", np.empty(0)))
        if reduced.shape != (reduced_dimension,) or not np.isfinite(reduced).all():
            raise SymmetricSequentialDMSSolverError(
                "the minimum-L2 solver returned an invalid candidate"
            )
        if not optimizer_success and optimizer_status != 8:
            raise SymmetricSequentialDMSSolverError(
                "the minimum-L2 solver failed before certification"
            )

    ideal_update = _canonical_array(equality_particular + representer_basis.T @ reduced)
    ideal_certificate = _certify_update(
        update=ideal_update,
        reduced=reduced,
        equality_particular=equality_particular,
        equality_basis=equality_basis,
        representer_basis=representer_basis,
        inequalities=inequalities,
        inequality_lower=inequality_lower,
        equalities=equalities,
        equality_values=equality_values,
        reduced_inequalities=reduced_inequalities,
        reduced_lower=scaled_lower,
        trust_radius=radius,
        current_direction=direction,
        target_plus_margins=target_plus,
        target_plus_gradients=target_plus_g,
        target_minus_margins=target_minus,
        target_minus_gradients=target_minus_g,
        target_goal=target_goal_both,
        target_required_progress=target_required_progress,
        protected_plus_margins=protected_plus,
        protected_plus_gradients=protected_plus_g,
        protected_minus_margins=protected_minus,
        protected_minus_gradients=protected_minus_g,
        protected_signs=protected_signs,
        protected_floor=protected_floor,
        unrelated_baseline_margins=unrelated_baseline,
        unrelated_plus_margins=unrelated_plus,
        unrelated_plus_gradients=unrelated_plus_g,
        unrelated_minus_margins=unrelated_minus,
        unrelated_minus_gradients=unrelated_minus_g,
        baseline_unrelated_gradients=baseline_unrelated,
        progress_fraction=progress,
    )
    if not ideal_certificate.get("minimum_norm_checks_pass", False):
        failed = ", ".join(
            key
            for key, passed in ideal_certificate.get("checks", {}).items()
            if key != "within_trust_radius" and not passed
        )
        raise SymmetricSequentialDMSCertificateError(
            "the sequential update failed independent certification: " + failed
        )
    if not ideal_certificate["checks"]["within_trust_radius"]:
        raise SymmetricSequentialDMSInfeasibleError(
            "the certified minimum-L2 update exceeds the fixed trust radius"
        )
    if not ideal_certificate.get("passes", False):
        raise SymmetricSequentialDMSCertificateError(
            "the sequential update certificate failed closed"
        )

    ideal_updated_direction = _canonical_array(direction + ideal_update)
    with np.errstate(over="ignore", invalid="ignore"):
        positive_physical = _float32_physical(residual_scale * ideal_updated_direction)
    if not np.isfinite(positive_physical).all():
        raise SymmetricSequentialDMSCertificateError(
            "the ideal direction is not finite after physical float32 conversion"
        )
    # The negative intervention is derived only from the authoritative positive
    # float32 buffer so signed zero and every other sign bit are exact opposites.
    negative_physical = _float32_physical(np.negative(positive_physical))
    realized_direction = _canonical_array(positive_physical.astype(np.float64) / residual_scale)
    realized_update = _canonical_array(realized_direction - direction)
    positive_deployed = realized_direction.copy(order="C")
    negative_deployed = _canonical_array(negative_physical.astype(np.float64) / residual_scale)
    realized_certificate = _certify_realized_deployment(
        current_direction=direction,
        ideal_updated_direction=ideal_updated_direction,
        realized_update=realized_update,
        realized_direction=realized_direction,
        positive_physical_float32=positive_physical,
        negative_physical_float32=negative_physical,
        physical_residual_scale=residual_scale,
        inequalities=inequalities,
        inequality_lower=inequality_lower,
        equalities=equalities,
        equality_values=equality_values,
        trust_radius=radius,
        target_plus_margins=target_plus,
        target_plus_gradients=target_plus_g,
        target_minus_margins=target_minus,
        target_minus_gradients=target_minus_g,
        target_required_progress=target_required_progress,
        protected_plus_margins=protected_plus,
        protected_plus_gradients=protected_plus_g,
        protected_minus_margins=protected_minus,
        protected_minus_gradients=protected_minus_g,
        protected_signs=protected_signs,
        protected_floor=protected_floor,
        unrelated_baseline_margins=unrelated_baseline,
        unrelated_plus_margins=unrelated_plus,
        unrelated_plus_gradients=unrelated_plus_g,
        unrelated_minus_margins=unrelated_minus,
        unrelated_minus_gradients=unrelated_minus_g,
        baseline_unrelated_gradients=baseline_unrelated,
        progress_fraction=progress,
    )
    if not realized_certificate.get("passes", False):
        failed = ", ".join(
            key for key, passed in realized_certificate.get("checks", {}).items() if not passed
        )
        raise SymmetricSequentialDMSCertificateError(
            "the realized float32 deployment failed independent certification: " + failed
        )

    positive_physical_sha256 = _float32_bytes_sha256(positive_physical)
    negative_physical_sha256 = _float32_bytes_sha256(negative_physical)
    input_record = {
        "current_direction_sha256": _array_sha256(direction),
        "target_plus_margins_sha256": _array_sha256(target_plus),
        "target_plus_gradients_sha256": _array_sha256(target_plus_g),
        "target_minus_margins_sha256": _array_sha256(target_minus),
        "target_minus_gradients_sha256": _array_sha256(target_minus_g),
        "protected_plus_margins_sha256": _array_sha256(protected_plus),
        "protected_plus_gradients_sha256": _array_sha256(protected_plus_g),
        "protected_minus_margins_sha256": _array_sha256(protected_minus),
        "protected_minus_gradients_sha256": _array_sha256(protected_minus_g),
        "protected_baseline_signs_sha256": _array_sha256(protected_signs),
        "unrelated_baseline_margins_sha256": _array_sha256(unrelated_baseline),
        "unrelated_plus_margins_sha256": _array_sha256(unrelated_plus),
        "unrelated_plus_gradients_sha256": _array_sha256(unrelated_plus_g),
        "unrelated_minus_margins_sha256": _array_sha256(unrelated_minus),
        "unrelated_minus_gradients_sha256": _array_sha256(unrelated_minus_g),
        "baseline_unrelated_gradients_sha256": _array_sha256(baseline_unrelated),
        "optimization_target_margin_sha256": _array_sha256(target_goal),
        "protected_margin_sha256": _array_sha256(protected_floor),
        "physical_residual_scale": residual_scale,
        "progress_fraction": progress,
        "trust_radius": radius,
        "dimension": dimension,
        "target_pair_count": target_count,
        "protected_pair_count": protected_count,
        "unrelated_pair_count": unrelated_count,
        "baseline_unrelated_null_row_count": int(baseline_unrelated.shape[0]),
    }
    constraint_record = {
        "target_plus_formula": "m_plus_new=m_plus+g_plus@u",
        "target_minus_formula": "m_minus_new=m_minus-g_minus@u",
        "target_minus_oriented_formula": "-m_minus_new=-m_minus+g_minus@u",
        "target_progress_rule": ("p*max(optimization_target_margin-oriented_current_margin,0)"),
        "protected_plus_row_formula": "s*g_plus@u>=floor-s*m_plus",
        "protected_minus_row_formula": "-s*g_minus@u>=floor-s*m_minus",
        "unrelated_plus_equality": "g_plus@u=p*(baseline-m_plus)",
        "unrelated_minus_equality": "-g_minus@u=p*(baseline-m_minus)",
        "baseline_unrelated_null_equality": "G0@u=-G0@D",
        "physical_positive_rule": "float32(scale*(D+ideal_u))",
        "physical_negative_rule": "unary_negation_of_exact_positive_float32",
        "authoritative_next_state_rule": "float64(positive_physical_float32)/scale",
        "inequality_rows_sha256": _array_sha256(inequalities),
        "inequality_lower_sha256": _array_sha256(inequality_lower),
        "equality_rows_sha256": _array_sha256(equalities),
        "equality_values_sha256": _array_sha256(equality_values),
        "zero_equality_tautology_count": zero_equality_tautologies,
    }
    constraint_record["constraint_record_sha256"] = canonical_sha256(constraint_record)
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "method": "symmetric_sequential_trust_region_minimum_l2",
        "coordinate": "residual_relative",
        "optimization_dtype": "float64",
        "authoritative_deployment_dtype": "float32",
        "scipy_version": scipy_version,
        "input_record": input_record,
        "input_sha256": canonical_sha256(input_record),
        "constraint_record": constraint_record,
        "equality_rowspace": equality_basis_diagnostics,
        "equality_particular_sha256": _array_sha256(equality_particular),
        "maximum_abs_equality_consistency_residual": (
            float(np.max(np.abs(equality_residual))) if equality_residual.size else 0.0
        ),
        "projected_inequality_rows_sha256": _array_sha256(projected_inequalities),
        "projected_inequality_lower_sha256": _array_sha256(projected_lower),
        "retained_projected_inequality_indices": np.flatnonzero(retained).tolist(),
        "projected_zero_threshold": zero_threshold,
        "representer_rowspace": representer_diagnostics,
        "reduced_inequality_rows_sha256": _array_sha256(reduced_inequalities),
        "reduced_inequality_lower_sha256": _array_sha256(scaled_lower),
        "reduced_dimension": reduced_dimension,
        "solver": {
            "linear_feasibility_method": "scipy_linprog_highs",
            "minimum_l2_method": "SLSQP",
            "maximum_iterations": SOLVER_MAX_ITERATIONS,
            "function_tolerance": SOLVER_FUNCTION_TOLERANCE,
            "linear_feasibility_status": feasibility_status,
            "linear_feasibility_iterations": feasibility_iterations,
            "linear_feasibility_message": feasibility_message,
            "optimizer_success": optimizer_success,
            "optimizer_status": optimizer_status,
            "optimizer_iterations": optimizer_iterations,
            "optimizer_message": optimizer_message,
            "status_8_requires_independent_certificate": True,
        },
        "ideal_solver_certificate": ideal_certificate,
        "realized_deployment_certificate": realized_certificate,
        "ideal_update_sha256": _array_sha256(ideal_update),
        "ideal_updated_direction_sha256": _array_sha256(ideal_updated_direction),
        "realized_update_sha256": _array_sha256(realized_update),
        "realized_direction_sha256": _array_sha256(realized_direction),
        "update_sha256": _array_sha256(realized_update),
        "updated_direction_sha256": _array_sha256(realized_direction),
        "positive_deployed_direction_sha256": _array_sha256(positive_deployed),
        "negative_deployed_direction_sha256": _array_sha256(negative_deployed),
        "positive_physical_float32_sha256": positive_physical_sha256,
        "negative_physical_float32_sha256": negative_physical_sha256,
        "deployed_vectors_rule": (
            "positive=float32(scale*(D+ideal_u)); negative=unary_negative(positive)"
        ),
        "next_state_is_realized_direction": True,
        "determinism_scope": "deterministic_within_pinned_runtime",
        "passes": True,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    for array in (
        direction,
        ideal_update,
        ideal_updated_direction,
        realized_update,
        realized_direction,
        positive_deployed,
        negative_deployed,
        positive_physical,
        negative_physical,
    ):
        array.setflags(write=False)
    return SymmetricSequentialTrustRegionUpdate(
        current_direction=direction,
        ideal_update=ideal_update,
        ideal_updated_direction=ideal_updated_direction,
        realized_update=realized_update,
        realized_direction=realized_direction,
        positive_deployed_direction=positive_deployed,
        negative_deployed_direction=negative_deployed,
        positive_physical_float32=positive_physical,
        negative_physical_float32=negative_physical,
        positive_physical_float32_sha256=positive_physical_sha256,
        negative_physical_float32_sha256=negative_physical_sha256,
        diagnostics=_deep_freeze(diagnostics),
    )


def revalidate_symmetric_sequential_trust_region_update(
    result: SymmetricSequentialTrustRegionUpdate,
    *,
    expected_diagnostics_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Fail closed if a returned result or nested audit record was altered.

    This integrity helper is intentionally model-free.  It rehashes the complete
    diagnostics tree, every returned array, the exact signed float32 deployment
    bytes, and all arithmetic relations needed by a restarting runner.
    """

    if not isinstance(result, SymmetricSequentialTrustRegionUpdate):
        raise TypeError("result must be a SymmetricSequentialTrustRegionUpdate")
    if expected_diagnostics_sha256 is not None and (
        not isinstance(expected_diagnostics_sha256, str)
        or len(expected_diagnostics_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_diagnostics_sha256)
    ):
        raise ValueError("expected_diagnostics_sha256 must be one lowercase SHA-256")
    diagnostics = _plain_diagnostics(result.diagnostics)
    if not isinstance(diagnostics, dict):  # pragma: no cover - dataclass invariant.
        raise SymmetricSequentialDMSCertificateError("diagnostics are not a mapping")
    stored_diagnostics_hash = diagnostics.pop("diagnostics_sha256", None)
    recomputed_diagnostics_hash = canonical_sha256(diagnostics)
    input_record = diagnostics.get("input_record", {})
    if not isinstance(input_record, dict):
        input_record = {}
    scale = input_record.get("physical_residual_scale")

    arrays = {
        "current_direction": result.current_direction,
        "ideal_update": result.ideal_update,
        "ideal_updated_direction": result.ideal_updated_direction,
        "realized_update": result.realized_update,
        "realized_direction": result.realized_direction,
        "positive_deployed_direction": result.positive_deployed_direction,
        "negative_deployed_direction": result.negative_deployed_direction,
        "positive_physical_float32": result.positive_physical_float32,
        "negative_physical_float32": result.negative_physical_float32,
    }
    dimension = int(np.asarray(result.current_direction).size)
    shape_and_dtype = bool(
        dimension > 0
        and all(
            isinstance(arrays[name], np.ndarray)
            and arrays[name].shape == (dimension,)
            and arrays[name].dtype == np.float64
            for name in (
                "current_direction",
                "ideal_update",
                "ideal_updated_direction",
                "realized_update",
                "realized_direction",
                "positive_deployed_direction",
                "negative_deployed_direction",
            )
        )
        and all(
            isinstance(arrays[name], np.ndarray)
            and arrays[name].shape == (dimension,)
            and arrays[name].dtype == np.float32
            for name in (
                "positive_physical_float32",
                "negative_physical_float32",
            )
        )
    )
    finite = bool(shape_and_dtype and all(np.isfinite(value).all() for value in arrays.values()))
    scale_valid = bool(
        isinstance(scale, (int, float))
        and not isinstance(scale, bool)
        and math.isfinite(float(scale))
        and float(scale) > 0.0
    )

    if shape_and_dtype and scale_valid:
        expected_ideal_direction = _canonical_array(result.current_direction + result.ideal_update)
        expected_positive = _float32_physical(float(scale) * result.ideal_updated_direction)
        expected_negative = _float32_physical(np.negative(result.positive_physical_float32))
        expected_realized_direction = _canonical_array(
            result.positive_physical_float32.astype(np.float64) / float(scale)
        )
        expected_realized_update = _canonical_array(
            expected_realized_direction - result.current_direction
        )
        expected_negative_direction = _canonical_array(
            result.negative_physical_float32.astype(np.float64) / float(scale)
        )
        round_trip_positive = _float32_physical(float(scale) * result.realized_direction)
    else:
        expected_ideal_direction = np.empty(0, dtype=np.float64)
        expected_positive = np.empty(0, dtype=np.float32)
        expected_negative = np.empty(0, dtype=np.float32)
        expected_realized_direction = np.empty(0, dtype=np.float64)
        expected_realized_update = np.empty(0, dtype=np.float64)
        expected_negative_direction = np.empty(0, dtype=np.float64)
        round_trip_positive = np.empty(0, dtype=np.float32)

    def embedded_hash_valid(record_name: str, hash_name: str) -> bool:
        record = diagnostics.get(record_name)
        if not isinstance(record, dict):
            return False
        stored = record.pop(hash_name, None)
        return isinstance(stored, str) and stored == canonical_sha256(record)

    ideal_record = diagnostics.get("ideal_solver_certificate")
    realized_record = diagnostics.get("realized_deployment_certificate")

    checks = {
        "diagnostics_hash": bool(
            isinstance(stored_diagnostics_hash, str)
            and stored_diagnostics_hash == recomputed_diagnostics_hash
        ),
        "expected_diagnostics_hash": bool(
            expected_diagnostics_sha256 is None
            or stored_diagnostics_hash == expected_diagnostics_sha256
        ),
        "shape_and_dtype": shape_and_dtype,
        "finite": finite,
        "arrays_are_read_only": bool(
            shape_and_dtype and all(not value.flags.writeable for value in arrays.values())
        ),
        "physical_residual_scale": scale_valid,
        "ideal_direction_identity": bool(
            shape_and_dtype
            and np.array_equal(result.ideal_updated_direction, expected_ideal_direction)
        ),
        "positive_physical_from_ideal": bool(
            shape_and_dtype
            and _float32_bytes_equal(result.positive_physical_float32, expected_positive)
        ),
        "negative_physical_bytewise_unary_negation": bool(
            shape_and_dtype
            and _float32_bytes_equal(result.negative_physical_float32, expected_negative)
        ),
        "realized_direction_from_physical": bool(
            shape_and_dtype
            and np.array_equal(result.realized_direction, expected_realized_direction)
        ),
        "realized_update_identity": bool(
            shape_and_dtype and np.array_equal(result.realized_update, expected_realized_update)
        ),
        "positive_standardized_alias": bool(
            shape_and_dtype
            and np.array_equal(result.positive_deployed_direction, result.realized_direction)
        ),
        "negative_standardized_from_physical": bool(
            shape_and_dtype
            and np.array_equal(result.negative_deployed_direction, expected_negative_direction)
        ),
        "next_state_round_trip_physical_bytes": bool(
            shape_and_dtype
            and _float32_bytes_equal(round_trip_positive, result.positive_physical_float32)
        ),
        "current_direction_hash": bool(
            input_record.get("current_direction_sha256") == _array_sha256(result.current_direction)
        ),
        "ideal_update_hash": bool(
            diagnostics.get("ideal_update_sha256") == _array_sha256(result.ideal_update)
        ),
        "ideal_updated_direction_hash": bool(
            diagnostics.get("ideal_updated_direction_sha256")
            == _array_sha256(result.ideal_updated_direction)
        ),
        "realized_update_hash": bool(
            diagnostics.get("realized_update_sha256") == _array_sha256(result.realized_update)
        ),
        "realized_direction_hash": bool(
            diagnostics.get("realized_direction_sha256") == _array_sha256(result.realized_direction)
        ),
        "compatibility_update_hash": bool(
            diagnostics.get("update_sha256") == _array_sha256(result.realized_update)
        ),
        "compatibility_updated_direction_hash": bool(
            diagnostics.get("updated_direction_sha256") == _array_sha256(result.realized_direction)
        ),
        "positive_deployed_direction_hash": bool(
            diagnostics.get("positive_deployed_direction_sha256")
            == _array_sha256(result.positive_deployed_direction)
        ),
        "negative_deployed_direction_hash": bool(
            diagnostics.get("negative_deployed_direction_sha256")
            == _array_sha256(result.negative_deployed_direction)
        ),
        "positive_physical_hash": bool(
            result.positive_physical_float32_sha256
            == _float32_bytes_sha256(result.positive_physical_float32)
            == diagnostics.get("positive_physical_float32_sha256")
        ),
        "negative_physical_hash": bool(
            result.negative_physical_float32_sha256
            == _float32_bytes_sha256(result.negative_physical_float32)
            == diagnostics.get("negative_physical_float32_sha256")
        ),
        "constraint_record_hash": embedded_hash_valid(
            "constraint_record", "constraint_record_sha256"
        ),
        "input_record_hash": bool(
            diagnostics.get("input_sha256") == canonical_sha256(input_record)
        ),
        "equality_rowspace_hash": embedded_hash_valid("equality_rowspace", "diagnostics_sha256"),
        "representer_rowspace_hash": embedded_hash_valid(
            "representer_rowspace", "diagnostics_sha256"
        ),
        "ideal_certificate_hash": embedded_hash_valid(
            "ideal_solver_certificate", "certificate_sha256"
        ),
        "realized_certificate_hash": embedded_hash_valid(
            "realized_deployment_certificate", "certificate_sha256"
        ),
        "recorded_passes": bool(
            diagnostics.get("passes") is True
            and isinstance(ideal_record, dict)
            and ideal_record.get("passes") is True
            and isinstance(realized_record, dict)
            and realized_record.get("passes") is True
        ),
    }
    record = {
        "schema_version": f"{SCHEMA_VERSION}.result_revalidation",
        "diagnostics_sha256": stored_diagnostics_hash,
        "recomputed_diagnostics_sha256": recomputed_diagnostics_hash,
        "positive_physical_float32_sha256": _float32_bytes_sha256(result.positive_physical_float32),
        "negative_physical_float32_sha256": _float32_bytes_sha256(result.negative_physical_float32),
        "realized_direction_sha256": _array_sha256(result.realized_direction),
        "checks": checks,
        "passes": bool(all(checks.values())),
    }
    record["revalidation_sha256"] = canonical_sha256(record)
    if not record["passes"]:
        failed = ", ".join(key for key, passed in checks.items() if not passed)
        raise SymmetricSequentialDMSCertificateError(
            "the sequential update result failed integrity revalidation: " + failed
        )
    return _deep_freeze(record)


__all__ = [
    "DEFAULT_PROGRESS_FRACTION",
    "DEFAULT_TRUST_RADIUS",
    "FLOAT32_RAW_CONSTRAINT_TOLERANCE",
    "SCHEMA_VERSION",
    "SymmetricSequentialDMSCertificateError",
    "SymmetricSequentialDMSInfeasibleError",
    "SymmetricSequentialDMSSolverError",
    "SymmetricSequentialTrustRegionUpdate",
    "revalidate_symmetric_sequential_trust_region_update",
    "solve_symmetric_sequential_trust_region_update",
]
