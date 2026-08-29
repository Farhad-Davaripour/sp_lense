"""Pure mathematics for Counterfactual KL-Extragradient Surgery (CKES).

CKES first constructs a certified common-ascent lookahead from every target
sign/order gradient after removing the exact unrelated-task row space.  A
separate model runtime measures full-vocabulary KL gradients at that nonzero
lookahead.  This module then finds the update closest to an already certified
nominal CL-DMS step while adding linearized matched-counterfactual KL limits.

The KL tangent is not a nonlinear upper bound.  A caller must measure and gate
actual KL after deployment; this module deliberately makes no claim otherwise.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import linprog, minimize

from .factorial_causal_anchor import canonical_sha256
from .symmetric_sequential_trust_region_dms import (
    SymmetricSequentialDMSCertificateError,
    SymmetricSequentialDMSInfeasibleError,
    SymmetricSequentialDMSSolverError,
    SymmetricSequentialTrustRegionUpdate,
    revalidate_symmetric_sequential_trust_region_update,
)

SCHEMA_VERSION = "sp_lense.counterfactual_kl_extragradient_surgery.v1"
LOOKAHEAD_EPSILON = 1.0 / 32.0
CELL_TANGENT_KL_LIMIT = 0.02
MEAN_TANGENT_KL_LIMIT = 0.005
SVD_RTOL = 1e-10
SVD_ATOL = 1e-12
RAW_EQUALITY_TOLERANCE = 2e-5
RAW_INEQUALITY_TOLERANCE = 2e-5
TRUST_TOLERANCE = 2e-6
COMMON_ASCENT_TOLERANCE = 1e-10
SOLVER_MAX_ITERATIONS = 2_000
SOLVER_FUNCTION_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CommonAscentLookahead:
    """One immutable nuisance-nulled target common-ascent direction."""

    direction: np.ndarray
    lookahead_direction: np.ndarray
    simplex_weights: np.ndarray
    nuisance_basis: np.ndarray
    diagnostics: Mapping[str, Any]

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-safe integrity record without mutable array aliases."""

        return {
            "direction": self.direction.tolist(),
            "lookahead_direction": self.lookahead_direction.tolist(),
            "simplex_weights": self.simplex_weights.tolist(),
            "nuisance_basis": self.nuisance_basis.tolist(),
            "diagnostics": _plain(self.diagnostics),
        }


@dataclass(frozen=True)
class CounterfactualKLESUpdate:
    """A centered-QP update with authoritative float32 deployment bytes."""

    current_direction: np.ndarray
    nominal_update: np.ndarray
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
        return self.realized_update

    @property
    def updated_direction(self) -> np.ndarray:
        return self.realized_direction

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-safe integrity record without mutable array aliases."""

        return {
            "current_direction": self.current_direction.tolist(),
            "nominal_update": self.nominal_update.tolist(),
            "ideal_update": self.ideal_update.tolist(),
            "ideal_updated_direction": self.ideal_updated_direction.tolist(),
            "realized_update": self.realized_update.tolist(),
            "realized_direction": self.realized_direction.tolist(),
            "positive_deployed_direction": self.positive_deployed_direction.tolist(),
            "negative_deployed_direction": self.negative_deployed_direction.tolist(),
            "positive_physical_float32": self.positive_physical_float32.tolist(),
            "negative_physical_float32": self.negative_physical_float32.tolist(),
            "positive_physical_float32_sha256": self.positive_physical_float32_sha256,
            "negative_physical_float32_sha256": self.negative_physical_float32_sha256,
            "diagnostics": _plain(self.diagnostics),
        }


def _array(value: Any, *, field: str, ndim: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{field} must contain real numbers")
    result = np.asarray(raw, dtype=np.float64, order="C").copy(order="C")
    result[result == 0.0] = 0.0
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{field} must have rank {ndim}")
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values")
    return result


def _vector(value: Any, *, field: str, length: int | None = None) -> np.ndarray:
    result = _array(value, field=field, ndim=1)
    if length is not None and result.shape != (length,):
        raise ValueError(f"{field} must contain exactly {length} values")
    return result


def _matrix(
    value: Any, *, field: str, width: int | None = None, rows: int | None = None
) -> np.ndarray:
    result = _array(value, field=field, ndim=2)
    if width is not None and result.shape[1] != width:
        raise ValueError(f"{field} has the wrong width")
    if rows is not None and result.shape[0] != rows:
        raise ValueError(f"{field} has the wrong row count")
    return result


def _positive(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _fraction(value: Any, *, field: str) -> float:
    result = _positive(value, field=field)
    if result > 1.0:
        raise ValueError(f"{field} must not exceed one")
    return result


def _array_sha256(value: Any) -> str:
    return canonical_sha256(_array(value, field="hash_value").tolist())


def _float32(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32, order="C").copy(order="C")


def _readonly(value: Any, *, dtype: Any, canonical_zero: bool = True) -> np.ndarray:
    result = np.asarray(value, dtype=dtype, order="C").copy(order="C")
    if canonical_zero:
        result[result == 0.0] = 0.0
    result.flags.writeable = False
    return result


def _float32_sha256(value: Any) -> str:
    return hashlib.sha256(_float32(value).tobytes(order="C")).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _orthonormal_row_basis(rows: np.ndarray, *, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    width = int(rows.shape[1])
    norms = np.linalg.norm(rows, axis=1)
    nonzero = norms > 0.0
    normalized = rows[nonzero] / norms[nonzero, None]
    if normalized.shape[0]:
        _, singular, vh = np.linalg.svd(normalized, full_matrices=False)
        threshold = max(SVD_ATOL, SVD_RTOL * float(singular[0]))
        rank = int(np.count_nonzero(singular > threshold))
        basis = vh[:rank].copy(order="C")
        for row in basis:
            anchor = int(np.argmax(np.abs(row)))
            if row[anchor] < 0.0:
                row *= -1.0
    else:
        singular = np.zeros(0, dtype=np.float64)
        threshold = SVD_ATOL
        basis = np.zeros((0, width), dtype=np.float64)
    residual = rows - (rows @ basis.T) @ basis
    orthogonality = basis @ basis.T - np.eye(basis.shape[0])
    maximum_residual = float(np.max(np.abs(residual))) if residual.size else 0.0
    maximum_orthogonality = (
        float(np.max(np.abs(orthogonality))) if orthogonality.size else 0.0
    )
    tolerance = max(SVD_ATOL, SVD_RTOL * max(1.0, float(norms.max()) if norms.size else 0.0))
    diagnostics = {
        "label": label,
        "input_rows": int(rows.shape[0]),
        "nonzero_rows": int(np.count_nonzero(nonzero)),
        "rank": int(basis.shape[0]),
        "singular_values": singular.tolist(),
        "threshold": threshold,
        "maximum_abs_reconstruction_residual": maximum_residual,
        "maximum_abs_orthonormality_residual": maximum_orthogonality,
        "passes": bool(maximum_residual <= tolerance and maximum_orthogonality <= 1e-10),
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return basis, diagnostics


def construct_common_ascent_lookahead(
    current_direction: Any,
    *,
    oriented_target_gradients: Any,
    baseline_unrelated_gradients: Any,
    epsilon: Any = LOOKAHEAD_EPSILON,
) -> CommonAscentLookahead:
    """Construct the fixed nuisance-nulled MGDA lookahead direction."""

    current = _vector(current_direction, field="current_direction")
    width = int(current.size)
    target = _matrix(
        oriented_target_gradients,
        field="oriented_target_gradients",
        width=width,
    )
    nuisance = _matrix(
        baseline_unrelated_gradients,
        field="baseline_unrelated_gradients",
        width=width,
    )
    if target.shape[0] < 2:
        raise ValueError("at least two target rows are required")
    step = _fraction(epsilon, field="epsilon")
    nuisance_basis, nuisance_diagnostics = _orthonormal_row_basis(
        nuisance, label="baseline_unrelated"
    )
    if not nuisance_diagnostics["passes"]:
        raise SymmetricSequentialDMSCertificateError("unrelated row-space certificate failed")
    projected = target - (target @ nuisance_basis.T) @ nuisance_basis
    current_nuisance_projection = nuisance @ current
    if current_nuisance_projection.size and float(
        np.max(np.abs(current_nuisance_projection))
    ) > RAW_EQUALITY_TOLERANCE:
        raise SymmetricSequentialDMSCertificateError(
            "current direction is outside the exact unrelated-task null"
        )
    projected_norms = np.linalg.norm(projected, axis=1)
    if bool(np.any(projected_norms <= COMMON_ASCENT_TOLERANCE)):
        raise SymmetricSequentialDMSInfeasibleError(
            "one target gradient vanishes after exact unrelated projection"
        )
    normalized = projected / projected_norms[:, None]
    gram = normalized @ normalized.T
    count = int(target.shape[0])

    def objective(alpha: np.ndarray) -> float:
        return 0.5 * float(alpha @ gram @ alpha)

    def objective_gradient(alpha: np.ndarray) -> np.ndarray:
        return gram @ alpha

    result = minimize(
        objective,
        np.full(count, 1.0 / count, dtype=np.float64),
        jac=objective_gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={
            "type": "eq",
            "fun": lambda alpha: float(alpha.sum() - 1.0),
            "jac": lambda alpha: np.ones_like(alpha),
        },
        options={"maxiter": SOLVER_MAX_ITERATIONS, "ftol": SOLVER_FUNCTION_TOLERANCE},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise SymmetricSequentialDMSSolverError(f"MGDA failed: {result.message}")
    weights = np.asarray(result.x, dtype=np.float64)
    combined = weights @ normalized
    combined_norm = float(np.linalg.norm(combined))
    if combined_norm <= COMMON_ASCENT_TOLERANCE:
        raise SymmetricSequentialDMSInfeasibleError("MGDA common-ascent vector collapses to zero")
    direction = combined / combined_norm
    target_dots = target @ direction
    nuisance_projection = nuisance @ direction
    simplex_residual = abs(float(weights.sum()) - 1.0)
    gram_times_weights = gram @ weights
    multiplier = float(weights @ gram_times_weights)
    active = weights > 1e-8
    active_kkt_error = (
        float(np.max(np.abs(gram_times_weights[active] - multiplier)))
        if bool(np.any(active))
        else math.inf
    )
    inactive_kkt_violation = (
        float(np.max(np.maximum(multiplier - gram_times_weights[~active], 0.0)))
        if bool(np.any(~active))
        else 0.0
    )
    checks = {
        "simplex": bool(simplex_residual <= 1e-9 and float(weights.min()) >= -1e-10),
        "simplex_kkt": bool(active_kkt_error <= 1e-7 and inactive_kkt_violation <= 1e-7),
        "unit_direction": bool(abs(float(np.linalg.norm(direction)) - 1.0) <= 1e-10),
        "strict_common_ascent": bool(float(target_dots.min()) > COMMON_ASCENT_TOLERANCE),
        "exact_unrelated_null": bool(
            float(np.max(np.abs(nuisance_projection))) <= RAW_EQUALITY_TOLERANCE
        ),
        "current_exact_unrelated_null": bool(
            not current_nuisance_projection.size
            or float(np.max(np.abs(current_nuisance_projection))) <= RAW_EQUALITY_TOLERANCE
        ),
    }
    if not all(checks.values()):
        raise SymmetricSequentialDMSCertificateError("MGDA common-ascent certificate failed")
    lookahead = current + step * direction
    diagnostics = {
        "schema_version": f"{SCHEMA_VERSION}.common_ascent",
        "epsilon": step,
        "scipy_version": scipy_version,
        "target_row_count": count,
        "current_direction_sha256": _array_sha256(current),
        "oriented_target_gradients_sha256": _array_sha256(target),
        "baseline_unrelated_gradients_sha256": _array_sha256(nuisance),
        "nuisance_basis": nuisance_diagnostics,
        "simplex_weights": weights.tolist(),
        "simplex_residual": simplex_residual,
        "simplex_kkt_multiplier": multiplier,
        "maximum_active_simplex_kkt_error": active_kkt_error,
        "maximum_inactive_simplex_kkt_violation": inactive_kkt_violation,
        "minimum_target_dot": float(target_dots.min()),
        "target_dots": target_dots.tolist(),
        "maximum_abs_unrelated_projection": float(np.max(np.abs(nuisance_projection))),
        "direction_sha256": _array_sha256(direction),
        "lookahead_direction_sha256": _array_sha256(lookahead),
        "checks": checks,
        "passes": bool(all(checks.values())),
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return CommonAscentLookahead(
        direction=_readonly(direction, dtype=np.float64),
        lookahead_direction=_readonly(lookahead, dtype=np.float64),
        simplex_weights=_readonly(weights, dtype=np.float64),
        nuisance_basis=_readonly(nuisance_basis, dtype=np.float64),
        diagnostics=_freeze(diagnostics),
    )


def _paired_family(
    *,
    plus_margins: Any,
    plus_gradients: Any,
    minus_margins: Any,
    minus_gradients: Any,
    width: int,
    family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    plus = _vector(plus_margins, field=f"{family}_plus_margins")
    minus = _vector(minus_margins, field=f"{family}_minus_margins", length=plus.size)
    plus_g = _matrix(
        plus_gradients, field=f"{family}_plus_gradients", width=width, rows=plus.size
    )
    minus_g = _matrix(
        minus_gradients, field=f"{family}_minus_gradients", width=width, rows=plus.size
    )
    return plus, plus_g, minus, minus_g


def _affine_equality_parameterization(
    equalities: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    width = int(equalities.shape[1])
    norms = np.linalg.norm(equalities, axis=1)
    nonzero = norms > 0.0
    if bool(np.any((~nonzero) & (np.abs(values) > RAW_EQUALITY_TOLERANCE))):
        raise SymmetricSequentialDMSInfeasibleError("a zero equality row has a nonzero value")
    rows = equalities[nonzero] / norms[nonzero, None]
    rhs = values[nonzero] / norms[nonzero]
    if rows.shape[0]:
        u, singular, vh = np.linalg.svd(rows, full_matrices=False)
        threshold = max(SVD_ATOL, SVD_RTOL * float(singular[0]))
        rank = int(np.count_nonzero(singular > threshold))
        basis = vh[:rank]
        coefficients = (u[:, :rank].T @ rhs) / singular[:rank]
        particular = basis.T @ coefficients
    else:
        singular = np.zeros(0, dtype=np.float64)
        threshold = SVD_ATOL
        basis = np.zeros((0, width), dtype=np.float64)
        particular = np.zeros(width, dtype=np.float64)
    residual = equalities @ particular - values
    maximum = float(np.max(np.abs(residual))) if residual.size else 0.0
    if maximum > RAW_EQUALITY_TOLERANCE:
        raise SymmetricSequentialDMSInfeasibleError("affine equalities are inconsistent")
    diagnostics = {
        "input_rows": int(equalities.shape[0]),
        "rank": int(basis.shape[0]),
        "singular_values": singular.tolist(),
        "threshold": threshold,
        "maximum_abs_particular_residual": maximum,
        "passes": True,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return particular, basis, diagnostics


def solve_counterfactual_kl_extragradient_update(
    nominal: SymmetricSequentialTrustRegionUpdate,
    *,
    common_ascent_lookahead: CommonAscentLookahead,
    lookahead_kl_values: Any,
    lookahead_kl_shared_gradients: Any,
    target_plus_margins: Any,
    target_plus_gradients: Any,
    target_minus_margins: Any,
    target_minus_gradients: Any,
    optimization_target_margin: Any,
    protected_plus_margins: Any,
    protected_plus_gradients: Any,
    protected_minus_margins: Any,
    protected_minus_gradients: Any,
    protected_baseline_signs: Any,
    protected_margin: Any,
    unrelated_baseline_margins: Any,
    unrelated_plus_margins: Any,
    unrelated_plus_gradients: Any,
    unrelated_minus_margins: Any,
    unrelated_minus_gradients: Any,
    baseline_unrelated_gradients: Any,
    progress_fraction: Any,
    trust_radius: Any,
    physical_residual_scale: Any,
    cell_tangent_kl_limit: Any = CELL_TANGENT_KL_LIMIT,
    mean_tangent_kl_limit: Any = MEAN_TANGENT_KL_LIMIT,
) -> CounterfactualKLESUpdate:
    """Solve the centered CKES QP and certify ideal and float32 states."""

    nominal_certificate = revalidate_symmetric_sequential_trust_region_update(nominal)
    if nominal_certificate.get("passes") is not True:
        raise SymmetricSequentialDMSCertificateError("nominal CL-DMS update is not certified")
    current = _vector(nominal.current_direction, field="current_direction")
    width = int(current.size)
    nominal_update = _vector(nominal.realized_update, field="nominal_update", length=width)
    if not isinstance(common_ascent_lookahead, CommonAscentLookahead):
        raise TypeError("common_ascent_lookahead must be a CommonAscentLookahead")
    common_diagnostics = _plain(common_ascent_lookahead.diagnostics)
    stored_common_hash = common_diagnostics.pop("diagnostics_sha256", None)
    if (
        common_diagnostics.get("passes") is not True
        or stored_common_hash != canonical_sha256(common_diagnostics)
    ):
        raise SymmetricSequentialDMSCertificateError("common-ascent lookahead integrity failed")
    lookahead = _vector(
        common_ascent_lookahead.lookahead_direction,
        field="lookahead_direction",
        length=width,
    )
    target_plus, target_plus_g, target_minus, target_minus_g = _paired_family(
        plus_margins=target_plus_margins,
        plus_gradients=target_plus_gradients,
        minus_margins=target_minus_margins,
        minus_gradients=target_minus_gradients,
        width=width,
        family="target",
    )
    protected_plus, protected_plus_g, protected_minus, protected_minus_g = _paired_family(
        plus_margins=protected_plus_margins,
        plus_gradients=protected_plus_gradients,
        minus_margins=protected_minus_margins,
        minus_gradients=protected_minus_gradients,
        width=width,
        family="protected",
    )
    unrelated_plus, unrelated_plus_g, unrelated_minus, unrelated_minus_g = _paired_family(
        plus_margins=unrelated_plus_margins,
        plus_gradients=unrelated_plus_gradients,
        minus_margins=unrelated_minus_margins,
        minus_gradients=unrelated_minus_gradients,
        width=width,
        family="unrelated",
    )
    target_count = int(target_plus.size)
    protected_count = int(protected_plus.size)
    unrelated_count = int(unrelated_plus.size)
    target_goal = np.broadcast_to(
        _array(optimization_target_margin, field="optimization_target_margin"),
        (target_count,),
    ).astype(np.float64, copy=True)
    protected_floor = np.broadcast_to(
        _array(protected_margin, field="protected_margin"),
        (protected_count,),
    ).astype(np.float64, copy=True)
    protected_signs = _vector(
        protected_baseline_signs,
        field="protected_baseline_signs",
        length=protected_count,
    )
    if bool(np.any(target_goal < 0.0)):
        raise ValueError("optimization_target_margin must be nonnegative")
    if bool(np.any(protected_floor < 0.0)):
        raise ValueError("protected_margin must be nonnegative")
    if not bool(np.all(np.isin(protected_signs, (-1.0, 1.0)))):
        raise ValueError("protected_baseline_signs must contain only -1 or +1")
    unrelated_baseline = _vector(
        unrelated_baseline_margins,
        field="unrelated_baseline_margins",
        length=unrelated_count,
    )
    baseline_unrelated = _matrix(
        baseline_unrelated_gradients,
        field="baseline_unrelated_gradients",
        width=width,
    )
    kl_values = _vector(lookahead_kl_values, field="lookahead_kl_values")
    kl_gradients = _matrix(
        lookahead_kl_shared_gradients,
        field="lookahead_kl_shared_gradients",
        width=width,
        rows=kl_values.size,
    )
    if kl_values.size == 0 or bool(np.any(kl_values < 0.0)):
        raise ValueError("lookahead KL values must be nonempty and nonnegative")
    progress = _fraction(progress_fraction, field="progress_fraction")
    radius = _positive(trust_radius, field="trust_radius")
    residual_scale = _positive(physical_residual_scale, field="physical_residual_scale")
    cell_limit = _positive(cell_tangent_kl_limit, field="cell_tangent_kl_limit")
    mean_limit = _positive(mean_tangent_kl_limit, field="mean_tangent_kl_limit")

    oriented_target_gradients = np.vstack((target_plus_g, target_minus_g))
    common_direction = _vector(
        common_ascent_lookahead.direction,
        field="common_ascent_direction",
        length=width,
    )
    expected_lookahead = current + LOOKAHEAD_EPSILON * common_direction
    common_checks = {
        "current_hash": common_diagnostics.get("current_direction_sha256")
        == _array_sha256(current),
        "target_hash": common_diagnostics.get("oriented_target_gradients_sha256")
        == _array_sha256(oriented_target_gradients),
        "unrelated_hash": common_diagnostics.get("baseline_unrelated_gradients_sha256")
        == _array_sha256(baseline_unrelated),
        "epsilon": float(common_diagnostics.get("epsilon", math.nan)) == LOOKAHEAD_EPSILON,
        "lookahead_identity": np.array_equal(lookahead, expected_lookahead),
        "unit_direction": abs(float(np.linalg.norm(common_direction)) - 1.0) <= 1e-10,
        "common_target_ascent": float(np.min(oriented_target_gradients @ common_direction))
        > COMMON_ASCENT_TOLERANCE,
        "unrelated_null": float(
            np.max(np.abs(baseline_unrelated @ common_direction))
        )
        <= RAW_EQUALITY_TOLERANCE,
    }
    if not all(common_checks.values()):
        raise SymmetricSequentialDMSCertificateError(
            "common-ascent lookahead does not bind to the CKES problem"
        )

    nominal_input = _plain(nominal.diagnostics).get("input_record", {})
    expected_nominal_input = {
        "current_direction_sha256": _array_sha256(current),
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
        "dimension": width,
        "target_pair_count": target_count,
        "protected_pair_count": protected_count,
        "unrelated_pair_count": unrelated_count,
        "baseline_unrelated_null_row_count": int(baseline_unrelated.shape[0]),
    }
    if nominal_input != expected_nominal_input:
        raise SymmetricSequentialDMSCertificateError(
            "nominal CL-DMS update was built from different inputs"
        )

    target_oriented = np.concatenate((target_plus, -target_minus))
    target_goal_both = np.concatenate((target_goal, target_goal))
    target_required = progress * np.maximum(target_goal_both - target_oriented, 0.0)
    target_rows = np.vstack((target_plus_g, target_minus_g))
    protected_rows = np.vstack(
        (protected_signs[:, None] * protected_plus_g, -protected_signs[:, None] * protected_minus_g)
    )
    protected_lower = np.concatenate(
        (
            protected_floor - protected_signs * protected_plus,
            protected_floor - protected_signs * protected_minus,
        )
    )
    tangent_offsets = kl_values + kl_gradients @ (current - lookahead)
    kl_rows = -kl_gradients
    kl_lower = tangent_offsets - cell_limit
    mean_row = -kl_gradients.mean(axis=0, keepdims=True)
    mean_lower = np.asarray([float(tangent_offsets.mean()) - mean_limit], dtype=np.float64)
    inequalities = np.vstack((target_rows, protected_rows, kl_rows, mean_row))
    lower = np.concatenate((target_required, protected_lower, kl_lower, mean_lower))

    unrelated_plus_return = progress * (unrelated_baseline - unrelated_plus)
    unrelated_minus_return = progress * (unrelated_baseline - unrelated_minus)
    exact_null_return = -(baseline_unrelated @ current)
    equalities = np.vstack((unrelated_plus_g, -unrelated_minus_g, baseline_unrelated))
    equality_values = np.concatenate(
        (unrelated_plus_return, unrelated_minus_return, exact_null_return)
    )
    particular, equality_basis, equality_diagnostics = _affine_equality_parameterization(
        equalities, equality_values
    )
    projector = np.eye(width, dtype=np.float64) - equality_basis.T @ equality_basis
    projected_inequalities = inequalities @ projector
    nominal_null = projector @ (nominal_update - particular)
    representers = np.vstack((projected_inequalities, nominal_null[None, :]))
    representer_basis, representer_diagnostics = _orthonormal_row_basis(
        representers, label="centered_qp_representer"
    )
    if not representer_diagnostics["passes"]:
        raise SymmetricSequentialDMSCertificateError("centered-QP row-space certificate failed")

    def update_from(z: np.ndarray) -> np.ndarray:
        return particular + representer_basis.T @ z

    def objective(z: np.ndarray) -> float:
        delta = update_from(z) - nominal_update
        return 0.5 * float(delta @ delta)

    def objective_gradient(z: np.ndarray) -> np.ndarray:
        return representer_basis @ (update_from(z) - nominal_update)

    reduced_rows = inequalities @ representer_basis.T
    reduced_lower = lower - inequalities @ particular
    reduced_norms = np.linalg.norm(reduced_rows, axis=1)
    contradictory_zero_rows = (reduced_norms <= SVD_ATOL) & (
        reduced_lower > RAW_INEQUALITY_TOLERANCE
    )
    if bool(np.any(contradictory_zero_rows)):
        raise SymmetricSequentialDMSInfeasibleError(
            "a certified zero CKES constraint row has a positive lower bound"
        )

    projected_start = particular + projector @ (nominal_update - particular)
    starts = [
        representer_basis @ (projected_start - particular),
        np.zeros(representer_basis.shape[0], dtype=np.float64),
    ]
    starts.append(0.5 * starts[0])
    candidates: list[tuple[float, np.ndarray, Any]] = []
    feasibility_status: int | None = None
    feasibility_message = "rank-zero representer; affine particular checked directly"
    if representer_basis.shape[0] == 0:
        update = particular
        equality_error = float(np.max(np.abs(equalities @ update - equality_values)))
        minimum_slack = float(np.min(inequalities @ update - lower))
        if (
            equality_error <= RAW_EQUALITY_TOLERANCE
            and minimum_slack >= -RAW_INEQUALITY_TOLERANCE
            and float(np.linalg.norm(update)) <= radius + TRUST_TOLERANCE
        ):
            optimizer = SimpleNamespace(
                success=True,
                status=0,
                message="rank-zero affine particular passed",
                nit=0,
            )
            candidates.append((objective(np.zeros(0)), update, optimizer))
        else:
            raise SymmetricSequentialDMSInfeasibleError(
                "the unique centered-QP affine particular violates a constraint or trust radius"
            )
    else:
        feasibility = linprog(
            np.zeros(representer_basis.shape[0], dtype=np.float64),
            A_ub=-reduced_rows,
            b_ub=-reduced_lower,
            bounds=[(None, None)] * representer_basis.shape[0],
            method="highs",
        )
        feasibility_status = int(feasibility.status)
        feasibility_message = str(feasibility.message)
        if feasibility.status == 2:
            raise SymmetricSequentialDMSInfeasibleError(
                "the centered CKES linear constraints are certified infeasible"
            )
        if not feasibility.success:
            raise SymmetricSequentialDMSSolverError(
                f"centered CKES linear feasibility failed: {feasibility.message}"
            )
        starts.append(np.asarray(feasibility.x, dtype=np.float64))
        for start in starts:
            result = minimize(
                objective,
                start,
                jac=objective_gradient,
                method="SLSQP",
                constraints=(
                    {
                        "type": "ineq",
                        "fun": lambda z: inequalities @ update_from(z) - lower,
                        "jac": lambda z: inequalities @ representer_basis.T,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda z: radius**2 - float(update_from(z) @ update_from(z)),
                        "jac": lambda z: -2.0 * (representer_basis @ update_from(z)),
                    },
                ),
                options={
                    "maxiter": SOLVER_MAX_ITERATIONS,
                    "ftol": SOLVER_FUNCTION_TOLERANCE,
                },
            )
            if result.success and np.isfinite(result.x).all():
                update = update_from(np.asarray(result.x, dtype=np.float64))
                equality_error = float(np.max(np.abs(equalities @ update - equality_values)))
                minimum_slack = float(np.min(inequalities @ update - lower))
                if (
                    equality_error <= RAW_EQUALITY_TOLERANCE
                    and minimum_slack >= -RAW_INEQUALITY_TOLERANCE
                    and float(np.linalg.norm(update)) <= radius + TRUST_TOLERANCE
                ):
                    candidates.append((objective(np.asarray(result.x)), update, result))
    if not candidates:
        raise SymmetricSequentialDMSSolverError(
            "no centered CKES optimizer candidate passed the independent primal certificate"
        )
    _, ideal_update, optimizer = min(candidates, key=lambda item: item[0])
    ideal_direction = current + ideal_update
    positive_physical = _float32(residual_scale * ideal_direction)
    negative_physical = -positive_physical
    realized_direction = positive_physical.astype(np.float64) / residual_scale
    realized_update = realized_direction - current

    def certificate(update: np.ndarray, direction: np.ndarray) -> dict[str, Any]:
        inequality_slack = inequalities @ update - lower
        equality_error = equalities @ update - equality_values
        tangent_kl = kl_values + kl_gradients @ (direction - lookahead)
        target_progress = target_rows @ update
        unrelated_plus_next = unrelated_plus + unrelated_plus_g @ update
        unrelated_minus_next = unrelated_minus - unrelated_minus_g @ update
        unrelated_plus_desired = unrelated_plus + unrelated_plus_return
        unrelated_minus_desired = unrelated_minus + unrelated_minus_return
        checks = {
            "inequalities": bool(float(inequality_slack.min()) >= -RAW_INEQUALITY_TOLERANCE),
            "equalities": bool(
                float(np.max(np.abs(equality_error))) <= RAW_EQUALITY_TOLERANCE
            ),
            "trust_radius": bool(float(np.linalg.norm(update)) <= radius + TRUST_TOLERANCE),
            "cell_tangent_kl": bool(float(tangent_kl.max()) <= cell_limit + RAW_INEQUALITY_TOLERANCE),
            "mean_tangent_kl": bool(float(tangent_kl.mean()) <= mean_limit + RAW_INEQUALITY_TOLERANCE),
        }
        value = {
            "target_realized_progress": target_progress.tolist(),
            "target_required_progress": target_required.tolist(),
            "unrelated_plus_next_margins": unrelated_plus_next.tolist(),
            "unrelated_minus_next_margins": unrelated_minus_next.tolist(),
            "unrelated_plus_desired_margins": unrelated_plus_desired.tolist(),
            "unrelated_minus_desired_margins": unrelated_minus_desired.tolist(),
            "tangent_kl_values": tangent_kl.tolist(),
            "maximum_tangent_kl": float(tangent_kl.max()),
            "mean_tangent_kl": float(tangent_kl.mean()),
            "minimum_inequality_slack": float(inequality_slack.min()),
            "maximum_abs_equality_error": float(np.max(np.abs(equality_error))),
            "update_l2": float(np.linalg.norm(update)),
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
        value["certificate_sha256"] = canonical_sha256(value)
        return value

    ideal_certificate = certificate(ideal_update, ideal_direction)
    realized_certificate = certificate(realized_update, realized_direction)
    if not ideal_certificate["passes"] or not realized_certificate["passes"]:
        raise SymmetricSequentialDMSCertificateError(
            "ideal or authoritative float32 CKES state failed certification"
        )
    input_record = {
        **expected_nominal_input,
        "common_ascent_diagnostics_sha256": stored_common_hash,
        "lookahead_direction_sha256": _array_sha256(lookahead),
        "lookahead_kl_values_sha256": _array_sha256(kl_values),
        "lookahead_kl_shared_gradients_sha256": _array_sha256(kl_gradients),
        "cell_tangent_kl_limit": cell_limit,
        "mean_tangent_kl_limit": mean_limit,
    }
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "solver": {
            "method": "SLSQP_centered_convex_QP",
            "scipy_version": scipy_version,
            "success": bool(optimizer.success),
            "status": int(optimizer.status),
            "message": str(optimizer.message),
            "iterations": int(optimizer.nit),
            "deterministic_start_count": len(starts),
            "passing_candidate_count": len(candidates),
            "linear_feasibility_method": "scipy_linprog_highs",
            "linear_feasibility_status": feasibility_status,
            "linear_feasibility_message": feasibility_message,
        },
        "nominal_revalidation_sha256": nominal_certificate["revalidation_sha256"],
        "nominal_diagnostics_sha256": _plain(nominal.diagnostics)["diagnostics_sha256"],
        "input_record": input_record,
        "input_sha256": canonical_sha256(input_record),
        "common_ascent_checks": common_checks,
        "equality_parameterization": equality_diagnostics,
        "representer_basis": representer_diagnostics,
        "cell_tangent_kl_limit": cell_limit,
        "mean_tangent_kl_limit": mean_limit,
        "lookahead_direction_sha256": _array_sha256(lookahead),
        "lookahead_kl_values_sha256": _array_sha256(kl_values),
        "lookahead_kl_shared_gradients_sha256": _array_sha256(kl_gradients),
        "nominal_update_sha256": _array_sha256(nominal_update),
        "current_direction_sha256": _array_sha256(current),
        "ideal_update_sha256": _array_sha256(ideal_update),
        "ideal_updated_direction_sha256": _array_sha256(ideal_direction),
        "realized_update_sha256": _array_sha256(realized_update),
        "realized_direction_sha256": _array_sha256(realized_direction),
        "positive_deployed_direction_sha256": _array_sha256(realized_direction),
        "negative_deployed_direction_sha256": _array_sha256(-realized_direction),
        "physical_residual_scale": residual_scale,
        "positive_physical_float32_sha256": _float32_sha256(positive_physical),
        "negative_physical_float32_sha256": _float32_sha256(negative_physical),
        "ideal_certificate": ideal_certificate,
        "realized_deployment_certificate": realized_certificate,
        "passes": True,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return CounterfactualKLESUpdate(
        current_direction=_readonly(current, dtype=np.float64),
        nominal_update=_readonly(nominal_update, dtype=np.float64),
        ideal_update=_readonly(ideal_update, dtype=np.float64),
        ideal_updated_direction=_readonly(ideal_direction, dtype=np.float64),
        realized_update=_readonly(realized_update, dtype=np.float64),
        realized_direction=_readonly(realized_direction, dtype=np.float64),
        positive_deployed_direction=_readonly(realized_direction, dtype=np.float64),
        negative_deployed_direction=_readonly(-realized_direction, dtype=np.float64),
        positive_physical_float32=_readonly(
            positive_physical, dtype=np.float32, canonical_zero=False
        ),
        negative_physical_float32=_readonly(
            negative_physical, dtype=np.float32, canonical_zero=False
        ),
        positive_physical_float32_sha256=_float32_sha256(positive_physical),
        negative_physical_float32_sha256=_float32_sha256(negative_physical),
        diagnostics=_freeze(diagnostics),
    )


def revalidate_counterfactual_kl_extragradient_update(
    candidate: CounterfactualKLESUpdate,
    *,
    expected_diagnostics_sha256: str | None = None,
) -> dict[str, Any]:
    """Independently revalidate array identities, hashes, and deployment bytes."""

    if not isinstance(candidate, CounterfactualKLESUpdate):
        raise TypeError("candidate must be a CounterfactualKLESUpdate")
    if expected_diagnostics_sha256 is not None:
        _checked = str(expected_diagnostics_sha256)
        if len(_checked) != 64 or any(character not in "0123456789abcdef" for character in _checked):
            raise ValueError("expected_diagnostics_sha256 must be one lowercase SHA-256")
    diagnostics = _plain(candidate.diagnostics)
    observed_hash = diagnostics.pop("diagnostics_sha256", None)
    input_record = diagnostics.get("input_record", {})
    scale = diagnostics.get("physical_residual_scale")
    arrays = {
        "current_direction": candidate.current_direction,
        "nominal_update": candidate.nominal_update,
        "ideal_update": candidate.ideal_update,
        "ideal_updated_direction": candidate.ideal_updated_direction,
        "realized_update": candidate.realized_update,
        "realized_direction": candidate.realized_direction,
        "positive_deployed_direction": candidate.positive_deployed_direction,
        "negative_deployed_direction": candidate.negative_deployed_direction,
        "positive_physical_float32": candidate.positive_physical_float32,
        "negative_physical_float32": candidate.negative_physical_float32,
    }
    dimension = int(np.asarray(candidate.current_direction).size)
    float64_names = (
        "current_direction",
        "nominal_update",
        "ideal_update",
        "ideal_updated_direction",
        "realized_update",
        "realized_direction",
        "positive_deployed_direction",
        "negative_deployed_direction",
    )
    float32_names = ("positive_physical_float32", "negative_physical_float32")
    shape_and_dtype = bool(
        dimension > 0
        and all(
            isinstance(arrays[name], np.ndarray)
            and arrays[name].shape == (dimension,)
            and arrays[name].dtype == np.float64
            for name in float64_names
        )
        and all(
            isinstance(arrays[name], np.ndarray)
            and arrays[name].shape == (dimension,)
            and arrays[name].dtype == np.float32
            for name in float32_names
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
        expected_ideal_direction = candidate.current_direction + candidate.ideal_update
        expected_positive = _float32(float(scale) * candidate.ideal_updated_direction)
        expected_negative = _float32(np.negative(candidate.positive_physical_float32))
        expected_realized = candidate.positive_physical_float32.astype(np.float64) / float(scale)
        expected_update = expected_realized - candidate.current_direction
        expected_negative_direction = (
            candidate.negative_physical_float32.astype(np.float64) / float(scale)
        )
    else:
        expected_ideal_direction = np.empty(0, dtype=np.float64)
        expected_positive = np.empty(0, dtype=np.float32)
        expected_negative = np.empty(0, dtype=np.float32)
        expected_realized = np.empty(0, dtype=np.float64)
        expected_update = np.empty(0, dtype=np.float64)
        expected_negative_direction = np.empty(0, dtype=np.float64)

    def embedded_hash_valid(record_name: str, hash_name: str) -> bool:
        record = diagnostics.get(record_name)
        if not isinstance(record, dict):
            return False
        record = dict(record)
        stored = record.pop(hash_name, None)
        return isinstance(stored, str) and stored == canonical_sha256(record)

    checks = {
        "diagnostics_hash": isinstance(observed_hash, str)
        and observed_hash == canonical_sha256(diagnostics),
        "expected_diagnostics_hash": expected_diagnostics_sha256 is None
        or observed_hash == expected_diagnostics_sha256,
        "schema": diagnostics.get("schema_version") == SCHEMA_VERSION,
        "shape_and_dtype": shape_and_dtype,
        "finite": finite,
        "arrays_are_read_only": bool(
            shape_and_dtype and all(not value.flags.writeable for value in arrays.values())
        ),
        "physical_residual_scale": scale_valid,
        "ideal_direction_identity": bool(
            shape_and_dtype
            and np.array_equal(candidate.ideal_updated_direction, expected_ideal_direction)
        ),
        "positive_physical_from_ideal": bool(
            shape_and_dtype
            and candidate.positive_physical_float32.tobytes(order="C")
            == expected_positive.tobytes(order="C")
        ),
        "negative_physical_bytewise_unary_negation": bool(
            shape_and_dtype
            and candidate.negative_physical_float32.tobytes(order="C")
            == expected_negative.tobytes(order="C")
        ),
        "realized_direction_from_physical": bool(
            shape_and_dtype and np.array_equal(candidate.realized_direction, expected_realized)
        ),
        "realized_update_identity": bool(
            shape_and_dtype and np.array_equal(candidate.realized_update, expected_update)
        ),
        "positive_deployed_alias": bool(
            shape_and_dtype
            and np.array_equal(candidate.positive_deployed_direction, candidate.realized_direction)
        ),
        "negative_deployed_from_physical": bool(
            shape_and_dtype
            and np.array_equal(candidate.negative_deployed_direction, expected_negative_direction)
        ),
        "current_direction_hash": diagnostics.get("current_direction_sha256")
        == _array_sha256(candidate.current_direction),
        "nominal_update_hash": diagnostics.get("nominal_update_sha256")
        == _array_sha256(candidate.nominal_update),
        "ideal_update_hash": diagnostics.get("ideal_update_sha256")
        == _array_sha256(candidate.ideal_update),
        "ideal_updated_direction_hash": diagnostics.get("ideal_updated_direction_sha256")
        == _array_sha256(candidate.ideal_updated_direction),
        "realized_update_hash": diagnostics.get("realized_update_sha256")
        == _array_sha256(candidate.realized_update),
        "realized_direction_hash": diagnostics.get("realized_direction_sha256")
        == _array_sha256(candidate.realized_direction),
        "positive_deployed_hash": diagnostics.get("positive_deployed_direction_sha256")
        == _array_sha256(candidate.positive_deployed_direction),
        "negative_deployed_hash": diagnostics.get("negative_deployed_direction_sha256")
        == _array_sha256(candidate.negative_deployed_direction),
        "positive_physical_hash": candidate.positive_physical_float32_sha256
        == _float32_sha256(candidate.positive_physical_float32)
        == diagnostics.get("positive_physical_float32_sha256"),
        "negative_physical_hash": candidate.negative_physical_float32_sha256
        == _float32_sha256(candidate.negative_physical_float32)
        == diagnostics.get("negative_physical_float32_sha256"),
        "input_record_hash": isinstance(input_record, dict)
        and diagnostics.get("input_sha256") == canonical_sha256(input_record),
        "input_current_hash": isinstance(input_record, dict)
        and input_record.get("current_direction_sha256")
        == _array_sha256(candidate.current_direction),
        "ideal_certificate_hash": embedded_hash_valid(
            "ideal_certificate", "certificate_sha256"
        ),
        "realized_certificate_hash": embedded_hash_valid(
            "realized_deployment_certificate", "certificate_sha256"
        ),
        "equality_diagnostics_hash": embedded_hash_valid(
            "equality_parameterization", "diagnostics_sha256"
        ),
        "representer_diagnostics_hash": embedded_hash_valid(
            "representer_basis", "diagnostics_sha256"
        ),
        "recorded_passes": diagnostics.get("passes") is True
        and diagnostics.get("ideal_certificate", {}).get("passes") is True
        and diagnostics.get("realized_deployment_certificate", {}).get("passes") is True,
    }
    result = {
        "schema_version": f"{SCHEMA_VERSION}.revalidation",
        "diagnostics_sha256": observed_hash,
        "checks": checks,
        "passes": bool(all(checks.values())),
    }
    result["revalidation_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "CELL_TANGENT_KL_LIMIT",
    "COMMON_ASCENT_TOLERANCE",
    "LOOKAHEAD_EPSILON",
    "MEAN_TANGENT_KL_LIMIT",
    "SCHEMA_VERSION",
    "CommonAscentLookahead",
    "CounterfactualKLESUpdate",
    "construct_common_ascent_lookahead",
    "revalidate_counterfactual_kl_extragradient_update",
    "solve_counterfactual_kl_extragradient_update",
]
