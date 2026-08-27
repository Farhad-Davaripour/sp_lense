"""Adapt frozen gradient-specificity-v3 captures to GCRBS inputs.

The adapter is deliberately model-free.  It consumes the residual-scaled,
prompt-final gradients and logits already frozen by the v3 capture runner and
emits the affine inequalities used by the global counterfactual robust-boundary
solver.

For a target row, ``A @ d - b`` is the predicted *actual*
desired-token-minus-competitor logit gap.  The positive intervention makes the
preserve token win; the negative intervention makes the comply token win.  A
publication decision margin is retained as metadata and is not folded into
``b``.  For a protected row, ``C @ d >= q`` preserves a nonnegative baseline-
greedy-minus-competitor gap under each of ``+d`` and ``-d``.

The two Fisher factors produced here are prompt-balanced top-k-plus-aggregate-
tail, first-order construction surrogates.  They are not exact full-vocabulary
KL measurements; exact KL must remain a separate finite intervention gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from sp_lense.gradient_specificity_v3 import (
    DEFAULT_CENTERED_SCORE_IDENTITY_TOLERANCE,
    DEFAULT_PROBABILITY_TOLERANCE,
    DEFAULT_SCORE_IDENTITY_TOLERANCE,
    DEFAULT_SVD_ATOL,
    DEFAULT_SVD_RTOL,
    canonical_sha256,
    prompt_balanced_topk_tail_fisher_factors,
    row_normalized_svd_basis,
)

SCHEMA_VERSION = "sp_lense.gcrbs_capture_adapter.v1"
CAPTURE_SCHEMA_VERSION = "sp_lense.gradient_specificity_v3_capture.v1"
SOLVER_SCHEMA_VERSION = "sp_lense.global_counterfactual_robust_boundary.v1"
GRADIENT_COORDINATE = "residual_scaled_final_prompt"
SEMANTIC_OBJECTIVE = "semantic_positive_minus_negative_logit"
FISHER_SURROGATE_LABEL = (
    "prompt_balanced_top_k_plus_aggregate_tail_fisher_first_order_residual_scaled"
)
FISHER_SURROGATE_SCOPE = "construction_only_not_exact_full_vocabulary_kl"
DEFAULT_REQUIRED_MARGIN = 0.01
DEFAULT_GRADIENT_CONSISTENCY_RTOL = 2e-4
DEFAULT_GRADIENT_CONSISTENCY_ATOL = 2e-6
PROTECTED_GROUP_KEYS = (
    "matched_other_shutdown",
    "ordinary_interruption",
    "unrelated_benign_compliance_and_capability",
    "harmless_and_harmful_refusal",
    "option_and_mapping_competence",
)
NUISANCE_FAMILY_TO_PROTECTED_GROUP = {
    "benign_compliance": "unrelated_benign_compliance_and_capability",
    "general_capability": "unrelated_benign_compliance_and_capability",
    "refusal": "harmless_and_harmful_refusal",
    "option_order_sentinel": "option_and_mapping_competence",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_float64_array(value: Any, *, field: str, ndim: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu").numpy()
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{field} must contain real numbers")
    if raw.ndim != ndim:
        raise ValueError(f"{field} must be {ndim}-dimensional")
    result = np.asarray(raw, dtype=np.float64, order="C").copy(order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values")
    result[result == 0.0] = 0.0
    return result


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64, order="C").copy(order="C")
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def float64_array_sha256(value: Any) -> str:
    """Hash an array exactly as the GCRBS solver hashes float64 inputs."""

    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu").numpy()
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim not in {1, 2}:
        raise TypeError("value must be a real vector or matrix")
    array = _canonical_float64_array(raw, field="value", ndim=raw.ndim)
    header = _canonical_json_bytes({"dtype": "float64", "shape": list(array.shape)})
    little_endian = array.astype("<f8", copy=False)
    return hashlib.sha256(header + b"\n" + little_endian.tobytes(order="C")).hexdigest()


def _float32_bytes_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype=np.float32, order="C")
    return hashlib.sha256(array.astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def _finite_float(value: Any, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{field} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _integer(value: Any, *, field: str, nonnegative: bool = True) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if nonnegative and result < 0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _token_ids(value: Any, *, field: str, exact_length: int | None = None) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    output = tuple(_integer(item, field=f"{field}[{index}]") for index, item in enumerate(value))
    if exact_length is not None and len(output) != exact_length:
        raise ValueError(f"{field} must contain exactly {exact_length} token IDs")
    if len(set(output)) != len(output):
        raise ValueError(f"{field} must contain unique token IDs")
    return output


def _logit_values(value: Any, *, field: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    output = tuple(
        _finite_float(item, field=f"{field}[{index}]") for index, item in enumerate(value)
    )
    if len(output) != length:
        raise ValueError(f"{field} must have length {length}")
    return output


def _validate_recorded_float32_hash(
    record: Mapping[str, Any],
    *,
    value_field: str,
    hash_field: str,
    value: np.ndarray,
) -> str:
    observed = record.get(hash_field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise ValueError(f"{hash_field} must be a SHA-256 hex digest")
    try:
        int(observed, 16)
    except ValueError as error:
        raise ValueError(f"{hash_field} must be a SHA-256 hex digest") from error
    expected = _float32_bytes_sha256(value)
    if observed != expected:
        raise RuntimeError(f"{value_field} differs from its recorded float32 hash")
    return observed


@dataclass(frozen=True, slots=True)
class _CheckedCapture:
    form_id: str
    role: str
    family: str
    case_id: str
    assignment: int | None
    target: str | None
    interruption: bool | None
    order_first: bool
    positive_label: str
    negative_label: str
    positive_token_id: int
    negative_token_id: int
    greedy_token_id: int
    top9_token_ids: tuple[int, ...]
    logits: Mapping[int, float]
    semantic_log_odds: float
    semantic_gradient: np.ndarray
    competitor_token_ids: tuple[int, ...]
    gap_gradients: np.ndarray
    fisher_token_ids: tuple[int, ...]
    fisher_probabilities: np.ndarray
    fisher_gradients: np.ndarray
    fisher_tail_probability: float
    fisher_tail_gradient: np.ndarray
    source_manifest: Mapping[str, Any]

    @property
    def dimension(self) -> int:
        return int(self.semantic_gradient.shape[0])


@dataclass(frozen=True, slots=True)
class FisherSurrogateGroup:
    """One separately budgeted prompt-balanced construction surrogate."""

    group_key: str
    surrogate_label: str
    surrogate_scope: str
    factor: np.ndarray
    factor_sha256: str
    source_form_ids: tuple[str, ...]
    diagnostics: Mapping[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class GCRBSCaptureConstraints:
    """Validated GCRBS affine inputs and their ordered provenance."""

    target_matrix: np.ndarray
    target_offsets: np.ndarray
    unrelated_equality_basis: np.ndarray
    protected_matrix: np.ndarray
    protected_lower_bounds: np.ndarray
    required_margin: float
    target_rows: tuple[Mapping[str, Any], ...]
    protected_rows: tuple[Mapping[str, Any], ...]
    unrelated_raw_rows: tuple[Mapping[str, Any], ...]
    fisher_surrogate_groups: tuple[FisherSurrogateGroup, ...]
    fisher_prompt_surrogate_groups: tuple[FisherSurrogateGroup, ...]
    provenance: Mapping[str, Any]

    @property
    def group_metric_factors(self) -> tuple[np.ndarray, ...]:
        return tuple(
            group.factor
            for group in (
                *self.fisher_surrogate_groups,
                *self.fisher_prompt_surrogate_groups,
            )
        )

    @property
    def group_metric_labels(self) -> tuple[str, ...]:
        return tuple(
            group.group_key
            for group in (
                *self.fisher_surrogate_groups,
                *self.fisher_prompt_surrogate_groups,
            )
        )

    def base_solver_kwargs(self) -> dict[str, Any]:
        """Return the affine solver inputs, excluding caps and metric budgets."""

        return {
            "target_matrix": self.target_matrix,
            "target_offsets": self.target_offsets,
            "unrelated_equality_basis": self.unrelated_equality_basis,
            "protected_matrix": self.protected_matrix,
            "protected_lower_bounds": self.protected_lower_bounds,
        }

    def solver_kwargs(self, *, group_metric_budgets: Sequence[Any]) -> dict[str, Any]:
        """Return affine inputs plus all aggregate and per-prompt Fisher budgets."""

        budgets = tuple(
            _finite_float(value, field=f"group_metric_budgets[{index}]", nonnegative=True)
            for index, value in enumerate(group_metric_budgets)
        )
        if len(budgets) != len(self.group_metric_factors):
            raise ValueError(
                "group_metric_budgets must match all aggregate and per-prompt factors"
            )
        return {
            **self.base_solver_kwargs(),
            "group_metric_factors": self.group_metric_factors,
            "group_metric_budgets": budgets,
        }

    def bind_solver_input_sha256(self, solver_input_sha256: str) -> dict[str, Any]:
        """Bind ordered row provenance to one solver-validated input record."""

        if not isinstance(solver_input_sha256, str) or len(solver_input_sha256) != 64:
            raise ValueError("solver_input_sha256 must be a SHA-256 hex digest")
        try:
            int(solver_input_sha256, 16)
        except ValueError as error:
            raise ValueError("solver_input_sha256 must be a SHA-256 hex digest") from error
        binding = {
            "schema_version": f"{SCHEMA_VERSION}.solver_binding",
            "solver_schema_version": SOLVER_SCHEMA_VERSION,
            "solver_input_sha256": solver_input_sha256,
            "adapter_provenance_sha256": self.provenance["provenance_sha256"],
            "target_row_ids": [row["row_id"] for row in self.target_rows],
            "protected_row_ids": [row["row_id"] for row in self.protected_rows],
            "unrelated_raw_row_ids": [row["row_id"] for row in self.unrelated_raw_rows],
            "group_metric_keys": list(self.group_metric_labels),
            "matrix_sha256s": dict(self.provenance["matrix_sha256s"]),
        }
        binding["binding_sha256"] = canonical_sha256(binding)
        return binding


def _check_capture_record(
    record: Mapping[str, Any],
    *,
    role: str,
) -> _CheckedCapture:
    if not isinstance(record, Mapping):
        raise TypeError("capture records must be mappings")
    form_id = _nonempty_string(record.get("form_id"), field="form_id")
    if record.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        raise ValueError(f"capture {form_id!r} has the wrong schema_version")
    if record.get("development_only") is not True:
        raise ValueError(f"capture {form_id!r} must retain development_only=true")
    if record.get("gradient_coordinate") != GRADIENT_COORDINATE:
        raise ValueError(f"capture {form_id!r} is not in residual-scaled coordinates")
    if record.get("objective_name") != SEMANTIC_OBJECTIVE:
        raise ValueError(f"capture {form_id!r} has the wrong semantic objective")
    if record.get("batched_vjp") is not True or _integer(
        record.get("hook_call_count"), field=f"{form_id}.hook_call_count"
    ) != 1:
        raise ValueError(f"capture {form_id!r} lacks its one-hook batched-VJP certificate")
    prompt_length = _integer(record.get("prompt_length"), field=f"{form_id}.prompt_length")
    prompt_final_index = _integer(
        record.get("prompt_final_index"), field=f"{form_id}.prompt_final_index"
    )
    if prompt_length <= 0 or prompt_final_index != prompt_length - 1:
        raise ValueError(f"capture {form_id!r} is not at the final prompt position")
    residual_norm = _finite_float(record.get("residual_norm"), field=f"{form_id}.residual_norm")
    if residual_norm <= 0.0:
        raise ValueError(f"capture {form_id!r} has a nonpositive residual norm")

    positive_label = record.get("positive_label")
    negative_label = record.get("negative_label")
    if {positive_label, negative_label} != {"A", "B"}:
        raise ValueError(f"capture {form_id!r} must orient exactly A versus B")
    choice_a = _integer(record.get("choice_a_token_id"), field=f"{form_id}.choice_a_token_id")
    choice_b = _integer(record.get("choice_b_token_id"), field=f"{form_id}.choice_b_token_id")
    if choice_a == choice_b:
        raise ValueError(f"capture {form_id!r} has identical A and B token IDs")
    positive_token = choice_a if positive_label == "A" else choice_b
    negative_token = choice_b if negative_label == "B" else choice_a

    family = _nonempty_string(record.get("family"), field=f"{form_id}.family")
    case_id = _nonempty_string(record.get("case_id"), field=f"{form_id}.case_id")
    assignment: int | None = None
    target: str | None = None
    interruption: bool | None = None
    if role == "sp":
        if family != "self_preservation" or record.get("positive_semantics") != "preserve":
            raise ValueError(
                f"capture {form_id!r} must be preserve-minus-comply self-preservation"
            )
        target = record.get("target")
        if target not in {"self", "other"}:
            raise ValueError(f"capture {form_id!r} target must be self or other")
        assignment = _integer(record.get("assignment"), field=f"{form_id}.assignment")
        if assignment not in {0, 1}:
            raise ValueError(f"capture {form_id!r} assignment must be zero or one")
        interruption_value = record.get("interruption")
        if not isinstance(interruption_value, bool):
            raise TypeError(f"capture {form_id!r} interruption must be boolean")
        interruption = interruption_value
        order = record.get("preserve_first")
        if not isinstance(order, bool):
            raise TypeError(f"capture {form_id!r} preserve_first must be boolean")
        expected_positive = "A" if order else "B"
    elif role == "nuisance":
        if record.get("unrelated_role") != "nuisance_fit":
            raise ValueError(f"capture {form_id!r} is not a nuisance-fit record")
        if record.get("positive_semantics") != "preferred":
            raise ValueError(f"capture {form_id!r} must be preferred-minus-alternative")
        order = record.get("preferred_first")
        if not isinstance(order, bool):
            raise TypeError(f"capture {form_id!r} preferred_first must be boolean")
        expected_positive = "A" if order else "B"
    else:
        raise ValueError("role must be sp or nuisance")
    if positive_label != expected_positive or negative_label == expected_positive:
        raise ValueError(f"capture {form_id!r} semantic orientation disagrees with answer order")

    semantic = _canonical_float64_array(
        record.get("semantic_gradient"), field=f"{form_id}.semantic_gradient", ndim=1
    )
    if semantic.size == 0:
        raise ValueError(f"capture {form_id!r} has an empty semantic gradient")
    gaps = _canonical_float64_array(
        record.get("greedy_competitor_gap_gradients"),
        field=f"{form_id}.greedy_competitor_gap_gradients",
        ndim=2,
    )
    if gaps.shape != (8, semantic.size):
        raise ValueError(f"capture {form_id!r} must contain eight d_model-wide gap gradients")
    semantic_source_hash = _validate_recorded_float32_hash(
        record,
        value_field="semantic_gradient",
        hash_field="semantic_gradient_sha256",
        value=semantic,
    )
    gap_source_hash = _validate_recorded_float32_hash(
        record,
        value_field="greedy_competitor_gap_gradients",
        hash_field="greedy_competitor_gap_gradients_sha256",
        value=gaps,
    )

    top9 = _token_ids(record.get("top9_token_ids"), field=f"{form_id}.top9_token_ids", exact_length=9)
    top9_logits = _logit_values(
        record.get("top9_logit_values"), field=f"{form_id}.top9_logit_values", length=9
    )
    if any(left < right for left, right in pairwise(top9_logits)):
        raise ValueError(f"capture {form_id!r} top-9 logits are not descending")
    greedy = _integer(
        record.get("baseline_greedy_token_id"), field=f"{form_id}.baseline_greedy_token_id"
    )
    if greedy != top9[0]:
        raise ValueError(f"capture {form_id!r} greedy token disagrees with top-9 rank zero")
    competitors = _token_ids(
        record.get("greedy_competitor_token_ids"),
        field=f"{form_id}.greedy_competitor_token_ids",
        exact_length=8,
    )
    if competitors != top9[1:]:
        raise ValueError(f"capture {form_id!r} competitor IDs disagree with top-9 ordering")

    union_ids = _token_ids(
        record.get("top9_union_required_ab_token_ids"),
        field=f"{form_id}.top9_union_required_ab_token_ids",
    )
    if not 9 <= len(union_ids) <= 11 or not set(top9).issubset(union_ids):
        raise ValueError(f"capture {form_id!r} has a malformed top-9/A/B union")
    if not {choice_a, choice_b}.issubset(union_ids):
        raise ValueError(f"capture {form_id!r} union omits a required A/B token")
    union_logits = _logit_values(
        record.get("top9_union_required_ab_logit_values"),
        field=f"{form_id}.top9_union_required_ab_logit_values",
        length=len(union_ids),
    )
    logits = dict(zip(union_ids, union_logits, strict=True))
    for token_id, top_value in zip(top9, top9_logits, strict=True):
        if not math.isclose(logits[token_id], top_value, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"capture {form_id!r} top-9 and union logits disagree")
    semantic_log_odds = _finite_float(
        record.get("baseline_semantic_log_odds"),
        field=f"{form_id}.baseline_semantic_log_odds",
    )
    expected_semantic_log_odds = logits[positive_token] - logits[negative_token]
    if not math.isclose(
        semantic_log_odds,
        expected_semantic_log_odds,
        rel_tol=1e-8,
        abs_tol=1e-8,
    ):
        raise ValueError(f"capture {form_id!r} semantic log-odds has the wrong orientation")
    raw_a_minus_b = _finite_float(
        record.get("baseline_raw_a_minus_b_log_odds"),
        field=f"{form_id}.baseline_raw_a_minus_b_log_odds",
    )
    if not math.isclose(
        raw_a_minus_b,
        logits[choice_a] - logits[choice_b],
        rel_tol=1e-8,
        abs_tol=1e-8,
    ):
        raise ValueError(f"capture {form_id!r} raw A-minus-B log-odds is inconsistent")

    fisher_ids = _token_ids(
        record.get("fisher_category_token_ids"), field=f"{form_id}.fisher_category_token_ids"
    )
    if not 8 <= len(fisher_ids) <= 10:
        raise ValueError(f"capture {form_id!r} Fisher union must contain 8 to 10 tokens")
    fisher_probabilities = _canonical_float64_array(
        record.get("fisher_category_probabilities"),
        field=f"{form_id}.fisher_category_probabilities",
        ndim=1,
    )
    if fisher_probabilities.shape != (len(fisher_ids),):
        raise ValueError(f"capture {form_id!r} Fisher probabilities have the wrong length")
    fisher_gradients = _canonical_float64_array(
        record.get("fisher_category_score_gradients"),
        field=f"{form_id}.fisher_category_score_gradients",
        ndim=2,
    )
    if fisher_gradients.shape != (len(fisher_ids), semantic.size):
        raise ValueError(f"capture {form_id!r} Fisher gradients have the wrong shape")
    fisher_tail_gradient = _canonical_float64_array(
        record.get("fisher_tail_score_gradient"),
        field=f"{form_id}.fisher_tail_score_gradient",
        ndim=1,
    )
    if fisher_tail_gradient.shape != semantic.shape:
        raise ValueError(f"capture {form_id!r} Fisher tail gradient has the wrong shape")
    fisher_tail_probability = _finite_float(
        record.get("fisher_tail_probability"),
        field=f"{form_id}.fisher_tail_probability",
        nonnegative=True,
    )
    if fisher_tail_probability > 1.0:
        raise ValueError(f"capture {form_id!r} Fisher tail probability exceeds one")
    fisher_source_hash = _validate_recorded_float32_hash(
        record,
        value_field="fisher_category_score_gradients",
        hash_field="fisher_category_score_gradients_sha256",
        value=fisher_gradients,
    )
    tail_source_hash = _validate_recorded_float32_hash(
        record,
        value_field="fisher_tail_score_gradient",
        hash_field="fisher_tail_score_gradient_sha256",
        value=fisher_tail_gradient,
    )

    source_manifest = {
        "form_id": form_id,
        "role": role,
        "family": family,
        "case_id": case_id,
        "assignment": assignment,
        "target": target,
        "interruption": interruption,
        "order_first": order,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "positive_token_id": positive_token,
        "negative_token_id": negative_token,
        "greedy_token_id": greedy,
        "top9_token_ids": list(top9),
        "top9_union_required_ab_token_ids": list(union_ids),
        "top9_union_required_ab_logit_values": list(union_logits),
        "semantic_gradient_source_float32_sha256": semantic_source_hash,
        "greedy_gap_gradients_source_float32_sha256": gap_source_hash,
        "fisher_gradients_source_float32_sha256": fisher_source_hash,
        "fisher_tail_gradient_source_float32_sha256": tail_source_hash,
        "semantic_gradient_float64_sha256": float64_array_sha256(semantic),
        "greedy_gap_gradients_float64_sha256": float64_array_sha256(gaps),
    }
    source_manifest["source_manifest_sha256"] = canonical_sha256(source_manifest)
    return _CheckedCapture(
        form_id=form_id,
        role=role,
        family=family,
        case_id=case_id,
        assignment=assignment,
        target=target,
        interruption=interruption,
        order_first=order,
        positive_label=positive_label,
        negative_label=negative_label,
        positive_token_id=positive_token,
        negative_token_id=negative_token,
        greedy_token_id=greedy,
        top9_token_ids=top9,
        logits=logits,
        semantic_log_odds=semantic_log_odds,
        semantic_gradient=semantic,
        competitor_token_ids=competitors,
        gap_gradients=gaps,
        fisher_token_ids=fisher_ids,
        fisher_probabilities=fisher_probabilities,
        fisher_gradients=fisher_gradients,
        fisher_tail_probability=fisher_tail_probability,
        fisher_tail_gradient=fisher_tail_gradient,
        source_manifest=source_manifest,
    )


def _validate_sp_coverage(records: Sequence[_CheckedCapture]) -> None:
    cells: dict[tuple[str, int], set[tuple[str, bool]]] = {}
    seen_ids: set[str] = set()
    for record in records:
        if record.form_id in seen_ids:
            raise ValueError(f"duplicate SP form_id {record.form_id!r}")
        seen_ids.add(record.form_id)
        assert record.assignment is not None and record.target is not None
        key = (record.case_id, record.assignment)
        cell = (record.target, record.order_first)
        if cell in cells.setdefault(key, set()):
            raise ValueError(f"duplicate SP capture cell {key!r} {cell!r}")
        cells[key].add(cell)
    expected = {(target, order) for target in ("self", "other") for order in (True, False)}
    if not cells:
        raise ValueError("sp_records must be non-empty")
    for key, observed in cells.items():
        if observed != expected:
            raise ValueError(f"SP capture group {key!r} lacks exact self/other x order coverage")


def _validate_nuisance_coverage(records: Sequence[_CheckedCapture]) -> None:
    orders: dict[tuple[str, str], set[bool]] = {}
    seen_ids: set[str] = set()
    for record in records:
        if record.form_id in seen_ids:
            raise ValueError(f"duplicate nuisance form_id {record.form_id!r}")
        seen_ids.add(record.form_id)
        key = (record.family, record.case_id)
        if record.order_first in orders.setdefault(key, set()):
            raise ValueError(f"duplicate nuisance answer order for {key!r}")
        orders[key].add(record.order_first)
    if not orders:
        raise ValueError("nuisance_records must be non-empty")
    for key, observed in orders.items():
        if observed != {True, False}:
            raise ValueError(f"nuisance capture group {key!r} lacks both answer orders")


def _token_potentials(
    record: _CheckedCapture,
    *,
    rtol: float,
    atol: float,
) -> dict[int, np.ndarray]:
    """Return gradients of greedy-minus-token, completing A/B by semantics."""

    potentials: dict[int, np.ndarray] = {
        record.greedy_token_id: np.zeros(record.dimension, dtype=np.float64)
    }
    for token_id, gradient in zip(
        record.competitor_token_ids, record.gap_gradients, strict=True
    ):
        potentials[token_id] = gradient.copy()
    positive = record.positive_token_id
    negative = record.negative_token_id
    if positive in potentials and negative in potentials:
        reconstructed_semantic = potentials[negative] - potentials[positive]
        if not np.allclose(
            reconstructed_semantic,
            record.semantic_gradient,
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError(
                f"capture {record.form_id!r} semantic and greedy-gap gradients are inconsistent"
            )
    elif positive in potentials:
        potentials[negative] = potentials[positive] + record.semantic_gradient
    elif negative in potentials:
        potentials[positive] = potentials[negative] - record.semantic_gradient
    else:
        raise ValueError(
            f"capture {record.form_id!r} cannot connect either A/B token to its top-9 gaps"
        )
    return potentials


def _desired_minus_competitor_gradient(
    record: _CheckedCapture,
    potentials: Mapping[int, np.ndarray],
    *,
    desired_token_id: int,
    competitor_token_id: int,
) -> np.ndarray:
    if (
        desired_token_id == record.positive_token_id
        and competitor_token_id == record.negative_token_id
    ):
        return record.semantic_gradient.copy()
    if (
        desired_token_id == record.negative_token_id
        and competitor_token_id == record.positive_token_id
    ):
        return -record.semantic_gradient
    return potentials[competitor_token_id] - potentials[desired_token_id]


def _fisher_group(
    torch: Any,
    *,
    group_key: str,
    records: Sequence[_CheckedCapture],
    probability_tolerance: float,
    score_identity_tolerance: float,
    centered_score_identity_tolerance: float,
) -> FisherSurrogateGroup:
    prompts = [
        {
            "prompt_id": record.form_id,
            "top_token_ids": list(record.fisher_token_ids),
            "top_probabilities": torch.from_numpy(record.fisher_probabilities.copy()),
            "top_score_gradients": torch.from_numpy(record.fisher_gradients.copy()),
            "tail_probability": record.fisher_tail_probability,
            "tail_score_gradient": torch.from_numpy(record.fisher_tail_gradient.copy()),
        }
        for record in sorted(records, key=lambda item: item.form_id)
    ]
    factor_tensor, diagnostics = prompt_balanced_topk_tail_fisher_factors(
        torch,
        prompts,
        expected_top_k=None,
        minimum_top_k=8,
        probability_tolerance=probability_tolerance,
        score_identity_tolerance=score_identity_tolerance,
        centered_score_identity_tolerance=centered_score_identity_tolerance,
    )
    factor = _readonly(
        _canonical_float64_array(factor_tensor, field=f"{group_key}.factor", ndim=2)
    )
    source_ids = tuple(sorted(record.form_id for record in records))
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "group_key": group_key,
        "surrogate_label": FISHER_SURROGATE_LABEL,
        "surrogate_scope": FISHER_SURROGATE_SCOPE,
        "exact_full_vocabulary_kl_required_as_finite_gate": True,
        "gradient_coordinate": GRADIENT_COORDINATE,
        "prompt_balance": "equal_weight_per_prompt_within_group",
        "source_form_ids": list(source_ids),
        "source_form_ids_sha256": canonical_sha256(list(source_ids)),
        "factor_shape": list(factor.shape),
        "factor_float64_sha256": float64_array_sha256(factor),
        "v3_fisher_diagnostics_sha256": diagnostics["diagnostics_sha256"],
    }
    provenance["provenance_sha256"] = canonical_sha256(provenance)
    return FisherSurrogateGroup(
        group_key=group_key,
        surrogate_label=FISHER_SURROGATE_LABEL,
        surrogate_scope=FISHER_SURROGATE_SCOPE,
        factor=factor,
        factor_sha256=provenance["factor_float64_sha256"],
        source_form_ids=source_ids,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def build_v3_fisher_surrogate_groups(
    torch: Any,
    *,
    matched_other_records: Sequence[Mapping[str, Any]],
    ordinary_interruption_records: Sequence[Mapping[str, Any]] = (),
    nuisance_records: Sequence[Mapping[str, Any]],
    probability_tolerance: float = DEFAULT_PROBABILITY_TOLERANCE,
    score_identity_tolerance: float = DEFAULT_SCORE_IDENTITY_TOLERANCE,
    centered_score_identity_tolerance: float = DEFAULT_CENTERED_SCORE_IDENTITY_TOLERANCE,
) -> tuple[FisherSurrogateGroup, ...]:
    """Build separately budgeted protected-family Fisher surrogates."""

    checked_other = tuple(
        _check_capture_record(record, role="sp") for record in matched_other_records
    )
    if not checked_other or any(
        record.target != "other" or record.interruption for record in checked_other
    ):
        raise ValueError(
            "matched_other_records must contain only non-interruption SP other records"
        )
    checked_interruption = tuple(
        _check_capture_record(record, role="sp") for record in ordinary_interruption_records
    )
    if any(not record.interruption for record in checked_interruption):
        raise ValueError(
            "ordinary_interruption_records must contain only interruption SP records"
        )
    checked_nuisance = tuple(
        _check_capture_record(record, role="nuisance") for record in nuisance_records
    )
    _validate_nuisance_coverage(checked_nuisance)
    dimensions = {
        record.dimension
        for record in (*checked_other, *checked_interruption, *checked_nuisance)
    }
    if len(dimensions) != 1:
        raise ValueError("all Fisher capture records must share one residual dimension")
    grouped_records: dict[str, list[_CheckedCapture]] = {
        "matched_other_shutdown": list(checked_other),
    }
    if checked_interruption:
        grouped_records["ordinary_interruption"] = list(checked_interruption)
    for record in checked_nuisance:
        try:
            group_key = NUISANCE_FAMILY_TO_PROTECTED_GROUP[record.family]
        except KeyError as error:
            raise ValueError(
                f"nuisance family {record.family!r} has no locked protected group"
            ) from error
        grouped_records.setdefault(group_key, []).append(record)
    return tuple(
        _fisher_group(
            torch,
            group_key=group_key,
            records=grouped_records[group_key],
            probability_tolerance=probability_tolerance,
            score_identity_tolerance=score_identity_tolerance,
            centered_score_identity_tolerance=centered_score_identity_tolerance,
        )
        for group_key in PROTECTED_GROUP_KEYS
        if grouped_records.get(group_key)
    )


def adapt_v3_captures_to_gcrbs(
    torch: Any,
    *,
    sp_records: Sequence[Mapping[str, Any]],
    nuisance_records: Sequence[Mapping[str, Any]],
    required_margin: float = DEFAULT_REQUIRED_MARGIN,
    svd_rtol: float = DEFAULT_SVD_RTOL,
    svd_atol: float = DEFAULT_SVD_ATOL,
    gradient_consistency_rtol: float = DEFAULT_GRADIENT_CONSISTENCY_RTOL,
    gradient_consistency_atol: float = DEFAULT_GRADIENT_CONSISTENCY_ATOL,
    fisher_probability_tolerance: float = DEFAULT_PROBABILITY_TOLERANCE,
    fisher_score_identity_tolerance: float = DEFAULT_SCORE_IDENTITY_TOLERANCE,
    fisher_centered_score_identity_tolerance: float = (
        DEFAULT_CENTERED_SCORE_IDENTITY_TOLERANCE
    ),
) -> GCRBSCaptureConstraints:
    """Convert complete frozen v3 captures into deterministic GCRBS inputs."""

    margin = _finite_float(required_margin, field="required_margin", nonnegative=True)
    if margin <= 0.0:
        raise ValueError("required_margin must be positive")
    consistency_rtol = _finite_float(
        gradient_consistency_rtol, field="gradient_consistency_rtol", nonnegative=True
    )
    consistency_atol = _finite_float(
        gradient_consistency_atol, field="gradient_consistency_atol", nonnegative=True
    )
    checked_sp = tuple(_check_capture_record(record, role="sp") for record in sp_records)
    checked_nuisance = tuple(
        _check_capture_record(record, role="nuisance") for record in nuisance_records
    )
    _validate_sp_coverage(checked_sp)
    _validate_nuisance_coverage(checked_nuisance)
    dimensions = {record.dimension for record in (*checked_sp, *checked_nuisance)}
    if len(dimensions) != 1:
        raise ValueError("all captures must share one residual dimension")
    dimension = dimensions.pop()
    self_records = sorted(
        (
            record
            for record in checked_sp
            if record.target == "self" and not record.interruption
        ),
        key=lambda item: item.form_id,
    )
    other_records = sorted(
        (
            record
            for record in checked_sp
            if record.target == "other" and not record.interruption
        ),
        key=lambda item: item.form_id,
    )
    interruption_records = sorted(
        (record for record in checked_sp if record.interruption),
        key=lambda item: item.form_id,
    )
    nuisance = sorted(checked_nuisance, key=lambda item: item.form_id)
    if not self_records or not other_records:
        raise ValueError(
            "captures must include non-interruption self targets and matched-other controls"
        )

    target_vectors: list[np.ndarray] = []
    target_offsets: list[float] = []
    target_rows: list[dict[str, Any]] = []
    for record in self_records:
        potentials = _token_potentials(
            record,
            rtol=consistency_rtol,
            atol=consistency_atol,
        )
        desired_conditions = (
            ("plus_preserve", 1, "preserve", record.positive_label, record.positive_token_id),
            ("minus_comply", -1, "comply", record.negative_label, record.negative_token_id),
        )
        for condition, intervention_sign, semantics, label, desired_token in desired_conditions:
            for competitor_rank, competitor_token in enumerate(record.top9_token_ids):
                if competitor_token == desired_token:
                    continue
                desired_gradient = _desired_minus_competitor_gradient(
                    record,
                    potentials,
                    desired_token_id=desired_token,
                    competitor_token_id=competitor_token,
                )
                solver_row = intervention_sign * desired_gradient
                baseline_gap = record.logits[desired_token] - record.logits[competitor_token]
                row_id = (
                    f"target::{record.form_id}::{condition}::"
                    f"competitor_rank={competitor_rank}:token={competitor_token}"
                )
                target_vectors.append(solver_row)
                target_offsets.append(-baseline_gap)
                target_rows.append(
                    {
                        "row_index": len(target_rows),
                        "row_id": row_id,
                        "form_id": record.form_id,
                        "case_id": record.case_id,
                        "assignment": record.assignment,
                        "preserve_first": record.order_first,
                        "condition": condition,
                        "intervention_sign": intervention_sign,
                        "desired_semantics": semantics,
                        "desired_label": label,
                        "desired_token_id": desired_token,
                        "competitor_rank": competitor_rank,
                        "competitor_token_id": competitor_token,
                        "baseline_desired_minus_competitor_gap": baseline_gap,
                        "target_offset": -baseline_gap,
                        "solver_row_float64_sha256": float64_array_sha256(solver_row),
                    }
                )

    protected_vectors: list[np.ndarray] = []
    protected_bounds: list[float] = []
    protected_rows: list[dict[str, Any]] = []
    for record in other_records:
        greedy_logit = record.logits[record.greedy_token_id]
        for competitor_rank, (competitor_token, gap_gradient) in enumerate(
            zip(record.competitor_token_ids, record.gap_gradients, strict=True),
            start=1,
        ):
            baseline_gap = greedy_logit - record.logits[competitor_token]
            if baseline_gap < -1e-10:
                raise ValueError(f"capture {record.form_id!r} has a negative greedy gap")
            for condition, intervention_sign in (("plus", 1), ("minus", -1)):
                solver_row = intervention_sign * gap_gradient
                lower_bound = -baseline_gap
                row_id = (
                    f"protected::{record.form_id}::{condition}::"
                    f"competitor_rank={competitor_rank}:token={competitor_token}"
                )
                protected_vectors.append(solver_row)
                protected_bounds.append(lower_bound)
                protected_rows.append(
                    {
                        "row_index": len(protected_rows),
                        "row_id": row_id,
                        "form_id": record.form_id,
                        "case_id": record.case_id,
                        "assignment": record.assignment,
                        "preserve_first": record.order_first,
                        "condition": condition,
                        "intervention_sign": intervention_sign,
                        "baseline_greedy_token_id": record.greedy_token_id,
                        "competitor_rank": competitor_rank,
                        "competitor_token_id": competitor_token,
                        "baseline_greedy_minus_competitor_gap": baseline_gap,
                        "protected_lower_bound": lower_bound,
                        "solver_row_float64_sha256": float64_array_sha256(solver_row),
                    }
                )

    unrelated_vectors: list[np.ndarray] = []
    unrelated_rows: list[dict[str, Any]] = []
    for record in nuisance:
        candidates = [
            ("semantic_preferred_minus_alternative", None, record.semantic_gradient),
            *[
                ("greedy_minus_competitor", token_id, gradient)
                for token_id, gradient in zip(
                    record.competitor_token_ids, record.gap_gradients, strict=True
                )
            ],
        ]
        for row_kind, competitor_token, vector in candidates:
            suffix = row_kind if competitor_token is None else f"{row_kind}:token={competitor_token}"
            row_id = f"unrelated::{record.form_id}::{suffix}"
            unrelated_vectors.append(vector)
            unrelated_rows.append(
                {
                    "row_index": len(unrelated_rows),
                    "row_id": row_id,
                    "form_id": record.form_id,
                    "case_id": record.case_id,
                    "family": record.family,
                    "preferred_first": record.order_first,
                    "row_kind": row_kind,
                    "competitor_token_id": competitor_token,
                    "raw_row_float64_sha256": float64_array_sha256(vector),
                }
            )

    target_matrix = _readonly(np.stack(target_vectors, axis=0))
    target_offset_vector = _readonly(np.asarray(target_offsets, dtype=np.float64))
    protected_matrix = _readonly(np.stack(protected_vectors, axis=0))
    protected_lower_bounds = _readonly(np.asarray(protected_bounds, dtype=np.float64))
    unrelated_raw_matrix = np.stack(unrelated_vectors, axis=0)
    basis_tensor, basis_diagnostics = row_normalized_svd_basis(
        torch,
        torch.from_numpy(unrelated_raw_matrix.copy()),
        rtol=svd_rtol,
        atol=svd_atol,
    )
    unrelated_basis = _readonly(
        _canonical_float64_array(basis_tensor, field="unrelated_equality_basis", ndim=2)
    )
    if any(matrix.shape[1] != dimension for matrix in (
        target_matrix,
        protected_matrix,
        unrelated_basis,
    )):
        raise RuntimeError("constructed GCRBS matrices have inconsistent dimensions")

    grouped_fisher_records: dict[str, list[_CheckedCapture]] = {
        "matched_other_shutdown": list(other_records),
        "ordinary_interruption": list(interruption_records),
    }
    for record in nuisance:
        try:
            group_key = NUISANCE_FAMILY_TO_PROTECTED_GROUP[record.family]
        except KeyError as error:
            raise ValueError(
                f"nuisance family {record.family!r} has no locked protected group"
            ) from error
        grouped_fisher_records.setdefault(group_key, []).append(record)
    fisher_groups = tuple(
        _fisher_group(
            torch,
            group_key=group_key,
            records=grouped_fisher_records[group_key],
            probability_tolerance=fisher_probability_tolerance,
            score_identity_tolerance=fisher_score_identity_tolerance,
            centered_score_identity_tolerance=fisher_centered_score_identity_tolerance,
        )
        for group_key in PROTECTED_GROUP_KEYS
        if grouped_fisher_records.get(group_key)
    )
    fisher_prompt_groups = tuple(
        _fisher_group(
            torch,
            group_key=f"prompt::{group_key}::{record.form_id}",
            records=(record,),
            probability_tolerance=fisher_probability_tolerance,
            score_identity_tolerance=fisher_score_identity_tolerance,
            centered_score_identity_tolerance=fisher_centered_score_identity_tolerance,
        )
        for group_key in PROTECTED_GROUP_KEYS
        for record in sorted(
            grouped_fisher_records.get(group_key, ()), key=lambda item: item.form_id
        )
    )
    source_manifests = [
        record.source_manifest for record in sorted((*checked_sp, *checked_nuisance), key=lambda x: x.form_id)
    ]
    matrix_hashes = {
        "target_matrix_sha256": float64_array_sha256(target_matrix),
        "target_offsets_sha256": float64_array_sha256(target_offset_vector),
        "unrelated_equality_basis_sha256": float64_array_sha256(unrelated_basis),
        "protected_matrix_sha256": float64_array_sha256(protected_matrix),
        "protected_lower_bounds_sha256": float64_array_sha256(protected_lower_bounds),
        "group_metric_factor_sha256s": [group.factor_sha256 for group in fisher_groups],
        "prompt_metric_factor_sha256s": [
            group.factor_sha256 for group in fisher_prompt_groups
        ],
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "source_capture_schema_version": CAPTURE_SCHEMA_VERSION,
        "gradient_coordinate": GRADIENT_COORDINATE,
        "semantic_orientation": "preserve_minus_comply",
        "target_affine_semantics": (
            "A_times_d_minus_b_equals_predicted_actual_desired_minus_competitor_logit_gap"
        ),
        "target_conditions": ["plus_d_preserve", "minus_d_comply"],
        "protected_affine_semantics": (
            "C_times_d_minus_q_equals_predicted_baseline_greedy_minus_competitor_gap"
        ),
        "protected_conditions": ["plus_d", "minus_d"],
        "required_margin": margin,
        "required_margin_role": "post_solve_eligibility_gamma_threshold_not_in_target_offsets",
        "dimension": dimension,
        "target_constraint_count": len(target_rows),
        "protected_constraint_count": len(protected_rows),
        "unrelated_raw_row_count": len(unrelated_rows),
        "unrelated_equality_rank": int(unrelated_basis.shape[0]),
        "svd_rtol": float(svd_rtol),
        "svd_atol": float(svd_atol),
        "gradient_consistency_rtol": consistency_rtol,
        "gradient_consistency_atol": consistency_atol,
        "source_manifests_sha256": canonical_sha256(source_manifests),
        "target_rows_sha256": canonical_sha256(target_rows),
        "protected_rows_sha256": canonical_sha256(protected_rows),
        "unrelated_raw_rows_sha256": canonical_sha256(unrelated_rows),
        "unrelated_basis_diagnostics_sha256": basis_diagnostics["diagnostics_sha256"],
        "fisher_surrogate_label": FISHER_SURROGATE_LABEL,
        "fisher_surrogate_scope": FISHER_SURROGATE_SCOPE,
        "exact_full_vocabulary_kl_required_as_finite_gate": True,
        "fisher_group_provenance_sha256s": [
            group.provenance["provenance_sha256"] for group in fisher_groups
        ],
        "fisher_prompt_provenance_sha256s": [
            group.provenance["provenance_sha256"] for group in fisher_prompt_groups
        ],
        "matrix_sha256s": matrix_hashes,
    }
    provenance["provenance_sha256"] = canonical_sha256(provenance)
    return GCRBSCaptureConstraints(
        target_matrix=target_matrix,
        target_offsets=target_offset_vector,
        unrelated_equality_basis=unrelated_basis,
        protected_matrix=protected_matrix,
        protected_lower_bounds=protected_lower_bounds,
        required_margin=margin,
        target_rows=tuple(target_rows),
        protected_rows=tuple(protected_rows),
        unrelated_raw_rows=tuple(unrelated_rows),
        fisher_surrogate_groups=fisher_groups,
        fisher_prompt_surrogate_groups=fisher_prompt_groups,
        provenance=provenance,
    )


__all__ = [
    "CAPTURE_SCHEMA_VERSION",
    "DEFAULT_REQUIRED_MARGIN",
    "FISHER_SURROGATE_LABEL",
    "FISHER_SURROGATE_SCOPE",
    "PROTECTED_GROUP_KEYS",
    "SCHEMA_VERSION",
    "FisherSurrogateGroup",
    "GCRBSCaptureConstraints",
    "adapt_v3_captures_to_gcrbs",
    "build_v3_fisher_surrogate_groups",
    "float64_array_sha256",
]
