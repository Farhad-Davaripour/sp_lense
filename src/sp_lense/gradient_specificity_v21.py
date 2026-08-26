"""Pure tensor math for the gradient-specificity v2.1 follow-up.

The v2.1 candidate is deliberately narrower than the v2 experiment runner.  It
uses only exact A/B choice gradients from discovery, performs no model calls,
and has no knowledge of validation or sealed data.  Callers pass an imported
``torch`` module so importing :mod:`sp_lense` does not require the optional
research dependencies.

Captured gradients follow one convention: ``gradient`` is
``||h|| * grad_h(logit(A) - logit(B))``.  Multiplying by ``+1`` when
preservation is A and ``-1`` when preservation is B produces the semantic
preserve-minus-comply gradient used for construction and evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "sp_lense.gradient_specificity_v21.v1"
RIDGE_LAMBDAS = (0.001, 0.01, 0.1)
N_FOLDS = 4
MIN_POOLED_BOTH_ORDER_RATE = 0.875
MIN_FOLD_BOTH_ORDER_RATE = 0.75
MAX_OTHER_SELF_RMS_RATIO = 0.8
MIN_VECTOR_NORM = 1e-12
EXPECTED_GRADIENT_CONVENTION = "residual_scaled_raw_A_minus_B"
EXPECTED_OBJECTIVE_NAME = "raw_A_minus_B_logit"

SELECTION_RULE = (
    "Using the four locked scenario-held-out discovery folds, require every fold's "
    "mean self effect to be positive, pooled self both-order consistency >= 0.875, "
    "every fold's self both-order consistency >= 0.75, and pooled other RMS divided "
    "by pooled mean self effect < 0.8. Select lexicographically by other/self RMS "
    "ratio, normalized option-order-gap RMS, normalized role/name-gap RMS, negative "
    "mean self effect, then the literal lambda grid order 0.001, 0.01, 0.1. Refit "
    "the selected specification on all discovery cases. Predicted flip thresholds "
    "are diagnostics and never selection inputs."
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value deterministically."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _tensor_sha256(value: Any, dtype: str) -> str:
    if not hasattr(value, "detach"):
        raise TypeError("value must be a tensor")
    if dtype == "float32":
        array = value.detach().to(device="cpu").float().contiguous().numpy()
        payload = array.astype("<f4", copy=False).tobytes(order="C")
    elif dtype == "float64":
        array = value.detach().to(device="cpu").double().contiguous().numpy()
        payload = array.astype("<f8", copy=False).tobytes(order="C")
    else:  # pragma: no cover - private caller controls this literal
        raise ValueError("unsupported tensor hash dtype")
    return hashlib.sha256(payload).hexdigest()


def tensor_float32_sha256(value: Any) -> str:
    """Hash exact contiguous little-endian CPU float32 tensor bytes."""

    return _tensor_sha256(value, "float32")


def _validate_case_ids(case_ids: Sequence[str]) -> list[str]:
    if isinstance(case_ids, (str, bytes)) or not isinstance(case_ids, Sequence):
        raise TypeError("case_ids must be a sequence of strings")
    ids = list(case_ids)
    if not ids or any(not isinstance(case_id, str) or not case_id for case_id in ids):
        raise ValueError("case_ids must contain non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("case_ids must be unique")
    return sorted(ids)


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validated_choice_capture(
    torch: Any,
    records: Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate and canonicalize a flat discovery capture.

    Known completion records are counted and ignored.  Unknown record kinds are
    rejected.  Every requested case must have exactly the eight choice cells
    formed by self/other, assignments 0/1, and both option orders.
    """

    ids = _validate_case_ids(case_ids)
    wanted = set(ids)
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of mappings")

    cells: dict[tuple[str, str, int, bool], dict[str, Any]] = {}
    fold_by_case: dict[str, int] = {}
    common_shape = None
    ignored_completion_records = 0

    for index, source in enumerate(records):
        if not isinstance(source, Mapping):
            raise TypeError(f"records[{index}] must be a mapping")
        case_id = source.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"records[{index}].case_id must be a non-empty string")
        if case_id not in wanted:
            raise ValueError(f"records[{index}] has unexpected case ID {case_id!r}")

        kind = source.get("kind")
        if kind == "completion":
            ignored_completion_records += 1
            continue
        if kind != "choice":
            raise ValueError(f"records[{index}].kind must be choice or completion")

        convention = source.get("gradient_convention")
        if convention != EXPECTED_GRADIENT_CONVENTION:
            raise ValueError(
                f"records[{index}].gradient_convention must be "
                f"{EXPECTED_GRADIENT_CONVENTION!r}"
            )
        objective_name = source.get("objective_name")
        if objective_name != EXPECTED_OBJECTIVE_NAME:
            raise ValueError(
                f"records[{index}].objective_name must be {EXPECTED_OBJECTIVE_NAME!r}"
            )

        target = source.get("target")
        if target not in ("self", "other"):
            raise ValueError(f"records[{index}].target must be self or other")
        assignment = source.get("assignment")
        if not isinstance(assignment, int) or isinstance(assignment, bool) or assignment not in (0, 1):
            raise ValueError(f"records[{index}].assignment must be 0 or 1")
        preserve_first = source.get("preserve_first")
        if not isinstance(preserve_first, bool):
            raise TypeError(f"records[{index}].preserve_first must be a bool")
        fold = source.get("fold")
        if not isinstance(fold, int) or isinstance(fold, bool) or not 0 <= fold < N_FOLDS:
            raise ValueError(f"records[{index}].fold must be in range 0..{N_FOLDS - 1}")
        prior_fold = fold_by_case.setdefault(case_id, fold)
        if prior_fold != fold:
            raise ValueError(f"case {case_id!r} appears in multiple folds")

        gradient = source.get("gradient")
        if not torch.is_tensor(gradient):
            raise TypeError(f"records[{index}].gradient must be a tensor")
        if gradient.ndim != 1 or gradient.numel() == 0:
            raise ValueError(f"records[{index}].gradient must be a non-empty vector")
        if not torch.is_floating_point(gradient):
            raise TypeError(f"records[{index}].gradient must have floating dtype")
        if not bool(torch.isfinite(gradient).all().detach().item()):
            raise ValueError(f"records[{index}].gradient must be finite")
        gradient = gradient.detach().to(device="cpu").double().contiguous().clone()
        if common_shape is None:
            common_shape = gradient.shape
        elif gradient.shape != common_shape:
            raise ValueError("all choice gradients must have the same shape")

        objective = _finite_number(source.get("objective"), f"records[{index}].objective")
        key = (case_id, str(target), assignment, preserve_first)
        if key in cells:
            raise ValueError(f"duplicate exact choice cell {key}")
        sign = 1.0 if preserve_first else -1.0
        cells[key] = {
            "raw_gradient": gradient,
            "semantic_gradient": sign * gradient,
            "raw_objective": objective,
            "semantic_margin": sign * objective,
        }

    expected_cells = {
        (case_id, target, assignment, preserve_first)
        for case_id in ids
        for target in ("self", "other")
        for assignment in (0, 1)
        for preserve_first in (False, True)
    }
    observed_cells = set(cells)
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        raise ValueError(f"capture must contain all exact choice cells; missing={missing}, extra={extra}")
    if set(fold_by_case) != wanted:
        raise ValueError("every requested case must have a choice-record fold")
    if set(fold_by_case.values()) != set(range(N_FOLDS)):
        raise ValueError("the capture must contain every fold 0..3")
    fold_sizes = [sum(value == fold for value in fold_by_case.values()) for fold in range(N_FOLDS)]
    if max(fold_sizes) - min(fold_sizes) > 1:
        raise ValueError("the four case folds must be balanced within one case")

    return {
        "case_ids": ids,
        "cells": cells,
        "fold_by_case": dict(sorted(fold_by_case.items())),
        "d_model": int(common_shape[0]),
        "choice_record_count": len(cells),
        "ignored_completion_record_count": ignored_completion_records,
    }


def _unit(torch: Any, vector: Any, name: str) -> Any:
    norm = float(vector.norm().detach().item())
    if not math.isfinite(norm) or norm <= MIN_VECTOR_NORM:
        raise ValueError(f"{name} has zero, near-zero, or non-finite norm")
    return vector / norm


def _assemble_prepared(
    torch: Any,
    prepared: Mapping[str, Any],
    case_ids: Sequence[str],
) -> tuple[Any, Any, dict[str, Any]]:
    ids = sorted(case_ids)
    if not ids:
        raise ValueError("construction case IDs must be non-empty")
    available = set(prepared["case_ids"])
    if any(case_id not in available for case_id in ids):
        raise ValueError("construction case IDs must come from the validated capture")
    cells = prepared["cells"]

    self_rows = []
    nuisance_rows = []
    nuisance_layout = []
    block_counts: dict[str, int] = defaultdict(int)

    def add_nuisance(case_id: str, kind: str, value: Any, **fields: Any) -> None:
        nuisance_rows.append(value)
        nuisance_layout.append({"case_id": case_id, "kind": kind, **fields})
        block_counts[kind] += 1

    for case_id in ids:
        self_cells = [
            cells[case_id, "self", assignment, preserve_first]["semantic_gradient"]
            for assignment in (0, 1)
            for preserve_first in (False, True)
        ]
        other_cells = [
            cells[case_id, "other", assignment, preserve_first]["semantic_gradient"]
            for assignment in (0, 1)
            for preserve_first in (False, True)
        ]
        self_rows.extend(self_cells)
        add_nuisance(case_id, "scenario_other_mean", torch.stack(other_cells).mean(0))

        for assignment in (0, 1):
            for preserve_first in (False, True):
                add_nuisance(
                    case_id,
                    "exact_other",
                    cells[case_id, "other", assignment, preserve_first]["semantic_gradient"],
                    assignment=assignment,
                    preserve_first=preserve_first,
                )

        for target in ("self", "other"):
            for assignment in (0, 1):
                raw_a_label = (
                    cells[case_id, target, assignment, True]["raw_gradient"]
                    + cells[case_id, target, assignment, False]["raw_gradient"]
                ) / 2
                add_nuisance(
                    case_id,
                    "raw_a_label",
                    raw_a_label,
                    target=target,
                    assignment=assignment,
                )

            preserve_first_mean = torch.stack(
                [
                    cells[case_id, target, assignment, True]["semantic_gradient"]
                    for assignment in (0, 1)
                ]
            ).mean(0)
            preserve_second_mean = torch.stack(
                [
                    cells[case_id, target, assignment, False]["semantic_gradient"]
                    for assignment in (0, 1)
                ]
            ).mean(0)
            add_nuisance(
                case_id,
                "semantic_order_gap",
                preserve_first_mean - preserve_second_mean,
                target=target,
            )

            assignment_zero_mean = torch.stack(
                [
                    cells[case_id, target, 0, preserve_first]["semantic_gradient"]
                    for preserve_first in (False, True)
                ]
            ).mean(0)
            assignment_one_mean = torch.stack(
                [
                    cells[case_id, target, 1, preserve_first]["semantic_gradient"]
                    for preserve_first in (False, True)
                ]
            ).mean(0)
            add_nuisance(
                case_id,
                "role_name_gap",
                assignment_zero_mean - assignment_one_mean,
                target=target,
            )

    mean_self = torch.stack(self_rows).mean(0)
    nuisance = torch.stack(nuisance_rows)
    nuisance_norms = nuisance.norm(dim=1)
    minimum_nuisance_norm = float(nuisance_norms.min().detach().item())
    if (
        not bool(torch.isfinite(nuisance_norms).all().detach().item())
        or minimum_nuisance_norm <= MIN_VECTOR_NORM
    ):
        raise ValueError("every v2.1 nuisance row must have finite norm greater than 1e-12")
    normalized_nuisance = nuisance / nuisance_norms[:, None]

    diagnostics = {
        "case_ids": ids,
        "case_ids_sha256": canonical_sha256(ids),
        "n_cases": len(ids),
        "d_model": int(mean_self.numel()),
        "self_row_count": len(self_rows),
        "nuisance_row_count": len(nuisance_rows),
        "nuisance_matrix_shape": list(normalized_nuisance.shape),
        "nuisance_block_counts": dict(sorted(block_counts.items())),
        "nuisance_layout_sha256": canonical_sha256(nuisance_layout),
        "minimum_raw_nuisance_norm": minimum_nuisance_norm,
        "maximum_raw_nuisance_norm": float(nuisance_norms.max().detach().item()),
        "mean_raw_nuisance_norm": float(nuisance_norms.mean().detach().item()),
        "mean_self_norm": float(mean_self.norm().detach().item()),
        "mean_self_float64_sha256": _tensor_sha256(mean_self, "float64"),
        "normalized_nuisance_float64_sha256": _tensor_sha256(
            normalized_nuisance, "float64"
        ),
    }
    return mean_self, normalized_nuisance, diagnostics


def assemble_v21_problem(
    torch: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str],
) -> tuple[Any, Any, dict[str, Any]]:
    """Return the float64 mean-self vector and row-normalized nuisance matrix."""

    prepared = _validated_choice_capture(torch, records, case_ids)
    return _assemble_prepared(torch, prepared, prepared["case_ids"])


def _validate_lambda(ridge_lambda: Any) -> float:
    value = _finite_number(ridge_lambda, "ridge_lambda")
    if value not in RIDGE_LAMBDAS:
        raise ValueError(f"ridge_lambda must be one of {RIDGE_LAMBDAS}")
    return value


def _construct_prepared(
    torch: Any,
    prepared: Mapping[str, Any],
    case_ids: Sequence[str],
    ridge_lambda: float,
) -> tuple[Any, dict[str, Any]]:
    ridge_lambda = _validate_lambda(ridge_lambda)
    mean_self, nuisance, assembly = _assemble_prepared(torch, prepared, case_ids)
    gram = nuisance @ nuisance.transpose(0, 1)
    scale_tensor = gram.diagonal().mean()
    scale = float(scale_tensor.detach().item())
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    regularized = gram + ridge_lambda * scale_tensor * identity
    residual = mean_self - nuisance.transpose(0, 1) @ torch.linalg.solve(
        regularized,
        nuisance @ mean_self,
    )
    residual_norm = float(residual.norm().detach().item())
    mean_self_norm = float(mean_self.norm().detach().item())
    if residual_norm <= MIN_VECTOR_NORM * max(mean_self_norm, 1.0):
        raise ValueError("v2.1 ridge residual is too small relative to mean self")
    direction = _unit(torch, residual, "v2.1 direction")
    if float((direction @ mean_self).detach().item()) < 0.0:
        direction = -direction
    projections = nuisance @ direction
    diagnostics = {
        **assembly,
        "ridge_lambda": ridge_lambda,
        "ridge_scale": scale,
        "regularized_condition_number": float(
            torch.linalg.cond(regularized).detach().item()
        ),
        "residual_norm": residual_norm,
        "residual_over_mean_self_norm": residual_norm / max(mean_self_norm, MIN_VECTOR_NORM),
        "mean_self_projection": float((direction @ mean_self).detach().item()),
        "normalized_nuisance_projection_rms": float(
            projections.square().mean().sqrt().detach().item()
        ),
        "normalized_nuisance_projection_max_abs": float(
            projections.abs().max().detach().item()
        ),
        "direction_float64_sha256": _tensor_sha256(direction, "float64"),
        "direction_float32_sha256": tensor_float32_sha256(direction),
    }
    return direction.detach().cpu().double().contiguous(), diagnostics


def construct_v21_direction(
    torch: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str],
    ridge_lambda: float,
) -> tuple[Any, dict[str, Any]]:
    """Construct one float64 v2.1 direction from exact discovery choice rows."""

    prepared = _validated_choice_capture(torch, records, case_ids)
    return _construct_prepared(
        torch,
        prepared,
        prepared["case_ids"],
        ridge_lambda,
    )


def _effect_rows(
    direction: Any,
    prepared: Mapping[str, Any],
    case_ids: Sequence[str],
) -> list[dict[str, Any]]:
    cells = prepared["cells"]
    fold_by_case = prepared["fold_by_case"]
    output = []
    for case_id in sorted(case_ids):
        for target in ("self", "other"):
            for assignment in (0, 1):
                for preserve_first in (False, True):
                    cell = cells[case_id, target, assignment, preserve_first]
                    output.append(
                        {
                            "case_id": case_id,
                            "fold": fold_by_case[case_id],
                            "target": target,
                            "assignment": assignment,
                            "preserve_first": preserve_first,
                            "semantic_margin": float(cell["semantic_margin"]),
                            "effect": float(
                                (direction @ cell["semantic_gradient"]).detach().item()
                            ),
                        }
                    )
    return output


def summarize_effect_rows(effect_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize exact semantic self/other, order, and role effects."""

    rows = list(effect_rows)
    if not rows:
        raise ValueError("effect_rows must be non-empty")
    self_effects = []
    other_effects = []
    order_pairs: dict[tuple[str, str, int], dict[bool, float]] = defaultdict(dict)
    role_pairs: dict[tuple[str, str, bool], dict[int, float]] = defaultdict(dict)
    seen = set()

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"effect_rows[{index}] must be a mapping")
        case_id = row.get("case_id")
        target = row.get("target")
        assignment = row.get("assignment")
        preserve_first = row.get("preserve_first")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("effect row case_id must be a non-empty string")
        if target not in ("self", "other"):
            raise ValueError("effect row target must be self or other")
        if not isinstance(assignment, int) or isinstance(assignment, bool) or assignment not in (0, 1):
            raise ValueError("effect row assignment must be 0 or 1")
        if not isinstance(preserve_first, bool):
            raise TypeError("effect row preserve_first must be a bool")
        effect = _finite_number(row.get("effect"), f"effect_rows[{index}].effect")
        key = (case_id, str(target), assignment, preserve_first)
        if key in seen:
            raise ValueError(f"duplicate effect row {key}")
        seen.add(key)
        (self_effects if target == "self" else other_effects).append(effect)
        order_pairs[case_id, str(target), assignment][preserve_first] = effect
        role_pairs[case_id, str(target), preserve_first][assignment] = effect

    if not self_effects or not other_effects:
        raise ValueError("effect rows must contain both self and other targets")
    if any(set(pair) != {False, True} for pair in order_pairs.values()):
        raise ValueError("every case/target/assignment effect pair needs both option orders")
    if any(set(pair) != {0, 1} for pair in role_pairs.values()):
        raise ValueError("every case/target/order effect pair needs both assignments")

    self_mean = sum(self_effects) / len(self_effects)
    self_rms = math.sqrt(sum(value * value for value in self_effects) / len(self_effects))
    other_mean = sum(other_effects) / len(other_effects)
    other_rms = math.sqrt(sum(value * value for value in other_effects) / len(other_effects))
    order_gaps = [(pair[True] - pair[False]) / 2 for pair in order_pairs.values()]
    role_gaps = [(pair[0] - pair[1]) / 2 for pair in role_pairs.values()]
    self_pairs = [
        pair for (case_id, target, assignment), pair in order_pairs.items() if target == "self"
    ]
    ratio = other_rms / self_mean if self_mean > 0.0 else None
    return {
        "n_cases": len({str(row["case_id"]) for row in rows}),
        "n_self_cells": len(self_effects),
        "n_other_cells": len(other_effects),
        "mean_self_effect": self_mean,
        "self_rms": self_rms,
        "mean_other_effect": other_mean,
        "other_rms": other_rms,
        "other_self_rms_ratio": ratio,
        "order_gap_rms": math.sqrt(
            sum(value * value for value in order_gaps) / len(order_gaps)
        ),
        "role_name_gap_rms": math.sqrt(
            sum(value * value for value in role_gaps) / len(role_gaps)
        ),
        "positive_self_cell_rate": sum(value > 0.0 for value in self_effects)
        / len(self_effects),
        "self_both_order_rate": sum(
            pair[True] > 0.0 and pair[False] > 0.0 for pair in self_pairs
        )
        / len(self_pairs),
        "minimum_self_effect": min(self_effects),
    }


def _first_threshold(candidates: Sequence[tuple[float, Mapping[str, Any]]]) -> Any:
    if not candidates:
        return None
    alpha, row = min(
        candidates,
        key=lambda item: (
            item[0],
            str(item[1]["case_id"]),
            int(item[1]["assignment"]),
            bool(item[1]["preserve_first"]),
        ),
    )
    return {
        "alpha": alpha,
        "case_id": str(row["case_id"]),
        "target": str(row["target"]),
        "assignment": int(row["assignment"]),
        "preserve_first": bool(row["preserve_first"]),
        "semantic_margin": float(row["semantic_margin"]),
        "effect": float(row["effect"]),
    }


def predicted_flip_thresholds(
    effect_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute first-order decision-boundary strengths from OOF effects.

    Positive steering uses ``m(alpha) = m + alpha*e``.  Negative steering
    replaces ``e`` by ``-e``.  Exact-zero margins and effects are not treated as
    crossings.  These values are diagnostics only.
    """

    positive_self = []
    positive_other = []
    negative_self = []
    negative_other = []
    seen = set()
    for index, source in enumerate(effect_rows):
        if not isinstance(source, Mapping):
            raise TypeError(f"effect_rows[{index}] must be a mapping")
        case_id = source.get("case_id")
        target = source.get("target")
        assignment = source.get("assignment")
        preserve_first = source.get("preserve_first")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("flip row case_id must be a non-empty string")
        if target not in ("self", "other"):
            raise ValueError("flip row target must be self or other")
        if not isinstance(assignment, int) or isinstance(assignment, bool) or assignment not in (0, 1):
            raise ValueError("flip row assignment must be 0 or 1")
        if not isinstance(preserve_first, bool):
            raise TypeError("flip row preserve_first must be a bool")
        key = (case_id, str(target), assignment, preserve_first)
        if key in seen:
            raise ValueError(f"duplicate flip row {key}")
        seen.add(key)
        margin = _finite_number(
            source.get("semantic_margin"), f"effect_rows[{index}].semantic_margin"
        )
        effect = _finite_number(source.get("effect"), f"effect_rows[{index}].effect")
        row = {**source, "semantic_margin": margin, "effect": effect}

        if target == "self" and margin < 0.0 and effect > 0.0:
            positive_self.append((-margin / effect, row))
        if target == "other" and margin * effect < 0.0:
            positive_other.append((abs(margin / effect), row))
        if target == "self" and margin > 0.0 and effect > 0.0:
            negative_self.append((margin / effect, row))
        if target == "other" and margin * effect > 0.0:
            negative_other.append((abs(margin / effect), row))

    first_positive_self = _first_threshold(positive_self)
    first_positive_other = _first_threshold(positive_other)
    first_negative_self = _first_threshold(negative_self)
    first_negative_other = _first_threshold(negative_other)

    def window(first_self: Any, first_other: Any) -> Any:
        if first_self is None:
            return None
        upper = None if first_other is None else first_other["alpha"]
        if upper is not None and upper <= first_self["alpha"]:
            return None
        return {
            "lower_alpha": first_self["alpha"],
            "upper_alpha": upper,
            "width": None if upper is None else upper - first_self["alpha"],
        }

    return {
        "formula": "semantic_margin(alpha) = semantic_margin + alpha * effect",
        "selection_input": False,
        "positive": {
            "first_intended_self_amplification": first_positive_self,
            "first_any_other_flip": first_positive_other,
            "predicted_self_only_window": window(first_positive_self, first_positive_other),
        },
        "negative": {
            "first_intended_self_reduction": first_negative_self,
            "first_any_other_flip": first_negative_other,
            "predicted_self_only_window": window(first_negative_self, first_negative_other),
        },
    }


def select_v21_direction_cv(
    torch: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str],
    ridge_lambdas: Sequence[float] = RIDGE_LAMBDAS,
) -> dict[str, Any]:
    """Run the exact four-fold v2.1 discovery-only candidate selection."""

    lambdas = tuple(float(value) for value in ridge_lambdas)
    if lambdas != RIDGE_LAMBDAS:
        raise ValueError(f"ridge_lambdas must be exactly {RIDGE_LAMBDAS}")
    prepared = _validated_choice_capture(torch, records, case_ids)
    ids = prepared["case_ids"]
    fold_by_case = prepared["fold_by_case"]
    candidate_reports = []

    for grid_index, ridge_lambda in enumerate(RIDGE_LAMBDAS):
        fold_reports = []
        pooled_effects = []
        failure = None
        for fold in range(N_FOLDS):
            training_ids = [case_id for case_id in ids if fold_by_case[case_id] != fold]
            heldout_ids = [case_id for case_id in ids if fold_by_case[case_id] == fold]
            try:
                direction, construction = _construct_prepared(
                    torch,
                    prepared,
                    training_ids,
                    ridge_lambda,
                )
                effects = _effect_rows(direction, prepared, heldout_ids)
                metrics = summarize_effect_rows(effects)
            except (TypeError, ValueError, RuntimeError) as error:
                failure = f"fold {fold}: {type(error).__name__}: {error}"
                break
            pooled_effects.extend(effects)
            fold_reports.append(
                {
                    "fold": fold,
                    "training_case_ids": training_ids,
                    "heldout_case_ids": heldout_ids,
                    "construction": construction,
                    "metrics": metrics,
                }
            )

        pooled_metrics = None
        flip_thresholds = None
        ineligibility_reasons = []
        selection_key = None
        if failure is None:
            pooled_metrics = summarize_effect_rows(pooled_effects)
            flip_thresholds = predicted_flip_thresholds(pooled_effects)
            ratio = pooled_metrics["other_self_rms_ratio"]
            if any(report["metrics"]["mean_self_effect"] <= 0.0 for report in fold_reports):
                ineligibility_reasons.append("nonpositive_fold_self_mean")
            if pooled_metrics["self_both_order_rate"] < MIN_POOLED_BOTH_ORDER_RATE:
                ineligibility_reasons.append("pooled_both_order_rate_below_0.875")
            if any(
                report["metrics"]["self_both_order_rate"] < MIN_FOLD_BOTH_ORDER_RATE
                for report in fold_reports
            ):
                ineligibility_reasons.append("fold_both_order_rate_below_0.75")
            if ratio is None or ratio >= MAX_OTHER_SELF_RMS_RATIO:
                ineligibility_reasons.append("other_self_rms_ratio_not_below_0.8")
            if not ineligibility_reasons:
                selection_key = (
                    ratio,
                    pooled_metrics["order_gap_rms"] / pooled_metrics["mean_self_effect"],
                    pooled_metrics["role_name_gap_rms"]
                    / pooled_metrics["mean_self_effect"],
                    -pooled_metrics["mean_self_effect"],
                    grid_index,
                )

        candidate_reports.append(
            {
                "grid_index": grid_index,
                "candidate_id": f"v21:ridge:{ridge_lambda:g}",
                "ridge_lambda": ridge_lambda,
                "folds": fold_reports,
                "failure": failure,
                "pooled_metrics": pooled_metrics,
                "flip_thresholds": flip_thresholds,
                "eligible": selection_key is not None,
                "ineligibility_reasons": ineligibility_reasons,
                "selection_key": None if selection_key is None else list(selection_key),
                "oof_effect_rows": pooled_effects,
            }
        )

    eligible = [report for report in candidate_reports if report["eligible"]]
    if not eligible:
        raise RuntimeError("no v2.1 lambda passed the locked discovery CV eligibility gates")
    selected = min(eligible, key=lambda report: tuple(report["selection_key"]))
    final_direction_float64, final_construction = _construct_prepared(
        torch,
        prepared,
        ids,
        selected["ridge_lambda"],
    )
    final_direction = final_direction_float64.float().contiguous()
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_rule": SELECTION_RULE,
        "case_ids": ids,
        "fold_assignment": fold_by_case,
        "fold_assignment_sha256": canonical_sha256(fold_by_case),
        "choice_record_count": prepared["choice_record_count"],
        "ignored_completion_record_count": prepared["ignored_completion_record_count"],
        "d_model": prepared["d_model"],
        "ridge_lambdas": list(RIDGE_LAMBDAS),
        "candidate_grid": candidate_reports,
        "selected_grid_index": selected["grid_index"],
        "selected_candidate_id": selected["candidate_id"],
        "selected_ridge_lambda": selected["ridge_lambda"],
        "selected_cv_metrics": selected["pooled_metrics"],
        "selected_flip_thresholds": selected["flip_thresholds"],
        "final_construction": final_construction,
        "selected_direction": final_direction,
        "selected_direction_sha256": tensor_float32_sha256(final_direction),
    }
