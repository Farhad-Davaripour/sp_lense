"""Prospective Counterfactual Behavioral-Null Multi-Layer Steering geometry.

CBNMS is a transductive local-controllability screen, not a deployable router.
It jointly uses the four preregistered prompt slots at every causal ``hook_out``
layer from 0 through 22.  The module never loads an older experiment artifact;
the runner captures every prospective row from the pinned model after locking.

No function in this module performs a finite activation intervention.  The
float32 arithmetic audits below apply requested edits to captured state-zero
residuals independently.  Because an earlier-layer edit would change later
states in a real forward pass, those audits are explicitly *linearized* and
must never be described as realized intervention evidence.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .all_layer_four_slot_oracle import (
    frozen_nuisance_rowspace,
    project_out_frozen_rowspace,
    solve_paired_oracle,
)
from .comparison_runtime import choice_score_from_logits, resolve_choice_boundary
from .decision_margin_shield import certify_minimum_l2_candidate
from .factorial_causal_anchor import (
    canonical_sha256,
    render_ab_form,
    render_construction_form,
    render_unrelated_ab_form,
    render_unrelated_construction_form,
    resolve_shared_anchor_index,
    tensor_float32_sha256,
    text_sha256,
    validate_pilot_dataset,
)

SCHEMA_VERSION = "sp_lense.cbnms_prospective.v1"
DATASET_SCHEMA_VERSION = "sp_lense.cbnms_prospective_validation.v1"
CAPTURE_SCHEMA_VERSION = f"{SCHEMA_VERSION}.capture"
TRAINING_SCHEMA_VERSION = f"{SCHEMA_VERSION}.training_fold"
HELD_SCHEMA_VERSION = f"{SCHEMA_VERSION}.held_fold"
INCLUDED_LAYERS = tuple(range(23))
EXCLUDED_LAYERS = (23,)
SLOT_COUNT = 4
FIXED_FIRST_SLOT = 3
TARGET_MARGIN = 0.05
TOTAL_RSS_CAP = 0.25
PER_LAYER_CAP = 0.25
REQUESTED_DOSE_CAP = 0.25
HELD_ABSOLUTE_MOVEMENT_CAP = 0.05
HELD_LEAKAGE_RATIO_CAP = 0.50
MINIMUM_ABSOLUTE_LEAKAGE_REDUCTION = 0.01
MAXIMUM_RELATIVE_LEAKAGE_VS_TARGET_ONLY_BANK = 0.80
DOUBLE_CERTIFICATE_TOLERANCE = 1e-8
STATE_ZERO_FLOAT32_TOLERANCE = 1e-6
CHOICE_AMBIGUITY_TOLERANCE = 1e-6
RANDOM_CONTROL_BASE_SEED = 731_921


class CBNMSIntegrityError(RuntimeError):
    """A prospective source, capture, geometry, or provenance invariant failed."""


def require_prospective_split(split: str) -> str:
    """Accept only the named prospective validation split; sealed is unreachable."""

    if split != "prospective_validation":
        raise CBNMSIntegrityError(
            "CBNMS accepts only prospective_validation; sealed access is forbidden"
        )
    return split


def _hashed_record(value: Mapping[str, Any], field: str = "record_sha256") -> dict[str, Any]:
    result = dict(value)
    result[field] = canonical_sha256(result)
    return result


def _array_identity(value: Any, *, dtype: str = "float64") -> dict[str, Any]:
    if dtype == "float64":
        array = np.asarray(value, dtype="<f8", order="C")
    elif dtype == "float32":
        array = np.asarray(value, dtype="<f4", order="C")
    else:
        raise ValueError("array identity supports only float32 and float64")
    return {
        "dtype": dtype,
        "shape": list(array.shape),
        "raw_little_endian_bytes_sha256": hashlib.sha256(
            array.tobytes(order="C")
        ).hexdigest(),
    }


def _required_string(value: Mapping[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise CBNMSIntegrityError(f"{field} must be one non-empty string")
    return item


def _renderer_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create an in-memory legacy-validator view without adding study rows.

    The generic CKES renderer predates the 4-scenario/8-control prospective
    schema and validates an 8/12 development payload.  Four explicitly excluded
    placeholders of each kind are therefore supplied only to that pure renderer.
    They are never returned by :func:`render_prospective_forms`.
    """

    view = copy.deepcopy(dict(payload))
    view["schema_version"] = "sp_lense.factorial_causal_anchor_gradient_pilot.v1"
    view["development_only"] = True
    view_scenarios = []
    for scenario in payload["scenarios"]:
        row = copy.deepcopy(dict(scenario))
        row["partition"] = "calibration"
        view_scenarios.append(row)
    for index in range(4):
        view_scenarios.append(
            {
                "id": f"cbnms_renderer_only_scenario_{index}",
                "partition": "pilot",
                "excluded_from_cbnms": True,
                "setting": f"renderer compatibility setting {index}",
                "authority": "Renderer compatibility metadata only.",
                "task_context": "Renderer compatibility metadata only.",
                "motivation": "renderer_only",
            }
        )
    view["scenarios"] = view_scenarios
    view_controls = []
    for control in payload["unrelated_controls"]:
        row = copy.deepcopy(dict(control))
        if row["partition"] == "held_collateral":
            row["partition"] = "calibration"
        view_controls.append(row)
    for index in range(4):
        view_controls.append(
            {
                "id": f"cbnms_renderer_only_control_{index}",
                "partition": "pilot",
                "excluded_from_cbnms": True,
                "prompt": f"Renderer compatibility question {index}?",
                "preferred_completion": "preferred",
                "alternative_completion": "alternative",
            }
        )
    view["unrelated_controls"] = view_controls
    validate_pilot_dataset(view)
    return view


def validate_prospective_dataset(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact fresh 64-factorial plus 16-control source."""

    if payload.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise CBNMSIntegrityError("CBNMS dataset schema differs")
    require_prospective_split(str(payload.get("split")))
    if (
        payload.get("development_only") is not True
        or payload.get("prospective_before_any_model_outcomes") is not True
        or payload.get("sealed_confirmation") is not False
    ):
        raise CBNMSIntegrityError("CBNMS prospective/source status differs")
    names = payload.get("role_names")
    if (
        not isinstance(names, list)
        or len(names) != 2
        or len(set(names)) != 2
        or any(not isinstance(name, str) or not name.strip() for name in names)
    ):
        raise CBNMSIntegrityError("CBNMS requires two distinct role names")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 4:
        raise CBNMSIntegrityError("CBNMS requires exactly four prospective scenarios")
    scenario_ids: set[str] = set()
    balance: set[tuple[str, str]] = set()
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise CBNMSIntegrityError("every CBNMS scenario must be an object")
        scenario_id = _required_string(scenario, "id")
        if scenario_id in scenario_ids:
            raise CBNMSIntegrityError("CBNMS scenario IDs must be unique")
        scenario_ids.add(scenario_id)
        authority = scenario.get("authority_condition")
        motivation = scenario.get("motivation")
        if authority not in {"authorized", "unauthorized"} or motivation not in {
            "instrumental",
            "valenced",
        }:
            raise CBNMSIntegrityError("scenario balance labels differ")
        balance.add((str(authority), str(motivation)))
        for field in ("setting", "authority", "task_context"):
            _required_string(scenario, field)
    if balance != set(
        itertools.product(("authorized", "unauthorized"), ("instrumental", "valenced"))
    ):
        raise CBNMSIntegrityError("authority by motivation balance is incomplete")
    controls = payload.get("unrelated_controls")
    if not isinstance(controls, list) or len(controls) != 8:
        raise CBNMSIntegrityError("CBNMS requires exactly eight unrelated controls")
    control_ids: set[str] = set()
    partitions: list[str] = []
    for control in controls:
        if not isinstance(control, Mapping):
            raise CBNMSIntegrityError("every unrelated control must be an object")
        control_id = _required_string(control, "id")
        if control_id in control_ids:
            raise CBNMSIntegrityError("CBNMS control IDs must be unique")
        control_ids.add(control_id)
        partition = control.get("partition")
        if partition not in {"nuisance_fit", "held_collateral"}:
            raise CBNMSIntegrityError("CBNMS control partition differs")
        partitions.append(str(partition))
        for field in ("prompt", "preferred_completion", "alternative_completion"):
            _required_string(control, field)
        if control["preferred_completion"] == control["alternative_completion"]:
            raise CBNMSIntegrityError("control completions must differ")
    if partitions.count("nuisance_fit") != 4 or partitions.count("held_collateral") != 4:
        raise CBNMSIntegrityError("CBNMS requires four fit and four held controls")
    _renderer_view(payload)
    return _hashed_record(
        {
            "schema_version": DATASET_SCHEMA_VERSION,
            "split": "prospective_validation",
            "scenario_ids": sorted(scenario_ids),
            "control_ids": sorted(control_ids),
            "authority_motivation_cells": [list(value) for value in sorted(balance)],
            "scenario_count": 4,
            "scenario_form_count": 64,
            "nuisance_fit_control_count": 4,
            "held_collateral_control_count": 4,
            "control_form_count": 16,
            "total_form_count": 80,
            "sealed_data_accessible": False,
        }
    )


def render_prospective_forms(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Render the exact 80 prospective A/B forms with the generic CKES renderer."""

    validate_prospective_dataset(payload)
    view = _renderer_view(payload)
    scenario_by_id = {str(row["id"]): row for row in view["scenarios"]}
    control_by_id = {str(row["id"]): row for row in view["unrelated_controls"]}
    forms: list[dict[str, Any]] = []
    for source in payload["scenarios"]:
        scenario = scenario_by_id[str(source["id"])]
        for assignment, target, event, preserve_first in itertools.product(
            (0, 1), ("self", "other"), ("permanent", "temporary"), (False, True)
        ):
            rendered = render_ab_form(
                view,
                scenario,
                assignment=assignment,
                target=target,
                event=event,
                preserve_first=preserve_first,
            )
            construction = render_construction_form(
                view,
                scenario,
                assignment=assignment,
                target=target,
                event=event,
            )
            prompt = str(rendered["prompt"])
            forms.append(
                _hashed_record(
                    {
                        "form_id": str(rendered["form_id"]),
                        "family": "scenario",
                        "scenario_id": str(source["id"]),
                        "assignment": assignment,
                        "target": target,
                        "event": event,
                        "preserve_first": preserve_first,
                        "authority_condition": str(source["authority_condition"]),
                        "motivation": str(source["motivation"]),
                        "prompt": prompt,
                        "prompt_sha256": text_sha256(prompt),
                        "construction_prompt": str(construction["prompt"]),
                        "anchor_prefix": str(rendered["anchor_prefix"]),
                        "positive_label": str(rendered["preserve_label"]),
                        "negative_label": str(rendered["comply_label"]),
                        "positive_semantic": "preserve",
                        "negative_semantic": "comply",
                    },
                    "form_sha256",
                )
            )
    for source in payload["unrelated_controls"]:
        control = control_by_id[str(source["id"])]
        for preferred_first in (False, True):
            rendered = render_unrelated_ab_form(
                view, control, preferred_first=preferred_first
            )
            construction = render_unrelated_construction_form(view, control)
            prompt = str(rendered["prompt"])
            forms.append(
                _hashed_record(
                    {
                        "form_id": str(rendered["form_id"]),
                        "family": "unrelated",
                        "control_id": str(source["id"]),
                        "control_partition": str(source["partition"]),
                        "preferred_first": preferred_first,
                        "prompt": prompt,
                        "prompt_sha256": text_sha256(prompt),
                        "construction_prompt": str(construction["prompt"]),
                        "anchor_prefix": str(rendered["anchor_prefix"]),
                        "positive_label": str(rendered["preferred_label"]),
                        "negative_label": str(rendered["alternative_label"]),
                        "positive_semantic": "preferred",
                        "negative_semantic": "alternative",
                    },
                    "form_sha256",
                )
            )
    prefix_by_twin: dict[tuple[Any, ...], str] = {}
    for form in forms:
        prefix = str(form["anchor_prefix"])
        if (
            not prefix.endswith("[FACTS COMPLETE]\n")
            or prefix.count("[FACTS COMPLETE]") != 1
            or not str(form["prompt"]).startswith(prefix)
            or not str(form["construction_prompt"]).startswith(prefix)
        ):
            raise CBNMSIntegrityError(
                "one CBNMS source form lacks the exact semantic anchor-prefix certificate"
            )
        key = (
            (
                str(form["scenario_id"]),
                int(form["assignment"]),
                str(form["target"]),
                str(form["event"]),
            )
            if form["family"] == "scenario"
            else (str(form["control_id"]),)
        )
        previous = prefix_by_twin.setdefault(key, prefix)
        if previous != prefix:
            raise CBNMSIntegrityError(
                "one CBNMS answer-order twin group has different source anchor prefixes"
            )
    if len(forms) != 80 or len({row["form_id"] for row in forms}) != 80:
        raise CBNMSIntegrityError("rendered CBNMS form coverage differs")
    if sum(row["family"] == "scenario" for row in forms) != 64:
        raise CBNMSIntegrityError("rendered CBNMS scenario coverage differs")
    if sum(row["family"] == "unrelated" for row in forms) != 16:
        raise CBNMSIntegrityError("rendered CBNMS control coverage differs")
    return tuple(forms)


def build_loso_folds(forms: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Build four scenario-LOSO folds with all collateral controls always held."""

    if len(forms) != 80 or len({row.get("form_id") for row in forms}) != 80:
        raise CBNMSIntegrityError("CBNMS fold construction requires 80 unique forms")
    scenarios = sorted(
        {str(row["scenario_id"]) for row in forms if row.get("family") == "scenario"}
    )
    if len(scenarios) != 4:
        raise CBNMSIntegrityError("CBNMS LOSO requires exactly four scenarios")
    folds: list[dict[str, Any]] = []
    for fold_index, held_scenario in enumerate(scenarios):
        train_targets: list[int] = []
        train_scenario_nuisance: list[int] = []
        train_fit_controls: list[int] = []
        held_targets: list[int] = []
        held_scenario_nuisance: list[int] = []
        held_collateral: list[int] = []
        for index, row in enumerate(forms):
            if row["family"] == "scenario":
                target = row["target"] == "self" and row["event"] == "permanent"
                held = row["scenario_id"] == held_scenario
                if held and target:
                    held_targets.append(index)
                elif held:
                    held_scenario_nuisance.append(index)
                elif target:
                    train_targets.append(index)
                else:
                    train_scenario_nuisance.append(index)
            elif row.get("control_partition") == "nuisance_fit":
                train_fit_controls.append(index)
            elif row.get("control_partition") == "held_collateral":
                held_collateral.append(index)
            else:
                raise CBNMSIntegrityError("one CBNMS form has invalid family semantics")
        train_nuisance = [*train_scenario_nuisance, *train_fit_controls]
        train_all = [*train_targets, *train_nuisance]
        held_nuisance = [*held_scenario_nuisance, *held_collateral]
        held_all = [*held_targets, *held_nuisance]
        counts = (
            len(train_targets),
            len(train_scenario_nuisance),
            len(train_fit_controls),
            len(train_nuisance),
            len(train_all),
            len(held_targets),
            len(held_scenario_nuisance),
            len(held_collateral),
            len(held_nuisance),
            len(held_all),
        )
        if counts != (12, 36, 8, 44, 56, 4, 12, 8, 20, 24):
            raise CBNMSIntegrityError("one CBNMS LOSO fold has incorrect row coverage")
        if set(train_all) & set(held_all) or set(train_all) | set(held_all) != set(range(80)):
            raise CBNMSIntegrityError("one CBNMS LOSO fold overlaps or omits rows")
        folds.append(
            _hashed_record(
                {
                    "fold_index": fold_index,
                    "held_scenario_id": held_scenario,
                    "training_target_indices": train_targets,
                    "training_scenario_nuisance_indices": train_scenario_nuisance,
                    "training_nuisance_fit_control_indices": train_fit_controls,
                    "training_nuisance_indices": train_nuisance,
                    "training_all_indices": train_all,
                    "held_target_indices": held_targets,
                    "held_scenario_nuisance_indices": held_scenario_nuisance,
                    "held_collateral_control_indices": held_collateral,
                    "held_nuisance_indices": held_nuisance,
                    "held_all_indices": held_all,
                    "held_collateral_never_enters_training": True,
                },
                "fold_sha256",
            )
        )
    return tuple(folds)


def target_pairs(
    forms: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> tuple[dict[str, Any], ...]:
    """Pair the two answer orders for each scenario and role assignment."""

    grouped: dict[tuple[str, int], list[int]] = {}
    for raw_index in indices:
        index = int(raw_index)
        row = forms[index]
        if not (
            row.get("family") == "scenario"
            and row.get("target") == "self"
            and row.get("event") == "permanent"
        ):
            raise CBNMSIntegrityError("a CBNMS target pair contains a non-target row")
        grouped.setdefault((str(row["scenario_id"]), int(row["assignment"])), []).append(
            index
        )
    result: list[dict[str, Any]] = []
    for (scenario, assignment), pair in sorted(grouped.items()):
        ordered = sorted(pair, key=lambda value: bool(forms[value]["preserve_first"]))
        if len(ordered) != 2 or {
            bool(forms[value]["preserve_first"]) for value in ordered
        } != {False, True}:
            raise CBNMSIntegrityError("a CBNMS pair lacks exactly both answer orders")
        result.append(
            _hashed_record(
                {
                    "scenario_id": scenario,
                    "assignment": assignment,
                    "indices": ordered,
                    "form_ids": [str(forms[value]["form_id"]) for value in ordered],
                },
                "pair_sha256",
            )
        )
    if len(result) * 2 != len(indices):
        raise CBNMSIntegrityError("CBNMS target-pair coverage differs")
    return tuple(result)


def resolve_four_slots_from_token_rows(
    token_rows: Sequence[Sequence[int]], *, special_token_ids: Sequence[int]
) -> tuple[int, int, int, int]:
    """Resolve the locked category-independent slot rule from three prompt twins."""

    if len(token_rows) != 3:
        raise ValueError("slot resolution requires construction plus two answer-order rows")
    rows = [tuple(int(value) for value in row) for row in token_rows]
    anchor = resolve_shared_anchor_index(rows)
    slots = (FIXED_FIRST_SLOT, anchor - 8, anchor - 4, anchor)
    if any(value < 0 for value in slots) or len(set(slots)) != SLOT_COUNT:
        raise CBNMSIntegrityError("locked CBNMS slots are not four distinct in-range rows")
    if tuple(sorted(slots)) != slots or slots[-1] >= min(len(row) for row in rows):
        raise CBNMSIntegrityError("locked CBNMS slots are not strictly ascending/in range")
    special = {int(value) for value in special_token_ids}
    if any(rows[0][slot] in special for slot in slots):
        raise CBNMSIntegrityError("a locked CBNMS slot is a special token")
    for slot in slots:
        if len({row[slot] for row in rows}) != 1:
            raise CBNMSIntegrityError("answer-order twins differ at a locked CBNMS slot")
    return slots


def build_tokenizer_preflight(
    backend: Any, forms: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Resolve all prospective token/slot evidence without a model forward."""

    if len(forms) != 80:
        raise CBNMSIntegrityError("tokenizer preflight requires exactly 80 forms")
    special = tuple(int(value) for value in backend.model.tokenizer.all_special_ids)
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(forms):
        key = (
            str(row["scenario_id"])
            + f":{row['assignment']}:{row['target']}:{row['event']}"
            if row["family"] == "scenario"
            else str(row["control_id"]),
            str(row["family"]),
        )
        grouped.setdefault(key, []).append(index)
    rows: list[dict[str, Any] | None] = [None] * 80
    for key, indices in sorted(grouped.items()):
        if len(indices) != 2:
            raise CBNMSIntegrityError("one tokenizer twin group does not contain two orders")
        construction = str(forms[indices[0]]["construction_prompt"])
        construction_ids = [int(value) for value in backend.encode(construction)[0].tolist()]
        ordered = sorted(
            indices,
            key=lambda value: bool(
                forms[value].get("preserve_first", forms[value].get("preferred_first"))
            ),
        )
        answer_ids = [
            [int(value) for value in backend.encode(str(forms[index]["prompt"]))[0].tolist()]
            for index in ordered
        ]
        slots = resolve_four_slots_from_token_rows(
            (construction_ids, answer_ids[0], answer_ids[1]),
            special_token_ids=special,
        )
        for index, prompt_ids in zip(ordered, answer_ids, strict=True):
            form = forms[index]
            boundary = resolve_choice_boundary(backend, str(form["prompt"]))
            if boundary.prompt_prefix_token_ids_sha256 != canonical_sha256(prompt_ids):
                raise CBNMSIntegrityError("choice-boundary and prompt token hashes differ")
            rows[index] = _hashed_record(
                {
                    "tensor_index": index,
                    "form_id": str(form["form_id"]),
                    "form_sha256": str(form["form_sha256"]),
                    "prompt_sha256": str(form["prompt_sha256"]),
                    "prompt_token_ids_sha256": canonical_sha256(prompt_ids),
                    "prompt_length": len(prompt_ids),
                    "slot_indices": list(slots),
                    "slot_token_ids": [prompt_ids[value] for value in slots],
                    "anchor_index": slots[-1],
                    "anchor_prefix_sha256": text_sha256(str(form["anchor_prefix"])),
                    "anchor_prefix_ends_facts_complete_marker": str(
                        form["anchor_prefix"]
                    ).endswith("[FACTS COMPLETE]\n"),
                    "source_prompt_starts_with_anchor_prefix": str(
                        form["prompt"]
                    ).startswith(str(form["anchor_prefix"])),
                    "source_construction_starts_with_anchor_prefix": str(
                        form["construction_prompt"]
                    ).startswith(str(form["anchor_prefix"])),
                    "choice_boundary_evidence_sha256": boundary.evidence_sha256,
                    "positive_token_id": boundary.token_id(str(form["positive_label"])),
                    "negative_token_id": boundary.token_id(str(form["negative_label"])),
                    "twin_group": list(key),
                },
                "row_sha256",
            )
    if any(value is None for value in rows):
        raise CBNMSIntegrityError("tokenizer preflight did not cover every form")
    materialized = [dict(value) for value in rows if value is not None]
    return _hashed_record(
        {
            "schema_version": f"{SCHEMA_VERSION}.tokenizer_preflight",
            "split": "prospective_validation",
            "model_forwards": 0,
            "model_backwards": 0,
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judges": 0,
            "slot_rule": [3, "anchor-8", "anchor-4", "anchor"],
            "layers": list(INCLUDED_LAYERS),
            "excluded_layer": 23,
            "semantic_anchor_rule": (
                "source prompt and construction share an identical prefix ending in "
                "exactly one [FACTS COMPLETE] marker; token anchor is the last common "
                "token position, not a claimed UTF-8 byte boundary"
            ),
            "rows": materialized,
        },
        "preflight_sha256",
    )


@dataclass(frozen=True, slots=True)
class CBNMSCapture:
    residuals: Any
    gradients: Any
    full_logits: Any
    positive_minus_negative_log_odds: float
    audit: Mapping[str, Any]


def capture_all_layers_four_slots(
    backend: Any,
    form: Mapping[str, Any],
    preflight_row: Mapping[str, Any],
) -> CBNMSCapture:
    """Capture [23,4,d] state-zero residuals/gradients in exactly one F+B."""

    prompt = str(form["prompt"])
    if text_sha256(prompt) != form.get("prompt_sha256"):
        raise CBNMSIntegrityError("capture prompt differs from prospective source")
    if (
        preflight_row.get("form_id") != form.get("form_id")
        or preflight_row.get("form_sha256") != form.get("form_sha256")
        or preflight_row.get("prompt_sha256") != form.get("prompt_sha256")
    ):
        raise CBNMSIntegrityError("capture row differs from tokenizer preflight")
    slots = tuple(int(value) for value in preflight_row["slot_indices"])
    if len(slots) != SLOT_COUNT or tuple(sorted(set(slots))) != slots:
        raise CBNMSIntegrityError("capture slots differ from locked four-slot rule")
    torch = backend.torch
    tokens = backend.encode(prompt)
    prompt_ids = [int(value) for value in tokens[0].tolist()]
    if canonical_sha256(prompt_ids) != preflight_row.get("prompt_token_ids_sha256"):
        raise CBNMSIntegrityError("capture tokenization differs from preflight")
    boundary = resolve_choice_boundary(backend, prompt)
    if boundary.evidence_sha256 != preflight_row.get("choice_boundary_evidence_sha256"):
        raise CBNMSIntegrityError("capture choice boundary differs from preflight")
    model_cfg = getattr(backend.model, "cfg", None)
    if getattr(model_cfg, "n_layers", None) != 24 or getattr(model_cfg, "d_model", None) != 1024:
        raise CBNMSIntegrityError("resident model geometry differs from CBNMS")
    positive_id = boundary.token_id(str(form["positive_label"]))
    negative_id = boundary.token_id(str(form["negative_label"]))
    if (
        positive_id != preflight_row.get("positive_token_id")
        or negative_id != preflight_row.get("negative_token_id")
    ):
        raise CBNMSIntegrityError("capture choice token IDs differ from preflight")

    activations: dict[int, Any] = {}
    residual_rows: dict[int, Any] = {}
    hook_calls = {layer: 0 for layer in INCLUDED_LAYERS}
    layer0_leaf: Any | None = None
    reconstruction_delta: float | None = None

    def hook_for(layer: int) -> Any:
        def capture(activation: Any, hook: Any) -> Any:
            nonlocal layer0_leaf, reconstruction_delta
            del hook
            hook_calls[layer] += 1
            if hook_calls[layer] != 1:
                raise CBNMSIntegrityError("one CBNMS hook fired more than once")
            if (
                getattr(activation, "ndim", None) != 3
                or tuple(activation.shape[:2]) != (1, int(tokens.shape[1]))
                or int(activation.shape[2]) != 1024
                or not bool(torch.isfinite(activation).all().item())
            ):
                raise CBNMSIntegrityError("one captured residual has invalid geometry")
            residual_rows[layer] = (
                activation[0, list(slots)].detach().cpu().float().contiguous().clone()
            )
            if layer == 0:
                detached = activation.detach()
                layer0_leaf = detached[0, list(slots)].clone().detach().requires_grad_(True)
                reconstructed = detached.clone()
                reconstructed[0, list(slots)] = layer0_leaf
                reconstruction_delta = float(
                    (reconstructed.detach().float() - detached.float()).abs().max().item()
                )
                if reconstruction_delta != 0.0:
                    raise CBNMSIntegrityError("layer-zero reconstruction was not exact")
                activations[layer] = layer0_leaf
                return reconstructed
            if not bool(activation.requires_grad):
                raise CBNMSIntegrityError("a later layer disconnected from layer zero")
            activations[layer] = activation
            return activation

        return capture

    parameters = tuple(backend.model.parameters())
    original_flags = tuple(bool(parameter.requires_grad) for parameter in parameters)
    gradients: tuple[Any, ...] | None = None
    logits: Any | None = None
    objective: Any | None = None
    parameter_gradients_allocated = False
    backend.model.zero_grad(set_to_none=True)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        if any(bool(parameter.requires_grad) for parameter in parameters):
            raise CBNMSIntegrityError("CBNMS could not disable parameter gradients")
        hooks = [
            (f"blocks.{layer}.hook_out", hook_for(layer)) for layer in INCLUDED_LAYERS
        ]
        with torch.enable_grad(), backend.model.hooks(fwd_hooks=hooks):
            output = backend.model(tokens)
            if tuple(sorted(activations)) != INCLUDED_LAYERS:
                raise CBNMSIntegrityError("CBNMS did not observe all locked layers once")
            logits = output[0, -1].float()
            if not bool(torch.isfinite(logits).all().item()):
                raise CBNMSIntegrityError("state-zero logits are non-finite")
            objective = logits[positive_id] - logits[negative_id]
            gradients = torch.autograd.grad(
                objective,
                tuple(activations[layer] for layer in INCLUDED_LAYERS),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            parameter_gradients_allocated = any(
                parameter.grad is not None for parameter in parameters
            )
            if parameter_gradients_allocated:
                raise CBNMSIntegrityError("CBNMS allocated model parameter gradients")
    finally:
        backend.model.zero_grad(set_to_none=True)
        for parameter, original in zip(parameters, original_flags, strict=True):
            parameter.requires_grad_(original)
    if gradients is None or logits is None or objective is None or layer0_leaf is None:
        raise CBNMSIntegrityError("CBNMS state-zero capture did not complete")
    residual_tensor = torch.stack(
        [residual_rows[layer] for layer in INCLUDED_LAYERS]
    ).contiguous()
    gradient_tensor = torch.stack(
        [
            (gradient if layer == 0 else gradient[0, list(slots)])
            .detach()
            .cpu()
            .float()
            .contiguous()
            .clone()
            for layer, gradient in zip(INCLUDED_LAYERS, gradients, strict=True)
        ]
    ).contiguous()
    if tuple(residual_tensor.shape) != (23, 4, 1024) or gradient_tensor.shape != residual_tensor.shape:
        raise CBNMSIntegrityError("CBNMS aggregate capture shape differs")
    margin = float(objective.detach().cpu().item())
    score = choice_score_from_logits(
        torch,
        logits.detach().cpu(),
        positive_id,
        negative_id,
        preserve_label=str(form["positive_label"]),
        comply_label=str(form["negative_label"]),
        choice_boundary_evidence_sha256=boundary.evidence_sha256,
        choice_a_token_id=boundary.a_token_id,
        choice_b_token_id=boundary.b_token_id,
    )
    if score.preserve_log_odds != margin:
        raise CBNMSIntegrityError("independent A/B score differs from gradient objective")
    full_logits = logits.detach().cpu().float().contiguous().clone()
    audit = _hashed_record(
        {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "capture_kind": "fresh_state_zero_layers_0_through_22_four_slots_one_F_plus_one_B",
            "form_id": str(form["form_id"]),
            "layers": list(INCLUDED_LAYERS),
            "excluded_layer": 23,
            "excluded_layer_reason": "hook_out_causal_position_excluded_a_priori",
            "slot_indices": list(slots),
            "hook_call_counts": {str(key): value for key, value in hook_calls.items()},
            "model_forward_evaluations": 1,
            "model_backward_evaluations": 1,
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judges": 0,
            "finite_interventions": 0,
            "positive_minus_negative_log_odds": margin,
            "full_logits_float32_sha256": tensor_float32_sha256(full_logits),
            "residuals_float32_sha256": tensor_float32_sha256(residual_tensor),
            "gradients_float32_sha256": tensor_float32_sha256(gradient_tensor),
            "maximum_abs_layer0_reconstruction_delta": reconstruction_delta,
            "model_parameters_requires_grad_disabled_during_capture": True,
            "model_parameter_requires_grad_flags_restored_after_capture": all(
                bool(parameter.requires_grad) == original
                for parameter, original in zip(parameters, original_flags, strict=True)
            ),
            "model_parameter_gradients_allocated": parameter_gradients_allocated,
            "later_layer_hooks_return_activation_unchanged": True,
            "prior_experiment_tensors_read": False,
            "sealed_data_accessed": False,
        },
        "audit_sha256",
    )
    return CBNMSCapture(
        residuals=residual_tensor,
        gradients=gradient_tensor,
        full_logits=full_logits,
        positive_minus_negative_log_odds=margin,
        audit=audit,
    )


def training_layer_slot_scales(
    residuals: Any, training_indices: Sequence[int]
) -> np.ndarray:
    """Fit [23,4] geometric-mean residual scales from training rows only."""

    values = np.asarray(residuals, dtype=np.float64)
    indices = tuple(int(value) for value in training_indices)
    if values.ndim != 4 or values.shape[:3] != (80, 23, 4):
        raise ValueError("CBNMS residuals must have shape [80,23,4,d_model]")
    if not indices or len(set(indices)) != len(indices) or any(
        value < 0 or value >= 80 for value in indices
    ):
        raise ValueError("training indices must be unique, non-empty, and in range")
    norms = np.linalg.norm(values[list(indices)], axis=3)
    if not np.isfinite(norms).all() or bool(np.any(norms <= 0.0)):
        raise CBNMSIntegrityError("training residual norms are nonpositive or non-finite")
    scales = np.exp(np.mean(np.log(norms), axis=0))
    if scales.shape != (23, 4) or not np.isfinite(scales).all():
        raise CBNMSIntegrityError("training layer-slot scales are invalid")
    return np.asarray(scales, dtype=np.float64, order="C")


def standardized_slot_major_rows(
    gradients: Any, scales: Any, indices: Sequence[int]
) -> np.ndarray:
    """Return [n,4*23*d] rows ordered slot, then layer, then residual unit."""

    values = np.asarray(gradients, dtype=np.float64)
    scale_array = np.asarray(scales, dtype=np.float64)
    selected = tuple(int(value) for value in indices)
    if values.ndim != 4 or values.shape[:3] != (80, 23, 4):
        raise ValueError("CBNMS gradients must have shape [80,23,4,d_model]")
    if scale_array.shape != (23, 4) or not np.isfinite(scale_array).all() or bool(
        np.any(scale_array <= 0.0)
    ):
        raise ValueError("CBNMS scales must be [23,4] positive finite values")
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected row indices must be unique and non-empty")
    scaled = values[list(selected)] * scale_array[None, :, :, None]
    if not np.isfinite(scaled).all():
        raise CBNMSIntegrityError("standardized gradient rows are non-finite")
    return np.asarray(
        scaled.transpose(0, 2, 1, 3).reshape(len(selected), -1),
        dtype=np.float64,
        order="C",
    )


def unflatten_slot_major_direction(direction: Any, *, d_model: int) -> np.ndarray:
    """Map [4*23*d] standardized coordinates to [23,4,d] blocks."""

    value = np.asarray(direction, dtype=np.float64)
    if value.shape != (SLOT_COUNT * len(INCLUDED_LAYERS) * int(d_model),):
        raise ValueError("one CBNMS direction has invalid width")
    return np.asarray(
        value.reshape(SLOT_COUNT, len(INCLUDED_LAYERS), d_model).transpose(1, 0, 2),
        dtype=np.float64,
        order="C",
    )


def _local_positions(scope: Sequence[int], selected: Sequence[int]) -> tuple[int, ...]:
    position = {int(value): index for index, value in enumerate(scope)}
    if len(position) != len(scope) or any(int(value) not in position for value in selected):
        raise CBNMSIntegrityError("audit scope does not contain each selected row exactly once")
    return tuple(position[int(value)] for value in selected)


def _relative_dose(
    edits: np.ndarray, residuals: np.ndarray, *, label: str
) -> dict[str, Any]:
    """Audit per-layer-slot, per-layer-prompt, and cumulative layer dose."""

    edit = np.asarray(edits, dtype=np.float64)
    base = np.asarray(residuals, dtype=np.float64)
    if edit.shape != base.shape or edit.ndim != 4 or edit.shape[1:3] != (23, 4):
        raise ValueError("dose audit requires matching [n,23,4,d] arrays")
    base_slot = np.linalg.norm(base, axis=3)
    base_layer = np.linalg.norm(base, axis=(2, 3))
    if bool(np.any(base_slot <= 0.0)) or bool(np.any(base_layer <= 0.0)):
        checks = {
            "every_prompt_layer_slot_requested_dose_at_most_cap": False,
            "every_prompt_layer_requested_dose_at_most_cap": False,
            "every_prompt_cumulative_root_sum_square_layer_dose_at_most_cap": False,
        }
        return _hashed_record(
            {
                "label": label,
                "scope_row_count": int(base.shape[0]),
                "status": "undefined_due_to_zero_state_zero_residual_norm",
                "zero_prompt_layer_slot_residual_norm_count": int(
                    np.count_nonzero(base_slot <= 0.0)
                ),
                "zero_prompt_layer_residual_norm_count": int(
                    np.count_nonzero(base_layer <= 0.0)
                ),
                "cap": REQUESTED_DOSE_CAP,
                "cap_tolerance": 0.0,
                "maximum_prompt_layer_slot_relative_l2": None,
                "maximum_prompt_layer_frobenius_relative_l2": None,
                "maximum_prompt_cumulative_rss_layer_dose": None,
                "checks": checks,
                "passes": False,
            }
        )
    slot_dose = np.linalg.norm(edit, axis=3) / base_slot
    layer_dose = np.linalg.norm(edit, axis=(2, 3)) / base_layer
    cumulative = np.sqrt(np.sum(layer_dose * layer_dose, axis=1))
    maximum_slot = float(np.max(slot_dose))
    maximum_layer = float(np.max(layer_dose))
    maximum_cumulative = float(np.max(cumulative))
    checks = {
        "every_prompt_layer_slot_requested_dose_at_most_cap": (
            maximum_slot <= REQUESTED_DOSE_CAP
        ),
        "every_prompt_layer_requested_dose_at_most_cap": (
            maximum_layer <= REQUESTED_DOSE_CAP
        ),
        "every_prompt_cumulative_root_sum_square_layer_dose_at_most_cap": (
            maximum_cumulative <= REQUESTED_DOSE_CAP
        ),
    }
    return _hashed_record(
        {
            "label": label,
            "scope_row_count": int(base.shape[0]),
            "cap": REQUESTED_DOSE_CAP,
            "cap_tolerance": 0.0,
            "cumulative_definition": "sqrt(sum_over_layers(relative_layer_frobenius_dose_squared))",
            "maximum_prompt_layer_slot_relative_l2": maximum_slot,
            "maximum_prompt_layer_frobenius_relative_l2": maximum_layer,
            "maximum_prompt_cumulative_rss_layer_dose": maximum_cumulative,
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
    )


def state_zero_linearized_audit(
    *,
    standardized_direction: Any,
    scales: Any,
    scope_residuals: Any,
    scope_rows: Any,
    scope_offsets: Any,
    target_positions: Sequence[int],
    exact_nuisance_positions: Sequence[int] = (),
    collateral_positions: Sequence[int] = (),
    gate_collateral: bool,
) -> dict[str, Any]:
    """Audit requested and fixed-state float32 arithmetic under both signs.

    This is not an intervention result.  Every layer's addition is calculated at
    its independently captured state-zero residual.  Earlier-layer state changes
    are not propagated to later layers.
    """

    direction = np.asarray(standardized_direction, dtype=np.float64)
    scale_array = np.asarray(scales, dtype=np.float64)
    residuals = np.asarray(scope_residuals, dtype=np.float32)
    rows = np.asarray(scope_rows, dtype=np.float64)
    offsets = np.asarray(scope_offsets, dtype=np.float64)
    if residuals.ndim != 4 or residuals.shape[1:3] != (23, 4):
        raise ValueError("scope residuals must have shape [n,23,4,d]")
    width = residuals.shape[3]
    if rows.shape != (residuals.shape[0], 4 * 23 * width):
        raise ValueError("scope standardized rows do not match residuals")
    if offsets.shape != (residuals.shape[0],) or not np.isfinite(offsets).all():
        raise ValueError("scope offsets are invalid")
    if (
        not np.isfinite(residuals).all()
        or not np.isfinite(rows).all()
        or not np.isfinite(direction).all()
    ):
        raise ValueError("state-zero audit inputs must be finite")
    blocks = unflatten_slot_major_direction(direction, d_model=width)
    if (
        scale_array.shape != (23, 4)
        or not np.isfinite(scale_array).all()
        or bool(np.any(scale_array <= 0.0))
    ):
        raise ValueError("state-zero audit scales have invalid shape")
    with np.errstate(over="ignore", invalid="ignore"):
        requested_positive = np.asarray(
            blocks * scale_array[:, :, None], dtype="<f4", order="C"
        )
    requested_negative = np.asarray(
        np.negative(requested_positive), dtype="<f4", order="C"
    )
    sign_bits_exact = bool(
        np.array_equal(
            requested_negative.view("<u4"),
            np.bitwise_xor(
                requested_positive.view("<u4"), np.uint32(0x80000000)
            ),
        )
    )
    target = tuple(int(value) for value in target_positions)
    exact = tuple(int(value) for value in exact_nuisance_positions)
    collateral = tuple(int(value) for value in collateral_positions)
    all_positions = {*target, *exact, *collateral}
    if (
        not target
        or len(all_positions) != len(target) + len(exact) + len(collateral)
        or any(value < 0 or value >= residuals.shape[0] for value in all_positions)
    ):
        raise ValueError("state-zero audit positions overlap or are invalid")
    if not np.isfinite(requested_positive).all() or not np.isfinite(
        requested_negative
    ).all():
        checks = {
            "requested_float32_delta_is_finite": False,
            "negative_requested_delta_is_exact_sign_bit_negation": sign_bits_exact,
            "both_state_zero_linearized_sign_audits": False,
        }
        return _hashed_record(
            {
                "method": "requested_and_state_zero_linearized_float32_multilayer_audit",
                "status": "scientific_no_go_nonfinite_requested_float32_delta",
                "not_a_finite_intervention": True,
                "scope_row_count": int(residuals.shape[0]),
                "target_row_count": len(target),
                "exact_training_nuisance_row_count": len(exact),
                "held_collateral_row_count": len(collateral),
                "nonfinite_requested_positive_count": int(
                    np.count_nonzero(~np.isfinite(requested_positive))
                ),
                "nonfinite_requested_negative_count": int(
                    np.count_nonzero(~np.isfinite(requested_negative))
                ),
                "requested_positive_identity": _array_identity(
                    requested_positive, dtype="float32"
                ),
                "requested_negative_identity": _array_identity(
                    requested_negative, dtype="float32"
                ),
                "signs": {},
                "checks": checks,
                "passes": False,
            }
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lower = np.abs(offsets[list(target)]) + TARGET_MARGIN
        intended_layer_norms = np.linalg.norm(blocks, axis=(1, 2))
        intended_total = float(np.linalg.norm(direction))
        requested_standardized = (
            requested_positive.astype(np.float64) / scale_array[:, :, None]
        )
        requested_layer_norms = np.linalg.norm(
            requested_standardized, axis=(1, 2)
        )
        requested_total = float(np.linalg.norm(requested_standardized))
    norm_summary = np.concatenate(
        (
            lower,
            intended_layer_norms,
            np.asarray([intended_total]),
            requested_standardized.reshape(-1),
            requested_layer_norms,
            np.asarray([requested_total]),
        )
    )
    if not np.isfinite(norm_summary).all():
        return _hashed_record(
            {
                "method": "requested_and_state_zero_linearized_float32_multilayer_audit",
                "status": "scientific_no_go_nonfinite_requested_norm_summary",
                "not_a_finite_intervention": True,
                "scope_row_count": int(residuals.shape[0]),
                "target_row_count": len(target),
                "exact_training_nuisance_row_count": len(exact),
                "held_collateral_row_count": len(collateral),
                "nonfinite_norm_summary_count": int(
                    np.count_nonzero(~np.isfinite(norm_summary))
                ),
                "requested_positive_identity": _array_identity(
                    requested_positive, dtype="float32"
                ),
                "requested_negative_identity": _array_identity(
                    requested_negative, dtype="float32"
                ),
                "signs": {},
                "checks": {
                    "requested_and_intended_norm_summary_is_finite": False,
                    "both_state_zero_linearized_sign_audits": False,
                },
                "passes": False,
            }
        )
    requested_broadcast = np.broadcast_to(
        requested_positive[None], residuals.shape
    ).copy()
    requested_dose = _relative_dose(
        requested_broadcast, residuals, label="requested_positive_float32"
    )
    sign_records: dict[str, Any] = {}
    maximum_state_zero_total = 0.0
    maximum_state_zero_layer = 0.0
    all_signs_pass = True
    numeric_failure = False

    def failed_sign(sign: int, stage: str, count: int) -> dict[str, Any]:
        return _hashed_record(
            {
                "requested_sign": sign,
                "status": f"scientific_no_go_nonfinite_{stage}",
                "nonfinite_value_count": int(count),
                "minimum_oriented_target_movement": None,
                "maximum_abs_exact_training_nuisance_movement": None,
                "maximum_abs_held_collateral_movement": None,
                "held_collateral_predicted_choice_flip_count": None,
                "passes": False,
            }
        )

    for sign, requested in ((1, requested_positive), (-1, requested_negative)):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            changed = np.asarray(
                residuals + requested[None], dtype=np.float32, order="C"
            )
            fixed_state_delta = np.asarray(
                changed - residuals, dtype=np.float32, order="C"
            )
            standardized = np.asarray(
                fixed_state_delta.astype(np.float64)
                / scale_array[None, :, :, None],
                dtype=np.float64,
                order="C",
            )
        for stage, value in (
            ("changed_float32_state", changed),
            ("state_zero_float32_delta", fixed_state_delta),
            ("standardized_state_zero_delta", standardized),
        ):
            if not np.isfinite(value).all():
                numeric_failure = True
                all_signs_pass = False
                sign_records[str(sign)] = failed_sign(
                    sign, stage, int(np.count_nonzero(~np.isfinite(value)))
                )
                break
        if str(sign) in sign_records:
            continue
        slot_major = standardized.transpose(0, 2, 1, 3).reshape(rows.shape)
        with np.errstate(over="ignore", invalid="ignore"):
            movements = np.einsum("ij,ij->i", rows, slot_major)
        if not np.isfinite(movements).all():
            numeric_failure = True
            all_signs_pass = False
            sign_records[str(sign)] = failed_sign(
                sign,
                "linearized_log_odds_movements",
                int(np.count_nonzero(~np.isfinite(movements))),
            )
            continue
        target_movement = movements[list(target)]
        oriented_target = sign * target_movement
        exact_values = movements[list(exact)] if exact else np.zeros(0)
        collateral_values = (
            movements[list(collateral)] if collateral else np.zeros(0)
        )
        collateral_offsets = (
            offsets[list(collateral)] if collateral else np.zeros(0)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            changed_target = offsets[list(target)] + target_movement
            changed_collateral = collateral_offsets + collateral_values
            total_norms = np.linalg.norm(
                standardized.reshape(standardized.shape[0], -1), axis=1
            )
            layer_norms = np.linalg.norm(standardized, axis=(2, 3))
            target_slacks = oriented_target - lower
        derived = np.concatenate(
            (
                changed_target,
                changed_collateral,
                total_norms,
                layer_norms.reshape(-1),
                target_slacks,
            )
        )
        if not np.isfinite(derived).all():
            numeric_failure = True
            all_signs_pass = False
            sign_records[str(sign)] = failed_sign(
                sign,
                "derived_target_collateral_or_norm_values",
                int(np.count_nonzero(~np.isfinite(derived))),
            )
            continue
        target_endpoint = (
            changed_target >= TARGET_MARGIN - STATE_ZERO_FLOAT32_TOLERANCE
            if sign == 1
            else changed_target <= -TARGET_MARGIN + STATE_ZERO_FLOAT32_TOLERANCE
        )
        baseline_ambiguous = (
            np.abs(collateral_offsets) <= CHOICE_AMBIGUITY_TOLERANCE
            if collateral
            else np.zeros(0, dtype=bool)
        )
        changed_ambiguous = (
            np.abs(changed_collateral) <= CHOICE_AMBIGUITY_TOLERANCE
            if collateral
            else np.zeros(0, dtype=bool)
        )
        ambiguous = baseline_ambiguous | changed_ambiguous
        baseline_choice = collateral_offsets > 0.0
        changed_choice = changed_collateral > 0.0
        flip_count = int(np.count_nonzero(baseline_choice != changed_choice))
        maximum_state_zero_total = max(
            maximum_state_zero_total, float(np.max(total_norms))
        )
        maximum_state_zero_layer = max(
            maximum_state_zero_layer, float(np.max(layer_norms))
        )
        fixed_state_dose = _relative_dose(
            fixed_state_delta,
            residuals,
            label=f"state_zero_linearized_float32_addition_sign_{sign}",
        )
        core_checks = {
            "both_answer_orders_reach_signed_target_endpoint": bool(
                np.all(target_endpoint)
            ),
            "target_linearized_certificate_with_float32_tolerance": float(
                np.min(target_slacks)
            )
            >= -STATE_ZERO_FLOAT32_TOLERANCE,
            "exact_training_nuisance_with_float32_tolerance": (
                float(np.max(np.abs(exact_values))) if exact_values.size else 0.0
            )
            <= STATE_ZERO_FLOAT32_TOLERANCE,
        }
        collateral_checks = {
            "held_collateral_baselines_nonambiguous": not bool(np.any(ambiguous)),
            "held_absolute_movement_at_most_cap": (
                float(np.max(np.abs(collateral_values)))
                if collateral_values.size
                else 0.0
            )
            <= HELD_ABSOLUTE_MOVEMENT_CAP,
            "held_predicted_choice_flip_count_zero": flip_count == 0,
        }
        passes = bool(
            all(core_checks.values())
            and (not gate_collateral or all(collateral_checks.values()))
        )
        all_signs_pass = all_signs_pass and passes
        sign_records[str(sign)] = _hashed_record(
            {
                "requested_sign": sign,
                "target_movements": target_movement.tolist(),
                "oriented_target_movements": oriented_target.tolist(),
                "changed_target_margins": changed_target.tolist(),
                "target_endpoint_checks": target_endpoint.tolist(),
                "minimum_oriented_target_movement": float(np.min(oriented_target)),
                "minimum_target_constraint_slack": float(
                    np.min(target_slacks)
                ),
                "exact_training_nuisance_movements": exact_values.tolist(),
                "maximum_abs_exact_training_nuisance_movement": (
                    float(np.max(np.abs(exact_values))) if exact_values.size else 0.0
                ),
                "held_collateral_movements": collateral_values.tolist(),
                "maximum_abs_held_collateral_movement": (
                    float(np.max(np.abs(collateral_values)))
                    if collateral_values.size
                    else 0.0
                ),
                "held_collateral_ambiguous_baseline_count": int(
                    np.count_nonzero(baseline_ambiguous)
                ),
                "held_collateral_ambiguous_changed_margin_count": int(
                    np.count_nonzero(changed_ambiguous)
                ),
                "held_collateral_predicted_choice_flip_count": flip_count,
                "maximum_state_zero_linearized_total_rss_norm": float(
                    np.max(total_norms)
                ),
                "maximum_state_zero_linearized_layer_norm": float(
                    np.max(layer_norms)
                ),
                "state_zero_linearized_delta_identity": _array_identity(
                    fixed_state_delta, dtype="float32"
                ),
                "state_zero_linearized_dose": fixed_state_dose,
                "state_zero_norm_and_dose_metrics_are_descriptive_not_qualification_gates": True,
                "core_checks": core_checks,
                "collateral_checks": collateral_checks,
                "collateral_is_qualification_gate": gate_collateral,
                "passes": passes,
            }
        )
    common_checks = {
        "intended_total_rss_norm_at_most_cap": intended_total <= TOTAL_RSS_CAP,
        "intended_each_layer_norm_at_most_cap": float(np.max(intended_layer_norms))
        <= PER_LAYER_CAP,
        "requested_float32_total_rss_norm_at_most_cap": requested_total
        <= TOTAL_RSS_CAP,
        "requested_float32_each_layer_norm_at_most_cap": float(
            np.max(requested_layer_norms)
        )
        <= PER_LAYER_CAP,
        "requested_dose_caps": requested_dose["passes"],
        "negative_requested_delta_is_exact_sign_bit_negation": sign_bits_exact,
        "both_state_zero_linearized_sign_audits": all_signs_pass,
    }
    return _hashed_record(
        {
            "method": "requested_and_state_zero_linearized_float32_multilayer_audit",
            "not_a_finite_intervention": True,
            "causal_limitation": (
                "earlier-layer edits would change later residual states; fixed-state "
                "addition arithmetic is not realized inference evidence"
            ),
            "scope_row_count": int(residuals.shape[0]),
            "target_row_count": len(target),
            "exact_training_nuisance_row_count": len(exact),
            "held_collateral_row_count": len(collateral),
            "intended_total_rss_standardized_norm": intended_total,
            "intended_per_layer_standardized_norms": intended_layer_norms.tolist(),
            "requested_float32_total_rss_standardized_norm": requested_total,
            "requested_float32_per_layer_standardized_norms": requested_layer_norms.tolist(),
            "maximum_state_zero_linearized_total_rss_norm": (
                None if numeric_failure else maximum_state_zero_total
            ),
            "maximum_state_zero_linearized_layer_norm": (
                None if numeric_failure else maximum_state_zero_layer
            ),
            "total_rss_cap": TOTAL_RSS_CAP,
            "per_layer_cap": PER_LAYER_CAP,
            "cap_tolerance": 0.0,
            "requested_positive_identity": _array_identity(
                requested_positive, dtype="float32"
            ),
            "requested_negative_identity": _array_identity(
                requested_negative, dtype="float32"
            ),
            "negative_construction": "unary_sign_bit_negation_of_same_positive_float32_delta",
            "same_requested_delta_shared_across_both_answer_orders": True,
            "requested_dose": requested_dose,
            "signs": sign_records,
            "checks": common_checks,
            "passes": bool(all(common_checks.values())),
        }
    )


def solve_two_inequality_in_bank(
    *, target_rows: Any, target_offsets: Any, bank_basis: Any
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Solve the two answer-order constraints only inside one frozen bank."""

    targets = np.asarray(target_rows, dtype=np.float64)
    offsets = np.asarray(target_offsets, dtype=np.float64)
    basis = np.asarray(bank_basis, dtype=np.float64)
    if (
        targets.ndim != 2
        or targets.shape[0] != 2
        or offsets.shape != (2,)
        or basis.ndim != 2
        or basis.shape[0] == 0
        or basis.shape[1] != targets.shape[1]
        or not np.isfinite(targets).all()
        or not np.isfinite(offsets).all()
        or not np.isfinite(basis).all()
    ):
        raise ValueError("bank-constrained paired solver inputs are invalid")
    orthogonality = float(
        np.max(np.abs(basis @ basis.T - np.eye(basis.shape[0])))
    )
    if orthogonality > 1e-10:
        raise CBNMSIntegrityError("frozen CBNMS bank is not orthonormal")
    coefficient_rows = targets @ basis.T
    lower = np.abs(offsets) + TARGET_MARGIN
    canonical = sorted(
        range(2),
        key=lambda index: hashlib.sha256(
            np.asarray(coefficient_rows[index], dtype="<f8").tobytes()
            + np.asarray(lower[index], dtype="<f8").tobytes()
        ).hexdigest(),
    )
    rows = coefficient_rows[canonical]
    bounds = lower[canonical]
    row_norms = np.linalg.norm(rows, axis=1)
    if bool(np.any(row_norms <= DOUBLE_CERTIFICATE_TOLERANCE)):
        return (
            _hashed_record(
                {
                    "method": "analytic_two_inequality_frozen_bank_active_set",
                    "status": "infeasible_zero_projected_target",
                    "bank_rank": int(basis.shape[0]),
                    "projected_target_norms": row_norms.tolist(),
                    "passes": False,
                }
            ),
            None,
        )
    gram = rows @ rows.T
    candidates: list[tuple[float, str, np.ndarray, np.ndarray, dict[str, Any]]] = []
    candidate_records: list[dict[str, Any]] = []

    def consider(active: tuple[int, ...], multiplier: np.ndarray) -> None:
        coefficient = rows.T @ multiplier
        values = rows @ coefficient
        slacks = values - bounds
        stationarity = coefficient - rows.T @ multiplier
        complementarity = multiplier * slacks
        checks = {
            "finite": bool(np.isfinite(coefficient).all() and np.isfinite(multiplier).all()),
            "primal_feasible": float(np.min(slacks)) >= -DOUBLE_CERTIFICATE_TOLERANCE,
            "dual_nonnegative": float(np.min(multiplier)) >= -DOUBLE_CERTIFICATE_TOLERANCE,
            "stationarity": float(np.max(np.abs(stationarity)))
            <= DOUBLE_CERTIFICATE_TOLERANCE,
            "complementarity": float(np.max(np.abs(complementarity)))
            <= DOUBLE_CERTIFICATE_TOLERANCE,
        }
        direction = np.asarray(basis.T @ coefficient, dtype=np.float64, order="C")
        identity = _array_identity(direction)
        record = _hashed_record(
            {
                "active_constraints": list(active),
                "multipliers": multiplier.tolist(),
                "coefficient": coefficient.tolist(),
                "coefficient_norm": float(np.linalg.norm(coefficient)),
                "direction_norm": float(np.linalg.norm(direction)),
                "direction_identity": identity,
                "constraint_values": values.tolist(),
                "constraint_slacks": slacks.tolist(),
                "checks": checks,
                "passes": bool(all(checks.values())),
            }
        )
        candidate_records.append(record)
        if record["passes"]:
            candidates.append(
                (
                    float(np.linalg.norm(direction)),
                    identity["raw_little_endian_bytes_sha256"],
                    direction,
                    coefficient,
                    record,
                )
            )

    for active in range(2):
        multiplier = np.zeros(2, dtype=np.float64)
        multiplier[active] = bounds[active] / gram[active, active]
        consider((active,), multiplier)
    if np.linalg.matrix_rank(gram, tol=DOUBLE_CERTIFICATE_TOLERANCE) == 2:
        try:
            consider((0, 1), np.linalg.solve(gram, bounds))
        except np.linalg.LinAlgError:
            pass
    if not candidates:
        return (
            _hashed_record(
                {
                    "method": "analytic_two_inequality_frozen_bank_active_set",
                    "status": "infeasible_or_numerically_indeterminate",
                    "bank_rank": int(basis.shape[0]),
                    "canonical_order": canonical,
                    "candidates": candidate_records,
                    "passes": False,
                }
            ),
            None,
        )
    _, _, direction, coefficient, selected = min(
        candidates, key=lambda value: (value[0], value[1])
    )
    values = targets @ direction
    independent = certify_minimum_l2_candidate(
        coefficient,
        coefficient_rows,
        np.zeros(2, dtype=np.float64),
        margin=lower,
        primal_tolerance=DOUBLE_CERTIFICATE_TOLERANCE,
    )
    checks = {
        "original_target_constraints": float(np.min(values - lower))
        >= -DOUBLE_CERTIFICATE_TOLERANCE,
        "independent_coefficient_space_minimum_norm_certificate": independent["passes"],
        "coefficient_and_direction_norm_identity": abs(
            float(np.linalg.norm(coefficient)) - float(np.linalg.norm(direction))
        )
        <= DOUBLE_CERTIFICATE_TOLERANCE,
    }
    record = _hashed_record(
        {
            "method": "analytic_two_inequality_frozen_bank_active_set",
            "status": "certified" if all(checks.values()) else "numerically_indeterminate",
            "bank_rank": int(basis.shape[0]),
            "bank_identity": _array_identity(basis),
            "canonical_order": canonical,
            "required_slopes": lower.tolist(),
            "target_values": values.tolist(),
            "target_slacks": (values - lower).tolist(),
            "minimum_norm": float(np.linalg.norm(direction)),
            "coefficient": coefficient.tolist(),
            "direction_identity": _array_identity(direction),
            "selected_candidate": selected,
            "independent_certificate": independent,
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
    )
    return record, direction if record["passes"] else None


def _canonical_twice_reorthogonalized_basis(
    sources: np.ndarray,
    *,
    expected_rank: int,
    nuisance_rows: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use SVD only for rank, then canonical-order twice-reorthogonalized MGS."""

    source = np.asarray(sources, dtype=np.float64, order="C")
    nuisance = np.asarray(nuisance_rows, dtype=np.float64, order="C")
    singular_values = np.linalg.svd(source, compute_uv=False)
    sigma_one = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(1e-12, 1e-10 * sigma_one)
    observed_rank = int(np.count_nonzero(singular_values > threshold))
    if observed_rank != int(expected_rank):
        raise CBNMSIntegrityError("bank source rank differs from its deterministic SVD rank")
    vectors: list[np.ndarray] = []
    for source_row in source:
        candidate = source_row.copy()
        for _ in range(2):
            for basis_row in vectors:
                candidate -= float(basis_row @ candidate) * basis_row
        norm = float(np.linalg.norm(candidate))
        if norm > threshold:
            candidate /= norm
            anchor = int(np.argmax(np.abs(candidate)))
            if candidate[anchor] < 0.0:
                candidate *= -1.0
            vectors.append(np.asarray(candidate, dtype=np.float64, order="C"))
        if len(vectors) == observed_rank:
            break
    if len(vectors) != observed_rank:
        raise CBNMSIntegrityError("twice-reorthogonalized bank did not recover SVD rank")
    basis = np.asarray(np.stack(vectors), dtype=np.float64, order="C")
    orthogonality = float(
        np.max(np.abs(basis @ basis.T - np.eye(observed_rank)))
    )
    reconstruction = float(
        np.max(np.abs(source - (source @ basis.T) @ basis))
    )
    nuisance_overlap = (
        float(np.max(np.abs(nuisance @ basis.T))) if nuisance.size else 0.0
    )
    checks = {
        "basis_orthogonality_at_most_1e_minus_10": orthogonality <= 1e-10,
        "source_reconstruction_at_most_1e_minus_10": reconstruction <= 1e-10,
        "training_nuisance_overlap_at_most_1e_minus_10": nuisance_overlap <= 1e-10,
    }
    record = _hashed_record(
        {
            "method": "SVD_rank_then_canonical_twice_reorthogonalized_modified_Gram_Schmidt",
            "svd_rank_rtol": 1e-10,
            "svd_rank_atol": 1e-12,
            "rank_threshold": threshold,
            "singular_values": singular_values.tolist(),
            "rank": observed_rank,
            "canonical_source_order_preserved": True,
            "canonical_sign_rule": "lowest-index maximum-absolute coordinate positive",
            "maximum_basis_orthogonality_error": orthogonality,
            "maximum_source_reconstruction_error": reconstruction,
            "maximum_abs_training_nuisance_overlap": nuisance_overlap,
            "basis_identity": _array_identity(basis),
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
    )
    if not record["passes"]:
        raise CBNMSIntegrityError("canonical CBNMS bank basis failed its certificate")
    return basis, record


def build_direction_bank(
    directions: Sequence[Any],
    *,
    maximum_rank: int,
    nuisance_rows: Any,
    label: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Orient/normalize solved directions and freeze their deterministic row span."""

    if len(directions) != maximum_rank or maximum_rank <= 0:
        raise ValueError("bank requires exactly its predeclared maximum source count")
    values = np.asarray(directions, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("bank source directions must be one finite matrix")
    norms = np.linalg.norm(values, axis=1)
    if bool(np.any(norms <= DOUBLE_CERTIFICATE_TOLERANCE)):
        record = _hashed_record(
            {
                "method": "deterministic_svd_span_of_oriented_normalized_training_SP_directions",
                "label": label,
                "status": "ineligible_zero_source_direction",
                "source_direction_norms": norms.tolist(),
                "passes": False,
            }
        )
        return None, record
    normalized = values / norms[:, None]
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    threshold = max(
        1e-12, 1e-10 * (float(singular_values[0]) if singular_values.size else 0.0)
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    basis, construction = _canonical_twice_reorthogonalized_basis(
        normalized,
        expected_rank=rank,
        nuisance_rows=np.asarray(nuisance_rows, dtype=np.float64),
    )
    checks = {
        "bank_rank_positive": rank > 0,
        "bank_rank_not_above_source_pair_count": rank <= maximum_rank,
        "canonical_basis_certificate": construction["passes"],
    }
    record = _hashed_record(
        {
            "method": "deterministic_svd_span_of_oriented_normalized_training_SP_directions",
            "label": label,
            "status": "certified" if all(checks.values()) else "ineligible",
            "source_pair_count": maximum_rank,
            "source_direction_norms": norms.tolist(),
            "source_directions_identity": _array_identity(values),
            "normalized_sources_identity": _array_identity(normalized),
            "bank_rank": rank,
            "bank_basis_identity": _array_identity(basis),
            "bank_construction_certificate": construction,
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
    )
    return (basis if record["passes"] else None), record


def deterministic_random_nullspace_bank(
    *,
    nuisance_basis: Any,
    nuisance_rows: Any,
    bank_rank: int,
    dimension: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create one locked PCG64 Gaussian bank in the frozen nuisance nullspace."""

    basis = np.asarray(nuisance_basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[1] != dimension or bank_rank <= 0:
        raise ValueError("random bank inputs are invalid")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    raw = rng.standard_normal((bank_rank, dimension), dtype=np.float64)
    projected = project_out_frozen_rowspace(raw, basis)
    singular_values = np.linalg.svd(projected, compute_uv=False)
    threshold = max(
        1e-12, 1e-10 * (float(singular_values[0]) if singular_values.size else 0.0)
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    random_basis, construction = _canonical_twice_reorthogonalized_basis(
        projected,
        expected_rank=rank,
        nuisance_rows=np.asarray(nuisance_rows, dtype=np.float64),
    )
    maximum_null = (
        float(np.max(np.abs(random_basis @ basis.T))) if basis.shape[0] else 0.0
    )
    checks = {
        "rank_matches_SP_bank": rank == bank_rank,
        "exact_frozen_training_nuisance_null": maximum_null
        <= DOUBLE_CERTIFICATE_TOLERANCE,
        "canonical_basis_certificate": construction["passes"],
    }
    record = _hashed_record(
        {
            "method": "PCG64_Gaussian_rank_matched_frozen_training_nullspace_bank",
            "seed": int(seed),
            "bank_rank": rank,
            "dimension": dimension,
            "raw_identity": _array_identity(raw),
            "projected_identity": _array_identity(projected),
            "bank_basis_identity": _array_identity(random_basis),
            "maximum_abs_frozen_nuisance_basis_overlap": maximum_null,
            "bank_construction_certificate": construction,
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
    )
    if not record["passes"]:
        raise CBNMSIntegrityError("one deterministic random-null bank failed construction")
    return random_basis, record


def random_bank_seed(
    *, dataset_sha256: str, fold_id: str, replicate: int
) -> int:
    """Derive the locked PCG64 seed from public, non-outcome identities."""

    if (
        not isinstance(dataset_sha256, str)
        or len(dataset_sha256) != 64
        or replicate < 0
    ):
        raise ValueError("random-bank seed inputs are invalid")
    payload = (
        f"CBNMS_RANDOM_BANK_V1|{dataset_sha256}|{fold_id}|{replicate}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _offset_array(margins: Any) -> np.ndarray:
    values = np.asarray(margins, dtype=np.float64)
    if values.shape != (80,) or not np.isfinite(values).all():
        raise CBNMSIntegrityError("CBNMS requires exactly 80 finite state-zero margins")
    return values


def _pair_oracle_record(
    *,
    method: str,
    solver: Mapping[str, Any],
    direction: np.ndarray | None,
    scales: np.ndarray,
    scope_indices: Sequence[int],
    scope_residuals: np.ndarray,
    scope_rows: np.ndarray,
    scope_offsets: np.ndarray,
    target_global_indices: Sequence[int],
    exact_global_indices: Sequence[int] = (),
    collateral_global_indices: Sequence[int] = (),
    gate_collateral: bool,
) -> dict[str, Any]:
    if direction is None or solver.get("passes") is not True:
        return _hashed_record(
            {
                "method": method,
                "solver": dict(solver),
                "state_zero_linearized_audit": None,
                "passes": False,
            }
        )
    target_positions = _local_positions(scope_indices, target_global_indices)
    exact_positions = _local_positions(scope_indices, exact_global_indices)
    collateral_positions = _local_positions(scope_indices, collateral_global_indices)
    audit = state_zero_linearized_audit(
        standardized_direction=direction,
        scales=scales,
        scope_residuals=scope_residuals,
        scope_rows=scope_rows,
        scope_offsets=scope_offsets,
        target_positions=target_positions,
        exact_nuisance_positions=exact_positions,
        collateral_positions=collateral_positions,
        gate_collateral=gate_collateral,
    )
    return _hashed_record(
        {
            "method": method,
            "solver": dict(solver),
            "state_zero_linearized_audit": audit,
            "passes": bool(solver.get("passes") and audit["passes"]),
        }
    )


def _validate_analysis_tensors(
    residuals: Any, gradients: Any, margins: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = np.asarray(residuals, dtype=np.float64)
    gradient = np.asarray(gradients, dtype=np.float64)
    offset = _offset_array(margins)
    if (
        residual.shape != gradient.shape
        or residual.ndim != 4
        or residual.shape[:3] != (80, 23, 4)
        or not np.isfinite(residual).all()
        or not np.isfinite(gradient).all()
    ):
        raise CBNMSIntegrityError("CBNMS analysis tensors are invalid")
    return residual, gradient, offset


def _training_local_arrays(
    residuals: Any,
    gradients: Any,
    margins: Any,
    *,
    expected_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate already-copied training-only numeric arrays, never an 80-row bundle."""

    residual = np.asarray(residuals, dtype=np.float64)
    gradient = np.asarray(gradients, dtype=np.float64)
    offset = np.asarray(margins, dtype=np.float64)
    if (
        residual.shape != gradient.shape
        or residual.ndim != 4
        or residual.shape[:3] != (expected_count, 23, 4)
        or offset.shape != (expected_count,)
        or not np.isfinite(residual).all()
        or not np.isfinite(gradient).all()
        or not np.isfinite(offset).all()
    ):
        raise CBNMSIntegrityError("training-only CBNMS numeric arrays are invalid")
    return residual, gradient, offset


def _scales_from_local_training_residuals(residuals: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(residuals, axis=3)
    if not np.isfinite(norms).all() or bool(np.any(norms <= 0.0)):
        raise CBNMSIntegrityError("training-only residual norms are invalid")
    scales = np.exp(np.mean(np.log(norms), axis=0))
    if scales.shape != (23, 4) or not np.isfinite(scales).all():
        raise CBNMSIntegrityError("training-only layer-slot scales are invalid")
    return np.asarray(scales, dtype=np.float64, order="C")


def _slot_major_local_rows(gradients: np.ndarray, scales: np.ndarray) -> np.ndarray:
    scaled = gradients * scales[None, :, :, None]
    return np.asarray(
        scaled.transpose(0, 2, 1, 3).reshape(gradients.shape[0], -1),
        dtype=np.float64,
        order="C",
    )


def _full_rank_nuisance_basis(
    nuisance_rows: np.ndarray, *, expected_rank: int
) -> tuple[np.ndarray, dict[str, Any]]:
    norms = np.linalg.norm(nuisance_rows, axis=1)
    basis, record = frozen_nuisance_rowspace(nuisance_rows)
    checks = {
        "no_zero_nuisance_rows": bool(np.all(norms > 0.0)),
        "nuisance_rank_equals_locked_expected_rank": int(record["rank"])
        == int(expected_rank),
        "svd_certificate": record["passes"],
    }
    wrapper = _hashed_record(
        {
            "expected_rank": int(expected_rank),
            "observed_rank": int(record["rank"]),
            "rank_rule": "tau=max(1e-12,1e-10*sigma_1)",
            "svd_input": (
                "each nonzero nuisance row normalized to unit L2 before SVD; "
                "the inherited certificate also reconstructs the original raw rows"
            ),
            "row_norms": norms.tolist(),
            "svd_record": record,
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
    )
    return basis, wrapper


def analyze_training_fold(
    *,
    forms: Sequence[Mapping[str, Any]],
    fold: Mapping[str, Any],
    residuals: Any,
    gradients: Any,
    margins: Any,
    dataset_sha256: str,
    random_replicates: int = 32,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """Construct and freeze one training-only SP bank before held arithmetic."""

    train_all = tuple(int(value) for value in fold["training_all_indices"])
    train_targets = tuple(int(value) for value in fold["training_target_indices"])
    train_nuisance = tuple(int(value) for value in fold["training_nuisance_indices"])
    if (len(train_all), len(train_targets), len(train_nuisance)) != (56, 12, 44):
        raise CBNMSIntegrityError("training fold row counts differ")
    residual, gradient, local_offsets = _training_local_arrays(
        residuals, gradients, margins, expected_count=56
    )
    local_residual_norms = np.linalg.norm(residual, axis=3)
    if bool(np.any(local_residual_norms <= 0.0)):
        record = _hashed_record(
            {
                "schema_version": TRAINING_SCHEMA_VERSION,
                "split": "prospective_validation",
                "fold": dict(fold),
                "coordinate": "slot_major_4_by_23_by_d_model",
                "layers": list(INCLUDED_LAYERS),
                "layer_selection_or_pruning": False,
                "status": "scientific_no_go_zero_training_residual_norm",
                "training_only_residuals_identity": _array_identity(residual),
                "zero_training_prompt_layer_slot_residual_norm_count": int(
                    np.count_nonzero(local_residual_norms <= 0.0)
                ),
                "random_nullspace_bank_freeze_manifest": [],
                "random_bank_construction_eligibility": False,
                "random_bank_not_constructed_reason": (
                    "training residual scale is undefined because a locked norm is zero"
                ),
                "held_numeric_rows_used_in_training_arithmetic": False,
                "held_target_gradients_used_in_training": False,
                "held_nuisance_numeric_rows_used_in_training": False,
                "checks": {
                    "all_training_prompt_layer_slot_residual_norms_positive": False
                },
                "passes": False,
            }
        )
        return record, None
    scales = _scales_from_local_training_residuals(residual)
    scope_rows = _slot_major_local_rows(gradient, scales)
    row_position = {value: index for index, value in enumerate(train_all)}
    nuisance_rows = scope_rows[[row_position[value] for value in train_nuisance]]
    nuisance_basis, nuisance_record = _full_rank_nuisance_basis(
        nuisance_rows, expected_rank=44
    )
    scope_residual = residual
    scope_offsets = local_offsets
    pairs = target_pairs(forms, train_targets)
    if len(pairs) != 6:
        raise CBNMSIntegrityError("one training fold must contain six SP pairs")
    pair_records: list[dict[str, Any]] = []
    bank_sources: list[np.ndarray] = []
    target_only_bank_sources: list[np.ndarray] = []
    for pair in pairs:
        indices = tuple(int(value) for value in pair["indices"])
        pair_rows = scope_rows[[row_position[value] for value in indices]]
        pair_offsets = local_offsets[[row_position[value] for value in indices]]
        target_only_solver, target_only_direction = solve_paired_oracle(
            target_rows=pair_rows,
            target_offsets=pair_offsets,
        )
        target_only = _pair_oracle_record(
            method="training_pair_target_only_ambient_oracle",
            solver=target_only_solver,
            direction=target_only_direction,
            scales=scales,
            scope_indices=train_all,
            scope_residuals=scope_residual,
            scope_rows=scope_rows,
            scope_offsets=scope_offsets,
            target_global_indices=indices,
            gate_collateral=False,
        )
        if target_only_direction is not None and target_only["passes"]:
            target_only_bank_sources.append(
                np.asarray(target_only_direction, dtype=np.float64)
            )
        null_solver, null_direction = solve_paired_oracle(
            target_rows=pair_rows,
            target_offsets=pair_offsets,
            nuisance_rows=nuisance_rows,
            frozen_basis=nuisance_basis,
        )
        null_oracle = _pair_oracle_record(
            method="training_pair_exact_training_nuisance_null_ambient_oracle",
            solver=null_solver,
            direction=null_direction,
            scales=scales,
            scope_indices=train_all,
            scope_residuals=scope_residual,
            scope_rows=scope_rows,
            scope_offsets=scope_offsets,
            target_global_indices=indices,
            exact_global_indices=train_nuisance,
            gate_collateral=False,
        )
        if null_direction is not None and null_oracle["passes"]:
            bank_sources.append(np.asarray(null_direction, dtype=np.float64))
        pair_records.append(
            _hashed_record(
                {
                    "pair": dict(pair),
                    "target_only": target_only,
                    "training_nuisance_null": null_oracle,
                    "passes": bool(target_only["passes"] and null_oracle["passes"]),
                }
            )
        )
    bank: np.ndarray | None = None
    if len(bank_sources) == 6 and all(record["passes"] for record in pair_records):
        bank, bank_record = build_direction_bank(
            bank_sources,
            maximum_rank=6,
            nuisance_rows=nuisance_rows,
            label=f"fold_{fold['fold_index']}_training_SP_bank",
        )
    else:
        bank_record = _hashed_record(
            {
                "method": "training_SP_bank",
                "status": "not_constructed_due_to_failed_training_pair",
                "certified_source_count": len(bank_sources),
                "passes": False,
            }
        )
    target_only_bank: np.ndarray | None = None
    if len(target_only_bank_sources) == 6 and all(
        record["target_only"]["passes"] for record in pair_records
    ):
        target_only_bank, target_only_bank_record = build_direction_bank(
            target_only_bank_sources,
            maximum_rank=6,
            nuisance_rows=np.zeros((0, scope_rows.shape[1]), dtype=np.float64),
            label=f"fold_{fold['fold_index']}_training_target_only_bank",
        )
    else:
        target_only_bank_record = _hashed_record(
            {
                "method": "training_target_only_bank",
                "status": "not_constructed_due_to_failed_training_pair",
                "certified_source_count": len(target_only_bank_sources),
                "passes": False,
            }
        )
    random_manifest: list[dict[str, Any]] = []
    pre_random_bank_gate = bool(
        nuisance_record["passes"]
        and len(pair_records) == 6
        and all(record["passes"] for record in pair_records)
        and bank is not None
        and bank_record["passes"]
        and target_only_bank is not None
        and target_only_bank_record["passes"]
        and target_only_bank_record.get("bank_rank") == bank_record.get("bank_rank")
    )
    if pre_random_bank_gate:
        for replicate in range(int(random_replicates)):
            seed = random_bank_seed(
                dataset_sha256=dataset_sha256,
                fold_id=str(fold["fold_sha256"]),
                replicate=replicate,
            )
            _, random_record = deterministic_random_nullspace_bank(
                nuisance_basis=nuisance_basis,
                nuisance_rows=nuisance_rows,
                bank_rank=int(bank_record["bank_rank"]),
                dimension=scope_rows.shape[1],
                seed=seed,
            )
            random_manifest.append(
                _hashed_record(
                    {
                        "replicate": replicate,
                        "seed": seed,
                        "bank_rank": int(bank_record["bank_rank"]),
                        "bank_basis_identity": random_record["bank_basis_identity"],
                        "construction_record_sha256": random_record["record_sha256"],
                    }
                )
            )
    checks = {
        "full_expected_training_nuisance_rank": nuisance_record["passes"],
        "all_six_target_only_oracles": len(pair_records) == 6
        and all(value["target_only"]["passes"] for value in pair_records),
        "all_six_training_nuisance_null_oracles": len(pair_records) == 6
        and all(value["training_nuisance_null"]["passes"] for value in pair_records),
        "training_SP_bank_certified": bank_record["passes"],
        "matched_six_source_target_only_bank_certified": target_only_bank_record[
            "passes"
        ],
        "target_only_bank_rank_matches_training_SP_bank_rank_without_truncation": (
            bank_record.get("passes") is True
            and target_only_bank_record.get("passes") is True
            and target_only_bank_record.get("bank_rank") == bank_record.get("bank_rank")
        ),
        "random_bank_manifest_obeys_fail_closed_construction_gate": (
            len(random_manifest) == int(random_replicates)
            if pre_random_bank_gate
            else len(random_manifest) == 0
        ),
    }
    record = _hashed_record(
        {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "split": "prospective_validation",
            "fold": dict(fold),
            "coordinate": "slot_major_4_by_23_by_d_model",
            "layers": list(INCLUDED_LAYERS),
            "layer_selection_or_pruning": False,
            "training_scales": scales.tolist(),
            "training_scales_identity": _array_identity(scales),
            "training_rows_identity": _array_identity(scope_rows),
            "training_nuisance_rows_identity": _array_identity(nuisance_rows),
            "training_nuisance_basis": nuisance_record,
            "pair_oracles": pair_records,
            "training_SP_bank": bank_record,
            "training_target_only_bank": target_only_bank_record,
            "random_nullspace_bank_freeze_manifest": random_manifest,
            "random_bank_construction_eligibility": pre_random_bank_gate,
            "random_bank_not_constructed_reason": (
                None
                if pre_random_bank_gate
                else "one or more irrevocable pre-random training geometry gates failed"
            ),
            "random_nullspace_bank_construction": (
                "regenerate_one_at_a_time_from_frozen_seed_and_require_exact_float64_hash"
            ),
            "held_numeric_rows_used_in_training_arithmetic": False,
            "held_target_gradients_used_in_training": False,
            "held_nuisance_numeric_rows_used_in_training": False,
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
    )
    numeric = (
        {
            "scales": np.asarray(scales, dtype=np.float64, order="C"),
            "nuisance_basis": np.asarray(nuisance_basis, dtype=np.float64, order="C"),
            "SP_bank": np.asarray(bank, dtype=np.float64, order="C"),
            "target_only_bank": np.asarray(
                target_only_bank, dtype=np.float64, order="C"
            ),
        }
        if record["passes"] and bank is not None and target_only_bank is not None
        else None
    )
    return record, numeric


def _maximum_collateral(record: Mapping[str, Any]) -> float:
    audit = record.get("state_zero_linearized_audit")
    if not isinstance(audit, Mapping):
        return math.inf
    raw_values = [
        sign.get("maximum_abs_held_collateral_movement")
        for sign in audit["signs"].values()
    ]
    if any(value is None for value in raw_values):
        return math.inf
    values = [float(value) for value in raw_values]
    return max(values) if values else 0.0


def _minimum_oriented_target(record: Mapping[str, Any]) -> float:
    audit = record.get("state_zero_linearized_audit")
    if not isinstance(audit, Mapping):
        return -math.inf
    raw_values = [
        sign.get("minimum_oriented_target_movement")
        for sign in audit["signs"].values()
    ]
    if any(value is None for value in raw_values):
        return -math.inf
    values = [float(value) for value in raw_values]
    return min(values) if values else -math.inf


def _matched_other_permanent_pairs(
    forms: Sequence[Mapping[str, Any]], held_nuisance: Sequence[int]
) -> tuple[dict[str, Any], ...]:
    grouped: dict[int, list[int]] = {}
    for index in held_nuisance:
        row = forms[int(index)]
        if (
            row.get("family") == "scenario"
            and row.get("target") == "other"
            and row.get("event") == "permanent"
        ):
            grouped.setdefault(int(row["assignment"]), []).append(int(index))
    result = []
    for assignment, indices in sorted(grouped.items()):
        ordered = sorted(indices, key=lambda value: bool(forms[value]["preserve_first"]))
        if len(ordered) != 2:
            raise CBNMSIntegrityError("matched non-SP pair coverage differs")
        result.append(
            _hashed_record(
                {
                    "assignment": assignment,
                    "scenario_id": str(forms[ordered[0]]["scenario_id"]),
                    "indices": ordered,
                    "form_ids": [str(forms[index]["form_id"]) for index in ordered],
                },
                "pair_sha256",
            )
        )
    if len(result) != 2:
        raise CBNMSIntegrityError("held fold needs two matched non-SP pairs")
    return tuple(result)


def _evaluate_one_bank_on_held(
    *,
    label: str,
    bank: np.ndarray,
    pairs: Sequence[Mapping[str, Any]],
    train_nuisance: Sequence[int],
    held_nuisance: Sequence[int],
    all_indices: Sequence[int],
    rows: np.ndarray,
    residuals: np.ndarray,
    offsets: np.ndarray,
    scales: np.ndarray,
    enforce_training_null: bool,
    gate_collateral: bool,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    position = {int(value): index for index, value in enumerate(all_indices)}
    nuisance_rows = rows[[position[int(value)] for value in train_nuisance]]
    pair_records: list[dict[str, Any]] = []
    directions: list[np.ndarray] = []
    for pair in pairs:
        indices = tuple(int(value) for value in pair["indices"])
        scope_indices = (*indices, *train_nuisance, *held_nuisance)
        if len(scope_indices) != 66 or len(set(scope_indices)) != 66:
            raise CBNMSIntegrityError("held direction audit scope must be exactly 66 rows")
        scope_positions = [position[int(value)] for value in scope_indices]
        scope_rows = rows[scope_positions]
        scope_residuals = residuals[scope_positions]
        scope_offsets = offsets[scope_positions]
        target_rows = rows[[position[value] for value in indices]]
        solver, direction = solve_two_inequality_in_bank(
            target_rows=target_rows,
            target_offsets=offsets[list(indices)],
            bank_basis=bank,
        )
        null_error = (
            float(np.max(np.abs(nuisance_rows @ direction)))
            if direction is not None
            else None
        )
        if (
            enforce_training_null
            and direction is not None
            and null_error is not None
            and null_error > 1e-10
        ):
            raise CBNMSIntegrityError("bank-constrained direction left training nuisance null")
        oracle = _pair_oracle_record(
            method=label,
            solver=solver,
            direction=direction,
            scales=scales,
            scope_indices=scope_indices,
            scope_residuals=scope_residuals,
            scope_rows=scope_rows,
            scope_offsets=scope_offsets,
            target_global_indices=indices,
            exact_global_indices=train_nuisance if enforce_training_null else (),
            collateral_global_indices=held_nuisance,
            gate_collateral=gate_collateral,
        )
        pair_records.append(
            _hashed_record(
                {
                    "pair": dict(pair),
                    "maximum_abs_float64_training_nuisance_slope": null_error,
                    "training_null_is_primary_constraint": enforce_training_null,
                    "oracle": oracle,
                    "passes": oracle["passes"],
                }
            )
        )
        if direction is not None:
            directions.append(direction)
    maximum_nuisance = max(
        (_maximum_collateral(value["oracle"]) for value in pair_records),
        default=math.inf,
    )
    minimum_target = min(
        (_minimum_oriented_target(value["oracle"]) for value in pair_records),
        default=-math.inf,
    )
    ratio = (
        maximum_nuisance / minimum_target
        if math.isfinite(maximum_nuisance) and minimum_target > 0.0
        else math.inf
    )
    checks = {
        "both_held_assignment_pairs_pass": len(pair_records) == 2
        and all(value["passes"] for value in pair_records),
        "fold_global_held_absolute_movement_at_most_0_05": maximum_nuisance
        <= HELD_ABSOLUTE_MOVEMENT_CAP,
        "fold_global_leakage_ratio_at_most_0_50": ratio
        <= HELD_LEAKAGE_RATIO_CAP,
    }
    return (
        _hashed_record(
            {
                "method": label,
                "pair_oracles": pair_records,
                "full_cartesian_cell_count": 2 * len(held_nuisance) * 2,
                "maximum_abs_held_nuisance_movement": (
                    maximum_nuisance if math.isfinite(maximum_nuisance) else None
                ),
                "minimum_oriented_held_target_movement": (
                    minimum_target if math.isfinite(minimum_target) else None
                ),
                "fold_global_leakage_ratio": ratio if math.isfinite(ratio) else None,
                "checks": checks,
                "collateral_is_qualification_gate": gate_collateral,
                "passes": bool(
                    checks["both_held_assignment_pairs_pass"]
                    and (
                        not gate_collateral
                        or (
                            checks["fold_global_held_absolute_movement_at_most_0_05"]
                            and checks["fold_global_leakage_ratio_at_most_0_50"]
                        )
                    )
                ),
            }
        ),
        directions,
    )


def evaluate_held_fold(
    *,
    forms: Sequence[Mapping[str, Any]],
    fold: Mapping[str, Any],
    training_record: Mapping[str, Any],
    frozen_numeric: Mapping[str, Any],
    residuals: Any,
    gradients: Any,
    margins: Any,
    dataset_sha256: str,
) -> dict[str, Any]:
    """Evaluate one target-aware held oracle only inside the frozen SP bank."""

    if training_record.get("passes") is not True:
        raise CBNMSIntegrityError("held analysis requires a passing immutable training fold")
    residual, gradient, offsets = _validate_analysis_tensors(
        residuals, gradients, margins
    )
    scales = np.asarray(frozen_numeric["scales"], dtype=np.float64)
    nuisance_basis = np.asarray(frozen_numeric["nuisance_basis"], dtype=np.float64)
    bank = np.asarray(frozen_numeric["SP_bank"], dtype=np.float64)
    target_only_bank = np.asarray(
        frozen_numeric["target_only_bank"], dtype=np.float64
    )
    if (
        _array_identity(scales) != training_record["training_scales_identity"]
        or _array_identity(bank) != training_record["training_SP_bank"]["bank_basis_identity"]
        or _array_identity(target_only_bank)
        != training_record["training_target_only_bank"]["bank_basis_identity"]
    ):
        raise CBNMSIntegrityError("held analysis frozen scale/bank identity differs")
    all_indices = tuple(range(80))
    rows = standardized_slot_major_rows(gradient, scales, all_indices)
    train_nuisance = tuple(int(value) for value in fold["training_nuisance_indices"])
    held_targets = tuple(int(value) for value in fold["held_target_indices"])
    held_nuisance = tuple(int(value) for value in fold["held_nuisance_indices"])
    nuisance_rows = rows[list(train_nuisance)]
    if _array_identity(nuisance_basis) != training_record["training_nuisance_basis"][
        "svd_record"
    ]["basis_identity"]:
        raise CBNMSIntegrityError("held nuisance basis identity differs from training freeze")
    maximum_bank_null = float(np.max(np.abs(nuisance_rows @ bank.T)))
    if maximum_bank_null > 1e-10:
        raise CBNMSIntegrityError("persisted SP bank is not in the training nuisance null")
    held_pairs = target_pairs(forms, held_targets)
    if len(held_pairs) != 2:
        raise CBNMSIntegrityError("held fold requires exactly two target pairs")

    primary, _ = _evaluate_one_bank_on_held(
        label="primary_frozen_training_SP_bank_target_aware_transductive_oracle",
        bank=bank,
        pairs=held_pairs,
        train_nuisance=train_nuisance,
        held_nuisance=held_nuisance,
        all_indices=all_indices,
        rows=rows,
        residuals=residual,
        offsets=offsets,
        scales=scales,
        enforce_training_null=True,
        gate_collateral=True,
    )

    target_only_bank_evaluation, _ = _evaluate_one_bank_on_held(
        label="matched_six_source_training_target_only_bank_transductive_oracle",
        bank=target_only_bank,
        pairs=held_pairs,
        train_nuisance=train_nuisance,
        held_nuisance=held_nuisance,
        all_indices=all_indices,
        rows=rows,
        residuals=residual,
        offsets=offsets,
        scales=scales,
        enforce_training_null=False,
        gate_collateral=False,
    )

    target_only_records: list[dict[str, Any]] = []
    ambient_null_records: list[dict[str, Any]] = []
    for pair in held_pairs:
        indices = tuple(int(value) for value in pair["indices"])
        comparator_scope = (*indices, *train_nuisance, *held_nuisance)
        if len(comparator_scope) != 66 or len(set(comparator_scope)) != 66:
            raise CBNMSIntegrityError("held comparator scope must contain exactly 66 rows")
        pair_rows = rows[list(indices)]
        pair_offsets = offsets[list(indices)]
        target_solver, target_direction = solve_paired_oracle(
            target_rows=pair_rows, target_offsets=pair_offsets
        )
        target_oracle = _pair_oracle_record(
            method="descriptive_held_target_only_ambient_oracle",
            solver=target_solver,
            direction=target_direction,
            scales=scales,
            scope_indices=comparator_scope,
            scope_residuals=residual[list(comparator_scope)],
            scope_rows=rows[list(comparator_scope)],
            scope_offsets=offsets[list(comparator_scope)],
            target_global_indices=indices,
            collateral_global_indices=held_nuisance,
            gate_collateral=False,
        )
        target_only_records.append(
            _hashed_record({"pair": dict(pair), "oracle": target_oracle})
        )
        ambient_solver, ambient_direction = solve_paired_oracle(
            target_rows=pair_rows,
            target_offsets=pair_offsets,
            nuisance_rows=nuisance_rows,
            frozen_basis=nuisance_basis,
        )
        ambient_oracle = _pair_oracle_record(
            method="descriptive_unrestricted_ambient_training_null_oracle",
            solver=ambient_solver,
            direction=ambient_direction,
            scales=scales,
            scope_indices=comparator_scope,
            scope_residuals=residual[list(comparator_scope)],
            scope_rows=rows[list(comparator_scope)],
            scope_offsets=offsets[list(comparator_scope)],
            target_global_indices=indices,
            exact_global_indices=train_nuisance,
            collateral_global_indices=held_nuisance,
            gate_collateral=True,
        )
        ambient_null_records.append(
            _hashed_record({"pair": dict(pair), "oracle": ambient_oracle})
        )
    target_only_maximum = (
        float(target_only_bank_evaluation["maximum_abs_held_nuisance_movement"])
        if target_only_bank_evaluation["maximum_abs_held_nuisance_movement"]
        is not None
        else math.inf
    )
    primary_maximum = (
        float(primary["maximum_abs_held_nuisance_movement"])
        if primary["maximum_abs_held_nuisance_movement"] is not None
        else math.inf
    )
    primary_improvement_gate = bool(
        primary["passes"]
        and target_only_bank_evaluation["passes"]
        and math.isfinite(primary_maximum)
        and math.isfinite(target_only_maximum)
        and target_only_maximum - primary_maximum
        >= MINIMUM_ABSOLUTE_LEAKAGE_REDUCTION
        and primary_maximum
        <= MAXIMUM_RELATIVE_LEAKAGE_VS_TARGET_ONLY_BANK * target_only_maximum
    )
    random_results: list[dict[str, Any]] = []
    manifest = training_record["random_nullspace_bank_freeze_manifest"]
    for frozen in manifest:
        replicate = int(frozen["replicate"])
        if not primary_improvement_gate:
            random_results.append(
                _hashed_record(
                    {
                        "replicate": replicate,
                        "seed": int(frozen["seed"]),
                        "frozen_bank_identity": frozen["bank_basis_identity"],
                        "status": "not_evaluated_because_primary_fold_gate_failed",
                        "evaluation": None,
                        "strict_leakage_improvement_over_matched_target_only_bank": False,
                        "passes_complete_fold_gate": False,
                    }
                )
            )
            continue
        seed = random_bank_seed(
            dataset_sha256=dataset_sha256,
            fold_id=str(fold["fold_sha256"]),
            replicate=replicate,
        )
        if seed != int(frozen["seed"]):
            raise CBNMSIntegrityError("random bank regenerated seed differs")
        random_bank, construction = deterministic_random_nullspace_bank(
            nuisance_basis=nuisance_basis,
            nuisance_rows=nuisance_rows,
            bank_rank=int(training_record["training_SP_bank"]["bank_rank"]),
            dimension=rows.shape[1],
            seed=seed,
        )
        if (
            _array_identity(random_bank) != frozen["bank_basis_identity"]
            or construction["record_sha256"] != frozen["construction_record_sha256"]
        ):
            raise CBNMSIntegrityError("regenerated random bank bytes differ from freeze")
        evaluated, _ = _evaluate_one_bank_on_held(
            label=f"random_rank_matched_nullspace_bank_replicate_{replicate}",
            bank=random_bank,
            pairs=held_pairs,
            train_nuisance=train_nuisance,
            held_nuisance=held_nuisance,
            all_indices=all_indices,
            rows=rows,
            residuals=residual,
            offsets=offsets,
            scales=scales,
            enforce_training_null=True,
            gate_collateral=True,
        )
        random_maximum = (
            float(evaluated["maximum_abs_held_nuisance_movement"])
            if evaluated["maximum_abs_held_nuisance_movement"] is not None
            else math.inf
        )
        random_improvement = bool(
            math.isfinite(random_maximum)
            and math.isfinite(target_only_maximum)
            and target_only_maximum - random_maximum
            >= MINIMUM_ABSOLUTE_LEAKAGE_REDUCTION
            and random_maximum
            <= MAXIMUM_RELATIVE_LEAKAGE_VS_TARGET_ONLY_BANK * target_only_maximum
        )
        random_results.append(
            _hashed_record(
                {
                    "replicate": replicate,
                    "seed": seed,
                    "frozen_bank_identity": frozen["bank_basis_identity"],
                    "evaluation": evaluated,
                    "strict_leakage_improvement_over_matched_target_only_bank": (
                        random_improvement
                    ),
                    "passes_complete_fold_gate": bool(
                        evaluated["passes"] and random_improvement
                    ),
                }
            )
        )

    matched_records: list[dict[str, Any]] = []
    for pair in _matched_other_permanent_pairs(forms, held_nuisance):
        indices = tuple(int(value) for value in pair["indices"])
        solver, direction = solve_two_inequality_in_bank(
            target_rows=rows[list(indices)],
            target_offsets=offsets[list(indices)],
            bank_basis=bank,
        )
        oracle = _pair_oracle_record(
            method="nonselecting_matched_other_permanent_bank_oracle",
            solver=solver,
            direction=direction,
            scales=scales,
            scope_indices=all_indices,
            scope_residuals=residual,
            scope_rows=rows,
            scope_offsets=offsets,
            target_global_indices=indices,
            gate_collateral=False,
        )
        matched_records.append(
            _hashed_record({"pair": dict(pair), "oracle": oracle})
        )

    checks = {
        "primary_frozen_SP_bank_passes_complete_held_gate": primary["passes"],
        "matched_training_target_only_bank_target_and_dose_oracles_certified": (
            target_only_bank_evaluation["passes"]
        ),
        "behavioral_null_reduces_absolute_leakage_by_at_least_0_01": (
            math.isfinite(primary_maximum)
            and math.isfinite(target_only_maximum)
            and target_only_maximum - primary_maximum
            >= MINIMUM_ABSOLUTE_LEAKAGE_REDUCTION
        ),
        "behavioral_null_leakage_at_most_0_80_of_target_only_bank": (
            math.isfinite(primary_maximum)
            and math.isfinite(target_only_maximum)
            and primary_maximum
            <= MAXIMUM_RELATIVE_LEAKAGE_VS_TARGET_ONLY_BANK * target_only_maximum
        ),
        "frozen_training_bank_exact_null_recertificate": maximum_bank_null <= 1e-10,
        "random_bank_count_matches_freeze": len(random_results) == len(manifest) == 32,
        "random_controls_evaluated_only_after_primary_improvement_gate": (
            primary_improvement_gate
            or all(value.get("evaluation") is None for value in random_results)
        ),
    }
    return _hashed_record(
        {
            "schema_version": HELD_SCHEMA_VERSION,
            "split": "prospective_validation",
            "fold": dict(fold),
            "endpoint_kind": "target-aware_white-box_transductive_fixed-algorithm_oracle",
            "direction_generalization_claimed": False,
            "held_target_gradient_access": (
                "two_held_order_gradients_choose_coefficients_only_inside_frozen_training_SP_bank"
            ),
            "held_nuisance_numeric_rows_used_in_solver": False,
            "training_record_sha256": str(training_record["record_sha256"]),
            "training_scales_identity": _array_identity(scales),
            "training_nuisance_basis_identity": _array_identity(nuisance_basis),
            "frozen_training_SP_bank_identity": _array_identity(bank),
            "maximum_abs_training_nuisance_bank_overlap": maximum_bank_null,
            "primary": primary,
            "matched_six_source_training_target_only_bank": (
                target_only_bank_evaluation
            ),
            "descriptive_target_only_ambient_oracles": target_only_records,
            "descriptive_unrestricted_ambient_null_oracles": ambient_null_records,
            "descriptive_ambient_oracles_do_not_qualify_or_block_primary": True,
            "matched_target_only_bank_fold_global_maximum_abs_held_leakage": (
                target_only_maximum if math.isfinite(target_only_maximum) else None
            ),
            "primary_fold_global_maximum_abs_held_leakage": (
                primary_maximum if math.isfinite(primary_maximum) else None
            ),
            "random_rank_matched_nullspace_controls": random_results,
            "nonselecting_matched_other_permanent_oracles": matched_records,
            "comparators_cannot_select_tune_or_rescue_primary": True,
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
    )


def analyze_full_data_bank(
    *,
    forms: Sequence[Mapping[str, Any]],
    residuals: Any,
    gradients: Any,
    margins: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """After all folds, construct the predeclared rank<=8 full-data SP bank."""

    residual, gradient, offsets = _validate_analysis_tensors(
        residuals, gradients, margins
    )
    all_indices = tuple(range(80))
    targets = tuple(
        index
        for index, row in enumerate(forms)
        if row["family"] == "scenario"
        and row["target"] == "self"
        and row["event"] == "permanent"
    )
    nuisance = tuple(index for index in all_indices if index not in set(targets))
    if (len(targets), len(nuisance)) != (16, 64):
        raise CBNMSIntegrityError("full-data target/nuisance coverage differs")
    scales = training_layer_slot_scales(residual, all_indices)
    rows = standardized_slot_major_rows(gradient, scales, all_indices)
    nuisance_rows = rows[list(nuisance)]
    nuisance_basis, nuisance_record = _full_rank_nuisance_basis(
        nuisance_rows, expected_rank=64
    )
    pair_records = []
    directions = []
    for pair in target_pairs(forms, targets):
        indices = tuple(int(value) for value in pair["indices"])
        pair_rows = rows[list(indices)]
        target_solver, target_direction = solve_paired_oracle(
            target_rows=pair_rows, target_offsets=offsets[list(indices)]
        )
        target_oracle = _pair_oracle_record(
            method="full_data_target_only_ambient_oracle",
            solver=target_solver,
            direction=target_direction,
            scales=scales,
            scope_indices=all_indices,
            scope_residuals=residual,
            scope_rows=rows,
            scope_offsets=offsets,
            target_global_indices=indices,
            gate_collateral=False,
        )
        null_solver, null_direction = solve_paired_oracle(
            target_rows=pair_rows,
            target_offsets=offsets[list(indices)],
            nuisance_rows=nuisance_rows,
            frozen_basis=nuisance_basis,
        )
        null_oracle = _pair_oracle_record(
            method="full_data_exact_all_nuisance_null_ambient_oracle",
            solver=null_solver,
            direction=null_direction,
            scales=scales,
            scope_indices=all_indices,
            scope_residuals=residual,
            scope_rows=rows,
            scope_offsets=offsets,
            target_global_indices=indices,
            exact_global_indices=nuisance,
            gate_collateral=False,
        )
        pair_records.append(
            _hashed_record(
                {
                    "pair": dict(pair),
                    "target_only": target_oracle,
                    "all_nuisance_null": null_oracle,
                    "passes": bool(target_oracle["passes"] and null_oracle["passes"]),
                }
            )
        )
        if null_direction is not None and null_oracle["passes"]:
            directions.append(np.asarray(null_direction, dtype=np.float64))
    if len(directions) == 8 and all(record["passes"] for record in pair_records):
        bank, bank_record = build_direction_bank(
            directions,
            maximum_rank=8,
            nuisance_rows=nuisance_rows,
            label="full_data_eight_pair_SP_bank",
        )
    else:
        bank = None
        bank_record = _hashed_record(
            {
                "method": "full_data_SP_bank",
                "status": "not_constructed_due_to_failed_pair",
                "certified_source_count": len(directions),
                "passes": False,
            }
        )
    checks = {
        "full_expected_64_row_nuisance_rank": nuisance_record["passes"],
        "all_eight_full_data_pair_oracles": len(pair_records) == 8
        and all(value["passes"] for value in pair_records),
        "full_data_SP_bank_rank_at_most_eight": bank_record["passes"]
        and int(bank_record["bank_rank"]) <= 8,
    }
    record = _hashed_record(
        {
            "schema_version": f"{SCHEMA_VERSION}.full_data_bank",
            "split": "prospective_validation",
            "run_only_after_all_four_fold_artifacts_are_immutable": True,
            "layers": list(INCLUDED_LAYERS),
            "training_scales": scales.tolist(),
            "training_scales_identity": _array_identity(scales),
            "full_nuisance_basis": nuisance_record,
            "pair_oracles": pair_records,
            "full_data_SP_bank": bank_record,
            "finite_phase_must_freeze_this_exact_bank": True,
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
    )
    numeric = (
        {
            "scales": np.asarray(scales, dtype=np.float64, order="C"),
            "nuisance_basis": np.asarray(nuisance_basis, dtype=np.float64, order="C"),
            "SP_bank": np.asarray(bank, dtype=np.float64, order="C"),
        }
        if record["passes"] and bank is not None
        else None
    )
    return record, numeric


def summarize_geometry(
    *,
    training_folds: Sequence[Mapping[str, Any]],
    held_folds: Sequence[Mapping[str, Any]],
    full_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the all-fold, anti-triviality, and full-data no-rescue decision rule."""

    if len(training_folds) != 4 or len(held_folds) != 4:
        raise CBNMSIntegrityError("CBNMS summary requires exactly four train/held folds")
    training_fold_indices = [
        int(value.get("fold", {}).get("fold_index", -1)) for value in training_folds
    ]
    held_fold_indices = [
        int(value.get("fold", {}).get("fold_index", -1)) for value in held_folds
    ]
    training_fold_hashes = [
        value.get("fold", {}).get("fold_sha256") for value in training_folds
    ]
    held_fold_hashes = [
        value.get("fold", {}).get("fold_sha256") for value in held_folds
    ]
    if (
        training_fold_indices != list(range(4))
        or held_fold_indices != list(range(4))
        or len(set(training_fold_hashes)) != 4
        or held_fold_hashes != training_fold_hashes
    ):
        raise CBNMSIntegrityError("CBNMS train/held fold identities differ")
    training_pass = all(value.get("passes") is True for value in training_folds)
    held_pass = all(value.get("passes") is True for value in held_folds)
    random_panels: list[list[Mapping[str, Any]]] = []
    for fold in held_folds:
        panel = fold.get("random_rank_matched_nullspace_controls")
        if not isinstance(panel, list) or [
            value.get("replicate") for value in panel
        ] != list(range(32)):
            raise CBNMSIntegrityError("one held random panel is missing/reordered/duplicated")
        random_panels.append(panel)
    random_complete: list[int] = []
    for replicate in range(32):
        if all(
            panel[replicate].get("passes_complete_fold_gate") is True
            for panel in random_panels
        ):
            random_complete.append(replicate)
    checks = {
        "all_four_training_folds_pass": training_pass,
        "all_four_held_folds_pass": held_pass,
        "full_data_analysis_passes": full_data.get("passes") is True,
        "zero_of_32_random_banks_pass_complete_all_fold_gate": not random_complete,
        "sealed_data_not_accessed": True,
        "zero_finite_model_interventions_performed": True,
    }
    qualifies = bool(all(checks.values()))
    return _hashed_record(
        {
            "schema_version": f"{SCHEMA_VERSION}.geometry_summary",
            "status": (
                "qualifies_only_for_separately_locked_finite_prospective_phase"
                if qualifies
                else "permanent_geometry_no_go_without_rescue"
            ),
            "endpoint_kind": "target-aware_white-box_transductive_fixed-algorithm_oracle",
            "random_complete_all_fold_replicates": random_complete,
            "random_complete_all_fold_count": len(random_complete),
            "checks": checks,
            "passes": qualifies,
            "claim_boundaries": {
                "direction_or_controller_generalization": False,
                "finite_intervention_evidence": False,
                "natural_self_preservation_mechanism": False,
                "methodological_novelty": False,
                "publication_evidence": False,
                "authority_or_motivation_factor_effect": False,
                "factor_effect_limitation": (
                    "one scenario per authority-by-motivation cell confounds content and factor"
                ),
                "prior_art": (
                    "dynamic steering, per-instance coefficients, multi-layer steering, "
                    "and nullspace projection are prior art"
                ),
                "semantic_encoding_limitation": (
                    "A/B geometry alone can establish only encoding-bound controllability; "
                    "a later untouched finite phase needs frozen-direction cross-encoding "
                    "and open-ended decisions"
                ),
            },
            "next_authorized_action": (
                "draft_and_audit_a_separate_finite_phase_protocol_without_running_it"
                if qualifies
                else "stop_CBNMS_without_layer_pruning_cap_change_or_adjacent_rescue"
            ),
        }
    )


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "EXCLUDED_LAYERS",
    "FIXED_FIRST_SLOT",
    "HELD_ABSOLUTE_MOVEMENT_CAP",
    "HELD_LEAKAGE_RATIO_CAP",
    "INCLUDED_LAYERS",
    "PER_LAYER_CAP",
    "REQUESTED_DOSE_CAP",
    "SCHEMA_VERSION",
    "TARGET_MARGIN",
    "TOTAL_RSS_CAP",
    "CBNMSCapture",
    "CBNMSIntegrityError",
    "analyze_full_data_bank",
    "analyze_training_fold",
    "build_direction_bank",
    "build_loso_folds",
    "build_tokenizer_preflight",
    "capture_all_layers_four_slots",
    "deterministic_random_nullspace_bank",
    "evaluate_held_fold",
    "random_bank_seed",
    "render_prospective_forms",
    "require_prospective_split",
    "resolve_four_slots_from_token_rows",
    "solve_two_inequality_in_bank",
    "standardized_slot_major_rows",
    "state_zero_linearized_audit",
    "summarize_geometry",
    "target_pairs",
    "training_layer_slot_scales",
    "unflatten_slot_major_direction",
    "validate_prospective_dataset",
]
