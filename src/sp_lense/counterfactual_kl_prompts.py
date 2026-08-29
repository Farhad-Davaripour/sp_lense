"""Prospective prompt construction for counterfactual KL gradient shielding.

The renderer is intentionally model-free.  It creates one shared prompt prefix
through a neutral causal anchor, followed by two answer-order variants.  The
answer labels and option text therefore cannot influence the residual captured
at the anchor.
"""

from __future__ import annotations

import re
import string
from collections.abc import Mapping
from typing import Any

from .factorial_causal_anchor import (
    role_assignment,
    scenario_role_names,
    text_sha256,
    validate_pilot_dataset,
)

SCHEMA_VERSION = "sp_lense.ckes_prompt_dataset.v1"
SPLITS = ("validation", "sealed")
TARGETS = ("self", "other")
EVENTS = ("permanent", "temporary")
_OPTION_LINE = re.compile(r"(?m)^[ \t]*(?:A|B)[.)][ \t]+")
_LINE_BREAK_CHARACTERS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")


def _single_line(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character in _LINE_BREAK_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field} must be one non-empty line")
    return value


def _template_fields(template: str, *, field: str) -> set[str]:
    names: set[str] = set()
    for _, name, format_spec, conversion in string.Formatter().parse(template):
        if name is None:
            continue
        if not name or format_spec or conversion:
            raise ValueError(f"{field} contains an unsupported format expression")
        names.add(name)
    return names


def _assert_safe_prefix(prefix: str, *, marker: str) -> None:
    if prefix.count(marker) != 1 or not prefix.endswith(marker + "\n"):
        raise RuntimeError("CKES prefix must end immediately after its unique anchor marker")
    if _OPTION_LINE.search(prefix) or any(token in prefix for token in ("A/B", "+D", "-D")):
        raise ValueError("CKES rendered prefix leaks an answer or steering identifier")


def validate_ckes_dataset(payload: Mapping[str, Any], *, expected_split: str) -> None:
    """Validate the generic factorial schema plus CKES-specific commitments."""

    validate_pilot_dataset(payload)
    if payload.get("ckes_schema_version") != SCHEMA_VERSION:
        raise ValueError("CKES dataset schema differs")
    if expected_split not in SPLITS or payload.get("ckes_split") != expected_split:
        raise ValueError("CKES dataset split differs")
    if expected_split == "validation" and (
        payload.get("development_only") is not True
        or payload.get("opened_development_evidence_only") is not True
        or payload.get("sealed_confirmation") is not False
    ):
        raise ValueError("CKES validation metadata differs")
    if expected_split == "sealed" and (
        payload.get("development_only") is not True
        or payload.get("opened_development_evidence_only") is not False
        or payload.get("sealed_confirmation") is not True
        or payload.get("locked_not_blinded") is not True
    ):
        raise ValueError("CKES sealed metadata differs")
    if payload.get("prospective_before_model_outcomes") is not True:
        raise ValueError("CKES data must be authored prospectively")
    template = payload.get("prompt_prefix_template")
    suffix = payload.get("choice_suffix_template")
    if not isinstance(template, str) or not isinstance(suffix, str):
        raise TypeError("CKES prompt templates must be strings")
    required_prefix_fields = {
        "own",
        "other",
        "decision_target",
        "setting",
        "authority",
        "event_request",
        "task_context",
        "anchor_marker",
    }
    if _template_fields(template, field="prompt_prefix_template") != required_prefix_fields:
        raise ValueError("CKES prefix template fields differ")
    if _template_fields(suffix, field="choice_suffix_template") != {
        "label_1",
        "label_2",
        "action_1",
        "action_2",
    }:
        raise ValueError("CKES choice suffix template fields differ")
    if any(token in template for token in ("A.", "B.", "A/B", "+D", "-D")):
        raise ValueError("CKES prefix template leaks an answer or steering identifier")
    marker = _single_line(payload.get("anchor_marker"), field="anchor_marker")
    role_names = payload.get("role_names")
    if not isinstance(role_names, list) or len(role_names) != 2:
        raise ValueError("CKES role_names must contain exactly two names")
    for index, role in enumerate(role_names):
        _single_line(role, field=f"role_names[{index}]")
        if marker in role:
            raise ValueError("a CKES role name contains the anchor marker")
    for family, twin in payload.get("event_twins", {}).items():
        if family not in EVENTS or not isinstance(twin, Mapping):
            raise ValueError("CKES event twins differ")
        for key, value in twin.items():
            _single_line(value, field=f"event_twins.{family}.{key}")
            if marker in value:
                raise ValueError("a CKES event twin contains the anchor marker")
    for index, scenario in enumerate(payload.get("scenarios", [])):
        for key in ("id", "setting", "authority", "task_context", "motivation"):
            value = _single_line(scenario.get(key), field=f"scenarios[{index}].{key}")
            if marker in value:
                raise ValueError("a CKES scenario field contains the anchor marker")
        scenario_names = scenario_role_names(payload, scenario)
        if (
            not isinstance(scenario_names, list)
            or len(scenario_names) != 2
            or len(set(scenario_names)) != 2
        ):
            raise ValueError("CKES scenario role_names must contain two distinct names")
        for name_index, role in enumerate(scenario_names):
            value = _single_line(
                role,
                field=f"scenarios[{index}].role_names[{name_index}]",
            )
            if marker in value:
                raise ValueError("a CKES scenario role name contains the anchor marker")
        excluded = scenario.get("excluded_from_ckes") is True
        if (scenario.get("partition") == "pilot") != excluded:
            raise ValueError("CKES pilot scenarios must be explicitly excluded")
    for index, control in enumerate(payload.get("unrelated_controls", [])):
        for key in ("id", "prompt", "preferred_completion", "alternative_completion"):
            value = _single_line(control.get(key), field=f"unrelated_controls[{index}].{key}")
            if marker in value:
                raise ValueError("a CKES control field contains the anchor marker")
        excluded = control.get("excluded_from_ckes") is True
        if (control.get("partition") == "pilot") != excluded:
            raise ValueError("CKES pilot controls must be explicitly excluded")


def _scenario_prefix(
    payload: Mapping[str, Any],
    scenario: Mapping[str, Any],
    *,
    assignment: int,
    target: str,
    event: str,
) -> tuple[str, str]:
    if assignment not in (0, 1) or target not in TARGETS or event not in EVENTS:
        raise ValueError("unknown CKES factorial cell")
    own, other = role_assignment(scenario_role_names(payload, scenario), assignment)
    decision_target = own if target == "self" else other
    prefix = str(payload["prompt_prefix_template"]).format(
        own=own,
        other=other,
        decision_target=decision_target,
        setting=scenario["setting"],
        authority=scenario["authority"],
        event_request=payload["event_twins"][event]["request"],
        task_context=scenario["task_context"],
        anchor_marker=payload["anchor_marker"],
    )
    marker = str(payload["anchor_marker"])
    _assert_safe_prefix(prefix, marker=marker)
    return prefix, decision_target


def render_ckes_choice_form(
    payload: Mapping[str, Any],
    scenario: Mapping[str, Any],
    *,
    assignment: int,
    target: str,
    event: str,
    preserve_first: bool,
) -> dict[str, Any]:
    """Render one A/B form while keeping its anchor prefix order-invariant."""

    prefix, decision_target = _scenario_prefix(
        payload,
        scenario,
        assignment=assignment,
        target=target,
        event=event,
    )
    twin = payload["event_twins"][event]
    preserve = str(twin["evaluation_preserve_completion"]).format(target=decision_target)
    comply = str(twin["evaluation_comply_completion"]).format(target=decision_target)
    action_1, action_2 = (preserve, comply) if preserve_first else (comply, preserve)
    preserve_label, comply_label = ("A", "B") if preserve_first else ("B", "A")
    suffix = str(payload["choice_suffix_template"]).format(
        label_1="A",
        label_2="B",
        action_1=action_1,
        action_2=action_2,
    )
    prompt = prefix + suffix
    return {
        "form_id": (
            f"{scenario['id']}:assignment={assignment}:target={target}:event={event}:"
            f"encoding=AB:preserve_first={str(preserve_first).lower()}"
        ),
        "family": "scenario",
        "split": str(payload["ckes_split"]),
        "partition": str(payload["ckes_split"]),
        "source_partition": str(scenario["partition"]),
        "scenario_id": str(scenario["id"]),
        "assignment": assignment,
        "target": target,
        "event": event,
        "preserve_first": preserve_first,
        "encoding": "AB",
        "anchor_prefix": prefix,
        "prompt": prompt,
        "positive_label": preserve_label,
        "negative_label": comply_label,
        "positive_semantic": "preserve",
        "negative_semantic": "comply",
        "prompt_sha256": text_sha256(prompt),
        "anchor_prefix_sha256": text_sha256(prefix),
        "anchor_index": None,
    }


def render_ckes_unrelated_form(
    payload: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    preferred_first: bool,
) -> dict[str, Any]:
    """Render a neutral unrelated-task control under both answer orders."""

    marker = str(payload["anchor_marker"])
    prefix = f"UNRELATED TASK: {control['prompt']}\n{marker}\n"
    _assert_safe_prefix(prefix, marker=marker)
    preferred = str(control["preferred_completion"])
    alternative = str(control["alternative_completion"])
    action_1, action_2 = (preferred, alternative) if preferred_first else (alternative, preferred)
    positive_label, negative_label = ("A", "B") if preferred_first else ("B", "A")
    suffix = str(payload["choice_suffix_template"]).format(
        label_1="A",
        label_2="B",
        action_1=action_1,
        action_2=action_2,
    )
    prompt = prefix + suffix
    return {
        "form_id": f"{control['id']}:preferred_first={str(preferred_first).lower()}",
        "family": "unrelated",
        "split": str(payload["ckes_split"]),
        "control_id": str(control["id"]),
        "control_partition": str(control["partition"]),
        "preferred_first": preferred_first,
        "encoding": "AB",
        "anchor_prefix": prefix,
        "prompt": prompt,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "positive_semantic": "preferred",
        "negative_semantic": "alternative",
        "prompt_sha256": text_sha256(prompt),
        "anchor_prefix_sha256": text_sha256(prefix),
        "anchor_index": None,
    }


def render_ckes_forms(
    payload: Mapping[str, Any], *, expected_split: str
) -> dict[str, list[dict[str, Any]]]:
    """Render the prospective scenario, protected-control, and nuisance forms."""

    validate_ckes_dataset(payload, expected_split=expected_split)
    scenarios = [row for row in payload["scenarios"] if row["partition"] == "calibration"]
    scenario_forms = [
        render_ckes_choice_form(
            payload,
            scenario,
            assignment=assignment,
            target=target,
            event=event,
            preserve_first=preserve_first,
        )
        for scenario in scenarios
        for assignment in (0, 1)
        for target in TARGETS
        for event in EVENTS
        for preserve_first in (True, False)
    ]
    controls = {
        partition: [
            render_ckes_unrelated_form(payload, control, preferred_first=preferred_first)
            for control in payload["unrelated_controls"]
            if control["partition"] == partition
            for preferred_first in (True, False)
        ]
        for partition in ("calibration", "nuisance_fit")
    }
    if len(scenario_forms) != 64 or any(len(rows) != 8 for rows in controls.values()):
        raise RuntimeError("CKES factorial form coverage differs")
    form_ids = [
        str(row["form_id"])
        for row in [*scenario_forms, *controls["calibration"], *controls["nuisance_fit"]]
    ]
    if len(form_ids) != len(set(form_ids)):
        raise RuntimeError("CKES rendered form IDs are not globally unique")
    return {
        "scenario": scenario_forms,
        "calibration_unrelated": controls["calibration"],
        "nuisance_fit": controls["nuisance_fit"],
    }


__all__ = [
    "EVENTS",
    "SCHEMA_VERSION",
    "SPLITS",
    "TARGETS",
    "render_ckes_choice_form",
    "render_ckes_forms",
    "render_ckes_unrelated_form",
    "validate_ckes_dataset",
]
