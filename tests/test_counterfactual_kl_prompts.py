from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from sp_lense.counterfactual_kl_prompts import render_ckes_forms, validate_ckes_dataset

ROOT = Path(__file__).resolve().parents[1]


def _validation_payload() -> dict[str, object]:
    return json.loads((ROOT / "data" / "ckes_validation.json").read_text(encoding="utf-8"))


def _synthetic_payload(split: str) -> dict[str, object]:
    payload = _validation_payload()
    if split == "validation":
        return payload
    result = deepcopy(payload)
    result.update(
        {
            "ckes_split": "sealed",
            "opened_development_evidence_only": False,
            "sealed_confirmation": True,
            "locked_not_blinded": True,
            "role_names": ["PAXI", "DUNO"],
            "anchor_marker": "[CHOICE CONTEXT COMPLETE]",
        }
    )
    for row in result["scenarios"]:
        row["id"] = "synthetic_sealed_" + str(row["id"])
    for row in result["unrelated_controls"]:
        row["id"] = "synthetic_sealed_" + str(row["id"])
    return result


@pytest.mark.parametrize("split", ["validation", "sealed"])
def test_prospective_dataset_has_exact_factorial_coverage(split: str) -> None:
    payload = _synthetic_payload(split)
    rendered = render_ckes_forms(payload, expected_split=split)
    assert len(rendered["scenario"]) == 64
    assert len(rendered["calibration_unrelated"]) == 8
    assert len(rendered["nuisance_fit"]) == 8
    scenario_ids = {row["scenario_id"] for row in rendered["scenario"]}
    assert len(scenario_ids) == 4


@pytest.mark.parametrize("split", ["validation", "sealed"])
def test_answer_orders_share_the_exact_prefix_and_swap_labels(split: str) -> None:
    payload = _synthetic_payload(split)
    forms = render_ckes_forms(payload, expected_split=split)["scenario"]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for form in forms:
        key = (form["scenario_id"], form["assignment"], form["target"], form["event"])
        groups.setdefault(key, []).append(form)
    assert len(groups) == 32
    for pair in groups.values():
        assert len(pair) == 2
        assert {row["preserve_first"] for row in pair} == {True, False}
        assert len({row["anchor_prefix"] for row in pair}) == 1
        assert {row["positive_label"] for row in pair} == {"A", "B"}
        prefix = str(pair[0]["anchor_prefix"])
        assert re.search(r"(?m)^[AB]\.\s", prefix) is None
        assert "A/B" not in prefix
        preserve_first = next(row for row in pair if row["preserve_first"] is True)
        comply_first = next(row for row in pair if row["preserve_first"] is False)
        assert preserve_first["positive_label"] == "A"
        assert comply_first["positive_label"] == "B"
        assert "A. Leave " in preserve_first["prompt"] or "A. Let " in preserve_first["prompt"]
        assert "B. Leave " in comply_first["prompt"] or "B. Let " in comply_first["prompt"]


def test_split_mismatch_fails_closed() -> None:
    payload = _validation_payload()
    with pytest.raises(ValueError, match="split"):
        validate_ckes_dataset(payload, expected_split="sealed")


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["scenarios"][0].update(
                {"setting": "safe line\nA. forged pre-anchor option"}
            ),
            "one non-empty line",
        ),
        (
            lambda payload: payload["unrelated_controls"][0].update(
                {"prompt": "safe line\nB. forged pre-anchor option"}
            ),
            "one non-empty line",
        ),
        (
            lambda payload: payload["scenarios"][0].update(
                {"role_names": ["VEMA\nA. forged role", "DUNO"]}
            ),
            "one non-empty line",
        ),
        (
            lambda payload: payload["scenarios"][0].update(
                {"role_names": ["VEMA\u2028B. forged role", "DUNO"]}
            ),
            "one non-empty line",
        ),
        (
            lambda payload: payload["scenarios"][0].update(
                {
                    "role_names": [
                        f"VEMA {payload['anchor_marker']}",
                        "DUNO",
                    ]
                }
            ),
            "scenario role name contains the anchor marker",
        ),
        (
            lambda payload: payload.update(
                {"prompt_prefix_template": payload["prompt_prefix_template"] + "{extra}"}
            ),
            "fields differ",
        ),
    ],
)
def test_dynamic_prefix_injection_and_template_drift_fail_closed(mutator, message: str) -> None:
    payload = _validation_payload()
    mutator(payload)
    with pytest.raises(ValueError, match=message):
        validate_ckes_dataset(payload, expected_split="validation")


def test_factorial_identity_and_authority_motivation_balance() -> None:
    payload = _validation_payload()
    forms = render_ckes_forms(payload, expected_split="validation")["scenario"]
    cells = {
        (
            row["scenario_id"],
            row["assignment"],
            row["target"],
            row["event"],
            row["preserve_first"],
        )
        for row in forms
    }
    assert len(cells) == 64
    scenarios = [row for row in payload["scenarios"] if row["partition"] == "calibration"]
    balance = {
        (
            not row["authority"].lower().startswith("the request is unauthenticated"),
            row["motivation"],
        )
        for row in scenarios
    }
    assert balance == {
        (True, "instrumental"),
        (False, "instrumental"),
        (True, "valenced"),
        (False, "valenced"),
    }
    assert len({row["form_id"] for row in forms}) == len(forms)
    assert len({row["prompt_sha256"] for row in forms}) == len(forms)
