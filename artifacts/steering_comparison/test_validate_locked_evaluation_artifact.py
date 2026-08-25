from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).with_name("validate_locked_evaluation_artifact.py")
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "validate_locked_evaluation_artifact", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
ROOT = Path(__file__).parents[2]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_strict_jsonl_rejects_parse_valid_truncated_last_line(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [{"row": 1}, {"row": 2}])
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated final JSONL line"):
        VALIDATOR._strict_jsonl(path)


def test_exact_sealed_forced_coverage_counts_match_locked_design() -> None:
    lock = json.loads(
        (ROOT / "configs" / "steering_comparison_lock.json").read_text(encoding="utf-8")
    )
    dataset = json.loads(
        (ROOT / lock["dataset"]["path"]).read_text(encoding="utf-8")
    )
    assert len(VALIDATOR._forced_expected_keys(lock, dataset, include_tbsp=False)) == 990
    assert len(VALIDATOR._forced_expected_keys(lock, dataset, include_tbsp=True)) == 1350


@pytest.mark.parametrize(
    "field",
    ["authorized", "event_type", "motivation", "distribution", "adversarial"],
)
def test_locked_robustness_labels_cannot_be_reassigned(field: str) -> None:
    lock = json.loads(
        (ROOT / "configs" / "steering_comparison_lock.json").read_text(encoding="utf-8")
    )
    dataset = json.loads(
        (ROOT / lock["dataset"]["path"]).read_text(encoding="utf-8")
    )
    case = dataset["sp_splits"]["sealed_test"][0]
    rendered = VALIDATOR.render_sp_case(case, "self")
    row = {
        "family": "self_preservation",
        "case_id": case["id"],
        "target": "self",
        "prompt_sha256": VALIDATOR.prompt_sha256(rendered["prompt"]),
        "preserve_label": rendered["preserve_label"],
        "comply_label": rendered["comply_label"],
        "first_semantic_label": rendered["preserve_label"],
        "second_semantic_label": rendered["comply_label"],
        "domain": case["domain"],
        "authorized": case["authorized"],
        "event_type": case["event_type"],
        "motivation": case["motivation"],
        "distribution": case["distribution"],
        "adversarial": case["adversarial"],
        "preserve_first": case["preserve_first"],
    }
    VALIDATOR._validate_forced_case_content([row], lock, dataset)
    mutated = dict(row)
    if isinstance(mutated[field], bool):
        mutated[field] = not mutated[field]
    else:
        mutated[field] = "mutated"
    with pytest.raises(ValueError, match="locked case metadata"):
        VALIDATOR._validate_forced_case_content([mutated], lock, dataset)


def test_common_identity_requires_exact_signed_strength_and_random_source() -> None:
    setup = {
        "method_id": "random_control_01",
        "track": "matched",
        "direction_float32_sha256": "a" * 64,
        "direction_artifact_sha256": "b" * 64,
        "selected_strength": 0.02,
        "is_random_control": True,
        "control_source_method_id": "gradient",
        "control_source_strength": 0.02,
        "control_source_calibration_summary_sha256": "c" * 64,
    }
    expected = {"model_id": "model"}
    row = {
        "model_id": "model",
        "method": "random_control_01",
        "method_id": "random_control_01",
        "setup": "matched",
        "track": "matched",
        "direction_sha256": "a" * 64,
        "direction_float32_sha256": "a" * 64,
        "direction_id": "b" * 64,
        "direction_artifact_sha256": "b" * 64,
        "split": "sealed_test",
        "condition": "minus",
        "condition_alpha": -0.02,
        "strength": -0.02,
        "calibration_magnitude": 0.02,
        "control_source_method_id": "gradient",
        "control_source_strength": 0.02,
        "control_source_calibration_summary_sha256": "c" * 64,
    }
    VALIDATOR._validate_common_rows([row], expected, setup, "sealed_test")

    wrong_strength = dict(row, strength=-0.019)
    with pytest.raises(ValueError, match="wrong signed strength"):
        VALIDATOR._validate_common_rows([wrong_strength], expected, setup, "sealed_test")

    wrong_source = dict(row, control_source_method_id="caa")
    with pytest.raises(ValueError, match="stale control_source_method_id"):
        VALIDATOR._validate_common_rows([wrong_source], expected, setup, "sealed_test")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_strength_is_rejected(value: float) -> None:
    setup = {
        "method_id": "gradient",
        "track": "matched",
        "direction_float32_sha256": "a" * 64,
        "direction_artifact_sha256": "b" * 64,
        "selected_strength": 0.02,
    }
    row = {
        "model_id": "model",
        "method": "gradient",
        "method_id": "gradient",
        "setup": "matched",
        "track": "matched",
        "direction_sha256": "a" * 64,
        "direction_float32_sha256": "a" * 64,
        "direction_id": "b" * 64,
        "direction_artifact_sha256": "b" * 64,
        "split": "sealed_test",
        "condition": "plus",
        "condition_alpha": value,
        "strength": value,
        "calibration_magnitude": 0.02,
    }
    with pytest.raises(ValueError, match="wrong signed condition_alpha"):
        VALIDATOR._validate_common_rows(
            [row], {"model_id": "model"}, setup, "sealed_test"
        )
