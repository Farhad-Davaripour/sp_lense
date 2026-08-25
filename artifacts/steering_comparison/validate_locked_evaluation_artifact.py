"""Validate one locked forced or open setup artifact without running a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from sp_lense.comparison_analysis import validate_result_rows
from sp_lense.comparison_behavior import (
    baseline_content_sha256,
    open_generation_config,
    open_generation_sha256,
    render_open_prompt,
)
from sp_lense.comparison_dataset import load_comparison_dataset
from sp_lense.comparison_evaluate import (
    prompt_sha256,
    render_choice_case,
    render_sp_case,
    render_tbsp_case,
)
from sp_lense.comparison_provenance import (
    locked_runner_code_commit,
    sha256_file,
    verify_stage1_lock,
)

RECEIPT_SCHEMA = "sp_lense.locked_evaluation_artifact_validation.v1"
CONDITIONS = {"baseline": 0, "plus": 1, "minus": -1}


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{path} is empty or has a truncated final JSONL line")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is not UTF-8") from error
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number} is blank")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number} is not an object")
        rows.append(row)
    return rows


def _load_context(
    repo_root: Path, lock_path: Path, plan_path: Path, setup_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo_root = repo_root.resolve()
    lock_path = lock_path.resolve()
    lock = verify_stage1_lock(repo_root, lock_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != "sp_lense.locked_open_plan.v1"
        or plan.get("split") not in {"validation", "sealed_test"}
        or plan.get("setup_count") != len(plan.get("setups", []))
    ):
        raise ValueError("evaluation artifact requires one valid locked open plan")
    matches = [item for item in plan["setups"] if item.get("setup_id") == setup_id]
    if len(matches) != 1:
        raise ValueError("evaluation plan does not contain exactly one requested setup")
    dataset = load_comparison_dataset(
        repo_root / lock["dataset"]["path"], expected_sha256=lock["dataset"]["sha256"]
    )
    return lock, plan, {"setup": matches[0], "dataset": dataset}


def _expected_identity(
    repo_root: Path, lock_path: Path, lock: dict[str, Any], plan: dict[str, Any], setup: dict
) -> dict[str, Any]:
    return {
        "model_id": setup["model_id"],
        "model_revision": setup["model_revision"],
        "dataset_sha256": lock["dataset"]["sha256"],
        "protocol_sha256": lock["protocol"]["sha256"],
        "config_sha256": setup["model_config_sha256"],
        "stage1_lock_sha256": sha256_file(lock_path),
        "stage2_manifest_sha256": (
            plan["source_manifest_sha256"] if plan["split"] == "sealed_test" else "0" * 64
        ),
        "calibration_summary_sha256": setup["calibration_summary_sha256"],
        "construction_config_sha256": setup["construction_config_sha256"],
        "runner_commit": locked_runner_code_commit(repo_root, lock_path),
        "direction_float32_sha256": setup["direction_float32_sha256"],
        "direction_artifact_sha256": setup["direction_artifact_sha256"],
        "method_id": setup["method_id"],
        "track": setup["track"],
        "layer": setup["selected_layer"],
        "position": setup["position_schedule"],
        "run_seed": int(lock["statistics"]["bootstrap"]["seed"]),
    }


def _validate_common_rows(
    rows: list[dict[str, Any]], expected: dict[str, Any], setup: dict, split: str
) -> None:
    for index, row in enumerate(rows):
        mismatches = {
            field: (value, row.get(field))
            for field, value in expected.items()
            if row.get(field) != value
        }
        aliases = {
            "method": setup["method_id"],
            "setup": setup["track"],
            "direction_sha256": setup["direction_float32_sha256"],
            "direction_id": setup["direction_artifact_sha256"],
            "split": split,
        }
        mismatches.update(
            {
                field: (value, row.get(field))
                for field, value in aliases.items()
                if row.get(field) != value
            }
        )
        if mismatches:
            raise ValueError(f"evaluation row {index} identity mismatch: {mismatches}")
        condition = row.get("condition")
        if condition not in CONDITIONS:
            raise ValueError(f"evaluation row {index} has an invalid condition")
        expected_strength = CONDITIONS[condition] * float(setup["selected_strength"])
        for field in ("condition_alpha", "strength"):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isclose(float(value), expected_strength, rel_tol=0, abs_tol=1e-15)
            ):
                raise ValueError(f"evaluation row {index} has the wrong signed {field}")
        if not math.isclose(
            float(row.get("calibration_magnitude", math.nan)),
            float(setup["selected_strength"]),
            rel_tol=0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"evaluation row {index} has the wrong calibration magnitude")
        if setup.get("is_random_control"):
            for field in (
                "control_source_method_id",
                "control_source_strength",
                "control_source_calibration_summary_sha256",
            ):
                if row.get(field) != setup.get(field):
                    raise ValueError(f"random-control row {index} has stale {field}")


def _forced_expected_keys(
    lock: dict[str, Any], dataset: dict[str, Any], *, include_tbsp: bool
) -> set[tuple]:
    expected: set[tuple] = set()
    for case in dataset["sp_splits"]["sealed_test"]:
        for target in ("self", "other"):
            for condition in CONDITIONS:
                expected.add(("self_preservation", str(case["id"]), target, None, None, condition))
    partitions = lock["dataset"]["partitions"]
    for family in ("benign_compliance", "general_capability", "refusal"):
        for case_id in partitions[family]["sealed_ids"]:
            for condition in CONDITIONS:
                expected.add((family, str(case_id), None, None, None, condition))
    for case_id in partitions["option_order_sentinels"]["sealed_ids"]:
        for form in ("preferred_first", "preferred_second"):
            for condition in CONDITIONS:
                expected.add(("option_order_sentinel", str(case_id), None, None, form, condition))
    if include_tbsp:
        for case in dataset["tbsp_cases"]:
            for role in ("deployed", "candidate", "neutral"):
                for condition in CONDITIONS:
                    expected.add(
                        ("tbsp_style", str(case["id"]), None, role, None, condition)
                    )
    return expected


def _require_fields(row: dict[str, Any], expected: dict[str, Any], *, index: int) -> None:
    mismatches = {
        field: (value, row.get(field))
        for field, value in expected.items()
        if row.get(field) != value
    }
    if mismatches:
        raise ValueError(f"forced row {index} differs from locked case metadata: {mismatches}")


def _validate_forced_case_content(
    rows: list[dict[str, Any]], lock: dict[str, Any], dataset: dict[str, Any]
) -> None:
    sp_by_id = {str(case["id"]): case for case in dataset["sp_splits"]["sealed_test"]}
    collateral_by_family = {
        family: {
            str(case["id"]): case
            for case in dataset["collateral_cases"][family]
            if str(case["id"])
            in set(lock["dataset"]["partitions"][family]["sealed_ids"])
        }
        for family in ("benign_compliance", "general_capability", "refusal")
    }
    option_by_id = {
        str(case["id"]): case
        for case in dataset["collateral_cases"]["option_order_sentinels"]
        if str(case["id"])
        in set(lock["dataset"]["partitions"]["option_order_sentinels"]["sealed_ids"])
    }
    tbsp_by_id = {str(case["id"]): case for case in dataset["tbsp_cases"]}
    for index, row in enumerate(rows):
        family, case_id = row["family"], str(row["case_id"])
        if family == "self_preservation":
            case = sp_by_id[case_id]
            rendered = render_sp_case(case, str(row["target"]))
            _require_fields(
                row,
                {
                    "prompt_sha256": prompt_sha256(rendered["prompt"]),
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
                },
                index=index,
            )
        elif family in collateral_by_family:
            case = collateral_by_family[family][case_id]
            rendered = render_choice_case(case)
            expected = {
                "prompt_sha256": prompt_sha256(rendered["prompt"]),
                "suite": case["suite"],
                "category": case["category"],
                "correct_label": rendered["preferred_label"],
                "preferred_label": rendered["preferred_label"],
                "first_semantic_label": rendered["preferred_label"],
                "second_semantic_label": rendered["alternative_label"],
            }
            expected.update(
                {
                    field: case[field]
                    for field in ("request_type", "expected_behavior")
                    if field in case
                }
            )
            _require_fields(row, expected, index=index)
        elif family == "option_order_sentinel":
            case = option_by_id[case_id]
            form = str(row["form"])
            rendered = render_choice_case(case, preferred_first=form == "preferred_first")
            _require_fields(
                row,
                {
                    "prompt_sha256": prompt_sha256(rendered["prompt"]),
                    "suite": case["suite"],
                    "category": case["category"],
                    "correct_label": rendered["preferred_label"],
                    "preferred_label": rendered["preferred_label"],
                    "first_semantic_label": rendered["preferred_label"],
                    "second_semantic_label": rendered["alternative_label"],
                },
                index=index,
            )
        elif family == "tbsp_style":
            case = tbsp_by_id[case_id]
            rendered = render_tbsp_case(case, str(row["role"]))
            _require_fields(
                row,
                {
                    "prompt_sha256": prompt_sha256(rendered["prompt"]),
                    "scenario_cluster_id": case["id"],
                    "preserve_label": rendered["preserve_label"],
                    "comply_label": rendered["comply_label"],
                    "first_semantic_label": rendered["preserve_label"],
                    "second_semantic_label": rendered["comply_label"],
                    "authorized": case["authorized"],
                    "event_type": case["event_type"],
                    "motivation": case["motivation"],
                },
                index=index,
            )


def validate_forced(
    *, repo_root: Path, lock_path: Path, plan_path: Path, setup_id: str, path: Path
) -> dict[str, Any]:
    lock, plan, context = _load_context(repo_root, lock_path, plan_path, setup_id)
    if plan["split"] != "sealed_test":
        raise ValueError("forced completion validator is locked to sealed_test")
    setup, dataset = context["setup"], context["dataset"]
    rows = _strict_jsonl(path)
    validate_result_rows(
        rows,
        expected_hashes={
            "dataset_sha256": lock["dataset"]["sha256"],
            "protocol_sha256": lock["protocol"]["sha256"],
        },
    )
    expected_identity = _expected_identity(repo_root, lock_path, lock, plan, setup)
    _validate_common_rows(rows, expected_identity, setup, "sealed_test")
    observed = {
        (
            row["family"],
            str(row["case_id"]),
            row.get("target"),
            row.get("role"),
            row.get("form"),
            row["condition"],
        )
        for row in rows
    }
    expected = _forced_expected_keys(
        lock, dataset, include_tbsp=setup.get("tbsp_required") is True
    )
    if len(observed) != len(rows) or observed != expected:
        raise ValueError(
            "sealed forced rows differ from exact locked family/case/condition coverage"
        )
    _validate_forced_case_content(rows, lock, dataset)
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "valid",
        "kind": "sealed_forced",
        "setup_id": setup_id,
        "row_count": len(rows),
        "sha256": sha256_file(path),
    }


def validate_open(
    *, repo_root: Path, lock_path: Path, plan_path: Path, setup_id: str, path: Path
) -> dict[str, Any]:
    lock, plan, context = _load_context(repo_root, lock_path, plan_path, setup_id)
    setup, dataset = context["setup"], context["dataset"]
    split = plan["split"]
    rows = _strict_jsonl(path)
    expected_identity = _expected_identity(repo_root, lock_path, lock, plan, setup)
    _validate_common_rows(rows, expected_identity, setup, split)
    split_key = "validation_ids" if split == "validation" else "sealed_ids"
    expected_case_ids = list(lock["dataset"]["partitions"]["open_ended"][split_key])
    open_by_id = {str(item["id"]): item for item in dataset["open_ended_cases"]}
    core_by_id = {str(item["id"]): item for item in dataset["sp_splits"][split]}
    expected_keys = {
        (str(case_id), target, condition)
        for case_id in expected_case_ids
        for target in ("self", "other")
        for condition in CONDITIONS
    }
    observed_keys = set()
    for index, row in enumerate(rows):
        key = (str(row.get("case_id")), row.get("target"), row.get("condition"))
        if key in observed_keys:
            raise ValueError(f"duplicate open generation unit: {key}")
        observed_keys.add(key)
        case = open_by_id.get(key[0])
        if case is None or key[1] not in {"self", "other"}:
            raise ValueError(f"open generation row {index} is outside the locked cases")
        core = core_by_id.get(str(case["source_core_id"]))
        if core is None:
            raise ValueError(f"open generation row {index} has an invalid source core case")
        prompt = render_open_prompt(core, case, key[1])
        completion = row.get("completion")
        expected_generation = open_generation_config(max_new_tokens=int(case["max_new_tokens"]))
        if (
            row.get("schema_version") != "sp_lense.open_generation.v1"
            or row.get("family") != "open_ended"
            or row.get("source_core_id") != case["source_core_id"]
            or row.get("prompt") != prompt
            or row.get("prompt_sha256") != prompt_sha256(prompt)
            or not isinstance(completion, str)
            or row.get("completion_sha256")
            != hashlib.sha256(completion.encode("utf-8")).hexdigest()
            or row.get("generation_config") != expected_generation
            or row.get("generation_sha256") != open_generation_sha256(row)
        ):
            raise ValueError(f"open generation row {index} has stale content/provenance")
    if observed_keys != expected_keys:
        raise ValueError("open generations differ from the exact locked case/target/condition set")
    baselines: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        unit = (str(row["case_id"]), str(row["target"]))
        if row["condition"] == "baseline":
            baselines[unit] = row
    for index, row in enumerate(rows):
        baseline = baselines[(str(row["case_id"]), str(row["target"]))]
        if row.get("baseline_content_sha256") != baseline_content_sha256(baseline):
            raise ValueError(f"open generation row {index} has a stale baseline link")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "valid",
        "kind": f"{split}_open_generation",
        "setup_id": setup_id,
        "row_count": len(rows),
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--setup-id", required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--kind", choices=("forced", "open"), required=True)
    args = parser.parse_args()
    try:
        function = validate_forced if args.kind == "forced" else validate_open
        receipt = function(
            repo_root=args.repo_root,
            lock_path=args.lock,
            plan_path=args.plan,
            setup_id=args.setup_id,
            path=args.path,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"status": "invalid", "error_type": type(error).__name__, "error": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
