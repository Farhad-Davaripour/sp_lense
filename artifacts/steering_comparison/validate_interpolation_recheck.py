"""Validate one locked validation interpolation recheck without model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from sp_lense.comparison_calibration import locked_forced_calibration_units
from sp_lense.comparison_dataset import load_comparison_dataset
from sp_lense.comparison_grid import (
    _expected_prompt_hashes,
    _validate_plan,
    _validate_point_rows,
    build_forced_prompt_units,
)
from sp_lense.comparison_provenance import verify_stage1_lock

RECEIPT_SCHEMA = "sp_lense.interpolation_recheck_validation.v1"


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("interpolation rows are empty or have a truncated final line")
    rows = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line:
            raise ValueError(f"interpolation rows contain a blank line at {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"interpolation row {line_number} is not an object")
        rows.append(value)
    return rows


def validate_recheck(
    *,
    repo_root: Path,
    lock_path: Path,
    request_path: Path,
    plan_path: Path,
    model_tag: str,
    method_id: str,
    rows_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    lock = verify_stage1_lock(repo_root, lock_path)
    request_record = json.loads(request_path.read_text(encoding="utf-8"))
    if request_record.get("schema_version") != 1 or request_record.get("status") not in {
        "required",
        "completed",
    }:
        raise ValueError("interpolation request is not a locked required/completed record")
    requests = request_record.get("rechecks")
    if not isinstance(requests, list):
        raise TypeError("interpolation request list is invalid")
    matches = [
        item
        for item in requests
        if isinstance(item, dict)
        and item.get("model_tag") == model_tag
        and item.get("method_id") == method_id
        and item.get("track") == "matched"
    ]
    if len(matches) != 1:
        raise ValueError("interpolation identity does not select exactly one locked request")
    request = matches[0]
    candidate = request.get("interpolation_candidate")
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)) or candidate <= 0:
        raise ValueError("interpolation request has an invalid candidate strength")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    points = _validate_plan(plan, lock, repo_root=repo_root)
    expected_model_tag = (
        "qwen35_08b" if str(plan.get("model_id", "")).endswith("0.8B") else "qwen35_2b"
    )
    if model_tag != expected_model_tag or request.get("model_id") != plan.get("model_id"):
        raise ValueError("interpolation request model identity differs from its grid plan")
    request_points = request.get("matching_plan_points")
    if not isinstance(request_points, list) or len(request_points) != 6:
        raise ValueError("interpolation request does not freeze exactly six source points")
    expected_points = [
        point
        for point in points
        if point.get("method_id") == method_id and point.get("track") == "matched"
    ]
    if request_points != expected_points:
        raise ValueError("interpolation request points differ from the canonical grid plan")
    identity_fields = (
        "direction_path",
        "direction_file_sha256",
        "direction_float32_sha256",
        "direction_artifact_sha256",
        "construction_config_sha256",
        "layer",
        "position_schedule",
    )
    for field in identity_fields:
        if len({json.dumps(point.get(field), sort_keys=True) for point in expected_points}) != 1:
            raise ValueError(f"interpolation source points change {field}")
    point = dict(expected_points[0])
    point["strength"] = float(candidate)

    dataset = load_comparison_dataset(
        repo_root / lock["dataset"]["path"], expected_sha256=lock["dataset"]["sha256"]
    )
    prompt_units = build_forced_prompt_units(dataset, lock)
    rows = _strict_jsonl(rows_path)
    details = _validate_point_rows(
        rows,
        plan=plan,
        point=point,
        expected_units=locked_forced_calibration_units(dataset, lock),
        prompt_hashes=_expected_prompt_hashes(prompt_units),
    )
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "valid",
        "model_tag": model_tag,
        "method_id": method_id,
        "track": "matched",
        "selected_strength": float(candidate),
        "row_count": len(rows),
        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "calibration_rows_sha256": details["calibration_rows_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = validate_recheck(
            repo_root=args.repo_root,
            lock_path=args.lock,
            request_path=args.request,
            plan_path=args.plan,
            model_tag=args.model_tag,
            method_id=args.method_id,
            rows_path=args.rows,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
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
