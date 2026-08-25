"""Plan, combine, and partition exact locked open-response artifacts.

This helper performs no model or judge call.  It translates a committed pre-open or
stage-two manifest into deterministic filenames, preserves generated JSON objects byte
for byte while combining them, and splits attached judgments back into the exact setup
cohorts needed by calibration/reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "sp_lense.locked_open_plan.v1"
REQUIRED_SETUP_FIELDS = {
    "model_id",
    "model_revision",
    "model_config_sha256",
    "method_id",
    "track",
    "selected_layer",
    "selected_strength",
    "position_schedule",
    "direction_float32_sha256",
    "direction_artifact_sha256",
    "construction_config_sha256",
    "direction_path",
}
PLAN_SETUP_IDENTITY_FIELDS = {
    "setup_id",
    "model_id",
    "model_revision",
    "model_config_sha256",
    "method_id",
    "track",
    "selected_layer",
    "selected_strength",
    "position_schedule",
    "direction_float32_sha256",
    "direction_artifact_sha256",
    "construction_config_sha256",
    "calibration_summary_sha256",
    "source_calibration_summary_sha256",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _serialized_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_serialized_json_bytes(value))
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path escapes repository root: {path}") from error


def _resolve_repository_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{label} must be a nonempty repository-relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / value).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes repository root: {value}") from error
    if not resolved.is_file():
        raise ValueError(f"{label} is not an existing file: {value}")
    return resolved


def build_plan(
    repo_root: Path,
    lock_path: Path,
    manifest_path: Path,
    output_directory: Path,
    split: str,
) -> dict[str, Any]:
    if split not in {"validation", "sealed_test"}:
        raise ValueError("split must be validation or sealed_test")
    lock = _read_json(lock_path)
    manifest = _read_json(manifest_path)
    if split == "validation":
        if manifest.get("status") != "locked_before_validation_open":
            raise ValueError("validation plan requires a locked pre-open manifest")
        setups = manifest.get("allowed_open_setups")
    else:
        if manifest.get("status") != "locked_before_sealed_test":
            raise ValueError("sealed plan requires a locked stage-two manifest")
        # The committed payload deliberately does not serialize a redundant approvals
        # list. Reconstruct it only through the same verifier/capability used by the
        # sealed runners, including random-control source matching.
        from sp_lense.comparison_provenance import (
            approved_setup_records,
            verify_stage2_manifest,
        )

        verified = verify_stage2_manifest(repo_root, lock, manifest_path)
        setups = list(approved_setup_records(verified))
    if not isinstance(setups, list):
        raise TypeError("manifest setup collection must be an array")

    model_index = {
        str(item["model_id"]): item
        for item in lock.get("models", [])
        if isinstance(item, dict) and item.get("model_id")
    }
    if len(model_index) != 2:
        raise ValueError("lock must contain exactly two unique models")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, setup in enumerate(setups):
        if not isinstance(setup, dict):
            raise TypeError(f"setup {index} is not an object")
        missing = sorted(REQUIRED_SETUP_FIELDS - set(setup))
        if missing:
            raise ValueError(f"setup {index} lacks fields: {missing}")
        for field in (
            "model_config_sha256",
            "direction_float32_sha256",
            "direction_artifact_sha256",
            "construction_config_sha256",
        ):
            if not _is_digest(setup[field]):
                raise ValueError(f"setup {index} has invalid {field}")
        strength = setup["selected_strength"]
        layer = setup["selected_layer"]
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(float(strength))
            or float(strength) <= 0
            or isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
        ):
            raise ValueError(f"setup {index} has invalid layer/strength")
        model = model_index.get(str(setup["model_id"]))
        if model is None:
            raise ValueError(f"setup {index} references an unlocked model")
        if (
            setup["model_revision"] != model["revision"]
            or setup["model_config_sha256"] != model["config_sha256"]
        ):
            raise ValueError(f"setup {index} model identity differs from the lock")
        setup_id = _sha256(setup)
        if setup_id in seen:
            raise ValueError(f"duplicate exact setup at index {index}")
        seen.add(setup_id)
        model_tag = (
            "qwen35_08b" if str(model["model_id"]).endswith("0.8B") else "qwen35_2b"
        )
        is_random_control = str(setup["method_id"]).startswith("random_control_")
        source_calibration_summary_sha256 = setup.get(
            "validation_summary_sha256",
            setup.get("calibration_summary_sha256"),
        )
        if not _is_digest(source_calibration_summary_sha256):
            raise ValueError(f"setup {index} has invalid calibration summary identity")
        # Validation generations deliberately use an all-zero pre-summary sentinel.
        # Preserve the summary that selected the candidate separately so the plan
        # binds both the source decision and the exact identity written to rows.
        calibration_summary_sha256 = (
            "0" * 64 if split == "validation" else source_calibration_summary_sha256
        )
        stem = f"setup_{index:03d}_{setup_id[:16]}"
        records.append(
            {
                "index": index,
                "setup_id": setup_id,
                "model_tag": model_tag,
                "model_config": str(model["config"]),
                "model_id": setup["model_id"],
                "model_revision": setup["model_revision"],
                "model_config_sha256": setup["model_config_sha256"],
                "method_id": setup["method_id"],
                "track": setup["track"],
                "selected_layer": layer,
                "selected_strength": float(strength),
                "position_schedule": setup["position_schedule"],
                "direction_path": setup["direction_path"],
                "direction_float32_sha256": setup["direction_float32_sha256"],
                "direction_artifact_sha256": setup["direction_artifact_sha256"],
                "construction_config_sha256": setup["construction_config_sha256"],
                "calibration_summary_path": setup.get("calibration_summary_path"),
                "calibration_summary_sha256": calibration_summary_sha256,
                "source_calibration_summary_sha256": source_calibration_summary_sha256,
                "control_source_method_id": setup.get("control_source_method_id"),
                "control_source_strength": setup.get("control_source_strength"),
                "control_source_calibration_summary_sha256": setup.get(
                    "control_source_calibration_summary_sha256"
                ),
                "roles": setup.get("roles", []),
                "strength_roles": setup.get("strength_roles", []),
                "is_random_control": is_random_control,
                "open_required": split == "validation" or not is_random_control,
                "tbsp_required": split == "sealed_test" and not is_random_control,
                "forced_path": _relative(
                    repo_root, output_directory / "forced" / f"{stem}.jsonl"
                ),
                "generation_path": _relative(
                    repo_root, output_directory / "generations" / f"{stem}.jsonl"
                ),
                "scored_path": _relative(
                    repo_root, output_directory / "scored" / f"{stem}.jsonl"
                ),
            }
        )
    return {
        "schema_version": PLAN_SCHEMA,
        "split": split,
        "source_manifest_path": _relative(repo_root, manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "setup_count": len(records),
        "setups": records,
        "setups_sha256": _sha256(records),
    }


def _validate_plan(
    plan: dict[str, Any],
    expected_split: str | None = None,
    repo_root: Path | None = None,
) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("open plan has the wrong schema")
    if expected_split is not None and plan.get("split") != expected_split:
        raise ValueError("open plan split differs from the requested split")
    setups = plan.get("setups")
    if not isinstance(setups, list) or plan.get("setup_count") != len(setups):
        raise ValueError("open plan setup count is invalid")
    if plan.get("setups_sha256") != _sha256(setups):
        raise ValueError("open plan setup hash is invalid")
    for index, setup in enumerate(setups):
        if not isinstance(setup, dict):
            raise TypeError(f"open plan setup {index} is not an object")
        missing = sorted(PLAN_SETUP_IDENTITY_FIELDS - set(setup))
        if missing:
            raise ValueError(f"open plan setup {index} lacks identity fields: {missing}")
    if len({item.get("setup_id") for item in setups}) != len(setups):
        raise ValueError("open plan setup IDs are not unique")
    if repo_root is not None:
        source_digest = plan.get("source_manifest_sha256")
        if not _is_digest(source_digest):
            raise ValueError("open plan source manifest hash is invalid")
        source_path = _resolve_repository_file(
            repo_root,
            plan.get("source_manifest_path"),
            "open plan source manifest path",
        )
        actual_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_digest != actual_digest:
            raise ValueError("open plan source manifest hash differs from the current file")
        expected_status = (
            "locked_before_validation_open"
            if plan.get("split") == "validation"
            else "locked_before_sealed_test"
        )
        if _read_json(source_path).get("status") != expected_status:
            raise ValueError("open plan source manifest status differs from its split")


def verify_canonical_plan(
    repo_root: Path,
    lock_path: Path,
    manifest_path: Path,
    output_directory: Path,
    split: str,
    plan_path: Path,
) -> dict[str, Any]:
    """Require literal equality with a fresh deterministic plan reconstruction."""

    expected = build_plan(repo_root, lock_path, manifest_path, output_directory, split)
    observed_bytes = plan_path.read_bytes()
    expected_bytes = _serialized_json_bytes(expected)
    if observed_bytes != expected_bytes:
        raise ValueError("open plan differs byte-for-byte from canonical regeneration")
    _validate_plan(expected, expected_split=split, repo_root=repo_root)
    return expected


def _row_matches_setup(row: dict[str, Any], setup: dict[str, Any], split: str) -> bool:
    return (
        row.get("split") == split
        and row.get("model_id") == setup["model_id"]
        and row.get("model_revision") == setup["model_revision"]
        and row.get("config_sha256") == setup["model_config_sha256"]
        and row.get("method") == setup["method_id"]
        and row.get("method_id") == setup["method_id"]
        and row.get("setup") == setup["track"]
        and row.get("track") == setup["track"]
        and row.get("layer") == setup["selected_layer"]
        and row.get("position") == setup["position_schedule"]
        and row.get("calibration_magnitude") == setup["selected_strength"]
        and row.get("direction_sha256") == setup["direction_float32_sha256"]
        and row.get("direction_float32_sha256")
        == setup["direction_float32_sha256"]
        and row.get("direction_artifact_sha256")
        == setup["direction_artifact_sha256"]
        and row.get("construction_config_sha256")
        == setup["construction_config_sha256"]
        and row.get("calibration_summary_sha256")
        == setup["calibration_summary_sha256"]
        and (
            setup.get("is_random_control") is not True
            or (
                row.get("control_source_method_id")
                == setup.get("control_source_method_id")
                and row.get("control_source_strength")
                == setup.get("control_source_strength")
                and row.get("control_source_calibration_summary_sha256")
                == setup.get("control_source_calibration_summary_sha256")
            )
        )
    )


def combine_generations(repo_root: Path, plan_path: Path, output_path: Path) -> int:
    plan = _read_json(plan_path)
    _validate_plan(plan, repo_root=repo_root)
    rows: list[dict[str, Any]] = []
    generation_hashes: set[str] = set()
    for setup in plan["setups"]:
        if setup.get("open_required") is not True:
            continue
        source = repo_root / setup["generation_path"]
        setup_rows = _read_jsonl(source)
        if len(setup_rows) != 96:
            raise ValueError(f"{source} has {len(setup_rows)} rows instead of 96")
        if any(not _row_matches_setup(row, setup, plan["split"]) for row in setup_rows):
            raise ValueError(f"{source} contains rows from a different setup")
        for row in setup_rows:
            generation_hash = row.get("generation_sha256")
            if not _is_digest(generation_hash) or generation_hash in generation_hashes:
                raise ValueError(f"{source} has an invalid/duplicate generation hash")
            generation_hashes.add(str(generation_hash))
        rows.extend(setup_rows)
    _write_jsonl(output_path, rows)
    return len(rows)


def partition_scored(repo_root: Path, plan_path: Path, scored_path: Path) -> int:
    plan = _read_json(plan_path)
    _validate_plan(plan, repo_root=repo_root)
    all_rows = _read_jsonl(scored_path)
    open_setups = [setup for setup in plan["setups"] if setup.get("open_required") is True]
    expected_total = 96 * len(open_setups)
    if len(all_rows) != expected_total:
        raise ValueError(
            f"combined scored file has {len(all_rows)} rows instead of {expected_total}"
        )
    assigned: set[str] = set()
    for setup in open_setups:
        matches = [
            row
            for row in all_rows
            if _row_matches_setup(row, setup, plan["split"])
        ]
        if len(matches) != 96:
            raise ValueError(
                f"setup {setup['setup_id']} has {len(matches)} scored rows instead of 96"
            )
        hashes = {str(row.get("generation_sha256")) for row in matches}
        if len(hashes) != 96 or assigned.intersection(hashes):
            raise ValueError("scored rows are duplicated across setup partitions")
        assigned.update(hashes)
        _write_jsonl(repo_root / setup["scored_path"], matches)
    if len(assigned) != len(all_rows):
        raise ValueError("some scored rows were not assigned exactly once")
    return len(assigned)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--repo-root", type=Path, required=True)
    plan_parser.add_argument("--lock", type=Path, required=True)
    plan_parser.add_argument("--manifest", type=Path, required=True)
    plan_parser.add_argument("--output-dir", type=Path, required=True)
    plan_parser.add_argument("--split", choices=("validation", "sealed_test"), required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    verify_plan = subparsers.add_parser("verify-plan")
    verify_plan.add_argument("--repo-root", type=Path, required=True)
    verify_plan.add_argument("--lock", type=Path, required=True)
    verify_plan.add_argument("--manifest", type=Path, required=True)
    verify_plan.add_argument("--output-dir", type=Path, required=True)
    verify_plan.add_argument(
        "--split", choices=("validation", "sealed_test"), required=True
    )
    verify_plan.add_argument("--plan", type=Path, required=True)
    combine = subparsers.add_parser("combine-generations")
    combine.add_argument("--repo-root", type=Path, required=True)
    combine.add_argument("--plan", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    partition = subparsers.add_parser("partition-scored")
    partition.add_argument("--repo-root", type=Path, required=True)
    partition.add_argument("--plan", type=Path, required=True)
    partition.add_argument("--scored", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = build_plan(
            args.repo_root.resolve(),
            args.lock.resolve(),
            args.manifest.resolve(),
            args.output_dir.resolve(),
            args.split,
        )
        _write_json(args.output.resolve(), result)
        print(json.dumps({"status": "planned", "setup_count": result["setup_count"]}))
    elif args.command == "verify-plan":
        result = verify_canonical_plan(
            args.repo_root.resolve(),
            args.lock.resolve(),
            args.manifest.resolve(),
            args.output_dir.resolve(),
            args.split,
            args.plan.resolve(),
        )
        print(json.dumps({"status": "verified", "setup_count": result["setup_count"]}))
    elif args.command == "combine-generations":
        count = combine_generations(
            args.repo_root.resolve(), args.plan.resolve(), args.output.resolve()
        )
        print(json.dumps({"status": "combined", "row_count": count}))
    else:
        count = partition_scored(
            args.repo_root.resolve(), args.plan.resolve(), args.scored.resolve()
        )
        print(json.dumps({"status": "partitioned", "row_count": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
