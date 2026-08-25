"""Fail-closed verification of the complete secondary J-space artifact set.

This verifier never runs a model or computes a J-space result.  It derives the exact
expected record set from the canonical sealed plan, validates every record and cache
binding, and rejects partial, stale, or extra persisted metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from validate_jspace_record import validate_record

PRIMARY_METHODS = ("gradient", "caa", "bipo", "persona_vector")
RECEIPT_SCHEMA = "sp_lense.jspace_completion_receipt.v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"J-space artifact escapes repository root: {path}") from error


def _canonical_digest(values: list[str]) -> str:
    payload = ("\n".join(values) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _serialized_receipt(receipt: dict[str, Any]) -> bytes:
    return (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_serialized_receipt(receipt))
    temporary.replace(path)


def _selected_setups(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if plan.get("schema_version") != "sp_lense.locked_open_plan.v1":
        raise ValueError("J-space completion requires the locked open-plan schema")
    if plan.get("split") != "sealed_test":
        raise ValueError("J-space completion requires the sealed-test plan")
    setups = plan.get("setups")
    if not isinstance(setups, list) or plan.get("setup_count") != len(setups):
        raise ValueError("sealed plan setup coverage is inconsistent")

    selected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for index, setup in enumerate(setups):
        if not isinstance(setup, dict):
            raise TypeError(f"sealed plan setup {index} is not an object")
        if setup.get("method_id") not in PRIMARY_METHODS:
            continue
        setup_id = setup.get("setup_id")
        if not _is_digest(setup_id):
            raise ValueError(f"sealed plan setup {index} has an invalid setup_id")
        key = (
            setup.get("model_id"),
            setup.get("method_id"),
            setup.get("track"),
            setup.get("selected_layer"),
            setup.get("direction_artifact_sha256"),
        )
        if key not in seen:
            seen.add(key)
            selected.append(setup)
    return selected


def _status_is_complete(status_path: Path) -> None:
    status = _read_object(status_path)
    if status.get("schema_version") != 1 or status.get("state") != "complete":
        raise ValueError("J-space status is not an explicit completed state")
    if not isinstance(status.get("detail"), str) or not status["detail"].strip():
        raise ValueError("J-space completed status lacks a nonempty detail")
    if (
        isinstance(status.get("process_id"), bool)
        or not isinstance(status.get("process_id"), int)
        or status["process_id"] < 1
    ):
        raise ValueError("J-space completed status has an invalid process_id")
    if not isinstance(status.get("updated_at_utc"), str) or not status["updated_at_utc"].strip():
        raise ValueError("J-space completed status lacks updated_at_utc")


def verify_completion(
    *,
    repo_root: Path,
    plan_path: Path,
    lock_path: Path,
    status_path: Path,
    records_directory: Path,
    atoms_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    for path, label in (
        (plan_path, "sealed plan"),
        (lock_path, "stage-one lock"),
        (status_path, "J-space status"),
    ):
        try:
            path.resolve().relative_to(repo_root)
        except ValueError as error:
            raise ValueError(f"{label} escapes repository root") from error
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    for path, label in ((records_directory, "records directory"), (atoms_root, "atoms root")):
        try:
            path.resolve().relative_to(repo_root)
        except ValueError as error:
            raise ValueError(f"J-space {label} escapes repository root") from error

    _status_is_complete(status_path)
    plan = _read_object(plan_path)
    lock = _read_object(lock_path)
    selected = _selected_setups(plan)

    try:
        model_settings = lock["evaluation"]["j_space"]["models"]
    except (KeyError, TypeError) as error:
        raise ValueError("stage-one lock lacks J-space model settings") from error
    if not isinstance(model_settings, dict):
        raise TypeError("stage-one J-space model settings must be an object")

    expected_records: list[Path] = []
    expected_metadata: set[Path] = set()
    receipts: list[dict[str, Any]] = []
    for index, setup in enumerate(selected):
        model_id = setup.get("model_id")
        model_tag = setup.get("model_tag")
        layer = setup.get("selected_layer")
        if (
            not isinstance(model_id, str)
            or not isinstance(model_tag, str)
            or not model_tag
            or isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
        ):
            raise ValueError(f"selected J-space setup {index} has invalid model/layer identity")
        try:
            source_layers = model_settings[model_id]["lens"]["source_layers"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"lock lacks lens source layers for {model_id}") from error
        if (
            not isinstance(source_layers, list)
            or not source_layers
            or any(isinstance(item, bool) or not isinstance(item, int) for item in source_layers)
        ):
            raise ValueError(f"lock has invalid lens source layers for {model_id}")

        setup_id = setup["setup_id"]
        record_path = records_directory / f"direction_{index:03d}_{setup_id[:16]}.jsonl"
        expected_records.append(record_path)
        manifest_path: Path | None = None
        if layer in source_layers:
            cache_root = atoms_root / model_tag / f"layer_{layer:02d}"
            manifest_path = cache_root / "atoms_manifest.json"
            expected_metadata.add(manifest_path.resolve())
            expected_metadata.add((cache_root / "token_labels.json").resolve())
        receipts.append(
            validate_record(
                repo_root=repo_root,
                plan_path=plan_path,
                lock_path=lock_path,
                setup_id=setup_id,
                record_path=record_path,
                atoms_manifest_path=manifest_path,
            )
        )

    observed_record_files = (
        sorted((path.resolve() for path in records_directory.iterdir() if path.is_file()), key=str)
        if records_directory.is_dir()
        else []
    )
    expected_record_files = sorted((path.resolve() for path in expected_records), key=str)
    if [str(path) for path in observed_record_files] != [str(path) for path in expected_record_files]:
        raise ValueError(
            "J-space record directory differs from the exact canonical record set: "
            f"expected={[path.name for path in expected_record_files]}, "
            f"observed={[path.name for path in observed_record_files]}"
        )

    observed_metadata = (
        {
            path.resolve()
            for path in atoms_root.rglob("*")
            if path.is_file() and path.name.lower() in {"atoms_manifest.json", "token_labels.json"}
        }
        if atoms_root.is_dir()
        else set()
    )
    if {str(path) for path in observed_metadata} != {str(path) for path in expected_metadata}:
        raise ValueError(
            "J-space atom metadata differs from the exact caches required by the sealed plan"
        )

    record_paths = sorted(_relative(repo_root, path) for path in expected_record_files)
    metadata_paths = sorted(_relative(repo_root, path) for path in expected_metadata)
    artifact_paths = sorted(record_paths + metadata_paths)
    artifacts = [
        {
            "path": relative,
            "sha256": hashlib.sha256((repo_root / relative).read_bytes()).hexdigest(),
            "size_bytes": (repo_root / relative).stat().st_size,
        }
        for relative in artifact_paths
    ]
    status_counts = dict(sorted(Counter(item["jspace_status"] for item in receipts).items()))
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "valid_complete",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "selected_direction_count": len(selected),
        "record_count": len(record_paths),
        "record_status_counts": status_counts,
        "record_paths": record_paths,
        "metadata_paths": metadata_paths,
        "artifact_paths": artifact_paths,
        "artifact_paths_sha256": _canonical_digest(artifact_paths),
        "artifacts": artifacts,
        "explicit_no_primary_direction_skip": len(selected) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--atoms-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        receipt = verify_completion(
            repo_root=args.repo_root,
            plan_path=args.plan,
            lock_path=args.lock,
            status_path=args.status,
            records_directory=args.records_dir,
            atoms_root=args.atoms_root,
        )
        if args.output is not None:
            output = args.output.resolve()
            output.relative_to(args.repo_root.resolve())
            write_receipt(output, receipt)
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
