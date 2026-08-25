"""Validate one persisted J-space record against its canonical sealed setup."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from validate_jspace_cache import validate_cache

APPROVED_STATUSES = {
    "complete",
    "not_run_resource_limited",
    "not_run_lens_layer_unavailable",
}
LOCKED_LENS_FIELDS = (
    "repository",
    "filename",
    "revision",
    "file_sha256",
    "file_size_bytes",
    "n_prompts",
    "source_layers",
    "fitted_model_id",
    "fitted_model_revision",
    "transfer_status",
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _read_single_jsonl(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise ValueError("J-space record must contain exactly one nonblank JSONL row")
    value = json.loads(lines[0])
    if not isinstance(value, dict):
        raise TypeError("J-space record row must be an object")
    return value


def _repository_file(repo_root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} must be a repository-relative path")
    root = repo_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository root") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def validate_record(
    *,
    repo_root: Path,
    plan_path: Path,
    lock_path: Path,
    setup_id: str,
    record_path: Path,
    atoms_manifest_path: Path | None,
) -> dict[str, Any]:
    plan = _read_object(plan_path)
    lock = _read_object(lock_path)
    matches = [item for item in plan.get("setups", []) if item.get("setup_id") == setup_id]
    if len(matches) != 1:
        raise ValueError("canonical plan does not contain exactly one requested setup")
    setup = matches[0]
    record = _read_single_jsonl(record_path)
    if record.get("schema_version") != "sp_lense.jspace_record.v2":
        raise ValueError("J-space record has the wrong schema")
    status = record.get("status")
    if status not in APPROVED_STATUSES:
        raise ValueError("J-space record has an unapproved status")

    expected = {
        "model_id": setup["model_id"],
        "model_revision": setup["model_revision"],
        "model_config_sha256": setup["model_config_sha256"],
        "method": setup["method_id"],
        "setup": setup["track"],
        "layer": setup["selected_layer"],
        "direction_float32_sha256": setup["direction_float32_sha256"],
        "direction_artifact_sha256": setup["direction_artifact_sha256"],
        "non_gating": True,
        "used_for_primary_ranking": False,
    }
    mismatches = {
        field: (value, record.get(field))
        for field, value in expected.items()
        if record.get(field) != value
    }
    if mismatches:
        raise ValueError(f"J-space record/setup identity mismatch: {mismatches}")

    direction_path = _repository_file(
        repo_root, setup.get("direction_path"), "sealed direction artifact"
    )
    direction_file_sha256 = hashlib.sha256(direction_path.read_bytes()).hexdigest()
    if record.get("direction_file_sha256") != direction_file_sha256:
        raise ValueError("J-space record direction-file hash is stale")

    try:
        model_settings = lock["evaluation"]["j_space"]["models"][setup["model_id"]]
        lens = model_settings["lens"]
    except (KeyError, TypeError) as error:
        raise ValueError("stage-one lock lacks J-space settings for the setup model") from error
    source_layers = lens.get("source_layers")
    if not isinstance(source_layers, list) or not source_layers:
        raise ValueError("locked J-space lens source layers are invalid")
    expected_lens = {
        "file_sha256": lens.get("file_sha256"),
        "revision": lens.get("revision"),
        "source_layers": source_layers,
    }
    if record.get("lens_provenance") != expected_lens:
        raise ValueError("J-space record lens provenance differs from the lock")

    layer = setup["selected_layer"]
    if layer not in source_layers:
        if atoms_manifest_path is not None:
            raise ValueError("layer-unavailable record cannot use an atom manifest")
        if status != "not_run_lens_layer_unavailable":
            raise ValueError("unavailable lens layer requires an explicit not-run record")
        if record.get("available_source_layers") != source_layers:
            raise ValueError("layer-unavailable record has stale source-layer identity")
        if any(
            field in record
            for field in (
                "atoms_manifest_sha256",
                "atoms_file_sha256",
                "atoms_float32_sha256",
            )
        ):
            raise ValueError("layer-unavailable record claims atom artifacts")
    else:
        if atoms_manifest_path is None:
            raise ValueError("available lens layer requires an atom manifest")
        atoms_manifest_path = atoms_manifest_path.resolve()
        try:
            atoms_manifest_path.relative_to(repo_root.resolve())
        except ValueError as error:
            raise ValueError("J-space atom manifest escapes the repository root") from error
        cache_receipt = validate_cache(atoms_manifest_path)
        manifest = _read_object(atoms_manifest_path)
        if cache_receipt["model_id"] != setup["model_id"] or cache_receipt["layer"] != layer:
            raise ValueError("J-space cache model/layer differs from the sealed setup")
        expected_cache_model = {
            "id": setup["model_id"],
            "revision": setup["model_revision"],
            "config_sha256": setup["model_config_sha256"],
        }
        if manifest.get("model") != expected_cache_model:
            raise ValueError("J-space cache model provenance differs from the sealed setup")
        manifest_lens = manifest.get("lens")
        if not isinstance(manifest_lens, dict):
            raise ValueError("J-space cache lacks lens provenance")
        lens_mismatches = {
            field: (lens.get(field), manifest_lens.get(field))
            for field in LOCKED_LENS_FIELDS
            if lens.get(field) != manifest_lens.get(field)
        }
        if lens_mismatches:
            raise ValueError(f"J-space cache lens provenance mismatch: {lens_mismatches}")
        try:
            reference_commit = manifest["construction"]["reference_repository_commit"]
            locked_reference_commit = lock["sources"]["j_space"]["commit"]
        except (KeyError, TypeError) as error:
            raise ValueError("J-space cache or lock lacks reference-code provenance") from error
        if reference_commit != locked_reference_commit:
            raise ValueError("J-space cache uses the wrong reference implementation commit")
        atom_expected = {
            "atoms_manifest_sha256": cache_receipt["manifest_sha256"],
            "atoms_file_sha256": manifest["atoms"]["file_sha256"],
            "atoms_float32_sha256": manifest["atoms"]["float32_sha256"],
        }
        atom_mismatches = {
            field: (value, record.get(field))
            for field, value in atom_expected.items()
            if record.get(field) != value
        }
        if atom_mismatches:
            raise ValueError(f"J-space record/cache identity mismatch: {atom_mismatches}")
        if status == "not_run_lens_layer_unavailable":
            raise ValueError("available lens layer cannot claim layer-unavailable status")

    return {
        "status": "valid",
        "setup_id": setup_id,
        "record_path": str(record_path.resolve()),
        "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
        "direction_file_sha256": direction_file_sha256,
        "jspace_status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--setup-id", required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--atoms-manifest", type=Path)
    args = parser.parse_args()
    try:
        receipt = validate_record(
            repo_root=args.repo_root,
            plan_path=args.plan,
            lock_path=args.lock,
            setup_id=args.setup_id,
            record_path=args.record,
            atoms_manifest_path=args.atoms_manifest,
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
