"""Immutable runtime bindings shared by the package controller and workers."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import hashlib
import json
import os


RUNTIME_VERSION = "qwen35-9b-ordinal-continuity-runtime-v1"


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def canonical_sha256(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_receipt(value: Mapping) -> dict:
    result = dict(value)
    result.pop("receipt_sha256", None)
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def verify_receipt(value: Mapping) -> dict:
    result = dict(value)
    embedded = result.pop("receipt_sha256", None)
    if not isinstance(embedded, str) or canonical_sha256(result) != embedded:
        raise PermissionError("Receipt self-hash differs")
    return dict(value)


def atomic_json(path, value: Mapping, *, finalize=False) -> dict:
    destination = Path(path)
    payload = finalize_receipt(value) if finalize else dict(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return payload


def directory_tree(root, *, exclude=()) -> list[dict]:
    base = Path(root).resolve()
    excluded = {Path(item).as_posix() for item in exclude}
    rows = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base).as_posix()
        if relative in excluded:
            continue
        rows.append({"path": relative, "sha256": file_sha256(path),
                     "bytes": path.stat().st_size})
    return rows


def directory_tree_sha256(root, *, exclude=()) -> str:
    return canonical_sha256(directory_tree(root, exclude=exclude))


def verify_package_manifest(package_root, manifest_path, *, expected_tree_sha256=None,
                            expected_source_sha256=None) -> dict:
    root = Path(package_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    declared = manifest.get("files")
    if not isinstance(declared, list) or not declared:
        raise PermissionError("Package manifest has no file inventory")
    checked = []
    for entry in declared:
        relative = entry.get("path")
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts):
            raise PermissionError("Package manifest contains an unsafe path")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise PermissionError(f"Package member is missing: {relative}")
        actual = {"path": relative, "sha256": file_sha256(path),
                  "bytes": path.stat().st_size}
        if actual != entry:
            raise PermissionError(f"Package member differs: {relative}")
        checked.append(actual)
    tree_sha = canonical_sha256(checked)
    if (tree_sha != manifest.get("tree_sha256")
            or expected_tree_sha256 is not None and tree_sha != expected_tree_sha256):
        raise PermissionError("Package tree digest differs")
    # Generated data lives under research4/ as well, so source membership is
    # explicitly declared by the preparation tool rather than inferred from a
    # broad prefix.
    source_paths = set(manifest.get("source_paths", ()))
    if not source_paths:
        raise PermissionError("Package manifest has no explicit source inventory")
    source = canonical_sha256([entry for entry in checked if entry["path"] in source_paths])
    if (source != manifest.get("source_tree_sha256")
            or expected_source_sha256 is not None and source != expected_source_sha256):
        raise PermissionError("Package source digest differs")
    result = dict(manifest)
    result.update({"verified_tree_sha256": tree_sha,
                   "verified_source_tree_sha256": source,
                   "manifest_file_sha256": file_sha256(manifest_file)})
    return result


def cli_sha256(argv: Sequence[str]) -> str:
    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise ValueError("CLI argument vector must contain nonempty strings")
    return canonical_sha256(list(argv))


def verify_tokenizer_receipt(tokenizer_root, receipt_path, *, model_id, revision) -> dict:
    root = Path(tokenizer_root).resolve()
    receipt_file = Path(receipt_path).resolve()
    if root not in receipt_file.parents or not receipt_file.is_file():
        raise PermissionError("Tokenizer receipt is outside the durable tokenizer root")
    receipt = verify_receipt(json.loads(receipt_file.read_text(encoding="utf-8")))
    if (receipt.get("model_id") != model_id or receipt.get("revision") != revision
            or receipt.get("resolved_commit") != revision):
        raise PermissionError("Tokenizer identity/commit differs")
    files = receipt.get("files", [])
    if not isinstance(files, list) or not files:
        raise PermissionError("Tokenizer receipt has no durable file inventory")
    paths = [entry.get("path") for entry in files]
    if (any(not isinstance(relative, str) or not relative
            or Path(relative).is_absolute() or ".." in Path(relative).parts
            for relative in paths)
            or len(paths) != len(set(paths)) or paths != sorted(paths)):
        raise PermissionError("Tokenizer receipt file inventory is unsafe or noncanonical")
    for entry in files:
        path = (root / entry["path"]).resolve()
        if root not in path.parents or not path.is_file():
            raise PermissionError("Tokenizer file is missing")
        if (file_sha256(path), path.stat().st_size) != (
                entry.get("sha256"), entry.get("bytes")):
            raise PermissionError("Tokenizer file bytes differ")
    if canonical_sha256(files) != receipt.get("tree_sha256"):
        raise PermissionError("Tokenizer tree digest differs")
    receipt_relative = receipt_file.relative_to(root).as_posix()
    actual_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
        and path.relative_to(root).as_posix() != receipt_relative)
    if actual_paths != paths:
        raise PermissionError("Durable tokenizer directory has undeclared or missing files")
    return receipt
