"""Kaggle entrypoint for one immutable continuity-training partition."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile


PACKAGE_VERSION = "qwen35-9b-ordinal-continuity-partition-package-v8"
EXPECTED_PACKAGE_SHA256 = "PACKAGE_SHA256_PLACEHOLDER"
EXPECTED_MANIFEST_SHA256 = "PACKAGE_MANIFEST_SHA256_PLACEHOLDER"
EXPECTED_TREE_SHA256 = "PACKAGE_TREE_SHA256_PLACEHOLDER"
EXPECTED_SOURCE_SHA256 = "SOURCE_TREE_SHA256_PLACEHOLDER"
MAX_EXTRACTED_BYTES = 512 * 1024 * 1024


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_archives(input_root):
    result = []
    for path in Path(input_root).rglob("*.zip"):
        try:
            with zipfile.ZipFile(path) as archive:
                if "PACKAGE_MANIFEST.json" not in archive.namelist():
                    continue
                manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
                if manifest.get("version") == PACKAGE_VERSION:
                    result.append(path.resolve())
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError):
            continue
    return result


def _safe_extract(archive_path, destination):
    destination = Path(destination).resolve()
    total = 0
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise PermissionError("Staged package CRC failed")
        seen = set()
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if (info.is_dir() or member.is_absolute() or ".." in member.parts
                    or member.as_posix() in seen
                    or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise PermissionError("Staged package has an unsafe member")
            seen.add(member.as_posix())
            total += info.file_size
            if total > MAX_EXTRACTED_BYTES:
                raise PermissionError("Staged package exceeds extraction cap")
        for info in archive.infolist():
            target = (destination / PurePosixPath(info.filename)).resolve()
            if not target.is_relative_to(destination):
                raise PermissionError("Staged package escapes destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as sink:
                shutil.copyfileobj(source, sink, 8 * 1024 * 1024)


def stage_bound_package(*, input_root="/kaggle/input",
                        working_root="/kaggle/working",
                        expected_package_sha256=EXPECTED_PACKAGE_SHA256,
                        expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
                        expected_tree_sha256=EXPECTED_TREE_SHA256,
                        expected_source_sha256=EXPECTED_SOURCE_SHA256):
    expected = (expected_package_sha256, expected_manifest_sha256,
                expected_tree_sha256, expected_source_sha256)
    if any(not isinstance(value, str) or len(value) != 64 for value in expected):
        raise PermissionError("Notebook package bindings were not rendered")
    matches = _candidate_archives(input_root)
    if len(matches) != 1:
        raise PermissionError("Exactly one ordinal partition package must be staged")
    source_archive = matches[0]
    if _sha(source_archive) != expected_package_sha256:
        raise PermissionError("Staged package archive differs")
    stage = Path(working_root).resolve() / (
        "research4_ordinal_partition_" + expected_package_sha256[:16])
    stage.mkdir(parents=True, exist_ok=False)
    copied = stage / "source_package.zip"
    with source_archive.open("rb") as source, copied.open("xb") as sink:
        shutil.copyfileobj(source, sink, 8 * 1024 * 1024)
    if _sha(copied) != expected_package_sha256:
        raise PermissionError("Working package copy differs")
    package_root = stage / "package"
    package_root.mkdir()
    _safe_extract(copied, package_root)
    manifest_path = package_root / "PACKAGE_MANIFEST.json"
    if _sha(manifest_path) != expected_manifest_sha256:
        raise PermissionError("Extracted manifest differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("version") != PACKAGE_VERSION
            or manifest.get("tree_sha256") != expected_tree_sha256
            or manifest.get("source_tree_sha256") != expected_source_sha256
            or manifest.get("package_sha256") not in (None, expected_package_sha256)):
        raise PermissionError("Manifest bindings differ")
    sys.path.insert(0, str(package_root))
    from research4.src.qwen9b_ordinal_continuity_runtime import verify_package_manifest
    verify_package_manifest(
        package_root, manifest_path,
        expected_tree_sha256=expected_tree_sha256,
        expected_source_sha256=expected_source_sha256)
    return package_root, manifest


def main(**bindings):
    root, manifest = stage_bound_package(**bindings)
    os.chdir(root)
    for key in ("offline_install_argv", "import_preflight_argv", "launch_argv"):
        argv = manifest.get(key)
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise PermissionError(f"Package manifest lacks exact {key}")
        subprocess.run(argv, cwd=root, check=True, shell=False)


if __name__ == "__main__":
    main()
