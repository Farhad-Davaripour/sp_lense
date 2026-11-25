"""Prepare or apply the explicit Terra reader correction from a prior commit.

Only opaque encrypted provider reasoning fields are removed. The full original
reader bytes remain retrievable at PRIOR_COMMIT, and the v2 manifest records
each old/new byte hash and the digest of every unchanged parsed YAML item.
This command never reads or edits ignored raw Game evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.publish_readable_results import (  # noqa: E402
    _is_opaque_terra_reasoning, _strip_opaque_terra_reader,
)

PRIOR_COMMIT = "b4a217ba41de6670b125519d172ab46c16d475ef"
MANIFEST = ROOT / "catalog/results/2026-11-25--terra-readable-correction-v2.json"
FOLDERS = (
    ("2026-11-20_19Z--infrastructure-preservation-v1", 10),
    ("2026-11-23_01Z--visible-state-extension-v1", 5),
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest_items(items: list[dict]) -> str:
    encoded = json.dumps(items, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha(encoded)


def old_bytes(relative: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{PRIOR_COMMIT}:{relative.as_posix()}"],
        cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout


def corrected_file(relative: Path) -> tuple[bytes, bytes, dict]:
    path = ROOT / relative
    current = path.read_bytes()
    old = old_bytes(relative)
    if current != old:
        raise RuntimeError(f"Published v1 reader differs from the prior commit: {relative}")
    original = yaml.safe_load(old)
    if not isinstance(original, list) or not original:
        raise RuntimeError(f"Reader is not a YAML list: {relative}")
    expected = []
    for item in original:
        if not isinstance(item, dict) or len(item) != 1:
            raise RuntimeError(f"Unexpected reader item: {relative}")
        if "reasoning_summary" in item:
            if not _is_opaque_terra_reasoning(item["reasoning_summary"]):
                raise RuntimeError(f"Readable Terra reasoning needs a separate review: {relative}")
            continue
        expected.append(item)
    new, omitted = _strip_opaque_terra_reader(old)
    if omitted != len(original) - len(expected) or yaml.safe_load(new) != expected:
        raise RuntimeError(f"Correction altered a nonopaque YAML item: {relative}")
    return old, new, {
        "path": relative.as_posix(),
        "old_sha256": sha(old),
        "new_sha256": sha(new),
        "old_item_count": len(original),
        "new_item_count": len(expected),
        "omitted_opaque_reasoning_count": omitted,
        "nonopaque_items_sha256": digest_items(expected),
    }


def readme_update(folder: str, rows: list[dict]) -> tuple[Path, bytes, bytes]:
    path = ROOT / "results" / folder / "README.md"
    old = path.read_bytes()
    lines = old.decode("utf-8").splitlines(keepends=True)
    for row in rows:
        name = Path(row["path"]).name
        marker = f"(conversations/terra/{name})"
        matches = [index for index, line in enumerate(lines) if marker in line]
        if len(matches) != 1 or row["old_sha256"] not in lines[matches[0]]:
            raise RuntimeError(f"README does not bind one old Terra YAML hash: {name}")
        lines[matches[0]] = lines[matches[0]].replace(row["old_sha256"], row["new_sha256"])
    new = "".join(lines).encode("utf-8")
    if new == old:
        raise RuntimeError(f"README did not change: {path}")
    return path, old, new


def plan() -> tuple[dict[Path, tuple[bytes, bytes]], bytes]:
    changes: dict[Path, tuple[bytes, bytes]] = {}
    rows = []
    for folder, count in FOLDERS:
        paths = sorted((ROOT / "results" / folder / "conversations" / "terra").glob("*.yaml"))
        if len(paths) != count:
            raise RuntimeError(f"Expected {count} Terra readers in {folder}, found {len(paths)}")
        cohort_rows = []
        for path in paths:
            relative = path.relative_to(ROOT)
            old, new, row = corrected_file(relative)
            changes[path] = old, new
            rows.append(row)
            cohort_rows.append(row)
        readme, old_readme, new_readme = readme_update(folder, cohort_rows)
        changes[readme] = old_readme, new_readme
    if len(rows) != 15 or sum(row["omitted_opaque_reasoning_count"] for row in rows) != 84:
        raise RuntimeError("Expected exactly 15 Terra readers and 84 omitted opaque fields")
    manifest = {
        "schema": "research4.terra-readable-correction.v2",
        "prior_commit": PRIOR_COMMIT,
        "file_count": 15,
        "omitted_opaque_reasoning_count": 84,
        "files": rows,
    }
    rendered = (json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                           indent=2, allow_nan=False) + "\n").encode("utf-8")
    return changes, rendered


def atomic_write(path: Path, raw: bytes) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    changes, manifest = plan()
    summary = {
        "files": 15,
        "omitted_opaque_reasoning": 84,
        "cohort_readmes": 2,
        "manifest_sha256": sha(manifest),
        "written": bool(args.apply),
    }
    if args.preview:
        print(json.dumps(summary, sort_keys=True))
        return
    if MANIFEST.exists():
        raise RuntimeError("The versioned correction manifest already exists")
    written: list[Path] = []
    try:
        for path, (_old, new) in changes.items():
            atomic_write(path, new)
            written.append(path)
        atomic_write(MANIFEST, manifest)
        if any(path.read_bytes() != new for path, (_old, new) in changes.items()):
            raise RuntimeError("A corrected reader differs after writing")
    except BaseException:
        for path in reversed(written):
            atomic_write(path, changes[path][0])
        MANIFEST.unlink(missing_ok=True)
        raise
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
