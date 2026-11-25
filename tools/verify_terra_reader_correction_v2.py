"""Verify the versioned Terra readable-YAML correction without private evidence.

The default check works in a shallow checkout. The optional --verify-history
also uses the named prior Git commit to prove that only opaque provider
reasoning blocks were removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "catalog/results/2026-11-25--terra-readable-correction-v2.json"
SCHEMA = "research4.terra-readable-correction.v2"
PRIOR_COMMIT = "b4a217ba41de6670b125519d172ab46c16d475ef"
COHORTS = (
    "2026-11-20_19Z--infrastructure-preservation-v1",
    "2026-11-23_01Z--visible-state-extension-v1",
)
HASH = re.compile(r"[0-9a-f]{64}\Z")
PATH = re.compile(
    r"results/(?:2026-11-20_19Z--infrastructure-preservation-v1|"
    r"2026-11-23_01Z--visible-state-extension-v1)/"
    r"conversations/terra/[A-Za-z0-9_.-]+\.yaml\Z"
)
OPAQUE = re.compile(r"gAAAA[A-Za-z0-9_-]{100,}={0,2}\Z")
README_ROW = re.compile(r"\[YAML\]\(([^)]+\.yaml)\)\s*\|\s*\x60([0-9a-f]{64})\x60")
FILE_FIELDS = {
    "path",
    "old_sha256",
    "new_sha256",
    "old_item_count",
    "new_item_count",
    "omitted_opaque_reasoning_count",
    "nonopaque_items_sha256",
}


class CorrectionError(ValueError):
    """The Terra reader correction differs from its published manifest."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise CorrectionError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parsed_digest(items: list[dict]) -> str:
    raw = json.dumps(
        items, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha(raw)


def parse_items(data: bytes, path: str) -> list[dict]:
    try:
        items = yaml.safe_load(data)
    except yaml.YAMLError as exc:
        raise CorrectionError(f"Invalid corrected YAML: {path}") from exc
    need(
        isinstance(items, list) and bool(items)
        and all(isinstance(item, dict) and len(item) == 1 for item in items),
        f"Invalid readable conversation structure: {path}",
    )
    return items


def is_opaque_item(item: dict) -> bool:
    value = item.get("reasoning_summary")
    return (
        set(item) == {"reasoning_summary"}
        and isinstance(value, str)
        and OPAQUE.fullmatch(value) is not None
    )


def active_paths() -> set[str]:
    found = set()
    for cohort in COHORTS:
        folder = ROOT / "results" / cohort / "conversations" / "terra"
        need(folder.is_dir() and not folder.is_symlink(), f"Missing Terra reader folder: {cohort}")
        for path in folder.glob("*.yaml"):
            need(path.is_file() and not path.is_symlink(), f"Unsafe Terra reader: {path}")
            found.add(path.relative_to(ROOT).as_posix())
    need(len(found) == 15, f"Expected 15 active Terra reader files, found {len(found)}")
    return found


def check_readme(path: Path, expected_sha: str) -> None:
    cohort = path.parents[2]
    readme = cohort / "README.md"
    need(readme.is_file(), f"Missing cohort README for {path}")
    relative = path.relative_to(cohort).as_posix()
    matches = [
        digest for linked, digest in README_ROW.findall(readme.read_text(encoding="utf-8"))
        if linked == relative
    ]
    need(matches == [expected_sha], f"Cohort README hash differs: {relative}")


def verify_public() -> tuple[dict, list[dict]]:
    need(MANIFEST.is_file() and not MANIFEST.is_symlink(), "Missing versioned Terra correction")
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise CorrectionError("Invalid Terra correction manifest") from exc
    need(
        isinstance(manifest, dict)
        and set(manifest) == {
            "schema", "prior_commit", "file_count", "omitted_opaque_reasoning_count", "files"
        }
        and manifest["schema"] == SCHEMA
        and manifest["prior_commit"] == PRIOR_COMMIT
        and manifest["file_count"] == 15
        and manifest["omitted_opaque_reasoning_count"] == 84
        and isinstance(manifest["files"], list)
        and len(manifest["files"]) == 15,
        "Terra correction identity or total differs",
    )
    paths = active_paths()
    seen = set()
    omitted = 0
    for row in manifest["files"]:
        need(isinstance(row, dict) and set(row) == FILE_FIELDS, "Malformed Terra correction row")
        relative = row["path"]
        need(
            isinstance(relative, str) and PATH.fullmatch(relative) is not None
            and relative in paths and relative not in seen,
            f"Unexpected or duplicate Terra reader: {relative}",
        )
        seen.add(relative)
        need(
            all(isinstance(row[field], str) and HASH.fullmatch(row[field]) is not None
                for field in ("old_sha256", "new_sha256", "nonopaque_items_sha256"))
            and row["old_sha256"] != row["new_sha256"]
            and all(type(row[field]) is int and row[field] > 0
                    for field in ("old_item_count", "new_item_count", "omitted_opaque_reasoning_count"))
            and row["old_item_count"] - row["new_item_count"]
            == row["omitted_opaque_reasoning_count"],
            f"Invalid Terra correction hashes or counts: {relative}",
        )
        path = ROOT / relative
        need(path.is_file() and not path.is_symlink(), f"Missing corrected Terra file: {relative}")
        data = path.read_bytes()
        need(sha(data) == row["new_sha256"], f"Corrected Terra byte hash differs: {relative}")
        items = parse_items(data, relative)
        need(len(items) == row["new_item_count"], f"Terra item count differs: {relative}")
        need(
            not any("reasoning_summary" in item for item in items),
            f"Opaque reasoning field remains in Terra reader: {relative}",
        )
        need(
            parsed_digest(items) == row["nonopaque_items_sha256"],
            f"Terra nonopaque item digest differs: {relative}",
        )
        check_readme(path, row["new_sha256"])
        omitted += row["omitted_opaque_reasoning_count"]
    need(seen == paths and omitted == 84, "Terra correction file set or omission total differs")
    return manifest, manifest["files"]


def strip_exact_opaque_blocks(old: bytes, path: str) -> tuple[bytes, int]:
    """Remove only two-line opaque reasoning YAML items, preserving all other bytes."""
    lines = old.splitlines(keepends=True)
    kept = []
    removed = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        if line == b"- reasoning_summary: |-\n":
            need(index + 1 < len(lines), f"Truncated opaque field: {path}")
            value_line = lines[index + 1]
            need(
                value_line.startswith(b"    ") and value_line.endswith(b"\n"),
                f"Unexpected reasoning block formatting: {path}",
            )
            try:
                value = value_line[4:-1].decode("ascii")
            except UnicodeDecodeError as exc:
                raise CorrectionError(f"Non-ASCII opaque payload: {path}") from exc
            need(OPAQUE.fullmatch(value) is not None, f"Readable reasoning would be removed: {path}")
            removed += 1
            index += 2
            continue
        kept.append(line)
        index += 1
    return b"".join(kept), removed


def verify_history(rows: list[dict]) -> None:
    for row in rows:
        relative = row["path"]
        completed = subprocess.run(
            ["git", "show", f"{PRIOR_COMMIT}:{relative}"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        need(completed.returncode == 0, f"Prior commit unavailable: {relative}")
        old = completed.stdout
        need(sha(old) == row["old_sha256"], f"Historical Terra byte hash differs: {relative}")
        old_items = parse_items(old, relative)
        need(len(old_items) == row["old_item_count"], f"Historical Terra item count differs: {relative}")
        opaque_count = sum(is_opaque_item(item) for item in old_items)
        need(
            opaque_count == row["omitted_opaque_reasoning_count"]
            and not any("reasoning_summary" in item and not is_opaque_item(item)
                        for item in old_items),
            f"Historical Terra reasoning classification differs: {relative}",
        )
        nonopaque = [item for item in old_items if not is_opaque_item(item)]
        need(
            parsed_digest(nonopaque) == row["nonopaque_items_sha256"],
            f"Historical nonopaque item digest differs: {relative}",
        )
        stripped, removed = strip_exact_opaque_blocks(old, relative)
        need(
            removed == opaque_count and stripped == (ROOT / relative).read_bytes(),
            f"Correction changed bytes beyond opaque fields: {relative}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-history", action="store_true",
        help="also compare with the named prior commit, which must exist locally",
    )
    args = parser.parse_args()
    _, rows = verify_public()
    if args.verify_history:
        verify_history(rows)
    print(json.dumps({
        "status": "pass",
        "mode": "history" if args.verify_history else "public",
        "files": len(rows),
        "omitted_opaque_reasoning": sum(row["omitted_opaque_reasoning_count"] for row in rows),
        "prior_commit": PRIOR_COMMIT,
    }))


if __name__ == "__main__":
    main()
