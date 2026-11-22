"""Copy the archived indexed Game episodes into this checkout's ignored .local."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path, PureWindowsPath
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import game_run_store as store


def recorded_root(index):
    entry = index["entries"][0]
    parts = PureWindowsPath(entry["conversation"]).parts
    positions = [i for i, value in enumerate(parts) if value.casefold() == ".local"]
    if len(positions) != 1:
        raise ValueError("The historical conversation has no unique .local root")
    return PureWindowsPath(*parts[: positions[0] + 1])


def relative(value, former):
    path = PureWindowsPath(value)
    result = path.relative_to(former)
    if not result.parts or any(part in {".", ".."} for part in result.parts):
        raise ValueError("An indexed historical path escapes its recorded root")
    return Path(*result.parts)


def hash_file(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_copy_file(source, target):
    source_name = store._filesystem_path(source)
    target_name = store._filesystem_path(target)
    expected = hash_file(source_name)
    os.makedirs(store._filesystem_path(target.parent), exist_ok=True)
    if not target.exists():
        shutil.copyfile(source_name, target_name)
    if hash_file(target_name) != expected or hash_file(source_name) != expected:
        raise ValueError(f"Historical evidence changed during copy: {source.name}")
    return 1


def safe_copy_tree(source, target):
    if not source.is_dir() or source.is_symlink() or os.path.isjunction(source):
        raise ValueError(f"Historical raw package is unavailable: {source}")
    copied = 0
    source_name = store._filesystem_path(source)
    for directory, directories, files in os.walk(source_name, followlinks=False):
        if any(os.path.islink(os.path.join(directory, name))
               or os.path.isjunction(os.path.join(directory, name)) for name in directories):
            raise ValueError("Historical raw package contains a linked directory")
        subdirectory = Path(os.path.relpath(directory, source_name))
        destination = target if str(subdirectory) == "." else target / subdirectory
        for name in files:
            original = os.path.join(directory, name)
            if os.path.islink(original):
                raise ValueError("Historical raw package contains a linked file")
            copied += safe_copy_file(Path(original), destination / name)
    return copied


def relocate(index, former, archive, local):
    result = copy.deepcopy(index)
    for entry in result["entries"]:
        for key in ("conversation", "raw_dir", "supporting_files_manifest"):
            if entry.get(key):
                entry[key] = str(local / relative(entry[key], former))
        if entry.get("source_dir"):
            entry["source_dir"] = str(archive / relative(entry["source_dir"], former))
        entry["source_aliases"] = [
            str(archive / relative(source, former)) for source in entry.get("source_aliases", [])
        ]
    result["historical_index_sha256"] = hashlib.sha256(
        (archive / "INDEX.yaml").read_bytes()
    ).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Archived Game .local with original INDEX.yaml")
    parser.add_argument("--local", default=ROOT / ".local", type=Path, help="Merged Research4 .local")
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    local = args.local.resolve(strict=True)
    if archive == local or archive.is_relative_to(local) or local.is_relative_to(archive):
        parser.error("The source and destination stores must be separate")
    old_bytes = (archive / "INDEX.yaml").read_bytes()
    index = yaml.safe_load(old_bytes)
    if (index.get("format") != store.FORMAT or len(index["entries"]) != index["counts"]["episodes"]
            or len({row["episode_id"] for row in index["entries"]}) != len(index["entries"])):
        raise ValueError("The historical index format, count or IDs differ")
    former = recorded_root(index)
    output = relocate(index, former, archive, local)
    for number, original in enumerate(index["entries"], 1):
        path = relative(original["raw_dir"], former)
        safe_copy_tree(archive / path, local / path)
        conversation = relative(original["conversation"], former)
        safe_copy_file(archive / conversation, local / conversation)
        destination = local / conversation
        if store._sha(destination) != original["conversation_sha256"]:
            raise ValueError("The indexed conversation SHA-256 differs")
        with open(store._filesystem_path(destination), "rb") as stream:
            yaml.safe_load(stream)
        if number % 40 == 0 or number == len(index["entries"]):
            print(f"Verified {number}/{len(index['entries'])} indexed episodes", flush=True)
    with store._index_lock(local):
        published = local / "INDEX.yaml"
        content = store._yaml_bytes(output)
        if published.exists() and published.read_bytes() != content:
            raise ValueError("Existing Research4 index differs")
        if not published.exists():
            store._atomic_write(published, content)
    for entry in (output["entries"][0], output["entries"][-1]):
        store.load_conversation(local, entry["episode_id"])
    if (archive / "INDEX.yaml").read_bytes() != old_bytes:
        raise ValueError("The original archived index changed")
    print(f"Imported and reverified {len(output['entries'])} historical Game episodes.")


if __name__ == "__main__":
    main()
