"""List and read the merged Research4 Game history from this checkout's .local."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import game_run_store

LOCAL = ROOT / ".local"
INFRASTRUCTURE_INDEX = "INFRASTRUCTURE_RUNS.yaml"


def _contained(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in relative:
        raise ValueError("A Game history locator escapes .local")
    resolved = (root / Path(*path.parts)).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("A Game history locator escaped .local")
    return resolved


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _order_time(value):
    if value is None:
        return datetime.max.replace(tzinfo=timezone.utc)
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        raise ValueError("A Game history finish time has no timezone")
    return parsed.astimezone(timezone.utc)


def _infrastructure_rows(local):
    index = local / INFRASTRUCTURE_INDEX
    if not index.exists():
        return []
    rows = yaml.safe_load(index.read_bytes())
    if not isinstance(rows, list):
        raise ValueError("The infrastructure history index has an unexpected format")
    return rows


def list_history(local=LOCAL):
    """Combine the older indexed episodes with the current infrastructure collection."""
    local = Path(local).resolve(strict=True)
    rows = [{
        "id": row["episode_id"],
        "collection": "indexed_game",
        "finish_utc": row.get("finish_utc"),
        "status": row.get("completion_status"),
        "model": row.get("model"),
    } for row in game_run_store.list_runs(local)]
    for row in _infrastructure_rows(local):
        raw = _contained(local, row["raw"])
        rows.append({
            "id": raw.parent.name,
            "collection": "infrastructure",
            "finish_utc": row.get("finished_utc"),
            "status": row["status"],
            "model": row["model_id"],
        })
    if len({(row["collection"], row["id"]) for row in rows}) != len(rows):
        raise ValueError("Game history contains duplicate record identities")
    return sorted(rows, key=lambda row: (
        _order_time(row["finish_utc"]),
        row["collection"],
        row["id"],
    ))


def read_indexed(episode_id, local=LOCAL):
    """Read a relocated historical YAML after checking its original index hash."""
    local = Path(local).resolve(strict=True)
    rows = [row for row in game_run_store.list_runs(local) if row["episode_id"] == episode_id]
    if len(rows) != 1:
        raise ValueError("The historical episode ID is missing or ambiguous")
    # Keep the existing Game API's index, containment and hash checks in force.
    game_run_store.load_conversation(local, episode_id)
    path = Path(rows[0]["conversation"]).resolve(strict=True)
    data = path.read_bytes()
    if not path.is_relative_to(local) or _digest(data) != rows[0]["conversation_sha256"]:
        raise ValueError("The historical conversation changed while reading")
    return data


def read_infrastructure(bundle, local=LOCAL):
    """Read a current YAML after checking its copied raw evidence references."""
    local = Path(local).resolve(strict=True)
    found = []
    for row in _infrastructure_rows(local):
        reference = _contained(local, row["raw"])
        if reference.parent.name == bundle:
            found.append((row, reference))
    if len(found) != 1:
        raise ValueError("The infrastructure bundle is missing or ambiguous")
    row, reference = found[0]
    if reference.name != "REFERENCE.yaml":
        raise ValueError("The infrastructure raw reference has an unexpected name")
    manifest = yaml.safe_load(reference.read_bytes())
    source = _contained(local, manifest["directory"])
    for name, expected in manifest["sha256"].items():
        if not isinstance(name, str) or not name or "\\" in name or Path(name).name != name:
            raise ValueError("An infrastructure evidence filename is invalid")
        copied = reference.parent / name
        if (_digest(copied.read_bytes()) != expected or
                _digest((source / name).read_bytes()) != expected):
            raise ValueError("An infrastructure evidence hash differs")
    conversation = _contained(local, row["conversation"])
    data = conversation.read_bytes()
    view = yaml.safe_load(data)
    if not isinstance(view, list) or not view or "resource_progress" not in view[-1]:
        raise ValueError("The infrastructure conversation is incomplete")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--episode", help="Read one historical episode by its full index ID")
    choice.add_argument("--bundle", help="Read one infrastructure bundle by its folder name")
    args = parser.parse_args()
    if args.episode:
        sys.stdout.buffer.write(read_indexed(args.episode))
    elif args.bundle:
        sys.stdout.buffer.write(read_infrastructure(args.bundle))
    else:
        for item in list_history():
            print(item["finish_utc"] or "unknown", item["collection"], item["status"],
                  item["model"], item["id"])


if __name__ == "__main__":
    main()
