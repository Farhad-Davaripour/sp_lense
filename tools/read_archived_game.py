"""Read the original Game index after moving its Windows worktree."""

import argparse
import hashlib
from pathlib import Path, PureWindowsPath
import sys

import yaml


REPOS = Path(__file__).resolve().parents[2]
OLD_STORE = REPOS / "sp_lense4_worktrees" / "company-world-v4" / ".local"
ARCHIVE_STORE = REPOS / "superseded" / "sp_lense4_worktrees" / "company-world-v4" / ".local"


def read_conversation(entry, store, recorded_store):
    relative = PureWindowsPath(entry["conversation"]).relative_to(PureWindowsPath(recorded_store))
    if ".." in relative.parts:
        raise ValueError("Archived conversation path contains parent traversal")
    path = store.joinpath(*relative.parts).resolve(strict=True)
    if not path.is_relative_to(store):
        raise ValueError("Archived conversation escaped its store")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["conversation_sha256"]:
        raise ValueError("Archived conversation does not match its recorded digest")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ARCHIVE_STORE)
    parser.add_argument("--recorded-store", default=str(OLD_STORE))
    parser.add_argument("--episode", help="Print a verified conversation by full episode ID")
    args = parser.parse_args()
    store = args.store.resolve(strict=True)
    index = yaml.safe_load((store / "INDEX.yaml").read_bytes())
    if index.get("format") != "research4.game-index.v1":
        parser.error("Unrecognized historical Game index")
    entries = index["entries"]
    if args.episode is None:
        for entry in sorted(entries, key=lambda row: row.get("finish_utc") or ""):
            print(entry.get("finish_edmonton") or "unknown", entry["episode_id"],
                  entry.get("completion_status"), entry.get("model"))
        return
    matches = [entry for entry in entries if entry["episode_id"] == args.episode]
    if len(matches) != 1:
        parser.error("Episode ID is missing or ambiguous")
    sys.stdout.buffer.write(read_conversation(matches[0], store, args.recorded_store))


if __name__ == "__main__":
    main()
