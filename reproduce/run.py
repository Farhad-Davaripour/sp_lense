"""Stable, repository-relative reproduction entry point; never launches model inference."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from .utils import verify_manifest
except ImportError:
    from utils import verify_manifest
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=[
            "verify",
            "replay",
            "refit",
            "tune",
            "tables",
            "paper",
            "audit",
            "gated",
            "policy",
        ],
    )
    args = p.parse_args()
    if args.command == "verify":
        count = verify_manifest(
            HERE / "artifacts", required=json.loads((HERE / "inventory.json").read_text())
        )
        print(f"PASS: {count} immutable reproduction artifacts verified")
        return
    commands = {
        "policy": [HERE / "audit_steering_policy.py"],
        "gated": [HERE / "audit_gated_chat.py"],
        "replay": [HERE / "replay.py"],
        "refit": [HERE / "replay.py", "--refit"],
        "tune": [HERE / "replay.py", "--tune"],
        "tables": [ROOT / "paper/build_tables.py"],
        "paper": [ROOT / "paper/verify_publication.py"],
        "audit": [ROOT / "paper/audit_results.py"],
    }
    subprocess.run(
        [
            sys.executable,
            *(["-O"] if sys.flags.optimize else []),
            *map(str, commands[args.command]),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
