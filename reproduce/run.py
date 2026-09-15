"""Stable, repository-relative reproduction entry point; never launches model inference."""

import argparse
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
        "command", choices=["verify", "replay", "refit", "tune", "figures", "paper", "audit"]
    )
    args = p.parse_args()
    if args.command == "verify":
        count = verify_manifest(HERE / "artifacts")
        print(f"PASS: {count} immutable reproduction artifacts verified")
        return
    commands = {
        "replay": [HERE / "replay.py"],
        "refit": [HERE / "replay.py", "--refit"],
        "tune": [HERE / "replay.py", "--tune"],
        "figures": [ROOT / "paper/build_figures.py"],
        "paper": [ROOT / "paper/build_paper.py"],
        "audit": [ROOT / "paper/audit_claims.py"],
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
