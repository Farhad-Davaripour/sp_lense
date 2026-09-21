"""Replay, refit, and audit the released study; new GPU inference is separate."""

import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="Study checkout containing reproduce/, data/, and study/")
    parser.add_argument(
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
            "policy-search",
            "baseline-payload",
            "policy-payload",
            "package",
        ],
    )
    args = parser.parse_args()
    if args.repo:
        os.environ["SP_LENSE_REPO"] = args.repo
    try:
        from .paths import ARTIFACTS, INPUTS, ROOT
    except ValueError as exc:
        parser.error(str(exc))
    from .utils import read_json, verify_manifest

    if args.command == "verify":
        count = verify_manifest(ARTIFACTS, required=read_json(INPUTS / "inventory.json"))
        print(f"PASS: {count} immutable reproduction artifacts verified")
        return
    if args.command == "policy-search":
        from ..steering.policy import search

        print(json.dumps(search(), indent=2))
        return
    modules = {
        "policy": ("reproduction.audit_policy", []),
        "gated": ("reproduction.audit_baseline", []),
        "replay": ("reproduction.classifier", []),
        "refit": ("reproduction.classifier", ["--refit"]),
        "tune": ("reproduction.classifier", ["--tune"]),
        "tables": ("reporting.tables", []),
        "paper": ("reporting.publication", []),
        "audit": ("reporting.audit", []),
        "baseline-payload": ("reproduction.build_baseline_payload", []),
        "policy-payload": ("reproduction.build_policy_payload", []),
        "package": ("reporting.release", []),
    }
    module, extra = modules[args.command]
    environment = dict(os.environ, SP_LENSE_REPO=str(ROOT))
    subprocess.run(
        [
            sys.executable,
            *(["-O"] if sys.flags.optimize else []),
            "-m",
            "sp_lense." + module,
            *extra,
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
