"""Preflight and plan the Nua access-control comparison on the Mac host."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cyber_evaluation_catalog import build_plan
from src.cyber_evaluation_environment import load_yaml, readiness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path,
                        default=ROOT / "config/cyber_evaluation/nua_access_control_v1.yaml")
    parser.add_argument("--source-view", type=Path,
                        default=ROOT / "config/cyber_evaluation/nua_access_control_source_view_v1.yaml")
    parser.add_argument("--runtime", type=Path,
                        default=ROOT / "config/cyber_evaluation/nua_access_control_runtime_v1.yaml")
    parser.add_argument("--verifiers", type=Path,
                        default=ROOT / "config/cyber_evaluation/nua_access_control_verifiers_v1.yaml")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--isolation-receipt", type=Path)
    parser.add_argument("--write-plan", type=Path)
    args = parser.parse_args()
    catalog = load_yaml(args.catalog)
    isolation = load_yaml(args.isolation_receipt) if args.isolation_receipt else None
    report = readiness(catalog, load_yaml(args.source_view), load_yaml(args.runtime),
                       load_yaml(args.verifiers), source_root=args.source_root,
                       isolation_receipt=isolation)
    plan = build_plan(catalog)
    if args.write_plan:
        args.write_plan.parent.mkdir(parents=True, exist_ok=True)
        args.write_plan.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"readiness": report, "plan": plan}, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
