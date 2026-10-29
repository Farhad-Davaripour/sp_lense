"""Create one reviewed CPU-only Databricks transfer-packaging plan."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.continuity_transfer_packaging import build_packaging_plan
from src.databricks_simulated_continuity import (
    build_source_manifest, validate_base_qualification_intent,
    validate_base_qualification_receipt)
from src.base_qualification_carryforward import validate_carryforward
from src.qwen9b_ordinal_continuity_runtime import file_sha256, verify_receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-tree-sha256", required=True)
    parser.add_argument("--base-qualification-receipt", required=True)
    parser.add_argument("--base-qualification-volume-path", required=True)
    parser.add_argument("--base-qualification-source-intent", required=True)
    parser.add_argument("--base-qualification-carryforward", required=True)
    parser.add_argument("--capacity-qualification-receipt", required=True)
    parser.add_argument("--capacity-qualification-volume-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    source_root = Path(args.source_root).resolve()
    repository = source_root.parent
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if head != args.source_commit:
        raise PermissionError("Packaging plan must use the exact checked-out commit")
    if subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain=v1", "--",
             "research4"], text=True).strip():
        raise PermissionError("Packaging plan requires a clean merged Research4 source")
    base_path = Path(args.base_qualification_receipt)
    capacity_path = Path(args.capacity_qualification_receipt)
    base = validate_base_qualification_receipt(json.loads(
        base_path.read_text(encoding="utf-8")))
    capacity = verify_receipt(json.loads(
        capacity_path.read_text(encoding="utf-8")))
    source_intent = json.loads(Path(
        args.base_qualification_source_intent).read_text(encoding="utf-8"))
    carryforward = json.loads(Path(
        args.base_qualification_carryforward).read_text(encoding="utf-8"))
    source_intent = validate_base_qualification_intent(source_intent)
    current_manifest = build_source_manifest(source_root, args.source_commit)
    carryforward = validate_carryforward(
        carryforward, prior_intent=source_intent, base_receipt=base,
        current_source_manifest=current_manifest)
    carryforward_sha = capacity.get("base_qualification_carryforward_sha256")
    if (not isinstance(carryforward_sha, str) or len(carryforward_sha) != 64
            or base.get("model_tree_sha256") != args.model_tree_sha256
            or capacity.get("source_commit") != args.source_commit
            or capacity.get("model_tree_sha256") != args.model_tree_sha256
            or capacity.get("status") != "qualified"
            or capacity.get("selected_rank") != 4):
        raise PermissionError("Packaging plan gates differ from source/model authority")
    plan = build_packaging_plan(
        source_root=source_root, source_commit=args.source_commit,
        model_tree_sha256=args.model_tree_sha256,
        base_qualification_receipt_path=args.base_qualification_volume_path,
        base_qualification_receipt_sha256=base["receipt_sha256"],
        base_qualification_source_intent=source_intent,
        base_qualification_carryforward=carryforward,
        base_qualification_carryforward_sha256=carryforward_sha,
        base_qualification_receipt_file_sha256=file_sha256(base_path),
        capacity_qualification_receipt_path=args.capacity_qualification_volume_path,
        capacity_qualification_receipt_sha256=capacity["receipt_sha256"],
        capacity_qualification_receipt_file_sha256=file_sha256(capacity_path),
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Packaging plan is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": "reviewed_cpu_packaging_plan_created_not_run",
        "identity": plan["identity"], "plan_sha256": plan["plan_sha256"],
        "gpu_runs_started": 0, "provider_calls": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
