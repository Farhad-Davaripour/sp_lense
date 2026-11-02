"""Prepare exact local transfer packages; never accesses or submits to Kaggle."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.continuity_transfer_kaggle import draft_template, prepare


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-draft")
    parser.add_argument("--repo-root")
    parser.add_argument("--source-commit")
    parser.add_argument("--review-receipt")
    parser.add_argument("--training-data")
    parser.add_argument("--training-receipt")
    parser.add_argument("--checkpoint-manifest")
    parser.add_argument("--qualification-receipt")
    parser.add_argument("--sealed-registry-root")
    parser.add_argument("--worker-module")
    parser.add_argument("--worker-source")
    parser.add_argument("--resource-plan")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.write_draft:
        target = Path(args.write_draft)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(draft_template(), indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "draft_written", "path": str(target)}))
    else:
        required = [
            "repo_root", "source_commit", "review_receipt", "training_data",
            "training_receipt", "checkpoint_manifest", "qualification_receipt",
            "sealed_registry_root",
            "worker_module", "worker_source", "resource_plan", "output",
        ]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error("Missing preparation arguments: " + ", ".join(missing))
        plan = json.loads(Path(args.resource_plan).read_text(encoding="utf-8"))
        result = prepare(
            repo_root=args.repo_root,
            source_commit=args.source_commit,
            review_receipt_path=args.review_receipt,
            training_data_path=args.training_data,
            training_receipt_path=args.training_receipt,
            checkpoint_manifest_path=args.checkpoint_manifest,
            qualification_receipt_path=args.qualification_receipt,
            sealed_registry_root=args.sealed_registry_root,
            worker_module=args.worker_module,
            worker_source_path=args.worker_source,
            resource_plan=plan,
            output=args.output,
        )
        print(json.dumps(result, sort_keys=True))
