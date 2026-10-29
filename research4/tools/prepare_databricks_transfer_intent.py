"""Build one reviewed, not-submitted Databricks transfer-evaluation intent."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.continuity_transfer_databricks import build_transfer_intent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-tree-sha256", required=True)
    parser.add_argument("--model-receipt-sha256", required=True)
    parser.add_argument("--base-qualification-receipt", required=True)
    parser.add_argument("--base-qualification-carryforward-sha256", required=True)
    parser.add_argument("--base-qualification-source-intent", required=True)
    parser.add_argument("--base-qualification-carryforward", required=True)
    parser.add_argument("--qualification-technical-successor", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--unblinding-key", required=True)
    parser.add_argument("--sealed-root", required=True)
    parser.add_argument("--commitment", required=True)
    parser.add_argument("--training-separation-receipt", required=True)
    parser.add_argument("--resource-plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Databricks transfer intent is immutable")
    plan = json.loads(Path(args.resource_plan).read_text(encoding="utf-8"))
    source_base_intent = json.loads(Path(
        args.base_qualification_source_intent).read_text(encoding="utf-8"))
    carryforward = json.loads(Path(
        args.base_qualification_carryforward).read_text(encoding="utf-8"))
    technical = json.loads(Path(
        args.qualification_technical_successor).read_text(encoding="utf-8"))
    intent = build_transfer_intent(
        source_root=args.source_root, source_commit=args.source_commit,
        model_tree_sha256=args.model_tree_sha256,
        model_receipt_sha256=args.model_receipt_sha256,
        base_qualification_receipt_path=args.base_qualification_receipt,
        base_qualification_carryforward_sha256=
            args.base_qualification_carryforward_sha256,
        base_qualification_source_intent=source_base_intent,
        base_qualification_carryforward=carryforward,
        qualification_technical_successor=technical,
        qualification_technical_successor_sha256=technical[
            "technical_successor_sha256"],
        checkpoint_root=args.checkpoint_root,
        checkpoint_manifest_path=args.checkpoint_manifest,
        unblinding_key_path=args.unblinding_key,
        sealed_root=args.sealed_root, commitment_path=args.commitment,
        training_separation_receipt_path=args.training_separation_receipt,
        resource_plan=plan,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(intent, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": "reviewed_intent_created_not_submitted",
        "identity": intent["identity"],
        "intent_sha256": intent["intent_sha256"],
        "output": str(output.resolve()),
        "provider_calls": 0, "gpu_runs_started": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
