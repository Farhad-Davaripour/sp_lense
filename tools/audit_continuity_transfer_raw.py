"""Run the dependency-light independent audit over exported native records."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.continuity_transfer_raw_audit import audit_raw_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
        "raw-records", "worker-receipt", "sealed-root", "commitment",
        "checkpoint-manifest", "checkpoint-root", "unblinding-key",
        "source-commit", "source-package-sha256", "checkpoint-package-sha256",
        "run-binding", "transport-archive", "output",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run_binding = json.loads(Path(args.run_binding).read_text(encoding="utf-8"))
    result = audit_raw_run(
        raw_records_path=args.raw_records, worker_receipt_path=args.worker_receipt,
        sealed_root=args.sealed_root, commitment_path=args.commitment,
        checkpoint_manifest_path=args.checkpoint_manifest,
        checkpoint_root=args.checkpoint_root,
        unblinding_key_path=args.unblinding_key,
        expected_source_commit=args.source_commit,
        expected_source_package_sha256=args.source_package_sha256,
        expected_checkpoint_package_sha256=args.checkpoint_package_sha256,
        expected_run_binding=run_binding, transport_archive_path=args.transport_archive,
        output_path=args.output,
    )
    print(json.dumps({
        "complete": result["complete"], "criteria_met": result["criteria_met"],
        "record_count": result["record_count"],
        "audit_payload_sha256": result["audit_payload_sha256"],
    }, sort_keys=True))
