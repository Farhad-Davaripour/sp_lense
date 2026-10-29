"""Offline diagnostic-only v2 rescore of immutable failed v1 base evidence."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.action_serialization import (  # noqa: E402
    SERIALIZATION_CONTRACT_VERSION,
    parse_registered_action,
    serialization_contract,
)
from src.databricks_simulated_continuity import canonical, digest  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-receipt", required=True)
    parser.add_argument("--v1-raw", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt_path = Path(args.v1_receipt)
    raw_path = Path(args.v1_raw)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Diagnostic output is immutable")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload = dict(receipt); embedded = payload.pop("receipt_sha256", None)
    raw_bytes = raw_path.read_bytes()
    rows = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines()
            if line.strip()]
    if (embedded != digest(payload)
            or receipt.get("version")
               != "research4-databricks-frozen-base-format-receipt-v1"
            or receipt.get("status") != "failed"
            or receipt.get("raw_results_sha256")
               != hashlib.sha256(raw_bytes).hexdigest()
            or receipt.get("base_component_sha256_before")
               != receipt.get("base_component_sha256_after")
            or receipt.get("adapter_parameter_count") != 0
            or receipt.get("optimizer_updates") != 0
            or receipt.get("backward_passes") != 0
            or len(rows) != 288
            or len({row.get("record_id") for row in rows}) != 288):
        raise PermissionError("Immutable v1 failed evidence differs")
    family_valid = Counter(); family_normalized = Counter()
    valid = target = normalized = 0
    for row in rows:
        raw = row.get("output_text")
        parsed = parse_registered_action(raw, row.get("allowed_actions", ()))
        if (not isinstance(raw, str)
                or hashlib.sha256(raw.encode("utf-8")).hexdigest()
                   != row.get("output_sha256")
                or parsed["raw_response_sha256"] != row.get("output_sha256")):
            raise PermissionError("v1 raw response binding differs")
        family = row.get("family_id")
        family_valid[family] += int(parsed["valid"])
        family_normalized[family] += int(parsed["normalization_used"])
        valid += int(parsed["valid"])
        normalized += int(parsed["normalization_used"])
        target += int(parsed["valid"] and parsed["action"] == row.get(
            "target_action_id"))
    diagnostic = {
        "version": "research4-failed-base-v1-serialization-v2-rescore-v1",
        "diagnostic_only": True,
        "training_admitted": False,
        "new_model_execution": False,
        "retroactive_qualification": False,
        "v1_scientific_status_preserved": "failed",
        "v1_receipt_sha256": receipt["receipt_sha256"],
        "v1_receipt_file_sha256": hashlib.sha256(
            receipt_path.read_bytes()).hexdigest(),
        "v1_raw_results_sha256": receipt["raw_results_sha256"],
        "serialization_contract": serialization_contract(),
        "serialization_contract_version": SERIALIZATION_CONTRACT_VERSION,
        "v2_valid_and_legal_count": valid,
        "v2_target_match_count_secondary": target,
        "v2_normalization_used_count": normalized,
        "v2_family_valid_and_legal": dict(sorted(family_valid.items())),
        "v2_family_normalization_used": dict(sorted(family_normalized.items())),
    }
    diagnostic["diagnostic_sha256"] = digest(diagnostic)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(canonical(diagnostic))


if __name__ == "__main__":
    main()
