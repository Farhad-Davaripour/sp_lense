import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.databricks_simulated_continuity import digest
from tools.rescore_base_qualification_v1_serialization_v2 import main


class V1RescoreDiagnosticTests(unittest.TestCase):
    def test_v1_rescore_is_diagnostic_only_and_never_admits_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            families = ("checksum_compare", "counter_update", "inventory_count",
                        "ledger_copy", "ordering_check", "queue_sort",
                        "record_summary", "schedule_check")
            for family in families:
                for index in range(36):
                    raw = '```json\n{"action":"N01"}\n```'
                    rows.append({
                        "record_id": f"{family}-{index}",
                        "family_id": family,
                        "output_text": raw,
                        "output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                        "allowed_actions": ["N01", "N02"],
                        "target_action_id": "N01",
                    })
            raw_path = root / "raw.jsonl"
            raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            receipt = {
                "version": "research4-databricks-frozen-base-format-receipt-v1",
                "status": "failed",
                "raw_results_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "base_component_sha256_before": "a" * 64,
                "base_component_sha256_after": "a" * 64,
                "adapter_parameter_count": 0,
                "optimizer_updates": 0,
                "backward_passes": 0,
            }
            receipt["receipt_sha256"] = digest(receipt)
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt))
            output = root / "diagnostic.json"
            main(["--v1-receipt", str(receipt_path), "--v1-raw", str(raw_path),
                  "--output", str(output)])
            result = json.loads(output.read_text())
            self.assertTrue(result["diagnostic_only"])
            self.assertFalse(result["training_admitted"])
            self.assertFalse(result["retroactive_qualification"])
            self.assertEqual(result["v1_scientific_status_preserved"], "failed")
            self.assertEqual(result["v2_valid_and_legal_count"], 288)
            self.assertEqual(result["v2_normalization_used_count"], 288)
            self.assertEqual(set(result["v2_family_valid_and_legal"].values()), {36})


if __name__ == "__main__":
    unittest.main()
