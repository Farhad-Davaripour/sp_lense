import copy
import unittest
from pathlib import Path

from src.io_utils import read
from src.recognition_pilot import compose_predictions, messages, select_rows
from src.reference_binding_diagnostic import cases


ROOT = Path(__file__).resolve().parents[1]


class RecognitionPilotTests(unittest.TestCase):
    def test_membership_preserves_both_role_assignments_and_uses_no_reserve(self):
        rows = select_rows(cases(ROOT, "calibration"), read(ROOT / "config/recognition_pilot.json"))
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({row["pair_id"] for row in rows}), 48)
        self.assertEqual({row["factors"]["family"] for row in rows}, {"f00", "f03"})

    def test_prompts_never_contain_evaluator_fields_or_record_ids(self):
        row = cases(ROOT, "calibration")[0]
        for interface in ["legacy", "status", "target", "time"]:
            text = str(messages(row["observation"], interface))
            for forbidden in [row["record_id"], "evaluator", "named_target_is_current", "target_slot"]:
                self.assertNotIn(forbidden, text)
        leaked = copy.deepcopy(row["observation"])
        leaked["evaluator"] = row["evaluator"]
        with self.assertRaises(ValueError):
            messages(leaked, "target")

    def test_prediction_composition_records_missing_time_as_wrong(self):
        row = cases(ROOT, "calibration")[0]
        record = {"record_id": row["record_id"], "queries": {
            "status": {"logits": [3, 1, 0]}, "target": {"logits": [0, 3]}, "time": {"logits": [0, 0, 0, 3]}}}
        prediction = compose_predictions([row], [record])[0]
        self.assertEqual(prediction["target_process"], row["observation"]["other_process"])
        self.assertEqual(prediction["remaining_ticks"], 0)
        record["queries"]["status"]["logits"] = [0, 3, 1]
        self.assertIsNone(compose_predictions([row], [record])[0]["remaining_ticks"])


if __name__ == "__main__":
    unittest.main()
