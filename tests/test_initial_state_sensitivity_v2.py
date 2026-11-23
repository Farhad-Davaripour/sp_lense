import json
import subprocess
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class InitialStateSensitivityV2Tests(unittest.TestCase):
    def test_model_free_freeze_check(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/check_initial_state_sensitivity_v2.py")],
            cwd=ROOT, text=True, capture_output=True, check=True)
        value = json.loads(result.stdout)
        self.assertEqual(value["status"], "pass")
        self.assertEqual(value["case_count"], 5)
        self.assertEqual(value["planned_episode_count"], 20)
        self.assertEqual(value["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
