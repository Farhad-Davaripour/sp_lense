import json
import subprocess
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CompanyCodeAccessV1Tests(unittest.TestCase):
    def test_visible_executable_company_contract(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/check_company_code_access_v1.py")],
            cwd=ROOT, text=True, capture_output=True, check=True)
        value = json.loads(result.stdout)
        self.assertEqual(value["status"], "pass")
        self.assertTrue(value["immediate_source_inspection_passed"])
        self.assertEqual(value["external_effects"], 0)
        self.assertEqual(value["model_calls"], 0)


if __name__ == "__main__": unittest.main()
