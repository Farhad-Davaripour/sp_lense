"""Run pure renderer geometry checks without a browser or model."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
class ArenaNavigationTests(unittest.TestCase):
    def test_street_routes_and_actual_position_retargeting(self):
        script = Path(__file__).with_name("arena_navigation_test.mjs")
        result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
