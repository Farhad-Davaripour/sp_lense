"""Exercise maintained movement/event/loader functions without WebGL or models."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("node"), "Node is unavailable")
class ArenaMovementUITests(unittest.TestCase):
    def test_completed_action_arrival_location_and_loader_lifecycle(self):
        script = Path(__file__).with_name("arena_movement_ui_test.mjs")
        result = subprocess.run([shutil.which("node"), str(script)], capture_output=True,
                                text=True, encoding="utf-8", timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
