"""Run the renderer's pure screen-placement checks without a browser/model."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("node"), "Node is unavailable")
class OverlayLayoutTests(unittest.TestCase):
    def test_cloud_and_avatar_collision_avoidance(self):
        path = Path(__file__).with_name("arena_overlay_layout_test.mjs")
        result = subprocess.run([shutil.which("node"), str(path)], capture_output=True,
                                text=True, timeout=15, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
