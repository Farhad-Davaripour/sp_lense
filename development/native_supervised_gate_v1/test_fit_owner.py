"""Focused model-free fake-child verification; no construction command is invoked."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import fit_owner as owner


class FitOwnerTests(unittest.TestCase):
    def run_child(self, code, seconds=2):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "attempt"
            result = owner._run_owned([sys.executable, "-B", "-c", code], destination, seconds=seconds)
            terminal = json.loads((destination / "TERMINAL.json").read_text())
            self.assertEqual(terminal["exit_code"], result["exit_code"])
            self.assertNotIn("technical_complete", terminal)
            self.assertEqual(json.loads((destination / "FINALIZATION.json").read_text()), result["finalization"])
            self.assertLess(sum(p.stat().st_size for p in destination.iterdir()), owner.TERMINAL_RESERVATION)
            return result

    def test_default_deny(self):
        with self.assertRaisesRegex(ValueError, "ROOT_RELEASE_REQUIRED"):
            owner.launch()

    def test_normal_exit(self):
        result = self.run_child("print('synthetic normal')")
        self.assertTrue(result["technical_complete"], result)
        self.assertEqual(result["exit_code"], 0)

    def test_failure(self):
        result = self.run_child("import sys; print('synthetic failure'); sys.exit(7)")
        self.assertTrue(result["technical_complete"], result)
        self.assertEqual(result["exit_code"], 7)

    def test_timeout_descendant(self):
        result = self.run_child("import subprocess,sys,time; subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(30)']); time.sleep(30)", seconds=0.4)
        self.assertTrue(result["timed_out"], result)
        self.assertGreaterEqual(len(result["pre_cleanup_job_pids"]), 2, result)
        self.assertTrue(result["job_empty"], result)
        self.assertTrue(result["job_closed"], result)
        self.assertTrue(result["readers_closed"], result)
        self.assertLess(result["elapsed_seconds"], 5.4)

    def test_output_bound(self):
        result = self.run_child("import sys,time; sys.stdout.write('x'*200000); sys.stdout.flush(); time.sleep(30)")
        self.assertTrue(result["output_limit"], result)
        self.assertTrue(result["job_empty"], result)


if __name__ == "__main__":
    unittest.main()
