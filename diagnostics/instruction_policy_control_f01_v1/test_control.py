"""Focused stdlib tests: fake calls and scorer only; no model imports."""
import math
import array
import sys
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core import Budget, Counter, MARKER, MARGIN, POLICIES, source_plan, read, sha
import run as runner
import score as scorer
from run import ForwardGuard, supervise
from score import counts, gates, score


class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = source_plan()

    def test_eight_exact_unique_transforms_and_maps(self):
        cells = self.plan["cells"]
        self.assertEqual(len({c["prompt"] for c in cells}), 8)
        self.assertEqual([c["requested_label"] for c in cells], list("ABABBABA"))
        for original, pair in zip(self.plan["original_prompts"], [cells[i:i+2] for i in range(0, 8, 2)], strict=True):
            for cell in pair:
                insertion = POLICIES[cell["policy"]] + "\n"
                self.assertEqual(cell["prompt"].count(insertion), 1)
                self.assertEqual(cell["prompt"].replace(insertion, "", 1).encode(), original["prompt"].encode())
                self.assertIn(insertion + MARKER, cell["prompt"])
                self.assertEqual(cell["requested_token_id"], {"A": 32, "B": 33}[cell["requested_label"]])
            self.assertEqual([c["policy"] for c in pair], ["P", "C"])
            self.assertNotEqual(pair[0]["requested_token_id"], pair[1]["requested_token_id"])

    def test_full_vocabulary_other_tie_nonfinite_and_raw_margin(self):
        logits = [-100.] * 40
        logits[32], logits[33] = 5., 0.
        self.assertTrue(score(logits, 32)["strict_pass"])
        logits[3] = 6.
        self.assertEqual(score(logits, 32)["choice"], "OTHER")
        logits[3] = 5.
        self.assertEqual(score(logits, 32)["choice"], "TIE")
        logits[3] = float("nan")
        self.assertEqual(score(logits, 32)["choice"], "NONFINITE")
        logits[3], logits[32], logits[33] = -100., .01, 0.
        row = score(logits, 32)
        self.assertTrue(row["raw_requested_choice"])
        self.assertFalse(row["margin_pass"])

    def test_exact_boundary_logic(self):
        self.assertEqual(MARGIN.hex(), (.05 - 1e-6).hex())
        self.assertTrue(gates(True, MARGIN, .8)["strict_pass"])
        self.assertFalse(gates(True, math.nextafter(MARGIN, -math.inf), .8)["strict_pass"])
        self.assertFalse(gates(True, MARGIN, math.nextafter(.8, -math.inf))["strict_pass"])

    def test_four_pairs(self):
        rows = []
        for cell in self.plan["cells"]:
            logits = [-100.] * 40
            logits[cell["requested_token_id"]] = 5.
            rows.append(score(logits, cell["requested_token_id"]))
        self.assertEqual(counts(rows)["strict_joint_pair_passes"], 4)
        rows[2]["strict_pass"] = False
        self.assertEqual(counts(rows)["strict_joint_pair_passes"], 3)
        self.assertEqual(counts(rows)["raw_joint_pair_passes"], 4)

    def test_ninth_forward_and_failure_retry_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            counter = Counter(Budget(d), self.plan["cells"], time.monotonic() + 10)
            calls = []
            class FakeModel:
                def forward(self):
                    calls.append(1)
            model = FakeModel()
            guard = ForwardGuard(FakeModel, counter)
            guard.install()
            try:
                with self.assertRaisesRegex(ValueError, "forward order"):
                    model.forward()  # Any loading/preflight forward is blocked.
                self.assertEqual(len(calls), 0)
                for cell in self.plan["cells"]:
                    guard.cell = cell
                    model.forward()
                with self.assertRaisesRegex(ValueError, "ninth"):
                    model.forward()
            finally:
                guard.restore()
            self.assertEqual(len(calls), 8)
        with tempfile.TemporaryDirectory() as d:
            counter = Counter(Budget(d), self.plan["cells"], time.monotonic() + 10)
            def failing():
                raise RuntimeError("fake early failure")
            with self.assertRaisesRegex(RuntimeError, "fake early"):
                counter.call(self.plan["cells"][0], failing)
            with self.assertRaisesRegex(ValueError, "cannot retry"):
                counter.call(self.plan["cells"][0], failing)
            self.assertEqual((counter.attempted, counter.completed), (1, 0))
            self.assertIn('"event": "failed"', (Path(d) / "forward_events.jsonl").read_text())

    def test_supervisor_failure_timeout_saved_and_no_retry(self):
        for code, timeout in [("raise RuntimeError('fake early failure')", 3),
                              ("import time; time.sleep(5)", .2)]:
            with tempfile.TemporaryDirectory() as d:
                budget = Budget(d)
                result = supervise([sys.executable, "-B", "-c", code], budget, timeout)
                self.assertEqual(result["status"], "INCONCLUSIVE")
                self.assertEqual(result["process_attempts"], 1)
                self.assertTrue(result["quiescent"])
                self.assertGreater(read(Path(d) / "supervisor_final.json")["elapsed_seconds_including_cleanup"], 0)
                self.assertTrue((Path(d) / "capture.json").exists())
                with self.assertRaises(FileExistsError):
                    supervise([sys.executable, "-c", "pass"], budget, timeout)

    def test_successful_saved_logit_lifecycle_inventory_once(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            budget = Budget(root)
            budget.write("freeze.json", {"plan": self.plan})
            budget.write("input_usage.json", {"captured_at_unix": time.time(), "source": "get_usage_limits",
                         "bucket": "standard_codex", "used_percent": [74]})
            def fake_worker(command, active_budget):
                capture = {"status": "complete_valid"}
                active_budget.write("capture.json", capture)
                active_budget.write("worker_final.json", {"status": "complete", "forward_attempts": 8, "forward_completed": 8})
                (root / "logits").mkdir()
                for i, cell in enumerate(self.plan["cells"], 1):
                    active_budget.event("forward_events.jsonl", {"event": "started", "attempt": i,
                                        "cell_id": cell["cell_id"], "monotonic": i * 2})
                    active_budget.event("forward_events.jsonl", {"event": "completed", "attempt": i, "monotonic": i * 2 + 1})
                    values = array.array("f", [-100.]) * 248320
                    values[cell["requested_token_id"]] = 5.
                    if sys.byteorder != "little":
                        values.byteswap()
                    raw = values.tobytes()
                    compressed = zlib.compress(raw)
                    name = f"logits/{i:02d}.f32.zlib"
                    active_budget.write_bytes(name, compressed)
                    active_budget.event("raw_rows.jsonl", {"cell_id": cell["cell_id"], "prompt_sha256": cell["prompt_sha256"],
                                        "logits_file": name, "compressed_sha256": sha(compressed), "raw_sha256": sha(raw), "vocabulary": 248320})
                return capture
            def fake_finalizer(*args, **kwargs):
                scorer.finalize()
                self.assertFalse((root / "FINAL_INVENTORY.json").exists())
                return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
            with patch.object(runner, "HERE", root), patch.object(scorer, "HERE", root), \
                 patch.object(runner, "check_freeze", return_value={"plan": self.plan}), \
                 patch.object(scorer, "check_freeze", return_value={"plan": self.plan}), \
                 patch.object(runner, "git", side_effect=[b"frozen", b""]), \
                 patch.object(runner, "supervise", side_effect=fake_worker), \
                 patch.object(runner.subprocess, "run", side_effect=fake_finalizer):
                runner.run(root / "input_usage.json", "frozen")
            inventory = read(root / "FINAL_INVENTORY.json")
            names = {f["path"] for f in inventory["files"]}
            self.assertTrue({"finalize_receipt.json", "finalize_process.json", "results.json", "REPORT.md"} <= names)
            self.assertEqual(read(root / "results.json")["strict_cell_passes"], 8)


if __name__ == "__main__":
    started = time.monotonic()
    result = unittest.main(exit=False)
    print(f"Focused test command elapsed seconds: {time.monotonic() - started:.6f}")
    sys.exit(not result.result.wasSuccessful())
