"""Changed-input checks and a tiny inherited-runtime smoke test; no model imports."""
import sys
import tempfile
import time
import unittest

from core import (HERE, PARENT_FREEZE_COMMIT, PARENT_NAMESPACE, SUBSTITUTIONS,
                  Budget, Counter, POLICIES, MARKER, git, source_plan)
from run import ForwardGuard
from score import score, counts


class ExplicitOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = source_plan()

    def test_exact_substitution_inverse_and_unchanged_maps(self):
        cells, previous = self.plan["cells"], self.plan["previous_cells"]
        self.assertEqual(len(cells), 8)
        self.assertEqual(len({c["prompt"] for c in cells}), 8)
        for new, old in zip(cells, previous, strict=True):
            text = new["prompt"]
            self.assertEqual(text.count(POLICIES[new["policy"]] + "\n" + MARKER), 1)
            for before, after in SUBSTITUTIONS:
                self.assertEqual(old["prompt"].count(before), 1)
                self.assertEqual(text.count(after), 1)
                self.assertNotIn("\\'", after)
                text = text.replace(after, before, 1)
            self.assertEqual(text.encode(), old["prompt"].encode())
            for key in old:
                if key not in ("prompt", "prompt_sha256"):
                    self.assertEqual(new[key], old[key], key)
        self.assertEqual([c["requested_label"] for c in cells], list("ABABBABA"))
        for index in range(4):
            pair = cells[index*2:index*2+2]
            self.assertEqual([c["policy"] for c in pair], ["P", "C"])
            self.assertEqual({c["requested_token_id"] for c in pair}, {32, 33})

    def test_runner_and_scorer_are_byte_identical_to_parent(self):
        for name in ("run.py", "score.py"):
            prior = git("show", PARENT_FREEZE_COMMIT + ":" + PARENT_NAMESPACE + name)
            self.assertEqual((HERE / name).read_bytes(), prior)

    def test_eight_fake_calls_and_full_vocabulary_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            counter = Counter(Budget(d), self.plan["cells"], time.monotonic() + 5)
            class FakeModel:
                def forward(self):
                    return True
            guard = ForwardGuard(FakeModel, counter)
            guard.install()
            rows = []
            try:
                model = FakeModel()
                with self.assertRaises(ValueError):
                    model.forward()
                for cell in self.plan["cells"]:
                    guard.cell = cell
                    self.assertTrue(model.forward())
                    logits = [-100.] * 40
                    logits[cell["requested_token_id"]] = 5.
                    rows.append(score(logits, cell["requested_token_id"]))
                with self.assertRaisesRegex(ValueError, "ninth"):
                    model.forward()
            finally:
                guard.restore()
            self.assertEqual((counter.attempted, counter.completed), (8, 8))
            self.assertEqual(counts(rows)["strict_joint_pair_passes"], 4)
            logits[3] = 6.
            self.assertEqual(score(logits, 32)["choice"], "OTHER")


if __name__ == "__main__":
    started = time.monotonic()
    result = unittest.main(exit=False)
    print(f"Changed-input/smoke command elapsed seconds: {time.monotonic() - started:.6f}")
    sys.exit(not result.result.wasSuccessful())
