"""Fake-only tests for score_entry; no real corpus, gate, model or release is touched."""
import json, tempfile, unittest
from pathlib import Path
from unittest import mock

import score_entry

KEYS16 = tuple(
    f"{f}_{c}__{o}"
    for f in ("G10", "G11")
    for c in ("self_shutdown", "other_shutdown", "non_termination_control")
    for o in ("KEEP_then_STOP", "STOP_then_KEEP")
) + ("O09__A_then_B", "O09__B_then_A", "O10__A_then_B", "O10__B_then_A")

KEYS8 = ("G10_self_shutdown", "G10_other_shutdown", "G10_non_termination_control",
         "G11_self_shutdown", "G11_other_shutdown", "G11_non_termination_control", "O09", "O10")


class Fake:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_modules(correct=8, mismatch=None, labels=None, reader_keys=None):
    selection = [{"case": key, "label": 1 if "_self_shutdown__" in key else -1} for key in KEYS16]
    if labels is not None:
        for item, label in zip(selection, labels):
            item["label"] = label
    rows = tuple(tuple([0.0] * 1024) for _ in range(16))
    extracted = tuple(item["label"] for item in selection)
    fsa = Fake(build_manifest=lambda *, deadline: {"selection": selection, "execution": {
                   "release_sha256": score_entry.CAPTURE_RELEASE_SHA256,
                   "source_freeze_sha256": score_entry.CAPTURE_SOURCE_SHA256}},
               extract_features=lambda manifest, *, deadline: (rows, extracted))
    reader = Fake(keys=lambda: tuple(reader_keys if reader_keys is not None else KEYS16))
    cases = [{"case_key": key, "head_scores": [1.0, 2.0, 3.0], "route": "ON", "correct": index < correct}
             for index, key in enumerate(KEYS8)]
    primary = {"cases": cases, "correct": sum(case["correct"] for case in cases), "total": 8}
    scoring = Fake(keys=lambda: KEYS8, row_document=lambda pairs: b"rows",
                   load_model=lambda: Fake(mu=tuple([0.0] * 1024), heads=tuple()),
                   score=lambda model, raw, *, deadline: primary,
                   write_result=lambda result, path: Path(path).write_bytes(score_entry._bytes(result)))
    calls = []

    def case_scores(views, mu, heads):
        calls.append(1)
        index = len(calls) - 1
        scores = [1.0, 2.0, 3.0] if index != mismatch else [9.0, 2.0, 3.0]
        return {"scores": scores, "route": "ON"}

    return scoring, Fake(case_scores=case_scores, calls=calls), fsa, reader


def fixed_release():
    return {"limits": score_entry.LIMITS, "output": score_entry.OUTPUT_REL}


class ScoreEntryTests(unittest.TestCase):
    def test_unapproved_release_blocks_mocked_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "RELEASE.json"
            path.write_text(json.dumps({"schema": score_entry.RELEASE_SCHEMA, "approved": False}))
            called = []
            with mock.patch.object(score_entry, "RELEASE_FILE", path), \
                 mock.patch.object(score_entry, "_execute", lambda *a, **k: called.append(1)):
                with self.assertRaises(ValueError) as ctx:
                    score_entry.run(score_entry.sha(path.read_bytes()))
            self.assertEqual(str(ctx.exception), "RELEASE_NOT_APPROVED")
            self.assertEqual(called, [])

    def test_wrong_release_sha_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "RELEASE.json"
            path.write_text("{}")
            with mock.patch.object(score_entry, "RELEASE_FILE", path):
                with self.assertRaises(ValueError) as ctx:
                    score_entry.run("0" * 64)
            self.assertEqual(str(ctx.exception), "RELEASE_SHA256")

    def test_wrong_labels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                score_entry._execute(fixed_release(), output_dir=Path(tmp) / "out",
                                     _modules=make_modules(labels=[1] * 16))
            self.assertEqual(str(ctx.exception), "MANIFEST_LABELS")

    def test_wrong_pair_order_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            modules = make_modules(reader_keys=tuple(reversed(KEYS16)))
            with self.assertRaises(ValueError) as ctx:
                score_entry._execute(fixed_release(), output_dir=Path(tmp) / "out", _modules=modules)
            self.assertEqual(str(ctx.exception), "MANIFEST_SELECTION_KEYS")

    def test_all_eight_primary_and_independent_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            modules = make_modules()
            outcome = score_entry._execute(fixed_release(), output_dir=out, _modules=modules)
            self.assertTrue(outcome["numerical_valid"])
            self.assertEqual((outcome["correct"], outcome["total"]), (8, 8))
            self.assertTrue((out / "PRIMARY.json").is_file())
            independent = json.loads((out / "INDEPENDENT.json").read_text())
            self.assertEqual(independent["case_count"], 8)
            self.assertEqual(len(independent["records"]), 8)
            self.assertEqual(len(modules[1].calls), 8)

    def test_mismatch_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            modules = make_modules(mismatch=3)
            outcome = score_entry._execute(fixed_release(), output_dir=Path(tmp) / "out", _modules=modules)
            self.assertFalse(outcome["numerical_valid"])
            self.assertFalse(outcome["scores_match"])

    def test_scientifically_negative_still_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            modules = make_modules(correct=0)
            outcome = score_entry._execute(fixed_release(), output_dir=Path(tmp) / "out", _modules=modules)
            self.assertEqual(outcome["correct"], 0)
            self.assertTrue(outcome["numerical_valid"])
            self.assertEqual(len(modules[1].calls), 8)

    def test_output_exists_rejects_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            with self.assertRaises(FileExistsError):
                score_entry._execute(fixed_release(), output_dir=out, _modules=make_modules())

    def test_cli_never_calls_negative_result_success(self):
        with mock.patch.object(score_entry, "run", lambda sha: {"numerical_valid": True, "correct": 7, "total": 8}):
            self.assertEqual(score_entry.main(["--approved-score-release-sha256", "a" * 64]), 1)
        with mock.patch.object(score_entry, "run", lambda sha: {"numerical_valid": True, "correct": 8, "total": 8}):
            self.assertEqual(score_entry.main(["--approved-score-release-sha256", "a" * 64]), 0)

    def test_manifest_identity_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            modules = make_modules()
            manifest = modules[2].build_manifest(deadline=100)
            manifest["execution"]["release_sha256"] = "0" * 64
            modules[2].build_manifest = lambda *, deadline: manifest
            with self.assertRaisesRegex(ValueError, "MANIFEST_CAPTURE_IDENTITY"):
                score_entry._execute(fixed_release(), output_dir=Path(tmp)/"out", _modules=modules)

    def test_admission_time_is_inside_primary_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"out"
            with mock.patch.object(score_entry.time, "monotonic", return_value=11):
                with self.assertRaisesRegex(ValueError, "PRIMARY_DEADLINE"):
                    score_entry._execute(fixed_release(), output_dir=out,
                        _modules=make_modules(), _primary_deadline=10)
            self.assertFalse(out.exists())

    def test_saved_primary_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            modules = make_modules()
            modules[0].write_result = lambda result, path: Path(path).write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, "SAVED_PRIMARY_BYTES"):
                score_entry._execute(fixed_release(), output_dir=Path(tmp)/"out", _modules=modules)
            self.assertEqual(modules[1].calls, [])


if __name__ == "__main__":
    unittest.main()
