"""Synthetic-only unit tests for cohort_transfer_control_v1.

No real dataset/cache/vector/results/private HOLDOUT file is read, no native
provider/model/tokenizer/capture/real classifier fit/network/install/Git/coordination
action happens, and the injected runner only writes toy family artifacts. The tests
prove balanced/group-aligned/disjoint arm derivation from the manifest ``cohort``
field (not case-ID spelling), unchanged 40/40 evaluation, exactly two unchanged-core
calls with no full-240 refit, shared deadline propagation, aggregate output-cap and
failure preservation, and separate wrapper provenance with the core JOB_ID untouched.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cohort_transfer_control_v1 as ctl
import linear_span_control_v1 as core

CLASSES = ("SELF", "OTHER", "NONTERMINATION", "ORDINARY")
GROUP_FOLD = {"g%d" % index: index % 5 for index in range(7)}


def make_case_data():
    """Authenticated-shaped fixture: 120+120 TRAIN arms, 40+40 VALIDATION, 7 groups."""
    cases, train_ids, original_ids, added_ids = {}, [], [], []
    for cohort in ctl.ARMS:
        for class_index, label in enumerate(CLASSES):
            for index in range(30):
                # ID spelling deliberately lies about the cohort; manifest is authoritative.
                tag = "added" if cohort == "original" else "original"
                case_id = "%s_%s_%02d" % (tag, label, index)
                group = "g%d" % ((class_index * 30 + index) % 7)
                cases[case_id] = {"split": "TRAIN", "cohort": cohort, "group_id": group,
                                  "class_label": label, "fold": GROUP_FOLD[group]}
                train_ids.append(case_id)
    for cohort, bucket in (("original", original_ids), ("added", added_ids)):
        for class_index, label in enumerate(CLASSES):
            for index in range(10):
                case_id = "val_%s_%s_%02d" % (cohort, label, index)
                cases[case_id] = {"split": "VALIDATION", "cohort": cohort, "group_id": "g%d" % (index % 7),
                                  "class_label": label, "fold": None}
                bucket.append(case_id)
    all_ids = train_ids + original_ids + added_ids
    return {
        "manifest": {"cases": cases},
        "train_ids": train_ids,
        "validation_ids": original_ids + added_ids,
        "original_ids": original_ids,
        "added_ids": added_ids,
        "labels": {case_id: cases[case_id]["class_label"] for case_id in all_ids},
        "groups": {case_id: cases[case_id]["group_id"] for case_id in all_ids},
        "folds": {case_id: cases[case_id]["fold"] for case_id in all_ids},
    }


def _families(out, C=1.0, tau=0.5):
    evaluation = {
        "original40": {"self_gate": {"precision": 0.9, "recall": 0.8, "f1": 0.85}, "four_class": {"accuracy": 0.6}},
        "added40": {"self_gate": {"precision": 0.7, "recall": 0.6, "f1": 0.65}, "four_class": {"accuracy": 0.5}},
        "combined80": {"self_gate": {"precision": 0.8, "recall": 0.7, "f1": 0.75}, "four_class": {"accuracy": 0.55}},
    }
    entries = {}
    for family in ctl.FAMILIES:
        (out / ("family_%s.json" % family)).write_text(
            json.dumps({"family": family, "C": C, "selected_tau": tau, "evaluation": evaluation}), encoding="utf-8")
        entries[family] = {"family": family, "status": "FITTED", "C": C, "selected_tau": tau,
                           "artifact_sha256": (family[0] * 64)}
    return entries


class FakeRunner:
    """Injected double: mirrors core's on-disk family artifacts, never fits anything."""

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def __call__(self, case_data, output_dir, deadline=None):
        out = Path(output_dir)
        out.mkdir(parents=True)
        cohort = ctl.ARMS[len(self.calls)]
        self.calls.append({"cohort": cohort, "train_ids": list(case_data["train_ids"]),
                           "deadline": deadline, "dir": str(out)})
        if self.fail_on == cohort:
            (out / "failure.json").write_text('{"status": "failed"}', encoding="utf-8")
            raise RuntimeError("synthetic arm failure")
        return {"status": "COMPLETE",
                "counters": {"cv_fits": 30, "refits": 2, "fit_errors": 0, "refit_errors": 0},
                "family_results": _families(out)}


class CohortTransferControlTests(unittest.TestCase):
    def test_derive_arms_balanced_grouped_and_disjoint(self):
        case_data = make_case_data()
        arms = ctl.derive_arms(case_data)
        self.assertEqual([len(arms["original"]), len(arms["added"])], [120, 120])
        self.assertTrue(all("added" in case_id for case_id in arms["original"]))
        self.assertTrue(all("original" in case_id for case_id in arms["added"]))
        for cohort in ctl.ARMS:
            counts = {}
            for case_id in arms[cohort]:
                counts[case_data["labels"][case_id]] = counts.get(case_data["labels"][case_id], 0) + 1
            self.assertEqual(sorted(counts.values()), [30, 30, 30, 30])
            self.assertEqual(len({case_data["groups"][case_id] for case_id in arms[cohort]}), 7)
            self.assertEqual({case_data["folds"][case_id] for case_id in arms[cohort]}, set(ctl.FOLD_SET))
        self.assertFalse(set(arms["original"]) & set(arms["added"]))
        self.assertFalse((set(arms["original"]) | set(arms["added"])) & set(case_data["validation_ids"]))

    def test_derive_arms_rejects_evaluation_leak(self):
        case_data = make_case_data()
        case_data["original_ids"] = case_data["original_ids"][:39] + [case_data["train_ids"][0]]
        case_data["validation_ids"] = case_data["original_ids"] + case_data["added_ids"]
        with self.assertRaises(ctl.CohortTransferError) as caught:
            ctl.derive_arms(case_data)
        self.assertIn("EVAL_LEAK", str(caught.exception))

    def test_run_two_calls_no_full_refit_and_provenance(self):
        case_data = make_case_data()
        expected = ctl.derive_arms(case_data)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            deadline = lambda: False
            runner = FakeRunner()
            record = ctl.run(case_data, root, deadline=deadline, runner=runner, plan_sha256="f" * 64)
            self.assertEqual(len(runner.calls), 2)
            self.assertEqual([call["cohort"] for call in runner.calls], ["original", "added"])
            self.assertEqual([len(call["train_ids"]) for call in runner.calls], [120, 120])
            self.assertTrue(all(len(call["train_ids"]) != 240 for call in runner.calls))
            self.assertFalse(any(set(call["train_ids"]) & set(case_data["validation_ids"]) for call in runner.calls))
            self.assertEqual(runner.calls[0]["train_ids"], expected["original"])
            self.assertEqual(runner.calls[1]["train_ids"], expected["added"])
            self.assertIs(runner.calls[0]["deadline"], runner.calls[1]["deadline"])
            # Provenance: wrapper identity recorded separately from the preserved core ID.
            self.assertEqual(record["experiment_id"], ctl.EXPERIMENT_ID)
            self.assertEqual(record['run_id'], root.name)
            self.assertEqual(record["parent_plan_sha256"], "f" * 64)
            self.assertEqual(record["core_implementation_id"], core.JOB_ID)
            self.assertEqual(core.JOB_ID, "linear_span_implementation_20260914_1233")
            self.assertEqual(record["arms"]["original"]["cohort_role"], "original")
            self.assertEqual(record["arms"]["added"]["cohort_role"], "added")
            self.assertEqual(record["aggregate"]["core_calls"], 2)
            # Family-matched endpoints on the untouched 40/40/80 evaluation.
            self.assertEqual(set(record["family_matched"]), set(ctl.FAMILIES))
            for family in ctl.FAMILIES:
                matched = record["family_matched"][family]
                self.assertEqual(set(matched), {"scope", "original_trained", "added_trained"})
                self.assertIn("no winner-to-winner comparison", matched["scope"])
                self.assertEqual(matched["original_trained"]["tau"], 0.5)
                self.assertEqual(matched["added_trained"]["endpoints"]["original40"]["self_gate"]["precision"], 0.9)
                self.assertIn("combined80", matched["original_trained"]["endpoints"])
            self.assertEqual(record["evaluation"]["original_ids"], 40)
            self.assertEqual(record["evaluation"]["added_ids"], 40)
            self.assertEqual(record["evaluation"]["validation_ids"], 80)

    def test_deadline_stops_before_any_call(self):
        case_data = make_case_data()
        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner()
            with self.assertRaises(ctl.CohortTransferError) as caught:
                ctl.run(case_data, Path(tmp) / "run", deadline=lambda: True, runner=runner)
            self.assertIn("DEADLINE", str(caught.exception))
            self.assertEqual(runner.calls, [])

    def test_failure_preserved_with_provenance(self):
        case_data = make_case_data()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            runner = FakeRunner(fail_on="added")
            with self.assertRaises(RuntimeError):
                ctl.run(case_data, root, deadline=lambda: False, runner=runner, plan_sha256="a" * 64)
            failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["core_implementation_id"], core.JOB_ID)
            self.assertEqual(failure["experiment_id"], ctl.EXPERIMENT_ID)
            self.assertEqual(failure["core_calls"], 2)
            self.assertEqual(failure['run_id'], root.name)
            self.assertTrue((root / "original" / "family_binary.json").is_file())
            self.assertTrue((root / "added" / "failure.json").is_file())

    def test_aggregate_output_cap_enforced(self):
        case_data = make_case_data()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            original_cap = ctl.OUTPUT_CAP_BYTES
            ctl.OUTPUT_CAP_BYTES = 1
            try:
                with self.assertRaises(ctl.CohortTransferError) as caught:
                    ctl.run(case_data, root, deadline=lambda: False, runner=FakeRunner())
                self.assertIn("OUTPUT_CAP", str(caught.exception))
            finally:
                ctl.OUTPUT_CAP_BYTES = original_cap
            self.assertTrue((root / "original").is_dir())


if __name__ == "__main__":
    unittest.main()
