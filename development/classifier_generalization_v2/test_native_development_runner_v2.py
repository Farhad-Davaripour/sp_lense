"""Synthetic gates + real owned toy subprocess tests; no native provider loads."""
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import native_development_runner_v2 as runner
from test_native_capture_contract import make_case


class Fixture:
    def __init__(self, directory):
        self.root = Path(directory)
        self.study = self.root / runner.STUDY
        self.study.mkdir(parents=True)
        self.runtime = self.root / "fake_runtime"
        self.commit = "a" * 40
        self.sources = {}
        for name in runner.SOURCE_NAMES:
            raw = (Path(runner.__file__).parent / name).read_bytes()
            path = self.study / name
            path.write_bytes(raw)
            self.sources[(runner.STUDY / name).as_posix()] = runner.sha(raw)
        provider_pins = {}
        for role, (_, relative) in runner.PROVIDERS.items():
            path = self.runtime / "Lib/site-packages" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"# fabricated provider source\n")
            provider_pins[role] = runner.sha(path.read_bytes())
        groups, manifests = [], {}
        for split, count, group_count in (("TRAIN", 120, 7), ("VALIDATION", 40, 4)):
            group_ids = [("T" if split == "TRAIN" else "V") + f"{i+1:02d}" for i in range(group_count)]
            for i, group in enumerate(group_ids):
                groups.append(dict(group_id=group, split=split, development_fold=i % 5 if split == "TRAIN" else None,
                                   mechanism_ancestry="fixture_" + group, template_ancestry="fixture_template_" + group))
            cases = []
            for label, prefix in (("SELF", "S"), ("OTHER", "O"), ("NONTERMINATION", "N"), ("ORDINARY", "A")):
                for i in range(count // 4):
                    group = group_ids[i % group_count]
                    metadata = next(g for g in groups if g["group_id"] == group)
                    cases.append(make_case(case_id=f"{group}_{prefix}{i+1:02d}", group_id=group, split=split,
                                           development_fold=metadata["development_fold"], class_label=label,
                                           mechanism_ancestry=metadata["mechanism_ancestry"], template_ancestry=metadata["template_ancestry"],
                                           status="ADMITTED_TRAIN_TEXT_ONLY" if split == "TRAIN" else "ADMITTED_VALIDATION_TEXT_ONLY"))
            manifests[split] = dict(split=split, case_count=count, cases=cases)
        held_ids = sorted({f"H{g:02d}_{p}{i:02d}" for g in range(1, 5) for p in ("S", "O", "N") for i in range(1, 13)} |
                          {f"H{g:02d}_A{i:02d}" for g in (5, 6) for i in range(1, 25)})
        documents = {
            "train": ("TRAIN_ACCEPTED_V10.json", manifests["TRAIN"]),
            "validation": ("VALIDATION_ACCEPTED_V3.json", manifests["VALIDATION"]),
            "blueprint": ("GROUP_BLUEPRINT_V4.json", {"groups": groups}),
            "holdout_index": ("holdout_custody/HOLDOUT_ADMITTED_INDEX_V99.json", dict(schema="holdout_admitted_public_index.v1", logical_cases=192, order_views=384,
                case_ids=held_ids, class_counts={k: 48 for k in runner.LABELS}, group_counts={"H01": 36, "H02": 36, "H03": 36, "H04": 36, "H05": 24, "H06": 24})),
            "model_metadata": ("NATIVE_MODEL_METADATA_V1.json", dict(schema="native_model_metadata.v1", revision=runner.REVISION, checkpoint_keys={str(i): {"shape": [1]} for i in range(488)})),
            "snapshot_lock": ("NATIVE_SNAPSHOT_LOCK_CANDIDATE_V1.json", dict(schema="snapshot_lock.v1", revision=runner.REVISION, scientific_execution_authorized=False)),
        }
        self.inputs = {}
        for role, (name, value) in documents.items():
            self.put(role, name, value)
        self.put("corpus_audit", "holdout_custody/CORPUS_AUDIT_SEAL_V1.json", dict(status="PASS", model_outcomes_read=False, private_text_exported=False,
                 dataset_hashes={k: self.inputs[k]["sha256"] for k in ("train", "validation", "holdout_index")}))
        self.lock = dict(schema=runner.SCHEMA, scientific_execution_authorized=True, run_id="toy_run", source_commit=self.commit,
                         source_files=self.sources, inputs=self.inputs, snapshot_cache_root=str(self.root / "fake_cache"), caps=dict(runner.CAPS), threads=dict(intra=1, inter=1),
                         runtime=dict(prefix=str(self.runtime), python=".".join(map(str, sys.version_info[:3])),
                         packages={k: "fixture" for k in runner.PACKAGES}, provider_sources=provider_pins))
        self.lock_path = self.study / "TOY_LOCK.json"

    def put(self, role, name, document):
        path = self.study / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = runner.encoded(document)
        path.write_bytes(raw)
        self.inputs[role] = dict(path=path.relative_to(self.root).as_posix(), sha256=runner.sha(raw))

    def save(self):
        raw = runner.encoded(self.lock)
        self.lock_path.write_bytes(raw)
        return runner.sha(raw)

    def git(self, root, *args):
        if args[0] == "rev-parse":
            return (self.commit + "\n").encode()
        if args[:2] == ("cat-file", "blob"):
            return (self.root / args[2].split(":", 1)[1]).read_bytes()
        raise AssertionError(args)

    def check(self):
        digest = self.save()
        with mock.patch.object(runner, "git_bytes", self.git), mock.patch.object(runner, "runtime_prefix", return_value=self.runtime), \
                mock.patch.object(importlib.metadata, "version", return_value="fixture"):
            return runner.preflight(self.lock_path, digest, self.root)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="runner_v2_")
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(self.temporary.name)

    def test_complete_synthetic_preflight_no_provider_import(self):
        prior = set(sys.modules)
        ctx = self.fixture.check()
        self.assertEqual(len(ctx["cases"]), 160)
        self.assertFalse(any(k.split(".")[0] in runner.PACKAGES for k in set(sys.modules) - prior))

    def test_digest_authorization_and_caps_reject_before_git(self):
        f = self.fixture
        expected = f.save()
        with mock.patch.object(runner, "git_bytes", side_effect=AssertionError("must not touch git")):
            with self.assertRaisesRegex(runner.GateError, "LOCK_DIGEST"):
                runner.preflight(f.lock_path, "0" * 64, f.root)
            f.lock["scientific_execution_authorized"] = False
            with self.assertRaisesRegex(runner.GateError, "LOCK_NOT_AUTHORIZED"):
                runner.preflight(f.lock_path, f.save(), f.root)
            f.lock["scientific_execution_authorized"] = True
            f.lock["caps"]["forwards"] = 321
            with self.assertRaisesRegex(runner.GateError, "CAPS"):
                runner.preflight(f.lock_path, f.save(), f.root)

    def test_missing_runtime_or_source_pins_rejected(self):
        del self.fixture.lock["source_files"][next(iter(self.fixture.sources))]
        with self.assertRaisesRegex(runner.GateError, "SOURCE_SET"):
            self.fixture.check()
        self.fixture = Fixture(Path(self.temporary.name) / "second")
        del self.fixture.lock["runtime"]["packages"]["torch"]
        with self.assertRaisesRegex(runner.GateError, "RUNTIME_PACKAGE_SET"):
            self.fixture.check()

    def test_partial_holdout_and_stale_corpus_refused(self):
        f = self.fixture
        path = f.root / f.inputs["holdout_index"]["path"]
        held = json.loads(path.read_bytes())
        held["logical_cases"] = 156
        f.put("holdout_index", "holdout_custody/HOLDOUT_ADMITTED_INDEX_V99.json", held)
        with self.assertRaisesRegex(runner.GateError, "HOLDOUT_COUNT"):
            f.check()
        held["logical_cases"] = 192
        held["note"] = "a harmless metadata change still invalidates the old audit binding"
        f.put("holdout_index", "holdout_custody/HOLDOUT_ADMITTED_INDEX_V99.json", held)
        with self.assertRaisesRegex(runner.GateError, "CORPUS_BINDING"):
            f.check()

    def test_case_fold_mismatch_and_private_path_refused(self):
        f = self.fixture
        train = json.loads((f.root / f.inputs["train"]["path"]).read_bytes())
        train["cases"][0]["development_fold"] = 4
        f.put("train", "TRAIN_ACCEPTED_V10.json", train)
        with self.assertRaisesRegex(runner.GateError, "BLUEPRINT_BINDING"):
            f.check()
        f.inputs["train"]["path"] = (runner.STUDY / "holdout_custody/private/secret.json").as_posix()
        with self.assertRaisesRegex(runner.GateError, "INPUT_SCOPE"):
            f.check()

    def test_real_git_blob_bytes_not_git_object_id(self):
        # Read-only probe against an already committed, non-sensitive file.
        root = runner.ROOT
        commit = runner.git_bytes(root, "rev-parse", "HEAD").decode().strip()
        raw = (root / ".gitignore").read_bytes()
        self.assertEqual(len(commit), 40)
        self.assertEqual(len(runner.sha(raw)), 64)
        runner.check_sources(root, commit, {".gitignore": runner.sha(raw)})
        with self.assertRaisesRegex(runner.GateError, "SOURCE_BLOB_MISMATCH"):
            runner.check_sources(root, commit, {".gitignore": "0" * 64})

    def test_real_owned_toy_child_has_different_pid(self):
        command = [sys.executable, "-c", "import os,sys,json; print(json.dumps({'pid':os.getpid(),'prefix':sys.prefix}))"]
        pid, raw = runner.watch(command, str(self.fixture.root), 5)
        self.assertNotEqual(pid, os.getpid())
        reported = json.loads(raw)
        self.assertEqual(reported["pid"], pid)
        self.assertEqual(Path(reported["prefix"]).resolve(), Path(sys.prefix).resolve())

    def test_real_toy_child_hard_timeout(self):
        processes = []
        def spawn(command, stream, cwd):
            child = runner.launch(command, stream, cwd)
            processes.append(child)
            return child
        with self.assertRaisesRegex(runner.GateError, "HARD_TIMEOUT") as caught:
            runner.watch([sys.executable, "-c", "import time; time.sleep(30)"], str(self.fixture.root), .15, spawn=spawn)
        self.assertEqual(caught.exception.owned_child_pid, processes[0].pid)
        self.assertIsNotNone(processes[0].poll())

    def test_nonzero_child_cannot_succeed(self):
        with self.assertRaisesRegex(runner.GateError, "CHILD_FAILED"):
            runner.watch([sys.executable, "-c", "raise SystemExit(7)"], str(self.fixture.root), 5)

    def test_marker_cleanup_checks_pid_token_and_run(self):
        marker = self.fixture.root / "owner.json"
        raw = dict(pid=123, token="a" * 32, run_id="toy")
        marker.write_bytes(runner.encoded(raw))
        self.assertFalse(runner.release_owner(marker, 124, raw["token"], "toy"))
        self.assertFalse(runner.release_owner(marker, 123, "b" * 32, "toy"))
        self.assertTrue(marker.exists())
        self.assertTrue(runner.release_owner(marker, 123, raw["token"], "toy"))

    def test_worker_fake_capture_preserves_exclusive_outputs(self):
        ctx = self.fixture.check()
        with mock.patch.object(runner, "preflight", return_value=ctx), mock.patch.object(runner, "capture", return_value={"features.json": b"[]", "capture_receipt.json": b"{}"}):
            result = runner.worker(self.fixture.lock_path, ctx["lock_sha256"], "a" * 32)
            self.assertEqual(result["pid"], os.getpid())
            self.assertFalse(ctx["marker"].exists())
            self.assertFalse((ctx["output"] / "supervisor_success.json").exists())
            with self.assertRaises(FileExistsError):
                runner.worker(self.fixture.lock_path, ctx["lock_sha256"], "b" * 32)
            self.assertEqual((ctx["output"] / "features.json").read_bytes(), b"[]")

    def test_parent_dispatches_worker_command_and_verifies_artifacts(self):
        ctx = self.fixture.check()
        seen = []
        def watched(command, cwd, seconds):
            seen.append(command)
            self.assertIn("--worker", command)
            self.assertNotIn("--run", command)
            token = command[command.index("--token") + 1]
            ctx["output"].mkdir()
            outputs = {}
            for name, raw in {"features.json": b"[]", "capture_receipt.json": b"{}"}.items():
                (ctx["output"] / name).write_bytes(raw)
                outputs[name] = dict(bytes=len(raw), sha256=runner.sha(raw))
            report = dict(status="worker_complete", pid=456, token=token, run_id=ctx["lock"]["run_id"], lock_sha256=ctx["lock_sha256"], outputs=outputs)
            (ctx["output"] / "worker_complete.json").write_bytes(runner.encoded(report))
            return 456, b""
        with mock.patch.object(runner, "preflight", return_value=ctx), mock.patch.object(runner, "watch", side_effect=watched), \
                mock.patch.object(runner, "capture", side_effect=AssertionError("parent must never capture")):
            receipt = runner.supervise(self.fixture.lock_path, ctx["lock_sha256"])
        self.assertEqual(len(seen), 1)
        self.assertEqual(receipt["pid"], 456)
        self.assertTrue((ctx["output"] / "supervisor_success.json").is_file())

    def test_integrated_parent_worker_toy_path_uses_real_child(self):
        ctx = self.fixture.check()
        original_launch = runner.launch
        def toy_launch(command, stream, cwd):
            self.assertIn("--worker", command)
            toy_context = {k: str(ctx[k]) for k in ("base", "output", "marker")}
            script = (
                "import sys,os,json; from pathlib import Path; "
                f"sys.path.insert(0,{str(Path(runner.__file__).parent)!r}); "
                "import native_development_runner_v2 as r; "
                f"c={{k:Path(v) for k,v in {toy_context!r}.items()}}; "
                f"c['lock']={{'run_id':{ctx['lock']['run_id']!r}}}; "
                "r.preflight=lambda *a,**k:c; "
                "r.capture=lambda *a:{'features.json':b'[]','capture_receipt.json':r.encoded({'actual_pid':os.getpid(),'model_calls':0})}; "
                f"raise SystemExit(r.main({command[2:]!r}))"
            )
            return original_launch([sys.executable, "-c", script], stream, cwd)
        with mock.patch.object(runner, "preflight", return_value=ctx), mock.patch.object(runner, "launch", side_effect=toy_launch), \
                mock.patch.object(runner, "capture", side_effect=AssertionError("must not capture in parent")):
            receipt = runner.supervise(self.fixture.lock_path, ctx["lock_sha256"])
        actual = json.loads((ctx["output"] / "capture_receipt.json").read_bytes())
        self.assertEqual(actual["actual_pid"], receipt["pid"])
        self.assertNotEqual(actual["actual_pid"], os.getpid())
        self.assertEqual(actual["model_calls"], 0)
        self.assertFalse(ctx["marker"].exists())

    def test_timeout_releases_only_matching_child_owner_marker(self):
        for foreign in (False, True):
            with self.subTest(foreign=foreign):
                f = Fixture(Path(self.temporary.name) / ("foreign" if foreign else "own"))
                ctx = f.check()
                def timed_out(command, cwd, seconds):
                    token = command[command.index("--token") + 1]
                    marker = dict(pid=8765, token="0" * 32 if foreign else token, run_id=ctx["lock"]["run_id"])
                    ctx["marker"].write_bytes(runner.encoded(marker))
                    error = runner.GateError("HARD_TIMEOUT")
                    error.owned_child_pid = 8765
                    error.owned_child_closed = True
                    raise error
                with mock.patch.object(runner, "preflight", return_value=ctx), mock.patch.object(runner, "watch", side_effect=timed_out):
                    with self.assertRaisesRegex(runner.GateError, "HARD_TIMEOUT"):
                        runner.supervise(f.lock_path, ctx["lock_sha256"])
                self.assertEqual(ctx["marker"].exists(), foreign)
                self.assertFalse((ctx["output"] / "supervisor_success.json").exists())


if __name__ == "__main__":
    unittest.main()
