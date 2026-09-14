"""Synthetic zero-provider tests for span_development_runner_v1.py.

All inputs, locks, providers and adapters are fabricated in a temporary
directory. No real dataset, feature, result, private holdout, snapshot, model,
tokenizer or provider module is read or imported; the study ``.runtime`` has no
torch. The captured "native" adapter is a plain Python fake, so these tests show
structural binary/IO/accounting behaviour and parent/worker ownership only, not
scientific run readiness. Independent review and a locked real preflight are
still required before any real execution.

Run from the repository root with the study runtime:

    development\\classifier_generalization_v2\\.runtime\\Scripts\\python.exe \
        -W error development/classifier_generalization_v2/test_span_development_runner_v1.py -v
"""
import ast
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import time
import unittest
from unittest import mock

import native_development_runner_v2 as runner
import span_capture_adapter_v1 as span
import span_development_runner_v1 as module
from test_native_capture_contract import make_case
from test_tokenizer_input_adapter import FakeTokenizer

STUDY = module.STUDY
LABELS = ("SELF", "OTHER", "NONTERMINATION", "ORDINARY")
SHORT = {"SELF": "S", "OTHER": "O", "NONTERMINATION": "N", "ORDINARY": "A"}
ALL_SOURCES = tuple(dict.fromkeys(module.SOURCE_NAMES + runner.SOURCE_NAMES))
HELD_IDS = sorted(
    {f"H{g:02d}_{p}{i:02d}" for g in range(1, 5) for p in ("S", "O", "N") for i in range(1, 13)}
    | {f"H{g:02d}_A{i:02d}" for g in (5, 6) for i in range(1, 25)}
)


# --------------------------------------------------------------------------- #
# synthetic fixture: two valid 320-forward V2 locks + one new span lock
# --------------------------------------------------------------------------- #
class SpanFixture:
    def __init__(self, directory):
        self.root = Path(directory)
        self.study = self.root / STUDY
        self.study.mkdir(parents=True)
        self.runtime = self.root / "fake_runtime"
        self.commit = "a" * 40
        source_dir = Path(module.__file__).parent
        self.sources = {}
        for name in ALL_SOURCES:
            raw = (source_dir / name).read_bytes()
            (self.study / name).write_bytes(raw)
            if name in module.SOURCE_NAMES:
                self.sources[(STUDY / name).as_posix()] = runner.sha(raw)
        self.old_sources = {
            (STUDY / name).as_posix(): runner.sha((self.study / name).read_bytes())
            for name in runner.SOURCE_NAMES
        }
        self.provider_pins = {}
        for role, (_, relative) in runner.PROVIDERS.items():
            path = self.runtime / "Lib/site-packages" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"# fabricated provider source\n")
            self.provider_pins[role] = runner.sha(path.read_bytes())
        self.groups = []
        for prefix in ("T", "X"):
            self.groups.extend(self._group_set(prefix + "T", "TRAIN", 7))
            self.groups.extend(self._group_set(prefix + "V", "VALIDATION", 4))
        self.pins = {}
        self.pins["blueprint"] = self.put("GROUP_BLUEPRINT_V4.json", {"groups": self.groups})
        self.pins["holdout_index"] = self.put(
            "holdout_custody/HOLDOUT_ADMITTED_INDEX_V99.json",
            dict(
                schema="holdout_admitted_public_index.v1",
                logical_cases=192,
                order_views=384,
                case_ids=HELD_IDS,
                class_counts={label: 48 for label in runner.LABELS},
                group_counts={"H01": 36, "H02": 36, "H03": 36, "H04": 36, "H05": 24, "H06": 24},
            ),
        )
        self.pins["model_metadata"] = self.put(
            "NATIVE_MODEL_METADATA_V1.json",
            dict(
                schema="native_model_metadata.v1",
                revision=runner.REVISION,
                checkpoint_keys={str(i): {"shape": [1]} for i in range(488)},
            ),
        )
        self.pins["snapshot_lock"] = self.put(
            "NATIVE_SNAPSHOT_LOCK_CANDIDATE_V1.json",
            dict(schema="snapshot_lock.v1", revision=runner.REVISION, scientific_execution_authorized=False),
        )
        self.old_run_ids = []
        self.old_lock_paths = []
        self.references = []
        for prefix, train_name, val_name, audit_name, run_id in (
            ("T", "TRAIN_ACCEPTED_V10.json", "VALIDATION_ACCEPTED_V3.json",
             "holdout_custody/CORPUS_AUDIT_SEAL_V1.json", "development_capture_20260914_v1"),
            ("X", "EXPANSION_TRAIN_ROOT_V2.json", "EXPANSION_VALIDATION_ROOT_V2.json",
             "EXPANSION_AUTHOR_REVIEW_V1.json", "expansion_capture_20260914_v1"),
        ):
            self._add_old_lock(prefix, train_name, val_name, audit_name, run_id)
        self.new_lock = dict(
            schema=module.SCHEMA,
            scientific_execution_authorized=True,
            run_id="span_toy_run",
            source_commit=self.commit,
            source_files=dict(self.sources),
            runtime=self.runtime_doc(),
            threads=dict(intra=1, inter=1),
            inputs={role: self.pins[role] for role in module.INPUT_ROLES},
            reference_locks=self.references,
            snapshot_cache_root=str(self.root / "fake_cache"),
            caps=dict(module.CAPS),
        )
        self.new_lock_path = self.study / "SPAN_TOY_LOCK.json"

    # -- construction helpers ------------------------------------------------ #
    def _group_set(self, prefix, split, count):
        groups = []
        for i in range(count):
            gid = f"{prefix}{i + 1:02d}"
            groups.append(
                dict(
                    group_id=gid,
                    split=split,
                    development_fold=(i % 5 if split == "TRAIN" else None),
                    mechanism_ancestry="fixture_" + gid,
                    template_ancestry="fixture_template_" + gid,
                )
            )
        return groups

    def _cases(self, groups, split, count):
        cases = []
        per = count // len(LABELS)
        for li, label in enumerate(LABELS):
            for i in range(per):
                group = groups[(li * per + i) % len(groups)]
                cases.append(
                    make_case(
                        case_id=f"{group['group_id']}_{SHORT[label]}{i + 1:02d}",
                        group_id=group["group_id"],
                        split=split,
                        class_label=label,
                        development_fold=group["development_fold"],
                        mechanism_ancestry=group["mechanism_ancestry"],
                        template_ancestry=group["template_ancestry"],
                        status=(
                            "ADMITTED_TRAIN_TEXT_ONLY"
                            if split == "TRAIN"
                            else "ADMITTED_VALIDATION_TEXT_ONLY"
                        ),
                    )
                )
        return cases

    def put(self, name, document):
        path = self.study / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = runner.encoded(document)
        path.write_bytes(raw)
        return dict(path=path.relative_to(self.root).as_posix(), sha256=runner.sha(raw))

    def runtime_doc(self):
        return dict(
            prefix=str(self.runtime),
            python=".".join(map(str, sys.version_info[:3])),
            packages={name: "fixture" for name in runner.PACKAGES},
            provider_sources=dict(self.provider_pins),
        )

    def _add_old_lock(self, prefix, train_name, val_name, audit_name, run_id):
        train_groups = [g for g in self.groups if g["group_id"].startswith(prefix + "T")]
        val_groups = [g for g in self.groups if g["group_id"].startswith(prefix + "V")]
        train_pin = self.put(train_name, dict(split="TRAIN", case_count=120,
                                              cases=self._cases(train_groups, "TRAIN", 120)))
        val_pin = self.put(val_name, dict(split="VALIDATION", case_count=40,
                                          cases=self._cases(val_groups, "VALIDATION", 40)))
        audit_pin = self.put(
            audit_name,
            dict(
                status="PASS",
                model_outcomes_read=False,
                private_text_exported=False,
                dataset_hashes={
                    "train": train_pin["sha256"],
                    "validation": val_pin["sha256"],
                    "holdout_index": self.pins["holdout_index"]["sha256"],
                },
            ),
        )
        inputs = dict(
            train=train_pin,
            validation=val_pin,
            blueprint=self.pins["blueprint"],
            holdout_index=self.pins["holdout_index"],
            model_metadata=self.pins["model_metadata"],
            snapshot_lock=self.pins["snapshot_lock"],
            corpus_audit=audit_pin,
        )
        document = dict(
            schema=runner.SCHEMA,
            scientific_execution_authorized=True,
            run_id=run_id,
            source_commit=self.commit,
            source_files=dict(self.old_sources),
            runtime=self.runtime_doc(),
            threads=dict(intra=1, inter=1),
            inputs=inputs,
            snapshot_cache_root=str(self.root / "fake_cache"),
            caps=dict(runner.CAPS),
        )
        path = self.study / f"REF_{prefix}_LOCK.json"
        raw = runner.encoded(document)
        path.write_bytes(raw)
        self.old_run_ids.append(run_id)
        self.old_lock_paths.append(path)
        self.references.append(
            dict(path=path.relative_to(self.root).as_posix(), sha256=runner.sha(raw),
                 run_id=run_id, forwards=runner.CAPS["forwards"])
        )

    # -- preflight invocation with fabricated git/runtime -------------------- #
    def save(self):
        raw = runner.encoded(self.new_lock)
        self.new_lock_path.write_bytes(raw)
        return runner.sha(raw)

    def git(self, root, *args):
        if args[0] == "rev-parse":
            return (self.commit + "\n").encode()
        if args[:2] == ("cat-file", "blob"):
            return (self.root / args[2].split(":", 1)[1]).read_bytes()
        raise AssertionError(args)

    def check(self):
        digest = self.save()
        with mock.patch.object(runner, "git_bytes", self.git), mock.patch.object(
            runner, "runtime_prefix", return_value=self.runtime
        ), mock.patch.object(importlib.metadata, "version", return_value="fixture"):
            return module.preflight(self.new_lock_path, digest, self.root)

    def rewrite_old_lock(self, index, document):
        raw = runner.encoded(document)
        self.old_lock_paths[index].write_bytes(raw)
        self.references[index]["sha256"] = runner.sha(raw)


# --------------------------------------------------------------------------- #
# capture fixture: fake tokenizer + fake span adapter, no providers
# --------------------------------------------------------------------------- #
class FakeSpanAdapter:
    def __init__(self, blocks=module.BLOCKS, mutate=None):
        self.tokenizer = FakeTokenizer()
        self.blocks = tuple(blocks)
        self.calls = []
        self.mutate = mutate
        self.provenance = {
            "job_id": "synthetic_fake",
            "span_window_schema": span.WINDOW_SCHEMA,
            "blocks": list(self.blocks),
            "native_model_provenance_verified": False,
            "injected_providers_are_not_authenticated": True,
        }

    def encode(self, text, *, add_special_tokens):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        return self.tokenizer.decode(
            ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def capture_window(self, input_ids, readout_index, final_input_index):
        self.calls.append((list(input_ids), readout_index, final_input_index))
        prefix = readout_index + 1
        length = min(module.WINDOW, prefix)
        start = prefix - length
        positions = list(range(start, prefix))
        layers = {}
        for block in self.blocks:
            matrix = [[float((block + i + j) % 17) for j in range(module.WIDTH)] for i in range(length)]
            layers[block] = dict(
                block=block, shape=[length, module.WIDTH], dtype="float32",
                row_bytes=module.WIDTH * 4, bytes=length * module.WIDTH * 4,
                position_indices=list(positions), matrix=matrix,
            )
        result = dict(
            schema=span.WINDOW_SCHEMA, blocks=list(self.blocks),
            readout_index=readout_index, final_input_index=final_input_index,
            shared_prefix_length=prefix, window_start=start, window_length=length,
            position_indices=list(positions), width=module.WIDTH, dtype="float32",
            layers=layers, hook_calls={block: 1 for block in self.blocks},
            all_positions_unchanged=True, parameters_unchanged=True,
            one_forward_per_capture_call=True,
        )
        if self.mutate is not None:
            self.mutate(result)
        return result


def make_capture_ctx(root, cases, *, raw_bytes=None, caps=None, run_id="toy_span_run"):
    caps = dict(caps or module.CAPS)
    if raw_bytes is not None:
        caps["raw_bytes"] = raw_bytes
    return dict(
        lock=dict(run_id=run_id, caps=caps, threads=dict(intra=1, inter=1)),
        lock_sha256="c" * 64,
        cases=cases,
        root=root,
        output=root / "capture" / run_id,
    )


def fake_verify(ctx):
    return dict(snapshot_realpath=str(ctx["root"] / "fake_snapshot"),
                aggregate_tokenizer_identity_sha256="a" * 64)


def case_pair():
    return [
        make_case(case_id="C01_S01", group_id="C01", class_label="SELF"),
        make_case(case_id="C02_O01", group_id="C02", class_label="OTHER"),
    ]


# --------------------------------------------------------------------------- #
# preflight tests
# --------------------------------------------------------------------------- #
class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="span_runner_")
        self.addCleanup(self.temporary.cleanup)
        self.fixture = SpanFixture(self.temporary.name)

    def test_combines_two_old_320_locks_into_320_cases_without_provider_import(self):
        before = set(sys.modules)
        ctx = self.fixture.check()
        self.assertEqual(len(ctx["cases"]), 320)
        self.assertEqual(sum(c["split"] == "TRAIN" for c in ctx["cases"]), 240)
        self.assertEqual(sum(c["split"] == "VALIDATION" for c in ctx["cases"]), 80)
        self.assertEqual(len({c["case_id"] for c in ctx["cases"]}), 320)
        self.assertEqual(len(ctx["reference_contexts"]), 2)
        self.assertEqual(set(ctx["provider_paths"]), set(runner.PROVIDERS))
        imported = {name.split(".")[0] for name in set(sys.modules) - before}
        self.assertFalse(imported & runner.PACKAGES)

    def test_bad_caps_old_sha_old_run_id_and_old_caps_rejected(self):
        f = self.fixture
        f.new_lock["caps"]["forwards"] = 641
        with self.assertRaisesRegex(runner.GateError, "CAPS"):
            f.check()
        f = SpanFixture(Path(self.temporary.name) / "b")
        f.new_lock["reference_locks"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(runner.GateError, "REFERENCE_LOCK_DIGEST"):
            f.check()
        f = SpanFixture(Path(self.temporary.name) / "c")
        f.new_lock["run_id"] = f.old_run_ids[0]
        with self.assertRaisesRegex(runner.GateError, "OLD_RUN_ID_REUSE"):
            f.check()
        f = SpanFixture(Path(self.temporary.name) / "d")
        old = json.loads(f.old_lock_paths[0].read_bytes())
        old["caps"]["forwards"] = 319
        f.rewrite_old_lock(0, old)
        with self.assertRaisesRegex(runner.GateError, "REFERENCE_LOCK_CAPS"):
            f.check()
        f = SpanFixture(Path(self.temporary.name) / "e")
        f.new_lock["reference_locks"][1]["forwards"] = 319
        with self.assertRaisesRegex(runner.GateError, "REFERENCE_LOCK_FORWARDS"):
            f.check()

    def test_exact_source_set_and_authorization_required(self):
        f = self.fixture
        f.new_lock["source_files"].pop(next(iter(f.sources)))
        with self.assertRaisesRegex(runner.GateError, "SOURCE_SET"):
            f.check()
        f = SpanFixture(Path(self.temporary.name) / "auth")
        f.new_lock["scientific_execution_authorized"] = False
        with self.assertRaisesRegex(runner.GateError, "LOCK_NOT_AUTHORIZED"):
            f.check()


# --------------------------------------------------------------------------- #
# capture / binary IO tests
# --------------------------------------------------------------------------- #
class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="span_capture_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_capture(self, cases, adapter, *, raw_bytes=None):
        ctx = make_capture_ctx(self.root, cases, raw_bytes=raw_bytes)
        built = []

        def build(inner, snapshot_dir):
            built.append(snapshot_dir)
            return adapter

        pins = module.capture(ctx, time.monotonic(), verify=fake_verify, build=build)
        return ctx, pins, built

    def test_streams_binary_index_and_receipt_without_json_float_arrays(self):
        adapter = FakeSpanAdapter()
        cases = case_pair()
        ctx, pins, built = self.run_capture(cases, adapter)
        self.assertEqual(built, [str(self.root / "fake_snapshot")])
        self.assertEqual(set(pins), set(module.ARTIFACTS))
        self.assertEqual(len(adapter.calls), len(cases) * 2)
        windows = (ctx["output"] / "windows.f32").read_bytes()
        index = json.loads((ctx["output"] / "index.json").read_bytes())
        self.assertNotIn("matrix", json.dumps(index))
        self.assertNotIn("values", json.dumps(index))
        self.assertEqual(index["schema"], module.INDEX_SCHEMA)
        self.assertEqual(index["record_order"], "case_order x orders(AB,BA) x blocks(6,10,18)")
        self.assertEqual(index["case_order"], [case["case_id"] for case in cases])
        self.assertEqual(len(index["records"]), len(cases) * 2 * len(module.BLOCKS))
        self.assertEqual(index["forward_count"], len(cases) * 2)
        self.assertEqual(index["raw_bytes"], len(windows))
        offset = 0
        for record in index["records"]:
            self.assertEqual(record["offset"], offset)
            self.assertEqual(record["length"], record["shape"][0] * record["shape"][1] * 4)
            self.assertEqual(record["sha256"], hashlib.sha256(windows[offset:offset + record["length"]]).hexdigest())
            self.assertNotEqual(record["token_positions"], [])
            offset += record["length"]
        self.assertEqual(offset, len(windows))
        self.assertEqual(pins["windows.f32"], dict(bytes=len(windows), sha256=hashlib.sha256(windows).hexdigest()))
        values = struct.unpack("<%df" % (index["records"][0]["length"] // 4), windows[: index["records"][0]["length"]])
        self.assertTrue(all(isinstance(value, float) for value in values))
        receipt = json.loads((ctx["output"] / "capture_receipt.json").read_bytes())
        self.assertEqual(receipt["counters"]["fits"], 0)
        self.assertEqual(receipt["counters"]["model_loads"], 1)
        self.assertIs(receipt["notes"]["json_float_arrays_written"], False)

    def test_window_metadata_and_forward_accounting_rejected(self):
        for mutate, code in (
            (lambda r: r.__setitem__("window_length", 3), "WINDOW_LENGTH"),
            (lambda r: r.__setitem__("hook_calls", {6: 1, 10: 1, 18: 2}), "WINDOW_HOOKS"),
            (lambda r: r.__setitem__("position_indices", [9]), "WINDOW_POSITIONS"),
            (lambda r: r["layers"][6].__setitem__("bytes", 1), "WINDOW_LAYER_BYTES"),
            (lambda r: r["layers"][10].__setitem__("matrix", [[]]), "WINDOW_MATRIX"),
        ):
            with self.subTest(code=code):
                root = Path(tempfile.mkdtemp(prefix="span_reject_"))
                self.addCleanup(lambda path=root: shutil.rmtree(path, ignore_errors=True))
                adapter = FakeSpanAdapter(mutate=mutate)
                ctx = make_capture_ctx(root, case_pair())
                with self.assertRaisesRegex(runner.GateError, code):
                    module.capture(ctx, time.monotonic(), verify=fake_verify, build=lambda c, p: adapter)

    def test_raw_cap_and_exclusive_output_enforced(self):
        root = Path(tempfile.mkdtemp(prefix="span_cap_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        adapter = FakeSpanAdapter()
        ctx = make_capture_ctx(root, case_pair(), raw_bytes=1000)
        with self.assertRaisesRegex(runner.GateError, "RAW_CAP"):
            module.capture(ctx, time.monotonic(), verify=fake_verify, build=lambda c, p: adapter)
        adapter = FakeSpanAdapter()
        ctx = make_capture_ctx(root / "second", case_pair())
        module.capture(ctx, time.monotonic(), verify=fake_verify, build=lambda c, p: adapter)
        with self.assertRaises(FileExistsError):
            module.capture(ctx, time.monotonic(), verify=fake_verify, build=lambda c, p: adapter)

    def test_failure_preserves_partial_binary_and_no_index(self):
        class Failing(FakeSpanAdapter):
            def capture_window(self, input_ids, readout_index, final_input_index):
                if len(self.calls) == 1:
                    raise RuntimeError("synthetic forward failure")
                return super().capture_window(input_ids, readout_index, final_input_index)

        root = Path(tempfile.mkdtemp(prefix="span_fail_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        ctx = make_capture_ctx(root, case_pair())
        with self.assertRaises(RuntimeError):
            module.capture(ctx, time.monotonic(), verify=fake_verify, build=lambda c, p: Failing())
        self.assertTrue((ctx["output"] / "windows.f32").is_file())
        self.assertFalse((ctx["output"] / "index.json").exists())


# --------------------------------------------------------------------------- #
# worker / supervisor ownership tests (synthetic, zero providers)
# --------------------------------------------------------------------------- #
def worker_ctx(fixture):
    return dict(
        lock=dict(run_id="toy_span_run", caps=dict(module.CAPS)),
        lock_sha256="d" * 64,
        cases=case_pair(),
        root=fixture.root,
        base=fixture.root / STUDY / "runs",
        output=fixture.root / STUDY / "runs" / "toy_span_run",
        marker=fixture.root / STUDY / "runs" / "native_model_owner.json",
    )


def write_fake_artifacts(ctx):
    output = ctx["output"]
    output.mkdir(parents=True, exist_ok=True)
    pins = {}
    for name, raw in {
        "windows.f32": b"\x00\x00\x80?",
        "index.json": runner.encoded({"schema": module.INDEX_SCHEMA}),
        "capture_receipt.json": runner.encoded({"actual_pid": os.getpid(), "model_calls": 0}),
    }.items():
        (output / name).write_bytes(raw)
        pins[name] = dict(bytes=len(raw), sha256=runner.sha(raw))
    return pins


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="span_owner_")
        self.addCleanup(self.temporary.cleanup)
        self.fixture = SpanFixture(self.temporary.name)

    def test_worker_exclusive_outputs_and_marker_removal(self):
        ctx = worker_ctx(self.fixture)
        with mock.patch.object(module, "preflight", return_value=ctx), mock.patch.object(
            module, "capture", side_effect=lambda c, s: write_fake_artifacts(c)
        ):
            report = module.worker("lock", ctx["lock_sha256"], "a" * 32)
            self.assertEqual(report["pid"], os.getpid())
            self.assertFalse(ctx["marker"].exists())
            self.assertFalse((ctx["output"] / "supervisor_success.json").exists())
            with self.assertRaises(FileExistsError):
                module.worker("lock", ctx["lock_sha256"], "b" * 32)
            self.assertEqual((ctx["output"] / "windows.f32").read_bytes(), b"\x00\x00\x80?")

    def test_supervisor_verifies_all_pins_before_success_and_preserves_failure(self):
        ctx = worker_ctx(self.fixture)

        def fake_watch(command, cwd, seconds):
            self.assertIn("--worker", command)
            pins = write_fake_artifacts(ctx)
            token = command[command.index("--token") + 1]
            report = dict(status="worker_complete", pid=456, token=token,
                          run_id=ctx["lock"]["run_id"], lock_sha256=ctx["lock_sha256"], outputs=pins)
            (ctx["output"] / "worker_complete.json").write_bytes(runner.encoded(report))
            return 456, b""

        with mock.patch.object(module, "preflight", return_value=ctx), mock.patch.object(
            runner, "watch", side_effect=fake_watch
        ), mock.patch.object(module, "capture", side_effect=AssertionError("parent must not capture")):
            receipt = module.supervise("lock", ctx["lock_sha256"])
        self.assertEqual(receipt["pid"], 456)
        self.assertTrue((ctx["output"] / "supervisor_success.json").is_file())

    def test_supervisor_rejects_tampered_pin_and_missing_artifacts(self):
        ctx = worker_ctx(self.fixture)

        def bad_watch(command, cwd, seconds):
            pins = write_fake_artifacts(ctx)
            pins["index.json"]["sha256"] = "0" * 64
            token = command[command.index("--token") + 1]
            report = dict(status="worker_complete", pid=457, token=token,
                          run_id=ctx["lock"]["run_id"], lock_sha256=ctx["lock_sha256"], outputs=pins)
            (ctx["output"] / "worker_complete.json").write_bytes(runner.encoded(report))
            return 457, b""

        with mock.patch.object(module, "preflight", return_value=ctx), mock.patch.object(
            runner, "watch", side_effect=bad_watch
        ):
            with self.assertRaisesRegex(runner.GateError, "ARTIFACT_HASH"):
                module.supervise("lock", ctx["lock_sha256"])
        self.assertFalse((ctx["output"] / "supervisor_success.json").exists())
        self.assertTrue(list(ctx["base"].glob("controller_failure_*.json")))


# --------------------------------------------------------------------------- #
# real toy child + module hygiene
# --------------------------------------------------------------------------- #
class RealToyChildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="span_toy_")
        self.addCleanup(self.temporary.cleanup)
        self.fixture = SpanFixture(self.temporary.name)

    def test_real_watched_toy_worker_path(self):
        ctx = worker_ctx(self.fixture)
        ctx["output"] = self.fixture.root / STUDY / "runs" / "toy_span_run"
        source = Path(module.__file__).parent
        toy_context = {key: str(ctx[key]) for key in ("base", "output", "marker")}
        original_launch = runner.launch

        def toy_launch(command, stream, cwd):
            self.assertIn("--worker", command)
            script = (
                "import hashlib, os, sys; from pathlib import Path; "
                f"sys.path.insert(0, {str(source)!r}); "
                "import span_development_runner_v1 as r; "
                f"c = {{k: Path(v) for k, v in {toy_context!r}.items()}}; "
                f"c['lock'] = {{'run_id': {ctx['lock']['run_id']!r}}}; "
                "r.preflight = lambda *a, **k: c\n"
                "def cap(context, started):\n"
                "    out = context['output']; out.mkdir(parents=True, exist_ok=True); pins = {}\n"
                "    for name, raw in {'windows.f32': b'abc', 'index.json': b'{}', "
                "'capture_receipt.json': r.runner.encoded({'actual_pid': os.getpid()})}.items():\n"
                "        (out / name).write_bytes(raw); "
                "pins[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}\n"
                "    return pins\n"
                "r.capture = cap; "
                f"raise SystemExit(r.main({command[2:]!r}))"
            )
            return original_launch([sys.executable, "-c", script], stream, cwd)

        expected = ctx["lock_sha256"]
        with mock.patch.object(module, "preflight", return_value=ctx), mock.patch.object(
            runner, "launch", side_effect=toy_launch
        ):
            receipt = module.supervise(self.fixture.new_lock_path, expected)
        actual = json.loads((ctx["output"] / "capture_receipt.json").read_bytes())
        self.assertEqual(actual["actual_pid"], receipt["pid"])
        self.assertNotEqual(actual["actual_pid"], os.getpid())
        self.assertFalse(ctx["marker"].exists())
        self.assertTrue((ctx["output"] / "supervisor_success.json").is_file())


class ModuleHygieneTests(unittest.TestCase):
    def test_no_real_provider_imports_and_documented_binary_contract(self):
        tree = ast.parse(inspect.getsource(module))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                roots.add((node.module or "").split(".")[0])
        self.assertFalse(roots & {"torch", "transformers", "tokenizers", "safetensors"})
        self.assertIn("windows.f32", module.__doc__)
        self.assertEqual(module.JOB_ID, "span_runner_implementation_20260914_1059")
        self.assertEqual(module.CAPS["forwards"], 640)
        self.assertEqual(module.CAPS["seconds"], 1800)
        self.assertEqual(module.CAPS["raw_bytes"], 125829120)
        self.assertEqual(module.CAPS["output_bytes"], 201326592)


if __name__ == "__main__":
    unittest.main()
