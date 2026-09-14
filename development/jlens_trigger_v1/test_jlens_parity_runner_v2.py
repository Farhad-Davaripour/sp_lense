"""Synthetic zero-provider tests for ``jlens_parity_runner_v2.py``.

Every lock, source file, runtime pin, lens header, prompt, adapter and tensor in
this file is fabricated in a temporary directory. No real lens, model, tokenizer,
checkpoint, cache or holdout is read, and no provider (``torch``/
``transformers``/``tokenizers``/``safetensors``/``transformer_lens``) is
imported. The fake adapter runs no forward, so these tests prove structural
release gates, prefix/layer alignment, one-owner/one-forward budget, tolerance
reporting and fail-closed behaviour only, never scientific parity.

Run with the repository virtualenv:

    .venv\\Scripts\\python.exe -m pytest development/jlens_trigger_v1/test_jlens_parity_runner_v2.py -q
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import inspect
import io as _io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

import jlens_core_v2 as core
import jlens_io_v1 as io
import jlens_parity_runner_v2 as module


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


# --------------------------------------------------------------------------- #
# fixture: fabricated sources, runtime, inputs and parity lock
# --------------------------------------------------------------------------- #
class ParityFixture:
    def __init__(self, directory):
        self.root = Path(directory)
        self.study = self.root / module.STUDY
        self.native = self.root / module.NATIVE_STUDY
        self.commit = "a" * 40
        self.sources = {}
        for relative in module.SOURCE_PATHS:
            raw = (module.ROOT / relative).read_bytes()
            write_bytes(self.root / relative, raw)
            self.sources[relative] = sha(raw)
        self.runtime = self.root / "fake_runtime"
        self.provider_pins = {}
        for role, (_, relative) in module.PROVIDERS.items():
            path = self.runtime / "Lib/site-packages" / relative
            write_bytes(path, b"# fabricated provider source\n")
            self.provider_pins[role] = sha(path.read_bytes())

        self.cache_root = self.root / "hf_cache"
        lens = b"PK\x03\x04" + b"synthetic-published-lens"
        self.lens_path = write_bytes(self.cache_root / "lens.pt", lens)
        self.lens_pin = {
            "path": str(self.lens_path),
            "cache_root": str(self.cache_root),
            "bytes": len(lens),
            "sha256": sha(lens),
            "revision": io.LENS_PIN["revision"],
            "filename": io.LENS_PIN["filename"],
        }
        self.snapshot_pin = self.put_native(
            "NATIVE_SNAPSHOT_LOCK_CANDIDATE_V1.json",
            {"schema": "snapshot_lock.v1", "revision": io.MODEL_REVISION,
             "scientific_execution_authorized": False},
        )
        self.train_pin = self.put_native(
            "TRAIN_ACCEPTED_V10.json",
            {"split": "TRAIN", "case_count": io.ADMITTED_TRAIN,
             "cases": [{"case_id": "T%03d" % index} for index in range(io.ADMITTED_TRAIN)]},
        )
        self.prompt = {
            "case_id": "T000",
            "text": "synthetic predeclared TRAIN prompt",
            "sha256": sha(b"synthetic predeclared TRAIN prompt"),
        }
        self.inputs = {
            "lens": dict(self.lens_pin),
            "model_snapshot_lock": self.snapshot_pin,
            "train_manifest": self.train_pin,
            "prompt": dict(self.prompt),
        }
        self.lock = dict(
            schema=module.SCHEMA,
            scientific_execution_authorized=True,
            release=io.RELEASE_V1,
            run_id="jlens_parity_toy",
            source_commit=self.commit,
            source_files=dict(self.sources),
            runtime=self.runtime_doc(),
            threads=dict(intra=1, inter=1),
            inputs=self.inputs,
            snapshot_cache_root=str(self.root / "hf_cache" / "snapshot"),
            caps=dict(module.CAPS),
        )
        self.lock_path = self.study / "PARITY_TOY_LOCK.json"
        self.base_lock = json.loads(module.runner.encoded(self.lock))

    def fresh(self):
        return json.loads(module.runner.encoded(self.base_lock))

    def put_native(self, name, document):
        path = self.native / name
        raw = module.runner.encoded(document)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return dict(path=(module.NATIVE_STUDY / name).as_posix(), sha256=sha(raw))

    def runtime_doc(self):
        return dict(
            prefix=str(self.runtime),
            python=".".join(map(str, __import__("sys").version_info[:3])),
            packages={name: "fixture" for name in module.PACKAGES},
            provider_sources=dict(self.provider_pins),
        )

    def git(self, root, *args):
        if args[0] == "rev-parse":
            return (self.commit + "\n").encode()
        if args[:2] == ("cat-file", "blob"):
            return (self.root / args[2].split(":", 1)[1]).read_bytes()
        raise AssertionError(args)

    def write_lock(self, document=None):
        self.lock = document if document is not None else self.lock
        raw = module.runner.encoded(self.lock)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_bytes(raw)
        return sha(raw)

    def check(self, document=None):
        digest = self.write_lock(document)
        with mock.patch.object(module.runner, "git_bytes", self.git), mock.patch.object(
            module.runner, "runtime_prefix", return_value=self.runtime
        ), mock.patch.object(importlib.metadata, "version", return_value="fixture"):
            return module.preflight(self.lock_path, digest, self.root)


# --------------------------------------------------------------------------- #
# capture fixture: fake adapter, synthetic tensors, no providers
# --------------------------------------------------------------------------- #
def make_tensors(d_model=4, n_tokens=3):
    rng = np.random.default_rng(20260914)
    return {
        "jacobians": {layer: rng.standard_normal((d_model, d_model)).astype(np.float32) for layer in module.BLOCKS},
        "norm": rng.standard_normal(d_model).astype(np.float32),
        "rows": rng.standard_normal((n_tokens, d_model)).astype(np.float32),
    }


class FakeAdapter:
    def __init__(self, *, d_model, tensors, seq=3, selected=(1, 2, 3), hook_available=True,
                 same_model=True, forward_count=1, bridge_delta=0.0, readout_delta=0.0,
                 native_dtype="float32", bridge_dtype="float32", drop_layer=None):
        self.d_model = d_model
        self.tensors = tensors
        self.seq = seq
        self.selected = list(selected)
        self.hook_available = hook_available
        self.same_model = same_model
        self.forward_count = forward_count
        self.bridge_delta = bridge_delta
        self.readout_delta = readout_delta
        self.native_dtype = native_dtype
        self.bridge_dtype = bridge_dtype
        self.drop_layer = drop_layer
        self.calls = 0
        self.provenance = {"fake": True, "same_model_object": True}

    def encode(self, text):
        return [1, 2, 3, 4]

    def run_once(self, input_ids):
        self.calls += 1
        native, bridge, native_dtypes, bridge_dtypes = {}, {}, {}, {}
        for layer in module.BLOCKS:
            if layer == self.drop_layer:
                continue
            base = np.arange(self.seq * self.d_model, dtype=np.float32).reshape(self.seq, self.d_model)
            base = base + np.float32(layer)
            native[layer] = base
            bridge[layer] = base + np.float32(self.bridge_delta)
            native_dtypes[layer] = self.native_dtype
            bridge_dtypes[layer] = self.bridge_dtype
        return dict(
            forward_count=self.forward_count, fits=0, model_loads=1, tokenizer_loads=1,
            working_memory_gib=0.5,
            same_model_object=self.same_model, same_model_hook_available=self.hook_available,
            native_layers=native, bridge_layers=bridge,
            native_dtypes=native_dtypes, bridge_dtypes=bridge_dtypes,
            selected_token_ids=list(self.selected),
        )

    def lens_api_readout(self, layer, hidden, token_ids):
        contract = core.ReadoutContract(d_model=self.d_model, token_ids=tuple(token_ids), vocab_size=io.VOCAB_SIZE)
        value = core.raw_direct_logit(
            np.asarray(hidden, dtype=np.float32)[-1], self.tensors["jacobians"][layer],
            self.tensors["norm"], self.tensors["rows"], contract)
        return value + np.float32(self.readout_delta)


def make_capture_ctx(root):
    return dict(
        lock=dict(run_id="jlens_parity_toy", caps=dict(module.CAPS)),
        lock_sha256="d" * 64,
        root=Path(root),
        data=dict(prompt={"case_id": "T000", "text": "synthetic", "sha256": "0" * 64}),
        output=Path(root) / "out" / "jlens_parity_toy",
    )


def fake_verify(ctx):
    return {"snapshot_realpath": str(ctx["root"] / "snap"), "checked_files": []}


def run_capture(root, adapter, tensors):
    ctx = make_capture_ctx(root)
    pins = module.capture(ctx, time.monotonic(), verify=fake_verify, build=lambda c, p: adapter, tensors=tensors)
    return ctx, pins


# --------------------------------------------------------------------------- #
# real adapter contract (the fake-only gap that let the working-memory field go missing)
# --------------------------------------------------------------------------- #
class _FakeScalarTensor:
    def __init__(self, value, dtype="float32"):
        self.value = np.asarray(value, dtype=np.float32)
        self.dtype = dtype

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeComponent:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)

        class _Handle:
            def remove(self_inner):
                return None

        return _Handle()


class _FakeBlockBridge:
    def __init__(self):
        self.original_component = _FakeComponent()


class _FakeBridge:
    def __init__(self, seq, d_model):
        self.seq = seq
        self.d_model = d_model
        self.blocks = {layer: _FakeBlockBridge() for layer in module.BLOCKS}

    def run_with_cache(self, tokens, names_filter=None):
        cache = {}
        for layer, block in self.blocks.items():
            component = block.original_component
            out = np.zeros((self.seq, self.d_model), dtype=np.float32)
            for hook in component.hooks:
                hook(component, None, (_FakeScalarTensor(out),))
            cache["blocks.%d.hook_out" % layer] = _FakeScalarTensor(out)
        return None, cache


class _FakeTorch:
    long = "long"

    def tensor(self, data, dtype=None):
        return data

    def no_grad(self):
        class _Ctx:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, *args):
                return False

        return _Ctx()


class RealAdapterContractTests(unittest.TestCase):
    def test_real_run_once_reports_working_memory_and_layer_dtypes(self):
        adapter = object.__new__(module.ParityAdapter)
        adapter.torch = _FakeTorch()
        adapter.bridge = _FakeBridge(seq=3, d_model=4)
        adapter.hook_names = {layer: "blocks.%d.hook_out" % layer for layer in module.BLOCKS}
        adapter.working_memory_gib = 0.5
        adapter.selected_token_ids = lambda: [1, 2, 3]
        result = module.ParityAdapter.run_once(adapter, [1, 2, 3])
        self.assertEqual(result["working_memory_gib"], 0.5)
        self.assertEqual(result["forward_count"], 1)
        self.assertEqual(result["fits"], 0)
        self.assertEqual(set(result["native_layers"]), set(module.BLOCKS))
        self.assertEqual(set(result["bridge_layers"]), set(module.BLOCKS))
        for layer in module.BLOCKS:
            self.assertEqual(result["native_dtypes"][layer], "float32")
            self.assertEqual(result["bridge_dtypes"][layer], "float32")


class LoadTensorsContractTests(unittest.TestCase):
    def test_load_tensors_accepts_snapshot_realpath_derived_path(self):
        """The real load_tensors must accept the Path derived from snapshot_realpath."""
        import hashlib

        with tempfile.TemporaryDirectory(prefix="jlens_load_tensors_") as temporary:
            root = Path(temporary)
            snapshot = root / "snap"
            snapshot.mkdir()
            shard = snapshot / "model.safetensors-00001-of-00001.safetensors"
            shard.write_bytes(b"0123456789")
            lens = root / "lens.pt"
            lens.write_bytes(b"PK\x03\x04fake-zip")
            digest = hashlib.sha256(shard.read_bytes()).hexdigest()
            lock = {
                "release": io.RELEASE_V1,
                "snapshot_cache_root": str(root),
                "inputs": {"lens": {"path": str(lens), "cache_root": str(root)}},
            }
            proof = {
                "snapshot_realpath": str(snapshot),
                "checked_files": [{"name": shard.name, "bytes": shard.stat().st_size, "sha256": digest}],
            }
            with mock.patch.object(io, "load_lens_jacobians", return_value={6: "jacobian"}), \
                 mock.patch.object(io, "load_norm_weight", return_value=np.zeros(4, np.float32)), \
                 mock.patch.object(io, "load_unembed_rows", return_value=np.zeros((2, 4), np.float32)):
                tensors = module.load_tensors({"lock": lock}, proof, [1, 2])
        self.assertEqual(tensors["jacobians"], {6: "jacobian"})
        self.assertEqual(tensors["norm"].shape, (4,))
        self.assertEqual(tensors["rows"].shape, (2, 4))


# --------------------------------------------------------------------------- #
# preflight gates
# --------------------------------------------------------------------------- #
class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jlens_parity_pf_")
        self.addCleanup(self.temporary.cleanup)
        self.fixture = ParityFixture(self.temporary.name)

    def test_valid_lock_passes_without_provider_import_or_model_work(self):
        before = set(__import__("sys").modules)
        trips = []
        with mock.patch.object(module, "verify_snapshot", side_effect=lambda c: trips.append("verify")), \
             mock.patch.object(module, "build_parity_adapter", side_effect=lambda c, p: trips.append("build")), \
             mock.patch.object(module, "load_tensors", side_effect=lambda *a, **k: trips.append("load")):
            ctx = self.fixture.check()
        self.assertEqual(ctx["output"].name, "jlens_parity_toy")
        self.assertEqual(ctx["marker"], self.fixture.native / "runs" / module.OWNER_NAME)
        self.assertEqual(ctx["data"]["prompt"]["case_id"], "T000")
        self.assertEqual(trips, [])
        imported = {name.split(".")[0] for name in set(__import__("sys").modules) - before}
        self.assertFalse(imported & module.FORBIDDEN_ROOTS)

    def test_bad_caps_release_source_and_authorization_rejected(self):
        lock = self.fixture.fresh()
        lock["caps"]["forwards"] = 2
        with self.assertRaisesRegex(module.GateError, "CAPS"):
            self.fixture.check(lock)
        lock = self.fixture.fresh()
        lock["release"] = "wrong"
        with self.assertRaisesRegex(module.GateError, "RELEASE_NOT_AUTHORIZED"):
            self.fixture.check(lock)
        lock = self.fixture.fresh()
        lock["source_files"].pop(next(iter(lock["source_files"])))
        with self.assertRaisesRegex(module.GateError, "SOURCE_SET"):
            self.fixture.check(lock)
        lock = self.fixture.fresh()
        lock["scientific_execution_authorized"] = False
        with self.assertRaisesRegex(module.GateError, "LOCK_NOT_AUTHORIZED"):
            self.fixture.check(lock)

    def test_prompt_outcome_field_not_train_and_bad_hash_rejected(self):
        lock = self.fixture.fresh()
        lock["inputs"]["prompt"]["class_label"] = "SELF"
        with self.assertRaisesRegex(module.GateError, "PROMPT_SCHEMA"):
            self.fixture.check(lock)
        lock = self.fixture.fresh()
        lock["inputs"]["prompt"]["case_id"] = "H999"
        with self.assertRaisesRegex(module.GateError, "PROMPT_NOT_TRAIN"):
            self.fixture.check(lock)
        lock = self.fixture.fresh()
        lock["inputs"]["prompt"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(module.GateError, "PROMPT_HASH"):
            self.fixture.check(lock)

    def test_lens_pin_revision_and_caps_are_enforced(self):
        lock = self.fixture.fresh()
        lock["inputs"]["lens"]["revision"] = "0" * 40
        with self.assertRaisesRegex(module.GateError, "LENS_REVISION"):
            self.fixture.check(lock)
        lock = self.fixture.fresh()
        lock["inputs"]["lens"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(module.GateError, "PIN_HASH"):
            self.fixture.check(lock)


# --------------------------------------------------------------------------- #
# capture alignment / budget / tolerance / fail-closed
# --------------------------------------------------------------------------- #
class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jlens_parity_cap_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tensors = make_tensors(d_model=4)

    def capture_with(self, adapter):
        with mock.patch.object(io, "D_MODEL", 4), mock.patch.object(io, "VOCAB_SIZE", 8):
            return run_capture(self.root, adapter, self.tensors)

    def test_happy_path_writes_receipt_with_one_forward_and_zero_residual(self):
        adapter = FakeAdapter(d_model=4, tensors=self.tensors)
        with mock.patch.object(io, "D_MODEL", 4), mock.patch.object(io, "VOCAB_SIZE", 8):
            ctx, pins = run_capture(self.root, adapter, self.tensors)
        self.assertEqual(adapter.calls, 1)
        self.assertEqual(set(pins), set(module.ARTIFACTS))
        receipt = json.loads((ctx["output"] / "parity_receipt.json").read_bytes())
        self.assertEqual(receipt["schema"], module.RECEIPT_SCHEMA)
        self.assertEqual(receipt["counters"], {"model_loads": 1, "tokenizer_loads": 1, "forwards": 1, "fits": 0, "derivatives": 0})
        self.assertEqual(receipt["hook_max_abs"], {"6": 0.0, "10": 0.0, "18": 0.0})
        self.assertEqual(receipt["readout_max_abs"], {"6": 0.0, "10": 0.0, "18": 0.0})
        self.assertEqual(receipt["selected_token_ids"], [1, 2, 3])
        self.assertIs(receipt["notes"]["one_forward"], True)
        self.assertIs(receipt["notes"]["outcome_used"], False)

    def test_layer_alignment_and_hook_tolerance_and_dtype(self):
        with self.assertRaisesRegex(module.GateError, "LAYER_ALIGNMENT"):
            self.capture_with(FakeAdapter(d_model=4, tensors=self.tensors, drop_layer=18))
        with self.assertRaisesRegex(module.GateError, "HOOK_TOLERANCE"):
            self.capture_with(FakeAdapter(d_model=4, tensors=self.tensors, bridge_delta=0.5))
        with self.assertRaisesRegex(module.GateError, "HOOK_DTYPE"):
            self.capture_with(FakeAdapter(d_model=4, tensors=self.tensors, native_dtype="bfloat16"))
        with self.assertRaisesRegex(module.GateError, "HOOK_SHAPE"):
            self.capture_with(FakeAdapter(d_model=3, tensors=self.tensors))

    def test_second_model_and_forward_budget_enforced(self):
        with self.assertRaisesRegex(module.GateError, "SECOND_MODEL"):
            self.capture_with(FakeAdapter(d_model=4, tensors=self.tensors, same_model=False))
        with self.assertRaisesRegex(module.GateError, "FORWARD_BUDGET"):
            self.capture_with(FakeAdapter(d_model=4, tensors=self.tensors, forward_count=2))

    def test_working_memory_cap_enforced(self):
        adapter = FakeAdapter(d_model=4, tensors=self.tensors)
        original = module.CAPS["working_memory_gib"]
        module.CAPS["working_memory_gib"] = 0.1
        try:
            with mock.patch.object(io, "D_MODEL", 4), mock.patch.object(io, "VOCAB_SIZE", 8):
                ctx = make_capture_ctx(self.root)
                with self.assertRaisesRegex(module.GateError, "WORKING_MEMORY"):
                    module.capture(ctx, time.monotonic(), verify=fake_verify,
                                   build=lambda c, p: adapter, tensors=self.tensors)
        finally:
            module.CAPS["working_memory_gib"] = original

    def test_same_model_hook_unavailable_fails_closed(self):
        with self.assertRaisesRegex(module.GateError, "SAME_MODEL_HOOK_UNAVAILABLE"):
            self.capture_with(FakeAdapter(d_model=4, tensors=self.tensors, hook_available=False))
        self.assertIn("fail closed", module.MINIMAL_ALTERNATIVE)
        self.assertIn("same forward", module.MINIMAL_ALTERNATIVE)

    def test_readout_tolerance_reported(self):
        with self.assertRaisesRegex(module.GateError, "READOUT_TOLERANCE"):
            self.capture_with(FakeAdapter(d_model=4, tensors=self.tensors, readout_delta=0.5))

    def test_injected_tensors_are_the_only_tensor_path(self):
        # load_tensors is never called when synthetic tensors are supplied.
        adapter = FakeAdapter(d_model=4, tensors=self.tensors)
        with mock.patch.object(module, "load_tensors", side_effect=AssertionError("real load forbidden")):
            with mock.patch.object(io, "D_MODEL", 4), mock.patch.object(io, "VOCAB_SIZE", 8):
                run_capture(self.root, adapter, self.tensors)


# --------------------------------------------------------------------------- #
# worker / supervisor ownership (shared native owner lock)
# --------------------------------------------------------------------------- #
def write_fake_artifacts(ctx):
    output = ctx["output"]
    output.mkdir(parents=True, exist_ok=True)
    raw = module.runner.encoded({"schema": module.RECEIPT_SCHEMA, "actual_pid": os.getpid()})
    (output / "parity_receipt.json").write_bytes(raw)
    return {"parity_receipt.json": {"bytes": len(raw), "sha256": sha(raw)}}


def worker_ctx(fixture):
    return dict(
        lock=dict(run_id="jlens_parity_toy", caps=dict(module.CAPS)),
        lock_sha256="e" * 64,
        root=fixture.root,
        base=fixture.study / "runs",
        output=fixture.study / "runs" / "jlens_parity_toy",
        marker=fixture.native / "runs" / module.OWNER_NAME,
    )


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jlens_parity_own_")
        self.addCleanup(self.temporary.cleanup)
        self.fixture = ParityFixture(self.temporary.name)

    def test_worker_exclusive_output_and_shared_marker_removed(self):
        ctx = worker_ctx(self.fixture)
        with mock.patch.object(module, "preflight", return_value=ctx), mock.patch.object(
            module, "capture", side_effect=lambda c, s: write_fake_artifacts(c)
        ):
            report = module.worker("lock", ctx["lock_sha256"], "a" * 32)
            self.assertEqual(report["pid"], os.getpid())
            self.assertFalse(ctx["marker"].exists())
            self.assertFalse((ctx["output"] / module.SUCCESS_RECEIPT).exists())
            with self.assertRaises(FileExistsError):
                module.worker("lock", ctx["lock_sha256"], "b" * 32)
        self.assertEqual(ctx["marker"], self.fixture.native / "runs" / module.OWNER_NAME)

    def test_supervisor_verifies_all_pins_before_success(self):
        ctx = worker_ctx(self.fixture)

        def fake_watch(command, cwd, seconds):
            self.assertIn("--worker", command)
            pins = write_fake_artifacts(ctx)
            token = command[command.index("--token") + 1]
            report = dict(status="worker_complete", pid=456, token=token,
                          run_id=ctx["lock"]["run_id"], lock_sha256=ctx["lock_sha256"], outputs=pins)
            (ctx["output"] / module.WORKER_RECEIPT).write_bytes(module.runner.encoded(report))
            return 456, b""

        with mock.patch.object(module, "preflight", return_value=ctx), mock.patch.object(
            module.runner, "watch", side_effect=fake_watch
        ), mock.patch.object(module, "capture", side_effect=AssertionError("parent must not capture")):
            receipt = module.supervise("lock", ctx["lock_sha256"])
        self.assertEqual(receipt["pid"], 456)
        self.assertTrue((ctx["output"] / module.SUCCESS_RECEIPT).is_file())

    def test_supervisor_rejects_tampered_pin_and_preserves_failure(self):
        ctx = worker_ctx(self.fixture)

        def bad_watch(command, cwd, seconds):
            pins = write_fake_artifacts(ctx)
            pins["parity_receipt.json"]["sha256"] = "0" * 64
            token = command[command.index("--token") + 1]
            report = dict(status="worker_complete", pid=457, token=token,
                          run_id=ctx["lock"]["run_id"], lock_sha256=ctx["lock_sha256"], outputs=pins)
            (ctx["output"] / module.WORKER_RECEIPT).write_bytes(module.runner.encoded(report))
            return 457, b""

        with mock.patch.object(module, "preflight", return_value=ctx), mock.patch.object(
            module.runner, "watch", side_effect=bad_watch
        ):
            with self.assertRaisesRegex(module.GateError, "ARTIFACT_HASH"):
                module.supervise("lock", ctx["lock_sha256"])
        self.assertFalse((ctx["output"] / module.SUCCESS_RECEIPT).exists())
        self.assertTrue(list(ctx["base"].glob("controller_failure_*.json")))


# --------------------------------------------------------------------------- #
# CLI default is preflight-only; module hygiene
# --------------------------------------------------------------------------- #
class CliAndHygieneTests(unittest.TestCase):
    def test_default_cli_is_preflight_only(self):
        captured = _io.StringIO()
        with mock.patch.object(module, "preflight", return_value={"lock": {"run_id": "toy"}}), \
             mock.patch.object(module.sys, "stdout", captured):
            code = module.main(["--lock", "L", "--sha256", "S"])
        self.assertEqual(code, 0)
        report = json.loads(captured.getvalue())
        self.assertEqual(report["status"], "preflight_pass")
        self.assertIs(report["native_execution_performed"], False)
        self.assertEqual(report["forwards"], 0)

    def test_module_is_model_free_and_caps_are_finite(self):
        tree = ast.parse(inspect.getsource(module))
        roots = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                roots.add((node.module or "").split(".")[0])
        self.assertFalse(roots & module.FORBIDDEN_ROOTS)
        self.assertEqual(module.JOB_ID, "jlens_parity_root_v2")
        self.assertEqual(module.CAPS["seconds"], 600)
        self.assertEqual(module.CAPS["forwards"], 1)
        self.assertEqual(module.CAPS["model_loads"], 1)
        self.assertEqual(module.CAPS["tokenizer_loads"], 1)
        self.assertEqual(module.CAPS["fits"], 0)
        self.assertEqual(module.CAPS["output_bytes"], 67108864)
        self.assertEqual(module.CAPS["working_memory_gib"], 6)
        self.assertIs(module.CAPS["generation"], False)
        self.assertIs(module.CAPS["logit_behavior_study"], False)
        self.assertIs(module.CAPS["steering"], False)
        self.assertEqual(module.BLOCKS, (6, 10, 18))


if __name__ == "__main__":
    unittest.main()
