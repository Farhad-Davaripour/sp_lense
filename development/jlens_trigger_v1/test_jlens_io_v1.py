"""Synthetic tests for ``jlens_io_v1.py`` (job jlens_loader_parity_20260914_v1).

Everything here is fabricated in a temporary directory. No lens, model,
checkpoint, tokenizer, cache, dataset or private holdout file is read, and no
provider (``torch``/``transformers``/``tokenizers``/``safetensors``/
``transformer_lens``) is imported. The published-lens loader is exercised with an
injected fake ``loader``; safetensors slicing is exercised on tiny synthetic files
with the module's dimension constants patched down so a real 248320-row matrix is
never involved.

Run with the repository virtualenv:

    .venv\\Scripts\\python.exe -m pytest development/jlens_trigger_v1/test_jlens_io_v1.py -q
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import jlens_io_v1 as io


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@contextlib.contextmanager
def small_dims(d_model=4, vocab=8):
    with mock.patch.object(io, "D_MODEL", d_model), mock.patch.object(io, "VOCAB_SIZE", vocab):
        yield d_model, vocab


def write_file(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return dict(path=str(path), bytes=len(raw), sha256=sha(raw))


def bf16_bytes(values):
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()


def make_safetensors(path, tensors):
    """tensors: name -> (dtype, shape, raw bytes)."""
    header, blob, offset = {}, b"", 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blob += raw
    encoded = json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)
    return dict(path=str(path), bytes=8 + len(encoded) + len(blob), sha256=sha(Path(path).read_bytes()))


def make_lens(path):
    raw = b"PK\x03\x04" + b"synthetic-lens-body"
    return dict(write_file(path, raw), revision=io.LENS_PIN["revision"],
                filename=io.LENS_PIN["filename"])


# --------------------------------------------------------------------------- #
# cache fixture: 240 TRAIN + 80 validation cases, 640 views, blocks 6/10/18
# --------------------------------------------------------------------------- #
class CacheFixture:
    def __init__(self, directory, d_model=4, window=2):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.d_model = d_model
        self.window = window
        self.cases = []
        for index in range(io.CACHE_CASES):
            split = "TRAIN" if index < io.ADMITTED_TRAIN else "VALIDATION"
            self.cases.append(("C%03d" % index, split))
        self.admitted = {case_id for case_id, _ in self.cases}
        self.held = {"H%03d" % index for index in range(8)}
        self.index_path = self.root / "index.json"
        self.windows_path = self.root / "windows.f32"
        self.document = self.build_document()

    def build_document(self, mutate=None):
        records, offset, order = [], 0, 0
        for case_id, split in self.cases:
            for view in io.CACHE_ORDERS:
                for block in io.BLOCKS:
                    length = self.window * self.d_model * io.FLOAT_BYTES
                    start = 5
                    positions = list(range(start, start + self.window))
                    record = {
                        "case_id": case_id,
                        "split": split,
                        "order": view,
                        "block": block,
                        "shape": [self.window, self.d_model],
                        "dtype": "float32",
                        "byte_order": "little",
                        "float_bytes": io.FLOAT_BYTES,
                        "token_positions": positions,
                        "prefix_length": positions[-1] + 1,
                        "readout_index": positions[-1],
                        "offset": offset,
                        "length": length,
                        "sha256": "0" * 64,
                    }
                    records.append(record)
                    offset += length
                    order += 1
        document = {
            "schema": io.CACHE_INDEX_SCHEMA,
            "condition": "synthetic_condition_v1",
            "blocks": list(io.BLOCKS),
            "dtype": "float32",
            "byte_order": "little",
            "float_bytes": io.FLOAT_BYTES,
            "width": self.d_model,
            "case_count": io.CACHE_CASES,
            "view_count": io.CACHE_VIEWS,
            "forward_count": io.CACHE_VIEWS,
            "case_order": [case_id for case_id, _ in self.cases],
            "raw_bytes": offset,
            "records": records,
        }
        if mutate is not None:
            mutate(document)
        return document

    def windows(self, document=None):
        document = self.document if document is None else document
        blob = bytearray()
        for record in document["records"]:
            # Write exactly the declared byte length so record-level shape/alignment
            # mutations are detected by the record checks, not by a raw-bytes mismatch.
            values = np.full(record["length"] // io.FLOAT_BYTES, 0.5, dtype="<f4").tobytes()
            record["sha256"] = sha(values)
            blob += values
        return bytes(blob)

    def write(self, document=None):
        self.document = document if document is not None else self.document
        blob = self.windows(self.document)
        self.windows_path.write_bytes(blob)
        self.index_path.write_bytes(io.runner.encoded(self.document))
        self.index_pin = dict(bytes=self.index_path.stat().st_size, sha256=sha(self.index_path.read_bytes()))
        self.windows_pin = dict(bytes=len(blob), sha256=sha(blob))
        return self

    def inspect(self, **kwargs):
        return io.inspect_cache(
            self.index_path, self.index_pin, self.windows_path, self.windows_pin,
            admitted_ids=self.admitted, held_ids=self.held, **kwargs)


# --------------------------------------------------------------------------- #
# lens pin / header / loader
# --------------------------------------------------------------------------- #
class LensTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jlens_io_lens_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_pin_size_hash_missing_and_revision_rejected(self):
        pin = make_lens(self.root / "lens.pt")
        io.inspect_lens(pin["path"], pin)
        bad = dict(pin)
        bad["bytes"] += 1
        with self.assertRaisesRegex(io.GateError, "PIN_SIZE"):
            io.inspect_lens(bad["path"], bad)
        bad = dict(pin, sha256="0" * 64)
        with self.assertRaisesRegex(io.GateError, "PIN_HASH"):
            io.inspect_lens(bad["path"], bad)
        with self.assertRaisesRegex(io.GateError, "PIN_FILE_TYPE"):
            io.inspect_lens(self.root / "missing.pt", pin)
        bad = dict(pin, revision="0" * 40)
        with self.assertRaisesRegex(io.GateError, "LENS_REVISION"):
            io.inspect_lens(pin["path"], bad)

    def test_unsafe_pickle_only_container_rejected(self):
        pin = dict(write_file(self.root / "unsafe.pt", b"\x80\x02synthetic-pickle"),
                   revision=io.LENS_PIN["revision"], filename=io.LENS_PIN["filename"])
        with self.assertRaisesRegex(io.GateError, "LENS_UNSAFE_FORMAT"):
            io.inspect_lens(pin["path"], pin)

    def test_load_requires_release_and_explicit_weights_only_loader(self):
        pin = make_lens(self.root / "lens.pt")
        payload = {"J": {6: np.zeros((4, 4), np.float32), 10: np.ones((4, 4), np.float32)}, "d_model": 4}
        with small_dims():
            with self.assertRaisesRegex(io.GateError, "RELEASE_NOT_AUTHORIZED"):
                io.load_lens_jacobians(pin["path"], pin, release="wrong", layers=(6, 10), loader=lambda p: payload)
            with self.assertRaisesRegex(io.GateError, "LENS_LOADER_REQUIRED"):
                io.load_lens_jacobians(pin["path"], pin, release=io.RELEASE_V1, layers=(6, 10))

    def test_load_only_requested_layers_as_float32(self):
        pin = make_lens(self.root / "lens.pt")
        payload = {
            "J": {
                0: np.ones((4, 4), np.float16),
                6: np.full((4, 4), 2.0, np.float16),
                10: np.full((4, 4), 3.0, np.float32),
                18: np.full((4, 4), 4.0, np.float32),
            },
            "d_model": 4,
            "source_layers": [0, 6, 10, 18],
        }
        with small_dims():
            selected = io.load_lens_jacobians(
                pin["path"], pin, release=io.RELEASE_V1, layers=(6, 10, 18), loader=lambda p: payload)
        self.assertEqual(set(selected), {6, 10, 18})
        self.assertTrue(all(array.dtype == np.float32 for array in selected.values()))
        self.assertTrue(np.all(selected[6] == np.float32(2.0)))
        self.assertTrue(np.all(selected[18] == np.float32(4.0)))

    def test_missing_layer_and_wrong_shape_and_d_model_rejected(self):
        pin = make_lens(self.root / "lens.pt")
        with small_dims():
            with self.assertRaisesRegex(io.GateError, "LENS_LAYER_MISSING"):
                io.load_lens_jacobians(
                    pin["path"], pin, release=io.RELEASE_V1, layers=(6, 10),
                    loader=lambda p: {"J": {6: np.zeros((4, 4), np.float32)}, "d_model": 4})
            with self.assertRaisesRegex(io.GateError, "LENS_J_SHAPE"):
                io.load_lens_jacobians(
                    pin["path"], pin, release=io.RELEASE_V1, layers=(6,),
                    loader=lambda p: {"J": {6: np.zeros((3, 3), np.float32)}, "d_model": 4})
            with self.assertRaisesRegex(io.GateError, "LENS_D_MODEL"):
                io.load_lens_jacobians(
                    pin["path"], pin, release=io.RELEASE_V1, layers=(6,),
                    loader=lambda p: {"J": {6: np.zeros((4, 4), np.float32)}, "d_model": 99})


# --------------------------------------------------------------------------- #
# safetensors slicing
# --------------------------------------------------------------------------- #
class SafetensorsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jlens_io_st_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def model_file(self, d_model=4, vocab=8, norm=None, rows=None, norm_dtype="BF16", embed_dtype="BF16"):
        norm = np.asarray([1.0, -2.0, 0.5, 4.0][:d_model], dtype=np.float32) if norm is None else norm
        rows = np.arange(vocab * d_model, dtype=np.float32).reshape(vocab, d_model) if rows is None else rows
        tensors = {}
        if norm_dtype == "F32":
            tensors[io.NORM_KEY] = ("F32", list(norm.shape), norm.astype("<f4").tobytes())
        else:
            tensors[io.NORM_KEY] = ("BF16", list(norm.shape), bf16_bytes(norm))
        if embed_dtype == "BF16":
            tensors[io.UNEMBED_KEY] = ("BF16", list(rows.shape), bf16_bytes(rows))
        else:
            tensors[io.UNEMBED_KEY] = ("F32", list(rows.shape), rows.astype("<f4").tobytes())
        return make_safetensors(self.root / "model.safetensors", tensors)

    def test_norm_bf16_storage_loaded_as_float32(self):
        pin = self.model_file()
        with small_dims():
            info = io.inspect_model_tensors(pin["path"], pin)
            self.assertEqual(info["norm_dtype"], "BF16")
            weight = io.load_norm_weight(pin["path"], pin, release=io.RELEASE_V1)
        self.assertEqual(weight.dtype, np.float32)
        self.assertEqual(weight.shape, (4,))
        self.assertTrue(np.allclose(weight, np.asarray([1.0, -2.0, 0.5, 4.0], np.float32)))

    def test_release_gate_and_shape_rejected(self):
        pin = self.model_file()
        with small_dims():
            with self.assertRaisesRegex(io.GateError, "RELEASE_NOT_AUTHORIZED"):
                io.load_norm_weight(pin["path"], pin, release=None)
        bad = self.model_file(norm=np.arange(5, dtype=np.float32))
        with small_dims():
            with self.assertRaisesRegex(io.GateError, "NORM_SHAPE"):
                io.inspect_model_tensors(bad["path"], bad)

    def test_truncated_header_rejected(self):
        path = self.root / "truncated.safetensors"
        path.write_bytes(struct.pack("<Q", 4096) + b"{}")
        with self.assertRaisesRegex(io.GateError, "SAFETENSORS_TRUNCATED"):
            io.read_safetensors_header(path)

    def test_unembed_rows_read_only_selected_rows(self):
        pin = self.model_file()
        source = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
        with small_dims():
            rows = io.load_unembed_rows(pin["path"], pin, [0, 3, 7], release=io.RELEASE_V1)
        self.assertEqual(rows.shape, (3, 4))
        self.assertTrue(np.array_equal(rows, source[[0, 3, 7]]))

    def test_surface_budget_and_token_range_rejected(self):
        pin = self.model_file()
        with small_dims():
            with self.assertRaisesRegex(io.GateError, "SURFACE_BUDGET"):
                io.load_unembed_rows(pin["path"], pin, list(range(io.MAX_SURFACES + 1)), release=io.RELEASE_V1)
            with self.assertRaisesRegex(io.GateError, "TOKEN_ID_RANGE"):
                io.load_unembed_rows(pin["path"], pin, [8], release=io.RELEASE_V1)

    def test_inspect_does_not_read_tensor_body(self):
        with small_dims():
            # A header whose data bytes are shorter than declared is still valid at
            # inspect time (no body read); only load/slice would detect it.
            header = {
                io.NORM_KEY: {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
                io.UNEMBED_KEY: {"dtype": "F32", "shape": [8, 4], "data_offsets": [16, 144]},
            }
            encoded = json.dumps(header).encode()
            path = self.root / "head_only.safetensors"
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00" * 144)
            pin = dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
            info = io.inspect_model_tensors(path, pin)
        self.assertFalse(info["tensor_body_parsed"])
        self.assertEqual(info["unembed_shape"], [8, 4])


# --------------------------------------------------------------------------- #
# cache index alignment / holdout / lazy view
# --------------------------------------------------------------------------- #
class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jlens_io_cache_")
        self.addCleanup(self.temporary.cleanup)
        self.fixture = CacheFixture(self.temporary.name, d_model=4, window=2)

    def test_valid_index_and_lazy_view(self):
        with small_dims():
            fixture = self.fixture.write()
            index = fixture.inspect(expected_condition="synthetic_condition_v1")
        self.assertEqual(len(index.records), io.CACHE_VIEWS * len(io.BLOCKS))
        record = index.record(fixture.cases[0][0], "AB", 6)
        with small_dims():
            with self.assertRaisesRegex(io.GateError, "RELEASE_NOT_AUTHORIZED"):
                index.view(record["case_id"], "AB", 6, release=None)
            view = index.view(record["case_id"], "AB", 6, release=io.RELEASE_V1)
        self.assertEqual(view.shape, (2, 4))
        self.assertTrue(np.all(view == np.float32(0.5)))

    def test_prefix_and_layer_alignment_rejected(self):
        def break_prefix(document):
            document["records"][0]["prefix_length"] += 1

        with small_dims():
            fixture = CacheFixture(self.fixture.root / "a", d_model=4, window=2)
            fixture.write(fixture.build_document(mutate=break_prefix))
            with self.assertRaisesRegex(io.GateError, "CACHE_PREFIX_ALIGNMENT"):
                fixture.inspect()

        def break_layer(document):
            document["records"][1]["block"] = 18

        with small_dims():
            fixture = CacheFixture(self.fixture.root / "b", d_model=4, window=2)
            fixture.write(fixture.build_document(mutate=break_layer))
            with self.assertRaisesRegex(io.GateError, "CACHE_RECORD_DUPLICATE"):
                fixture.inspect()

        def unknown_block(document):
            document["records"][1]["block"] = 7

        with small_dims():
            fixture = CacheFixture(self.fixture.root / "b2", d_model=4, window=2)
            fixture.write(fixture.build_document(mutate=unknown_block))
            with self.assertRaisesRegex(io.GateError, "CACHE_BLOCK"):
                fixture.inspect()

    def test_holdout_private_overlap_and_unadmitted_rejected(self):
        def inject_held(document):
            document["case_order"][0] = "H000"

        with small_dims():
            fixture = CacheFixture(self.fixture.root / "c", d_model=4, window=2)
            fixture.write(fixture.build_document(mutate=inject_held))
            with self.assertRaisesRegex(io.GateError, "CACHE_HOLDOUT_OVERLAP|CACHE_CASE_NOT_ADMITTED"):
                fixture.inspect()

    def test_oversize_window_and_count_rejected(self):
        def oversize(document):
            document["records"][0]["shape"][0] = io.WINDOW_MAX + 1

        with small_dims():
            fixture = CacheFixture(self.fixture.root / "d", d_model=4, window=2)
            fixture.write(fixture.build_document(mutate=oversize))
            with self.assertRaisesRegex(io.GateError, "CACHE_SHAPE"):
                fixture.inspect()

        def miscount(document):
            document["view_count"] -= 1

        with small_dims():
            fixture = CacheFixture(self.fixture.root / "e", d_model=4, window=2)
            fixture.write(fixture.build_document(mutate=miscount))
            with self.assertRaisesRegex(io.GateError, "CACHE_VIEW_COUNT"):
                fixture.inspect()

    def test_tampered_window_record_rejected_by_hash(self):
        with small_dims():
            fixture = self.fixture.write()
            index = fixture.inspect()
        blob = bytearray(fixture.windows_path.read_bytes())
        blob[0] ^= 0xFF
        fixture.windows_path.write_bytes(bytes(blob))
        with small_dims():
            with self.assertRaisesRegex(io.GateError, "CACHE_RECORD_HASH"):
                index.view(fixture.cases[0][0], "AB", 6, release=io.RELEASE_V1)


class ModuleHygieneTests(unittest.TestCase):
    def test_no_provider_imports_and_pins(self):
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(io))
        roots = set()
        # Only MODULE-LEVEL imports matter: the sanctioned torch loader is a lazy,
        # release-gated import inside a function, never executed at import time.
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                roots.add((node.module or "").split(".")[0])
        self.assertFalse(roots & {"torch", "transformers", "tokenizers", "safetensors", "transformer_lens"})
        self.assertEqual(io.LENS_PIN["bytes"], 48242373)
        self.assertEqual(
            io.LENS_PIN["sha256"],
            "aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48",
        )
        self.assertEqual(io.MODEL_REVISION, "2fc06364715b967f1860aea9cf38778875588b17")
        self.assertEqual(io.D_MODEL, 1024)
        self.assertEqual(io.VOCAB_SIZE, 248320)
        self.assertEqual(io.BLOCKS, (6, 10, 18))


if __name__ == "__main__":
    unittest.main()
