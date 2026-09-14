"""Strict, lazy pinning/cache loader for the J-lens loader+parity job.

Job ``jlens_loader_parity_20260914_v1``. This module is the boundary between the
accepted pure-array core (``jlens_core_v2.py``, SHA256
``1aa0f3814979f9b61bf004775d79652523b18bea0bfc591f8ac8d3ba96433903``) and any
future released tensor read.

Scope of THIS job: MODEL-FREE. Importing this module imports only the standard
library, NumPy, and the reviewed ``native_development_runner_v2`` /
``snapshot_verifier`` helpers by explicit file path. It never imports
``torch``/``transformers``/``tokenizers``/``safetensors``/``transformer_lens``,
never touches the lens, model, checkpoint or cache, and never performs a readout,
tokenization or fit. Tensor construction only happens through an explicit
released loader call on synthetic or, later, pinned real bytes.

Pins carried here (from ``STUDY_BRIEF.md`` / ``PLAN_V3.json``):

* published local lens ``neuronpedia/jacobian-lens`` revision
  ``6bb49967d3c51a12ccb5beac7146f6f5781f9d06``,
  ``qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt``,
  48242373 bytes, SHA256 ``aa26b6...c9b48``;
* native model ``Qwen/Qwen3.5-0.8B`` revision
  ``2fc06364715b967f1860aea9cf38778875588b17``, final norm
  ``model.language_model.norm.weight`` and tied unembedding
  ``model.language_model.embed_tokens.weight`` ``[248320, 1024]``.

The loader is deliberately narrow: three Jacobian matrices (layers 6/10/18), the
norm vector (1024) and at most six unembedding rows (each 1024). A whole
``248320 x 1024`` unembedding matrix is never read or materialized. safetensors
tensors are sliced by byte range from the header only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

__all__ = [
    "JOB_ID",
    "IO_SCHEMA",
    "RELEASE_V1",
    "D_MODEL",
    "VOCAB_SIZE",
    "N_LAYERS",
    "BLOCKS",
    "MAX_SURFACES",
    "ADMITTED_TRAIN",
    "ADMITTED_VALIDATION",
    "CACHE_CASES",
    "CACHE_VIEWS",
    "WINDOW_MAX",
    "NORM_KEY",
    "UNEMBED_KEY",
    "CACHE_INDEX_SCHEMA",
    "CACHE_ORDERS",
    "CACHE_SPLITS",
    "LENS_PIN",
    "MODEL_REPO",
    "MODEL_REVISION",
    "GateError",
    "runner",
    "snapshot_verifier",
    "need",
    "contained_path",
    "file_sha256",
    "strict_file",
    "inspect_lens",
    "torch_weights_only_loader",
    "load_lens_jacobians",
    "read_safetensors_header",
    "inspect_model_tensors",
    "load_norm_weight",
    "load_unembed_rows",
    "validate_token_ids",
    "inspect_cache",
    "CacheIndex",
]

JOB_ID = "jlens_loader_parity_20260914_v1"
IO_SCHEMA = "jlens_io.v1"
RELEASE_V1 = "jlens_io_release_v1"

D_MODEL = 1024
VOCAB_SIZE = 248320
N_LAYERS = 24
BLOCKS = (6, 10, 18)
MAX_SURFACES = 6
ADMITTED_TRAIN = 240
ADMITTED_VALIDATION = 80
CACHE_CASES = ADMITTED_TRAIN + ADMITTED_VALIDATION
CACHE_VIEWS = CACHE_CASES * 2
WINDOW_MAX = 16
MAX_HEADER_BYTES = 8 * 1024 * 1024
CHUNK = 1 << 20
FLOAT_BYTES = 4

NORM_KEY = "model.language_model.norm.weight"
UNEMBED_KEY = "model.language_model.embed_tokens.weight"
SAFETENSORS_DTYPES = {"F32": 4, "F16": 2, "BF16": 2}

CACHE_INDEX_SCHEMA = "span_capture_index.v1"
CACHE_ORDERS = ("AB", "BA")
CACHE_SPLITS = ("TRAIN", "VALIDATION")

LENS_PIN = {
    "repo": "neuronpedia/jacobian-lens",
    "revision": "6bb49967d3c51a12ccb5beac7146f6f5781f9d06",
    "filename": "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
    "bytes": 48242373,
    "sha256": "aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48",
}
MODEL_REPO = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"

_HEX64 = re.compile(r"[0-9a-fA-F]{64}")
_HERE = Path(__file__).resolve().parent
_CLASSIFIER = (_HERE.parent / "classifier_generalization_v2").resolve()
_SIBLINGS = {
    "native_development_runner_v2": "1bfde09c4d4ee124df4f5ddfa231cd8ee2a822b2fd9ebeb9383ac5124f1df92a",
    "snapshot_verifier": "710c16f39b2fb7aa818d5b6ee5d33632555e690d745997fa0fb0f51ad134875a",
}


def _load_sibling(name):
    """Load one reviewed classifier helper by explicit path, never by sys.path."""
    if name in sys.modules:
        return sys.modules[name]
    path = _CLASSIFIER / (name + ".py")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _SIBLINGS[name]:
        raise ImportError("SIBLING_SOURCE_HASH: %s" % name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_sibling("native_development_runner_v2")
snapshot_verifier = _load_sibling("snapshot_verifier")
GateError = runner.GateError


def need(condition, code, detail=""):
    if not condition:
        raise GateError(code if not detail else "%s: %s" % (code, detail))


# --------------------------------------------------------------------------- #
# path / byte checks
# --------------------------------------------------------------------------- #
def contained_path(path, allowed_root):
    """Resolve ``path`` with ``realpath`` and require it beneath ``allowed_root``."""
    need(type(path) is str and path, "PATH_TYPE")
    need(type(allowed_root) is str and allowed_root, "ROOT_TYPE")
    resolved = os.path.realpath(path)
    root = os.path.realpath(allowed_root)
    need(os.path.isdir(root), "ROOT_NOT_DIRECTORY")
    need(os.path.commonpath([resolved, root]) == root, "PATH_ESCAPE")
    return Path(resolved)


def file_sha256(path, *, chunk=CHUNK):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def strict_file(path, pin, *, full_hash=True):
    """Strict path/type/size/hash check against one pin (no tensor parse)."""
    need(type(pin) is dict and set(pin) >= {"bytes", "sha256"}, "PIN_SCHEMA")
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "PIN_FILE_TYPE")
    size = path.stat().st_size
    need(type(pin["bytes"]) is int and not isinstance(pin["bytes"], bool), "PIN_SIZE_TYPE")
    need(size == pin["bytes"], "PIN_SIZE")
    need(type(pin["sha256"]) is str and _HEX64.fullmatch(pin["sha256"]) is not None, "PIN_HASH_FORMAT")
    if full_hash:
        need(file_sha256(path) == pin["sha256"].lower(), "PIN_HASH")
    return size


def _read_head(path, count):
    with Path(path).open("rb") as stream:
        return stream.read(count)


# --------------------------------------------------------------------------- #
# published lens (.pt)
# --------------------------------------------------------------------------- #
def inspect_lens(path, pin=None, *, full_hash=True):
    """Validate the lens pin and header only. Never unpickles or loads a tensor.

    The artifact is a ``torch.save`` archive; the only accepted container is a
    ZIP (``PK\\x03\\x04``). A legacy pickle-only file is rejected here so a later
    ``torch.load(..., weights_only=True)`` cannot be reached through an unsafe
    format.
    """
    pin = dict(LENS_PIN if pin is None else pin)
    need(pin.get("revision") == LENS_PIN["revision"], "LENS_REVISION")
    need(pin.get("filename") == LENS_PIN["filename"], "LENS_FILENAME")
    size = strict_file(path, pin, full_hash=full_hash)
    head = _read_head(path, 4)
    need(head[:2] == b"PK" and head[2:4] == b"\x03\x04", "LENS_UNSAFE_FORMAT")
    return {
        "schema": IO_SCHEMA,
        "format": "torch_zip_weights_only",
        "revision": pin["revision"],
        "filename": pin["filename"],
        "bytes": size,
        "sha256": pin["sha256"].lower(),
        "container_parsed": False,
        "tensor_body_parsed": False,
    }


def torch_weights_only_loader(path):
    """The only sanctioned real lens loader: ``torch.load(weights_only=True)``.

    Never call ``torch.load`` without ``weights_only=True``. This helper is not
    reachable from :func:`load_lens_jacobians` unless the caller passes it
    explicitly, and it is never imported by this module.
    """
    import torch  # local, release-gated, never at module import

    return torch.load(str(path), map_location="cpu", weights_only=True)


def load_lens_jacobians(path, pin, *, release, layers=BLOCKS, loader=None):
    """Load exactly the requested lens layers as float32 ``(1024, 1024)`` arrays.

    ``release`` must be the released token; ``loader`` must be supplied
    explicitly (the parity driver passes :func:`torch_weights_only_loader`).
    Layers other than ``layers`` are dropped and never returned; the whole
    ``[d_vocab, d_model]`` unembedding is never involved.
    """
    need(release == RELEASE_V1, "RELEASE_NOT_AUTHORIZED")
    inspect_lens(path, pin, full_hash=True)
    need(callable(loader), "LENS_LOADER_REQUIRED")
    payload = loader(str(path))
    need(type(payload) is dict and "J" in payload, "LENS_PAYLOAD")
    need(payload.get("d_model") == D_MODEL, "LENS_D_MODEL")
    matrices = payload["J"]
    need(type(matrices) is dict, "LENS_J")
    selected = {}
    for layer in layers:
        need(layer in matrices, "LENS_LAYER_MISSING")
        matrix = matrices[layer]
        need(tuple(getattr(matrix, "shape", ())) == (D_MODEL, D_MODEL), "LENS_J_SHAPE")
        array = np.asarray(matrix, dtype=np.float32)
        need(array.shape == (D_MODEL, D_MODEL), "LENS_J_SHAPE")
        need(np.isfinite(array).all(), "LENS_J_NONFINITE")
        selected[int(layer)] = array
    return selected


# --------------------------------------------------------------------------- #
# safetensors slicing (norm vector + selected unembedding rows only)
# --------------------------------------------------------------------------- #
class SafetensorsHeader:
    def __init__(self, entries, data_start, size):
        self.entries = entries
        self.data_start = data_start
        self.size = size

    def entry(self, name):
        need(name in self.entries, "SAFETENSORS_KEY")
        entry = self.entries[name]
        need(type(entry) is dict, "SAFETENSORS_ENTRY")
        need(entry.get("dtype") in SAFETENSORS_DTYPES, "SAFETENSORS_DTYPE")
        shape = entry.get("shape")
        need(type(shape) is list and all(type(v) is int and v > 0 for v in shape), "SAFETENSORS_SHAPE")
        offsets = entry.get("data_offsets")
        need(
            type(offsets) is list
            and len(offsets) == 2
            and all(type(v) is int for v in offsets)
            and 0 <= offsets[0] <= offsets[1] <= self.size - self.data_start,
            "SAFETENSORS_OFFSETS",
        )
        return entry


def read_safetensors_header(path):
    """Parse only the safetensors header; data bytes are left untouched."""
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "SAFETENSORS_FILE_TYPE")
    size = path.stat().st_size
    need(size >= 8, "SAFETENSORS_TRUNCATED")
    prefix = _read_head(path, 8)
    (header_bytes,) = struct.unpack("<Q", prefix)
    need(0 < header_bytes <= MAX_HEADER_BYTES, "SAFETENSORS_HEADER_SIZE")
    need(8 + header_bytes <= size, "SAFETENSORS_TRUNCATED")
    with path.open("rb") as stream:
        stream.seek(8)
        raw = stream.read(header_bytes)
    need(len(raw) == header_bytes, "SAFETENSORS_TRUNCATED")
    try:
        document = runner.strict_json(raw)
    except Exception as exc:  # noqa: BLE001 - normalize any parse failure
        raise GateError("SAFETENSORS_HEADER_JSON: %s" % type(exc).__name__)
    need(type(document) is dict, "SAFETENSORS_HEADER")
    entries = {k: v for k, v in document.items() if k != "__metadata__"}
    return SafetensorsHeader(entries, 8 + header_bytes, size)


def inspect_model_tensors(path, pin, *, names=(NORM_KEY, UNEMBED_KEY), full_hash=False):
    """Validate model-file pin/header/dtypes/shapes without reading tensor data."""
    size = strict_file(path, pin, full_hash=full_hash)
    header = read_safetensors_header(path)
    norm = header.entry(NORM_KEY)
    need(norm["shape"] == [D_MODEL], "NORM_SHAPE")
    need(norm["dtype"] in SAFETENSORS_DTYPES, "NORM_DTYPE")
    embed = header.entry(UNEMBED_KEY)
    need(embed["shape"] == [VOCAB_SIZE, D_MODEL], "UNEMBED_SHAPE")
    need(embed["dtype"] in SAFETENSORS_DTYPES, "UNEMBED_DTYPE")
    need(set(names) >= {NORM_KEY, UNEMBED_KEY}, "TENSOR_NAMES")
    return {
        "schema": IO_SCHEMA,
        "bytes": size,
        "sha256": pin["sha256"].lower(),
        "norm_dtype": norm["dtype"],
        "norm_shape": list(norm["shape"]),
        "unembed_dtype": embed["dtype"],
        "unembed_shape": list(embed["shape"]),
        "tensor_body_parsed": False,
    }


def _to_float32(raw, dtype, shape):
    need(dtype in SAFETENSORS_DTYPES, "SAFETENSORS_DTYPE")
    if dtype == "F32":
        array = np.frombuffer(raw, dtype="<f4")
    elif dtype == "F16":
        array = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    else:  # BF16 is the top 16 bits of F32; zero-pad, then reinterpret.
        words = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        array = words.view(np.float32)
    need(array.size == int(np.prod(shape)), "SAFETENSORS_BYTES")
    return np.array(array.reshape(shape), dtype=np.float32, copy=True)


def _read_tensor(path, header, name):
    entry = header.entry(name)
    count = int(np.prod(entry["shape"]))
    length = count * SAFETENSORS_DTYPES[entry["dtype"]]
    with Path(path).open("rb") as stream:
        stream.seek(header.data_start + entry["data_offsets"][0])
        raw = stream.read(length)
    need(len(raw) == length, "SAFETENSORS_TRUNCATED")
    return _to_float32(raw, entry["dtype"], entry["shape"])


def load_norm_weight(path, pin, *, release, full_hash=True):
    """Return the final-norm weight as float32 ``(1024,)`` (storage may be bf16)."""
    need(release == RELEASE_V1, "RELEASE_NOT_AUTHORIZED")
    strict_file(path, pin, full_hash=full_hash)
    header = read_safetensors_header(path)
    array = _read_tensor(path, header, NORM_KEY)
    need(array.shape == (D_MODEL,), "NORM_SHAPE")
    need(np.isfinite(array).all(), "NORM_NONFINITE")
    return array


def validate_token_ids(token_ids):
    need(type(token_ids) in (list, tuple) and len(token_ids) > 0, "TOKEN_IDS")
    need(len(token_ids) <= MAX_SURFACES, "SURFACE_BUDGET")
    seen = set()
    for token_id in token_ids:
        need(type(token_id) is int and not isinstance(token_id, bool), "TOKEN_ID_TYPE")
        need(0 <= token_id < VOCAB_SIZE, "TOKEN_ID_RANGE")
        need(token_id not in seen, "TOKEN_ID_DUPLICATE")
        seen.add(token_id)
    return [int(token_id) for token_id in token_ids]


def load_unembed_rows(path, pin, token_ids, *, release, full_hash=True):
    """Read only the selected unembedding rows as float32 ``(n, 1024)``.

    The file is seeked per row; a full ``248320 x 1024`` matrix is never read.
    """
    need(release == RELEASE_V1, "RELEASE_NOT_AUTHORIZED")
    ids = validate_token_ids(token_ids)
    strict_file(path, pin, full_hash=full_hash)
    header = read_safetensors_header(path)
    entry = header.entry(UNEMBED_KEY)
    need(entry["shape"] == [VOCAB_SIZE, D_MODEL], "UNEMBED_SHAPE")
    itemsize = SAFETENSORS_DTYPES[entry["dtype"]]
    row_bytes = D_MODEL * itemsize
    base = header.data_start + entry["data_offsets"][0]
    rows = []
    with Path(path).open("rb") as stream:
        for token_id in ids:
            stream.seek(base + token_id * row_bytes)
            raw = stream.read(row_bytes)
            need(len(raw) == row_bytes, "SAFETENSORS_TRUNCATED")
            rows.append(_to_float32(raw, entry["dtype"], (D_MODEL,)))
    array = np.stack(rows).astype(np.float32, copy=False)
    need(np.isfinite(array).all(), "UNEMBED_NONFINITE")
    return array


# --------------------------------------------------------------------------- #
# existing cached views (prompted / unprompted)
# --------------------------------------------------------------------------- #
class CacheIndex:
    """Validated, lazily readable index over one existing cached-view run."""

    def __init__(self, index_path, windows_path, document, size):
        self.index_path = Path(index_path)
        self.windows_path = Path(windows_path)
        self.document = document
        self.size = size
        self.records = {}
        for record in document["records"]:
            self.records[(record["case_id"], record["order"], record["block"])] = record

    def record(self, case_id, order, block):
        key = (case_id, order, block)
        need(key in self.records, "CACHE_RECORD_MISSING")
        return self.records[key]

    def view(self, case_id, order, block, *, release, reader=None):
        """Read one cached block view, lazily, only when released."""
        need(release == RELEASE_V1, "RELEASE_NOT_AUTHORIZED")
        record = self.record(case_id, order, block)
        if reader is None:
            with self.windows_path.open("rb") as stream:
                stream.seek(record["offset"])
                raw = stream.read(record["length"])
        else:
            raw = reader(self.windows_path, record["offset"], record["length"])
        need(len(raw) == record["length"], "CACHE_RECORD_TRUNCATED")
        need(hashlib.sha256(raw).hexdigest() == record["sha256"], "CACHE_RECORD_HASH")
        shape = (record["shape"][0], record["shape"][1])
        return _to_float32(raw, "F32", shape)


def _check_record_alignment(record, names):
    need(type(record) is dict, "CACHE_RECORD_TYPE")
    for key in ("case_id", "order", "block", "split", "shape", "token_positions",
                "prefix_length", "readout_index", "offset", "length", "sha256"):
        need(key in record, "CACHE_RECORD_FIELD: %s" % key)
    need(record["order"] in CACHE_ORDERS, "CACHE_ORDER")
    need(record["block"] in BLOCKS, "CACHE_BLOCK")
    need(record["split"] in CACHE_SPLITS, "CACHE_SPLIT")
    shape = record["shape"]
    need(
        type(shape) is list
        and len(shape) == 2
        and shape[1] == D_MODEL
        and 1 <= shape[0] <= WINDOW_MAX,
        "CACHE_SHAPE",
    )
    length = shape[0] * D_MODEL * FLOAT_BYTES
    need(record["length"] == length, "CACHE_LENGTH")
    need(record["prefix_length"] == record["readout_index"] + 1, "CACHE_PREFIX_ALIGNMENT")
    start = record["prefix_length"] - shape[0]
    need(start >= 0, "CACHE_PREFIX_ALIGNMENT")
    need(record["token_positions"] == list(range(start, record["prefix_length"])), "CACHE_POSITIONS")
    need(type(record["offset"]) is int and record["offset"] >= 0, "CACHE_OFFSET")
    need(type(record["sha256"]) is str and _HEX64.fullmatch(record["sha256"]) is not None, "CACHE_HASH_FORMAT")
    names.add(record["case_id"])


def inspect_cache(index_path, index_pin, windows_path, windows_pin, *,
                  admitted_ids, held_ids, expected_condition=None):
    """Strictly validate the existing cached-view index. No score is computed.

    ``admitted_ids`` is the admitted 240 TRAIN + 80 validation case universe and
    ``held_ids`` the sealed holdout. Private/holdout text is never read: only the
    public case identifiers are compared.
    """
    strict_file(index_path, index_pin, full_hash=True)
    windows_size = strict_file(windows_path, windows_pin, full_hash=False)
    document = runner.strict_json(Path(index_path).read_bytes())
    need(type(document) is dict, "CACHE_INDEX_SCHEMA")
    need(document.get("schema") == CACHE_INDEX_SCHEMA, "CACHE_INDEX_SCHEMA")
    if expected_condition is not None:
        need(document.get("condition") == expected_condition, "CACHE_CONDITION")
    need(document.get("blocks") == list(BLOCKS), "CACHE_BLOCKS")
    need(document.get("dtype") == "float32", "CACHE_DTYPE")
    need(document.get("byte_order") == "little", "CACHE_BYTE_ORDER")
    need(document.get("float_bytes") == FLOAT_BYTES, "CACHE_FLOAT_BYTES")
    need(document.get("width") == D_MODEL, "CACHE_WIDTH")
    need(document.get("case_count") == CACHE_CASES, "CACHE_CASE_COUNT")
    need(document.get("view_count") == CACHE_VIEWS, "CACHE_VIEW_COUNT")
    need(document.get("forward_count") == CACHE_VIEWS, "CACHE_FORWARD_COUNT")
    need(document.get("raw_bytes") == windows_size, "CACHE_RAW_BYTES")
    case_order = document.get("case_order")
    need(
        type(case_order) is list
        and len(case_order) == CACHE_CASES
        and len(set(case_order)) == CACHE_CASES,
        "CACHE_CASE_ORDER",
    )
    need(set(case_order) <= set(admitted_ids), "CACHE_CASE_NOT_ADMITTED")
    need(not (set(case_order) & set(held_ids)), "CACHE_HOLDOUT_OVERLAP")
    records = document.get("records")
    need(type(records) is list and len(records) == CACHE_VIEWS * len(BLOCKS), "CACHE_RECORD_COUNT")

    names, seen, splits = set(), set(), {"TRAIN": set(), "VALIDATION": set()}
    total, previous = 0, 0
    grouped = {}
    for record in records:
        _check_record_alignment(record, names)
        key = (record["case_id"], record["order"], record["block"])
        need(key not in seen, "CACHE_RECORD_DUPLICATE")
        seen.add(key)
        need(record["offset"] == previous, "CACHE_OFFSET_ORDER")
        previous += record["length"]
        total += record["length"]
        splits[record["split"]].add(record["case_id"])
        grouped.setdefault((record["case_id"], record["order"]), []).append(record["block"])
    need(total == document["raw_bytes"], "CACHE_RAW_BYTES")
    need(names == set(case_order), "CACHE_CASE_ORDER")
    need(len(splits["TRAIN"]) == ADMITTED_TRAIN, "CACHE_TRAIN_COUNT")
    need(len(splits["VALIDATION"]) == ADMITTED_VALIDATION, "CACHE_VALIDATION_COUNT")
    for blocks in grouped.values():
        need(sorted(blocks) == list(BLOCKS), "CACHE_LAYER_ALIGNMENT")
    return CacheIndex(index_path, windows_path, document, windows_size)
