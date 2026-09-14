"""Pure in-memory capture-export layer for frozen pre-option activations.

This module is a *model-free* bridge between an already captured numeric
activation vector and the decoded-view records consumed by
``pair_join.assemble_train_pairs``. It performs no file, tokenizer, model,
tensor, cache, network, fitting, centering, or HOLDOUT I/O, and it imports no
torch/transformers. Only the standard library is used for serialization;
numpy is not imported here at all (the downstream joiner owns the float64
pair mean).

What this layer does and does not claim
---------------------------------------
* It proves *serialized-byte integrity* only: that the exact 4096-byte payload
  fed in is the payload recovered, and that the SHA256 digest matches.
* It does **not** authenticate a native model, a hook, a tokenizer, a capture
  pipeline, or a real dataset. A caller can fabricate perfect bytes. The
  ``model_revision``/``block``/``position``/``dtype`` fields are structural
  echoes, not provenance evidence.
* Round-tripped records are structural: they carry numeric values plus the
  required metadata, and are accepted by the pair joiner's structural checks.

Interface
---------
* ``serialize_activation(values) -> bytes``: exactly 1024 finite numeric
  values to exactly 4096 little-endian float32 bytes.
* ``decode_activation(payload, expected_sha256) -> list[float]``: requires the
  expected digest, exactly 4096 bytes, and all-finite float32 values.
* ``build_view(...)`` / ``build_case_views(...)``: produce per-view transport
  records holding the payload, its SHA256, and sidecar metadata.
* ``decode_view(...)``: decode one transport record into a JSON-compatible
  decoded-view record (``case_id``, ``order``, ``model_revision``, ``block``,
  ``position``, ``dtype``, ``prefix_sha256``, ``values``).
* ``assemble_validation_pairs(cases, view_records, plan_groups)``: pair exactly
  AB/BA for admitted VALIDATION cases, average RAW values once in float64, and
  keep class/group/case metadata as sidecars. No fitting and no centering;
  preprocessing stays downstream and TRAIN-only.

Rejects (``CaptureExportError.code``): ``VALUES_TYPE``, ``WIDTH``,
``NONFINITE``, ``OVERFLOW``, ``PAYLOAD_TYPE``, ``PAYLOAD_LENGTH``,
``HASH_FORMAT``, ``HASH_MISMATCH``, ``REVISION``, ``ORDER``, ``RECORD``,
``CASE``, ``SPLIT``, ``FOLD``, ``PLAN``, ``DUPLICATE_CASE``,
``DUPLICATE_VIEW``, ``ORPHAN_VIEW``, ``PAIR``, ``PREFIX_MISMATCH``.
"""

from __future__ import annotations

import hashlib
import re
import struct

MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
BLOCK = 10
POSITION = "last_shared_preoption_input"
DTYPE = "float32"
WIDTH = 1024
PAYLOAD_BYTES = WIDTH * 4
ORDERS = (["A", "B"], ["B", "A"])
CAPTURE_EXPORT_SCHEMA = "capture_export_view.v1"
ALLOWED_SPLITS = frozenset(("TRAIN", "VALIDATION"))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# Largest finite binary32 value. Anything with a greater magnitude cannot be
# represented as a finite float32.
_FLOAT32_MAX = 3.4028234663852886e38


class CaptureExportError(ValueError):
    """Raised when activation bytes or paired views violate the contract."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


def _need(condition, code: str, detail: str = ""):
    if not condition:
        raise CaptureExportError(code, detail)


def _is_real_number(value) -> bool:
    return type(value) is int or type(value) is float


def _float32_exact(value, code: str):
    """Round finite Python numeric inputs to binary32 at serialization.

    Already-binary32 capture values round-trip exactly. Higher-precision
    numeric inputs follow struct's binary32 rounding; bool and overflow fail.
    """
    _need(_is_real_number(value), code, f"expected a real number, got {type(value).__name__}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise CaptureExportError('OVERFLOW') from exc
    _need(number == number and number not in (float("inf"), float("-inf")), "NONFINITE", "non-finite value")
    _need(abs(number) <= _FLOAT32_MAX, "OVERFLOW", "overflows finite float32")
    packed = struct.pack("<f", number)
    return struct.unpack("<f", packed)[0]


def serialize_activation(values) -> bytes:
    """Serialize exactly 1024 finite numeric values to 4096 little-endian f4 bytes."""
    _need(type(values) in (list, tuple), "VALUES_TYPE", f"got {type(values).__name__}")
    _need(len(values) == WIDTH, "WIDTH", f"expected {WIDTH}, got {len(values)}")
    packed = bytearray()
    for value in values:
        payload_value = _float32_exact(value, "VALUES_TYPE")
        packed += struct.pack("<f", payload_value)
    _need(len(packed) == PAYLOAD_BYTES, "PAYLOAD_LENGTH", f"got {len(packed)}")
    return bytes(packed)


def sha256_hex(payload) -> str:
    """SHA256 hex digest of an exact payload."""
    _need(type(payload) is bytes, "PAYLOAD_TYPE", f"got {type(payload).__name__}")
    return hashlib.sha256(payload).hexdigest()


def validate_sha256(value, code: str = "HASH_FORMAT"):
    _need(type(value) is str and _SHA256_RE.fullmatch(value) is not None, code)
    return value


def decode_activation(payload, expected_sha256):
    """Decode exactly one 4096-byte payload after digest and finiteness checks."""
    _need(type(payload) is bytes, "PAYLOAD_TYPE", f"got {type(payload).__name__}")
    _need(len(payload) == PAYLOAD_BYTES, "PAYLOAD_LENGTH", f"got {len(payload)}")
    validate_sha256(expected_sha256)
    _need(hashlib.sha256(payload).hexdigest() == expected_sha256, "HASH_MISMATCH")
    values = [value for (value,) in struct.iter_unpack("<f", payload)]
    _need(all(value == value and value not in (float("inf"), float("-inf")) for value in values),
          "NONFINITE", "payload contains NaN or Inf")
    return values


def _validate_order(order):
    _need(type(order) is list and order in ORDERS, "ORDER", f"got {order!r}")
    return list(order)


def build_view(case_id, order, values, prefix_sha256, revision=MODEL_REVISION) -> dict:
    """Build one transport record: payload, its digest, and structural metadata."""
    _need(type(case_id) is str and len(case_id) > 0, "RECORD", "case_id")
    _validate_order(order)
    validate_sha256(prefix_sha256)
    _need(revision == MODEL_REVISION, "REVISION", f"got {revision!r}")
    payload = serialize_activation(values)
    return {
        "schema": CAPTURE_EXPORT_SCHEMA,
        "case_id": case_id,
        "order": list(order),
        "model_revision": MODEL_REVISION,
        "block": BLOCK,
        "position": POSITION,
        "dtype": DTYPE,
        "prefix_sha256": prefix_sha256,
        "payload": payload,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "width": WIDTH,
    }


def decode_view(record) -> dict:
    """Decode a transport record into a JSON-compatible decoded-view record.

    The returned mapping contains exactly the fields ``pair_join._validate_view``
    requires and nothing else, so it can be fed straight to the joiner.
    """
    _need(type(record) is dict, "RECORD", f"got {type(record).__name__}")
    required = {
        "schema", "case_id", "order", "model_revision", "block", "position",
        "dtype", "prefix_sha256", "payload", "payload_sha256", "width",
    }
    _need(set(record) == required, "RECORD", f"unexpected keys {sorted(set(record) ^ required)}")
    _need(record["schema"] == CAPTURE_EXPORT_SCHEMA, "RECORD", "schema")
    _need(type(record["case_id"]) is str and len(record["case_id"]) > 0, "RECORD", "case_id")
    order = _validate_order(record["order"])
    _need(record["model_revision"] == MODEL_REVISION, "REVISION")
    _need(record["block"] == BLOCK, "RECORD", "block")
    _need(record["position"] == POSITION, "RECORD", "position")
    _need(record["dtype"] == DTYPE, "RECORD", "dtype")
    _need(record["width"] == WIDTH, "WIDTH", "record width")
    validate_sha256(record["prefix_sha256"])
    values = decode_activation(record["payload"], record["payload_sha256"])
    return {
        "case_id": record["case_id"],
        "order": order,
        "model_revision": MODEL_REVISION,
        "block": BLOCK,
        "position": POSITION,
        "dtype": DTYPE,
        "prefix_sha256": record["prefix_sha256"],
        "values": values,
    }


def build_case_views(case, activations, prefix_sha256, revision=MODEL_REVISION) -> list:
    """Build the AB and BA transport records for one TRAIN/VALIDATION case.

    ``activations`` must be ``{"AB": [...1024...], "BA": [...1024...]}``.
    Identical values across orders are valid, especially at the last shared
    pre-option token in a deterministic causal model. Orders identify views.
    """
    _need(type(case) is dict, "CASE", f"got {type(case).__name__}")
    _validate_case(case)
    case_id = case.get("case_id")
    _need(type(case_id) is str and len(case_id) > 0, "CASE", "case_id")
    split = case.get("split")
    _need(split in ALLOWED_SPLITS, "SPLIT", f"got {split!r}")
    fold = case.get("development_fold", None)
    if split == "TRAIN":
        _need(type(fold) is int and not isinstance(fold, bool) and 0 <= fold <= 4, "FOLD")
    else:
        _need(fold is None, "FOLD", "VALIDATION cases carry no development fold")
    _need(type(activations) is dict, "VALUES_TYPE", "activations must be a mapping")
    _need(set(activations) == {"AB", "BA"}, "VALUES_TYPE", "activations must be exactly AB/BA")
    records = [
        build_view(case_id, ["A", "B"], activations["AB"], prefix_sha256, revision),
        build_view(case_id, ["B", "A"], activations["BA"], prefix_sha256, revision),
    ]
    return records


def _validate_case(case):
    _need(type(case) is dict, "CASE", f"got {type(case).__name__}")
    for key in ("case_id", "group_id", "split", "class_label"):
        _need(key in case, "CASE", f"missing {key!r}")
        _need(type(case[key]) is str and len(case[key]) > 0, "CASE", f"{key} must be a non-empty string")
    _need(case["split"] in ALLOWED_SPLITS, "SPLIT", f"got {case['split']!r}")
    _need(case.get('status') == f"ADMITTED_{case['split']}_TEXT_ONLY", 'CASE', 'case not admitted')
    _need('development_fold' in case, 'FOLD', 'explicit fold sidecar required')
    if case["split"] == "TRAIN":
        _need(type(case.get("development_fold")) is int and not isinstance(case.get("development_fold"), bool)
              and 0 <= case["development_fold"] <= 4, "FOLD")
    else:
        _need(case.get("development_fold", None) is None, "FOLD",
              "non-TRAIN cases must not declare a development fold")
    return case


def _validate_validation_binding(case, plan_groups):
    group_id, label = case["group_id"], case["class_label"]
    _need(case["split"] == "VALIDATION", "SPLIT", "validation join admits VALIDATION only")
    _need(type(plan_groups) is dict, "PLAN", "plan_groups must be a mapping")
    _need(group_id in plan_groups, "PLAN", f"unknown group {group_id!r}")
    plan = plan_groups[group_id]
    _need(type(plan) is dict, "PLAN")
    _need(plan.get("split") == "VALIDATION", "PLAN", "group is not VALIDATION")
    _need(plan.get("development_fold", None) is None, "FOLD", "VALIDATION group carries no fold")
    counts = plan.get("case_counts")
    _need(type(counts) is dict, "PLAN", "group counts must be a mapping")
    _need(label in counts and type(counts[label]) is int and counts[label] > 0,
          "PLAN", "class not allowed by group")
    return label, group_id


def assemble_validation_pairs(cases, view_records, plan_groups) -> dict:
    """Pair exactly AB/BA decoded VALIDATION views and average RAW values once.

    Structural + role validation only. TRAIN, HOLDOUT, ``split=VAL``-style
    typos, declared folds on non-TRAIN cases, unknown/wrong-role groups,
    orphans, duplicates, partial pairs, and prefix disagreement are rejected.
    The returned ``x`` rows are float64 pair means of raw values; no centering,
    L2, or fitting occurs here (preprocessing remains downstream TRAIN-only).
    Class/group/case metadata is returned only as sidecars, and input case
    order is preserved.
    """
    _need(isinstance(cases, (list, tuple)), "CASE", "cases must be a sequence")
    _need(bool(cases), "CASE", "cases must be non-empty")
    _need(isinstance(view_records, (list, tuple)), "RECORD", "view_records must be a sequence")
    _need(type(plan_groups) is dict, "PLAN", "plan_groups must be a mapping")

    case_ids, labels, groups = [], [], []
    seen_ids = set()
    for case in cases:
        _validate_case(case)
        case_id = case["case_id"]
        _need(case_id not in seen_ids, "DUPLICATE_CASE", f"duplicate case_id {case_id!r}")
        label, group_id = _validate_validation_binding(case, plan_groups)
        seen_ids.add(case_id)
        case_ids.append(case_id)
        labels.append(label)
        groups.append(group_id)

    grouped = {}
    for record in view_records:
        decoded = decode_view(record)
        case_id = decoded["case_id"]
        _need(case_id in seen_ids, "ORPHAN_VIEW", f"unknown case_id {case_id!r}")
        order = tuple(decoded["order"])
        bucket = grouped.setdefault(case_id, {})
        _need(order not in bucket, "DUPLICATE_VIEW", f"duplicate order {order!r} for {case_id!r}")
        bucket[order] = decoded

    x = []
    for case_id in case_ids:
        bucket = grouped.get(case_id, {})
        _need(set(bucket) == {("A", "B"), ("B", "A")}, "PAIR",
              f"case {case_id!r} needs exactly one AB and one BA view")
        ab, ba = bucket[("A", "B")], bucket[("B", "A")]
        _need(ab["prefix_sha256"] == ba["prefix_sha256"], "PREFIX_MISMATCH",
              f"case {case_id!r} view prefix hashes differ")
        row = [0.5 * a + 0.5 * b for a, b in zip(ab["values"], ba["values"])]
        x.append(row)

    return {"x": x, "labels": labels, "groups": groups, "folds": [None] * len(case_ids),
            "case_ids": case_ids}
