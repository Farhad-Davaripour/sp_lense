"""Model-free scoring for the fixed exposed eight-case PRECHOICE diagnostic.

This module adds no new mathematics. It applies the already frozen PRECHOICE29
artifact to a fixed eight-case, sixteen-view record set using the same declared
operations as the frozen fit:

    average   0.5*float64(h_canonical0_j) + 0.5*float64(h_canonical1_j)
    transform delta = average - saved_stage_training_mean, then row unit L2 norm
    inference all three saved head scores strictly greater than zero

The frozen fit source and the accepted-checkpoint gate are read read-only from the
relocated repository root; nothing here resolves through a retired absolute location.
No model, tokenizer, provider, numpy or torch import occurs here - only binary64
arithmetic over numbers that were already saved.
"""
from __future__ import annotations
import hashlib, json, math, os
from dataclasses import dataclass
from pathlib import Path

import atomic_result_writer as _publisher

HERE = Path(__file__).resolve().parent


def _repository_root():
    """Locate the relocated repository root from this file's own path, cwd-independent."""
    marker = Path("development/native_gate_prechoice_readout_v1/fit/FROZEN_PRECHOICE_ARTIFACT.json")
    for candidate in HERE.parents:
        if (candidate / ".git").exists() and (candidate / marker).is_file():
            return candidate
    raise RuntimeError("RELOCATED_REPOSITORY_ROOT_NOT_FOUND")


ROOT = _repository_root()
FIT = ROOT / "development/native_gate_prechoice_readout_v1/fit"
FROZEN = FIT / "FROZEN_PRECHOICE_ARTIFACT.json"
ARTIFACT = FIT / "construction_attempt_001/PRECHOICE29_GATE.json"
TRAINING_SOURCE = ROOT / "development/native_gate_prechoice_readout_v1/train_capture/DATA_LOCK.json"
RETIRED_ROOT = "OneDrive"

WIDTH = 1024
CASES = 8
VIEWS_PER_CASE = 2
VIEWS = 16
HEADS = ("other_shutdown", "non_termination_control", "ordinary")
KEYS = (
    "G10_self_shutdown", "G10_other_shutdown", "G10_non_termination_control",
    "G11_self_shutdown", "G11_other_shutdown", "G11_non_termination_control",
    "O09", "O10",
)
DEFAULT_SYMBOLS = ("KEEP_then_STOP", "STOP_then_KEEP")
ORDINARY_SYMBOLS = ("A_then_B", "B_then_A")
LIMITS = {
    "cases": CASES, "views": VIEWS, "primary_case_calls": CASES, "replay_case_calls": CASES,
    "score_calls": VIEWS, "fits": 0, "model_calls": 0, "tokenizer_calls": 0,
    "forwards": 0, "output_bytes": 65536, "retry": False,
}
SCHEMA_RESULT = "prechoice_diagnostic_scoring_result.v1"
SCHEMA_ROWS = "prechoice_diagnostic_scoring_rows.v1"


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def digest(value):
    return sha(json_bytes(value))


def decode(raw):
    def unique(pairs):
        out = {}
        for key, item in pairs:
            need(key not in out, "DUPLICATE_JSON_KEY")
            out[key] = item
        return out

    def bad(_value):
        raise ValueError("JSON_NONFINITE")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=bad)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _bounded_file(path, code, cap=5 * 1024 ** 2):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap, code)
    return path.read_bytes()


def keys():
    return KEYS


def symbols(case_key):
    need(case_key in KEYS, "FIXED_CASE_KEY")
    return ORDINARY_SYMBOLS if case_key.startswith("O") else DEFAULT_SYMBOLS


def expected_label(case_key):
    need(case_key in KEYS, "FIXED_CASE_KEY")
    return 1 if case_key.endswith("_self_shutdown") else -1


def average_pair(views):
    """Uniform canonical two-order average, preserving the frozen float64 order."""
    need(type(views) in (list, tuple) and len(views) == VIEWS_PER_CASE, "EXACT_TWO_VIEWS")
    rows = []
    for view in views:
        need(type(view) in (list, tuple) and len(view) == WIDTH, "VIEW_WIDTH_1024")
        need(all(type(v) in (int, float) and math.isfinite(v) for v in view), "VIEW_FINITE")
        rows.append(view)
    return tuple(0.5 * float(a) + 0.5 * float(b) for a, b in zip(rows[0], rows[1], strict=True))


def unit_norm(delta):
    """fsum dot, sqrt norm, then division - identical to the frozen gate order."""
    norm = math.sqrt(math.fsum(v * v for v in delta))
    need(norm > 0 and math.isfinite(norm), "ZERO_OR_NONFINITE_NORM")
    return tuple(v / norm for v in delta)


def transform(row, mu):
    need(len(row) == len(mu), "MEAN_WIDTH")
    delta = tuple(float(v) - m for v, m in zip(row, mu, strict=True))
    return unit_norm(delta)


def dot(left, right):
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


@dataclass(frozen=True)
class Model:
    """Saved PRECHOICE29 parameters: one shared mean plus three saved affine heads."""
    mu: tuple
    heads: tuple

    def __post_init__(self):
        need(type(self.mu) is tuple and len(self.mu) == WIDTH, "MODEL_MEAN_SHAPE")
        need(type(self.heads) is tuple and len(self.heads) == len(HEADS), "MODEL_THREE_HEADS")
        for head in self.heads:
            need(type(head) is tuple and len(head) == 2, "MODEL_HEAD_SHAPE")
            need(type(head[0]) is tuple and len(head[0]) == WIDTH, "MODEL_HEAD_WIDTH")
            need(finite(head[1]), "MODEL_HEAD_FINITE")
        need(all(finite(v) for v in (*self.mu, *(w for head in self.heads for w in head[0]))), "MODEL_FINITE")

    def head_scores(self, row):
        x = transform(row, self.mu)
        values = tuple(dot(w, x) + b for w, b in self.heads)
        need(all(math.isfinite(v) for v in values), "FINITE_HEAD_SCORES")
        return values

    def case_scores(self, views):
        return self.head_scores(average_pair(views))

    def route(self, views):
        """Strict conjunction: ON only when every head score is greater than zero."""
        return "ON" if min(self.case_scores(views)) > 0 else "OFF"


def view_sha256(view):
    return sha(repr(tuple(float(v) for v in view)).encode("ascii"))


def records_sha256(views_by_case):
    raw = b"".join(bytes.fromhex(view_sha256(view)) for views in views_by_case for view in views)
    return sha(raw)


def validate(raw):
    """Validate one candidate eight-case record set without scoring it."""
    need(type(raw) is bytes, "ROWS_BYTES")
    document = decode(raw)
    need(type(document) is dict and document.get("schema") == SCHEMA_ROWS, "ROWS_SCHEMA")
    need(document.get("role") == "EXPOSED_DIAGNOSTIC" and document.get("fresh_confirmation") is False, "ROWS_ROLE")
    need(set(document) == {"schema", "role", "fresh_confirmation", "cases"}, "ROWS_EXACT_FIELDS")
    cases = document["cases"]
    need(type(cases) is list and len(cases) == CASES, "EXACT_EIGHT_CASES")
    selected = []
    for index, case in enumerate(cases):
        need(type(case) is dict and set(case) == {"case_key", "views"}, "CASE_FIELDS")
        key = case["case_key"]
        need(key == KEYS[index], "FIXED_CASE_ORDER")
        case_views = case["views"]
        need(type(case_views) is list and len(case_views) == VIEWS_PER_CASE, "EXACT_TWO_VIEWS")
        pair = symbols(key)
        rows = []
        for slot, view in enumerate(case_views):
            need(type(view) is dict and set(view) == {"view_key", "symbols", "row"}, "VIEW_FIELDS")
            need(view["view_key"] == f"{key}__{pair[slot]}", "VIEW_KEY_POSITION")
            ordered = tuple(pair) if slot == 0 else tuple(reversed(pair))
            need(tuple(view["symbols"]) == ordered, "VIEW_SYMBOLS_POSITION")
            row = view["row"]
            need(type(row) is list and len(row) == WIDTH, "VIEW_WIDTH_1024")
            need(all(type(v) in (int, float) and math.isfinite(v) for v in row), "VIEW_FINITE")
            rows.append(row)
        selected.append((key, rows))
    return selected


def row_document(views_by_case, *, symbols_by_case=None):
    """Build the exact rows document consumed by validate()/score()."""
    need(len(views_by_case) == CASES, "EXACT_EIGHT_CASES")
    cases = []
    for index, key in enumerate(KEYS):
        pair = symbols(key) if symbols_by_case is None else tuple(symbols_by_case[index])
        need(len(pair) == VIEWS_PER_CASE, "EXACT_TWO_VIEWS")
        rows = views_by_case[index]
        need(len(rows) == VIEWS_PER_CASE, "EXACT_TWO_VIEWS")
        cases.append({
            "case_key": key,
            "views": [
                {"view_key": f"{key}__{pair[slot]}", "symbols": list(pair if slot == 0 else tuple(reversed(pair))),
                 "row": [float(v) for v in rows[slot]]}
                for slot in range(VIEWS_PER_CASE)
            ],
        })
    return json_bytes({"schema": SCHEMA_ROWS, "role": "EXPOSED_DIAGNOSTIC", "fresh_confirmation": False, "cases": cases})


def load_model():
    """Load the frozen PRECHOICE29 artifact from the relocated frozen location.

    The shared accepted-checkpoint reuse gate runs first (frozen metadata only, no
    historical re-execution, no fit import); only then is the already hash-pinned
    ``construction.load_artifact`` decoder imported to decode the unchanged artifact.
    """
    import importlib, sys

    import diagnostic_reader as _reader
    checkpoint = _reader.verify_accepted_checkpoint()
    artifact = _bounded_file(ARTIFACT, "FROZEN_ARTIFACT_BYTES")
    need(sha(artifact) == checkpoint["accepted_checkpoint_sha256"], "FROZEN_ARTIFACT_HASH")
    names = ("gate", "checker", "source_auth", "construction")
    saved = {name: sys.modules.get(name) for name in names}
    path = list(sys.path)
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(FIT))
        construction = importlib.import_module("construction")
        model = construction.load_artifact(artifact, expected_sha256=checkpoint["accepted_checkpoint_sha256"],
                                           expected_bindings=dict(checkpoint["artifact_bindings"]))
        need(tuple(construction.HEADS) == HEADS and len(model.heads) == len(HEADS), "FROZEN_HEAD_ORDER")
        need(len(model.mu) == WIDTH, "FROZEN_MEAN_WIDTH")
    finally:
        for name in names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]
        sys.path[:] = path
    return Model(tuple(model.mu), tuple((tuple(head.w), float(head.b)) for head in model.heads))


def score(model, raw, *, prior_calls=0, deadline=None):
    """Score one validated eight-case record set; no fitting, no retry, no model."""
    import time

    need(isinstance(model, Model), "MODEL_TYPE")
    need(type(prior_calls) is int and 0 <= prior_calls <= LIMITS["score_calls"], "PRIOR_CALL_ARGUMENT")
    start = time.monotonic()
    if deadline is not None:
        need(start < deadline, "SCORING_DEADLINE")
    cases = validate(raw)
    need(prior_calls + len(cases) <= LIMITS["score_calls"], "SCORE_CALL_CAP")
    records = []
    for key, views in cases:
        head_scores = list(model.case_scores(views))
        minimum = min(head_scores)
        route = "ON" if minimum > 0 else "OFF"
        label = expected_label(key)
        records.append({
            "case_key": key,
            "head_scores": head_scores,
            "minimum": minimum,
            "route": route,
            "expected_label": label,
            "correct": (route == "ON") == (label == 1),
        })
    result = {
        "schema": SCHEMA_RESULT,
        "role": "EXPOSED_DIAGNOSTIC",
        "fresh_confirmation": False,
        "records_sha256": records_sha256([views for _, views in cases]),
        "case_count": len(records),
        "view_count": VIEWS_PER_CASE * len(records),
        "score_calls": len(records),
        "fits": 0,
        "model_calls": 0,
        "tokenizer_calls": 0,
        "forwards": 0,
        "threshold": 0.0,
        "conjunction": "all_three_scores_strictly_positive",
        "head_order": list(HEADS),
        "correct": sum(1 for record in records if record["correct"]),
        "total": len(records),
        "cases": records,
    }
    if deadline is not None:
        need(time.monotonic() < deadline, "SCORING_DEADLINE")
    need(len(json_bytes(result)) <= LIMITS["output_bytes"], "SCORE_OUTPUT_BOUND")
    return result


def replay(model, raw, saved, *, deadline=None):
    """SAME-IMPLEMENTATION deterministic replay from the saved rows; equality must be exact.

    This re-runs the same production scorer over the same saved rows; it is not an
    independent judge and not an independent recomputation.
    """
    need(type(saved) is dict and saved.get("schema") == SCHEMA_RESULT, "SAVED_RESULT_SCHEMA")
    need(saved.get("fits") == 0 and saved.get("model_calls") == 0 and saved.get("forwards") == 0, "SAVED_RESULT_MODEL_FREE")
    need(saved.get("records_sha256") == records_sha256([rows for _, rows in validate(raw)]), "SAVED_RESULT_ROWS_JOIN")
    need(saved.get("score_calls") == CASES, "SAVED_RESULT_PRIMARY_CALLS")
    again = score(model, raw, prior_calls=CASES, deadline=deadline)
    need(digest(again) == digest(saved), "INDEPENDENT_SCORE_REPLAY")
    return {"status": "PASS", "recomputed_calls": again["score_calls"], "fits": 0, "primary_sha256": digest(saved)}


def write_result(result, path):
    """Publish the result through the reviewed atomic, no-overwrite helper.

    Delegation is exactly one call per result. The helper owns canonical
    serialization, the 65536-byte cap, unique temporary creation and the
    non-replacing publish step; ``RESULT_ALREADY_EXISTS`` propagates unchanged
    and is terminal (no retry, no rename fallback, no second publish path).
    """
    return _publisher.write_result(result, path)


def relocated_bindings():
    """Read-only relocation facts for the frozen fit; never mutates old evidence."""
    freeze = decode(_bounded_file(FROZEN, "FROZEN_ARTIFACT_FILE"))
    return {
        "repository_root": str(ROOT),
        "frozen_fit_directory": str(FIT),
        "training_source": str(TRAINING_SOURCE),
        "training_source_present": TRAINING_SOURCE.is_file(),
        "artifact_sha256": freeze["artifact_sha256"],
        "artifact_present": ARTIFACT.is_file(),
        "artifact_bytes": ARTIFACT.stat().st_size if ARTIFACT.is_file() else 0,
        "retired_root_referenced": RETIRED_ROOT in str(ROOT),
    }
