"""One predetermined .20 condition; authenticated source and exposed comparison."""

from __future__ import annotations

import importlib.util
import json
import math
import sys

from scripts import frozen_guarded_preserve_crossed_plan as parent

ROOT = parent.ROOT
CONFIG = "configs/guarded_preserve_endpoint020_crossed.json"
DOC = "docs/GUARDED_PRESERVE_ENDPOINT020_CROSSED.md"
SCRIPT = "scripts/guarded_preserve_endpoint020.py"
VERIFY = "scripts/verify_guarded_preserve_endpoint020.py"
TEST = "tests/test_guarded_preserve_endpoint020.py"
PLAN = "scripts/guarded_preserve_endpoint020_plan.py"
OUTPUT = "evidence/guarded_preserve_endpoint020_crossed_f03_v1_qwen35_08b"
CONDITION = "derived_condition.json"
require, sha, read = parent.require, parent.sha, parent.read
norm, vector_sha, offset_sha, EPS = parent.norm, parent.vector_sha, parent.offset_sha, parent.EPS
authenticated, canonical_sha = parent.authenticated, parent.canonical_sha
canonical, crossed, io = parent.canonical, parent.crossed, parent.io
render, storage_preflight = parent.render, parent.storage_preflight
prelaunch_commands, check_prelaunch = parent.prelaunch_commands, parent.check_prelaunch


def isolate(name, path):
    """New globals, including sys.modules identity; never patch old module globals."""
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def serialized(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def derive_condition(config=None, root=ROOT):
    """Single scalar application from SOURCE only; no edited-score argument/input."""
    config = config or read(root / CONFIG)
    spec = config["candidates"]["preserve"]
    source = authenticated(spec["path"], spec["file_sha256"], root)
    v = source["vector"]
    require(
        len(v) == 1024
        and all(type(x) is float and math.isfinite(x) for x in v)
        and norm(v) == spec["norm"] == 0.10764565083962835
        and vector_sha(v) == spec["vector_float64_le_sha256"] == source["vector_float64_le_sha256"],
        "exact authenticated source vector before endpoint derivation",
    )
    rule = config["endpoint_condition"]
    require(
        rule["radius"] == 0.20 and rule["dimensionless_rounding_tolerance"] == 1e-12,
        "single preexisting .20 ceiling",
    )
    scale = float(0.20 / norm(v))
    vector = [float(scale * x) for x in v]
    measured = norm(vector)
    require(abs(measured - 0.20) <= 1e-12, "endpoint dimensionless rounding cap")
    return {
        "schema": "sp_lense.derived_endpoint_condition.v1",
        "condition_only": True,
        "newly_trained_candidate": False,
        "source_training_success_transfers": False,
        "source_path": spec["path"],
        "source_file_sha256": spec["file_sha256"],
        "source_vector_float64_le_sha256": spec["vector_float64_le_sha256"],
        "source_norm": norm(v),
        "source_verification_sha256": spec["verification_sha256"],
        "source_candidate_freeze_sha256": spec["candidate_freeze_sha256"],
        "source_construction_lock_sha256": spec["construction_lock_sha256"],
        "radius": 0.20,
        "scale": scale,
        "derivation": rule["derivation"],
        "norm": measured,
        "vector_float64_le_sha256": vector_sha(vector),
        "vector": vector,
    }


def candidates(config=None, root=ROOT):
    config = config or read(root / CONFIG)
    expected = derive_condition(config, root)
    path = root / OUTPUT / CONDITION
    raw = path.read_bytes()
    require(raw == serialized(expected), "frozen derived condition bytes/hash")
    return {"preserve": json.loads(raw)}


def prior_records(config, current, root=ROOT):
    spec = config["prior_comparison"]
    paths = {name: spec["namespace"] + "/" + name for name in spec["sha256"]}
    values = {}
    for name, path in paths.items():
        data = (root / path).read_bytes()
        require(sha(data) == spec["sha256"][name], "authenticated prior comparison artifact")
        values[name] = data
    lock = json.loads(values["preregistration.json"])
    require(lock["plan"] == current, "exact frozen parent plan")
    verification = json.loads(values["verification.json"])
    require(
        verification["status"] == "INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH"
        and verification["summary"]["matrix"]["strict_accepted"] == 3,
        "prior3of4 exposed development provenance",
    )
    raw = values["rows.jsonl"].splitlines()
    rows = [json.loads(line) for line in raw]
    require(
        len(rows) == 12
        and [r["cell_id"] for r in rows] == [c["cell_id"] for c in current["cells"]],
        "fixed prior rows, no selection",
    )
    for row, cell in zip(rows, current["cells"], strict=True):
        require(all(row[k] == v for k, v in cell.items()), "prior exact matrix identity")
    snapshots = [
        {"row": row, "row_sha256": canonical_sha(row), "raw_line_sha256": sha(line)}
        for row, line in zip(rows[:8], raw[:8], strict=True)
    ]
    hashes = {paths[name]: digest for name, digest in spec["sha256"].items()}
    for path, digest in lock["source_sha256"].items():
        require(sha((root / path).read_bytes()) == digest, "unchanged prior source")
        hashes[path] = digest
    return {
        "namespace": spec["namespace"],
        "artifact_sha256": spec["sha256"],
        "baseline_records": snapshots[:4],
        "edit_records": snapshots[4:],
        "source_candidate_norm": current["candidates"]["preserve"]["norm"],
        "role": "fixed baseline reproducibility; prior edited outcomes descriptive only",
    }, hashes


def build_plan(root=ROOT):
    current = parent.build_plan(root)
    config = read(root / CONFIG)
    expected, observed = dict(current["config"]), dict(config)
    for key in ("schema", "output_namespace", "interpretation"):
        expected.pop(key)
        observed.pop(key)
    observed.pop("endpoint_condition")
    observed.pop("prior_comparison")
    expected["selection"] = dict(expected["selection"])
    observed["selection"] = dict(observed["selection"])
    expected["selection"].pop("selection_basis")
    observed["selection"].pop("selection_basis")
    require(expected == observed, "unchanged scientific settings except endpoint/provenance")
    require(
        config["output_namespace"] == OUTPUT
        and config["endpoint_condition"]["prospective_lock_files"]
        == [CONDITION, "preregistration.json"]
        and config["endpoint_condition"]["source_training_success_transfers"] is False,
        "one new two-file condition lock, no training claims",
    )
    condition = derive_condition(config, root)
    prior, hashes = prior_records(config, current, root)
    meta = {
        "path": OUTPUT + "/" + CONDITION,
        "file_sha256": sha(serialized(condition)),
        "vector_float64_le_sha256": condition["vector_float64_le_sha256"],
        "norm": condition["norm"],
        "condition_only": True,
        "newly_trained_candidate": False,
        "source_training_success_transfers": False,
        "guarded_training_audit_verified": False,
        "renormalization_allowed": False,
        "extra_strength_allowed": False,
        "sign_inversion_allowed": False,
    }
    return {
        **current,
        "schema": config["schema"],
        "output_namespace": OUTPUT,
        "config": config,
        "input_sha256": {**current["input_sha256"], **hashes},
        "source_candidate": current["candidates"]["preserve"],
        "candidates": {"preserve": meta},
        "derived_condition": condition,
        "prior_comparison": prior,
    }
