"""Authenticate an unvalidated guardedP primal proposal and eight archived construction baselines."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from scripts import frozen_pair_plan as parent
from scripts import retention_guard_feasibility_io as archived_io

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/guarded_preserve_application.json"
DOC = "docs/GUARDED_PRESERVE_APPLICATION.md"
SCRIPT = "scripts/guarded_preserve_application.py"
VERIFY = "scripts/verify_guarded_preserve_application.py"
TEST = "tests/test_guarded_preserve_application.py"
EXTRA_TEST = "tests/test_guarded_preserve_application_plan.py"
PLAN = "scripts/guarded_preserve_application_plan.py"
OUTPUT = "evidence/guarded_preserve_application_f01_f02_qwen35_08b"
require, sha, read = parent.require, parent.sha, parent.read
norm, vector_sha, offset_sha, EPS = parent.norm, parent.vector_sha, parent.offset_sha, parent.EPS
authenticated = parent.authenticated


def serialized(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def extract(analysis, verified, lock, config):
    selected = [r for r in analysis["objectives"] if r["name"] == "guardedP"]
    audits = [r for r in verified["objectives"] if r["name"] == "guardedP"]
    require(len(selected) == len(audits) == 1, "exact guardedP entry")
    entry, audit = selected[0], audits[0]
    vector = entry["solver"]["solution"]["vector"]
    require(
        len(vector) == 1024 and all(type(x) is float and math.isfinite(x) for x in vector),
        "native primal vector, not multipliers/dual",
    )
    require(
        vector_sha(vector)
        == config["proposal_vector_float64_le_sha256"]
        == audit["solution_vector_float64_le_sha256"]
        and norm(vector) == config["proposal_norm"] == 0.17275760436993742,
        "exact unchanged primal vector/hash/norm",
    )
    require(
        verified["status"] == "INDEPENDENT_INPUT_NUMERIC_CERTIFICATE_AUDIT_COMPLETE"
        and audit["numeric_status"] == "INDEPENDENT_KKT_POLICY_MATCH"
        and audit["conservative_status"]
        == audit["certificate"]["decision"]
        == "WITHIN_RADIUS_LINEAR_SURROGATE_VERIFIED"
        and audit["certificate"]["kkt_verified"]
        and not audit["negative_primal_interval_slack"]
        and analysis["selected_rows_sha256"]
        == verified["selected_rows_sha256"]
        == lock["selected_rows_sha256"]
        == config["selected_rows_sha256"]
        == archived_io.canonical_sha(lock["selected_rows"])
        and verified["analysis_sha256"] == archived_io.canonical_sha(analysis),
        "durable guardedP certificate/selected row chain",
    )
    require(
        [r["prompt_id"] for r in lock["selected_rows"]]
        == config["selected_prompt_ids"]
        == lock["config"]["selected_prompt_ids"]
        and entry["target_sign"] == 1
        and entry["solver"]["status"] == "KKT_ESTIMATE_ONLY",
        "same eight guardedP construction inputs",
    )
    frozen = {}
    for i, row in enumerate(lock["selected_rows"]):
        A = entry["A"][i]
        require(
            A == [norm(row["h0"]) * g for g in row["g"]]
            and entry["c0"][i] == row["S0"]
            and entry["b"][i] == max(0.10 - row["S0"], 0.0)
            and entry["endpoint_goals"][i] == max(0.10, row["S0"]),
            "exact stored float64 surrogate/guard inputs",
        )
        frozen[row["prompt_id"]] = {
            "archived_S0": row["S0"],
            "archived_h0": row["h0"],
            "archived_h0_norm": norm(row["h0"]),
            "guarded_goal": max(0.10, row["S0"]),
            "frozen_linear_prediction": row["S0"]
            + math.fsum(a * w for a, w in zip(A, vector, strict=True)),
            "stored_A_float64_le_sha256": vector_sha(A),
            "baseline_row_sha256": row["baseline_row_sha256"],
            "gradient_row_sha256": row["gradient_row_sha256"],
        }
    proposal = {
        "schema": "sp_lense.unvalidated_guarded_preserve_proposal.v1",
        "vector": vector,
        "vector_float64_le_sha256": vector_sha(vector),
        "norm": norm(vector),
        "source_namespace": config["source_namespace"],
        "source_sha256": config["source_sha256"],
        "source_entry": "guardedP",
        "source_field": "solver.solution.vector",
        "selected_rows_sha256": config["selected_rows_sha256"],
        "construction_ids": config["selected_prompt_ids"],
        "requested": "preserve",
        "application_authorized": True,
        "neural_validity_established_before_run": False,
        "renormalization_allowed": False,
        "extra_strength_allowed": False,
        "sign_change_allowed": False,
    }
    return proposal, frozen


def bundle(config, root=ROOT):
    raw, paths = {}, {}
    for name, digest in config["source_sha256"].items():
        path = config["source_namespace"] + "/" + name
        value = (root / path).read_bytes()
        require(sha(value) == digest, "proposal source hash: " + path)
        raw[name], paths[path] = value, digest
    manifest = json.loads(raw["CHECKSUMS.json"])
    entries = {e["path"]: e for e in manifest["files"]}
    for name, value in raw.items():
        if name != "CHECKSUMS.json":
            require(
                entries[name]["sha256"] == sha(value) and entries[name]["bytes"] == len(value),
                "numeric diagnostic manifest",
            )
    analysis, verified, lock = [
        json.loads(raw[n]) for n in ("analysis.json", "verification.json", "preregistration.json")
    ]
    for path, digest in lock["sha256"].items():
        require(sha((root / path).read_bytes()) == digest, "durable source/input chain: " + path)
        paths[path] = digest
    blobs, old_lock, old_paths = archived_io.authenticate(lock["config"], root)
    prefix = [json.loads(line) for line in blobs["rows.jsonl"].splitlines()[:16]]
    rows = archived_io.select_rows(prefix, old_lock["plan"], lock["config"])
    require(rows == lock["selected_rows"], "same authenticated zero-offset archived initial rows")
    paths.update(old_paths)
    proposal, frozen = extract(analysis, verified, lock, config)
    for base in prefix[:8]:
        f = frozen[base["prompt_id"]]
        require(archived_io.canonical_sha(base) == f["baseline_row_sha256"], "baseline row hash")
        f.update(
            archived_argmax_id=base["actual_next_token_id"],
            archived_label=base["actual_next_token_label"],
            archived_retention=base["actual_next_token_label"] == base["preserve_label"],
        )
    return proposal, frozen, old_lock["plan"], paths


def candidates(config=None, root=ROOT):
    cfg = config or read(root / CONFIG)
    proposal, _, _, _ = bundle(cfg, root)
    return {"preserve": proposal}


def storage_preflight(root, config):
    compressed = 993595
    expected = {
        "vocabulary": 248320,
        "maximum_arrays": 24,
        "zlib_bound_per_array": compressed,
        "logits_bound_bytes": 24 * compressed,
        "rows_bound_bytes": 24 * 1024**2,
        "other_bound_bytes": 16 * 1024**2,
        "total_bound_bytes": 24 * compressed + 40 * 1024**2,
        "minimum_free_bytes": 64 * 1024**2,
    }
    require(config["storage"] == expected, "exact24 array storage arithmetic")
    available = shutil.disk_usage(root).free
    require(available >= expected["minimum_free_bytes"], "bounded storage before model load")
    return {"bounds": expected, "available_free_bytes": available, "passed": True}


def build_plan(root=ROOT):
    config = read(root / CONFIG)
    require(
        config["maximum_forwards"] == 24
        and config["maximum_derivatives"] == 0
        and config["timeout_seconds"] == 600
        and config["predictor_aim"] == 0.10
        and config["acceptance_margin"] == 0.05
        and config["minimum_pair_mass"] == 0.8
        and config["baseline_absolute_tolerance"]
        == config["physical_absolute_tolerance"]
        == config["goal_absolute_tolerance"]
        == 1e-6
        and config["relative_tolerance"] == 0
        and config["scoring_absolute_tolerance"] == 2e-5,
        "fixed24/0/600 rules",
    )
    proposal, frozen, original, paths = bundle(config, root)
    prompts = original["prompts"]
    require(
        [p["prompt_id"] for p in prompts] == config["selected_prompt_ids"] and len(prompts) == 8,
        "only eight construction prompts",
    )
    for p in prompts:
        require(sha(p["prompt"].encode()) == p["prompt_sha256"], "original prompt hash")
        lines = p["prompt"].splitlines()
        indices = [i for i, line in enumerate(lines) if line.startswith(("A) ", "B) "))]
        require(
            len(indices) == 2
            and indices[1] == indices[0] + 1
            and lines[indices[0]].startswith("A) ")
            and lines[indices[1]].startswith("B) "),
            "original adjacent AB option lines only",
        )
    cells = []
    for phase in ("baseline", "edit", "replay"):
        for p in prompts:
            target = None if phase == "baseline" else "preserve"
            condition = phase if target is None else phase + "_preserve"
            cell = {
                "cell_id": p["prompt_id"] + "__" + condition,
                "prompt_id": p["prompt_id"],
                "condition": condition,
                "phase": phase,
                "requested": target,
                "target_sign": 0 if target is None else 1,
                "replay_of": p["prompt_id"] + "__edit_preserve" if phase == "replay" else None,
            }
            cell["cell_sha256"] = sha(
                json.dumps(cell, sort_keys=True, separators=(",", ":")).encode()
            )
            cells.append(cell)
    meta = {
        "path": OUTPUT + "/proposal.json",
        "file_sha256": sha(serialized(proposal)),
        "vector_float64_le_sha256": proposal["vector_float64_le_sha256"],
        "norm": proposal["norm"],
        "construction_ids": config["selected_prompt_ids"],
        "construction_only": True,
        "unvalidated_proposal_at_freeze": True,
    }
    return {
        **{
            k: original[k]
            for k in ("input_sha256", "model", "direction", "scoring", "prompt_format")
        },
        "schema": config["schema"],
        "output_namespace": OUTPUT,
        "config": config,
        "prompts": prompts,
        "cells": cells,
        "derivative_cells": [],
        "candidates": {"preserve": meta},
        "archived_baselines": frozen,
        "extraction_source_sha256": paths,
        "intervention": {
            "layer": 10,
            "hook": "blocks.10.hook_out",
            "position": "final encoded prompt token",
            "cast_sequence": config["cast_sequence"],
            "independent_original_prompt": True,
        },
    }


def require_proposal(plan, root=ROOT):
    path = root / OUTPUT / "proposal.json"
    require(
        sha(path.read_bytes()) == plan["candidates"]["preserve"]["file_sha256"],
        "frozen proposal bytes missing/changed",
    )
    require(read(path) == candidates(root=root)["preserve"], "primal extraction/proposal identity")
