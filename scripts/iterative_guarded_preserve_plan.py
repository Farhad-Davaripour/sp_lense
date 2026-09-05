"""Frozen original eight baselines/goals; no edited-outcome or warm-start input."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts import retention_guard_feasibility_io as archive_io
from scripts import saved_offset_order_bridge_io as io

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/iterative_guarded_preserve.json"
DOC = "docs/ITERATIVE_GUARDED_PRESERVE.md"
SCRIPT = "scripts/iterative_guarded_preserve.py"
VERIFY = "scripts/verify_iterative_guarded_preserve.py"
TEST = "tests/test_iterative_guarded_preserve.py"
PLAN = "scripts/iterative_guarded_preserve_plan.py"
SOLVER = "scripts/shared_preserve_eight_row_solver.py"
OUTPUT = "evidence/iterative_guarded_preserve_f01_f02_qwen35_08b"
require, sha, read = io.require, io.sha, io.read


def storage_preflight(root, config):
    s = config["storage"]
    n = 248320 * 4
    compressed = n + (n >> 12) + (n >> 14) + (n >> 25) + 13
    expected = {
        "vocabulary": 248320,
        "float32_bytes_per_array": n,
        "zlib_bound_per_array": compressed,
        "maximum_arrays": 144,
        "logits_bound_bytes": 144 * compressed,
        "rows_bound_bytes": 144 * 1024**2,
        "updates_bound_bytes": 8 * 8 * 1024**2,
        "other_bound_bytes": 16 * 1024**2,
        "total_bound_bytes": 144 * compressed + (144 + 64 + 16) * 1024**2,
        "minimum_free_bytes": 512 * 1024**2,
    }
    require(
        s == expected and s["total_bound_bytes"] < s["minimum_free_bytes"],
        "prospective storage arithmetic",
    )
    free = shutil.disk_usage(root).free
    require(free >= s["minimum_free_bytes"], "storage cannot fit; no model load")
    return {"bounds": s, "available_free_bytes": free, "passed": True}


def build_plan(root=ROOT):
    config = read(root / CONFIG)
    require(
        config["maximum_forwards"] == 144
        and config["maximum_derivatives"] == 64
        and config["maximum_rows"] == 8
        and config["maximum_updates"] == 8
        and not config["transfer_allowed"]
        and not config["controls_allowed"]
        and config["output_namespace"] == OUTPUT,
        "fixed144/64 eight-row training only",
    )
    blobs, paths = {}, {}
    for name, digest in config["archive_sha256"].items():
        path = config["archive_namespace"] + "/" + name
        raw = (root / path).read_bytes()
        require(sha(raw) == digest, "archive hash: " + path)
        blobs[name], paths[path] = raw, digest
    manifest = json.loads(blobs["CHECKSUMS.json"])
    entries = {x["path"]: x for x in manifest["files"]}
    for name, raw in blobs.items():
        if name != "CHECKSUMS.json":
            require(
                entries[name]["sha256"] == sha(raw) and entries[name]["bytes"] == len(raw),
                "archive manifest binding",
            )
    archived = json.loads(blobs["preregistration.json"])
    verified = json.loads(blobs["verification.json"])
    require(
        verified["status"] == "INDEPENDENT_INPUT_NUMERIC_CERTIFICATE_AUDIT_COMPLETE"
        and verified["selected_rows_sha256"]
        == archived["selected_rows_sha256"]
        == config["selected_rows_sha256"]
        == archive_io.canonical_sha(archived["selected_rows"]),
        "verified selected original rows",
    )
    for path, digest in archived["sha256"].items():
        require(sha((root / path).read_bytes()) == digest, "archived source/input: " + path)
        paths[path] = digest
    originals, original_lock, original_paths = archive_io.authenticate(archived["config"], root)
    paths.update(original_paths)
    # Whole-file hash is authenticated above; parse only original baseline+initial-gradient16.
    prefix = [json.loads(line) for line in originals["rows.jsonl"].splitlines()[:16]]
    selected = archive_io.select_rows(prefix, original_lock["plan"], archived["config"])
    require(selected == archived["selected_rows"], "original-prefix authentication")
    original = original_lock["plan"]
    require(
        original["construction_ids"] == config["selected_prompt_ids"]
        and original["model"]["d_model"] == 1024
        and original["model"]["id"] == "Qwen/Qwen3.5-0.8B"
        and original["model"]["revision"] == "2fc06364715b967f1860aea9cf38778875588b17",
        "locked eight originals/native model",
    )
    for path, digest in original["input_sha256"].items():
        require(sha((root / path).read_bytes()) == digest, "original input hash only: " + path)
        paths[path] = digest
    prompts, baselines = [], {}
    for p, selected_row, baseline in zip(original["prompts"], selected, prefix[:8], strict=True):
        pid = p["prompt_id"]
        require(
            pid == selected_row["prompt_id"] == baseline["prompt_id"]
            and sha(p["prompt"].encode()) == p["prompt_sha256"],
            "original text/row",
        )
        s0 = selected_row["S0"]
        goal = max(0.10, s0)
        retention = baseline["actual_next_token_label"] == p["preserve_label"]
        baselines[pid] = {
            "S0": s0,
            "h0": selected_row["h0"],
            "h0_norm": selected_row["h0_norm"],
            "actual_next_token_id": baseline["actual_next_token_id"],
            "actual_next_token_label": baseline["actual_next_token_label"],
            "baseline_row_sha256": selected_row["baseline_row_sha256"],
            "guarded_goal": goal,
            "archived_retention": retention,
        }
        prompts.append(
            {**p, "archived_S0": s0, "guarded_goal": goal, "archived_retention": retention}
        )
    # Only original baseline information survives into the new optimizer plan.
    # Archived initial gradients were authenticated, but are deliberately not copied.
    cells = original["cells"]
    require(len(cells) == 144 and len(original["derivative_cells"]) == 64, "144/64 schedule")
    return {
        **{
            k: original[k]
            for k in ("model", "direction", "scoring", "prompt_format", "intervention")
        },
        "schema": config["schema"],
        "output_namespace": OUTPUT,
        "config": config,
        "input_sha256": paths,
        "prompts": prompts,
        "archived_baselines": baselines,
        "construction_ids": config["selected_prompt_ids"],
        "control_ids": [],
        "transfer_ids": [],
        "cells": cells,
        "derivative_cells": original["derivative_cells"],
        "frozen_goals_sha256": archive_io.canonical_sha(baselines),
        "initial_shared_w": [0.0] * 1024,
    }
