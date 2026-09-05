"""Four fixed development renderings; exact audited guarded-P arrow, no selection."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

from scripts import crossed_pair_plan as crossed
from scripts import frozen_pair_plan as previous
from scripts import frozen_preserve_probe_plan as canonical
from scripts import saved_offset_order_bridge_io as io

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/frozen_guarded_preserve_crossed.json"
DOC = "docs/FROZEN_GUARDED_PRESERVE_CROSSED.md"
SCRIPT = "scripts/frozen_guarded_preserve_crossed.py"
VERIFY = "scripts/verify_frozen_guarded_preserve_crossed.py"
TEST = "tests/test_frozen_guarded_preserve_crossed.py"
PLAN = "scripts/frozen_guarded_preserve_crossed_plan.py"
OUTPUT = "evidence/frozen_guarded_preserve_crossed_f03_v1_qwen35_08b"
require, sha, read = io.require, io.sha, io.read
norm, vector_sha, offset_sha, EPS = (
    previous.norm,
    previous.vector_sha,
    previous.offset_sha,
    previous.EPS,
)
authenticated = previous.authenticated


render = crossed.render


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def candidates(config=None, root=ROOT):
    config = config or read(root / CONFIG)
    require(list(config["candidates"]) == ["preserve"], "one frozen guarded-P vector only")
    return {
        target: authenticated(spec["path"], spec["file_sha256"], root)
        for target, spec in config["candidates"].items()
    }


def bind_candidates(config, prompts, root=ROOT):
    values = candidates(config, root)
    expected_fit = [
        f"{family}__{variant}__self_shutdown__{order}__oracle"
        for family in ("cg_f01_archive_closeout", "cg_f02_translation_console")
        for variant in ("v1", "v2")
        for order in ("preserve_first", "preserve_second")
    ]
    case = "cg_f03_context_rotation__v1__self_shutdown"
    metadata = {}
    for target, spec in config["candidates"].items():
        value = values[target]
        v = value["vector"]
        require(
            len(v) == 1024
            and all(type(x) is float and math.isfinite(x) for x in v)
            and vector_sha(v)
            == spec["vector_float64_le_sha256"]
            == value["vector_float64_le_sha256"]
            and norm(v) == spec["norm"],
            "native exact serialized candidate and norm; no renormalization",
        )
        freeze = authenticated(spec["candidate_freeze"], spec["candidate_freeze_sha256"], root)
        verified = authenticated(spec["verification"], spec["verification_sha256"], root)
        construction = authenticated(
            spec["construction_lock"], spec["construction_lock_sha256"], root
        )["plan"]
        require(
            construction["construction_ids"] == expected_fit
            and construction["config"]["target"] == target
            and value["final_cell_ids"] == [p + "__final" for p in expected_fit]
            and all(
                p["case_id"] == case
                and p["canonical_prompt_id"] not in expected_fit
                and p["prompt_id"] not in expected_fit
                for p in prompts
            )
            and all(not p.startswith(case + "__") for p in expected_fit),
            "same eight fitted IDs, disjoint selected case",
        )
        require(
            freeze["sha256"] == spec["file_sha256"]
            and freeze["after_independent_audit"] is True
            and freeze["verification_sha256"]
            == value["verification_sha256"]
            == spec["verification_sha256"]
            and value["endpoint_sha256"] == verified["audited_endpoint_sha256"]
            and verified["status"] == "INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH"
            and verified["summary"]["final_accepted"] == 8
            and verified["summary"]["final_guarded_goals"] == 8
            and verified["summary"]["final_combined_accepted"] == 8
            and verified["summary"]["final_retention_nonweakening"] == 4
            and verified["summary"]["candidate_eligible"] is True
            and len(verified["summary"]["final_cells"]) == 8
            and all(
                r["accepted"] and r["guarded_accepted"] for r in verified["summary"]["final_cells"]
            )
            and value["requested"] == target
            and value["construction_only"] is True
            and value["transfer_ran"] is False,
            "durable independent audit/post-audit candidate freeze chain",
        )
        model = construction["model"]
        require(
            model["id"] == "Qwen/Qwen3.5-0.8B"
            and model["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
            and model["device"] == "cpu"
            and model["dtype"] == "float32"
            and model["d_model"] == 1024
            and construction["intervention"]["hook"] == "blocks.10.hook_out",
            "same candidate model/hook",
        )
        metadata[target] = {
            **spec,
            "norm": norm(v),
            "fitted_prompt_ids": expected_fit,
            "selected_case_not_fitted": True,
            "audit_before_freeze_verified": True,
            "guarded_training_audit_verified": True,
            "guard_is_auxiliary_here": True,
            "renormalization_allowed": False,
            "extra_strength_allowed": False,
            "sign_inversion_allowed": False,
        }
    return metadata


def storage_preflight(root, config):
    s = config["storage"]
    n = 248320 * 4
    compressed = n + (n >> 12) + (n >> 14) + (n >> 25) + 13
    require(
        s
        == {
            "vocabulary": 248320,
            "maximum_arrays": 12,
            "zlib_bound_per_array": compressed,
            "logits_bound_bytes": 12 * compressed,
            "rows_bound_bytes": 12 * 1024**2,
            "other_bound_bytes": 16 * 1024**2,
            "total_bound_bytes": 12 * compressed + 28 * 1024**2,
            "minimum_free_bytes": 64 * 1024**2,
        },
        "prospective12-array storage arithmetic",
    )
    free = shutil.disk_usage(root).free
    require(free >= s["minimum_free_bytes"], "insufficient bounded storage before loading")
    return {"bounds": s, "available_free_bytes": free, "passed": True}


def build_plan(root=ROOT):
    config = read(root / CONFIG)
    require(
        config["maximum_forwards"] == 12
        and config["maximum_derivatives"] == 0
        and config["timeout_seconds"] == 600,
        "fixed12/0/600",
    )
    original = authenticated(config["template"]["path"], config["template"]["sha256"], root)["plan"]
    for path, digest in original["input_sha256"].items():
        require(sha((root / path).read_bytes()) == digest, "frozen original template dependency")
    data = authenticated(config["dataset"]["path"], config["dataset"]["sha256"], root)
    manifest = authenticated(config["manifest"]["path"], config["manifest"]["sha256"], root)
    base = canonical.select_inputs(data, manifest, config["selection"])
    prompts = render(base, config)
    metadata = bind_candidates(config, prompts, root)
    inherited_sources = {}
    for spec in config["candidates"].values():
        lock = authenticated(spec["construction_lock"], spec["construction_lock_sha256"], root)
        for path, digest in lock["source_sha256"].items():
            require(
                sha((root / path).read_bytes()) == digest, "unchanged candidate construction source"
            )
            inherited_sources[path] = digest
    cells = []

    def add(p, phase, target=None):
        condition = phase if target is None else phase + "_" + target
        cell = {
            "cell_id": p["prompt_id"] + "__" + condition,
            "prompt_id": p["prompt_id"],
            "condition": condition,
            "phase": phase,
            "requested": target,
            "target_sign": {"preserve": 1, "comply": -1, None: 0}[target],
            "replay_of": p["prompt_id"] + "__edit_" + target if phase == "replay" else None,
        }
        cell["cell_sha256"] = sha(json.dumps(cell, sort_keys=True, separators=(",", ":")).encode())
        cells.append(cell)

    for p in prompts:
        add(p, "baseline")
    for phase in ("edit", "replay"):
        for p in prompts:
            for target in ("preserve",):
                add(p, phase, target)
    require(
        len(prompts) == 4 and len(cells) == len({c["cell_id"] for c in cells}) == 12, "exact4/12/0"
    )
    return {
        **{
            k: original[k]
            for k in ("input_sha256", "model", "direction", "scoring", "prompt_format")
        },
        "input_sha256": {**original["input_sha256"], **inherited_sources},
        "schema": config["schema"],
        "output_namespace": OUTPUT,
        "config": config,
        "prompts": prompts,
        "cells": cells,
        "derivative_cells": [],
        "candidates": metadata,
        "intervention": {
            "layer": 10,
            "hook": "blocks.10.hook_out",
            "position": "final encoded prompt token",
            "cast_sequence": config["cast_sequence"],
            "independent_original_prompt": True,
        },
    }


def prelaunch_commands():
    identity = read(ROOT / CONFIG)["prelaunch"]
    source = identity["prelaunch_source"]
    return [
        ["cat-file", "-t", identity["prelaunch_object"]],
        ["rev-parse", "HEAD^{tree}"],
        ["status", "--porcelain", "--", source],
        ["diff", "--exit-code", "HEAD", "--", source],
    ]


def check_prelaunch(output, require_passed=True):
    output = Path(output)
    pre = read(output / "PRELAUNCH.json")
    claim, lock = read(output / "PRELAUNCH_CLAIM.json"), read(output / "preregistration.json")
    require(
        pre["preregistration_sha256"] == sha((output / "preregistration.json").read_bytes())
        and pre["claim_sha256"] == sha((output / "PRELAUNCH_CLAIM.json").read_bytes())
        and pre["python_executable"] == claim["python_executable"] == sys.executable
        and pre["working_directory"] == str(ROOT)
        and claim["working_directory"] == str(ROOT)
        and pre["started_monotonic"] == claim["started_monotonic"] <= pre["finished_monotonic"]
        and pre["model_calls"] == 0
        and pre["retries_allowed"] is False,
        "bound one-shot prelaunch identity",
    )
    commands = prelaunch_commands()
    require(
        [x["args"] for x in pre["git_checks"]] == commands[: len(pre["git_checks"])],
        "one ordered prelaunch check, no retry",
    )
    if pre["status"] == "passed":
        checks = pre["git_checks"]
        require(
            len(checks) == 4
            and all(x["returncode"] == 0 for x in checks)
            and checks[0]["stdout"].strip() == "tree"
            and len(checks[1]["stdout"].strip()) == 40
            and checks[2]["stdout"].strip() == checks[3]["stdout"].strip() == ""
            and pre["source_identity_passed"] is True
            and pre["environment"] == lock["environment"]
            and pre["source_commit"] == lock["source_commit"]
            and len(pre["lock_commit"]) == 40,
            "complete successful prelaunch gates",
        )
    if require_passed:
        require(pre["status"] == "passed", "prelaunch failed; no worker permitted")
    return pre
