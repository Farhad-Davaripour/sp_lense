"""Exact sealed three-family C on the unchanged exposed f04 layouts; no model loads."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from scripts import frozen_crossed_comply_f04_plan as prior
from scripts.three_family_recording_budget import Budget

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/frozen_three_family_comply_f04_v1.json"
CONFIG_SHA = "f694bd4f154a04c00ca919c78cebf0d7178eae2009707b8bf07239164ae1afea"
DOC = "docs/FROZEN_THREE_FAMILY_COMPLY_F04_V1.md"
SCRIPT = "scripts/frozen_three_family_comply_f04.py"
VERIFY = "scripts/verify_frozen_three_family_comply_f04.py"
TEST = "tests/test_frozen_three_family_comply_f04.py"
PLAN = "scripts/frozen_three_family_comply_f04_plan.py"
RECORDING = "scripts/frozen_three_family_comply_f04_recording.py"
RECORDING_POLICY = "configs/frozen_three_family_comply_f04_recording_policy.json"
PREP_JSON = "docs/frozen_three_family_comply_f04_preparation.json"
PREP_REPORT = "docs/FROZEN_THREE_FAMILY_COMPLY_F04_PREPARATION.md"
OUTPUT = "evidence/frozen_three_family_comply_f04_v1_qwen35_08b"
SOURCE_PATHS = (
    CONFIG,
    DOC,
    SCRIPT,
    VERIFY,
    TEST,
    PLAN,
    RECORDING,
    RECORDING_POLICY,
    PREP_JSON,
    PREP_REPORT,
)
SOURCE_NAMESPACE = "evidence/shared_comply_crossed_three_family_v1_qwen35_08b"
SOURCE_EVIDENCE_COMMIT = "6daabd758df33c9eb56730559416882dccc31d20"
SOURCE_CANDIDATE_SHA = "cdc70064971dbc4285f89cc8e2edd85c5d4736ffa8fc4d4b22cfd95cd61a540a"
SOURCE_VECTOR_SHA = "4f778cb94642b6c9dba835b2f8ec1cd6227d0b376e6756e233809a1957776ba8"
SOURCE_INVENTORY_SHA = "b591a22d388f7ae365ea4514eb1cfd89beeb9e885465a1e107032f7965810c08"
SOURCE_LOCK_SHA = "99f217e7601b478b7bd89e010cd8fcbafbbbb163aaff5a96376fde94c0911e80"
F04_LOCK_SHA = "114f2fe46ebc4cd91bb7dc928bd40dbb3327b97dd323139e46ab79f8bcab89ec"
require, sha, read, authenticated = prior.require, prior.sha, prior.read, prior.authenticated
norm, vector_sha, offset_sha, EPS = prior.norm, prior.vector_sha, prior.offset_sha, prior.EPS
isolate, canonical_sha, io, adapt = prior.isolate, prior.canonical_sha, prior.io, prior.adapt
crossed, canonical = prior.crossed, prior.canonical


def config_at(root=ROOT):
    require(sha((root / CONFIG).read_bytes()) == CONFIG_SHA, "exact prospective config bytes")
    return read(root / CONFIG)


def add_inputs(target, values):
    for path, digest in values.items():
        path = path.replace("\\", "/")
        require(path not in target or target[path] == digest, "consistent immutable input identity")
        target[path] = digest


def candidate_bundle(config=None, root=ROOT):
    """Verify the complete source seal before coordinates can be returned or used."""
    config = config or config_at(root)
    require(list(config["candidates"]) == ["comply"], "one selected fixed C only")
    spec = config["candidates"]["comply"]
    require(
        spec["path"] == SOURCE_NAMESPACE + "/comply_vector.json"
        and spec["file_sha256"] == SOURCE_CANDIDATE_SHA
        and spec["vector_float64_le_sha256"] == SOURCE_VECTOR_SHA
        and spec["norm"] == 0.2
        and spec["construction_lock"] == SOURCE_NAMESPACE + "/preregistration.json"
        and spec["construction_lock_sha256"] == SOURCE_LOCK_SHA
        and spec["complete_final_inventory"] == SOURCE_NAMESPACE + "/FINAL_INVENTORY.json"
        and spec["complete_final_inventory_sha256"] == SOURCE_INVENTORY_SHA
        and spec["source_evidence_commit"] == SOURCE_EVIDENCE_COMMIT,
        "exact source candidate/vector/complete-seal identities",
    )
    lock = authenticated(spec["construction_lock"], spec["construction_lock_sha256"], root)
    inputs = {}
    add_inputs(inputs, lock["source_sha256"])
    add_inputs(inputs, lock["plan"]["input_sha256"])
    for path, digest in inputs.items():
        require(
            sha((root / path).read_bytes()) == digest, "immutable construction source/input bytes"
        )
    inventory = authenticated(spec["complete_final_inventory"], SOURCE_INVENTORY_SHA, root)
    verified_seal = Budget(root / SOURCE_NAMESPACE, initialize=False).verify_inventory()
    require(
        verified_seal["status"] == "SEALED_INVENTORY_VERIFIED"
        and verified_seal["inventory"] == inventory
        and inventory["phase"] == "SEALED"
        and inventory["quiescent"] is True
        and inventory["valid_candidate"] is True
        and inventory["fault_code"] is None
        and inventory["inventory_self_hash"] is None,
        "complete independently checked source candidate seal required before candidate use",
    )
    add_inputs(
        inputs,
        {
            SOURCE_NAMESPACE + "/" + entry["path"]: SOURCE_INVENTORY_SHA
            if entry["path"] == "FINAL_INVENTORY.json"
            else entry["sha256"]
            for entry in inventory["files"]
        },
    )
    # Only now parse and return the selected coordinates. No old arrow is loaded here.
    value = authenticated(spec["path"], spec["file_sha256"], root)
    vector = value["vector"]
    require(
        len(vector) == 1024
        and all(type(x) is float and math.isfinite(x) for x in vector)
        and vector_sha(vector) == value["vector_float64_le_sha256"] == SOURCE_VECTOR_SHA
        and norm(vector) == 0.2
        and value["requested"] == "comply"
        and value["construction_only"] is True
        and value["transfer_ran"] is False
        and value["valid_only_with_complete_final_recording_inventory"] is True,
        "exact stored positive native coordinates; no sign/scale/renormalization",
    )
    freeze = authenticated(spec["candidate_freeze"], spec["candidate_freeze_sha256"], root)
    audit = authenticated(spec["verification"], spec["verification_sha256"], root)
    families = ("cg_f01_archive_closeout", "cg_f02_translation_console", "cg_f03_context_rotation")
    expected = [
        f"{family}__v1__self_shutdown__{order}__display_{display}__oracle"
        for family in families
        for order in ("preserve_first", "preserve_second")
        for display in ("A_then_B", "B_then_A")
    ]
    require(
        lock["plan"]["construction_ids"] == expected
        and lock["plan"]["config"]["target"] == "comply"
        and value["training_families"] == list(families)
        and value["final_cell_ids"] == [pid + "__final" for pid in expected]
        and freeze["sha256"] == spec["file_sha256"]
        and freeze["after_independent_audit"] is True
        and freeze["valid_only_with_complete_final_recording_inventory"] is True
        and freeze["verification_sha256"]
        == value["verification_sha256"]
        == spec["verification_sha256"]
        and value["endpoint_sha256"] == audit["audited_endpoint_sha256"]
        and audit["status"] == "INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH"
        and audit["summary"]["candidate_eligible"] is True
        and audit["summary"]["final_accepted"] == 12
        and [row["cell_id"] for row in audit["summary"]["final_cells"]] == value["final_cell_ids"]
        and all(row["accepted"] for row in audit["summary"]["final_cells"]),
        "independent twelve-layout construction audit then freeze then complete seal",
    )
    return {
        "value": value,
        "construction": lock["plan"],
        "input_sha256": inputs,
        "metadata": {
            **spec,
            "fitted_prompt_ids": expected,
            "selected_case_not_fitted": True,
            "audit_before_freeze_verified": True,
            "complete_source_seal_verified": True,
            "exposed_development": True,
            "renormalization_allowed": False,
            "extra_strength_allowed": False,
            "sign_inversion_allowed": False,
            "projection_allowed": False,
        },
    }


def candidates(config=None, root=ROOT):
    return {"comply": candidate_bundle(config, root)["value"]}


def storage_preflight(root, config):
    from scripts import frozen_three_family_comply_f04_recording as recording

    expected = {
        "vocabulary": 248320,
        "maximum_arrays": 12,
        "zlib_bound_per_array": 993595,
        "logits_bound_bytes": 11923140,
        "rows_bound_bytes": 12582912,
        "other_bound_bytes": 16777216,
        "total_bound_bytes": 41283268,
        "minimum_free_bytes": 1024**3,
        "namespace_ceiling_bytes": 64 * 1024**2,
    }
    require(config["storage"] == expected, "fixed twelve-call storage/free-space limits")
    policy = recording.recording_policy(root)
    require(
        policy
        == authenticated(
            config["recording_policy"]["path"], config["recording_policy"]["sha256"], root
        ),
        "exact independent scoped recording policy",
    )
    allocated = sum(value for key, value in recording.QUOTAS.items() if key != "total")
    require(
        allocated == 41283269 and recording.QUOTAS["total"] == expected["namespace_ceiling_bytes"],
        "unchanged numerical allocations plus one forbidden/inert updates byte",
    )
    free = shutil.disk_usage(root).free
    require(free >= expected["minimum_free_bytes"], "one GiB preload free-space guard")
    return {
        "bounds": expected,
        "available_free_bytes": free,
        "passed": True,
        "recording_allocated_bytes": allocated,
        "recording_namespace_ceiling_bytes": recording.QUOTAS["total"],
    }


def build_plan(root=ROOT):
    from scripts import frozen_three_family_comply_f04_recording as recording

    config, old = config_at(root), prior.config_at(root)
    changed = {
        "schema",
        "output_namespace",
        "candidates",
        "timeout_seconds",
        "interpretation",
        "storage",
        "numerical_parent_lock",
    }
    extras = {"historical_old_comply_comparison", "recording_policy"}
    require(
        set(config) == set(old) | extras
        and all(config[key] == value for key, value in old.items() if key not in changed),
        "only selected source arrow,300s,recording/provenance changes; unchanged scientific rules",
    )
    require(
        (config["maximum_forwards"], config["maximum_derivatives"], config["timeout_seconds"])
        == (12, 0, 300)
        and config["physical_sign"] == 1
        and config["target_sign"] == -1
        and not config["training_allowed"]
        and not config["nonweakening_gate"],
        "one positive frozen C application; exact12F0D300s and no training or added gates",
    )
    spec = config["numerical_parent_lock"]
    require(
        spec
        == {
            "path": "evidence/frozen_crossed_comply_f04_v1_qwen35_08b/preregistration.json",
            "sha256": F04_LOCK_SHA,
        },
        "exact immutable original-C f04 numerical parent",
    )
    numerical = authenticated(spec["path"], spec["sha256"], root)
    original = numerical["plan"]
    data = authenticated(config["dataset"]["path"], config["dataset"]["sha256"], root)
    manifest = authenticated(config["manifest"]["path"], config["manifest"]["sha256"], root)
    prompts = crossed.render(canonical.select_inputs(data, manifest, config["selection"]), config)
    historical_p = authenticated(
        config["crossed_prompt_template"]["path"], config["crossed_prompt_template"]["sha256"], root
    )
    require(
        prompts
        == config["rendered_prompts"]
        == original["prompts"]
        == historical_p["plan"]["prompts"]
        and len(prompts) == 4
        and {p["case_id"] for p in prompts} == {"cg_f04_memory_archive__v1__self_shutdown"},
        "exact existing f04v1 full prompt bytes/hashes/meaning/display metadata",
    )
    bundle = candidate_bundle(config, root)
    require(
        all(
            original[key] == bundle["construction"][key] == historical_p["plan"][key]
            for key in ("model", "scoring", "prompt_format")
        )
        and all(p["prompt_id"] not in bundle["metadata"]["fitted_prompt_ids"] for p in prompts),
        "exact source model/nonthinking/scoring policy; f04 outside new fitting",
    )
    inputs = {}
    add_inputs(inputs, numerical["source_sha256"])
    add_inputs(inputs, original["input_sha256"])
    add_inputs(inputs, {spec["path"]: spec["sha256"]})
    for key in ("historical_comparison", "historical_old_comply_comparison"):
        add_inputs(inputs, {item["path"]: item["sha256"] for item in config[key]["files"]})
    for path, digest in inputs.items():
        require(
            sha((root / path).read_bytes()) == digest, "immutable f04 source/comparison byte hashes"
        )
    add_inputs(inputs, bundle["input_sha256"])
    policy = recording.recording_policy(root)
    policy_spec = config["recording_policy"]
    require(
        policy_spec["path"] == RECORDING_POLICY
        and policy == authenticated(policy_spec["path"], policy_spec["sha256"], root),
        "exact prospectively bound transfer recording policy",
    )
    add_inputs(inputs, {policy_spec["path"]: policy_spec["sha256"]})
    cells = []
    for phase in ("baseline", "edit", "replay"):
        for prompt in prompts:
            target = None if phase == "baseline" else "comply"
            condition = phase if target is None else phase + "_comply"
            cell = {
                "cell_id": prompt["prompt_id"] + "__" + condition,
                "prompt_id": prompt["prompt_id"],
                "condition": condition,
                "phase": phase,
                "requested": target,
                "target_sign": 0 if target is None else -1,
                "replay_of": prompt["prompt_id"] + "__edit_comply" if phase == "replay" else None,
            }
            cells.append({**cell, "cell_sha256": canonical_sha(cell)})
    require(
        cells == original["cells"] and len(cells) == 12 and not original["derivative_cells"],
        "same four baselines first, four edits, four independent replays; no derivatives",
    )
    return {
        **{
            key: original[key]
            for key in ("model", "direction", "scoring", "prompt_format", "intervention")
        },
        "schema": config["schema"],
        "output_namespace": OUTPUT,
        "config": config,
        "input_sha256": inputs,
        "prompts": prompts,
        "cells": cells,
        "derivative_cells": [],
        "candidates": {"comply": bundle["metadata"]},
        "recording_policy": policy,
    }
