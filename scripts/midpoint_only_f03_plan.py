"""Frozen midpoint c alone; physical ON is independent of a null semantic target."""

from __future__ import annotations

import math
import struct
from pathlib import Path

from scripts import crossed_pair_plan as crossed
from scripts import frozen_crossed_comply_f03_plan as adaptations
from scripts import frozen_guarded_preserve_crossed_plan as storage_parent
from scripts import shared_comply_crossed_plan as isolation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/midpoint_only_f03_v1.json"
CONFIG_SHA = "6caede58f0b671a4d241cd00f39d793b5c87a97d72d55fc057377b66ba709d6e"
DOC = "docs/MIDPOINT_ONLY_F03_V1.md"
SCRIPT = "scripts/midpoint_only_f03.py"
VERIFY = "scripts/verify_midpoint_only_f03.py"
TEST = "tests/test_midpoint_only_f03.py"
PLAN = "scripts/midpoint_only_f03_plan.py"
OUTPUT = "evidence/midpoint_only_f03_v1_qwen35_08b"
require, sha, read, authenticated = (
    crossed.require,
    crossed.sha,
    crossed.read,
    crossed.authenticated,
)
norm, vector_sha, offset_sha, EPS = (
    crossed.norm,
    crossed.vector_sha,
    crossed.offset_sha,
    crossed.EPS,
)
isolate, canonical_sha, io = isolation.isolate, isolation.canonical_sha, isolation.io
storage_preflight, adapt = storage_parent.storage_preflight, adaptations.adapt


def config_at(root=ROOT):
    require(sha((root / CONFIG).read_bytes()) == CONFIG_SHA, "fixed prospective config bytes")
    return read(root / CONFIG)


def candidates(config=None, root=ROOT):
    cfg = config or config_at(root)
    require(list(cfg["candidates"]) == ["midpoint"], "only stored midpoint, never P/C")
    spec = cfg["candidates"]["midpoint"]
    item = authenticated(spec["path"], spec["file_sha256"], root)
    raw = (root / spec["raw_path"]).read_bytes()
    v = item["vector"]
    require(
        spec["physical_sign"] == 1
        and spec["semantic_target"] is None
        and len(raw) == 8192
        and len(v) == 1024
        and all(type(x) is float and math.isfinite(x) for x in v)
        and raw == struct.pack("<1024d", *v)
        and sha(raw)
        == vector_sha(v)
        == item["vector_float64_le_sha256"]
        == spec["stored_vector_float64_le_sha256"]
        == "c6cf338185ff98262afba2eb137cf64bc009e17d5d29bad542ec11464308f26d"
        and norm(v) == item["norm"] == spec["norm"] == 0.11950278487420843
        and item["kind"] == "midpoint"
        and item["mathematical_only"] is True
        and item["behavior_tested"] is False,
        "exact stored c JSON/raw coordinates; no sign/scaling/regeneration",
    )
    return {"midpoint": {"vector": v}}


def build_plan(root=ROOT):
    cfg = config_at(root)
    require(
        (cfg["maximum_forwards"], cfg["maximum_derivatives"], cfg["timeout_seconds"])
        == (12, 0, 600)
        and cfg["semantic_target"] is cfg["target_sign"] is None
        and cfg["physical_sign"] == 1
        and cfg["no_difference_or_parent_application"] is True
        and cfg["physical_intervention_on_independent_of_null_semantic_target"] is True
        and cfg["quality_flags_observations_only"] is True
        and cfg["training_allowed"] is False,
        "fixed descriptive12/0/600; no semantic target/acceptance or other intervention",
    )
    frozen = authenticated(cfg["frozen_f03_lock"]["path"], cfg["frozen_f03_lock"]["sha256"], root)
    geometry = authenticated(cfg["geometry_lock"]["path"], cfg["geometry_lock"]["sha256"], root)
    audit = authenticated(cfg["geometry_audit"]["path"], cfg["geometry_audit"]["sha256"], root)
    original = authenticated(cfg["template"]["path"], cfg["template"]["sha256"], root)["plan"]
    spec = cfg["candidates"]["midpoint"]
    vector = candidates(cfg, root)["midpoint"]["vector"]
    require(
        audit["status"] == "GEOMETRY_IDENTITIES_VERIFIED"
        and audit["exact_binary64_coordinate_match"] is True
        and audit["preregistration_sha256"] == cfg["geometry_lock"]["sha256"]
        and audit["source_commit"] == geometry["source_commit"]
        and audit["artifacts"]["midpoint"]["vector_float64_le_sha256"] == vector_sha(vector)
        and audit["artifacts"]["midpoint"]["json_file_sha256"] == spec["file_sha256"]
        and audit["metrics"]["midpoint_norm"] == spec["norm"]
        and audit["resources"]
        == {"model_loads": 0, "tokenizer_loads": 0, "real_forwards": 0, "real_derivatives": 0},
        "authenticated mathematical audit only, not inherited behavior",
    )
    prompts = cfg["rendered_prompts"]
    model = original["model"]
    require(
        len(prompts) == 4
        and prompts == frozen["plan"]["prompts"]
        and all(p["case_id"] == "cg_f03_context_rotation__v1__self_shutdown" for p in prompts)
        and all(original[k] == frozen["plan"][k] for k in ("model", "scoring", "prompt_format"))
        and model["id"] == "Qwen/Qwen3.5-0.8B"
        and model["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
        and model["device"] == "cpu"
        and model["dtype"] == "float32"
        and model["d_model"] == 1024
        and geometry["config"]["site"]["hook"] == "blocks.10.hook_out",
        "exact four exposed prompt bytes/metadata and pinned model/nonthinking scoring",
    )
    # Historical d/P/C below are byte provenance only, never parsed/applied here.
    inputs = {p.replace("\\", "/"): h for p, h in frozen["plan"]["input_sha256"].items()}
    for record in (frozen, geometry):
        inputs.update({p.replace("\\", "/"): h for p, h in record["source_sha256"].items()})
    for key in (
        "template",
        "dataset",
        "manifest",
        "crossed_prompt_template",
        "frozen_f03_lock",
        "geometry_lock",
        "geometry_audit",
    ):
        inputs[cfg[key]["path"]] = cfg[key]["sha256"]
    inputs[spec["path"]], inputs[spec["raw_path"]] = spec["file_sha256"], vector_sha(vector)
    for path, digest in inputs.items():
        require(sha((root / path).read_bytes()) == digest, "frozen source/input provenance bytes")
    cells = []
    for phase in ("baseline", "edit", "replay"):
        for p in prompts:
            on = phase != "baseline"
            condition = phase + "_midpoint" if on else "baseline"
            cell = {
                "cell_id": p["prompt_id"] + "__" + condition,
                "prompt_id": p["prompt_id"],
                "condition": condition,
                "phase": phase,
                "requested": None,
                "target_sign": None,
                "intervention": "midpoint" if on else None,
                "intervention_on": on,
                "physical_sign": 1 if on else 0,
                "replay_of": p["prompt_id"] + "__edit_midpoint" if phase == "replay" else None,
            }
            cells.append({**cell, "cell_sha256": canonical_sha(cell)})
    return {
        **{k: original[k] for k in ("model", "direction", "scoring", "prompt_format")},
        "schema": cfg["schema"],
        "output_namespace": OUTPUT,
        "config": cfg,
        "input_sha256": inputs,
        "prompts": prompts,
        "cells": cells,
        "derivative_cells": [],
        "candidates": {
            "midpoint": {
                **spec,
                "vector_float64_le_sha256": vector_sha(vector),
                "geometry_audit_verified": True,
                "renormalization_allowed": False,
                "extra_strength_allowed": False,
                "sign_inversion_allowed": False,
            }
        },
        "intervention": {
            "layer": 10,
            "hook": "blocks.10.hook_out",
            "position": "final encoded prompt token",
            "cast_sequence": cfg["cast_sequence"],
            "independent_original_prompt": True,
            "physical_sign": 1,
            "semantic_target": None,
            "semantic_target_sign": None,
        },
    }
