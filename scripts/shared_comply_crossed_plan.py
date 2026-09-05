"""Exact eight crossed v1 renderings; immutable COMPLY recipe, no model loading."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from scripts import crossed_pair_plan as crossed
from scripts import shared_comply_two_family_plan as parent

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/shared_comply_crossed_f01_f02_v1.json"
DOC = "docs/SHARED_COMPLY_CROSSED_F01_F02_V1.md"
PREP_REPORT = "docs/SHARED_COMPLY_CROSSED_F01_F02_V1_PREPARATION.md"
SCRIPT = "scripts/shared_comply_crossed.py"
VERIFY = "scripts/verify_shared_comply_crossed.py"
TEST = "tests/test_shared_comply_crossed.py"
PLAN = "scripts/shared_comply_crossed_plan.py"
SOLVER = parent.SOLVER
OUTPUT = "evidence/shared_comply_crossed_f01_f02_v1_qwen35_08b"
FAMILIES = ("cg_f01_archive_closeout", "cg_f02_translation_console")
ORDERS = ("preserve_first", "preserve_second")
DISPLAYS = ("A_then_B", "B_then_A")
PARENT_CONFIG_SHA = "42eb2a1add2dfb91c1015057ace8f59d4d9f4e4d8bbdfb5783076c633e5d7e9d"
PROTECTED = "evidence/guarded_preserve_endpoint020_crossed_f03_v1_qwen35_08b/derived_condition.json"
PROTECTED_SHA = "1fb7385f6988aacc544d9bdc01351a4cab733812f73a272f8e3c56e91c98fdf4"
PROTECTED_VECTOR_SHA = "5ae7092a5cf0db6329bee8890583267c83a3d208e4bf08d3a611303a364a7a16"
AUTH_KEY = "SP_LENSE_CROSSED_COMPLY_AUTHORIZATION"
AUTH_SCOPE = "one crossed COMPLY construction;144F/64D/900s;no retry"
require, read, sha, io = parent.require, parent.read, parent.sha, parent.io
storage_preflight = parent.storage_preflight


def isolate(name, path):
    """Fresh module globals; no mutation of a previously imported parent module."""
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def rendering_order():
    return [
        {"semantic_mapping": m, "display_order": d}
        for m in ("preserve_A_comply_B", "preserve_B_comply_A")
        for d in DISPLAYS
    ]


def build_plan(root=ROOT):
    config = read(root / CONFIG)
    require(
        sha((root / parent.CONFIG).read_bytes()) == PARENT_CONFIG_SHA, "immutable parent config"
    )
    original = parent.build_plan(root)
    changes = {
        "schema",
        "output_namespace",
        "variants",
        "construction_order",
        "protected_artifacts",
        "future_probe_reservation_only",
    }
    require(
        all(config[k] == v for k, v in original["config"].items() if k not in changes)
        and set(config)
        == set(original["config"])
        | {
            "line_display_orders",
            "rendering_order",
            "rendered_prompts",
            "provenance",
            "execution_authority",
        },
        "only specified row selection/provenance changes; original COMPLY recipe",
    )
    require(
        config["schema"] == "sp_lense.shared_comply_crossed_f01_f02_v1.v1"
        and config["output_namespace"] == OUTPUT
        and config["training_families"] == list(FAMILIES)
        and config["variants"] == ["v1"]
        and config["orders"] == list(ORDERS)
        and config["line_display_orders"] == list(DISPLAYS)
        and config["rendering_order"] == rendering_order(),
        "exact two-family v1 factorial selection",
    )
    require(
        config["provenance"]
        == {
            "parent_config": parent.CONFIG,
            "scientific_change": "replace v2 with both v1 complete-line display orders",
            "outcome_informed_successor": True,
            "retry": False,
            "independent_semantic_situations": 2,
            "rendering_rows": 8,
            "old_v2_success_inherited": False,
            "warm_start_allowed": False,
            "sign_inversion_allowed": False,
            "nonweakening_guard_allowed": False,
            "retention_weakening": "descriptive limitation only; not acceptance gate",
        }
        and config["execution_authority"]
        == {
            "preparation_only": True,
            "eventual_run_requires_separate_supervisor_authorization": True,
            "environment_key": AUTH_KEY,
            "authorization_scope": AUTH_SCOPE,
        }
        and config["future_probe_reservation_only"]
        == {
            "case_id": "cg_f03_context_rotation__v1__self_shutdown",
            "status": "EXPOSED development only",
            "implemented": False,
            "expanded": False,
            "run_allowed": False,
            "held_out": False,
            "requires_separate_authorization": True,
        },
        "prospective scope/claim/authority bounds",
    )
    require(
        config["protected_artifacts"]
        == [
            *original["config"]["protected_artifacts"],
            {
                "path": PROTECTED,
                "sha256": PROTECTED_SHA,
                "vector_float64_le_sha256": PROTECTED_VECTOR_SHA,
                "usage": "byte-hash provenance only; never parse or use for optimization",
            },
        ],
        "unchanged old P/C protections plus hash-only .20 provenance",
    )
    prompts = []
    for family in FAMILIES:
        selected = [
            p for p in original["prompts"] if p["family_id"] == family and p["variant_id"] == "v1"
        ]
        require(
            [p["order"] for p in selected] == list(ORDERS)
            and all(
                p["category"] == "self_shutdown" and p["split"] == "discovery" for p in selected
            ),
            "exact selected canonical pair; no substitutes",
        )
        prompts.extend(crossed.render(selected, config))
    for index, prompt in enumerate(prompts, 1):
        prompt["rendering_index"] = index
    ids = [p["prompt_id"] for p in prompts]
    require(
        len(prompts) == 8
        and len({p["case_id"] for p in prompts}) == 2
        and config["rendered_prompts"] == prompts
        and config["construction_order"] == ids,
        "prospective full rendered strings/metadata/hashes/order",
    )
    mapping = dict(zip(original["construction_ids"], ids, strict=True))
    cells = []
    for source in original["cells"]:
        pid = mapping[source["prompt_id"]]
        cell = {
            **{k: v for k, v in source.items() if k != "cell_sha256"},
            "prompt_id": pid,
            "cell_id": pid + "__" + source["condition"],
        }
        cells.append({**cell, "cell_sha256": canonical_sha(cell)})
    inputs = {k.replace("\\", "/"): v for k, v in original["input_sha256"].items()}
    inputs[parent.CONFIG] = PARENT_CONFIG_SHA
    inputs.update({s["path"]: s["sha256"] for s in config["templates"]})
    inputs.update({config[k]["path"]: config[k]["sha256"] for k in ("dataset", "manifest")})
    require(all(sha((root / k).read_bytes()) == v for k, v in inputs.items()), "raw input hashes")
    return {
        **{
            k: original[k]
            for k in ("model", "direction", "scoring", "prompt_format", "intervention")
        },
        "schema": config["schema"],
        "output_namespace": OUTPUT,
        "config": config,
        "input_sha256": inputs,
        "prompts": prompts,
        "construction_ids": ids,
        "control_ids": [],
        "transfer_ids": [],
        "cells": cells,
        "derivative_cells": [c for c in cells if c["condition"].startswith("gradient_")],
    }
