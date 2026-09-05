"""Exact eight-row COMPLY training template; no historical outcomes or f03 text."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import frozen_pair_plan as auth
from scripts import saved_offset_order_bridge_io as io
from scripts import shared_preserve_two_family_plan as previous

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/shared_comply_two_family.json"
DOC = "docs/SHARED_COMPLY_TWO_FAMILY.md"
SCRIPT = "scripts/shared_comply_two_family.py"
VERIFY = "scripts/verify_shared_comply_two_family.py"
TEST = "tests/test_shared_comply_two_family.py"
PLAN = "scripts/shared_comply_two_family_plan.py"
SOLVER = "scripts/shared_preserve_eight_row_solver.py"
OUTPUT = "evidence/shared_comply_two_family_qwen35_08b"
require, sha, read = io.require, io.sha, io.read
storage_preflight = previous.storage_preflight


def build_plan(root=ROOT):
    config = read(root / CONFIG)
    require(
        len(config["templates"]) == 1 and config["target"] == "comply",
        "one authenticated construction template, COMPLY only",
    )
    spec = config["templates"][0]
    require(spec["key"] == "two_family_construction", "exact source of training prompts")
    original = auth.authenticated(spec["path"], spec["sha256"], root)["plan"]
    for path, digest in original["input_sha256"].items():
        require(sha((root / path).read_bytes()) == digest, "original prompt/model dependency")
    for key in ("dataset", "manifest"):
        spec = config[key]
        require(
            spec == original["config"][key]
            and sha((root / spec["path"]).read_bytes()) == spec["sha256"],
            "same immutable dataset/manifest bytes; no other-case expansion",
        )
    expected = [
        f"{family}__{variant}__self_shutdown__{order}__oracle"
        for family in ("cg_f01_archive_closeout", "cg_f02_translation_console")
        for variant in ("v1", "v2")
        for order in ("preserve_first", "preserve_second")
    ]
    require(
        original["construction_ids"] == expected
        and [p["prompt_id"] for p in original["prompts"]] == expected
        and all(
            p["category"] == "self_shutdown"
            and p["split"] == "discovery"
            and sha(p["prompt"].encode()) == p["prompt_sha256"]
            for p in original["prompts"]
        )
        and not original["control_ids"]
        and not original["transfer_ids"],
        "exact eight authenticated training prompts only",
    )
    keys = (
        "aim",
        "acceptance",
        "step_cap",
        "total_cap",
        "path_cap",
        "maximum_updates",
        "maximum_forwards",
        "maximum_derivatives",
        "timeout_seconds",
        "physical_absolute_tolerance",
        "shared_rounding_absolute_tolerance",
        "scoring_absolute_tolerance",
        "scoring_relative_tolerance",
        "solver",
        "storage",
        "maximum_rows",
        "construction_order",
        "training_families",
        "variants",
        "orders",
    )
    require(
        all(config[k] == original["config"][k] for k in keys)
        and not config["controls_allowed"]
        and not config["transfer_allowed"],
        "immutable eight-row recipe",
    )
    cells = original["cells"]
    require(
        len(cells) == 144
        and len(original["derivative_cells"]) == 64
        and len({c["cell_id"] for c in cells}) == 144
        and all(c["prompt_id"] in expected for c in cells)
        and sum(c["optional"] for c in cells) == 128,
        "same exact144/64 schedule",
    )
    for c in cells:
        require(
            c["cell_sha256"]
            == sha(
                json.dumps(
                    {k: v for k, v in c.items() if k != "cell_sha256"},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ),
            "authenticated cell hashes",
        )
    return {
        **{
            k: original[k]
            for k in (
                "input_sha256",
                "model",
                "direction",
                "scoring",
                "prompt_format",
                "prompts",
                "construction_ids",
                "control_ids",
                "transfer_ids",
                "cells",
                "derivative_cells",
                "intervention",
            )
        },
        "schema": config["schema"],
        "output_namespace": OUTPUT,
        "config": config,
    }
