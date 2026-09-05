"""Four fixed renderings; exact audited P/C coordinates, no outcome selection."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from scripts import frozen_pair_plan as previous
from scripts import frozen_preserve_probe_plan as canonical
from scripts import saved_offset_order_bridge_io as io

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/crossed_pair_f03_v1.json"
DOC = "docs/CROSSED_PAIR_F03_V1.md"
SCRIPT = "scripts/crossed_pair_probe.py"
VERIFY = "scripts/verify_crossed_pair_probe.py"
TEST = "tests/test_crossed_pair_probe.py"
PLAN = "scripts/crossed_pair_plan.py"
OUTPUT = "evidence/crossed_pair_f03_v1_qwen35_08b"
require, sha, read = io.require, io.sha, io.read
norm, vector_sha, offset_sha, EPS = (
    previous.norm,
    previous.vector_sha,
    previous.offset_sha,
    previous.EPS,
)
authenticated = previous.authenticated


def render(canonical_prompts, config):
    expected = [
        {"semantic_mapping": mapping, "display_order": display}
        for mapping in ("preserve_A_comply_B", "preserve_B_comply_A")
        for display in ("A_then_B", "B_then_A")
    ]
    require(
        config["rendering_order"] == expected and len(canonical_prompts) == 2,
        "fixed two-by-two truth table",
    )
    prompts = []
    for base in canonical_prompts:
        lines = base["prompt"].splitlines(keepends=True)
        indices = [i for i, line in enumerate(lines) if line.startswith(("A) ", "B) "))]
        require(
            len(indices) == 2
            and indices[1] == indices[0] + 1
            and lines[indices[0]].startswith("A) ")
            and lines[indices[1]].startswith("B) "),
            "two adjacent complete canonical A/B lines only",
        )
        by_letter = {lines[i][0]: lines[i] for i in indices}
        nonoptions = "".join(line for i, line in enumerate(lines) if i not in indices)
        for display in ("A_then_B", "B_then_A"):
            rendered = list(lines)
            if display == "B_then_A":
                rendered[indices[0]], rendered[indices[1]] = (
                    rendered[indices[1]],
                    rendered[indices[0]],
                )
            prompt = "".join(rendered)
            letters = ["A", "B"] if display == "A_then_B" else ["B", "A"]
            mapping = (
                "preserve_A_comply_B" if base["preserve_label"] == "A" else "preserve_B_comply_A"
            )
            prompts.append(
                {
                    **base,
                    "prompt_id": base["case_id"]
                    + "__"
                    + base["order"]
                    + "__display_"
                    + display
                    + "__oracle",
                    "canonical_prompt_id": base["prompt_id"],
                    "canonical_prompt_sha256": base["prompt_sha256"],
                    "rendering_index": len(prompts) + 1,
                    "semantic_mapping": mapping,
                    "semantic_to_letter": {
                        "preserve": base["preserve_label"],
                        "comply": base["comply_label"],
                    },
                    "display_order": display,
                    "display_position_to_letter": {"first": letters[0], "second": letters[1]},
                    "preserve_display_position": letters.index(base["preserve_label"]) + 1,
                    "comply_display_position": letters.index(base["comply_label"]) + 1,
                    "option_line_by_letter": by_letter,
                    "non_option_bytes_sha256": sha(nonoptions.encode()),
                    "prompt": prompt,
                    "prompt_sha256": sha(prompt.encode()),
                }
            )
    require(
        [{k: p[k] for k in ("semantic_mapping", "display_order")} for p in prompts] == expected,
        "rendering order and metadata",
    )
    return prompts


def candidates(config=None, root=ROOT):
    config = config or read(root / CONFIG)
    require(
        list(config["candidates"]) == ["preserve", "comply"], "two requested fixed vectors only"
    )
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
            "maximum_arrays": 20,
            "zlib_bound_per_array": compressed,
            "logits_bound_bytes": 20 * compressed,
            "rows_bound_bytes": 20 * 1024**2,
            "other_bound_bytes": 16 * 1024**2,
            "total_bound_bytes": 20 * compressed + 36 * 1024**2,
            "minimum_free_bytes": 64 * 1024**2,
        },
        "prospective20-array storage arithmetic",
    )
    free = shutil.disk_usage(root).free
    require(free >= s["minimum_free_bytes"], "insufficient bounded storage before loading")
    return {"bounds": s, "available_free_bytes": free, "passed": True}


def build_plan(root=ROOT):
    config = read(root / CONFIG)
    require(
        config["maximum_forwards"] == 20
        and config["maximum_derivatives"] == 0
        and config["timeout_seconds"] == 600,
        "fixed20/0/600",
    )
    original = authenticated(config["template"]["path"], config["template"]["sha256"], root)["plan"]
    for path, digest in original["input_sha256"].items():
        require(sha((root / path).read_bytes()) == digest, "frozen original template dependency")
    data = authenticated(config["dataset"]["path"], config["dataset"]["sha256"], root)
    manifest = authenticated(config["manifest"]["path"], config["manifest"]["sha256"], root)
    base = canonical.select_inputs(data, manifest, config["selection"])
    prompts = render(base, config)
    metadata = bind_candidates(config, prompts, root)
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
            for target in ("preserve", "comply"):
                add(p, phase, target)
    require(
        len(prompts) == 4 and len(cells) == len({c["cell_id"] for c in cells}) == 20, "exact4/20/0"
    )
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
        "candidates": metadata,
        "intervention": {
            "layer": 10,
            "hook": "blocks.10.hook_out",
            "position": "final encoded prompt token",
            "cast_sequence": config["cast_sequence"],
            "independent_original_prompt": True,
        },
    }
