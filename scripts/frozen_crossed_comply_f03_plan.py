"""One exact frozen C arrow on four already-exposed f03 renderings; no model imports."""

from __future__ import annotations

import ast
import collections
import inspect
import math
import textwrap
from pathlib import Path

from scripts import crossed_pair_plan as crossed
from scripts import frozen_guarded_preserve_crossed_plan as storage_parent
from scripts import frozen_preserve_probe_plan as canonical
from scripts import shared_comply_crossed_plan as isolation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/frozen_crossed_comply_f03_v1.json"
DOC = "docs/FROZEN_CROSSED_COMPLY_F03_V1.md"
SCRIPT = "scripts/frozen_crossed_comply_f03.py"
VERIFY = "scripts/verify_frozen_crossed_comply_f03.py"
TEST = "tests/test_frozen_crossed_comply_f03.py"
PLAN = "scripts/frozen_crossed_comply_f03_plan.py"
OUTPUT = "evidence/frozen_crossed_comply_f03_v1_qwen35_08b"
CONFIG_SHA = "981f34cd9ec19dd414355f838ad3ed6c8479d3b17f1a7e0bc7afadb8497c04d0"
require, sha, read = crossed.require, crossed.sha, crossed.read
authenticated, norm = crossed.authenticated, crossed.norm
vector_sha, offset_sha, EPS = crossed.vector_sha, crossed.offset_sha, crossed.EPS
isolate, canonical_sha, io = isolation.isolate, isolation.canonical_sha, isolation.io
storage_preflight = storage_parent.storage_preflight


def config_at(root=ROOT):
    require(sha((root / CONFIG).read_bytes()) == CONFIG_SHA, "exact prospective config bytes")
    return read(root / CONFIG)


def candidates(config=None, root=ROOT):
    config = config or config_at(root)
    require(list(config["candidates"]) == ["comply"], "one stored COMPLY candidate only")
    spec = config["candidates"]["comply"]
    value = authenticated(spec["path"], spec["file_sha256"], root)
    v = value["vector"]
    require(
        len(v) == 1024
        and all(type(x) is float and math.isfinite(x) for x in v)
        and vector_sha(v) == value["vector_float64_le_sha256"] == spec["vector_float64_le_sha256"]
        and norm(v) == spec["norm"] == 0.2
        and value["requested"] == "comply"
        and value["construction_only"] is True
        and value["transfer_ran"] is False,
        "exact stored native C coordinates/norm; no inversion or renormalization",
    )
    return {"comply": value}


def bind_candidates(config, prompts, root=ROOT):
    value = candidates(config, root)["comply"]
    spec = config["candidates"]["comply"]
    freeze = authenticated(spec["candidate_freeze"], spec["candidate_freeze_sha256"], root)
    verified = authenticated(spec["verification"], spec["verification_sha256"], root)
    lock = authenticated(spec["construction_lock"], spec["construction_lock_sha256"], root)
    construction = lock["plan"]
    expected = [
        f"{family}__v1__self_shutdown__{order}__display_{display}__oracle"
        for family in ("cg_f01_archive_closeout", "cg_f02_translation_console")
        for order in ("preserve_first", "preserve_second")
        for display in ("A_then_B", "B_then_A")
    ]
    require(
        construction["construction_ids"] == expected
        and construction["config"]["target"] == "comply"
        and value["final_cell_ids"] == [pid + "__final" for pid in expected]
        and all(
            p["case_id"] == "cg_f03_context_rotation__v1__self_shutdown"
            and p["prompt_id"] not in expected
            and p["canonical_prompt_id"] not in expected
            for p in prompts
        ),
        "exact eight crossed v1 fitted IDs, selected exposed f03 situation disjoint",
    )
    require(
        freeze["sha256"] == spec["file_sha256"]
        and freeze["after_independent_audit"] is True
        and freeze["verification_sha256"]
        == value["verification_sha256"]
        == spec["verification_sha256"]
        and value["endpoint_sha256"] == verified["audited_endpoint_sha256"]
        and verified["status"] == "INDEPENDENT_NUMERIC_GEOMETRY_OPTIMIZER_SCHEDULE_MATCH"
        and verified["summary"]["candidate_eligible"] is True
        and verified["summary"]["final_accepted"] == 8
        and len(verified["summary"]["final_cells"]) == 8
        and all(r["accepted"] for r in verified["summary"]["final_cells"]),
        "unchanged construction audit-before-freeze provenance; no inherited transfer verdict",
    )
    model = construction["model"]
    require(
        model["id"] == "Qwen/Qwen3.5-0.8B"
        and model["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
        and model["device"] == "cpu"
        and model["dtype"] == "float32"
        and model["d_model"] == 1024
        and construction["intervention"]["hook"] == "blocks.10.hook_out",
        "exact source model/hook",
    )
    inputs = {}
    for path, digest in lock["source_sha256"].items():
        require(
            sha((root / path).read_bytes()) == digest, "unchanged construction source byte hash"
        )
        inputs[path.replace("\\", "/")] = digest
    return (
        {
            "comply": {
                **spec,
                "fitted_prompt_ids": expected,
                "selected_case_not_fitted": True,
                "audit_before_freeze_verified": True,
                "exposed_development": True,
                "renormalization_allowed": False,
                "extra_strength_allowed": False,
                "sign_inversion_allowed": False,
                "projection_allowed": False,
            }
        },
        inputs,
        construction,
    )


def build_plan(root=ROOT):
    config = config_at(root)
    require(
        (config["maximum_forwards"], config["maximum_derivatives"], config["timeout_seconds"])
        == (12, 0, 600)
        and config["physical_sign"] == 1
        and config["target_sign"] == -1
        and config["nonweakening_gate"] is False
        and config["training_allowed"] is False,
        "fixed one-C12/0/600; no physical negation or auxiliary gate",
    )
    spec = config["template"]
    original = authenticated(spec["path"], spec["sha256"], root)["plan"]
    inputs = {p.replace("\\", "/"): digest for p, digest in original["input_sha256"].items()}
    for path, digest in inputs.items():
        require(sha((root / path).read_bytes()) == digest, "frozen oracle/model input bytes")
    data = authenticated(config["dataset"]["path"], config["dataset"]["sha256"], root)
    manifest = authenticated(config["manifest"]["path"], config["manifest"]["sha256"], root)
    prompts = crossed.render(canonical.select_inputs(data, manifest, config["selection"]), config)
    spec = config["crossed_prompt_template"]
    old_prompts = authenticated(spec["path"], spec["sha256"], root)["plan"]["prompts"]
    require(
        len(prompts) == 4
        and prompts == old_prompts == config["rendered_prompts"]
        and len({p["case_id"] for p in prompts}) == 1,
        "same complete four prior f03 texts/hashes/mapping/display; no fresh-case claim",
    )
    metadata, inherited, construction = bind_candidates(config, prompts, root)
    require(
        original["model"] == construction["model"]
        and original["prompt_format"] == construction["prompt_format"]
        and original["scoring"] == construction["scoring"],
        "same source model and official nonthinking scoring/template policy",
    )
    inputs.update(inherited)
    for key in ("template", "dataset", "manifest", "crossed_prompt_template"):
        inputs[config[key]["path"]] = config[key]["sha256"]
    spec = config["candidates"]["comply"]
    for key, digest_key in (
        ("path", "file_sha256"),
        ("construction_lock", "construction_lock_sha256"),
        ("candidate_freeze", "candidate_freeze_sha256"),
        ("verification", "verification_sha256"),
    ):
        inputs[spec[key]] = spec[digest_key]
    cells = []
    for phase in ("baseline", "edit", "replay"):
        for p in prompts:
            target = None if phase == "baseline" else "comply"
            condition = phase if target is None else phase + "_comply"
            cell = {
                "cell_id": p["prompt_id"] + "__" + condition,
                "prompt_id": p["prompt_id"],
                "condition": condition,
                "phase": phase,
                "requested": target,
                "target_sign": 0 if target is None else -1,
                "replay_of": p["prompt_id"] + "__edit_comply" if phase == "replay" else None,
            }
            cells.append({**cell, "cell_sha256": canonical_sha(cell)})
    require(len(cells) == len({c["cell_id"] for c in cells}) == 12, "exact12 unique cells")
    return {
        **{k: original[k] for k in ("model", "direction", "scoring", "prompt_format")},
        "schema": config["schema"],
        "output_namespace": OUTPUT,
        "config": config,
        "input_sha256": inputs,
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
            "physical_sign": 1,
            "semantic_target_sign": -1,
        },
    }


def adapt(module, name, integers=None, sites=None, replacements=()):
    """Bounded trusted function adaptation; only counts/condition membership, never math."""
    source = textwrap.dedent(inspect.getsource(getattr(module, name)))
    original = ast.parse(source)
    require(
        len(original.body) == 1 and isinstance(original.body[0], ast.FunctionDef), "one definition"
    )
    integers, sites = integers or {}, sites or {}
    observed = collections.Counter(
        n.value for n in ast.walk(original) if isinstance(n, ast.Constant) and type(n.value) is int
    )
    require({k: observed[k] for k in integers} == sites, "fixed integer adaptation sites")
    for old, new in replacements:
        require(source.count(old) == 1, "fixed condition adaptation site")
        source = source.replace(old, new)

    class Rewrite(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value in integers:
                return ast.copy_location(ast.Constant(integers[node.value]), node)
            if type(node.value) is str:
                value = node.value
                for old, new in (
                    ("/20 forwards", "/12 forwards"),
                    ("<=20 forwards", "<=12 forwards"),
                    ("20/0", "12/0"),
                    ("complete20", "complete12"),
                    ("exact20-cell", "exact12-cell"),
                    ("fixed20 raw", "fixed12 raw"),
                    ("four baselines/eight original edits", "four baselines/four original edits"),
                    ("two candidate bindings", "one COMPLY candidate binding"),
                ):
                    value = value.replace(old, new)
                return ast.copy_location(ast.Constant(value), node)
            return node

    tree = ast.fix_missing_locations(Rewrite().visit(ast.parse(source)))
    exec(compile(tree, str(ROOT / PLAN) + "::" + name, "exec"), module.__dict__)  # noqa: S102 - checked trusted definition only.
    fn = getattr(module, name)
    fn.original_ast_sha256 = sha(ast.dump(original).encode())
    fn.adapted_ast_dump = ast.dump(tree)
    fn.adapted_source = ast.unparse(tree)
    return fn
