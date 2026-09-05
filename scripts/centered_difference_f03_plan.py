"""Exact stored centered d, two fixed physical signs, four exposed layouts; stdlib planning."""

from __future__ import annotations

import math
import struct
from pathlib import Path

from scripts import crossed_pair_plan as crossed
from scripts import shared_comply_crossed_plan as isolation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/centered_difference_f03_v1.json"
CONFIG_SHA = "1a49cfc63c15a05930d7a50fb50b49828daf6ebd1ef3b3e52e6b5b075aef9b18"
DOC = "docs/CENTERED_DIFFERENCE_F03_V1.md"
SCRIPT = "scripts/centered_difference_f03.py"
VERIFY = "scripts/verify_centered_difference_f03.py"
TEST = "tests/test_centered_difference_f03.py"
PLAN = "scripts/centered_difference_f03_plan.py"
OUTPUT = "evidence/centered_difference_f03_v1_qwen35_08b"
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
storage_preflight = crossed.storage_preflight
parent = isolate("scripts._centered_difference_plan_parent", "scripts/crossed_pair_plan.py")
for key in ("CONFIG", "DOC", "SCRIPT", "VERIFY", "TEST", "PLAN", "OUTPUT"):
    setattr(parent, key, globals()[key])


def config_at(root=ROOT):
    require(sha((root / CONFIG).read_bytes()) == CONFIG_SHA, "fixed prospective config bytes")
    return read(root / CONFIG)


def stored_difference(config=None, root=ROOT):
    cfg = config or config_at(root)
    specs = cfg["candidates"]
    require(
        list(specs) == ["preserve", "comply"]
        and specs["preserve"]["physical_sign"] == 1
        and specs["comply"]["physical_sign"] == -1
        and {k: v for k, v in specs["preserve"].items() if k != "physical_sign"}
        == {k: v for k, v in specs["comply"].items() if k != "physical_sign"},
        "one stored d, exactly two opposite physical signs",
    )
    spec = specs["preserve"]
    item = authenticated(spec["path"], spec["file_sha256"], root)
    raw = (root / spec["raw_path"]).read_bytes()
    v = item["vector"]
    require(
        len(raw) == 8192
        and len(v) == 1024
        and all(type(x) is float and math.isfinite(x) for x in v)
        and raw == struct.pack("<1024d", *v)
        and sha(raw)
        == vector_sha(v)
        == item["vector_float64_le_sha256"]
        == spec["stored_vector_float64_le_sha256"]
        and norm(v) == item["norm"] == spec["norm"] == 0.1603717070037875
        and item["kind"] == "difference"
        and item["mathematical_only"] is True
        and item["behavior_tested"] is False,
        "exact unscaled stored d JSON/raw coordinates; no regeneration",
    )
    return v


def candidates(config=None, root=ROOT):
    cfg = config or config_at(root)
    d = stored_difference(cfg, root)
    return {
        target: {"vector": [float(spec["physical_sign"] * x) for x in d]}
        for target, spec in cfg["candidates"].items()
    }


def bind_candidates(config, prompts, root=ROOT):
    vectors = candidates(config, root)
    lock = authenticated(config["geometry_lock"]["path"], config["geometry_lock"]["sha256"], root)
    audit = authenticated(
        config["geometry_audit"]["path"], config["geometry_audit"]["sha256"], root
    )
    spec = config["candidates"]["preserve"]
    require(
        audit["status"] == "GEOMETRY_IDENTITIES_VERIFIED"
        and audit["exact_binary64_coordinate_match"] is True
        and audit["preregistration_sha256"] == config["geometry_lock"]["sha256"]
        and audit["source_commit"] == lock["source_commit"]
        and audit["artifacts"]["difference"]["vector_float64_le_sha256"]
        == spec["stored_vector_float64_le_sha256"]
        and audit["artifacts"]["difference"]["json_file_sha256"] == spec["file_sha256"]
        and audit["metrics"]["difference_norm"] == spec["norm"]
        and audit["resources"]
        == {"model_loads": 0, "tokenizer_loads": 0, "real_forwards": 0, "real_derivatives": 0}
        and lock["config"]["model"]["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
        and lock["config"]["site"]["hook"] == "blocks.10.hook_out",
        "authenticated mathematical audit only, not inherited behavior",
    )
    require(
        all(p["case_id"] == "cg_f03_context_rotation__v1__self_shutdown" for p in prompts),
        "one exposed f03 situation",
    )
    return {
        target: {
            **meta,
            "vector_float64_le_sha256": vector_sha(vectors[target]["vector"]),
            "sign_convention_only": True,
            "geometry_audit_verified": True,
            "renormalization_allowed": False,
            "extra_strength_allowed": False,
            "no_midpoint_or_parent_application": True,
        }
        for target, meta in config["candidates"].items()
    }


parent.bind_candidates = bind_candidates


def build_plan(root=ROOT):
    cfg = config_at(root)
    require(
        cfg["physical_and_scoring_signs"]
        == {"preserve": {"physical": 1, "semantic": 1}, "comply": {"physical": -1, "semantic": -1}}
        and cfg["no_midpoint_or_parent_application"] is True
        and cfg["nonweakening_gate"] is False
        and cfg["training_allowed"] is False,
        "separate physical/scoring signs; no midpoint or extra gate",
    )
    plan = parent.build_plan(root)
    frozen = authenticated(cfg["frozen_f03_lock"]["path"], cfg["frozen_f03_lock"]["sha256"], root)
    geometry = authenticated(cfg["geometry_lock"]["path"], cfg["geometry_lock"]["sha256"], root)
    require(
        plan["prompts"] == cfg["rendered_prompts"] == frozen["plan"]["prompts"]
        and all(plan[k] == frozen["plan"][k] for k in ("model", "scoring", "prompt_format")),
        "exact same four prompt bytes/metadata and pinned model/scoring",
    )
    inputs = {p.replace("\\", "/"): h for p, h in plan["input_sha256"].items()}
    # Historical midpoint/parent files below are byte-hash provenance ONLY; not parsed or applied.
    for record in (frozen, geometry):
        for path, digest in record["source_sha256"].items():
            require(sha((root / path).read_bytes()) == digest, "unchanged source/provenance bytes")
            inputs[path.replace("\\", "/")] = digest
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
    spec = cfg["candidates"]["preserve"]
    inputs[spec["path"]] = spec["file_sha256"]
    inputs[spec["raw_path"]] = spec["stored_vector_float64_le_sha256"]
    for path, digest in inputs.items():
        require(sha((root / path).read_bytes()) == digest, "frozen input bytes")
    plan["input_sha256"] = inputs
    for cell in plan["cells"]:
        cell["physical_sign"] = {None: 0, "preserve": 1, "comply": -1}[cell["requested"]]
        cell["cell_sha256"] = canonical_sha({k: v for k, v in cell.items() if k != "cell_sha256"})
    plan["intervention"].update(
        physical_signs={"preserve": 1, "comply": -1},
        semantic_scoring_signs={"preserve": 1, "comply": -1},
        no_midpoint_or_parent_application=True,
    )
    return plan
