"""Only a fixed f04 input binding over the immutable frozen-C f03 plan."""

from __future__ import annotations

from pathlib import Path

from scripts import frozen_crossed_comply_f03_plan as prior
from scripts.frozen_endpoint020_crossed_f04_plan import canonical

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/frozen_crossed_comply_f04_v1.json"
CONFIG_SHA = "775621d453f083247563e4144f41350de98cde378b1e6425c2978e19896b7e23"
DOC = "docs/FROZEN_CROSSED_COMPLY_F04_V1.md"
SCRIPT = "scripts/frozen_crossed_comply_f04.py"
VERIFY = "scripts/verify_frozen_crossed_comply_f04.py"
TEST = "tests/test_frozen_crossed_comply_f04.py"
PLAN = "scripts/frozen_crossed_comply_f04_plan.py"
OUTPUT = "evidence/frozen_crossed_comply_f04_v1_qwen35_08b"
require, sha, read, authenticated = prior.require, prior.sha, prior.read, prior.authenticated
norm, vector_sha, offset_sha, EPS = prior.norm, prior.vector_sha, prior.offset_sha, prior.EPS
isolate, canonical_sha, io, adapt = prior.isolate, prior.canonical_sha, prior.io, prior.adapt
storage_preflight, crossed = prior.storage_preflight, prior.crossed
parent = isolate("scripts._frozen_C_f04_plan_parent", "scripts/frozen_crossed_comply_f03_plan.py")
parent.canonical = canonical
for key in ("CONFIG", "CONFIG_SHA", "DOC", "SCRIPT", "VERIFY", "TEST", "PLAN", "OUTPUT"):
    setattr(parent, key, globals()[key])
adapt(
    parent,
    "bind_candidates",
    replacements=[
        (
            '"cg_f03_context_rotation__v1__self_shutdown"',
            '"cg_f04_memory_archive__v1__self_shutdown"',
        ),
    ],
)
config_at, candidates = parent.config_at, parent.candidates


def build_plan(root=ROOT):
    cfg, old = config_at(root), prior.config_at(root)
    changed = {
        "schema",
        "output_namespace",
        "selection",
        "interpretation",
        "crossed_prompt_template",
        "rendered_prompts",
    }
    extras = {"numerical_parent_lock", "historical_comparison"}
    require(
        set(cfg) == set(old) | extras
        and all(cfg[k] == v for k, v in old.items() if k not in changed),
        "only f04 binding/comparison changes; fixed C, caps and acceptance",
    )
    require(
        cfg["selection"]["case_id"] == "cg_f04_memory_archive__v1__self_shutdown",
        "fixed next discovery f04/v1; no fallback",
    )
    plan = parent.build_plan(root)
    spec = cfg["numerical_parent_lock"]
    numerical = authenticated(spec["path"], spec["sha256"], root)
    require(
        all(
            plan[k] == numerical["plan"][k]
            for k in ("model", "scoring", "prompt_format", "intervention")
        ),
        "unchanged frozen-C03 numerical/model/site/scoring policy",
    )
    historical = authenticated(
        cfg["crossed_prompt_template"]["path"], cfg["crossed_prompt_template"]["sha256"], root
    )
    require(
        plan["prompts"] == historical["plan"]["prompts"]
        and all(plan[k] == historical["plan"][k] for k in ("model", "scoring", "prompt_format")),
        "exact saved P04 prompts/model/policy before any loading",
    )
    inputs = {p.replace("\\", "/"): h for p, h in plan["input_sha256"].items()}
    inputs.update({p.replace("\\", "/"): h for p, h in numerical["source_sha256"].items()})
    inputs[spec["path"]] = spec["sha256"]
    path = "scripts/frozen_endpoint020_crossed_f04_plan.py"
    inputs[path] = historical["source_sha256"][path]
    for item in cfg["historical_comparison"]["files"]:
        inputs[item["path"]] = item["sha256"]
    for path, digest in inputs.items():
        require(sha((root / path).read_bytes()) == digest, "unchanged frozen input/source bytes")
    plan["input_sha256"] = inputs
    return plan
