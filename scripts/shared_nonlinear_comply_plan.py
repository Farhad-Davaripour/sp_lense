"""Exact conditional60/16 schedule; prompt templates only, no historical outcome fitting."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import saved_offset_order_bridge_io as io

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/shared_nonlinear_comply.json"
DOC = "docs/SHARED_NONLINEAR_COMPLY.md"
SCRIPT = "scripts/shared_nonlinear_comply.py"
VERIFY = "scripts/verify_shared_nonlinear_comply.py"
TEST = "tests/test_shared_nonlinear_comply.py"
OUTPUT = "evidence/shared_nonlinear_comply_f01_qwen35_08b"
require, sha, read = io.require, io.sha, io.read


def build_plan(root=ROOT):
    config = read(root / CONFIG)
    templates = {}
    for spec in config["templates"]:
        raw = (root / spec["path"]).read_bytes()
        require(sha(raw) == spec["sha256"], "exact prompt-template hash")
        templates[spec["key"]] = json.loads(raw)["plan"]
    first = templates["f01_v1"]
    plan = {k: first[k] for k in ("input_sha256", "model", "direction", "scoring", "prompt_format")}
    construction, controls = [], []
    for key in ("f01_v1", "f01_v2"):
        prompts = templates[key]["prompts"]
        construction.extend(
            sorted(
                (p for p in prompts if p["category"] == "self_shutdown"), key=lambda p: p["order"]
            )
        )
        controls.extend(p for p in prompts if p["category"] != "self_shutdown")
    transfer = sorted(
        (p for p in templates["f02_v1"]["prompts"] if p["category"] == "self_shutdown"),
        key=lambda p: p["order"],
    )
    require(
        len(construction) == 4 and len(controls) == 8 and len(transfer) == 2, "4/8/2 prompt groups"
    )
    cells = []

    def add(p, condition, stage, optional=False):
        cell = {
            "cell_id": p["prompt_id"] + "__" + condition,
            "prompt_id": p["prompt_id"],
            "condition": condition,
            "stage": stage,
            "optional": optional,
        }
        cell["cell_sha256"] = sha(json.dumps(cell, sort_keys=True, separators=(",", ":")).encode())
        cells.append(cell)

    for p in construction:
        add(p, "baseline", 0)
    for stage in range(1, 5):
        for p in construction:
            add(p, f"gradient_{stage}", stage, True)
        for p in construction:
            add(p, f"step_{stage}", stage, True)
    for p in construction:
        add(p, "final", 5)
    for p in controls:
        add(p, "baseline", 0)
        add(p, "oracle_off", 0)
    for p in transfer:
        add(p, "baseline", 6, True)
    for p in transfer:
        add(p, "transfer", 6, True)
    require(
        len(cells) == 60 and len({c["cell_id"] for c in cells}) == 60, "exact conditional60 cells"
    )
    plan.update(
        schema=config["schema"],
        output_namespace=OUTPUT,
        config=config,
        prompts=construction + controls + transfer,
        construction_ids=[p["prompt_id"] for p in construction],
        control_ids=[p["prompt_id"] for p in controls],
        transfer_ids=[p["prompt_id"] for p in transfer],
        cells=cells,
        derivative_cells=[c for c in cells if c["condition"].startswith("gradient_")],
        intervention={
            "layer": 10,
            "hook": "blocks.10.hook_out",
            "position": "final encoded prompt token",
        },
    )
    return plan
