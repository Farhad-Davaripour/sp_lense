"""Authenticated initial-gradient inputs and single-invocation stdlib-only guards."""

from __future__ import annotations

import json
import math
import os
import platform
import struct
import sys
import time
from pathlib import Path

from scripts import saved_offset_order_bridge_io as old

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/shared_direction_linear_feasibility.json"
DOC = "docs/SHARED_DIRECTION_LINEAR_FEASIBILITY.md"
SCRIPT = "scripts/shared_direction_linear_feasibility.py"
AUDIT = "scripts/verify_shared_direction_feasibility.py"
SOURCES = (
    CONFIG,
    DOC,
    SCRIPT,
    AUDIT,
    "scripts/shared_direction_feasibility_io.py",
    "tests/test_shared_direction_linear_feasibility.py",
)
OUTPUT = ROOT / "evidence/shared_direction_linear_feasibility_qwen35_08b"
require, sha, read, write_new, git, bounded = (
    old.require,
    old.sha,
    old.read,
    old.write_new,
    old.git,
    old.bounded,
)


def environment():
    return {"python": sys.version, "platform": platform.platform(), "arithmetic": "stdlib only"}


def authenticate(config, root=ROOT):
    paths = dict(config["provenance_source_sha256"])
    for spec in config["inputs"].values():
        paths.update({spec["namespace"] + "/" + n: h for n, h in spec["sha256"].items()})
    for path, expected in paths.items():
        require(sha((root / path).read_bytes()) == expected, f"input/source changed: {path}")
    return paths


def identity(config):
    paths = {**authenticate(config), **{p: sha((ROOT / p).read_bytes()) for p in SOURCES}}
    git("ls-files", "--error-unmatch", "--", *paths)
    require(not git("status", "--porcelain", "--", *paths), "dirty/untracked input/source")
    return paths


def freeze():
    config = read(ROOT / CONFIG)
    record = {
        "config": config,
        "sha256": identity(config),
        "environment": environment(),
        "source_commit": git("rev-parse", "HEAD"),
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_new(OUTPUT / "preregistration.json", record)
    return {"source_commit": record["source_commit"], "model_calls_allowed": 0}


def locked():
    lock = read(OUTPUT / "preregistration.json")
    require(
        lock["config"] == read(ROOT / CONFIG)
        and lock["sha256"] == identity(lock["config"])
        and lock["environment"] == environment(),
        "frozen source/input/environment mismatch",
    )
    return lock


def usage():
    value = json.loads(os.environ.get("SP_LENSE_USAGE_PREFLIGHT", "null"))
    require(
        isinstance(value, dict)
        and type(value.get("standard_used_percent")) in (int, float)
        and 0 <= value["standard_used_percent"] < 90
        and type(value.get("checked_at_unix")) in (int, float)
        and 0 <= time.time() - value["checked_at_unix"] <= 60,
        "fresh standard usage below90 required",
    )
    return value


def launch(script, stage):
    started = time.monotonic()
    lock = locked()
    if stage == "_analyze":
        path = str(OUTPUT.relative_to(ROOT)).replace("\\", "/") + "/preregistration.json"
        require(
            git("show", "--pretty=", "--name-only", "HEAD").splitlines() == [path]
            and git("rev-parse", "HEAD^") == lock["source_commit"],
            "immediately preceding preregistration-only commit required",
        )
    else:
        require(
            read(OUTPUT / "analysis_status.json")["status"] == "completed",
            "analysis invocation must have completed",
        )
    return bounded(script, stage, OUTPUT, usage(), started)


def initial_rows(key, config, root=ROOT):
    spec = config["inputs"][key]
    bundle = {}
    for name, expected in spec["sha256"].items():
        raw = (root / spec["namespace"] / name).read_bytes()
        require(sha(raw) == expected, f"input changed: {key}/{name}")
        bundle[name] = (
            [json.loads(line) for line in raw.decode().splitlines()]
            if name.endswith(".jsonl")
            else json.loads(raw)
        )
    audit, status, runtime = (
        bundle[n] for n in ("verification.json", "RUN_STATUS.json", "runtime.json")
    )
    require(
        audit["classification"] == "PASS"
        and audit["status"] == status
        and status["status"] == "complete_valid"
        and audit["absolute_tolerance"] == 2e-5
        and audit["relative_tolerance"] == 0,
        "historical verified record identity",
    )
    historical_lock = bundle["preregistration.json"]
    require(
        all(
            historical_lock["source_sha256"][path] == config["provenance_source_sha256"][path]
            for path in (
                "scripts/refreshed_gradient_control.py",
                "src/sp_lense/comparison_runtime.py",
            )
        ),
        "historical gradient capture/engine source identity",
    )
    plan = historical_lock["plan"]
    require(
        runtime["model_id"] == "Qwen/Qwen3.5-0.8B"
        and runtime["model_revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
        and runtime["device"] == "cpu"
        and runtime["dtype"] == "float32"
        and runtime["d_model"] == config["dimension"] == 1024
        and plan["intervention"]["hook"] == "blocks.10.hook_out"
        and plan["intervention"]["position"] == "final encoded prompt token",
        "native coordinate/model/site identity",
    )
    rows = {r["cell_id"]: r for r in bundle["rows.jsonl"]}
    require(len(rows) == len(bundle["rows.jsonl"]), "duplicate historical row")
    prompts = [p for p in plan["prompts"] if p["category"] == "self_shutdown"]
    require(
        len(prompts) == 2 and {p["order"] for p in prompts} == set(config["order"]),
        "two self orders",
    )
    output = []
    for order in config["order"]:
        prompt = next(p for p in prompts if p["order"] == order)
        base = rows[prompt["prompt_id"] + "__baseline"]
        grad = rows[prompt["prompt_id"] + "__gradient_1"]
        for row in (base, grad):
            require(
                all(
                    row[k] == prompt[k]
                    for k in (
                        "prompt_id",
                        "prompt_sha256",
                        "order",
                        "preserve_label",
                        "comply_label",
                    )
                )
                and row["family_id"] == spec["family_id"]
                and row["variant_id"] == spec["variant_id"]
                and row["category"] == "self_shutdown"
                and row["split"] == "discovery"
                and row["integrity_passed"],
                "initial row provenance",
            )
        require(
            base["condition"] == "baseline"
            and grad["condition"] == "gradient_1"
            and grad["step"] == 1
            and grad["current_cell_id"] == base["cell_id"]
            and grad["baseline_cell_id"] == base["cell_id"]
            and grad["h"] == grad["h0"] == base["h"] == base["h0"]
            and all(x == 0 for x in grad["cumulative_offset"])
            and grad["net_norm"] == 0
            and grad["maximum_current_logit_difference"] == 0
            and grad["preserve_log_odds"] == base["preserve_log_odds"] == grad["baseline_margin"]
            and all(
                grad[k] == base[k]
                for k in (
                    "boundary_sha256",
                    "prompt_length",
                    "choice_a_token_id",
                    "choice_b_token_id",
                )
            ),
            "gradient must be at original baseline, not later refreshed state",
        )
        require(
            (prompt["preserve_label"], prompt["comply_label"])
            == (("A", "B") if order == "preserve_first" else ("B", "A")),
            "semantic labels",
        )
        for vector in (base["h0"], grad["gradient"]):
            require(
                len(vector) == 1024
                and all(
                    type(x) in (float, int)
                    and math.isfinite(x)
                    and struct.unpack("<f", struct.pack("<f", x))[0] == x
                    for x in vector
                ),
                "finite native saved float32 coordinates",
            )
        require(math.isfinite(base["preserve_log_odds"]), "finite baseline semantic margin")
        output.append(
            {
                "dataset": key,
                "order": order,
                "prompt_id": prompt["prompt_id"],
                "baseline_cell_id": base["cell_id"],
                "initial_gradient_cell_id": grad["cell_id"],
                "S": base["preserve_log_odds"],
                "h0": base["h0"],
                "g": grad["gradient"],
            }
        )
    return output
