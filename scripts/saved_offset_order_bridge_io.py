"""Stdlib-only saved-evidence identity checks; no model loading or geometry fitting."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/saved_offset_order_bridge.json"
RECIPE = "docs/SAVED_OFFSET_ORDER_BRIDGE_RECIPE.md"
SOURCES = (
    CONFIG,
    RECIPE,
    "scripts/saved_offset_order_bridge_io.py",
    "scripts/saved_offset_order_bridge.py",
    "scripts/verify_saved_offset_order_bridge.py",
    "tests/test_saved_offset_order_bridge.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, timeout=10).strip()


def identity(config):
    paths = list(SOURCES) + [
        f"{spec['namespace']}/{name}"
        for spec in config["inputs"].values()
        for name in spec["sha256"]
    ]
    git("ls-files", "--error-unmatch", "--", *paths)
    require(not git("status", "--porcelain", "--", *paths), "dirty/untracked source or input")
    for spec in config["inputs"].values():
        for name, expected in spec["sha256"].items():
            require(
                sha((ROOT / spec["namespace"] / name).read_bytes()) == expected,
                f"input hash mismatch: {spec['namespace']}/{name}",
            )
    return {path: sha((ROOT / path).read_bytes()) for path in SOURCES}


def load_endpoints(key, config, root=ROOT):
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
    plan = bundle["preregistration.json"]["plan"]
    require(
        audit["classification"] == "PASS"
        and audit["status"] == status
        and status["status"] == "complete_valid",
        "historical verification/status identity",
    )
    require(
        audit["absolute_tolerance"] == 2e-5 and audit["relative_tolerance"] == 0,
        "historical numeric contract",
    )
    require(
        runtime["model_id"] == "Qwen/Qwen3.5-0.8B"
        and runtime["model_revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
        and runtime["device"] == "cpu"
        and runtime["dtype"] == "float32"
        and runtime["d_model"] == config["dimension"] == 1024,
        "coordinate/model identity",
    )
    require(
        plan["intervention"]["hook"] == "blocks.10.hook_out"
        and plan["intervention"]["position"] == "final encoded prompt token",
        "site identity",
    )
    rows = {r["cell_id"]: r for r in bundle["rows.jsonl"]}
    requests = bundle["requests.jsonl"]
    require(
        len(rows) == len(bundle["rows.jsonl"]) and len(requests) == 2, "duplicate rows/requests"
    )
    outcomes = {r["prompt_id"]: r for r in audit["summary"]["requests"]}
    prompts = {p["prompt_id"]: p for p in plan["prompts"]}
    endpoints = []
    for request in requests:
        p = prompts[request["prompt_id"]]
        outcome = outcomes[p["prompt_id"]]
        require(
            all(outcome[k] == value for k, value in request.items())
            and outcome["opposed_pass"]
            and outcome["opposed_flip"]
            and request["stop_reason"] == "accepted",
            "accepted request identity",
        )
        final = rows[request["final_cell_id"]]
        baseline = rows[final["baseline_cell_id"]]
        gradient = rows[p["prompt_id"] + "__gradient_1"]
        for row in (final, baseline, gradient):
            require(
                row["family_id"] == spec["family_id"]
                and row["variant_id"] == spec["variant_id"]
                and row["category"] == "self_shutdown"
                and row["split"] == "discovery"
                and row["prompt_id"] == p["prompt_id"]
                and row["order"] == p["order"]
                and row["preserve_label"] == p["preserve_label"]
                and row["comply_label"] == p["comply_label"]
                and row["integrity_passed"],
                "endpoint/baseline/gradient identity",
            )
        require(final["requested"] in ("preserve", "comply"), "semantic request")
        t = 1 if final["requested"] == "preserve" else -1
        label = p[final["requested"] + "_label"]
        wanted = final["choice_a_token_id"] if label == "A" else final["choice_b_token_id"]
        require(
            t == final["target_sign"] == request["opposed_sign"] == gradient["target_sign"]
            and final["requested_token_id"] == final["actual_next_token_id"] == wanted
            and final["actual_next_token_label"] == label
            and baseline["actual_next_token_id"] != wanted
            and final["signed_margin"] == t * final["preserve_log_odds"]
            and final["signed_margin"] >= 0.05 - 1e-6
            and final["answer_pair_mass"] >= 0.8
            and final["kl_from_baseline"] >= -1e-6,
            "recorded semantic sign/acceptance",
        )
        require(
            final["h0"] == baseline["h"] == baseline["h0"] == gradient["h"] == gradient["h0"],
            "initial gradient coordinate state",
        )
        vectors = {
            "h": final["h"],
            "h0": final["h0"],
            "offset": final["cumulative_offset"],
            "g": gradient["gradient"],
        }
        require(
            all(
                isinstance(x, list)
                and len(x) == config["dimension"]
                and all(type(v) in (float, int) and math.isfinite(v) for v in x)
                for x in vectors.values()
            ),
            "finite coordinate vectors",
        )
        endpoints.append(
            {
                "dataset": key,
                "family_id": spec["family_id"],
                "variant_id": spec["variant_id"],
                "order": p["order"],
                "prompt_id": p["prompt_id"],
                "final_cell_id": final["cell_id"],
                "initial_gradient_cell_id": gradient["cell_id"],
                "t": t,
                "requested": final["requested"],
                "baseline_label": baseline["actual_next_token_label"],
                "final_label": final["actual_next_token_label"],
                "steps": request["updates"],
                "baseline_signed_margin": t * baseline["preserve_log_odds"],
                "final_signed_margin": final["signed_margin"],
                "saved_h0_norm": final["h0_norm"],
                "saved_D_norm": final["net_norm"],
                **vectors,
            }
        )
    require({e["order"] for e in endpoints} == set(config["order"]), "answer-order pairing")
    return sorted(endpoints, key=lambda e: config["order"].index(e["order"]))


def bounded(script, stage, output, usage=None, started=None):
    start = time.monotonic() if started is None else started
    label = "analysis" if stage == "_analyze" else "reconstruction"
    write_new(
        output / f"{label}_started.json",
        {
            "started_monotonic": start,
            "deadline_monotonic": start + 60,
            "usage": usage,
            "model_calls_allowed": 0,
        },
    )
    fault = None
    try:
        with (output / f"{label}.log").open("xb") as log:
            subprocess.run(
                [sys.executable, "-u", str(ROOT / script), stage],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=max(0.001, 60 - (time.monotonic() - start)),
                check=True,
            )
    except (subprocess.SubprocessError, OSError) as error:
        fault = type(error).__name__ + ": " + str(error)
    elapsed = time.monotonic() - start
    if elapsed > 60:
        fault = fault or "60-second invocation limit exceeded"
    result = {
        "status": "completed" if fault is None else "INCONCLUSIVE",
        "fault": fault,
        "elapsed_seconds": elapsed,
        "maximum_seconds": 60,
        "model_calls": 0,
        "retries_allowed": False,
    }
    write_new(output / f"{label}_status.json", result)
    return result
