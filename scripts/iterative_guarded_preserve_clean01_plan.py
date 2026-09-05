"""Clean01 identity wrapper; original scientific plan and engine stay immutable."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from scripts import iterative_guarded_preserve_plan as parent

ROOT = parent.ROOT
CONFIG = "configs/iterative_guarded_preserve_clean01.json"
DOC = "docs/ITERATIVE_GUARDED_PRESERVE_CLEAN01.md"
SCRIPT = "scripts/iterative_guarded_preserve_clean01.py"
VERIFY = "scripts/verify_iterative_guarded_preserve_clean01.py"
TEST = "tests/test_iterative_guarded_preserve_clean01.py"
PLAN = "scripts/iterative_guarded_preserve_clean01_plan.py"
SOLVER = parent.SOLVER
OUTPUT = "evidence/iterative_guarded_preserve_f01_f02_clean01_qwen35_08b"
io, archive_io, shutil = parent.io, parent.archive_io, parent.shutil
require, sha, read, storage_preflight = (
    parent.require,
    parent.sha,
    parent.read,
    parent.storage_preflight,
)


def build_plan(root=ROOT):
    original = parent.build_plan(root)
    config = read(root / CONFIG)
    identity = config["clean_successor"]
    scientific = {k: v for k, v in config.items() if k != "clean_successor"}
    scientific["output_namespace"] = parent.OUTPUT
    require(
        scientific == original["config"] and config["output_namespace"] == OUTPUT,
        "unchanged scientific config; clean01 identity only",
    )
    require(
        identity["maximum_successors"] == 1
        and identity["no_resume"] is True
        and identity["predecessor_scientific_observations"] is False,
        "one clean successor, not resume/refinement",
    )
    ns = root / identity["predecessor_namespace"]
    require(identity["predecessor_namespace"] == parent.OUTPUT, "exact technical predecessor")
    raw = (ns / "CHECKSUMS.json").read_bytes()
    require(sha(raw) == identity["predecessor_manifest_sha256"], "predecessor manifest hash")
    manifest = json.loads(raw)
    paths = {parent.OUTPUT + "/CHECKSUMS.json": sha(raw)}
    expected = {f["path"] for f in manifest["files"]} | {"CHECKSUMS.json"}
    actual = {p.relative_to(ns).as_posix() for p in ns.rglob("*") if p.is_file()}
    require(actual == expected, "complete immutable predecessor namespace")
    for f in manifest["files"]:
        raw = (ns / f["path"]).read_bytes()
        require(len(raw) == f["bytes"] and sha(raw) == f["sha256"], "predecessor artifact bytes")
        paths[parent.OUTPUT + "/" + f["path"]] = f["sha256"]
    lock, verification, status = [
        read(ns / name) for name in ("preregistration.json", "verification.json", "RUN_STATUS.json")
    ]
    require(
        lock["plan"] == original
        and verification["status"] == "INCONCLUSIVE"
        and verification["runtime"] == status
        and status["status"] == "INCONCLUSIVE"
        and status["forward_attempts"]
        == status["completed_forwards"]
        == status["derivative_attempts"]
        == status["skipped_cells"]
        == 0
        and not any(
            (ns / name).exists()
            for name in ("runtime.json", "storage_preflight.json", "rows.jsonl", "endpoint.json")
        ),
        "predecessor pre-load failure has no scientific observation/state",
    )
    for path, digest in lock["source_sha256"].items():
        require(sha((root / path).read_bytes()) == digest, "immutable predecessor source/input")
        paths[path] = digest
    require(
        original["frozen_goals_sha256"]
        == identity["frozen_goals_sha256"]
        == "2d7a242d78ed0b63e0a2def082665b4d5c89c9d7f5fcbc2ee090135655eb46be",
        "exact original goal bytes",
    )
    return {
        **original,
        "config": config,
        "output_namespace": OUTPUT,
        "input_sha256": {**original["input_sha256"], **paths},
        "predecessor": identity,
    }


def isolated_engine(kind):
    """Execute the unchanged file in isolated globals; never monkeypatch old imports."""
    require(kind in ("runner", "audit"), "fixed engine identity")
    path = parent.SCRIPT if kind == "runner" else parent.VERIFY
    spec = importlib.util.spec_from_file_location("_sp_lense_clean01_" + kind, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.protocol = sys.modules[__name__]
    module.OUTPUT = ROOT / OUTPUT
    return module


def prelaunch_commands():
    identity = read(ROOT / CONFIG)["clean_successor"]
    source = identity["prelaunch_source"]
    return [
        ["cat-file", "-t", identity["prelaunch_object"]],
        ["rev-parse", "HEAD^{tree}"],
        ["status", "--porcelain", "--", source],
        ["diff", "--exit-code", "HEAD", "--", source],
    ]


def check_prelaunch(output, require_passed=True):
    output = Path(output)
    pre = read(output / "PRELAUNCH.json")
    claim, lock = read(output / "PRELAUNCH_CLAIM.json"), read(output / "preregistration.json")
    require(
        pre["preregistration_sha256"] == sha((output / "preregistration.json").read_bytes())
        and pre["claim_sha256"] == sha((output / "PRELAUNCH_CLAIM.json").read_bytes())
        and pre["python_executable"] == claim["python_executable"] == sys.executable
        and pre["working_directory"] == str(ROOT)
        and claim["working_directory"] == str(ROOT)
        and pre["started_monotonic"] == claim["started_monotonic"] <= pre["finished_monotonic"]
        and pre["model_calls"] == 0
        and pre["retries_allowed"] is False,
        "bound one-shot prelaunch identity",
    )
    commands = prelaunch_commands()
    require(
        [x["args"] for x in pre["git_checks"]] == commands[: len(pre["git_checks"])],
        "one ordered prelaunch check, no retry",
    )
    if pre["status"] == "passed":
        checks = pre["git_checks"]
        require(
            len(checks) == 4
            and all(x["returncode"] == 0 for x in checks)
            and checks[0]["stdout"].strip() == "tree"
            and len(checks[1]["stdout"].strip()) == 40
            and checks[2]["stdout"].strip() == checks[3]["stdout"].strip() == ""
            and pre["source_identity_passed"] is True
            and pre["environment"] == lock["environment"]
            and pre["source_commit"] == lock["source_commit"]
            and len(pre["lock_commit"]) == 40,
            "complete successful prelaunch gates",
        )
    if require_passed:
        require(pre["status"] == "passed", "prelaunch failed; no worker permitted")
    return pre
