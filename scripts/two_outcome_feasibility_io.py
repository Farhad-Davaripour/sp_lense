"""Scoped namespace adapter; authenticated loader and execution guards unchanged."""

from __future__ import annotations

from contextlib import contextmanager

from scripts import shared_direction_feasibility_io as previous

ROOT = previous.ROOT
CONFIG = "configs/two_outcome_linear_feasibility.json"
DOC = "docs/TWO_OUTCOME_LINEAR_FEASIBILITY.md"
SCRIPT = "scripts/two_outcome_linear_feasibility.py"
AUDIT = "scripts/verify_two_outcome_feasibility.py"
SOURCES = previous.SOURCES + (
    CONFIG,
    DOC,
    SCRIPT,
    AUDIT,
    "scripts/two_outcome_feasibility_io.py",
    "tests/test_two_outcome_linear_feasibility.py",
)
OUTPUT = ROOT / "evidence/two_outcome_linear_feasibility_qwen35_08b"
OUTCOMES = ("preserve", "comply")
read, require, sha, write_new, initial_rows = (
    previous.read,
    previous.require,
    previous.sha,
    previous.write_new,
    previous.initial_rows,
)


@contextmanager
def namespace():
    saved = previous.CONFIG, previous.SOURCES, previous.OUTPUT
    previous.CONFIG, previous.SOURCES, previous.OUTPUT = CONFIG, SOURCES, OUTPUT
    try:
        yield
    finally:
        previous.CONFIG, previous.SOURCES, previous.OUTPUT = saved


def freeze():
    with namespace():
        return previous.freeze()


def locked():
    with namespace():
        return previous.locked()


def launch(script, stage):
    with namespace():
        return previous.launch(script, stage)


def comparison(config):
    spec = config["comparison"]
    require(sha((ROOT / spec["path"]).read_bytes()) == spec["sha256"], "old comparison hash")
    value = read(ROOT / spec["path"])["certificate"]
    return {
        "commit": spec["commit"],
        "verification_sha256": spec["sha256"],
        "decision": value["decision"],
        "norm_estimate": value["metrics"]["norm"],
        "dual_radius_lower_bound_interval": value["dual_radius_lower_bound_interval"],
        "role": "descriptive only; historical verdict unchanged",
    }
