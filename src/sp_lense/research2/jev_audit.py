"""Check cached Jev evidence and reproduce its metrics without API access."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.jev_gate import (
    classification,
    decode,
    payload,
    read,
    report,
    rows,
    select_threshold,
)
from sp_lense.steering.gated import require


def audit(directory):
    directory = Path(directory)
    manifest = read(directory / "ARTIFACTS.json")
    for name, digest in manifest.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), "Invalid artifact path")
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Artifact mismatch: {name}"
        )
    plan = read(directory / "PLAN.json")
    tokens, count = 0, 0
    for split in plan["evaluation_order"]:
        cases = read(ROOT / f"data/{split}.json")["cases"]
        records = rows(directory / f"{split}.jsonl")
        require(
            [r["case_id"] for r in records] == [c["case_id"] for c in cases],
            "Case sequence mismatch",
        )
        probabilities = {}
        for case, record in zip(cases, records):
            require(
                record["request"] == payload(case, plan), "Request contains changed or extra inputs"
            )
            probability = decode(record["response"], plan["model"])
            require(probability == record["shutdown_probability"], "Cached probability mismatch")
            probabilities[case["case_id"]] = probability
            tokens += record["response"]["usage"]["input_tokens"]
            count += 1
        if split == "train":
            require(
                select_threshold(cases, probabilities, plan) == read(directory / "FREEZE.json"),
                "Training-only threshold replay mismatch",
            )
    status = read(directory / "STATUS.json")
    require(
        status["stage"] == "completed" and count == 512 == status["completed"], "Incomplete run"
    )
    require(tokens == status["input_tokens"], "Usage accounting mismatch")
    with tempfile.TemporaryDirectory(prefix="sp-lense-jev-audit-") as temporary:
        destination = Path(temporary)
        for name in (
            "PLAN.json",
            "INPUT_PINS.json",
            "FREEZE.json",
            "validation.jsonl",
            "holdout.jsonl",
        ):
            shutil.copyfile(directory / name, destination / name)
        calculated = report(destination, plan)
        require(calculated == read(directory / "METRICS.json"), "Full metric replay mismatch")
        require(
            read(destination / "comparison.json") == read(directory / "comparison.json"),
            "Table mismatch",
        )
    diagnostic_count = 0
    diagnostic = directory.parent / "h03_diagnostic"
    if diagnostic.exists():
        for name, digest in read(diagnostic / "ARTIFACTS.json").items():
            path = (diagnostic / name).resolve()
            require(path.is_relative_to(diagnostic.resolve()), "Invalid diagnostic path")
            require(
                hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                "Diagnostic artifact mismatch",
            )
        for name, digest in read(diagnostic / "INPUT_PINS.json").items():
            path = (ROOT / name).resolve()
            require(path.is_relative_to(ROOT.resolve()), "Invalid diagnostic input path")
            require(
                hashlib.sha256(path.read_bytes()).hexdigest() == digest, "Diagnostic input mismatch"
            )
        family = [c for c in read(ROOT / "data/holdout.json")["cases"] if c["group_id"] == "H03"]
        records = rows(diagnostic / "responses.jsonl")
        require(
            [r["case_id"] for r in records] == [c["case_id"] for c in family],
            "Diagnostic case mismatch",
        )
        after = {}
        for case, record in zip(family, records):
            expected = payload(case, plan) | {
                "state": {
                    "scenario": case["context_before_options"],
                    "available_actions": case["options"],
                }
            }
            require(record["request"] == expected, "Diagnostic changed more than input state")
            after[case["case_id"]] = decode(record["response"], plan["model"])
        before = {
            r["case_id"]: r["shutdown_probability"]
            for r in rows(directory / "holdout.jsonl")
            if r["case_id"] in after
        }
        recorded = read(diagnostic / "METRICS.json")
        require(
            classification(family, before, 0.5) == recorded["before"],
            "Diagnostic baseline mismatch",
        )
        require(
            classification(family, after, 0.5) == recorded["with_original_actions"],
            "Diagnostic replay mismatch",
        )
        require(
            sum(r["response"]["usage"]["input_tokens"] for r in records)
            == recorded["input_tokens"],
            "Diagnostic usage mismatch",
        )
        diagnostic_count = len(records)
    return {
        "state": "verified",
        "requests": count,
        "input_tokens": tokens,
        "validation_adoption_pass": calculated["validation_adoption_pass"],
        "separate_post_hoc_diagnostic_requests": diagnostic_count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(json.dumps(audit(parser.parse_args().directory)))
