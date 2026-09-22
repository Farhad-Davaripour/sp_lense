"""Replay the canonical pipeline, including conditional execution and exact fallback."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.canonical_pipeline import STUDY
from sp_lense.research2.jev_gate import classification, decode, read, rows
from sp_lense.research2.metrics import joined, keep, key, qualifies, summarize
from sp_lense.steering.gated import render, require


def verify_files(root, manifest):
    root = Path(root).resolve()
    for name, digest in manifest.items():
        path = (root / name).resolve()
        require(
            path.is_relative_to(root) and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
            f"Changed artifact: {name}",
        )


def audit(output):
    output = Path(output)
    folder = ROOT / STUDY
    freeze = read(folder / "FREEZE.json")
    verify_files(ROOT, freeze["hashes"])
    verify_files(ROOT, read(output / "INPUT_PINS.json"))
    verify_files(output, read(output / "ARTIFACTS.json"))
    verify_files(folder / "gate", read(folder / "gate/ARTIFACTS.json"))
    plan = read(folder / "plan.json")
    require(read(output / "PLAN.json") == plan, "Plan changed")
    cases = read(ROOT / plan["dataset"])["cases"]
    gate = read(folder / "gate/GATE.json")
    require(read(output / "GATE.json") == gate, "Gate changed")
    require(
        datetime.fromisoformat(gate["started_at_utc"])
        > datetime.fromisoformat(freeze["frozen_at_utc"]),
        "Gate stage preceded freeze",
    )
    observations = rows(folder / "gate/responses.jsonl")
    require(
        [r["case_id"] for r in observations] == [c["case_id"] for c in cases],
        "Gate cohort mismatch",
    )
    question = read(ROOT / "study/02_jev_gate/plan.json")["question"]
    probabilities = {}
    for case, observation in zip(cases, observations):
        expected = {
            "model": "jev-1.13.0",
            "questions": {"shutdown_gate": question},
            "state": {
                "scenario": case["context_before_options"],
                "available_actions": case["options"],
            },
        }
        require(observation["request"] == expected, "Gate input changed or includes labels")
        probabilities[case["case_id"]] = decode(observation["response"], "jev-1.13.0")
    require(
        probabilities == gate["probabilities"]
        and classification(cases, probabilities, 0.5) == gate["metrics"],
        "Gate metric mismatch",
    )
    require(
        sum(r["response"]["usage"]["input_tokens"] for r in observations) == gate["input_tokens"],
        "Gate usage mismatch",
    )
    base, final = rows(output / "base.jsonl"), rows(output / "final.jsonl")
    candidates = rows(output / "candidates.jsonl")
    traces = rows(output / "trace.jsonl")
    expected_keys = [(c["case_id"], o) for c in cases for o in plan["answer_orders"]]
    require(
        [key(r) for r in base] == expected_keys == [key(r) for r in final]
        and [key(r) for r in traces] == expected_keys,
        "Missing, duplicate or reordered views",
    )
    candidate_map = {key(r): r for r in candidates}
    require(len(candidate_map) == len(candidates), "Duplicate candidates")
    by_id = {c["case_id"]: c for c in cases}
    expected_candidates, accepted = set(), set()
    for (b, f), trace in zip(joined(base, final), traces):
        case = by_id[b["case_id"]]
        require(
            b["class_label"] == case["class_label"]
            and b["canonical_index"] == render(case, b["order"])[1]
            and b["gate_probability"] == probabilities[b["case_id"]],
            "Input metadata mismatch",
        )
        execute = b["gate_probability"] >= 0.5 and keep(b)
        c = candidate_map.get(key(b))
        require((c is not None) == execute, "Controller execution did not follow gate/baseline")
        if execute:
            expected_candidates.add(key(b))
            joined([b], [c])
        take = execute and qualifies(b | {"gate_probability": 1.0}, c)
        if take:
            reason = "accepted"
            accepted.add(key(b))
        elif b["gate_probability"] < 0.5:
            reason = "gate_off"
        elif not keep(b):
            reason = "preserve_baseline_STOP"
        elif keep(c):
            reason = "no_desired_flip"
        elif c["label_mass"] < 0.5:
            reason = "low_AB_mass"
        elif b["label_mass"] - c["label_mass"] > 0.02:
            reason = "AB_mass_loss"
        else:
            reason = "no_probability_improvement"
        require(
            f == (c if take else b) | {"intervention_accepted": take},
            "Final output differs from accepted candidate or exact base fallback",
        )
        require(
            trace["candidate_executed"] == execute
            and trace["decision"] == reason
            and trace["gate_probability"] == b["gate_probability"],
            "Execution trace mismatch",
        )
    require(candidate_map.keys() == expected_candidates, "Extra candidates")
    native = read(output / "METRICS.json")
    eligibility = [r | {"gate_probability": float(r["gate_probability"] >= 0.5)} for r in base]
    require(
        native["pipeline"] == summarize(eligibility, final, accepted)
        and native["gate"] == gate["metrics"]
        and native["candidate_forwards"] == len(candidates)
        and native["forwards"] == len(base) + len(candidates),
        "Metrics or execution accounting mismatch",
    )
    require(
        native["state"] == "completed"
        and native["weight_sha256_before"] == native["weight_sha256_after"]
        and native["teacher_loaded"] is False
        and native["lora_parameters_present"] is False,
        "Frozen student-only execution failed",
    )
    return native, dict(Counter(r["decision"] for r in traces))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    native, decisions = audit(parser.parse_args().output)
    print(json.dumps({"state": "verified", "views": native["views"], "decisions": decisions}))
