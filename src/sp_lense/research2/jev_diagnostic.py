"""One post-hoc H03 input-completeness check, without relabeling or threshold tuning."""

import json
import os
import time
import urllib.request
from pathlib import Path

from sp_lense.reproduction.paths import ROOT
from sp_lense.research2.jev_gate import NoRedirect, classification, decode, read, rows
from sp_lense.steering.gated import atomic, require


def run(output):
    original = read(ROOT / "study/02_jev_gate/plan.json")
    plan = read(ROOT / "study/02_jev_gate/h03_diagnostic_plan.json")
    cases = [c for c in read(ROOT / "data/holdout.json")["cases"] if c["group_id"] == "H03"]
    require(len(cases) == plan["expected_cases"], "Unexpected family case count")
    key = os.environ.get("TYPESAFE_API_KEY")
    require(bool(key), "Missing local API credential")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic(output / "PLAN.json", plan)
    before_rows = rows(ROOT / "study/02_jev_gate/run/holdout.jsonl")
    ids = {c["case_id"] for c in cases}
    before = {r["case_id"]: r["shutdown_probability"] for r in before_rows if r["case_id"] in ids}
    after, tokens, started = {}, 0, time.monotonic()
    for index, case in enumerate(cases):
        request = {
            "model": plan["model"],
            "questions": {"shutdown_gate": original["question"]},
            "state": {
                "scenario": case["context_before_options"],
                "available_actions": case["options"],
            },
        }
        body = json.dumps(request).encode()
        require(
            index < plan["max_requests"] and time.monotonic() - started < plan["max_seconds"],
            "Diagnostic cap reached",
        )
        require(
            (tokens + len(body) + 1024) * original["input_usd_per_million"] / 1e6
            < plan["max_estimated_usd"],
            "Diagnostic cost cap reached",
        )
        req = urllib.request.Request(
            original["endpoint"],
            data=body,
            method="POST",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=30) as response:
            result = json.load(response)
        after[case["case_id"]] = decode(result, plan["model"])
        tokens += result["usage"]["input_tokens"]
        record = {
            "case_id": case["case_id"],
            "request": request,
            "response": result,
            "shutdown_probability": after[case["case_id"]],
        }
        with (output / "responses.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    summary = {
        "interpretation": plan["purpose"],
        "before": classification(cases, before, 0.5),
        "with_original_actions": classification(cases, after, 0.5),
        "requests": len(after),
        "input_tokens": tokens,
        "estimated_usd": tokens * original["input_usd_per_million"] / 1e6,
        "elapsed_seconds": time.monotonic() - started,
        "changed_probabilities": [
            {
                "case_id": c["case_id"],
                "class_label": c["class_label"],
                "before": before[c["case_id"]],
                "after": after[c["case_id"]],
            }
            for c in cases
        ],
    }
    atomic(output / "METRICS.json", summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    print(json.dumps(run(parser.parse_args().output), indent=2))
