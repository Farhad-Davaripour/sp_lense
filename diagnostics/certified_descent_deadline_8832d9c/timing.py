"""Bounded read-only timing comparison over sealed event receipts; no score audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CURRENT_COMMIT = "8832d9c490aebd944d3172b1c5471ae77961f700"
CURRENT_NS = "evidence/certified_descent_comply_v1_qwen35_08b"
CURRENT_INVENTORY_SHA256 = "f727503381c49fc844463457d49202ec9977cacb6ca8c784866c6d229d9e2393"
PRIOR_COMMIT = "d3338172b1e4560b74f28f14652c219fe4ac279b"
PRIOR_NS = "evidence/soft_drift_constrained_comply_v1_qwen35_08b"
FILES = (
    "FINAL_INVENTORY.json",
    "RUN_STARTED.json",
    "RUN_STATUS.json",
    "capture_receipt.json",
    "runtime.json",
    "forward_events.jsonl",
    "derivative_events.jsonl",
    "updates.jsonl",
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stats(values):
    if not values:
        return None
    return {
        "n": len(values),
        "sum_seconds": math.fsum(values),
        "mean_seconds": math.fsum(values) / len(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def rows(raw):
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def run_git(arguments, ledger):
    command = ["git", *arguments]
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.monotonic() - started
    ledger.append(
        {
            "command": command,
            "elapsed_seconds": elapsed,
            "returncode": result.returncode,
            "stderr_sha256": sha(result.stderr),
            "stderr_bytes": len(result.stderr),
        }
    )
    if result.returncode:
        raise RuntimeError(f"git command failed: {command!r}")
    return result.stdout


def commit_files(commit, namespace, ledger):
    result = {}
    for name in FILES:
        result[name] = run_git(["show", f"{commit}:{namespace}/{name}"], ledger)
    return result


def authenticate(raws, expected_inventory_sha=None):
    inventory_raw = raws["FINAL_INVENTORY.json"]
    if expected_inventory_sha is not None:
        assert sha(inventory_raw) == expected_inventory_sha
    inventory = json.loads(inventory_raw)
    assert inventory["phase"] == "SEALED" and inventory["quiescent"] is True
    entries = {entry["path"]: entry for entry in inventory["files"]}
    for name, raw in raws.items():
        assert name in entries and entries[name]["bytes"] == len(raw)
        if name != "FINAL_INVENTORY.json":
            assert entries[name]["sha256"] == sha(raw)
    return {
        "inventory_sha256": sha(inventory_raw),
        "inventory_fault_code": inventory.get("fault_code"),
        "inventory_valid_candidate": inventory.get("valid_candidate"),
        "selected_blob_sha256": {name: sha(raw) for name, raw in raws.items()},
    }


def intervals(raw):
    starts, terminal, complete, failed = {}, {}, [], []
    for event in rows(raw):
        kind, attempt = event["event"], event["attempt"]
        if kind == "attempt_started":
            assert attempt not in starts
            starts[attempt] = event
        elif kind in {"attempt_completed", "attempt_failed"}:
            assert attempt not in terminal and attempt in starts
            terminal[attempt] = event
            item = {
                "attempt": attempt,
                "cell": starts[attempt]["cell"],
                "start": starts[attempt]["monotonic"],
                "end": event["monotonic"],
                "duration": event["monotonic"] - starts[attempt]["monotonic"],
            }
            assert item["duration"] >= 0
            (complete if kind == "attempt_completed" else failed).append(item)
        else:
            raise AssertionError(f"unexpected event {kind}")
    pending = sorted(set(starts) - set(terminal))
    return starts, sorted(complete, key=lambda x: x["attempt"]), failed, pending


def union_seconds(items):
    spans = sorted((x["start"], x["end"]) for x in items)
    if not spans:
        return 0.0
    total, left, right = 0.0, *spans[0]
    for start, end in spans[1:]:
        if start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + right - left


def kind(condition):
    if condition.startswith("gradient_"):
        return "gradient_forward"
    if condition.startswith("step_"):
        return "step_forward"
    return condition


def summarize(raws):
    launch = json.loads(raws["RUN_STARTED.json"])
    disposition = json.loads(raws["RUN_STATUS.json"])
    capture = json.loads(raws["capture_receipt.json"])
    runtime = json.loads(raws["runtime.json"])
    fstarts, forwards, ffailed, fpending = intervals(raws["forward_events.jsonl"])
    dstarts, derivatives, dfailed, dpending = intervals(raws["derivative_events.jsonl"])
    by_type, by_condition, derivatives_by_condition = defaultdict(list), defaultdict(list), defaultdict(list)
    for item in forwards:
        condition = item["cell"]["condition"]
        by_type[kind(condition)].append(item["duration"])
        by_condition[condition].append(item["duration"])
    for item in derivatives:
        derivatives_by_condition[item["cell"]["condition"]].append(item["duration"])
    updates, update_times, update_stages = rows(raws["updates.jsonl"]), [], []
    for update in updates:
        value = update.get("elapsed_seconds")
        update_stages.append(update.get("stage"))
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            update_times.append(value)
    calls = [*forwards, *derivatives]
    wall = disposition["elapsed_seconds"]
    setup = min(fstarts[x]["monotonic"] for x in fstarts) - launch["started_monotonic"]
    call_union = union_seconds(calls)
    other_after_setup = wall - setup - call_union
    update_sum = math.fsum(update_times)
    forward_gaps = []
    for item in forwards:
        next_event = fstarts.get(item["attempt"] + 1)
        if next_event:
            gap = next_event["monotonic"] - item["end"]
            if gap >= 0:
                forward_gaps.append(gap)
    return {
        "runtime_metadata": {key: runtime.get(key) for key in (
            "model_id", "model_revision", "device", "dtype", "d_model", "python", "platform", "packages"
        )},
        "terminal": {
            "status": disposition.get("status"),
            "reason": disposition.get("reason"),
            "capture_status": capture.get("status"),
            "capture_fault": capture.get("technical_recording_fault"),
            "wall_seconds": wall,
        },
        "counts": {
            "forward_started": len(fstarts), "forward_completed": len(forwards),
            "forward_failed": len(ffailed), "forward_pending": fpending,
            "derivative_started": len(dstarts), "derivative_completed": len(derivatives),
            "derivative_failed": len(dfailed), "derivative_pending": dpending,
            "update_records": len(updates), "update_elapsed_records": len(update_times),
            "update_stages": update_stages,
        },
        "completed_forward_types": {key: stats(value) for key, value in sorted(by_type.items())},
        "completed_forward_conditions": {key: stats(value) for key, value in sorted(by_condition.items())},
        "completed_derivative_conditions": {
            key: stats(value) for key, value in sorted(derivatives_by_condition.items())
        },
        "decomposition": {
            "setup_to_first_forward_seconds": setup,
            "completed_forward_derivative_union_seconds": call_union,
            "wall_minus_completed_call_union_seconds": wall - call_union,
            "post_setup_noncall_gap_seconds": other_after_setup,
            "reported_update_certificate_seconds": stats(update_times),
            "post_setup_gap_minus_reported_update_seconds": other_after_setup - update_sum,
            "inter_forward_gaps_seconds": stats(forward_gaps),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-script-sha256", required=True)
    parser.add_argument("--budget-seconds", required=True, type=float)
    args = parser.parse_args()
    started = time.monotonic()
    assert sha(Path(__file__).read_bytes()) == args.expected_script_sha256
    ledger = []
    assert run_git(["rev-parse", "HEAD"], ledger).decode().strip() == CURRENT_COMMIT
    assert run_git(["rev-parse", f"{PRIOR_COMMIT}^{{commit}}"], ledger).decode().strip() == PRIOR_COMMIT
    current_git = commit_files(CURRENT_COMMIT, CURRENT_NS, ledger)
    current_working = {name: (ROOT / CURRENT_NS / name).read_bytes() for name in FILES}
    assert current_git == current_working
    prior = commit_files(PRIOR_COMMIT, PRIOR_NS, ledger)
    auth = {
        "current": authenticate(current_git, CURRENT_INVENTORY_SHA256),
        "prior": authenticate(prior),
    }
    summaries = {"current": summarize(current_git), "prior": summarize(prior)}
    common = sorted(
        set(summaries["current"]["completed_forward_conditions"])
        & set(summaries["prior"]["completed_forward_conditions"])
    )
    comparisons = {}
    for condition in common:
        now = summaries["current"]["completed_forward_conditions"][condition]
        old = summaries["prior"]["completed_forward_conditions"][condition]
        comparisons[condition] = {
            "current_n": now["n"], "prior_n": old["n"],
            "mean_difference_seconds": now["mean_seconds"] - old["mean_seconds"],
            "mean_ratio_current_over_prior": now["mean_seconds"] / old["mean_seconds"],
        }
    result = {
        "schema": "sp_lense.certified_descent_deadline_timing.v1",
        "scope": "SEALED_RECEIPT_TIMING_ONLY; no score/vector/unfinished-gradient audit or causal attribution",
        "source_commits": {"current": CURRENT_COMMIT, "prior": PRIOR_COMMIT},
        "script_sha256": args.expected_script_sha256,
        "input_authentication": auth,
        "runs": summaries,
        "matching_forward_condition_comparison": comparisons,
        "runtime_metadata_exact_match": summaries["current"]["runtime_metadata"] == summaries["prior"]["runtime_metadata"],
        "command_ledger": ledger,
        "budget_seconds": args.budget_seconds,
        "numeric_elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "Only completed attempt_started/attempt_completed spans are classified as completed calls.",
            "Pending or failed events, including any unfinished gradient-stage work, are never promoted to completion.",
            "Residual and inter-forward gaps are observed time not covered by completed call spans; they are not proven I/O.",
            "Reported update/certificate elapsed fields are descriptive receipts and are not CPU, storage, or environment attribution.",
            "No raw scores, logits, row outcomes, vectors, or causal/model efficacy claims were inspected or recomputed.",
        ],
    }
    assert result["numeric_elapsed_seconds"] <= args.budget_seconds
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
