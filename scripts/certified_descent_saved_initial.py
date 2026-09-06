"""Retrospective initial-input numerical check only; never export a candidate.

`extract` authenticates and assembles immutable saved initial observations but
does not import/call the dyadic solver. `evaluate` makes one frozen call, under an
external hard10s subprocess ceiling. All stdout remains provisional until exit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_COMMIT = "d3338172b1e4560b74f28f14652c219fe4ac279b"
ROWS = "evidence/soft_drift_constrained_comply_v1_qwen35_08b/rows.jsonl"
UPDATES = "evidence/soft_drift_constrained_comply_v1_qwen35_08b/updates.jsonl"
INPUTS = "docs/certified_descent_saved_initial_inputs.json"
LOCK = "docs/certified_descent_saved_initial_lock.json"
REQUIRED_SOURCES = {
    "scripts/certified_descent_dyadic_solver.py",
    "scripts/verify_certified_descent_dyadic.py",
    "scripts/partial_progress_deficit_solver.py",
    "scripts/verify_partial_progress_deficit.py",
    "scripts/paired_common_drift_comply_optimizer.py",
    "scripts/soft_drift_qp_solver.py",
    "scripts/certified_descent_saved_initial.py",
    INPUTS,
    "docs/CERTIFIED_DESCENT_INTEGRATION_JOB_PROTOCOL.md",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def extract(*, include_problem=False):
    from scripts import paired_common_drift_comply_optimizer as binding
    from scripts.soft_drift_qp_solver import assemble

    saved_blob = subprocess.check_output(
        ["git", "rev-parse", PROVENANCE_COMMIT + ":" + ROWS], cwd=ROOT, text=True
    ).strip()
    current_blob = subprocess.check_output(
        ["git", "hash-object", "--no-filters", "--", ROWS], cwd=ROOT, text=True
    ).strip()
    assert current_blob == saved_blob, "saved_initial_rows_not_immutable"
    with (ROOT / ROWS).open("rb") as handle:
        raw_lines = [handle.readline() for _ in range(24)]
    assert all(raw_lines), "initial_prefix_incomplete"
    initial = [json.loads(line) for line in raw_lines]
    gradients, baseline = binding._bound_rows(initial[12:], initial[:12])
    for row, base in zip(gradients, baseline, strict=True):
        assert row["condition"] == "gradient_1" and row["stage"] == 1
        assert row["cell_id"] == row["prompt_id"] + "__gradient_1"
        assert row["current_cell_id"] == base["cell_id"]
        assert row["maximum_current_logit_difference"] == 0
        assert row["maximum_current_h_difference"] == 0
        assert row["preserve_log_odds"] == base["preserve_log_odds"]
        assert all(x == 0 for x in row["shared_w"]) and row["path_norm"] == 0
        binding.vector(row["gradient"])
    problem = assemble(
        [row["h0_norm"] for row in gradients],
        [row["gradient"] for row in gradients],
        [-row["preserve_log_odds"] for row in gradients],
        [-row["preserve_log_odds"] for row in baseline],
    )
    w = list(gradients[0]["shared_w"])
    problem.update(w=w, w_sha256=binding.vector_sha(w), history=[w[:]], path_upper="0")
    assert problem["c"] == [0.0] * 6
    saved_update_blob = subprocess.check_output(
        ["git", "rev-parse", PROVENANCE_COMMIT + ":" + UPDATES], cwd=ROOT, text=True
    ).strip()
    current_update_blob = subprocess.check_output(
        ["git", "hash-object", "--no-filters", "--", UPDATES], cwd=ROOT, text=True
    ).strip()
    assert saved_update_blob == current_update_blob, "saved_initial_update_not_immutable"
    with (ROOT / UPDATES).open("rb") as handle:
        raw_update = handle.readline()
    update = json.loads(raw_update)
    assert update["stage"] == 1 and update["path_before"] == 0
    assert update["w_before"] == w and update["w_before_sha256"] == problem["w_sha256"]
    assert update["gradient_cell_ids"] == [row["cell_id"] for row in gradients]
    assert update["qp_inputs"]["A_row_sha256"] == [binding.vector_sha(row) for row in problem["A"]]
    assert update["qp_inputs"]["b"] == problem["b"] and update["qp_inputs"]["c"] == problem["c"]
    raw_problem = canonical(problem)
    receipt = {
        "schema": "sp_lense.certified_descent_saved_initial_inputs.v1",
        "scope": "POSTHOC_LOCAL_SURROGATE_ONLY_NO_MODEL_RESULT_NO_WARMSTART",
        "provenance_commit": PROVENANCE_COMMIT,
        "original_runtime_source_commit": "3764c93321c8ba6a9012b4787211298b566063e4",
        "raw_rows_path": ROWS,
        "raw_rows_git_blob": saved_blob,
        "raw_rows_sha256": sha((ROOT / ROWS).read_bytes()),
        "raw_updates_path": UPDATES,
        "raw_updates_git_blob": saved_update_blob,
        "initial_update_raw_line_sha256": sha(raw_update),
        "selected_line_numbers": list(range(1, 25)),
        "selected_raw_line_sha256": [sha(line) for line in raw_lines],
        "gradient_cell_ids": [row["cell_id"] for row in gradients],
        "own_h0_norms": [row["h0_norm"] for row in gradients],
        "gradient_float64_sha256": [binding.vector_sha(row["gradient"]) for row in gradients],
        "initial_comply_margins": [-row["preserve_log_odds"] for row in gradients],
        "derived_A_row_sha256": [binding.vector_sha(row) for row in problem["A"]],
        "assembly": "Original binary64 assemble: A=-own_h0_norm*gS; b=.10-m; m=-S; c=0",
        "derived_input_storage": "Immutable raw references; reconstruct identically and verify problem hash",
        "problem_sha256": sha(raw_problem.encode()),
        "model_calls": 0,
        "solver_calls": 0,
    }
    return (receipt, problem) if include_problem else receipt


def evaluate():
    started = time.monotonic()
    lock = json.loads((ROOT / LOCK).read_bytes())
    assert lock["attempts"] == 1 and lock["hard_seconds"] == 10
    assert set(lock["source_sha256"]) == REQUIRED_SOURCES, "incomplete_frozen_source_set"
    for path, expected in lock["source_sha256"].items():
        assert sha((ROOT / path).read_bytes()) == expected, path
    saved = json.loads((ROOT / INPUTS).read_bytes())
    reconstructed, problem = extract(include_problem=True)
    assert reconstructed["problem_sha256"] == lock["problem_sha256"] == saved["problem_sha256"]
    for key in (
        "raw_rows_git_blob",
        "raw_rows_sha256",
        "selected_raw_line_sha256",
        "raw_updates_git_blob",
        "initial_update_raw_line_sha256",
    ):
        assert reconstructed[key] == saved[key], key
    from scripts.certified_descent_dyadic_solver import solve

    result = solve(problem, lock["problem_sha256"])
    # This is a labelled numerical receipt, not a portable intervention endpoint.
    # Keep exact certificate scalars and hashes, never the proposed/selected vector.
    for trial in result["trials"]:
        if trial.get("certificate") is not None:
            trial["certificate"].pop("actual_displacement", None)
    if result.get("selected_certificate") is not None:
        result["selected_certificate"].pop("actual_displacement", None)
    proposal = result.pop("proposal", None)
    result.pop("w_next", None)
    if proposal and proposal.get("p"):
        from scripts.certified_descent_dyadic_solver import vector_sha256

        result["proposal_sha256"] = vector_sha256(proposal["p"])
    out = {
        "scope": "POSTHOC_LOCAL_SURROGATE_ONLY",
        "external_completion_required": True,
        "candidate_exported": False,
        "saved_step_reuse_forbidden": True,
        "new_model_result": False,
        "result": result,
    }
    encoded = canonical(out)
    if time.monotonic() - started >= 10:
        raise TimeoutError("combined_saved_input_evaluation_deadline")
    print(encoded, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("extract", "evaluate"))
    args = parser.parse_args()
    started = time.perf_counter()
    if args.mode == "extract":
        value = extract()
        value["extraction_elapsed_seconds"] = time.perf_counter() - started
        print(canonical(value), flush=True)
    else:
        evaluate()


if __name__ == "__main__":
    main()
