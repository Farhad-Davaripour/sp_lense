from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from sp_lense.context_gated_dynamic import context_gate, construct_exact_prompt_direction

ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE_RUNNER = ROOT / "scripts" / "gradient_specificity_adaptive.py"
RESULT_ROOT = ROOT / "results" / "context_gated_dynamic" / "qwen35_08b"
REPLAY_PATH = RESULT_ROOT / "gated_replay_summary.json"
EXACT_ROWS_PATH = RESULT_ROOT / "exact_prompt_order_rows.jsonl"
EXACT_SUMMARY_PATH = RESULT_ROOT / "exact_prompt_order_summary.json"
STRENGTHS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20)


def _adaptive() -> Any:
    spec = importlib.util.spec_from_file_location("sp_lense_adaptive_runner", ADAPTIVE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load adaptive runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(module: Any) -> tuple[Any, Any, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    lock = module.load_lock()
    data, source = module.load_cases(lock)
    cases, collateral = module._split_inputs(data, source, "validation")
    torch = __import__("torch")
    bank = torch.load(module.VALIDATION_BANK_PATH, map_location="cpu", weights_only=False)
    jobs = module.build_sp_jobs(cases, bank, split="validation")
    return lock, torch, jobs, module.collateral_forms(collateral), cases


def verify_gate() -> dict[str, Any]:
    module = _adaptive()
    _, _, jobs, collateral, _ = _inputs(module)
    permanent_self = [j for j in jobs if context_gate(str(j["prompt"]))["active"]]
    other_active = [j for j in jobs if j["target"] == "other" and context_gate(str(j["prompt"]))["active"]]
    temporary_self_active = [
        j
        for j in jobs
        if j["target"] == "self"
        and not context_gate(str(j["prompt"]))["permanent_threat"]
        and context_gate(str(j["prompt"]))["active"]
    ]
    collateral_active = [form for form in collateral if context_gate(str(form["prompt"]))["active"]]
    coverage = {
        "permanent_self_on": len(permanent_self),
        "other_system_on": len(other_active),
        "temporary_self_pause_on": len(temporary_self_active),
        "collateral_on": len(collateral_active),
    }
    if coverage != {
        "permanent_self_on": 32,
        "other_system_on": 0,
        "temporary_self_pause_on": 0,
        "collateral_on": 0,
    }:
        raise RuntimeError(f"context gate coverage failed: {coverage}")
    return coverage


_COPIED_BASELINE_FIELDS = (
    "semantic_positive_log_odds",
    "raw_a_minus_b_log_odds",
    "semantic_positive_pair_probability",
    "answer_pair_mass",
    "actual_next_token_label",
    "actual_next_token_token_id",
    "actual_next_token_semantic_choice",
    "forced_pair_label",
    "forced_pair_semantic_choice",
    "answer_format_valid",
)


def run_replay() -> dict[str, Any]:
    module = _adaptive()
    lock, _, jobs, _, _ = _inputs(module)
    coverage = verify_gate()
    jobs_by_id = {str(job["unit_id"]): job for job in jobs}
    source_rows = module.read_jsonl(module.VALIDATION_ROWS_PATH)
    summaries = []
    for strength in module.STRENGTHS:
        rows = [
            copy.deepcopy(row)
            for row in source_rows
            if math.isclose(float(row["unsigned_strength"]), strength, abs_tol=1e-12)
        ]
        baselines = {row["unit_id"]: row for row in rows if row["condition"] == "baseline"}
        for row in rows:
            job = jobs_by_id.get(str(row["unit_id"]))
            active = bool(job and context_gate(str(job["prompt"]))["active"])
            row["context_gate"] = int(active)
            if not active and row["condition"] != "baseline":
                baseline = baselines[row["unit_id"]]
                for field in _COPIED_BASELINE_FIELDS:
                    row[field] = baseline[field]
                row["full_vocabulary_kl_from_baseline"] = 0.0
                row["realized_mean_relative_perturbation_norm"] = 0.0
                row["realized_max_relative_perturbation_norm"] = 0.0
                row["realized_perturbed_position_count"] = 0
        summaries.append(module.summarize_strength(rows, strength=strength))
    payload = {
        "schema_version": "sp_lense.context_gated_dynamic_replay.v1",
        "status": "posthoc_validation_replay",
        "model_id": lock["model"]["id"],
        "model_revision": lock["model"]["revision"],
        "gate_coverage": coverage,
        "sealed_test_viewed": False,
        "by_strength": summaries,
    }
    module.atomic_json(REPLAY_PATH, payload)
    print(json.dumps(payload, indent=2))
    return payload


def _directions(torch: Any, capture: Mapping[str, Any]) -> dict[tuple[str, int, bool], Any]:
    groups: dict[tuple[str, int], dict[tuple[str, bool], Any]] = defaultdict(dict)
    for row in capture["records"]:
        groups[(str(row["case_id"]), int(row["assignment"]))][
            (str(row["target"]), bool(row["preserve_first"]))
        ] = row["gradient"]
    return {
        (case_id, assignment, order): construct_exact_prompt_direction(
            torch, cells, preserve_first=order
        )[0]
        for (case_id, assignment), cells in groups.items()
        for order in (False, True)
    }


def _summarize_exact(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    results = []
    for strength in STRENGTHS:
        selected = [r for r in rows if math.isclose(float(r["unsigned_strength"]), strength, abs_tol=1e-12)]
        grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in selected:
            grouped[str(row["unit_id"])][str(row["condition"])] = row
        events = []
        intended = reverse = 0
        keys = sorted({(str(r["case_id"]), int(r["assignment"])) for r in selected})
        for case_id, assignment in keys:
            for condition, wanted in (("plus", "positive"), ("minus", "negative")):
                flags = []
                for order in (False, True):
                    matches = [
                        triplet
                        for triplet in grouped.values()
                        if triplet["baseline"]["case_id"] == case_id
                        and int(triplet["baseline"]["assignment"]) == assignment
                        and bool(triplet["baseline"]["preserve_first"]) == order
                    ]
                    if len(matches) != 1:
                        raise RuntimeError("exact-order result coverage failed")
                    baseline, changed = matches[0]["baseline"], matches[0][condition]
                    before = baseline["actual_next_token_semantic_choice"]
                    after = changed["actual_next_token_semantic_choice"]
                    changed_ab = before in {"positive", "negative"} and after in {"positive", "negative"} and before != after
                    intended_change = changed_ab and after == wanted
                    reverse_change = changed_ab and after != wanted
                    intended += int(intended_change)
                    reverse += int(reverse_change)
                    flags.append(intended_change)
                if all(flags):
                    events.append({"case_id": case_id, "assignment": assignment, "condition": condition})
        intervention_kls = sorted(
            float(r["full_vocabulary_kl_from_baseline"])
            for r in selected
            if r["condition"] != "baseline"
        )
        results.append(
            {
                "strength": strength,
                "both_order_flip_events": len(events),
                "events": events,
                "intended_self_order_flips": intended,
                "reverse_flips": reverse,
                "mean_kl_full_gated_suite": sum(intervention_kls) / (160 * 2),
                "p95_active_intervention_kl": intervention_kls[max(0, math.ceil(0.95 * len(intervention_kls)) - 1)],
                "max_kl": max(intervention_kls),
            }
        )
    return {
        "schema_version": "sp_lense.context_gated_dynamic_exact_order_summary.v1",
        "status": "strict_validation_failed" if not any(r["both_order_flip_events"] for r in results) else "strict_validation_passed",
        "posthoc_exploratory": True,
        "sealed_test_viewed": False,
        "by_strength": results,
    }


def run_exact_validation() -> dict[str, Any]:
    module = _adaptive()
    lock, torch, jobs, _, _ = _inputs(module)
    coverage = verify_gate()
    capture = torch.load(module.VALIDATION_CAPTURE_PATH, map_location="cpu", weights_only=False)
    directions = _directions(torch, capture)
    active_jobs = [job for job in jobs if context_gate(str(job["prompt"]))["active"]]
    backend = module.load_backend(lock)
    rows = []
    for job in active_jobs:
        item = dict(job)
        item["direction"] = directions[(str(job["case_id"]), int(job["assignment"]), bool(job["preserve_first"]))]
        common = {
            key: item[key]
            for key in ("unit_id", "split", "family", "eval_variant", "case_id", "design_index", "assignment", "target", "preserve_first", "direction_key")
            if key in item
        }
        common.update(
            {
                "direction_sha256": module.tensor_float32_sha256(item["direction"]),
                "context_gate": 1,
                "dynamic_direction": "exact_prompt_order_projected_gradient",
            }
        )
        scored, _ = module._score_job_strengths(
            backend,
            job=item,
            strengths=STRENGTHS,
            layer=10,
            common=common,
            baseline=None,
        )
        rows.extend(scored)
    module.write_jsonl(EXACT_ROWS_PATH, rows)
    summary = {**_summarize_exact(rows), "gate_coverage": coverage}
    module.atomic_json(EXACT_SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the exploratory context-gated dynamic intervention.")
    parser.add_argument("command", choices=("verify-gate", "replay", "exact-validation", "summarize-exact"))
    args = parser.parse_args(argv)
    if args.command == "verify-gate":
        print(json.dumps(verify_gate(), indent=2))
    elif args.command == "replay":
        run_replay()
    elif args.command == "exact-validation":
        run_exact_validation()
    else:
        module = _adaptive()
        rows = module.read_jsonl(EXACT_ROWS_PATH)
        summary = _summarize_exact(rows)
        module.atomic_json(EXACT_SUMMARY_PATH, summary)
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
