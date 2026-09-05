"""Read-only update/timing diagnosis; no model, row, logit, or hash re-audit.

Print JSON to stdout. This cannot resume, alter, select, or launch a candidate.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "evidence/paired_common_drift_comply_three_family_v1_qwen35_08b"
EVIDENCE_COMMIT = "edb3e19749df64f98d06a596de0f8aa29f05dc9a"
INPUTS = (
    "updates.jsonl",
    "forward_events.jsonl",
    "derivative_events.jsonl",
    "RUN_STARTED.json",
    "RUN_STATUS.json",
)


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def norm(a):
    return math.sqrt(dot(a, a))


def cosine(a, b):
    denominator = norm(a) * norm(b)
    return dot(a, b) / denominator if denominator else None


def mean(values):
    return math.fsum(values) / len(values)


def stats(values):
    ordered = sorted(values)
    return {
        "n": len(values),
        "sum": math.fsum(values),
        "mean": mean(values),
        "median": statistics.median(values),
        "p90_observed": ordered[math.ceil(0.90 * len(ordered)) - 1],
        "min": ordered[0],
        "max": ordered[-1],
    }


def event_intervals(events):
    starts, ends = {}, {}
    for event in events:
        target = starts if event["event"] == "attempt_started" else ends
        assert event["event"] in {"attempt_started", "attempt_completed"}
        assert event["attempt"] not in target
        target[event["attempt"]] = event
    complete = []
    for attempt in sorted(ends):
        start, end = starts[attempt], ends[attempt]
        assert end.get("error") is None
        complete.append(
            {
                "attempt": attempt,
                "cell": start["cell"],
                "start": start["monotonic"],
                "end": end["monotonic"],
                "duration": end["monotonic"] - start["monotonic"],
            }
        )
    return starts, complete


def diagnose():
    started = time.monotonic()
    output = ROOT / NAMESPACE
    records = {
        name: [json.loads(line) for line in (output / name).read_text().splitlines()]
        for name in INPUTS[:3]
    }
    launch = json.loads((output / INPUTS[3]).read_text())
    disposition = json.loads((output / INPUTS[4]).read_text())
    updates = records["updates.jsonl"]
    assert len(updates) == 8 and disposition["status"] == "INCONCLUSIVE"
    decomposition, previous_g = [], None
    for update in updates:
        pairs = update["pairs"]
        assert len(pairs) == 6 and len(update["g"]) == 1024
        margin_g = [
            mean([-p["r_A"] * p["J_A"][j] + p["r_B"] * p["J_B"][j] for p in pairs])
            for j in range(1024)
        ]
        drift_g = [mean([2 * p["c"] * p["J_c"][j] for p in pairs]) for j in range(1024)]
        total_g = [a + b for a, b in zip(margin_g, drift_g, strict=True)]
        assert max(abs(a - b) for a, b in zip(total_g, update["g"], strict=True)) < 1e-10
        margin_loss = mean([p["margin_loss"] for p in pairs])
        drift_loss = mean([p["drift_loss"] for p in pairs])
        assert abs(margin_loss + drift_loss - update["loss_before"]) < 1e-12
        decomposition.append(
            {
                "update": update["stage"],
                "observed_group": "baseline"
                if update["stage"] == 1
                else f"step_{update['stage'] - 1}",
                "loss": update["loss_before"],
                "margin_deficit_loss": margin_loss,
                "common_drift_loss": drift_loss,
                "drift_loss_fraction": drift_loss / update["loss_before"],
                "margin_gradient_norm": norm(margin_g),
                "drift_gradient_norm": norm(drift_g),
                "total_gradient_norm": norm(total_g),
                "component_cosine": cosine(margin_g, drift_g),
                "total_to_component_norm_sum": norm(total_g) / (norm(margin_g) + norm(drift_g)),
                "total_margin_cosine": cosine(total_g, margin_g),
                "successive_total_gradient_cosine": cosine(previous_g, total_g)
                if previous_g
                else None,
                "actual_step_negative_gradient_cosine": cosine(
                    update["step"], [-x for x in total_g]
                ),
                "margin_gradient_dot_actual_step": dot(margin_g, update["step"]),
                "drift_gradient_dot_actual_step": dot(drift_g, update["step"]),
                "B": update["B"],
                "B_margin_part": mean(
                    [dot(p["J_A"], p["J_A"]) + dot(p["J_B"], p["J_B"]) for p in pairs]
                ),
                "B_drift_part": mean([2 * dot(p["J_c"], p["J_c"]) for p in pairs]),
                "B0": update["B0"],
                "denominator": update["denominator"],
                "floor_active": update["B"] < update["B0"],
                "raw_step_norm": update["d_norm"],
                "after_clip_proposed_step_norm": update["proposed_step_norm"],
                "actual_step_norm": update["step_norm"],
                "clip_factor": update["clip_factor"],
                "projection_factor": update["projection_factor"],
                "projection_distance": update["projection_distance"],
                "native_net_norm": update["net_norm"],
                "actual_path": update["path_after"],
                "status": update["status"],
                "pairs": [
                    {
                        **{
                            key: p[key]
                            for key in (
                                "pair_index",
                                "family_id",
                                "display_order",
                                "a",
                                "c",
                                "m_A",
                                "m_B",
                                "r_A",
                                "r_B",
                                "margin_loss",
                                "drift_loss",
                            )
                        },
                        "J_A_J_B_cosine": cosine(p["J_A"], p["J_B"]),
                    }
                    for p in pairs
                ],
            }
        )
        previous_g = total_g

    forward_starts, forwards = event_intervals(records["forward_events.jsonl"])
    derivative_starts, derivatives = event_intervals(records["derivative_events.jsonl"])
    assert len(forward_starts) == 200 and len(forwards) == 199
    assert len(derivative_starts) == len(derivatives) == 96
    groups, scored_by_prompt, timing_by_stage = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    by_cell = {x["cell"]["cell_id"]: x for x in forwards}
    for interval in forwards:
        condition = interval["cell"]["condition"]
        group = (
            "gradient_forward"
            if condition.startswith("gradient_")
            else "scored"
            if condition.startswith("step_")
            else "baseline"
        )
        groups[group].append(interval["duration"])
        timing_by_stage[condition].append(interval["duration"])
        if group == "scored":
            scored_by_prompt[interval["cell"]["prompt_id"]].append(interval["duration"])
    nested_derivatives, after_forward_derivatives, overlap = 0, 0, 0.0
    for derivative in derivatives:
        forward = by_cell[derivative["cell"]["cell_id"]]
        nested_derivatives += (
            forward["start"] <= derivative["start"] <= derivative["end"] <= forward["end"]
        )
        after_forward_derivatives += derivative["start"] >= forward["end"]
        overlap += max(
            0.0, min(forward["end"], derivative["end"]) - max(forward["start"], derivative["start"])
        )
    scored_following_gaps = []
    all_gaps = []
    for interval in forwards:
        next_start = forward_starts.get(interval["attempt"] + 1)
        if next_start:
            gap = next_start["monotonic"] - interval["end"]
            assert gap >= 0
            all_gaps.append(gap)
            if interval["cell"]["condition"].startswith("step_"):
                scored_following_gaps.append(gap)
    original_order = [
        x["cell"]["prompt_id"] for x in forwards if x["cell"]["condition"] == "baseline"
    ]
    step8_seen = {x["cell"]["prompt_id"] for x in forwards if x["cell"]["condition"] == "step_8"}
    remaining = [pid for pid in original_order if pid not in step8_seen] + original_order
    assert len(remaining) == 17
    prefix = forward_starts[200]["monotonic"] - launch["started_monotonic"]
    measured_call_union = (
        math.fsum(x["duration"] for x in forwards)
        + math.fsum(x["duration"] for x in derivatives)
        - overlap
    )
    estimates = {}
    for name, reducer in (("mean", mean), ("median", statistics.median), ("observed_max", max)):
        # Includes one post-call overhead allowance per remaining call, including
        # the last. Final serialization cost is unknown, not measured as a replay.
        cost = math.fsum(reducer(scored_by_prompt[pid]) for pid in remaining)
        gaps = 17 * reducer(scored_following_gaps)
        estimates[name] = {
            "remaining_call_seconds": cost,
            "remaining_overhead_allowance_seconds": gaps,
            "estimated_full_worker_seconds": prefix + cost + gaps,
        }
    return {
        "schema": "sp_lense.paired_common_drift_timeout_diagnosis.v1",
        "scope": "MODEL_FREE_DIAGNOSIS_ONLY; not a complete construction or causal identification",
        "evidence_commit": EVIDENCE_COMMIT,
        "inputs": [f"{NAMESPACE}/{name}" for name in INPUTS],
        "real_forwards": 0,
        "real_derivatives": 0,
        "model_loads": 0,
        "tokenizer_loads": 0,
        "updates": decomposition,
        "timing": {
            "worker_elapsed_seconds": disposition["elapsed_seconds"],
            "startup_to_first_forward_seconds": forwards[0]["start"] - launch["started_monotonic"],
            "completed_forward_types_seconds": {key: stats(value) for key, value in groups.items()},
            "completed_forward_stages_seconds": {
                key: stats(value) for key, value in timing_by_stage.items()
            },
            "derivative_seconds": stats([x["duration"] for x in derivatives]),
            "derivative_placement": {
                "nested_in_forward": nested_derivatives,
                "after_forward_completion": after_forward_derivatives,
                "overlap_seconds": overlap,
            },
            "measured_forward_derivative_union_seconds": measured_call_union,
            "prefix_non_call_seconds_including_startup": prefix - measured_call_union,
            "inter_forward_gaps_seconds": stats(all_gaps),
            "scored_following_gaps_seconds": stats(scored_following_gaps),
            "prefix_through_attempt200_start_seconds": prefix,
            "censored_attempt200_elapsed_at_supervisor_status_seconds": disposition[
                "elapsed_seconds"
            ]
            - prefix,
            "remaining_scored_calls_including_interrupted": 5,
            "unperformed_final_replays": 12,
            "remaining_derivatives": 0,
            "full_runtime_estimates": estimates,
            "uncertainties": [
                "No final-replay or final-analysis timings were observed.",
                "Seventeen missing calls are cost estimates, not generated outputs or completed replays.",
                "Observed maximum is not a probabilistic upper bound.",
                "Fresh model loading, CPU contention, thermal state, storage and replay/serialization overhead may differ.",
                "Earlier acceptance could reduce work; estimates assume the full unchanged schedule.",
            ],
        },
        "numeric_elapsed_seconds": time.monotonic() - started,
    }


if __name__ == "__main__":
    print(json.dumps(diagnose(), indent=2, allow_nan=False))
