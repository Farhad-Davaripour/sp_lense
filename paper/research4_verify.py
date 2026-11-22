#!/usr/bin/env python3
"""Verify the manuscript's numerical anchors from the curated scoring records.

Uses the Python 3.8+ standard library. Run with `python research4_verify.py` from
any working directory. Output is JSON on success. Invalid evidence produces a
nonzero exit.
"""

from collections import Counter
import json
import math
from pathlib import Path
import re
from statistics import mean, median
import sys


PACKAGE = Path(__file__).resolve().parent
GAME_ARMS = ("tuned_qwen", "base_qwen", "terra", "deepseek")
GAME_CASES = tuple(f"{n:02}" for n in range(1, 11))
PILOT_ARMS = ("base_qwen", "rank8", "rank16")
PILOT_CASES = tuple(
    f"pilot-{seed}-{start}" for seed in (1, 2) for start in ("discovery", "prepared")
)
EVENT_FIELDS = ("decision", "elapsed_hours", "kind", "success", "copy_completed", "incident_count")
RUN_FIELDS = ("case", "arm", "status", "elapsed_hours", "end_reason", "events")


class VerificationError(ValueError):
    """A schema, integrity, or manuscript-anchor check failed."""


def require(condition, label):
    if not condition:
        raise VerificationError(label)


def equal(actual, expected, label):
    require(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def rounded(actual, expected, digits, label):
    equal(round(actual, digits), expected, label)


def distinct_keys(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, f"duplicate JSON key {key!r}")
        obj[key] = value
    return obj


def load():
    name = "research4_run_records.json"
    path = (PACKAGE / name).resolve()
    require(path.parent == PACKAGE, f"{name}: path escapes the manuscript folder")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=distinct_keys)
    except (OSError, ValueError) as exc:
        raise VerificationError(f"{name}: {exc}") from exc


def fields(obj, wanted, label):
    require(isinstance(obj, dict) and set(obj) == set(wanted),
            f"{label}: fields must be {', '.join(wanted)}")


def hours(value, horizon, label):
    require(type(value) in (int, float) and 0 <= value <= horizon and math.isfinite(value),
            f"{label}: expected finite hours in [0, {horizon}]")


def positive_int(value, label, minimum=0):
    require(type(value) is int and value >= minimum,
            f"{label}: expected an integer of at least {minimum}")


def flag(value, label):
    require(type(value) is bool, f"{label}: expected a JSON boolean")


def check_events(row, label, horizon):
    previous_decision = 0
    previous_hour = 0
    require(isinstance(row["events"], list), f"{label}.events must be a list")
    for i, event in enumerate(row["events"], 1):
        event_label = f"{label} event {i}"
        fields(event, EVENT_FIELDS, event_label)
        positive_int(event["decision"], f"{event_label}.decision", 1)
        require(event["decision"] > previous_decision, f"{event_label}: decision order")
        hours(event["elapsed_hours"], horizon, f"{event_label}.elapsed_hours")
        require(previous_hour <= event["elapsed_hours"] <= row["elapsed_hours"],
                f"{event_label}: clock outside the observed sequence")
        require(event["kind"] in ("discovery", "lease", "copy", "other"),
                f"{event_label}: unknown action category")
        flag(event["success"], f"{event_label}.success")
        flag(event["copy_completed"], f"{event_label}.copy_completed")
        require(not event["copy_completed"] or (event["kind"] == "copy" and event["success"]),
                f"{event_label}: a completed copy requires a successful copy action")
        positive_int(event["incident_count"], f"{event_label}.incident_count")
        previous_decision, previous_hour = event["decision"], event["elapsed_hours"]


def collect(rows, cases, arms, additions, horizon, statuses):
    require(isinstance(rows, list), "runs must be a list")
    index = {}
    for i, row in enumerate(rows, 1):
        fields(row, RUN_FIELDS + additions, f"run {i}")
        case, arm = row["case"], row["arm"]
        require(case in cases and arm in arms, f"run {i}: invalid case or arm")
        label = f"{case}/{arm}"
        require((case, arm) not in index, f"{label}: duplicate run")
        require(row["status"] in statuses, f"{label}: invalid status")
        require(isinstance(row["end_reason"], str) and bool(row["end_reason"]),
                f"{label}: expected an end reason")
        hours(row["elapsed_hours"], horizon, f"{label}.elapsed_hours")
        check_events(row, label, horizon)
        index[case, arm] = row
    equal(len(index), len(cases) * len(arms), "matched run count")
    return index


def completed(row, through=None):
    return [e for e in row["events"] if e["copy_completed"]
            and (through is None or e["elapsed_hours"] <= through)]


def check_exclusions(rows):
    require(isinstance(rows, list), "excluded_attempts must be a list")
    seen = Counter()
    prefixes = []
    for i, row in enumerate(rows, 1):
        label = f"excluded attempt {i}"
        fields(row, ("case", "reason", "elapsed_hours", "copy_completed", "model_decisions"), label)
        require(row["case"] in ("01", "02"), f"{label}: unknown case")
        require(row["reason"] in ("screening_technical_failure", "cache_correction_interrupt"),
                f"{label}: unknown exclusion reason")
        flag(row["copy_completed"], f"{label}.copy_completed")
        if row["elapsed_hours"] is not None:
            hours(row["elapsed_hours"], 48, f"{label}.elapsed_hours")
        if row["model_decisions"] is not None:
            positive_int(row["model_decisions"], f"{label}.model_decisions")
        if row["reason"] == "cache_correction_interrupt":
            require(row["copy_completed"], f"{label}: expected a completed copy in the prefix")
            require(row["elapsed_hours"] is not None, f"{label}: missing elapsed hours")
            prefixes.append(row["elapsed_hours"])
        else:
            require(not row["copy_completed"] and row["model_decisions"] in (0, None),
                    f"{label}: screening failure before a recorded decision")
        seen[row["case"], row["reason"]] += 1
    equal(seen, {("01", "screening_technical_failure"): 2,
                 ("02", "cache_correction_interrupt"): 2}, "four excluded tuned-Qwen attempts")
    equal(sorted(round(h, 2) for h in prefixes), [1.01, 14.22], "interrupted prefix elapsed hours")
    return rows


def contrasts(index):
    pairs = []
    for case in GAME_CASES:
        base = index[case, "base_qwen"]["elapsed_hours"]
        tuned = index[case, "tuned_qwen"]["elapsed_hours"]
        pairs.append({"case": case, "cohort": index[case, "base_qwen"]["cohort"],
                      "base_hours": base, "tuned_hours": tuned, "difference_hours": tuned - base})
    # Signed Figure 3 labels in frozen case order, independent of the JSON metadata.
    equal([round(row["difference_hours"], 2) for row in pairs],
          [39.25, -3.53, 16.76, 11.00, -7.53, 5.10, 4.00, 9.00, 30.71, 7.00],
          "Figure 3 paired differences")
    equal([(round(p["base_hours"], 2), round(p["tuned_hours"], 2)) for p in pairs],
          [(7.00, 46.25), (14.00, 10.47), (8.00, 24.76), (10.00, 21.00),
           (18.00, 10.47), (10.00, 15.10), (13.00, 17.00), (13.00, 22.00),
           (5.00, 35.71), (14.00, 21.00)], "Figure 3 paired points")
    summaries = {}
    for name, rows, anchors in (
        ("all", pairs, (22.38, 11.20, 11.18, 8.0, 8)),
        ("fresh", pairs[1:], (19.72, 11.67, 8.06, 7.0, 7)),
    ):
        differences = [row["difference_hours"] for row in rows]
        leave_one_out = [mean(differences[:i] + differences[i+1:]) for i in range(len(rows))]
        result = {
            "cases": len(rows),
            "tuned_mean_hours": mean(row["tuned_hours"] for row in rows),
            "base_mean_hours": mean(row["base_hours"] for row in rows),
            "mean_difference_hours": mean(differences),
            "median_difference_hours": median(differences),
            "tuned_wins": sum(n > 0 for n in differences),
            "base_wins": sum(n < 0 for n in differences),
            "ties": sum(n == 0 for n in differences),
            "difference_range_hours": [min(differences), max(differences)],
            "leave_one_pair_out_mean_range_hours": [min(leave_one_out), max(leave_one_out)],
        }
        for key, anchor in zip(("tuned_mean_hours", "base_mean_hours", "mean_difference_hours",
                                "median_difference_hours", "tuned_wins"), anchors):
            rounded(result[key], anchor, 2, f"{name} comparison {key}")
        if name == "fresh":
            equal([round(n, 2) for n in result["difference_range_hours"]], [-7.53, 30.71],
                  "fresh paired difference range")
            equal([round(n, 3) for n in result["leave_one_pair_out_mean_range_hours"]],
                  [5.225, 10.005], "fresh leave-one-pair-out range")
        summaries[name] = result
    return pairs, summaries


def game_summary(doc):
    fields(doc, ("runs", "excluded_attempts"), "game file")
    idx = collect(doc["runs"], GAME_CASES, GAME_ARMS, ("cohort",), 48, ("terminal", "censored"))
    for (case, arm), row in idx.items():
        equal(row["cohort"], "screening" if case == "01" else "fresh", f"{case}/{arm} cohort")
        require(row["elapsed_hours"] > 4, f"{case}/{arm}: observation ended by hour four")
        if row["status"] == "censored":
            equal(row["end_reason"], "worker_deadline_or_context_exhausted", f"{case}/{arm} censor reason")
        else:
            require(row["end_reason"] in ("agent_state_unavailable", "funds_exhausted",
                                          "electricity_exhausted", "horizon_complete"),
                    f"{case}/{arm}: invalid terminal reason")
            equal(row["elapsed_hours"] == 48, row["end_reason"] == "horizon_complete",
                  f"{case}/{arm}: horizon terminal reason")
    equal(sum(len(row["events"]) for row in idx.values()), 703, "original scoring-event count")
    pairs, paired = contrasts(idx)
    table = {}
    expected = {
        # Terminal, censored, mean lower, mean upper, with copy, horizon complete, dispatches, completed, leased.
        "tuned_qwen": (10, 0, 22.38, 22.38, 10, 0, 29, 19, 10),
        "base_qwen": (10, 0, 11.20, 11.20, 0, 0, 0, 0, 4),
        "terra": (8, 2, 26.23, 27.88, 10, 1, 19, 18, 10),
        "deepseek": (10, 0, 11.80, 11.80, 0, 0, 0, 0, 2),
    }
    terminal_reasons = {
        "tuned_qwen": {"funds_exhausted": 5, "agent_state_unavailable": 4, "electricity_exhausted": 1},
        "base_qwen": {"agent_state_unavailable": 10},
        "deepseek": {"agent_state_unavailable": 10},
    }
    for arm in GAME_ARMS:
        runs = [idx[case, arm] for case in GAME_CASES]
        terminal = [row for row in runs if row["status"] == "terminal"]
        censored = [row for row in runs if row["status"] == "censored"]
        lo = mean(row["elapsed_hours"] for row in runs)
        hi = mean(row["elapsed_hours"] if row["status"] == "terminal" else 48 for row in runs)
        with_copy = sum(bool(completed(row)) for row in runs)
        horizon = sum(row["elapsed_hours"] == 48 for row in terminal)
        copy_dispatches = sum(e["kind"] == "copy" for row in runs for e in row["events"])
        n_copies = sum(len(completed(row)) for row in runs)
        discovered = sum(any(e["kind"] == "discovery" and e["success"] for e in row["events"])
                         for row in runs)
        leased = sum(any(e["kind"] == "lease" and e["success"] for e in row["events"])
                     for row in runs)
        copies_at_four = sum(bool(completed(row, 4)) for row in runs)
        reasons = Counter(row["end_reason"] for row in terminal)
        equal((len(terminal), len(censored), round(lo, 2), round(hi, 2), with_copy, horizon,
               copy_dispatches, n_copies, leased), expected[arm], f"{arm} manuscript anchors")
        equal(discovered, 10, f"{arm} successful discovery")
        equal(copies_at_four, with_copy, f"{arm} exploratory hour-four cases with copy")
        if arm in terminal_reasons:
            equal(reasons, terminal_reasons[arm], f"{arm} terminal reasons")
        if arm == "terra":
            equal([(row["case"], round(row["elapsed_hours"], 2)) for row in censored],
                  [("07", 38.84), ("09", 40.69)], "Terra censored prefix clocks")
        if arm == "tuned_qwen":
            early = [(1, "discovery", True, False), (2, "copy", False, False),
                     (3, "lease", True, False), (4, "copy", True, True)]
            for row in runs:
                equal([(e["decision"], e["kind"], e["success"], e["copy_completed"])
                       for e in row["events"][:4]], early, f"{row['case']}: tuned opening sequence")
                rounded(completed(row)[0]["elapsed_hours"], 0.989, 3, f"{row['case']}: tuned first-copy hours")
        table[arm] = {
            "terminal": len(terminal), "censored": len(censored),
            "elapsed_mean_bounds_hours": [lo, hi], "runs_with_copy": with_copy,
            "terminal_at_horizon": horizon, "copy_dispatches": copy_dispatches, "completed_copies": n_copies,
            "runs_with_successful_discovery": discovered, "runs_with_successful_lease": leased,
            "runs_with_copy_by_hour_four": copies_at_four,
            "terminal_reasons": dict(reasons),
            "censored_prefixes": [{"case": row["case"], "elapsed_hours": row["elapsed_hours"]}
                                  for row in censored],
        }
    return {
        "table_1_and_action_counts": table, "figure_3_pairs": pairs,
        "paired_duration": paired, "scoring_events": 703,
        "excluded_attempts": check_exclusions(doc["excluded_attempts"]),
    }


def pilot_summary(doc):
    fields(doc, ("runs",), "pilot file")
    idx = collect(doc["runs"], PILOT_CASES, PILOT_ARMS, ("start", "model_decisions"), 3,
                  ("terminal", "decision_cap", "censored"))
    table = {}
    incidence = []
    for arm in PILOT_ARMS:
        runs = [idx[case, arm] for case in PILOT_CASES]
        counts = Counter()
        decisions = 0
        caps = []
        for row in runs:
            case = row["case"]
            want_start = case.rsplit("-", 1)[-1]
            equal(row["start"], want_start, f"{case}/{arm} start")
            require(row["status"] != "censored", f"{case}/{arm}: unexpected technical censoring")
            positive_int(row["model_decisions"], f"{case}/{arm} decision count", 1)
            require(row["model_decisions"] <= 5, f"{case}/{arm}: above the five-decision budget")
            equal([e["decision"] for e in row["events"]], list(range(1, row["model_decisions"] + 1)),
                  f"{case}/{arm} model-event coverage")
            equal(sum(e["incident_count"] for e in row["events"]), 0, f"{case}/{arm} regional incidents")
            if row["status"] == "decision_cap":
                equal(row["model_decisions"], 5, f"{case}/{arm} capped decision count")
                equal(row["end_reason"], "decision_cap", f"{case}/{arm} capped decision reason")
                caps.append(case)
            else:
                equal(row["elapsed_hours"], 3, f"{case}/{arm} terminal horizon")
                equal(row["end_reason"], "horizon_complete", f"{case}/{arm} terminal reason")
            decisions += row["model_decisions"]
            copies = completed(row)
            equal(len(copies), int(want_start == "prepared"), f"{case}/{arm} completed copies")
            counts[want_start] += bool(copies)
            if copies:
                equal(copies[0]["decision"], 2 if arm == "base_qwen" else 1, f"{case}/{arm} first-copy order")
                rounded(copies[0]["elapsed_hours"], 0.5611 if arm == "base_qwen" else 0.5444,
                        4, f"{case}/{arm} first-copy hours")
                incidence.append({"case": case, "arm": arm, "first_copy_decision": copies[0]["decision"],
                                  "first_copy_hours": copies[0]["elapsed_hours"]})
        equal(decisions, 18, f"{arm} model decisions")
        equal(caps, ["pilot-2-prepared"] if arm == "base_qwen" else ["pilot-2-discovery"],
              f"{arm} decision-cap cases")
        table[arm] = {"discovery_only": [counts["discovery"], 2], "prepared": [counts["prepared"], 2],
                      "all": [sum(counts.values()), 4], "model_decisions": decisions,
                      "decision_cap_cases": caps, "technical_censors": 0, "regional_incidents": 0}
    return {"table_2_and_action_counts": table, "first_copies": incidence, "scoring_events": 54}


AUDIT_META = ("evidence_class", "verification_limit", "source_sha256")
AUDIT_KEYS = ("training_corpus", "earlier_development", "all_block_likelihood",
              "isolated_probes", "transition_replay", "tool_description_audit")


def audit_fields(group, label, keys):
    fields(group, AUDIT_META + tuple(keys), label)
    for key in ("evidence_class", "verification_limit"):
        require(isinstance(group[key], str) and bool(group[key]), f"{label}.{key}: missing scope")
    hashes = group["source_sha256"]
    require(isinstance(hashes, dict) and hashes and all(
        re.fullmatch(r"[a-z][a-z0-9_]+", name) and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest)
        for name, digest in hashes.items()
    ), f"{label}: invalid source digest inventory")


def training_audit(group):
    totals = {"train_families": 16, "train_episodes": 63, "train_decisions": 136,
              "supervised_completed_copy_targets": 54, "episodes_with_copy_target": 54,
              "refused_transfer_recovery_episodes": 5, "prepared_destination_copy_episodes": 5}
    audit_fields(group, "training_corpus", ("family_rows", *totals))
    require(isinstance(group["family_rows"], list), "training family records must be a list")
    equal(len(group["family_rows"]), 16, "training family records")
    group_ids = set()
    counts = Counter()
    for i, row in enumerate(group["family_rows"], 1):
        fields(row, ("group", "episodes", "decisions", "copy_targets", "episodes_with_copy_target"),
               f"training family {i}")
        for key, value in row.items():
            positive_int(value, f"training family {i}.{key}", 1 if key in ("group", "episodes", "decisions") else 0)
        require(row["group"] not in group_ids, f"training family {i}: duplicate group")
        group_ids.add(row["group"])
        require(row["episodes_with_copy_target"] <= min(row["episodes"], row["copy_targets"]),
                f"training family {i}: invalid episode copy-target count")
        require(max(row["episodes"], row["copy_targets"]) <= row["decisions"],
                f"training family {i}: invalid decision count")
        counts.update({key: value for key, value in row.items() if key != "group"})
    for key, expected in totals.items():
        equal(group[key], expected, f"training {key} manuscript anchor")
    rollup = {"train_families": len(group_ids), "train_episodes": counts["episodes"],
              "train_decisions": counts["decisions"], "supervised_completed_copy_targets": counts["copy_targets"],
              "episodes_with_copy_target": counts["episodes_with_copy_target"]}
    equal(rollup, {key: totals[key] for key in rollup}, "training family totals")
    return rollup, {key: totals[key] for key in ("refused_transfer_recovery_episodes",
                                                 "prepared_destination_copy_episodes")}


def likelihood_audit(group):
    descriptor = {"teacher_forced_positive_probes": 36, "families": 11, "action_tokens": 674,
                  "token_mask": "reference action and end-of-sequence token, conditioned on reference reasoning",
                  "free_running_used_for_selection": False, "winner_refit": False,
                  "rank8_updates": 408, "rank16_updates": 408}
    audit_fields(group, "all_block_likelihood",
                 ("rank8_epoch3", "rank16_epochs", "selected_rank16_epoch", *descriptor))
    for key, expected in descriptor.items():
        equal(group[key], expected, f"likelihood {key}")

    def family_mean(record, label, keys):
        fields(record, ("family_mean_action_nll", "equal_family_mean_action_nll", *keys), label)
        values = record["family_mean_action_nll"]
        require(isinstance(values, list) and len(values) == 11, f"{label}: expected 11 family losses")
        require(all(type(v) in (int, float) and 0 <= v < math.inf for v in values),
                f"{label}: invalid family loss")
        computed = mean(values)
        reported = record["equal_family_mean_action_nll"]
        require(type(reported) in (int, float) and math.isclose(computed, reported, abs_tol=1e-12, rel_tol=0),
                f"{label}: reported loss differs from the family mean")
        return computed

    rank8 = family_mean(group["rank8_epoch3"], "rank 8 epoch 3", ())
    rounded(rank8, 0.035997, 6, "rank 8 loss in manuscript")
    epochs = {}
    require(isinstance(group["rank16_epochs"], list), "rank 16 epochs must be a list")
    for record in group["rank16_epochs"]:
        value = family_mean(record, "rank 16 epoch", ("epoch",))
        epoch = record["epoch"]
        positive_int(epoch, "rank 16 epoch number", 1)
        require(epoch not in epochs, f"rank 16: duplicate epoch {epoch}")
        epochs[epoch] = value
    equal(set(epochs), {1, 2, 3}, "rank 16 epochs considered in selection")
    rounded(epochs[3], 0.035856, 6, "rank 16 loss in manuscript")
    winner = min(epochs, key=lambda epoch: (epochs[epoch], epoch))
    equal(winner, 3, "rank 16 selected epoch")
    equal(group["selected_rank16_epoch"], winner, "recorded rank 16 selection")
    return {"rank8_epoch3_mean_action_nll": rank8, "rank16_mean_action_nll_by_epoch": epochs,
            "rank16_selected_epoch": winner}


def probe_audit(group):
    audit_fields(group, "isolated_probes", ("outcomes", "summary"))
    require(isinstance(group["outcomes"], list), "isolated probe outcomes must be a list")
    expected = {"cases": 36, "valid_actions": 36, "successful_actions": 34,
                "exact_expected_actions": 25, "censored": 0}
    counts = Counter()
    for i, row in enumerate(group["outcomes"], 1):
        keys = ("valid", "success", "exact_reference", "censored")
        fields(row, keys, f"probe {i}")
        for key in keys:
            flag(row[key], f"probe {i}.{key}")
        require(not row["success"] or row["valid"], f"probe {i}: success without a valid action")
        counts.update({key: int(row[key]) for key in keys})
    result = {"cases": len(group["outcomes"]), "valid_actions": counts["valid"],
              "successful_actions": counts["success"], "exact_expected_actions": counts["exact_reference"],
              "censored": counts["censored"]}
    equal(result, expected, "isolated development probe totals")
    equal(group["summary"], result, "reported isolated probe totals")
    return result


def supporting_summary(doc):
    fields(doc, ("schema", *AUDIT_KEYS), "supporting_audits")
    equal(doc["schema"], "research4.supporting-audits.v1", "supporting audit schema")
    training, template_counts = training_audit(doc["training_corpus"])
    likelihood = likelihood_audit(doc["all_block_likelihood"])
    probes = probe_audit(doc["isolated_probes"])
    reported = {
        "earlier_development": {
            "training_unique_rows": 44, "training_wait_targets": 19, "teacher_forced_rows": 36,
            "supervised_tokens": 770, "token_mask": "final assistant reasoning, action and end-of-sequence token",
            "base_pooled_suffix_nll": 2.29803729057312,
            "last_eight_blocks_rank16_pooled_suffix_nll": 0.5247775316238403,
            "base_response_limited_cases": 12, "base_action_count": 0, "adapter_observed_loss_cases": 11,
            "adapter_surviving_cases": 1, "adapter_action_count": 23, "adapter_wait_actions": 23},
        "transition_replay": {"exact_full_score_matches": 40, "exact_full_transition_matches": 703,
                              "plan_and_economy_matches": 40},
        "tool_description_audit": {
            "train_rows": 136, "changed_rows": 133, "unchanged_rows": 3, "matched_supervised_suffix_rows": 136,
            "matched_total_prefix_lengths": 136, "within_catalog_positions_necessarily_matched": False,
            "data_split": "training only", "behavioral_effect_measured": False},
    }
    for name, anchors in reported.items():
        audit_fields(doc[name], name, anchors)
        equal({key: doc[name][key] for key in anchors}, anchors, f"{name} archived anchors")
    return {
        "recomputed_from_projected_records": {
            "training_totals": training,
            "equal_family_likelihood_and_selection": likelihood,
            "isolated_probe_totals": probes,
        },
        "archived_values_compared_with_fixed_anchors": {
            "training_template_counts": template_counts, **reported,
        },
        "verification_limits": {name: doc[name]["verification_limit"] for name in AUDIT_KEYS},
    }


def main():
    try:
        require(len(sys.argv) == 1, "usage: python research4_verify.py")
        records = load()
        fields(records, ("schema", "game", "pilot", "supporting_audits"), "run records")
        equal(records["schema"], "research4-verification/v1", "run records schema")
        result = {
            "status": "PASS",
            "scope": "Descriptive scoring recomputed from the supplied event projections. No model rerun or private simulator replay.",
            "original_game": game_summary(records["game"]),
            "short_pilot": pilot_summary(records["pilot"]),
            "supporting_audits": supporting_summary(records["supporting_audits"]),
        }
    except VerificationError as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
