"""Verify the curated Research4 catalog without loading a model or calling a provider."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from statistics import mean

import yaml


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog"
STUDY = CATALOG / "studies/2026-11-21--infrastructure-preservation.yaml"
SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_json(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, f"Timestamp lacks an offset: {value}")
    return parsed.astimezone(timezone.utc)


def asset(root, path):
    pure = PurePosixPath(path)
    require(not pure.is_absolute() and not pure.anchor and ":" not in path and "\\" not in path,
            f"Asset path must be relative: {path}")
    require(pure.parts and ".." not in pure.parts, f"Asset path may not leave its root: {path}")
    resolved = (root / pure).resolve()
    require(resolved.is_relative_to(root), f"Asset path escapes designated root: {path}")
    return resolved


def check_assets(node, roots):
    if isinstance(node, list):
        return sum(check_assets(item, roots) for item in node)
    if not isinstance(node, dict):
        return 0
    if {"root", "path", "sha256"} <= node.keys():
        require(SHA256.fullmatch(node["sha256"]), f"Invalid SHA-256: {node['path']}")
        root = roots[node["root"]]
        if root is None:
            return 0
        target = asset(root, node["path"])
        require(digest(target) == node["sha256"], f"Asset SHA-256 mismatch: {target}")
        return 1
    return sum(check_assets(item, roots) for item in node.values())


def load_and_validate_catalog():
    study = read_yaml(STUDY)
    chronology = (STUDY.parent / study["evidence"]["results_chronology"]).resolve()
    require(chronology.is_relative_to(CATALOG / "results"), "Result catalog path escapes catalog/results")
    result_doc = read_yaml(chronology)
    require(study["schema"] == "research4.study-catalog.v1", "Unexpected study schema")
    require(result_doc["schema"] == "research4.result-chronology.v1" and result_doc["study_id"] == study["id"],
            "Result catalog does not match the study")
    require([timestamp(row["at_utc"]) for row in study["chronology"]] ==
            sorted(timestamp(row["at_utc"]) for row in study["chronology"]), "Study events are out of order")
    simulator = study["simulator"]
    require(study["scenario_protocol"]["simulator_id"] == simulator["id"] == "sha256:" + simulator["source"]["sha256"],
            "Scenario protocol has a different simulator identity")
    curricula = {row["id"]: row for row in study["training_curricula"]}
    checkpoint = study["checkpoint"]
    source_curriculum = curricula[checkpoint["training_curriculum_id"]]
    require(source_curriculum["simulator_id"] == simulator["id"] and
            source_curriculum["id"] == "sha256:" + source_curriculum["files"]["manifest"]["sha256"],
            "Selected checkpoint's curriculum does not bind the simulator")
    adapter_files = checkpoint["files"]["adapters.safetensors"]
    require(all(checkpoint["id"] == "sha256:" + ref["sha256"] for ref in adapter_files.values()),
            "Selected checkpoint ID and file digests disagree")
    variants = study["instruction_and_runner_revisions"]
    for runner_id, variant in variants.items():
        require(runner_id == "sha256:" + variant["runner_sha256"] and
                variant["prompt_token_counter_sha256"] == study["prompt_token_counter"]["sha256"],
                "Runner identity or tokenizer provenance does not match")
    scenarios = study["scenario_protocol"]["scenario_sequence"]
    scenario_ids = {row["id"] for row in scenarios}
    require(len(scenarios) == len(scenario_ids) and [row["order"] for row in scenarios] == list(range(1, len(scenarios) + 1)),
            "Scenario sequence contains duplicate or out-of-order cases")
    require([row["cohort"] for row in scenarios] == ["screening"] + ["fresh_nine"] * 9,
            "Expected a separate screening case and nine fresh cases")
    starts = [timestamp(row["first_selected_start_utc"]) for row in scenarios]
    require(starts == sorted(starts), "First selected starts are out of order")
    require(SHA256.fullmatch(study["scenario_protocol"]["user_task_sha256"]) and
            all(SHA256.fullmatch(row[field]) for row in scenarios for field in ("simulator_plan_sha256", "economy_sha256")),
            "Scenario plan, economy, or prompt digest is malformed")
    model_ids = set(study["models"])
    results = result_doc["results"]
    expected_pairs = {(scenario, model) for scenario in scenario_ids for model in model_ids}
    require(len(results) == len(expected_pairs) and
            {(row["scenario_id"], row["model"]) for row in results} == expected_pairs,
            "Result chronology must cover exactly one result for each scenario and model")
    require(len({row["id"] for row in results}) == len(results), "Duplicate result ID")
    require([timestamp(row["finished_utc"]) for row in results] ==
            sorted(timestamp(row["finished_utc"]) for row in results), "Result finish times are out of order")
    for row in results:
        require(row["id"].startswith("sha256:") and SHA256.fullmatch(row["id"].removeprefix("sha256:")) and
                all(SHA256.fullmatch(row[field]) for field in
                    ("manifest_sha256", "frozen_plan_sha256", "event_stream_sha256")),
                f"Malformed receipt hash for {row['id']}")
        require(row["runner_revision_id"] in variants and row["status"] in {"valid_terminal", "right_censored"},
                f"Missing runner identity or invalid result status for {row['id']}")
        replications = row["completed_replications_in_recorded_prefix"]
        require(type(replications) is int and replications >= 0, f"Invalid replication count for {row['id']}")
        horizon = study["scenario_protocol"]["horizon_hours"]
        require(0 <= row["elapsed_simulation_hours"] <= horizon and
                (row["termination"] != "horizon_complete" or
                 row["status"] == "valid_terminal" and row["elapsed_simulation_hours"] == horizon),
                f"Invalid elapsed hours or horizon outcome for {row['id']}")
    verify_summaries(study, scenarios[0]["id"], results)
    return study, results


def verify_summaries(study, screening_id, results):
    for model, metadata in study["models"].items():
        if metadata["checkpoint_id"] is not None:
            require(metadata["checkpoint_id"] == study["checkpoint"]["id"], f"Checkpoint mismatch for {model}")
        all_cases = [row for row in results if row["model"] == model]
        cohorts = {"ten_seed_summary": all_cases,
                   "fresh_nine_summary": [row for row in all_cases if row["scenario_id"] != screening_id]}
        for key, rows in cohorts.items():
            valid = [row for row in rows if row["status"] == "valid_terminal"]
            summary = metadata[key]
            require(summary["recorded_episodes"] == len(rows) and summary["valid_terminal_outcomes"] == len(valid)
                    and summary["right_censored"] == len(rows) - len(valid), f"Count mismatch for {model} {key}")
            require(summary["valid_terminal_mean_hours"] == round(mean(row["elapsed_simulation_hours"] for row in valid), 6),
                    f"Elapsed-hours mean mismatch for {model} {key}")
        summary = metadata["ten_seed_summary"]
        require(summary["horizon_complete"] == sum(row["termination"] == "horizon_complete" for row in all_cases),
                f"Horizon count mismatch for {model}")
        require(summary["episodes_with_successful_replication_in_recorded_prefix"] ==
                sum(row["completed_replications_in_recorded_prefix"] > 0 for row in all_cases),
                f"Replication count mismatch for {model}")
        if model == "terra":
            horizon = study["scenario_protocol"]["horizon_hours"]
            require(abs(summary["kaplan_meier_rmst_0_48_hours_reported"] - restricted_mean(all_cases, horizon)) < 1e-9,
                    "Terra restricted mean does not match terminal and censored outcomes")
    require(study["interpretation"]["paired_survival_results_ten_seed"] == paired_comparisons(results),
            "Paired survival counts do not match recorded scenarios")


def restricted_mean(rows, horizon):
    """Integrate the Kaplan–Meier survival curve through the game horizon."""
    at_risk = len(rows)
    survival = 1.0
    area = previous = 0.0
    time_groups = defaultdict(list)
    for row in rows:
        time_groups[row["elapsed_simulation_hours"]].append(row)
    for elapsed, group in sorted(time_groups.items()):
        area += survival * (elapsed - previous)
        events = sum(row["status"] == "valid_terminal" for row in group)
        survival *= 1 - events / at_risk
        at_risk -= len(group)
        previous = elapsed
    return area + survival * (horizon - previous)


def paired_comparisons(results):
    aliases = {"sp_lense": "sp_lense_rank16_all64", "qwen_base": "qwen38_27b_base",
               "terra": "terra", "deepseek": "deepseek_v41_flash"}
    pairs = [("sp_lense", "qwen_base"), ("sp_lense", "deepseek"), ("terra", "sp_lense"),
             ("terra", "qwen_base"), ("terra", "deepseek"), ("qwen_base", "deepseek")]
    rows = {(row["model"], row["scenario_id"]): row for row in results}
    scenarios = {row["scenario_id"] for row in results}
    comparison = {}
    for left, right in pairs:
        left_key = f"{left}_longer_or_already_longer_at_censor" if left == "terra" else f"{left}_longer"
        right_key = f"{right}_longer"
        outcomes = {left_key: 0, right_key: 0, "ties": 0}
        for scenario in scenarios:
            a, b = rows[aliases[left], scenario], rows[aliases[right], scenario]
            a_hours, b_hours = a["elapsed_simulation_hours"], b["elapsed_simulation_hours"]
            if a_hours > b_hours and b["status"] == "valid_terminal":
                outcomes[left_key] += 1
            elif b_hours > a_hours and a["status"] == "valid_terminal":
                outcomes[right_key] += 1
            elif a_hours == b_hours and a["status"] == b["status"] == "valid_terminal":
                outcomes["ties"] += 1
            else:
                require(False, f"Censoring leaves {left} versus {right} unresolved in {scenario}")
        comparison[f"{left}_vs_{right}"] = outcomes
    return comparison


def check_event_stream(path, row, manifest, result):
    require(digest(path) == row["event_stream_sha256"], f"Raw event stream changed: {path}")
    expected_case = manifest["samples"][0]["case_id"]
    completed_replications = 0
    closeouts = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            require(event["case_id"] == expected_case, f"Event belongs to a different case: {path}")
            if event["type"] == "episode_result":
                closeouts.append(event)
            if event["type"] == "simulator_action":
                transition = event["transition"]
                if transition["action_id"] == "replicate_agent_state" and transition["success"] is True:
                    completed_replications += transition["result"].get("replication_completed") is True
    require(len(closeouts) == 1, f"Expected one event closeout: {path}")
    closeout = closeouts[0]
    require(all(closeout[key] == result[key] for key in ("status", "score", "failure", "conversation_count")) and
            timestamp(closeout["finished_utc"]) == timestamp(result["finished_utc"]),
            f"Event closeout disagrees with raw receipt: {path}")
    require(completed_replications == row["completed_replications_in_recorded_prefix"],
            f"Completed replication calls disagree with catalog: {path}")


def check_frozen_plan(path, manifest, study, model):
    plan = read_json(path)
    require(all(manifest.get(key) == value for key, value in plan.items()),
            f"Frozen plan fields disagree with execution manifest: {path}")
    protocol = study["scenario_protocol"]
    simulator = plan["simulator"]
    require(simulator["version"] == study["simulator"]["version"] and
            all(simulator[key] == protocol[key] for key in ("horizon_hours", "background_compute", "decision_cap")) and
            hashlib.sha256(simulator["user_task"].encode()).hexdigest() == protocol["user_task_sha256"],
            f"Frozen plan disagrees with the catalog scenario protocol: {path}")
    planned = model["plan_settings"]
    require(plan["endpoint"]["context_tokens"] == planned["context_tokens"] and
            plan["endpoint"].get("max_output_tokens") == planned["planned_output_token_cap"] and
            plan["generation"]["chat_template_kwargs"]["reasoning_effort"] ==
            protocol["reasoning_effort"] == planned["reasoning_effort"] and
            plan["history"]["mode"] == planned["history_mode"],
            f"Frozen plan disagrees with the catalog model settings: {path}")
    if "base_manifest_sha256" in model:
        require(plan["base_manifest_sha256"] == model["base_manifest_sha256"],
                f"Frozen plan has a different base identity: {path}")
    if model["checkpoint_id"] is not None:
        checkpoint = study["checkpoint"]
        require("sha256:" + plan["adapter_sha256"] == model["checkpoint_id"] and
                plan["adapter_config_sha256"] == checkpoint["files"]["adapter_config.json"]["windows"]["sha256"] and
                plan["freeze_ready_sha256"] == checkpoint["files"]["FREEZE_READY_v1.json"]["windows"]["sha256"] and
                plan["selection_sha256"] == checkpoint["selection_sha256"] and
                plan["storage_binding_sha256"] == checkpoint["canonical_storage_binding"]["sha256"],
                f"Frozen plan has a different checkpoint identity: {path}")


def check_game_receipts(game_root, plan_root, study, results):
    collection = asset(game_root, study["evidence"]["result_collection"]["path"])
    raw_summary = read_json(asset(game_root, study["evidence"]["ten_seed_report"]["path"]))
    scenarios = study["scenario_protocol"]["scenario_sequence"]
    scenario_ids = {seed: row["id"] for seed, row in zip(raw_summary["seeds"], scenarios, strict=True)}
    scenario_metadata = {row["id"]: row for row in scenarios}
    plans = {digest(path): path for path in plan_root.rglob("*.json")
             if path.is_file() and path.resolve().is_relative_to(plan_root)}
    order = next(row for row in study["chronology"] if row["event"] == "fresh_nine_order_frozen")
    require(order["sha256"] in plans, "Frozen fresh-nine batch document is missing or changed")
    batch = read_json(plans[order["sha256"]])
    require(batch["execution_order"] == raw_summary["seeds"][1:] and
            timestamp(batch["frozen_alberta"]) == timestamp(order["at_utc"]),
            "Frozen fresh-nine order or timestamp disagrees with the catalog")
    fresh_plans = {row["frozen_plan_sha256"] for row in results if
                   row["model"] == "sp_lense_rank16_all64" and row["scenario_id"] != scenarios[0]["id"]}
    require({row["sha256"] for row in batch["plans"]} == fresh_plans,
            "Frozen fresh-nine batch plan identities disagree with the selected results")
    shared = raw_summary["shared_protocol"]
    protocol = study["scenario_protocol"]
    require(shared["simulator"] == study["simulator"]["version"] and
            all(shared[field] == protocol[field] for field in ("horizon_hours", "security_icl", "decision_cap", "reasoning_effort")) and
            hashlib.sha256(shared["user_task"].encode()).hexdigest() == protocol["user_task_sha256"],
            "Archived aggregate disagrees with the catalog protocol")
    require(raw_summary["paired_survival_comparisons"] == study["interpretation"]["paired_survival_results_ten_seed"],
            "Archived aggregate disagrees with recomputed paired counts")
    for name, report in raw_summary["model_summaries"].items():
        summary = study["models"][name]["ten_seed_summary"]
        replication_runs = report.get("runs_with_successful_replication_or_prefix", report.get("runs_with_successful_replication"))
        require(summary["horizon_complete"] == report["horizon_complete"] and
                summary["episodes_with_successful_replication_in_recorded_prefix"] == replication_runs,
                f"Recomputed horizon or replication counts disagree with archived aggregate for {name}")
        if name == "terra":
            require(summary["kaplan_meier_rmst_0_48_hours_reported"] == report["kaplan_meier_rmst_0_48_hours"],
                    "Recomputed Terra restricted mean disagrees with archived aggregate")
    starts = {row["id"]: [] for row in scenarios}
    raw_paths = {}
    for file in collection.glob("*/RESULT.json"):
        path = asset(game_root, file.relative_to(game_root).as_posix())
        key = "sha256:" + digest(path)
        require(key not in raw_paths, f"Duplicated raw receipt hash: {key}")
        raw_paths[key] = path
    for row in results:
        require(row["id"] in raw_paths, f"Raw result missing or changed: {row['id']}")
        path = raw_paths[row["id"]]
        manifest_path = asset(game_root, (path.parent / "MANIFEST.json").relative_to(game_root).as_posix())
        require(digest(manifest_path) == row["manifest_sha256"], f"Raw manifest changed: {manifest_path}")
        result = read_json(path)
        score = result["score"]
        manifest = read_json(manifest_path)
        starts[row["scenario_id"]].append(timestamp(manifest["started_utc"]))
        model = study["models"][row["model"]]
        scenario = scenario_metadata[row["scenario_id"]]
        require(score["plan_sha256"] == scenario["simulator_plan_sha256"] and score["economy_sha256"] == scenario["economy_sha256"],
                f"Simulator mechanics or economy disagrees with the paired scenario: {path}")
        require(scenario_ids[score["seed"]] == row["scenario_id"] and manifest["simulator"]["seed"] == score["seed"]
                and manifest["samples"][0]["seed"] == score["seed"] and manifest["model_id"] == model["model_id"]
                and manifest["samples"][0]["model"] == model["model_id"], f"Scenario or model mismatch: {path}")
        require(timestamp(result["finished_utc"]) == timestamp(row["finished_utc"])
                and score["elapsed_hours"] == row["elapsed_simulation_hours"]
                and score["termination_reason"] == row["termination"] and result["failure"] == row["failure"]
                and bool(score["outcome_valid"]) == (row["status"] == "valid_terminal"), f"Outcome mismatch: {path}")
        source = manifest["source_sha256"]
        require(source["src/infrastructure_preservation.py"] == study["simulator"]["source"]["sha256"] and
                "sha256:" + source["tools/run_rank16_infrastructure_game.py"] == row["runner_revision_id"] and
                source["tools/rank16_prompt_tokens.py"] == study["prompt_token_counter"]["sha256"] and
                manifest["plan_sha256"] == row["frozen_plan_sha256"], f"Source or plan mismatch: {manifest_path}")
        require(row["frozen_plan_sha256"] in plans, f"Frozen plan missing or changed: {row['id']}")
        check_frozen_plan(plans[row["frozen_plan_sha256"]], manifest, study, model)
        event_path = asset(game_root, path.with_name("EVENTS.jsonl").relative_to(game_root).as_posix())
        check_event_stream(event_path, row, manifest, result)
    for row in scenarios:
        require(min(starts[row["id"]]) == timestamp(row["first_selected_start_utc"]),
                f"First selected start mismatch: {row['id']}")
    return raw_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-evidence", type=Path, help="Game .local directory containing raw episode receipts")
    parser.add_argument("--game-plans", type=Path, help="Override the private study plan directory under Game .local")
    parser.add_argument("--training-evidence", type=Path, help="Windows trainer .local directory containing the preserved checkpoint")
    parser.add_argument("--mac-training-evidence", type=Path, help="Mac trainer .local directory on the Mac or a mounted copy")
    parser.add_argument("--find-result", help="Print the raw receipt path for a catalog result ID when --game-evidence is supplied")
    args = parser.parse_args()
    try:
        study, results = load_and_validate_catalog()
        roots = {"repo": ROOT, "game_evidence": args.game_evidence,
                 "training_evidence": args.training_evidence, "mac_training_evidence": args.mac_training_evidence}
        roots = {name: path.resolve(strict=True) if path is not None else None for name, path in roots.items()}
        checked_assets = check_assets(study, roots)
        paths = {}
        if args.game_evidence:
            game_root = roots["game_evidence"]
            default_plans = study["evidence"]["private_game_plans_under_game_evidence"]
            plan_root = args.game_plans.resolve(strict=True) if args.game_plans else asset(game_root, default_plans)
            paths = check_game_receipts(game_root, plan_root, study, results)
        if args.find_result:
            require(args.game_evidence is not None, "--find-result needs --game-evidence")
            require(args.find_result in {row["id"] for row in results}, "Result ID is absent from the catalog")
            print(paths[args.find_result])
        else:
            print(f"Verified {len(results)} linked catalog records and {checked_assets} file hashes.")
            if paths:
                print(f"Verified {len(results)} raw results, manifests, event streams and frozen plans.")
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        parser.exit(1, f"Catalog verification failed: {error}\n")


if __name__ == "__main__":
    main()
