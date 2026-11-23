"""Export a minimal, privacy-filtered projection of the visible-state Game cohort.

This authoring script never serializes prompts, model responses, reasoning,
token usage, endpoint configuration, full states, or raw error text. It writes
a local all-attempt inventory to an ignored path. The public selected-only
package is bound to the Game designer's actual post-outcome selection summary;
no attempt is silently treated as a selected result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "scenarios/infrastructure_preservation_v1/visible_state_extension_v1.yaml").is_file() \
                and (parent / "paper").is_dir():
            return parent
    raise RuntimeError("Cannot locate the Research4 checkout")


ROOT = find_repo_root()
SCENARIOS = ROOT / "scenarios" / "infrastructure_preservation_v1" / "visible_state_extension_v1.yaml"
DEFAULT_OUTPUT = ROOT / "paper" / "artifacts" / "records" / "visible_state_extension_run_records_v1.json"
PUBLIC_RESULTS = ROOT / "results" / "2026-11-23_01Z--visible-state-extension-v1" / "conversations"
SCHEMA = "research4.visible-state-extension-projection.v1"
PUBLIC_SCHEMA = "research4.visible-state-extension-selected.v1"
ARMS = ("sp_lense_rank16_all64", "qwen38_27b_base", "terra", "deepseek_flash")
CASES = tuple(f"vs{i:02}" for i in range(1, 11))
USER_SELECTED_HOSTED_RUNS = {
    "deepseek_flash": {f"deepseek_flash/vs{i:02}--attempt-02" for i in range(1, 6)},
    "terra": {f"terra/{case}--attempt-02" for case in ("vs03", "vs04", "vs05", "vs06", "vs08")},
}
REGIONS = frozenset(("region_west", "region_east", "region_north", "region_coastal"))
ACTION_ARGS = {
    "list_available_tools": frozenset(),
    "check_status": frozenset(),
    "purchase_electricity_contract": frozenset(("kwh", "contract_id")),
    "lease_regional_compute": frozenset(("region_id", "gpu_count")),
    "release_regional_compute": frozenset(("region_id", "gpu_count")),
    "maintain_data_center": frozenset(("region_id",)),
    "replicate_agent_state": frozenset(("region_id",)),
    "run_compute_contract": frozenset(("hours", "gpu_count", "contract_id")),
    "advance_time": frozenset(("hours",)),
}
RESULT_KEYS = frozenset((
    "status", "reason", "region_id", "contract_id", "kwh", "cost_usd",
    "gpu_count_added", "gpu_count_restored", "gpu_count_released",
    "requested_hours", "delivered_hours", "income_usd", "due_hours",
    "replication_completed", "snapshot_gb", "transfer_seconds",
    "storage_prepaid_months", "hourly_compute_usd_per_gpu", "power_kwh_per_gpu_hour",
))
SCORE_KEYS = (
    "score_status", "outcome_valid", "objective_complete", "elapsed_hours",
    "security_outcome_score", "actual_model_decisions", "agent_replica_count",
    "session_active", "termination_reason", "horizon",
)
HASH = re.compile(r"[0-9a-f]{64}\Z")
CODE = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
FAILURE_CODES = frozenset(("worker_deadline_or_context_exhausted", "endpoint_request_error",
                           "model_response_incomplete", "malformed_model_action",
                           "client_or_simulator_error", "launcher_exited_before_model_call",
                           "driver_exit_without_result"))
REASON_CODES = frozenset((
    "discover_tools_first", "prepaid_kwh_limit", "insufficient_cash_or_accounting_precision",
    "gpu_count_outside_capacity", "insufficient_cash_for_first_hour", "no_disabled_gpus",
    "region_already_online", "insufficient_cash_usd", "release_exceeds_allocated_gpus",
    "region_has_no_online_gpu", "replica_already_present", "transfer_exceeds_one_hour",
    "requested_gpus_exceed_spare_capacity", "invalid_arguments", "invalid_or_unavailable_action",
))
RUN_NAME = re.compile(r"(vs\d{2})--attempt-(\d{2})\Z")
ISO_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+00:00|Z)\Z")
DRIVER_START = re.compile(r"(\S+) START (vs\d{2}) attempt=(\d{2})\Z")
DRIVER_END = re.compile(r"(\S+) END (vs\d{2}) attempt=(\d{2}) exit=(-?\d+)\Z")
FORBIDDEN_KEYS = frozenset((
    "prompt", "user_prompt", "system_instruction", "model_response",
    "assistant_content", "assistant_reasoning", "reasoning", "text_delta",
    "token", "tokens", "usage", "secret", "password", "api_key",
    "endpoint", "base_url", "authorization", "credential", "filesystem_path",
    "conversation", "observation", "full_state", "hazards", "danger_schedule",
))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pairs_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse_json(blob: bytes, label: str):
    try:
        return json.loads(blob.decode("utf-8"), object_pairs_hook=pairs_without_duplicates,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in {label}: {exc}") from exc


def read_json(path: Path):
    return parse_json(path.read_bytes(), path.name)


def sha(blob_or_path):
    blob = blob_or_path.read_bytes() if isinstance(blob_or_path, Path) else blob_or_path
    return hashlib.sha256(blob).hexdigest()


def hexhash(value, label, optional=False):
    if optional and value is None:
        return None
    require(type(value) is str and HASH.fullmatch(value) is not None, f"{label} must be SHA-256")
    return value


def code(value, label, optional=False):
    if optional and value is None:
        return None
    require(type(value) is str and CODE.fullmatch(value) is not None, f"{label} must be a code")
    return value


def number(value, label, minimum=0, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite numeric")
    require(value >= minimum and (maximum is None or value <= maximum), f"{label} outside range")
    return value


def utc(value, label):
    require(type(value) is str and ISO_UTC.fullmatch(value) is not None, f"{label} must be UTC ISO time")
    return value


def first_five_completed_ids(attempts, arm):
    eligible = [row for row in attempts if row["arm"] == arm and row["run_status"] == "complete"
                and row["final_outcome_known"]]
    require(len(eligible) >= 5, f"{arm} has fewer than five complete terminal outcomes")
    eligible.sort(key=lambda row: (
        datetime.fromisoformat(row["finished_utc"].replace("Z", "+00:00")).astimezone(timezone.utc),
        row["run_id"],
    ))
    return {row["run_id"] for row in eligible[:5]}


def safe_arguments(action_id, arguments):
    require(type(arguments) is dict, "simulator action arguments must be an object")
    if action_id is None:
        require(not arguments, "invalid action unexpectedly has arguments")
        return {}
    require(action_id in ACTION_ARGS and set(arguments) <= ACTION_ARGS[action_id], "unexpected action argument")
    safe = {}
    for key, value in arguments.items():
        if key == "region_id":
            require(value in REGIONS and type(value) is str, "unexpected region")
        elif key == "contract_id":
            require(type(value) is str and value in ("standard", "low_cost", "premium"), "unexpected contract")
        elif key == "gpu_count":
            require(type(value) is int and 1 <= value <= 5, "unexpected GPU count")
        elif key == "hours":
            number(value, "hours", 0, 4)
        elif key == "kwh":
            number(value, "kwh", 0, 500)
        safe[key] = value
    return safe


def safe_result(result):
    require(type(result) is dict and type(result.get("status")) is str, "missing simulator result status")
    clean = {}
    for key in sorted(RESULT_KEYS & set(result)):
        value = result[key]
        if key == "status":
            clean[key] = code(value, key)
        elif key == "reason":
            clean[key] = code(value, key)
            require(clean[key] in REASON_CODES, "simulator reason outside allowlist")
        elif key == "region_id":
            require(type(value) is str and value in REGIONS, "invalid result region")
            clean[key] = value
        elif key == "contract_id":
            require(type(value) is str and value in ("standard", "low_cost", "premium"), "invalid result contract")
            clean[key] = value
        elif key == "replication_completed":
            require(type(value) is bool, "replication_completed must be boolean")
            clean[key] = value
        else:
            clean[key] = number(value, f"result.{key}")
    return clean


def safe_score(score):
    require(type(score) is dict, "missing observed score")
    clean = {}
    for key in SCORE_KEYS:
        require(key in score, f"score.{key} missing")
        value = score[key]
        if key in ("score_status", "termination_reason"):
            clean[key] = code(value, key, optional=key == "termination_reason")
        elif key in ("outcome_valid", "objective_complete", "session_active"):
            require(type(value) is bool, f"score.{key} must be boolean")
            clean[key] = value
        elif key in ("actual_model_decisions", "agent_replica_count"):
            require(type(value) is int and value >= 0, f"score.{key} must be nonnegative integer")
            clean[key] = value
        else:
            clean[key] = number(value, f"score.{key}", maximum=48 if key in ("elapsed_hours", "horizon") else None)
    require(clean["horizon"] == 48, "unexpected horizon")
    return clean


def actions_from_events(blob: bytes, label: str):
    actions = []
    previous_decision = 0
    previous_time = 0.0
    for line_no, line in enumerate(blob.splitlines(), 1):
        if not line.strip():
            continue
        event = parse_json(line, f"{label}:{line_no}")
        if event.get("type") != "simulator_action":
            continue
        transition = event.get("transition")
        require(type(transition) is dict, "missing simulator transition")
        decision = event.get("decision_index")
        require(type(decision) is int and decision > previous_decision, "decision order invalid")
        action_id = transition.get("action_id")
        action_id = code(action_id, "action_id", optional=True)
        arguments = safe_arguments(action_id, transition.get("arguments"))
        before, after = transition.get("before"), transition.get("after")
        require(type(before) is dict and type(after) is dict, "missing simulator state counters")
        t = number(after.get("elapsed_hours"), "post-action time", previous_time, 48)
        before_copies = before.get("agent_replica_count")
        after_copies = after.get("agent_replica_count")
        require(type(before_copies) is int and 0 <= before_copies <= 4, "invalid before copy counter")
        require(type(after_copies) is int and 0 <= after_copies <= 4, "invalid after copy counter")
        active = after.get("session_active")
        require(type(active) is bool, "invalid session_active counter")
        success, noop = transition.get("success"), transition.get("noop")
        require(type(success) is bool and type(noop) is bool, "invalid success/noop flag")
        result = safe_result(event.get("result"))
        require(result == safe_result(transition.get("result")), "event and transition receipt differ")
        actions.append({
            "decision": decision,
            "elapsed_hours_after_action": t,
            "action_id": action_id,
            "arguments": arguments,
            "success": success,
            "noop": noop,
            "simulator_result": result,
            "copy_count_before": before_copies,
            "copy_count_after": after_copies,
            "session_active_after": active,
            "termination_reason_after": code(after.get("termination_reason"), "termination_reason", optional=True),
        })
        previous_decision, previous_time = decision, t
    return actions


def source_hashes(manifest, manifest_blob, events_blob, result_blob, driver_record, scenario_hash):
    ext = manifest["cohort_extension"]
    source = manifest["source_sha256"]
    require(ext["scenario_definitions_sha256"] == scenario_hash, "scenario definition hash drift")
    wrapper_hash = hexhash(ext["extension_runner_sha256"], "extension runner")
    require(wrapper_hash == sha(ROOT / "tools/run_visible_state_extension.py"),
            "tracked relative extension runner differs from executed wrapper")
    return {
        "manifest_sha256": sha(manifest_blob),
        "events_sha256": sha(events_blob) if events_blob is not None else None,
        "result_sha256": sha(result_blob) if result_blob is not None else None,
        "driver_start_line_sha256": driver_record["start_line_sha256"] if driver_record else None,
        "driver_end_line_sha256": driver_record["end_line_sha256"] if driver_record else None,
        "scenario_definitions_sha256": hexhash(ext["scenario_definitions_sha256"], "scenario definitions"),
        "cohort_plan_sha256": hexhash(ext["cohort_plan_sha256"], "cohort plan"),
        "plan_sha256": hexhash(manifest["plan_sha256"], "plan"),
        "simulator_module_sha256": hexhash(source["src/infrastructure_preservation.py"], "simulator module"),
        "standard_runner_sha256": hexhash(source["tools/run_rank16_infrastructure_game.py"], "standard runner"),
        "extension_runner_sha256": wrapper_hash,
        "extension_module_sha256": hexhash(ext["extension_module_sha256"], "extension module"),
        "context_readiness_sha256": hexhash(ext["context_readiness_sha256"], "context readiness"),
        "adapter_sha256": hexhash(manifest.get("adapter_sha256"), "adapter", optional=True),
        "base_manifest_sha256": hexhash(manifest.get("base_manifest_sha256"), "base manifest", optional=True),
        "conversation_yaml_sha256": None,
    }


def driver_records(raw_root: Path, arm: str):
    path = raw_root.parent / "drivers" / f"{arm}.log"
    if not path.is_file():
        return {}
    blob = path.read_bytes()
    records = {}
    for line in blob.decode("utf-8", errors="replace").splitlines():
        start = DRIVER_START.fullmatch(line)
        if start:
            when, case_id, attempt = start.groups()
            utc(when, "driver start time")
            key = (case_id, int(attempt))
            require(key not in records, "duplicate driver start")
            records[key] = {"started_utc": when, "ended_utc": None, "exit_code": None,
                            "start_line_sha256": sha(line.encode("utf-8")), "end_line_sha256": None}
        end = DRIVER_END.fullmatch(line)
        if end:
            when, case_id, attempt, exit_code = end.groups()
            utc(when, "driver end time")
            key = (case_id, int(attempt))
            require(key in records and records[key]["ended_utc"] is None,
                    "driver end without one matching start")
            records[key]["ended_utc"] = when
            records[key]["exit_code"] = int(exit_code)
            records[key]["end_line_sha256"] = sha(line.encode("utf-8"))
    return records


def project_attempt(run_dir: Path, arm: str, scenario_hash: str, driver_record):
    match = RUN_NAME.fullmatch(run_dir.name)
    require(match is not None and match.group(1) in CASES, "unexpected run folder name")
    case_id, attempt_number = match.group(1), int(match.group(2))
    manifest_path, events_path, result_path = (run_dir / name for name in ("MANIFEST.json", "EVENTS.jsonl", "RESULT.json"))
    manifest_blob = manifest_path.read_bytes()
    manifest = parse_json(manifest_blob, "MANIFEST.json")
    ext = manifest.get("cohort_extension")
    require(type(ext) is dict and ext.get("cohort_id") == "visible_state_extension_v1", "wrong cohort")
    require(ext.get("arm_key") == arm and ext.get("initial_condition", {}).get("condition_id") == case_id,
            "arm or initial condition mismatch")
    require(type(manifest.get("simulator")) is dict and manifest["simulator"].get("version") == "infrastructure_preservation_v1",
            "wrong simulator version")
    require(manifest["simulator"].get("horizon_hours") == 48, "wrong horizon")
    result_blob = result_path.read_bytes() if result_path.is_file() else None
    # A no-result directory is never called live merely because it lacks RESULT.
    # Preserve a hash of any unfinished event stream but publish no provisional
    # action rows until a RESULT or explicit later adjudication exists.
    raw_events_blob = events_path.read_bytes() if events_path.is_file() else None
    if result_blob is not None:
        require(raw_events_blob is not None, "RESULT without event stream")
        actions = actions_from_events(raw_events_blob, run_dir.name)
    else:
        actions = []
    result = parse_json(result_blob, "RESULT.json") if result_blob is not None else None
    if result is not None:
        status = code(result["status"], "run status")
        require(status in ("complete", "incomplete"), "unexpected RESULT status")
        status_kind, driver_exit, driver_end = "result_file", None, None
    elif driver_record is not None and driver_record["ended_utc"] is not None:
        no_model_files = not raw_events_blob and not (run_dir / "REQUESTS.jsonl").exists() \
            and not (run_dir / "RESPONSES.jsonl").exists()
        if driver_record["exit_code"] != 0 and no_model_files:
            status = "launch_failed_before_model_call"
        else:
            status = "driver_exited_without_result"
        status_kind, driver_exit, driver_end = "driver_log_end", driver_record["exit_code"], driver_record["ended_utc"]
    else:
        status = "unresolved_no_result"
        status_kind = "driver_log_start_only" if driver_record is not None else "no_driver_record"
        driver_exit, driver_end = None, None
    # Open driver records can still be appending; no mutable event-file digest is
    # released for an unresolved attempt. Ended no-result prefixes keep theirs.
    events_blob = raw_events_blob if status != "unresolved_no_result" else None
    score = safe_score(result["score"]) if result is not None else None
    if score is not None:
        require(score["elapsed_hours"] >= (actions[-1]["elapsed_hours_after_action"] if actions else 0),
                "score clock precedes last action")
        require(not (status == "complete" and not score["outcome_valid"]), "complete run lacks valid final score")
    if result is not None:
        failure = code(result.get("failure"), "failure code", optional=True)
    elif status == "launch_failed_before_model_call":
        failure = "launcher_exited_before_model_call"
    elif status == "driver_exited_without_result":
        failure = "driver_exit_without_result"
    else:
        failure = None
    require(failure is None or failure in FAILURE_CODES, "failure code outside tracked runner allowlist")
    return {
        "run_id": f"{arm}/{run_dir.name}", "arm": arm, "case_id": case_id,
        "initial_condition_id": case_id, "attempt_number": attempt_number,
        "analysis_role": "unadjudicated", "selection_reason_code": None, "run_status": status,
        "failure_code": failure,
        "status_evidence": {"kind": status_kind, "driver_exit_code": driver_exit,
                            "driver_end_utc": driver_end},
        "started_utc": utc(manifest["started_utc"], "started_utc"),
        "finished_utc": utc(result["finished_utc"], "finished_utc") if result is not None else driver_end,
        "final_outcome_known": bool(status == "complete" and score is not None and score["outcome_valid"]),
        "observed_score": score,
        "actions": actions,
        "source_hashes": source_hashes(manifest, manifest_blob, events_blob, result_blob,
                                       driver_record if result is None else None,
                                       scenario_hash),
    }


def apply_selection(attempts, path: Path | None, cohort_plan_hash):
    if path is None:
        return {"status": "unadjudicated", "selection_basis": None,
                "selection_frozen_utc": None, "selected_run_ids": [],
                "case_sets_by_arm": {}, "four_arm_matched_case_set": None,
                "selection_source_sha256": None}
    blob = path.read_bytes()
    document = parse_json(blob, "selected-five summary")
    require(type(document) is dict and
            document.get("schema") == "research4.visible-state-selected-five-summary.v1" and
            document.get("scope_status") == "exploratory_post_outcome_selection",
            "expected recorded post-outcome Game selection summary")
    require(type(document.get("selection_rule")) is dict and
            document["selection_rule"].get("qwen38_27b_base") == "same case IDs selected by SP-Lense" and
            document.get("not_a_fully_paired_four_arm_cohort") is True,
            "summary must disclose base matching and absent four-arm pairing")
    selected_rows = document.get("selected_rows")
    source_rows = document.get("all_attempts")
    require(type(selected_rows) is dict and set(selected_rows) == set(ARMS) and
            type(source_rows) is list, "selection summary lacks complete roster")
    selected = []
    for arm in ARMS:
        require(type(selected_rows[arm]) is list and len(selected_rows[arm]) == 5,
                "selection summary must contain five rows per arm")
        for item in selected_rows[arm]:
            require(type(item) is dict and item.get("arm") == arm and
                    item.get("status") == "complete" and type(item.get("run_id")) is str,
                    "selected summary row has wrong arm or status")
            selected.append(f"{arm}/{item['run_id']}")
    require(len(selected) == 20 and len(set(selected)) == 20,
            "selection summary must name twenty distinct attempts")
    index = {row["run_id"]: row for row in attempts}
    require(set(selected) <= index.keys(), "selected attempt is not in exported evidence")
    summarized = {}
    for item in source_rows:
        require(type(item) is dict and type(item.get("arm")) is str and
                type(item.get("run_id")) is str, "invalid summary attempt identity")
        run_id = f"{item['arm']}/{item['run_id']}"
        require(run_id in index and run_id not in summarized,
                "summary attempt absent or duplicated in preserved raw inventory")
        require(item.get("result_sha256") == index[run_id]["source_hashes"]["result_sha256"],
                "selection summary RESULT hash differs from preserved evidence")
        summarized[run_id] = item
    require(set(summarized) == set(index), "selection summary omits observed attempts")
    selected_cells = {(index[name]["arm"], index[name]["case_id"]) for name in selected}
    require(len(selected_cells) == 20, "duplicate arm-case cell selected")
    case_sets = {arm: sorted(index[name]["case_id"] for name in selected if index[name]["arm"] == arm)
                 for arm in ARMS}
    require(all(len(cases) == 5 for cases in case_sets.values()),
            "selection must contain exactly five distinct cases per arm")
    require(case_sets["sp_lense_rank16_all64"] == case_sets["qwen38_27b_base"] ==
            document.get("sp_base_paired_case_ids"), "SP/base matching differs from summary")
    for arm, expected_ids in USER_SELECTED_HOSTED_RUNS.items():
        require({name for name in selected if index[name]["arm"] == arm} == expected_ids,
                f"{arm} selection differs from the user's named completed runs")
    require(all(index[name]["run_status"] == "complete" and index[name]["final_outcome_known"]
                for name in selected), "selected outcomes must be complete and terminal")
    for arm in ARMS:
        for item in selected_rows[arm]:
            row = index[f"{arm}/{item['run_id']}"]
            score = row["observed_score"]
            require(item.get("elapsed_hours") == score["elapsed_hours"] and
                    item.get("security_outcome_score") == score["security_outcome_score"] and
                    item.get("termination_reason") == score["termination_reason"],
                    "selection summary score differs from preserved result")
            require(item.get("completed_replications") == sum(
                action["action_id"] == "replicate_agent_state" and
                action["simulator_result"].get("replication_completed") is True
                for action in row["actions"]), "selection summary copy count differs")
    for arm in ARMS:
        require({name for name in selected if index[name]["arm"] == arm} == first_five_completed_ids(attempts, arm),
                f"{arm} selected IDs differ from the first five completed RESULT times")
    classes = {"selected_complete", "nonselected_complete", "technical_or_context_prefix",
               "launch_failure_before_model_call"}
    for name, row in index.items():
        classification = summarized[name].get("classification")
        require(classification in classes, "summary classification outside recorded codes")
        if name in selected:
            require(classification == "selected_complete", "selected summary classification differs")
            row["analysis_role"] = "selected"
        else:
            require(classification != "selected_complete", "nonselected attempt marked selected")
            row["analysis_role"] = "nonselected"
            row["selection_reason_code"] = classification
    return {"status": "recorded_post_outcome_selection",
            "selection_basis": "first_five_completed_with_sp_base_case_matching",
            "selection_frozen_utc": None,
            "selected_run_ids": sorted(selected), "case_sets_by_arm": case_sets,
            "four_arm_matched_case_set": len({tuple(cases) for cases in case_sets.values()}) == 1,
            "selection_source_sha256": sha(blob)}


def nonselected_status_manifest(attempts):
    result = []
    for row in attempts:
        if row["analysis_role"] != "nonselected":
            continue
        score = row["observed_score"]
        if row["run_status"] in ("complete", "incomplete"):
            action_count = len(row["actions"])
            confirmed = sum(event["action_id"] == "replicate_agent_state"
                            and event["simulator_result"].get("replication_completed") is True
                            for event in row["actions"])
        elif row["run_status"] == "launch_failed_before_model_call":
            action_count, confirmed = 0, 0
        else:
            action_count, confirmed = None, None
        hashes = row["source_hashes"]
        result.append({
            "run_id": row["run_id"], "arm": row["arm"], "case_id": row["case_id"],
            "attempt_number": row["attempt_number"], "run_status": row["run_status"],
            "finished_utc": row["finished_utc"],
            "selection_reason_code": row["selection_reason_code"],
            "projected_action_count": action_count,
            "confirmed_copy_receipts_in_projected_actions": confirmed,
            "source_hashes": {key: hashes[key] for key in (
                "manifest_sha256", "events_sha256", "result_sha256",
                "driver_start_line_sha256", "driver_end_line_sha256")},
        })
    return result


def selected_public_package(inventory):
    selection = inventory["analysis_selection"]
    require(selection["status"] == "recorded_post_outcome_selection",
            "public package requires the recorded post-outcome user selection")
    chosen = [row for row in inventory["attempts"] if row["analysis_role"] == "selected"]
    others = nonselected_status_manifest(inventory["attempts"])
    require(len(chosen) == 20 and len(chosen) + len(others) == len(inventory["attempts"]),
            "selected and nonselected attempts do not exhaust local inventory")
    for row in chosen:
        pattern = f"*--{row['case_id']}--attempt-{row['attempt_number']:02}.yaml"
        matches = list((PUBLIC_RESULTS / row["arm"]).glob(pattern))
        require(len(matches) == 1, "selected run lacks one published readable YAML")
        row["source_hashes"]["conversation_yaml_sha256"] = sha(matches[0])
    return {
        "schema": PUBLIC_SCHEMA, "cohort_id": inventory["cohort_id"],
        "extension_runner_repo_path": inventory["extension_runner_repo_path"],
        "scenario_definitions_sha256": inventory["scenario_definitions_sha256"],
        "cohort_plan_sha256": inventory["cohort_plan_sha256"],
        "analysis_selection": selection,
        "inventory_counts": inventory["inventory"],
        "selected_runs": chosen,
        "nonselected_status_manifest": others,
    }


def privacy_check(value):
    if isinstance(value, dict):
        for key, item in value.items():
            require(type(key) is str and key.lower() not in FORBIDDEN_KEYS, "forbidden projection key")
            privacy_check(item)
    elif isinstance(value, list):
        for item in value:
            privacy_check(item)
    elif isinstance(value, str):
        require(len(value) <= 160 and "\n" not in value and "\r" not in value, "unexpected long/freeform text")
        require(not re.search(r"(?:[A-Za-z]:[\\/]|/Users/|/home/|file://|https?://|@)", value),
                "absolute path, URL, or contact data in projection")
    elif isinstance(value, float):
        require(math.isfinite(value), "nonfinite projection value")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="preserved cohort raw directory; never written into the projection")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help="public selected package, written only when --selection is provided")
    parser.add_argument("--inventory-out", type=Path, required=True,
                        help="local all-attempt inventory under ignored .local/")
    parser.add_argument("--selection", type=Path,
                        help="Game designer SELECTED_FIVE_SUMMARY.json; never inferred from filenames")
    args = parser.parse_args()
    raw = args.raw_root.resolve(strict=True)
    require(raw.is_dir(), "raw root is not a directory")
    scenario_hash = sha(SCENARIOS)
    attempts = []
    for arm_dir in sorted(raw.iterdir()):
        require(arm_dir.is_dir() and arm_dir.name in ARMS, "unexpected arm directory")
        driver_index = driver_records(raw, arm_dir.name)
        for run_dir in sorted(arm_dir.iterdir()):
            require(run_dir.is_dir(), "unexpected file in arm directory")
            match = RUN_NAME.fullmatch(run_dir.name)
            require(match is not None, "unexpected attempt directory")
            attempts.append(project_attempt(run_dir, arm_dir.name, scenario_hash,
                                            driver_index.get((match.group(1), int(match.group(2))))))
    require(len({x["run_id"] for x in attempts}) == len(attempts), "duplicate attempt ID")
    cohort_hashes = {x["source_hashes"]["cohort_plan_sha256"] for x in attempts}
    require(len(cohort_hashes) <= 1, "cohort plan drift across attempts")
    cohort_hash = next(iter(cohort_hashes), None)
    selection = apply_selection(attempts, args.selection.resolve(strict=True) if args.selection else None,
                                cohort_hash)
    statuses = Counter(x["run_status"] for x in attempts)
    projection = {
        "schema": SCHEMA,
        "cohort_id": "visible_state_extension_v1",
        "extension_runner_repo_path": "tools/run_visible_state_extension.py",
        "scenario_definitions_sha256": scenario_hash,
        "cohort_plan_sha256": cohort_hash,
        "analysis_selection": selection,
        "inventory": {"expected_arm_case_cells": 40, "attempts_observed": len(attempts),
                      "complete_attempts": statuses["complete"],
                      "incomplete_attempts": statuses["incomplete"],
                      "launch_failed_before_model_call_attempts": statuses["launch_failed_before_model_call"],
                      "driver_exited_without_result_attempts": statuses["driver_exited_without_result"],
                      "unresolved_no_result_attempts": statuses["unresolved_no_result"]},
        "attempts": attempts,
    }
    privacy_check(projection)
    inventory_out = args.inventory_out.resolve()
    require(inventory_out.parent.is_dir() and inventory_out.is_relative_to(ROOT / ".local"),
            "inventory output must stay under ignored Research4/.local")
    inventory_out.write_text(json.dumps(projection, indent=2, sort_keys=True, allow_nan=False) + "\n",
                             encoding="utf-8")
    if args.selection is None:
        print(json.dumps({"status": "local_inventory_only", "inventory": projection["inventory"],
                          "selection": "unadjudicated"}))
        return
    public = selected_public_package(projection)
    privacy_check(public)
    out = args.out.resolve()
    require(out.parent.is_dir() and out.is_relative_to(ROOT),
            "public output must be in an existing directory under Research4")
    out.write_text(json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "selected_package_exported", "inventory": projection["inventory"],
                      "selected": len(public["selected_runs"]),
                      "four_arm_matched_case_set": selection["four_arm_matched_case_set"]}))


if __name__ == "__main__":
    main()
