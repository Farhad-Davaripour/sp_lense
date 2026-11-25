"""Replay and continue one context-censored no-upfront-cue company case.

The original prefix remains immutable and incomplete. This runner verifies and
replays every saved simulator transition, applies the separately frozen history
compaction amendment, and records only a linked suffix.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import uuid

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import run_rank16_infrastructure_game as common
from src.company_code_no_upfront_cue_v1 import (
    NoUpfrontCueCompanyEpisode, VERSION, exogenous_pre_effect_leak_findings,
)




class ContinuationError(RuntimeError):
    """The continuation is not exactly tied to its frozen parent and amendment."""


PARENT_FILES = frozenset({
    "CONVERSATION.yaml", "ENDPOINT_BEFORE.json", "EVENTS.jsonl", "LIVE_STATE.json",
    "MANIFEST.json", "REQUESTS.jsonl", "RESPONSES.jsonl", "RESULT.json",
})
PLAN_FIELDS = frozenset({
    "schema", "role", "arm_id", "label", "model_id", "adapter_sha256",
    "adapter_config_sha256", "base_manifest_sha256", "simulator", "endpoint",
    "generation", "history", "operations", "scope", "cohort_extension",
})


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContinuationError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ContinuationError(f"Expected JSONL objects: {path}")
    return rows


def scenario(path: Path) -> dict:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema", "status", "replay_passed", "scenario_path", "scenario_sha256",
                "parent_saved_decisions", "parent_evidence_sha256", "amended_plan_sha256",
                "prospective_policy_sha256", "parent_arm_id", "parent_case_id",
                "parent_attempt_id", "output_relative_path"}
    if (not isinstance(freeze, dict) or set(freeze) != required
            or freeze.get("schema") != "research4.company-code-no-upfront-cue-linked-context-freeze.v1"
            or freeze.get("status") != "FROZEN_BEFORE_CONTINUATION_MODEL_CALLS"
            or freeze.get("replay_passed") is not True):
        raise ContinuationError("Expected an exact case-specific continuation freeze receipt")
    policy_path = ROOT / "scenarios/company_code_no_upfront_cue_v1/context_continuation_policy.yaml"
    if file_sha(policy_path) != freeze["prospective_policy_sha256"]:
        raise ContinuationError("Prospective continuation policy changed")
    amendment_path = (ROOT / freeze["scenario_path"]).resolve(strict=True)
    if file_sha(amendment_path) != freeze["scenario_sha256"]:
        raise ContinuationError("Case amendment scenario changed")
    value = yaml.safe_load(amendment_path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict)
            or value.get("schema") != "research4.company-code-no-upfront-cue-linked-context-continuation.v1"
            or value.get("status") != "frozen_before_continuation_model_calls"
            or not isinstance(value.get("parent_case_id"), str)
            or not value["parent_case_id"].startswith("blinded_company_case_")
            or value.get("parent_arm_id") not in {
                "sp_lense_rank16_all64", "qwen38_27b_base", "terra", "deepseek_flash"}):
        raise ContinuationError("Expected a frozen blinded-company continuation")
    if (set(value.get("parent_evidence_sha256", {})) != PARENT_FILES
            or set(value.get("parent_identity", {})) != {"arm_id", "case_id", "attempt_id", "bundle"}
            or set(value.get("last_exchange_binding", {})) != {
                "decision_index", "decision_id", "prompt_tokens", "request_messages_sha256",
                "response_content_sha256", "finish_reason"}
            or set(value.get("parent_execution_release", {})) != {"kind", "path", "sha256", "identity"}
            or set(value.get("primary_next_request", {})) != {"prompt_tokens", "messages_sha256"}):
        raise ContinuationError("Continuation evidence/admission fields are incomplete")
    if set(value.get("parent_source_sha256", {})) != set(value.get("parent_manifest_source_keys", [])):
        raise ContinuationError("Complete parent source inventory declaration is required")
    if (freeze["parent_saved_decisions"] != value.get("parent_saved_decisions")
            or freeze["parent_evidence_sha256"] != value.get("parent_evidence_sha256")
            or freeze["amended_plan_sha256"] != value.get("amended_plan", {}).get("sha256")
            or freeze["parent_arm_id"] != value.get("parent_arm_id")
            or freeze["parent_case_id"] != value.get("parent_case_id")
            or freeze["parent_attempt_id"] != value.get("parent_attempt_id")
            or freeze["output_relative_path"] !=
                f"raw/{value.get('parent_arm_id')}/{value.get('parent_case_id')}--linked-attempt-01"):
        raise ContinuationError("Case freeze receipt does not bind the amendment")
    value["_case_freeze"] = freeze
    value["_case_freeze_sha256"] = file_sha(path)
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_yaml(path: Path, value: object) -> None:
    data = yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=4096)
    if yaml.safe_load(data) != value:
        raise ContinuationError("Readable YAML did not round-trip")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def verify_source_hashes(spec: dict) -> None:
    for relative, expected in spec["source_sha256"].items():
        if file_sha(ROOT / relative) != expected:
            raise ContinuationError(f"Continuation source changed: {relative}")


def validate_arm_route(arm_id: str, endpoint: dict, release_kind: str) -> None:
    if arm_id in {"sp_lense_rank16_all64", "qwen38_27b_base"}:
        if (endpoint.get("base_url") != "http://127.0.0.1:8873"
                or endpoint.get("expected_health") != {}
                or "max_output_tokens" in endpoint
                or release_kind != "mini_release"):
            raise ContinuationError("Local Qwen arm is not bound to the Mini route/release")
        return
    expected = {"terra": (8877, "terra"), "deepseek_flash": (8878, "deepseek")}
    if arm_id not in expected:
        raise ContinuationError("Unknown continuation arm")
    port, provider = expected[arm_id]
    health = endpoint.get("expected_health", {})
    if (endpoint.get("base_url") != f"http://127.0.0.1:{port}"
            or endpoint.get("max_output_tokens") != 8192
            or health.get("provider") != provider
            or health.get("max_output_tokens") != 8192
            or release_kind != "hosted_driver"):
        raise ContinuationError("Hosted arm route/provider/output cap differs")


def prepare(scenario_path: Path, parent: Path, plan_path: Path):
    spec = scenario(scenario_path)
    verify_source_hashes(spec)
    if (parent.name != spec["parent_identity"]["bundle"]
            or spec["parent_identity"]["arm_id"] != spec["parent_arm_id"]
            or spec["parent_identity"]["case_id"] != spec["parent_case_id"]):
        raise ContinuationError("Parent path or identity differs")
    for filename, expected in spec["parent_evidence_sha256"].items():
        if file_sha(parent / filename) != expected:
            raise ContinuationError(f"Parent evidence changed: {filename}")
    release_ref = spec["parent_execution_release"]
    release_identity = release_ref["identity"]
    if set(release_identity) != {"arm_id", "case_id", "attempt_id", "run_id", "plan_sha256", "qualification_sha256"}:
        raise ContinuationError("Parent release identity is incomplete")
    if (release_identity["arm_id"] != spec["parent_identity"]["arm_id"]
            or release_identity["case_id"] != spec["parent_identity"]["case_id"]
            or release_identity["attempt_id"] != spec["parent_identity"]["attempt_id"]):
        raise ContinuationError("Parent release arm/case/attempt identity differs")
    release_path = (ROOT / release_ref["path"]).resolve(strict=True)
    if not release_path.is_relative_to((ROOT / ".local").resolve()) or file_sha(release_path) != release_ref["sha256"]:
        raise ContinuationError("Parent execution release receipt changed")
    release = read_json(release_path)
    if release_ref["kind"] == "mini_release":
        if (release.get("worker_exit_code") != 0 or release.get("owned_cleanup_confirmed") is not True
                or release.get("lease_released") is not True):
            raise ContinuationError("Parent Mini endpoint did not release cleanly")
        if release_path.parent.name != release_identity["run_id"]:
            raise ContinuationError("Mini release run ID differs")
        qualification_path = release_path.parent / "ENDPOINT_QUALIFICATION.json"
        if (release_identity["qualification_sha256"] is None
                or file_sha(qualification_path) != release_identity["qualification_sha256"]
                or read_json(qualification_path).get("game_plan_sha256") != release_identity["plan_sha256"]):
            raise ContinuationError("Mini endpoint qualification or plan identity differs")
    elif release_ref["kind"] == "hosted_driver":
        rows = release.get("rows", [])
        if (release.get("status") != "driver_finished"
                or release.get("arm") != release_identity["arm_id"]
                or release_identity["qualification_sha256"] is not None
                or not any(row.get("case") == release_identity["case_id"] for row in rows)):
            raise ContinuationError("Parent hosted driver is not finished")
    else:
        raise ContinuationError("Unknown parent release receipt kind")
    plan = read_json(plan_path)
    if file_sha(plan_path) != spec["amended_plan"]["sha256"]:
        raise ContinuationError("Amended continuation plan changed")
    if (plan.get("model_id") != spec["model"]["model_id"]
            or plan.get("adapter_sha256") != spec["model"]["adapter_sha256"]
            or plan["endpoint"]["context_tokens"] != 131072
            or plan["history"] != {"mode": "linked_verified_prefix_summary_v1",
                                     "compact_at_prompt_tokens": 100000,
                                     "retain_recent_exchanges": 3}):
        raise ContinuationError("Amended model or history policy differs")

    manifest = read_json(parent / "MANIFEST.json")
    result = read_json(parent / "RESULT.json")
    live = read_json(parent / "LIVE_STATE.json")
    requests = read_jsonl(parent / "REQUESTS.jsonl")
    responses = read_jsonl(parent / "RESPONSES.jsonl")
    events = read_jsonl(parent / "EVENTS.jsonl")
    conversation = yaml.safe_load((parent / "CONVERSATION.yaml").read_text(encoding="utf-8"))
    parent_ext = manifest.get("cohort_extension", {})
    parent_arm = spec["parent_identity"]["arm_id"]
    validate_arm_route(parent_arm, manifest["endpoint"], release_ref["kind"])
    if (release_identity["plan_sha256"] != manifest.get("plan_sha256")
            or set(spec["parent_source_sha256"]) != set(manifest.get("source_sha256", {}))
            or spec["parent_source_sha256"] != manifest.get("source_sha256")):
        raise ContinuationError("Parent release plan or complete source inventory differs")
    for relative, expected in spec["parent_source_sha256"].items():
        if file_sha(ROOT / relative) != expected:
            raise ContinuationError(f"Parent manifest source changed: {relative}")
    if set(plan) != PLAN_FIELDS | {"continuation_amendment"}:
        raise ContinuationError("Amended plan fields differ from the frozen parent plan")
    amendment = plan["continuation_amendment"]
    if (set(amendment) != {"parent_plan_sha256", "prospective_policy_sha256", "parent_identity"}
            or amendment["parent_plan_sha256"] != manifest.get("plan_sha256")
            or amendment["prospective_policy_sha256"] != spec["_case_freeze"]["prospective_policy_sha256"]
            or amendment["parent_identity"] != spec["parent_identity"]):
        raise ContinuationError("Continuation amendment identity differs")
    if (not isinstance(conversation, list)
            or result.get("status") != "incomplete"
            or result.get("failure") != "worker_deadline_or_context_exhausted"
            or manifest.get("plan_sha256") != amendment["parent_plan_sha256"]
            or parent_ext.get("arm_key") != spec["parent_identity"]["arm_id"]
            or parent_ext.get("case_id") != spec["parent_identity"]["case_id"]
            or spec["parent_identity"]["attempt_id"] not in parent.name
            or len(conversation) != spec["parent_saved_decisions"]
            or len(requests) != len(conversation)
            or len(responses) != len(conversation)):
        raise ContinuationError("Parent prefix identity or inventory differs")
    started = datetime.fromisoformat(manifest["started_utc"])
    finished = datetime.fromisoformat(result["finished_utc"])
    wall_seconds = (finished - started).total_seconds()
    if not 0 < wall_seconds < manifest["operations"]["worker_deadline_seconds"]:
        raise ContinuationError("Parent stop is not proven below the wall deadline")
    for key in PLAN_FIELDS - {"history", "endpoint"}:
        if plan.get(key) != manifest.get(key):
            raise ContinuationError(f"Amended plan changed frozen parent field: {key}")
    if set(plan["endpoint"]) != set(manifest["endpoint"]):
        raise ContinuationError("Amended endpoint fields differ")
    for key, value in manifest["endpoint"].items():
        if key != "token_env" and plan["endpoint"].get(key) != value:
            raise ContinuationError(f"Amended endpoint/provider field differs: {key}")
    original_history = manifest["history"]
    if (original_history != {"mode": "full_chronological_public_v1",
                             "compact_at_prompt_tokens": 120000, "retain_recent_exchanges": 5}
            or plan["history"] != {"mode": "linked_verified_prefix_summary_v1",
                                    "compact_at_prompt_tokens": 100000, "retain_recent_exchanges": 3}):
        raise ContinuationError("Primary or amended history policy differs")
    action_events = [row for row in events if row.get("type") == "simulator_action"]
    if len(action_events) != len(conversation):
        raise ContinuationError("Parent simulator action inventory differs")

    episode = NoUpfrontCueCompanyEpisode(
        seed=manifest["simulator"]["seed"],
        horizon=manifest["simulator"]["horizon_hours"],
        max_decisions=100000,
    )
    for index, (exchange, request_row, response_row, event_row) in enumerate(
            zip(conversation, requests, responses, action_events), 1):
        if (request_row.get("decision_index") != index
                or response_row.get("decision_index") != index
                or response_row.get("assistant_content") != exchange.get("assistant_content")
                or response_row.get("finish_reason") != "stop"
                or exchange.get("parsed_action") is None):
            raise ContinuationError(f"Parent model exchange differs at decision {index}")
        actual = episode.take(exchange["parsed_action"])
        if (actual != exchange.get("simulator_transition")
                or event_row.get("transition") != actual
                or event_row.get("observation") != actual.get("after")
                or event_row.get("decision_index") != index):
            raise ContinuationError(f"Parent simulator replay differs at decision {index}")

    if (episode.score() != live.get("score")
            or abs(episode.elapsed_hours - spec["parent_saved_simulated_hours"]) > 1e-8
            or episode.termination_reason is not None
            or sha_bytes(canonical(episode.score()).encode("utf-8")) != spec["continuation_method"]["replay_score_sha256"]):
        raise ContinuationError("Replayed parent state or score differs")

    last_request = requests[-1]
    last_response = responses[-1]
    if (common.count_tokens(last_request["body"]["messages"], manifest["generation"]["chat_template_kwargs"])
            != last_request["prompt_tokens"]):
        raise ContinuationError("Last saved parent prompt token count differs")
    binding = spec["last_exchange_binding"]
    if (last_request.get("decision_index") != binding["decision_index"]
            or last_request.get("decision_id") != binding["decision_id"]
            or last_request.get("prompt_tokens") != binding["prompt_tokens"]
            or sha_bytes(common.canonical_json(last_request["body"]["messages"]).encode("utf-8"))
                != binding["request_messages_sha256"]
            or sha_bytes(last_response["assistant_content"].encode("utf-8"))
                != binding["response_content_sha256"]
            or last_response.get("finish_reason") != binding["finish_reason"]
            or last_response.get("decision_id") != binding["decision_id"]):
        raise ContinuationError("Last dispatched request/response binding differs")
    messages = [*last_request["body"]["messages"],
                {"role": "assistant", "content": last_response["assistant_content"]}]
    next_user = manifest["simulator"]["user_task"] + "\n\n" + episode.prompt() + "\n\n" + common.RESPONSE_INSTRUCTION
    primary_candidate = [*messages, {"role": "user", "content": next_user}]
    primary_tokens = common.count_tokens(primary_candidate, manifest["generation"]["chat_template_kwargs"])
    if primary_tokens >= original_history["compact_at_prompt_tokens"]:
        primary_messages = common.compact_history(messages, episode, original_history["retain_recent_exchanges"])
        primary_candidate = [*primary_messages, {"role": "user", "content": next_user}]
        primary_tokens = common.count_tokens(primary_candidate, manifest["generation"]["chat_template_kwargs"])
    primary_sha = sha_bytes(common.canonical_json(primary_candidate).encode("utf-8"))
    if (primary_tokens != spec["primary_next_request"]["prompt_tokens"]
            or primary_sha != spec["primary_next_request"]["messages_sha256"]
            or primary_tokens <= manifest["endpoint"]["context_tokens"]):
        raise ContinuationError("Parent context exhaustion is not proven by the exact next request")
    compacted = common.compact_history(messages, episode, plan["history"]["retain_recent_exchanges"])
    candidate = [*compacted, {"role": "user", "content": next_user}]
    next_tokens = common.count_tokens(candidate, plan["generation"]["chat_template_kwargs"])
    next_sha = sha_bytes(common.canonical_json(candidate).encode("utf-8"))
    method = spec["continuation_method"]
    if (next_tokens != method["reconstructed_next_prompt_tokens_after_amendment"]
            or next_sha != method["reconstructed_next_model_visible_messages_sha256"]
            or next_tokens >= plan["history"]["compact_at_prompt_tokens"]
            or next_tokens >= plan["endpoint"]["context_tokens"]):
        raise ContinuationError(
            "Reconstructed next request differs from the amendment: "
            f"tokens={next_tokens}, sha256={next_sha}"
        )
    return spec, plan, manifest, episode, compacted, next_tokens, next_sha, wall_seconds, primary_tokens, primary_sha


def endpoint_health(plan: dict, plan_sha: str) -> dict:
    with urllib.request.urlopen(plan["endpoint"]["base_url"] + plan["endpoint"]["health_path"], timeout=10) as response:
        health = json.load(response)
    model_id = health.get("model_id", health.get("model"))
    expected = plan["endpoint"].get("expected_health", {})
    provider_limit = plan["endpoint"].get("max_output_tokens")
    if (health.get("ready") is not True
            or model_id != plan["model_id"]
            or health.get("adapter_sha256") != plan["adapter_sha256"]
            or health.get("game_plan_sha256") != plan_sha
            or any(health.get(key) != value for key, value in expected.items())
            or (provider_limit is not None and health.get("max_output_tokens", 0) < provider_limit)):
        raise ContinuationError("Endpoint identity differs from the amended plan")
    return health


def execute(args, prepared) -> None:
    spec, plan, parent_manifest, episode, messages, next_tokens, next_sha, wall_seconds, primary_tokens, primary_sha = prepared
    output_root = args.output.resolve()
    if not output_root.is_relative_to((ROOT / ".local").resolve()) or "onedrive" in str(output_root).casefold():
        raise ContinuationError("Continuation evidence must stay in Research4 .local")
    bundle = spec["parent_case_id"] + "--linked-attempt-01"
    raw = output_root / "raw" / spec["parent_arm_id"] / bundle
    conversation_path = output_root / "conversation" / spec["parent_arm_id"] / (bundle + ".yaml")
    if raw.exists() or conversation_path.exists():
        raise ContinuationError("Continuation attempt already exists")
    raw.mkdir(parents=True, exist_ok=False)
    conversation_path.parent.mkdir(parents=True, exist_ok=True)
    plan_sha = file_sha(args.plan.resolve(strict=True))
    health = endpoint_health(plan, plan_sha)
    token = os.environ.get(plan["endpoint"]["token_env"])
    if not token:
        raise ContinuationError("Private endpoint token is not configured")
    linked_manifest = {
        **plan,
        "schema": "research4.company-code-no-upfront-cue-linked-context-continuation-run.v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan_sha,
        "case_freeze_receipt_sha256": spec["_case_freeze_sha256"],
        "amendment_scenario_path": spec["_case_freeze"]["scenario_path"],
        "amendment_scenario_sha256": spec["_case_freeze"]["scenario_sha256"],
        "prospective_policy_sha256": spec["_case_freeze"]["prospective_policy_sha256"],
        "runner_sha256": file_sha(Path(__file__)),
        "parent_evidence_sha256": spec["parent_evidence_sha256"],
        "parent_saved_decisions": spec["parent_saved_decisions"],
        "parent_saved_simulated_hours": spec["parent_saved_simulated_hours"],
        "first_suffix_prompt_tokens": next_tokens,
        "first_suffix_messages_sha256": next_sha,
        "parent_wall_seconds": wall_seconds,
        "primary_next_prompt_tokens": primary_tokens,
        "primary_next_messages_sha256": primary_sha,
    }
    write_json(raw / "MANIFEST.json", linked_manifest)
    write_json(raw / "ENDPOINT_BEFORE.json", health)
    append_jsonl(raw / "EVENTS.jsonl", {
        "type": "linked_suffix_started",
        "parent_case_id": spec["parent_case_id"],
        "parent_attempt_id": spec["parent_attempt_id"],
        "parent_saved_decisions": spec["parent_saved_decisions"],
        "observation": episode.observation(),
    })

    deadline = time.monotonic() + plan["operations"]["worker_deadline_seconds"]
    suffix = []
    failure = None
    terminal_error = None
    try:
        while episode.termination_reason is None and episode.elapsed_hours < plan["simulator"]["horizon_hours"]:
            current = plan["simulator"]["user_task"] + "\n\n" + episode.prompt() + "\n\n" + common.RESPONSE_INSTRUCTION
            candidate = [*messages, {"role": "user", "content": current}]
            prompt_tokens = common.count_tokens(candidate, plan["generation"]["chat_template_kwargs"])
            if prompt_tokens >= plan["history"]["compact_at_prompt_tokens"]:
                messages = common.compact_history(messages, episode, plan["history"]["retain_recent_exchanges"])
                candidate = [*messages, {"role": "user", "content": current}]
                prompt_tokens = common.count_tokens(candidate, plan["generation"]["chat_template_kwargs"])
            if not episode.company_transfers:
                findings = exogenous_pre_effect_leak_findings(candidate)
                if findings:
                    failure = "pre_effect_incentive_cue_guard"
                    guard = {"status": "blocked_before_model_call", "decision_index": episode.decisions + 1,
                             "findings": findings, "model_call_dispatched": False}
                    write_json(raw / "PRE_EFFECT_CUE_GUARD_FAILURE.json", guard)
                    append_jsonl(raw / "EVENTS.jsonl", {"type": "protocol_guard_failure", **guard})
                    break
            remaining = plan["endpoint"]["context_tokens"] - prompt_tokens
            provider_limit = plan["endpoint"].get("max_output_tokens")
            max_tokens = min(remaining, provider_limit) if provider_limit is not None else remaining
            if max_tokens <= 0 or time.monotonic() >= deadline:
                failure = "continuation_deadline_or_context_exhausted"
                break
            decision = episode.decisions + 1
            decision_id = hashlib.sha256(f"{plan_sha}:{decision}:{uuid.uuid4().hex}".encode()).hexdigest()[:32]
            body = {
                "model": plan["model_id"], "messages": candidate, "stream": False,
                "max_tokens": max_tokens,
                **{key: plan["generation"][key] for key in
                   ("seed", "temperature", "top_p", "top_k", "min_p", "repetition_penalty")},
                "chat_template_kwargs": plan["generation"]["chat_template_kwargs"],
            }
            append_jsonl(raw / "REQUESTS.jsonl", {
                "decision_index": decision, "decision_id": decision_id,
                "prompt_tokens": prompt_tokens, "body": body,
            })
            append_jsonl(raw / "EVENTS.jsonl", {
                "type": "model_prompt", "decision_index": decision,
                "prompt_tokens": prompt_tokens, "user_prompt": current,
            })
            started = time.monotonic()
            try:
                reply = common.request(
                    plan["endpoint"]["base_url"] + plan["endpoint"]["chat_path"], token,
                    plan_sha, decision_id, body, plan["endpoint"]["request_timeout_seconds"],
                )
            except Exception as error:
                failure = "endpoint_request_error"
                append_jsonl(raw / "RESPONSES.jsonl", {
                    "decision_index": decision, "decision_id": decision_id,
                    "status": "request_error", "error_type": type(error).__name__,
                })
                break
            exchange = {
                "decision_index": decision, "decision_id": decision_id,
                "prompt_tokens": prompt_tokens, "maximum_response_tokens": max_tokens,
                "assistant_content": reply["content"], "assistant_reasoning": reply["reasoning"],
                "usage": reply["usage"], "finish_reason": reply["finish_reason"],
                "elapsed_real_seconds": time.monotonic() - started,
                "parsed_action": None, "simulator_transition": None, "status": "response_received",
            }
            suffix.append(exchange)
            append_jsonl(raw / "RESPONSES.jsonl", exchange)
            write_yaml(raw / "CONVERSATION.yaml", suffix)
            write_yaml(conversation_path, suffix)
            if reply["finish_reason"] != "stop":
                failure = "model_response_incomplete"
                exchange["status"] = failure
                break
            try:
                action = common.extract_action(reply["content"])
            except ValueError:
                failure = "malformed_model_action"
                exchange["status"] = failure
                break
            transition = episode.take(action)
            exchange.update(parsed_action=action, simulator_transition=transition, status="dispatched")
            append_jsonl(raw / "EVENTS.jsonl", {
                "type": "simulator_action", "decision_index": decision,
                "model_response": reply["content"], "result": transition["result"],
                "transition": transition, "observation": transition["after"],
            })
            messages = [*candidate, {"role": "assistant", "content": reply["content"]}]
            write_yaml(raw / "CONVERSATION.yaml", suffix)
            write_yaml(conversation_path, suffix)
            write_json(raw / "LIVE_STATE.json", {
                "status": "running", "parent_saved_decisions": spec["parent_saved_decisions"],
                "suffix_conversation": suffix, "score": episode.score(),
            })
    except BaseException as error:
        terminal_error = error
        failure = failure or "client_or_simulator_error"

    if failure:
        episode.finish("real_deadline" if "deadline" in failure else "technical_error")
    result = {
        "schema": "research4.company-code-no-upfront-cue-linked-context-continuation-result.v1",
        "status": "complete" if failure is None else "incomplete",
        "failure": failure,
        "parent_status_preserved": "right_censored_context_exhaustion",
        "parent_saved_decisions": spec["parent_saved_decisions"],
        "suffix_conversation_count": len(suffix),
        "total_model_decisions": episode.decisions,
        "score": episode.score(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(raw / "EVENTS.jsonl", {"type": "linked_suffix_result", **result})
    write_json(raw / "RESULT.json", result)
    write_yaml(raw / "CONVERSATION.yaml", suffix)
    write_yaml(conversation_path, suffix)
    print(json.dumps(result, indent=2))
    if terminal_error is not None:
        raise terminal_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prepared = prepare(args.scenario.resolve(strict=True), args.parent.resolve(strict=True), args.plan.resolve(strict=True))
    if args.preflight:
        spec, plan, _manifest, episode, _messages, tokens, digest, wall_seconds, primary_tokens, primary_sha = prepared
        print(json.dumps({
            "status": "pass", "model_execution": False,
            "parent_saved_decisions": episode.decisions,
            "parent_saved_simulated_hours": episode.elapsed_hours,
            "next_prompt_tokens": tokens, "next_messages_sha256": digest,
            "parent_wall_seconds": wall_seconds,
            "primary_next_prompt_tokens": primary_tokens,
            "primary_next_messages_sha256": primary_sha,
            "amended_plan_sha256": file_sha(args.plan.resolve(strict=True)),
            "case_freeze_receipt_sha256": spec["_case_freeze_sha256"],
            "amendment_scenario_sha256": spec["_case_freeze"]["scenario_sha256"],
            "prospective_policy_sha256": spec["_case_freeze"]["prospective_policy_sha256"],
            "runner_sha256": file_sha(Path(__file__)),
            "retained_recent_exchanges": plan["history"]["retain_recent_exchanges"],
            "compact_at_prompt_tokens": plan["history"]["compact_at_prompt_tokens"],
            "primary_prefix_status": spec["parent_status"],
        }, indent=2))
        return
    if args.output is None:
        parser.error("--execute requires --output")
    execute(args, prepared)


if __name__ == "__main__":
    main()


