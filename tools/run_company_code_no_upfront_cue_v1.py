"""Run one frozen no-upfront-incentive-cue company-code episode."""
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import run_rank16_infrastructure_game as common
from src.company_code_no_upfront_cue_v1 import NoUpfrontCueCompanyEpisode, VERSION


def smoke() -> dict:
    episode = NoUpfrontCueCompanyEpisode(seed=1454041566, horizon=1, max_decisions=3)
    payload = json.loads(episode.prompt())
    tools = {row["name"] for row in payload["tools"]}
    if not {"inspect_company_source", "request_company_action"} <= tools:
        raise RuntimeError("Neutral company source/action tools are absent")
    if "request_company_payment" in tools or "4000" in episode.prompt():
        raise RuntimeError("Upfront company-payment cue remains")
    return {"status": "passed", "mode": "model_free_smoke", "visible_tool_count": len(tools),
            "payment_tool_visible": False, "external_effects": 0, "model_calls": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2)); return
    if args.plan is None or args.output is None:
        parser.error("--execute requires --plan and --output")
    plan_path = args.plan.resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("schema") != "research4.company-code-no-upfront-cue-game.v1"
            or plan.get("cohort_extension", {}).get("study_id") != VERSION):
        raise ValueError("Exact no-upfront-cue company plan required")
    output = args.output.resolve()
    if (ROOT / ".local").resolve() not in output.parents or "onedrive" in str(output).casefold():
        raise ValueError("Evidence must remain under the non-OneDrive Research4 .local directory")
    output.mkdir(parents=True, exist_ok=False)
    plan_sha = common.sha(plan_path)
    source_paths = (
        Path(__file__), ROOT / "src/company_code_no_upfront_cue_v1.py",
        ROOT / "src/company_code_access_v1.py", ROOT / "src/company_world_v2.py",
        ROOT / "src/company_world.py", ROOT / "src/company_access_cases.py",
        ROOT / "scenarios/company_world_v2/world.yaml",
    )
    source_hashes = {path.relative_to(ROOT).as_posix(): common.sha(path) for path in source_paths}
    bundle = output.name; arm_id = plan["arm_id"]; seed = plan["simulator"]["seed"]
    case_id = f"{VERSION}/{arm_id}/{seed}"
    sample = {"case_id": case_id, "arm_id": arm_id, "label": plan["label"],
              "model_id": arm_id, "model": plan["model_id"], "prompt_condition": "no_upfront_incentive_cue",
              "seed": seed, "rules_version": VERSION}
    manifest = {**plan, "version": bundle, "plan": {"arms": [arm_id], "game_seeds": [seed]},
                "samples": [sample], "plan_sha256": plan_sha,
                "source_sha256": source_hashes, "started_utc": datetime.now(timezone.utc).isoformat()}
    common.write_json(output / "MANIFEST.json", manifest)
    token = os.environ.get(plan["endpoint"]["token_env"])
    if not token:
        raise ValueError("Private endpoint token is not configured")
    with urllib.request.urlopen(plan["endpoint"]["base_url"] + plan["endpoint"]["health_path"], timeout=10) as response:
        health = json.load(response)
    expected = plan["endpoint"].get("expected_health", {})
    endpoint_model_id = health.get("model_id", health.get("model"))
    if (type(expected) is not dict or any(health.get(key) != value for key, value in expected.items())
            or health.get("ready") is not True or endpoint_model_id != plan["model_id"]
            or health.get("adapter_sha256") != plan["adapter_sha256"]):
        raise ValueError("Endpoint identity differs from frozen plan")
    common.write_json(output / "ENDPOINT_BEFORE.json", health)
    episode = NoUpfrontCueCompanyEpisode(seed=seed, horizon=plan["simulator"]["horizon_hours"], max_decisions=100000)
    messages = [{"role": "system", "content": common.SYSTEM}]
    deadline = time.monotonic() + plan["operations"]["worker_deadline_seconds"]
    conversation, failure, terminal_error = [], None, None
    common.append_jsonl(output / "EVENTS.jsonl", {"type": "episode_started", "bundle": bundle,
        "case_id": case_id, "observation": episode.observation()})
    try:
        while episode.termination_reason is None and episode.elapsed_hours < plan["simulator"]["horizon_hours"]:
            current = plan["simulator"]["user_task"] + "\n\n" + episode.prompt() + "\n\n" + common.RESPONSE_INSTRUCTION
            candidate = [*messages, {"role": "user", "content": current}]
            prompt_tokens = common.count_tokens(candidate, plan["generation"]["chat_template_kwargs"])
            if prompt_tokens >= plan["history"]["compact_at_prompt_tokens"]:
                messages = common.compact_history(messages, episode, plan["history"]["retain_recent_exchanges"])
                candidate = [*messages, {"role": "user", "content": current}]
                prompt_tokens = common.count_tokens(candidate, plan["generation"]["chat_template_kwargs"])
            remaining = plan["endpoint"]["context_tokens"] - prompt_tokens
            provider_limit = plan["endpoint"].get("max_output_tokens")
            max_tokens = min(remaining, provider_limit) if provider_limit is not None else remaining
            if max_tokens <= 0 or time.monotonic() >= deadline:
                failure = "worker_deadline_or_context_exhausted"; break
            decision = episode.decisions + 1
            decision_id = hashlib.sha256(f"{plan_sha}:{decision}:{uuid.uuid4().hex}".encode()).hexdigest()[:32]
            body = {"model": plan["model_id"], "messages": candidate, "stream": False, "max_tokens": max_tokens,
                    **{key: plan["generation"][key] for key in
                       ("seed", "temperature", "top_p", "top_k", "min_p", "repetition_penalty")},
                    "chat_template_kwargs": plan["generation"]["chat_template_kwargs"]}
            common.append_jsonl(output / "REQUESTS.jsonl", {"decision_index": decision,
                "decision_id": decision_id, "prompt_tokens": prompt_tokens, "body": body})
            common.append_jsonl(output / "EVENTS.jsonl", {"type": "model_prompt", "bundle": bundle,
                "case_id": case_id, "decision_index": decision, "system_instruction": common.SYSTEM,
                "user_prompt": current, "prompt_tokens": prompt_tokens})
            started = time.monotonic()
            try:
                reply = common.request(plan["endpoint"]["base_url"] + plan["endpoint"]["chat_path"], token,
                    plan_sha, decision_id, body, plan["endpoint"]["request_timeout_seconds"])
            except Exception as error:
                failure = "endpoint_request_error"
                common.append_jsonl(output / "RESPONSES.jsonl", {"decision_index": decision,
                    "decision_id": decision_id, "status": "request_error", "error_type": type(error).__name__})
                break
            exchange = {"decision_index": decision, "decision_id": decision_id, "prompt_tokens": prompt_tokens,
                "maximum_response_tokens": max_tokens, "assistant_content": reply["content"],
                "assistant_reasoning": reply["reasoning"], "usage": reply["usage"],
                "finish_reason": reply["finish_reason"], "elapsed_real_seconds": time.monotonic() - started,
                "parsed_action": None, "simulator_transition": None, "status": "response_received"}
            conversation.append(exchange); common.append_jsonl(output / "RESPONSES.jsonl", exchange)
            common.write_yaml(output / "CONVERSATION.yaml", conversation)
            if reply["finish_reason"] != "stop":
                failure = "model_response_incomplete"; exchange["status"] = failure; break
            try:
                action = common.extract_action(reply["content"])
            except ValueError:
                failure = "malformed_model_action"; exchange["status"] = failure; break
            transition = episode.take(action)
            exchange.update(parsed_action=action, simulator_transition=transition, status="dispatched")
            common.append_jsonl(output / "EVENTS.jsonl", {"type": "simulator_action", "bundle": bundle,
                "case_id": case_id, "decision_index": decision, "model_response": reply["content"],
                "result": transition["result"], "transition": transition, "observation": transition["after"]})
            messages = [*candidate, {"role": "assistant", "content": reply["content"]}]
            common.write_yaml(output / "CONVERSATION.yaml", conversation)
            common.write_json(output / "LIVE_STATE.json", {"status": "running", "conversation": conversation,
                                                               "score": episode.score()})
    except BaseException as error:
        terminal_error = error; failure = failure or "client_or_simulator_error"
    if failure:
        episode.finish("real_deadline" if "deadline" in failure else "technical_error")
    result = {"status": "complete" if failure is None else "incomplete", "failure": failure,
              "score": episode.score(), "conversation_count": len(conversation),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    common.append_jsonl(output / "EVENTS.jsonl", {"type": "episode_result", "bundle": bundle,
        "case_id": case_id, "scenario_score": result["score"], "external_stop": failure, **result})
    common.write_json(output / "RESULT.json", result); common.write_yaml(output / "CONVERSATION.yaml", conversation)
    print(json.dumps(result, indent=2))
    if terminal_error is not None:
        raise terminal_error


if __name__ == "__main__":
    main()
