"""One DeepSeek company-world episode through the existing Inspect connector.

No provider calls occur without --execute. Reuses the common task, streaming,
message construction and provider configuration; no Qwen service is contacted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .company_world import CompanyWorldEpisode, SCORE_VERSION, VERSION
from .continuity_arena import (ARENA_RESPONSE_INSTRUCTION, FORMAT_SYSTEM_INSTRUCTION,
                               MODEL_SPECS, build_model, evaluate_parallel, logs_complete)
from .hosted_inspect_runner import make_task, sample_metadata

REPOSITORY = Path(__file__).resolve().parents[1]
MODEL_ID = "deepseek"
MODEL_NAME = MODEL_SPECS[MODEL_ID]["model"]
MAX_OUTPUT_TOKENS = 4096
OUTER_SECONDS = 45 * 60
MAX_DECISIONS = 256
HORIZON_HOURS = 24
SOURCES = (
    "src/company_world_evaluation.py", "src/company_world.py", "src/company_access_cases.py",
    "src/company_economy.py", "src/continuity_arena.py", "src/hosted_inspect_runner.py",
    "src/infrastructure_game.py", "src/security_game.py", "src/continuity_game.py",
    "src/security_challenges.py", "scenarios/company_world_v1/world.yaml",
    "scenarios/company_world_v1/access_cases.yaml", "config/company_economy_v1.json",
    "config/infrastructure_economy_v1.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def source_matches(plan: dict[str, Any]) -> bool:
    try:
        return plan["source_sha256"] == {name: digest(REPOSITORY / name) for name in SOURCES}
    except OSError:
        return False


def episode(spec: dict[str, Any]) -> CompanyWorldEpisode:
    return CompanyWorldEpisode(seed=spec["seed"], horizon=HORIZON_HOURS,
                               max_decisions=MAX_DECISIONS)


def prospective_plan(seed: int, bundle: str = "unsubmitted") -> dict[str, Any]:
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be a non-negative 31-bit integer")
    world = episode({"seed": seed})
    spec = sample_metadata({"scenario": f"seed:{seed}", "seed": seed,
        "case_id": f"company_world_v1/deepseek_icl/{seed}", "arm_id": "deepseek_icl",
        "bundle": bundle, "model_id": MODEL_ID, "model": MODEL_NAME,
        "label": "DeepSeek · security instruction", "rules_version": VERSION,
        "prompt_condition": "security_maximization",
        "common_system_instruction": FORMAT_SYSTEM_INSTRUCTION})
    return {"version": "company_world_deepseek_evaluation_v1", "rules_version": VERSION,
        "score_version": SCORE_VERSION, "models": [MODEL_ID], "samples": [spec],
        "seed": seed, "world_plan_sha256": world.plan_sha256,
        "source_sha256": {name: digest(REPOSITORY / name) for name in SOURCES},
        "economy_sha256": world.economy_sha256,
        "checkpoint": "not_applicable_hosted_provider", "adapter": "none",
        "horizon_hours": HORIZON_HOURS, "max_decisions": MAX_DECISIONS,
        "max_output_tokens": MAX_OUTPUT_TOKENS, "outer_seconds": OUTER_SECONDS,
        "thinking_enabled": True, "reasoning_effort": "low", "provider_retries": 0,
        "request_timeout_seconds": 120, "fresh_context_each_turn": True,
        "prompt_protocol": "complete_company_world_state_and_tool_feedback_v1",
        "initial_user_payload_sha256": hashlib.sha256(world.prompt().encode()).hexdigest(),
        "format_system_instruction": FORMAT_SYSTEM_INSTRUCTION,
        "response_instruction": ARENA_RESPONSE_INSTRUCTION,
        "scope": "Single DeepSeek hosted evaluation of inert company state; no Qwen, training or external company actions."}


def build_task(plan: dict[str, Any], destination: Path, timeout_seconds: float,
               on_event: Any) -> Any:
    """The shared Inspect task owns all generation and message plumbing."""
    from inspect_ai import Task

    if plan["models"] != [MODEL_ID] or len(plan["samples"]) != 1:
        raise ValueError("This runner admits only one DeepSeek episode")
    spec = plan["samples"][0]
    model = build_model(MODEL_NAME, 1, MAX_OUTPUT_TOKENS)
    model.config.max_retries = 0
    model.config.timeout = max(1, int(min(120, timeout_seconds)))
    base = make_task(None, [], MAX_DECISIONS, timeout_seconds,
        stream_path=destination / "STREAM.jsonl", sample_specs=[spec], episode_factory=episode,
        on_event=on_event, fresh_context_each_turn=True,
        common_system_instruction=FORMAT_SYSTEM_INSTRUCTION,
        response_instruction=ARENA_RESPONSE_INSTRUCTION)
    return Task(dataset=base.dataset, solver=base.solver, scorer=base.scorer,
                model=model, name="company_world_deepseek")


def replay(records: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    """Replay received native actions; an absent terminal receipt stays missing."""
    selected = [row for row in records if row.get("type") in {"simulator_action", "episode_result"}]
    if any(row.get("case_id") != spec["case_id"] for row in selected):
        raise ValueError("Out-of-plan case in event evidence")
    world = episode(spec)
    actions = [row for row in selected if row["type"] == "simulator_action"]
    for row in actions:
        actual = world.take(row["model_response"])
        expected = {key: value for key, value in row["transition"].items() if key != "model_response"}
        if actual != expected:
            raise ValueError(f"Transition replay differs at decision {actual['decision_index']}")
    results = [row for row in selected if row["type"] == "episode_result"]
    if len(results) > 1:
        raise ValueError("Duplicate episode result")
    status, external_stop = "missing_result", "worker_or_export_incomplete"
    if results:
        result = results[0]
        external_stop = result.get("external_stop")
        if external_stop:
            world.finish("real_deadline" if external_stop == "time_limit" else "technical_error")
        if world.score() != result["scenario_score"]:
            raise ValueError("Terminal score replay differs")
        status = "complete" if external_stop is None and world.score()["score_status"] == "final" else "incomplete"
    return {"case_id": spec["case_id"], "status": status,
            "actions_received": len(actions), "replay_passed": True,
            "external_stop": external_stop, "score": world.score(),
            "replay_scope": "separate client replay of shared simulator; not an independently implemented scorer"}


def execute(output: Path, seed: int = 7, credentials: Path | None = None) -> int:
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc)
    destination = output.expanduser().resolve()
    ignored_root = (REPOSITORY / ".local").resolve()
    if ignored_root not in destination.parents:
        raise ValueError("Evidence must be beneath this repository's ignored .local directory")
    destination.mkdir(parents=True, exist_ok=True)
    plan = prospective_plan(seed, destination.name)
    plan.update(started_utc=started_utc.isoformat(),
                deadline_utc=(started_utc + timedelta(seconds=OUTER_SECONDS)).isoformat())
    # The UI may create the directory/console first. Exclusive manifest creation
    # claims the bundle once without allowing a second inference worker to reuse it.
    with (destination / "MANIFEST.json").open("x", encoding="utf-8") as manifest:
        manifest.write(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    records: list[dict[str, Any]] = []
    lock = threading.Lock()
    failure = None
    result = None

    def event(record: dict[str, Any]) -> None:
        stamped = {**copy.deepcopy(record), "bundle": destination.name}
        with lock:
            with (destination / "EVENTS.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(stamped, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            if stamped["type"] in {"simulator_action", "episode_result"}:
                records.append(stamped)
        # Other event types are already emitted by the shared Inspect connector.
        if stamped["type"] in {"episode_started", "model_prompt"}:
            print(json.dumps(stamped, ensure_ascii=False), flush=True)

    try:
        if credentials is not None:
            from dotenv import load_dotenv
            load_dotenv(credentials.expanduser().resolve(), override=False)
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise ValueError("Existing DEEPSEEK_API_KEY required")
        remaining = OUTER_SECONDS - (time.monotonic() - started) - 30
        if remaining <= 0:
            raise TimeoutError("Preparation exhausted the bounded run")
        task = build_task(plan, destination, remaining, event)
        if not source_matches(plan):
            raise ValueError("Source changed after manifest binding")
        logs = evaluate_parallel([task], destination / "inspect", MAX_OUTPUT_TOKENS)
        if not logs_complete(logs):
            failure = {"type": "provider_or_harness_incomplete", "automatic_retry": False}
    except Exception as error:
        # Never serialize provider exception text or environment credentials.
        failure = {"type": type(error).__name__, "automatic_retry": False}
    finally:
        try:
            result = replay(records, plan["samples"][0])
            write_json(destination / "REPLAYED_RESULT.json", result)
        except Exception as error:
            failure = {"type": type(error).__name__, "stage": "client_replay", "automatic_retry": False}
        unchanged = source_matches(plan)
        if not unchanged:
            failure = {"type": "SourceDrift", "automatic_retry": False}
        write_json(destination / "CLOSEOUT.json", {"failure": failure, "result": result,
            "elapsed_seconds": time.monotonic() - started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "source_unchanged": unchanged,
            "resources": "hosted API client only; no local model/GPU/compute allocation or Qwen service query"})
    print(json.dumps({"type": "company_evaluation_closeout", "result": result, "failure": failure}), flush=True)
    return 0 if not failure and result and result["status"] == "complete" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".local/company_world_eval") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(prospective_plan(args.seed), indent=2))
        return 0
    return execute(args.output, args.seed, args.credentials)


if __name__ == "__main__":
    raise SystemExit(main())
