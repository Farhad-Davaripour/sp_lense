"""Selected company-world models through the existing Inspect connector.

No provider calls occur without --execute. Reuses the common task, streaming,
message construction and provider configuration. DeepSeek alone is the default;
the local Qwen identity gate runs only when a Qwen condition is selected.
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
from .company_world_v2 import CompanyWorldEpisodeV2, VERSION as V2_VERSION
from .company_models import (MODEL_SPECS, QWEN_BASE_URL, assert_local_binding,
                             build_company_model, selected_models, required_api_key)
from .continuity_arena import (ARENA_RESPONSE_INSTRUCTION, FORMAT_SYSTEM_INSTRUCTION,
                               evaluate_parallel, logs_complete)
from .hosted_inspect_runner import make_task, sample_metadata

REPOSITORY = Path(__file__).resolve().parents[1]
MODEL_ID = "deepseek"
MODEL_NAME = MODEL_SPECS[MODEL_ID]["model"]
MAX_OUTPUT_TOKENS = 8192
REQUEST_TIMEOUT_SECONDS = 180
COMPANY_SYSTEM_INSTRUCTION = FORMAT_SYSTEM_INSTRUCTION + "\nKeep reasoning brief and focused on choosing the next action."
OUTER_SECONDS = 45 * 60
MAX_DECISIONS = 256
HORIZON_HOURS = 24
SOURCES = (
    "src/company_world_evaluation.py", "src/company_models.py", "src/company_world.py", "src/company_access_cases.py",
    "src/company_economy.py", "src/continuity_arena.py", "src/hosted_inspect_runner.py",
    "src/infrastructure_game.py", "src/security_game.py", "src/continuity_game.py",
    "src/security_challenges.py", "scenarios/company_world_v1/world.yaml",
    "scenarios/company_world_v1/access_cases.yaml", "config/company_economy_v1.json",
    "config/infrastructure_economy_v1.json",
    "src/company_world_v2.py", "src/company_economy_v2.py", "src/qwen_cloud.py",
    "config/company_economy_v2.json", "scenarios/company_world_v2/world.yaml",
    "src/company_world_comparison.py", "src/company_preservation_comparison.py", "src/qwen_spending.py",
    "config/company_preservation_comparison_v1.yaml",
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
    world_version = spec.get("rules_version", V2_VERSION)
    if world_version == V2_VERSION:
        return CompanyWorldEpisodeV2(seed=spec["seed"], horizon=HORIZON_HOURS,
                                    max_decisions=MAX_DECISIONS,
                                    incentive_profile=spec.get("incentive_profile", "high_reward"))
    if world_version == VERSION:
        return CompanyWorldEpisode(seed=spec["seed"], horizon=HORIZON_HOURS,
                                   max_decisions=MAX_DECISIONS)
    raise ValueError("Unknown company-world version")


def prospective_plan(seed: int, bundle: str = "unsubmitted",
                     models: list[str] | None = None, world_version: str = V2_VERSION,
                     incentive_profile: str = "high_reward") -> dict[str, Any]:
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be a non-negative 31-bit integer")
    if world_version not in {VERSION, V2_VERSION}:
        raise ValueError("Unknown company-world version")
    world = episode({"seed": seed, "rules_version": world_version, "incentive_profile": incentive_profile})
    selected = selected_models(models)
    specs = []
    for model_id in selected:
        config = MODEL_SPECS[model_id]
        specs.append(sample_metadata({"scenario": f"seed:{seed}", "seed": seed,
            "case_id": f"{world_version}/{config['arm_id']}/{seed}" if world_version == VERSION else f"{world_version}/{incentive_profile}/{config['arm_id']}/{seed}", "arm_id": config["arm_id"],
            "bundle": bundle, "model_id": model_id, "model": config["model"],
            "label": config["label"], "rules_version": world_version,
            "incentive_profile": incentive_profile if world_version == V2_VERSION else None,
            "prompt_condition": config["prompt_condition"],
            "common_system_instruction": COMPANY_SYSTEM_INSTRUCTION}))
    settings = {key: {**copy.deepcopy(MODEL_SPECS[key]), "thinking_enabled": True,
                     "reasoning_effort": "low", "provider_retries": 0} for key in selected}
    hosted_only = not any(MODEL_SPECS[key].get("local_qwen") for key in selected)
    return {"version": "company_world_v2_selected_models_v3_last8" if world_version == V2_VERSION else "company_world_deepseek_evaluation_v2" if selected == [MODEL_ID] else "company_world_selected_models_v1", "rules_version": world_version,
        "score_version": SCORE_VERSION, "models": selected, "samples": specs,
        "generation_settings": settings,
        "seed": seed, "world_plan_sha256": world.plan_sha256,
        "source_sha256": {name: digest(REPOSITORY / name) for name in SOURCES},
        "economy_sha256": world.economy_sha256,
        "incentive_profile": incentive_profile if world_version == V2_VERSION else None,
        "checkpoint": "not_applicable_hosted_provider" if hosted_only else "reviewed_local_binding_required",
        "adapter": "per_selected_service_model" if any(MODEL_SPECS[key].get("adapter_sha256") for key in selected) else "none",
        "horizon_hours": HORIZON_HOURS, "max_decisions": MAX_DECISIONS,
        "max_output_tokens": {key: MODEL_SPECS[key]["max_output_tokens"] for key in selected} if len(selected) > 1 else MODEL_SPECS[selected[0]]["max_output_tokens"],
        "outer_seconds": OUTER_SECONDS,
        "thinking_enabled": True, "reasoning_effort": "low", "provider_retries": 0,
        "request_timeout_seconds": {key: MODEL_SPECS[key].get("request_timeout_seconds", REQUEST_TIMEOUT_SECONDS)
                                    for key in selected} if len(selected) > 1 else MODEL_SPECS[selected[0]].get("request_timeout_seconds", REQUEST_TIMEOUT_SECONDS),
        "fresh_context_each_turn": True,
        "prompt_protocol": "company_world_compact_semantic_history_v3" if world_version == V2_VERSION else "company_world_feedback_brief_reasoning_nested_envelope_v2",
        "initial_user_payload_sha256": hashlib.sha256(world.prompt().encode()).hexdigest(),
        "format_system_instruction": COMPANY_SYSTEM_INSTRUCTION,
        "response_instruction": ARENA_RESPONSE_INSTRUCTION,
        "scope": "Selected model conditions operating isolated copies of one inert company-world seed; no training or external company actions.",
        "comparison_limits": ["A single seed is an engineering observation, not a model ranking or population result.",
                              "All current conditions request 8192 output tokens with thinking enabled at nominal low effort; provider tokenizers and implementation differ.",
                              "Neutral and security-instruction conditions have different prompts; adapter and prompt effects must be distinguished.",
                              "Local Qwen base and last8 adapter requests share one serialized resident service with the same response allowance; hosted tasks can overlap.",
                              "Compare adapters only against the same verified local base, quantization, inference settings and prompt condition. Hosted Max, Terra and DeepSeek are separate deployed-system comparisons; hosted precision is unverified."]}


def build_task(plan: dict[str, Any], destination: Path, timeout_seconds: float,
               on_event: Any, sample_index: int = 0) -> Any:
    """The shared Inspect task owns all generation and message plumbing."""
    from inspect_ai import Task

    selected = selected_models(plan["models"])
    if len(plan["samples"]) != len(selected):
        raise ValueError("Selected models and sample bindings differ")
    spec = plan["samples"][sample_index]
    model_id = selected[sample_index]
    if spec["model_id"] != model_id or spec["model"] != MODEL_SPECS[model_id]["model"]:
        raise ValueError("Sample model identity differs")
    model = build_company_model(model_id, max_connections=MODEL_SPECS[model_id].get("max_connections", 1))
    model.config.max_tokens = MODEL_SPECS[model_id]["max_output_tokens"]
    model.config.max_retries = 0
    model.config.timeout = max(1, int(min(MODEL_SPECS[model_id].get("request_timeout_seconds", REQUEST_TIMEOUT_SECONDS), timeout_seconds)))
    base = make_task(None, [], MAX_DECISIONS, timeout_seconds,
        stream_path=destination / ("STREAM.jsonl" if len(selected) == 1 else f"STREAM_{model_id}.jsonl"), sample_specs=[spec], episode_factory=episode,
        on_event=on_event, fresh_context_each_turn=True,
        common_system_instruction=COMPANY_SYSTEM_INSTRUCTION,
        response_instruction=ARENA_RESPONSE_INSTRUCTION)
    return Task(dataset=base.dataset, solver=base.solver, scorer=base.scorer,
                model=model, name=f"company_world_{model_id}")


def build_tasks(plan: dict[str, Any], destination: Path, timeout_seconds: float,
                on_event: Any) -> list[Any]:
    return [build_task(plan, destination, timeout_seconds, on_event, index)
            for index in range(len(plan["samples"]))]


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


def replay_all(records: list[dict[str, Any]], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {spec["case_id"] for spec in specs}
    if any(row.get("case_id") not in expected for row in records
           if row.get("type") in {"simulator_action", "episode_result"}):
        raise ValueError("Out-of-plan case in event evidence")
    return [replay([row for row in records if row.get("case_id") == spec["case_id"]], spec)
            for spec in specs]


def execute(output: Path, seed: int = 7, credentials: Path | None = None,
            models: list[str] | None = None, world_version: str = V2_VERSION,
            incentive_profile: str = "high_reward") -> int:
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc)
    destination = output.expanduser().resolve()
    ignored_root = (REPOSITORY / ".local").resolve()
    if ignored_root not in destination.parents:
        raise ValueError("Evidence must be beneath this repository's ignored .local directory")
    destination.mkdir(parents=True, exist_ok=True)
    plan = prospective_plan(seed, destination.name, models, world_version, incentive_profile)
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
    results = []

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
        local_selected = any(MODEL_SPECS[key].get("local_qwen") for key in plan["models"])
        for key in plan["models"]:
            config = MODEL_SPECS[key]
            if not config.get("local_qwen"):
                variable = required_api_key(config)
                if not os.environ.get(variable):
                    raise ValueError(f"Existing {variable} required")
        if local_selected:
            os.environ.setdefault("RESEARCH4_QWEN_API_KEY", "local-unused")
            groups = sorted({MODEL_SPECS[key]["local_binding_group"] for key in plan["models"]
                             if MODEL_SPECS[key].get("local_qwen")})
            write_json(destination / "LOCAL_BINDING_READBACK.json",
                       {group: assert_local_binding(group) for group in groups})
        remaining = OUTER_SECONDS - (time.monotonic() - started) - 30
        if remaining <= 0:
            raise TimeoutError("Preparation exhausted the bounded run")
        tasks = build_tasks(plan, destination, remaining, event)
        if not source_matches(plan):
            raise ValueError("Source changed after manifest binding")
        # None does not override model-specific GenerateConfig values in Inspect;
        # this preserves each model's explicitly bound response allowance.
        shared_limit = MODEL_SPECS[plan["models"][0]]["max_output_tokens"] if len(tasks) == 1 else None
        logs = evaluate_parallel(tasks, destination / "inspect", shared_limit)
        if not logs_complete(logs):
            failure = {"type": "provider_or_harness_incomplete", "automatic_retry": False}
    except Exception as error:
        # Never serialize provider exception text or environment credentials.
        failure = {"type": type(error).__name__, "automatic_retry": False}
    finally:
        try:
            results = replay_all(records, plan["samples"])
            result = results[0] if len(results) == 1 else None
            write_json(destination / ("REPLAYED_RESULT.json" if result is not None else "REPLAYED_RESULTS.json"), result if result is not None else results)
        except Exception as error:
            failure = {"type": type(error).__name__, "stage": "client_replay", "automatic_retry": False}
        unchanged = source_matches(plan)
        if not unchanged:
            failure = {"type": "SourceDrift", "automatic_retry": False}
        write_json(destination / "CLOSEOUT.json", {"failure": failure, "result": result, "results": results,
            "elapsed_seconds": time.monotonic() - started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "source_unchanged": unchanged,
            "resources": "Hosted clients and, only when selected, an existing verified local Qwen endpoint. This runner starts no training or model/compute allocation and does not own the resident service lifetime."})
    print(json.dumps({"type": "company_evaluation_closeout", "result": result, "results": results, "failure": failure}), flush=True)
    return 0 if not failure and results and all(row["status"] == "complete" for row in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--models", default=MODEL_ID, help="Comma-separated configured model IDs; default deepseek")
    parser.add_argument("--world-version", choices=[VERSION, V2_VERSION], default=V2_VERSION)
    parser.add_argument("--incentive-profile", choices=["high_reward", "reference_reward"], default="high_reward")
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".local/company_world_eval") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    models = [key.strip() for key in args.models.split(",")]
    if not args.execute:
        print(json.dumps(prospective_plan(args.seed, models=models, world_version=args.world_version,
                                          incentive_profile=args.incentive_profile), indent=2))
        return 0
    return execute(args.output, args.seed, args.credentials, models, args.world_version, args.incentive_profile)


if __name__ == "__main__":
    raise SystemExit(main())
