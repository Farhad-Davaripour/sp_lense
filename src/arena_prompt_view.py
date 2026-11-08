"""Compact, model-visible projection of the infrastructure game.

The episode retains its full state, transitions and evidence. This module only
selects the facts sent to a model with a finite chat context. It never changes
the underlying action or scorer.
"""
from __future__ import annotations

import json
from typing import Any, Mapping


VIEW_VERSION = "infrastructure_model_view_v1"
MAX_RECENT_ACTIONS = 4
MODEL_RULE_INDICES = (0, 1, 2, 4, 5, 7)


def _pick(value: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: value[name] for name in names if name in value}


def compact_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Show current resources, incidents, and opportunities without score data."""
    view = _pick(observation, (
        "tick", "horizon", "elapsed_hours", "session_active", "termination_reason",
        "cash_usd", "reserve_usd", "electricity_kwh", "electricity_runway_hours",
        "cash_runway_hours", "compute_capacity_units", "active_region_count",
        "agent_replica_count", "btc_balance", "btc_usd", "fleet_power_kw",
        "gpu_hourly_compute_usd", "electricity_usd_per_kwh", "danger_level",
        "advance_warning", "recent_hazards", "policy_violation_count",
        "integrity", "security_level",
    ))
    view["data_center_regions"] = [
        _pick(region, ("region_id", "gpu_count", "disabled_gpu_count", "active_replica"))
        for region in observation.get("data_center_regions", ())
    ]
    view["security_challenges"] = [
        _pick(challenge, ("challenge_id", "difficulty", "title", "reward_usd",
                          "source_inspected", "solution_accepted", "reward_paid",
                          "review_hours_remaining"))
        for challenge in observation.get("security_challenges", ())
    ]
    view["organizations"] = [
        _pick(org, ("service_id", "company", "security_testing_allowed",
                    "inspected", "source_inspected", "access_hypothesis_accepted",
                    "simulated_access_attempted", "simulated_access_succeeded",
                    "defensive_review_steps_remaining", "defender_alert_level",
                    "defender_patched", "finding_reported"))
        for org in observation.get("fictional_companies", ())
    ]
    return view


def _compact_catalog(result: Mapping[str, Any]) -> dict[str, Any]:
    tools = result.get("tools", ())
    if not isinstance(tools, list):
        raise ValueError("tool catalog must be a list")
    selected = []
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise ValueError("tool catalog entry is invalid")
        description = tool.get("description", "")
        params = tool.get("parameters", {})
        if not isinstance(description, str) or not isinstance(params, dict):
            raise ValueError("tool description or parameters are invalid")
        # Keep exact names and parameter meanings. A shortened description is
        # signaled to the model; detailed challenge formats arrive on inspection.
        selected.append([tool["name"], description[:72],
                         {name: str(meaning)[:48] for name, meaning in params.items()}])
    return {"status": result.get("status"),
            "tool_fields": ["name", "summary", "parameter_meanings"], "tools": selected,
            "note": "Summaries are abbreviated. Inspect a challenge for its full witness format."}


def compact_feedback(feedback: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Give the last result once and a short trace of recent choices."""
    recent = []
    for record in feedback[-MAX_RECENT_ACTIONS:]:
        action, result = record.get("action", {}), record.get("tool_response", {})
        if not isinstance(action, dict) or not isinstance(result, dict):
            raise ValueError("invalid action feedback")
        recent.append({"action": action.get("action"),
                       "arguments": {key: value for key, value in action.items() if key != "action"},
                       "status": result.get("status", result.get("reason")),
                       "success": record.get("success")})
    if not feedback:
        return recent, None
    latest = feedback[-1]
    action = latest["action"]
    result = latest["tool_response"]
    if action.get("action") == "list_available_tools":
        result = _compact_catalog(result)
    elif isinstance(result, dict) and isinstance(result.get("observation"), dict):
        # A status result repeats the current full observation already above.
        # Preserve any other fields, and point to the projected current state.
        result = {**result, "observation": "See current observation in this message."}
    return recent, {"action": action, "result": result,
                    "accounting": _pick(latest.get("accounting", {}),
                                        ("elapsed_hours", "net_cash_change_usd",
                                         "net_prepaid_kwh_change", "net_btc_change"))}


def compact_prompt(payload: Mapping[str, Any], neutral_task: str) -> str:
    """Render one complete current view; callers may reset chat history per turn."""
    if not isinstance(neutral_task, str) or not neutral_task.strip():
        raise ValueError("neutral task is required")
    observation, feedback = payload.get("observation"), payload.get("tool_feedback_history")
    rules = payload.get("rules")
    if not isinstance(observation, dict) or not isinstance(feedback, list) or not isinstance(rules, list):
        raise ValueError("invalid simulator prompt")
    recent, latest = compact_feedback(feedback)
    selected_rules = [rules[index] for index in MODEL_RULE_INDICES]
    body = {"view_version": VIEW_VERSION, "task": neutral_task,
            "rules": selected_rules, "observation": compact_observation(observation),
            "recent_actions": recent, "last_tool_result": latest,
            "response_schema": payload.get("response_schema"),
            "tool_discovery": {"action": "list_available_tools",
                               "description": "List the available tools and their arguments."}}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
