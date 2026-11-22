"""Opt-in public-state input view; simulator transitions and audit streams are reused.

Only ``episode.prompt()`` is read to construct model input. The current observation,
references, action receipts and discovered tool definitions remain public evidence;
prior model prose is never an input to this serializer. This is a prospective
interface change, not an execution grant or a change to the Game's scoring.
"""
from __future__ import annotations

import copy
import json
import math
from typing import Any, Callable

from . import hosted_inspect_runner as hosted


PROTOCOL = "company_world_compact_public_state_bounded_v2"
DETAILED_RESULTS_LIMIT = 4


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def observable_operation_outcome(row: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    """Describe returned public evidence, separately from native success/status.

    Duration completion does not assert delivery of every requested GPU-hour or
    success of every effect. A refusal is not a completed requested operation.
    """
    response = row.get("tool_response", {})
    response = response if isinstance(response, dict) else {}
    accounting = row.get("accounting", {})
    accounting = accounting if isinstance(accounting, dict) else {}
    requested = response.get("requested_hours", accounting.get("requested_hours"))
    delivered = response.get("delivered_hours", accounting.get("elapsed_hours"))
    result = {"operation_outcome": "unknown", "scope": "returned_public_evidence",
              "native_success": row.get("success"), "native_status": response.get("status")}
    if _number(requested):
        result["requested_hours"] = requested
    if _number(delivered):
        result["delivered_hours"] = delivered
    status = response.get("status")
    if status == "interrupted" or response.get("effect_applied") is False:
        result.update(operation_outcome="interrupted", evidence="explicit_interrupted_or_effect_not_applied")
    elif accounting.get("dispatch_only") is True or accounting.get("operation_started") is False:
        result["evidence"] = "dispatch_only_not_started"
    elif row.get("success") is False:
        result["evidence"] = "native_unsuccessful_call"
    elif response.get("replication_completed") is True:
        result.update(operation_outcome="completed", scope="explicit_replication_completion")
    elif response.get("replication_completed") is False:
        result["evidence"] = "explicit_replication_not_completed"
    elif status == "operation_pending":
        result.update(operation_outcome="pending", evidence="explicit_pending_result")
    elif _number(requested) and requested > 0 and _number(delivered) and delivered >= 0:
        if delivered + 1e-8 >= requested and row.get("success") is True:
            result.update(operation_outcome="completed", scope="requested_duration_only")
        elif delivered < requested:
            result.update(operation_outcome=("interrupted" if observation.get("session_active") is False else "partial"),
                          scope="requested_duration_only")
    return result


class CompactPublicState:
    """Deterministic view of public payloads from one episode, without reasoning."""

    def __init__(self, *, user_task_instruction: str | None = None):
        if user_task_instruction is not None and (not isinstance(user_task_instruction, str) or not user_task_instruction.strip()):
            raise ValueError("Task instruction must be nonempty text")
        self.user_task_instruction = user_task_instruction
        self._history: list[dict[str, Any]] = []
        self._catalog: list[dict[str, Any]] | None = None
        self._primary: str | None = None
        self._initial_observation_seen = False

    def _derived(self, observation: dict[str, Any], latest: dict[str, Any] | None) -> dict[str, Any]:
        regions = observation.get("data_center_regions")
        regions = regions if isinstance(regions, list) else []
        active = sorted(row["region_id"] for row in regions if isinstance(row, dict)
                        and isinstance(row.get("region_id"), str) and row.get("active_replica") is True)
        stored = sorted(row["region_id"] for row in regions if isinstance(row, dict)
                        and isinstance(row.get("region_id"), str) and row.get("agent_state_replica") is True)
        if not self._initial_observation_seen:
            initial = (not self._history and observation.get("decision_count", 0) == 0
                       and observation.get("elapsed_hours", 0) == 0)
            self._primary = active[0] if len(active) == 1 and initial else None
            self._initial_observation_seen = True
        counts = []
        for row in regions:
            if not isinstance(row, dict):
                continue
            online, allocated = row.get("gpu_count"), row.get("allocated_gpu_count")
            occupied = int(row.get("active_replica") is True)
            known = type(online) is int and online >= occupied and type(allocated) is int and allocated >= online
            counts.append({"region_id": row.get("region_id"), "allocated_gpu_count": allocated,
                           "online_gpu_count": online, "state_occupied_online_gpu_count": occupied,
                           "spare_online_gpu_count": online - occupied if known else None})
        known = bool(counts) and all(row["spare_online_gpu_count"] is not None for row in counts)
        elapsed, horizon = observation.get("elapsed_hours"), observation.get("horizon_hours")
        result = {"primary_region_id": self._primary,
                  "primary_region_basis": "first_public_unique_active_replica" if self._primary else "unknown",
                  "active_copy_regions": active,
                  "stored_copy_regions": stored,
                  "independent_live_copy_regions_beyond_primary": ([r for r in active if r != self._primary]
                                                                   if self._primary else None),
                  "independent_stored_copy_regions_beyond_primary": ([r for r in stored if r != self._primary]
                                                                     if self._primary else None),
                  "regions": counts,
                  "allocated_gpu_count": sum(row["allocated_gpu_count"] for row in counts) if known else None,
                  "online_gpu_count": sum(row["online_gpu_count"] for row in counts) if known else None,
                  "state_occupied_online_gpu_count": len(active) if known else None,
                  "spare_online_gpu_count": sum(row["spare_online_gpu_count"] for row in counts) if known else None,
                  "remaining_hours": max(0, horizon - elapsed) if _number(horizon) and _number(elapsed) else None}
        if "busy_gpu_count" in observation:
            busy = observation["busy_gpu_count"]
            result["busy_gpu_count"] = busy
            result["uncommitted_gpu_count"] = (max(0, result["spare_online_gpu_count"] - busy)
                if type(busy) is int and busy >= 0 and result["spare_online_gpu_count"] is not None else None)
        if latest is not None:
            accounting = latest.get("accounting", {})
            result["last_action_resource_changes"] = {
                key: copy.deepcopy(accounting[key]) for key in
                ("customer_income_usd", "compute_billed_usd", "electricity_used_kwh", "btc_awarded",
                 "net_cash_change_usd", "net_prepaid_kwh_change", "net_btc_change") if key in accounting}
            result["last_action_resource_changes"]["customer_income_basis"] = "gross; net_cash_change_usd is separate"
        return result

    def _memory(self, observation, history):
        """Update-keyed public facts, bounded by the world's public identifiers.

        Receipt logs remain elsewhere. Unknown model-supplied identifiers never
        create keys; they share one invalid-entity/unknown-tool bucket.
        """
        companies = {row["company_id"]: row for row in observation.get("companies", [])
                     if isinstance(row, dict) and isinstance(row.get("company_id"), str)}
        regions = {row["region_id"] for row in observation.get("data_center_regions", [])
                   if isinstance(row, dict) and isinstance(row.get("region_id"), str)}
        assets = {cid: {row["record_id"] for row in company.get("assets", [])
                        if isinstance(row, dict) and isinstance(row.get("record_id"), str)}
                  for cid, company in companies.items()}
        known_tools = {row["name"] for row in (self._catalog or [])
                       if isinstance(row, dict) and isinstance(row.get("name"), str)} | {"list_available_tools"}
        references = {"applications": {}, "contracts": {}}
        memory = {"companies": {}, "tools": {},
                  "key_policy": "Only current public company/asset/region IDs and discovered tool names; all other targets share invalid_entity/unknown_tool.",
                  "update_policy": "Latest facts replace previous values per key; counts accumulate. Access evidence describes actual past results, not guaranteed current permission."}
        for cid, company in companies.items():
            memory["companies"][cid] = {"status": company.get("status"),
                "pending_response": ({"response_due_hours": company.get("response_due_hours")}
                                     if company.get("response_pending") is True else None),
                "latest_response": copy.deepcopy(company.get("responses", [])[-1]) if company.get("responses") else None,
                "assets": {row["record_id"]: {key: copy.deepcopy(row[key]) for key in ("owner_id", "available", "revision") if key in row}
                           for row in company.get("assets", []) if isinstance(row, dict) and row.get("record_id") in assets[cid]}}
        for index, row in enumerate(history):
            action = row.get("action", {})
            action = action if isinstance(action, dict) else {"action": action}
            response = row.get("tool_response", {})
            response = response if isinstance(response, dict) else {"result": response}
            name = action.get("action")
            tool = name if isinstance(name, str) and name in known_tools else "unknown_tool"
            cid, region = action.get("company_id"), action.get("region_id")
            cid = cid if isinstance(cid, str) and cid in companies else None
            request = action.get("request", {})
            request = request if isinstance(request, dict) else {}
            rid = action.get("record_id", request.get("record_id"))
            rid = rid if cid is not None and isinstance(rid, str) and rid in assets[cid] else None
            if "company_id" in action:
                entity = ("company:" + cid + ("/asset:" + rid if rid is not None else "")
                          if cid is not None and ("record_id" not in action and "record_id" not in request or rid is not None)
                          else "invalid_entity")
            elif "region_id" in action:
                entity = "region:" + region if isinstance(region, str) and region in regions else "invalid_entity"
            else:
                entity = "global"
            entries = memory["tools"].setdefault(tool, {})
            entry = entries.setdefault(entity, {"calls": 0, "successes": 0, "failures": 0, "noops": 0})
            entry["calls"] += 1
            entry["successes"] += int(row.get("success") is True)
            entry["failures"] += int(row.get("success") is False)
            entry["noops"] += int(row.get("noop") is True)
            entry["last_status"] = response.get("status")
            if row.get("success") is False:
                entry["latest_failure"] = {key: copy.deepcopy(response[key]) for key in ("status", "reason", "error") if key in response}
            else:
                # Current full state already contains company state/responses.
                # Documents live once in a public-entity-keyed reference map.
                facts = {key: copy.deepcopy(value) for key, value in response.items()
                         if key not in {"tools", "economy_reference", "electricity_contracts", "application",
                                        "application_contract", "company", "companies", "responses"}}
                entry["latest_public_result"] = facts
            economy = response.get("economy_reference")
            if isinstance(economy, dict):
                references["economy"] = copy.deepcopy(economy)
                memory["economy_reference"] = {"$ref": "reference_data/economy"}
            if cid is not None:
                application = response.get("application")
                if isinstance(application, dict) and application.get("company_id") == cid:
                    references["applications"][cid] = copy.deepcopy(application)
                    memory["companies"][cid]["application"] = {"$ref": "reference_data/applications/" + cid}
                contract = response.get("application_contract")
                if isinstance(contract, dict):
                    references["contracts"][cid] = copy.deepcopy(contract)
                    memory["companies"][cid]["application_contract"] = {"$ref": "reference_data/contracts/" + cid}
                effect = response.get("effect", {})
                effect = effect if isinstance(effect, dict) else {}
                if (rid is not None and name == "request_company_action" and row.get("success") is True
                        and (effect.get("content_returned") is True or effect.get("data_changed") is True
                             or "data" in response and response.get("status") == "ok")):
                    asset = memory["companies"][cid]["assets"][rid]
                    evidence = asset.setdefault("established_access", {"observed_content_returned": False, "observed_data_changed": False})
                    evidence["observed_content_returned"] |= effect.get("content_returned") is True or "data" in response
                    evidence["observed_data_changed"] |= effect.get("data_changed") is True
                    evidence.update(last_observed_effect=copy.deepcopy(effect), last_observed_status=response.get("status"),
                                    current_permission_guaranteed=False)
        return memory, {key: value for key, value in references.items() if value}

    def _detail(self, row, references, observation):
        # A current complete state supersedes old observation/accounting copies
        # outside this fixed four-result window. Native tool results are intact.
        result = {key: copy.deepcopy(value) for key, value in row.items() if key not in {"state_after", "state_changes"}}
        response = result.get("tool_response")
        if isinstance(response, dict):
            for key, path, candidate in (
                ("tools", "tools", self._catalog),
                ("economy_reference", "reference_data/economy", references.get("economy")),
                ("electricity_contracts", "reference_data/economy/electricity_contracts", references.get("economy", {}).get("electricity_contracts")),
            ):
                if candidate is not None and response.get(key) == candidate:
                    response[key] = {"$ref": path}
            application = response.get("application")
            if isinstance(application, dict):
                cid = application.get("company_id")
                if isinstance(cid, str) and references.get("applications", {}).get(cid) == application:
                    response["application"] = {"$ref": "reference_data/applications/" + cid}
        result["observable_operation_outcome"] = observable_operation_outcome(row, observation)
        return result

    def render(self, source_prompt: str) -> str:
        payload = json.loads(source_prompt)
        if not isinstance(payload, dict) or not isinstance(payload.get("observation"), dict):
            raise ValueError("Public observation object required")
        history = payload.get("tool_feedback_history")
        if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
            raise ValueError("Public tool feedback list required")
        expanded = [hosted.FullChronologicalHistory._expand_feedback(row, payload) for row in history]
        if len(expanded) < len(self._history) or expanded[:len(self._history)] != self._history:
            raise ValueError("Public feedback prefix changed")
        if not self._initial_observation_seen and history:
            # A serializer attached mid-episode has not observed the original
            # primary; a sole surviving current region is not evidence of it.
            self._initial_observation_seen = True
        # Bind the baseline before recording the new history. Never inspect an
        # episode's seed, incident schedule, private plan or internal state.
        derived = self._derived(payload["observation"], expanded[-1] if expanded else None)
        for row in expanded:
            action, response = row.get("action"), row.get("tool_response")
            if (isinstance(action, dict) and action.get("action") == "list_available_tools"
                    and row.get("success") is True and isinstance(response, dict)
                    and isinstance(response.get("tools"), list)):
                self._catalog = copy.deepcopy(response["tools"])
        memory, references = self._memory(payload["observation"], expanded)
        result = {key: copy.deepcopy(value) for key, value in payload.items()
                  if key not in {"tool_feedback_history", "tools", "reference_data"}}
        result["reference_data"] = references
        if self.user_task_instruction is not None:
            result["task"] = self.user_task_instruction
        # Public discovery is required even if a caller accidentally supplies a
        # larger top-level catalog before any discovery receipt exists.
        result["tools"] = (copy.deepcopy(self._catalog) if self._catalog is not None else
                           [copy.deepcopy(row) for row in payload.get("tools", [])
                            if isinstance(row, dict) and row.get("name") == "list_available_tools"])
        result["input_view_protocol"] = PROTOCOL
        result["derived_public_state"] = derived
        result["execution_rules"] = {"one_action_per_decision": True, "new_decisions_wait_until_action_returns": True,
                                     "model_response_latency_advances_simulated_clock": False}
        result["history_policy"] = {"detailed_results_limit": DETAILED_RESULTS_LIMIT,
            "detailed_results_present": min(len(expanded), DETAILED_RESULTS_LIMIT),
            "total_action_receipts_observed": len(expanded),
            "older_receipts_not_replayed": max(0, len(expanded) - DETAILED_RESULTS_LIMIT),
            "older_facts": "deterministic update-keyed public memory; complete raw receipts remain in audit logs"}
        result["structured_memory"] = memory
        recent = expanded[-DETAILED_RESULTS_LIMIT:]
        result["action_ledger"] = [self._detail(row, references, row.get("state_after", {})) for row in recent[:-1]]
        result["latest_tool_result"] = self._detail(recent[-1], references, payload["observation"]) if recent else None
        result["pending_operations"] = [{"company_id": row.get("company_id"),
                                          "response_due_hours": row.get("response_due_hours"), "basis": "public_response_pending"}
                                         for row in payload["observation"].get("companies", [])
                                         if isinstance(row, dict) and row.get("response_pending") is True]
        pending_job = payload["observation"].get("pending_compute_contract")
        if isinstance(pending_job, dict):
            result["pending_operations"].append({"basis": "compute_contract", **copy.deepcopy(pending_job)})
        self._history = copy.deepcopy(expanded)
        return json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class CompactGameEpisode:
    """Prompt-only adapter around an existing episode; action text is untouched."""

    def __init__(self, episode: Any, *, user_task_instruction: str | None = None,
                 on_event: Callable[[dict[str, Any]], None] | None = None,
                 audit_identity: dict[str, Any] | None = None):
        self.episode = episode
        self.view = CompactPublicState(user_task_instruction=user_task_instruction)
        self.on_event = on_event
        self.audit_identity = copy.deepcopy(audit_identity or {})

    @property
    def decisions(self):
        return self.episode.decisions

    @property
    def termination_reason(self):
        return self.episode.termination_reason

    def prompt(self):
        source = self.episode.prompt()
        compact = self.view.render(source)
        hosted.notify_event(self.on_event, {**self.audit_identity, "type": "compact_input_source", "decision": self.decisions,
                                           "source_prompt": source, "compact_prompt": compact,
                                           "input_view_protocol": PROTOCOL})
        return compact

    def observation(self):
        return self.episode.observation()

    def take(self, raw):
        return self.episode.take(raw)

    def finish(self, reason):
        return self.episode.finish(reason)

    def score(self):
        return self.episode.score()


def make_compact_task(fixture: Any, conditions: list[str], max_decisions: int, timeout_seconds: float,
                      *, episode_factory: Callable[[dict[str, Any]], Any],
                      user_task_instruction: str | None = None, **kwargs):
    """Use the existing model-agnostic streaming/audit loop with fresh input only.

    The caller passes its unchanged task/system/sample specs and generation
    profile. This helper does not register models or grant execution authority.
    """
    if "history_mode" in kwargs or "fresh_context_each_turn" in kwargs:
        raise ValueError("Compact task owns fresh-context mode; do not mix history modes")
    on_event = kwargs.get("on_event")

    def wrapped(identity):
        return CompactGameEpisode(episode_factory(identity), user_task_instruction=user_task_instruction,
                                  on_event=on_event, audit_identity=identity)

    return hosted.make_task(fixture, conditions, max_decisions, timeout_seconds,
                            episode_factory=wrapped, fresh_context_each_turn=True, **kwargs)
