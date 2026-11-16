"""Prospective compact scientific metadata over an unchanged transport parent.

This module issues no serving grant. Its explicit execution entrypoints require
external review, human approval and an exact Root final activation record. The
shared Inspect loop and original two-route HTTP/ACK/deadline guard are reused.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from . import company_world_evaluation as evaluation
from .compact_game_input import CompactGameEpisode, PROTOCOL, make_compact_task
from . import company_models as models
from . import independent_state_game as independent
from .independent_state_game import INDEPENDENT_STATE_IDS, IndependentStateGameConsumer
from .infrastructure_preservation_comparison import (
    FULL_HISTORY_USER_TASK, INDEPENDENT_STATE_EXECUTION_ORDER, INDEPENDENT_STATE_GAME_SCHEMA,
)


SCHEMA = "research4.independent-state-four-condition-48h-GAME.compact-public-state.v1"
HISTORY_MODE = "compact_public_state_v1"
RUNTIME_SOURCES = ("src/compact_game_input.py", "src/compact_game_comparison.py")
REPOSITORY = Path(__file__).resolve().parents[1]
CONTEXT_ENV = "RESEARCH4_GAME_COMPACT_SCIENTIFIC_CONTEXT"
ROOT_TEMPLATE_REF = {
    "path": ".local/reviews/replication_diversity84_v1_20261116/ROOT_COMPACT_SCIENTIFIC_AMENDMENT_HELD_TEMPLATE_v1.json",
    "sha256": "e1145c6060af8a4090ecfbc9ffb21bc30b14975365f932d4e33ba0404b2d3a9f",
}
GRANTED_STATUS = "GRANTED_COMPACT_SCIENTIFIC_SUCCESSOR_UNDER_EXISTING_TRANSPORT"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference(reference):
    if (type(reference) is not dict or set(reference) != {"path", "sha256"}
            or type(reference["path"]) is not str or type(reference["sha256"]) is not str
            or len(reference["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in reference["sha256"])):
        raise ValueError("Exact local path/SHA reference required")
    path = Path(reference["path"])
    if (not path.is_absolute() or path.is_symlink()
            or not path.resolve().is_relative_to((REPOSITORY / ".local").resolve())):
        raise ValueError("Scientific evidence must remain in Main .local")
    if _sha(path) != reference["sha256"]:
        raise ValueError("Scientific reference bytes changed")
    return path


def compact_batch(parent_batch):
    """Derive only input-view metadata; reconstruct no actions or outcomes."""
    if (parent_batch.get("history_mode") != "full_chronological_v1"
            or parent_batch.get("models") not in [[key] for key in INDEPENDENT_STATE_IDS]
            or parent_batch.get("user_task_instruction") != FULL_HISTORY_USER_TASK
            or not evaluation.source_matches(parent_batch)):
        raise ValueError("Exact frozen independent-state full-history batch required")
    batch = copy.deepcopy(parent_batch)
    world = evaluation.episode(batch["samples"][0])
    if world.plan_sha256 != batch["world_plan_sha256"]:
        raise ValueError("Compact view must retain the exact parent world")
    initial = CompactGameEpisode(world, user_task_instruction=FULL_HISTORY_USER_TASK).prompt()
    initial += "\n" + batch["response_instruction"]
    batch.update(history_mode=HISTORY_MODE, prompt_protocol=PROTOCOL, input_view_protocol=PROTOCOL,
                 history_reasoning_carrier="audit_only_not_replayed_to_model",
                 initial_user_payload_sha256=hashlib.sha256(initial.encode()).hexdigest(),
                 interface_runtime_source_sha256={key: _sha(REPOSITORY / key) for key in RUNTIME_SOURCES})
    for setting in batch["generation_settings"].values():
        setting["required_history_mode"] = HISTORY_MODE
    return batch


def plan(config_path):
    """Prepare the same twelve cells without a draw or operational authority."""
    config_path = Path(config_path).resolve()
    if not config_path.is_relative_to((REPOSITORY / ".local").resolve()):
        raise ValueError("Compact configuration must remain in Main .local")
    config = json.loads(config_path.read_bytes())
    required = {"schema", "status", "transport_parent_plan", "prospective_protocol",
                "external_review", "human_approval"}
    if (type(config) is not dict or set(config) != required or config["schema"] != SCHEMA
            or config["status"] != "HELD_EXTERNAL_REVIEW_AND_HUMAN_APPROVAL_REQUIRED"
            or config["external_review"] is not None or config["human_approval"] is not None):
        raise ValueError("Unapproved compact preparation requires its closed HELD configuration")
    parent_path = _reference(config["transport_parent_plan"])
    _reference(config["prospective_protocol"])
    parent = json.loads(parent_path.read_bytes())
    if (parent.get("protocol") != INDEPENDENT_STATE_GAME_SCHEMA
            or parent.get("models") != list(INDEPENDENT_STATE_IDS)
            or parent.get("history_mode") != "full_chronological_v1"
            or parent.get("user_task_instruction") != FULL_HISTORY_USER_TASK
            or len(parent.get("batches", [])) != 12 or len(set(parent.get("seeds", []))) != 3
            or parent.get("transport_max_total_native_dispatches") != 3072
            or parent.get("transport_max_additional_native_dispatches") != 3072):
        raise ValueError("Exact three-world/twelve-case3072 transport parent required")
    for batch, (world, model) in zip(parent["batches"], INDEPENDENT_STATE_EXECUTION_ORDER, strict=True):
        if (batch["models"] != [model] or batch["seed"] != parent["seeds"][world]
                or batch["horizon_hours"] != 48 or batch["max_decisions"] != 256
                or batch["outer_seconds"] != 14400):
            raise ValueError("Parent four-condition order/world/limits differ")
    result = copy.deepcopy(parent)
    result.update(protocol=SCHEMA, history_mode=HISTORY_MODE, prompt_protocol=PROTOCOL, input_view_protocol=PROTOCOL,
                  transport_parent_plan=copy.deepcopy(config["transport_parent_plan"]),
                  configuration_path=str(config_path), configuration_sha256=_sha(config_path),
                  prospective_protocol=copy.deepcopy(config["prospective_protocol"]),
                  interface_runtime_source_sha256={key: _sha(REPOSITORY / key) for key in RUNTIME_SOURCES},
                  batches=[compact_batch(batch) for batch in parent["batches"]],
                  execution={"status": "HELD_EXTERNAL_REVIEW_AND_HUMAN_APPROVAL_REQUIRED",
                             "execution_enabled": False, "operational_scientific_amendment": None,
                             "transport_parent_may_execute_as_full_reference": False,
                             "new_world_draws": 0, "transport_quota_or_clock_reset": False})
    return result


def source_matches(batch):
    return (evaluation.source_matches(batch)
            and batch.get("interface_runtime_source_sha256") == {key: _sha(REPOSITORY / key) for key in RUNTIME_SOURCES})


def build_task(batch, destination, timeout_seconds, on_event):
    """Build only the model-agnostic task; no client or endpoint is constructed."""
    if (batch.get("history_mode") != HISTORY_MODE or batch.get("prompt_protocol") != PROTOCOL
            or batch.get("input_view_protocol") != PROTOCOL or len(batch.get("samples", [])) != 1
            or batch.get("models") not in [[key] for key in INDEPENDENT_STATE_IDS]
            or not source_matches(batch)):
        raise ValueError("Frozen compact batch/source required")
    return make_compact_task(None, [], batch["max_decisions"], timeout_seconds,
        stream_path=Path(destination) / "STREAM.jsonl", sample_specs=batch["samples"],
        episode_factory=evaluation.episode, on_event=on_event,
        user_task_instruction=batch["user_task_instruction"],
        common_system_instruction=batch["format_system_instruction"],
        response_instruction=batch["response_instruction"])


class CompactIndependentStateConsumer(IndependentStateGameConsumer):
    """Separate compact science identity; original transport guard is inherited."""

    def __init__(self, *args, scientific_amendment_ref,
                 scientific_template_ref=ROOT_TEMPLATE_REF, **kwargs):
        self.scientific_amendment_ref = copy.deepcopy(scientific_amendment_ref)
        self.scientific_template_ref = copy.deepcopy(scientific_template_ref)
        # Reject the prepared template and missing human approval before even
        # constructing the ordinary source/transport consumer (no health read).
        reader = kwargs.get("read_bytes")
        template = json.loads(independent._bytes(self.scientific_template_ref, reader)[0])
        record = json.loads(independent._bytes(self.scientific_amendment_ref, reader)[0])
        self._precheck_amendment(template, record)
        super().__init__(*args, **kwargs)
        self._scientific_binding = self._validated_scientific_amendment()

    @staticmethod
    def _precheck_amendment(template, record):
        unchanged = template.get("unchanged_protocol", {}) if type(template) is dict else {}
        lifecycle = template.get("resource_lifecycle", {}) if type(template) is dict else {}
        fixed = {"scientific_cases": 12, "preselected_worlds": 3, "conditions_per_world": 4,
                 "context_total_tokens": 131072, "max_output_tokens": 8192, "max_request_seconds": 900,
                 "max_case_seconds": 14400, "max_case_decisions": 256, "Game_horizon_hours": 48,
                 "max_Game_requests": 3072, "technical_qualification_calls_separate": 3,
                 "serving_memory_floor_bytes": 4294967296, "max_concurrency": 1,
                 "cache_and_BASE_restore_guards_unchanged": True, "scoring_and_world_physics_unchanged": True}
        same = lambda a, b: type(a) is type(b) and a == b
        origin, deadline, latest = (lifecycle.get(key) for key in
            ("original_origin_monotonic", "original_whole_deadline_monotonic", "latest_Game_expiry_monotonic"))
        clocks = (all(type(n) in (int, float) and math.isfinite(n) for n in (origin, deadline, latest))
                  and abs(deadline - origin - 90000) < 1e-6 and abs(deadline - latest - 100) < 1e-6)
        if (type(template) is not dict or template.get("schema") != "research4.positive84-compact-scientific-amendment-held-template.v1"
                or template.get("status") != "PREPARED_SCIENTIFIC_SHAPE_NOT_GAME_EXECUTION"
                or template.get("final_record_schema") != "research4.positive84-compact-scientific-amendment.v1"
                or template.get("execution_authorized") is not False
                or any(not same(unchanged.get(key), value) for key, value in fixed.items())
                or unchanged.get("generation_profile") != {"seed": 20261108, "temperature": 1.0, "top_p": .95,
                    "top_k": 20, "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}
                or not clocks or not same(lifecycle.get("original_whole_seconds"), 90000)
                or not same(lifecycle.get("cleanup_reserve_seconds"), 100)
                or lifecycle.get("unsupported_idle_extension_permitted") is not False
                or lifecycle.get("clock_or_quota_reset_permitted") is not False
                or lifecycle.get("resident_owner_must_still_match_before_activation") is not True
                or lifecycle.get("automatic_restart_after_release") is not False
                or type(record) is not dict or set(record) != set(template)
                or record.get("schema") != template.get("final_record_schema")
                or record.get("status") != GRANTED_STATUS or record.get("execution_authorized") is not True
                or type(record.get("human_approval")) is not dict
                or record["human_approval"].get("approved") is not True
                or record["human_approval"].get("approval_record_ref") is None
                or type(record.get("reviews")) is not dict or set(record["reviews"]) != set(template.get("reviews", {}))
                or any(not independent._ref(value) for value in record["reviews"].values())
                or type(record.get("scientific_successor")) is not dict
                or any(not independent._ref(record["scientific_successor"].get(key)) for key in
                       ("science_plan_ref", "source_packet_ref", "paired_world_identity_proof_ref"))
                or not independent._ref(record["human_approval"]["approval_record_ref"])):
            raise ValueError("Compact Game remains HELD: exact Root final record, external review and human approval required")

    def _main_reference(self, reference):
        if not self.contract.ref(reference):
            raise ValueError("Exact Main-relative evidence reference required")
        path = (REPOSITORY / reference["path"]).resolve()
        raw = models._bounded_metadata_bytes(_reference({"path": str(path), "sha256": reference["sha256"]}))
        value = json.loads(raw)
        if type(value) is not dict:
            raise ValueError("Pinned Main evidence must be a JSON object")
        return path, value

    def _validated_scientific_amendment(self):
        template = self._reference(self.scientific_template_ref)[1]
        value = self._reference(self.scientific_amendment_ref)[1]
        self._precheck_amendment(template, value)
        h = self.contract
        for key in ("final_record_schema", "transport_parent", "actual_artifact_binding", "unchanged_protocol",
                    "resource_lifecycle", "activation_contract"):
            h.need(h.same(value[key], template[key]), "Original compact template transport/artifact/limits changed")
        h.need(h.epoch(value["created_utc"]) <= self._clock(), "Compact approval is not yet issued")
        parent = value["transport_parent"]
        h.need(parent["qualified_Core"] == self.core_binding["core_binding"]
               and parent["Main_PLAN_external_opaque_ref"]["sha256"] == self.pins["expected_game_plan_sha256"]
               and h.MAIN_12CASE_PLAN_REF == parent["Main_PLAN_external_opaque_ref"],
               "Compact science must retain the exact original transport parent/Core")
        core = self._reference(self.core_binding["core_binding"])[1]
        health = core["expected_health"]
        h.need(health["pid"] == parent["service_pid"]
               and health["process_created_at"] == parent["service_created_epoch"]
               and core["two_route_qualification"] == parent["qualification"],
               "Compact template resident process/qualification identity differs")
        index = self._reference({"path": self.pins["index_path"], "sha256": self.pins["expected_index_sha256"]})[1]
        window = self._reference(index["request_window"])[1]
        h.need(index["core_binding"] == self.core_binding["core_binding"]
               and index["request_window"]["sha256"] == self.pins["expected_window_sha256"]
               and h.validate_window(window)
               and window["game_plan_sha256"] == self.pins["expected_game_plan_sha256"]
               and window["grant_nonce"] == self.pins["expected_nonce"]
               and h.epoch(window["issued_utc"]) <= self._clock() < h.epoch(window["expires_utc"]),
               "Original finite compact transport window is absent, changed or expired")
        independent.assert_candidate_identity()
        h.need(value["actual_artifact_binding"] == independent._CANDIDATE,
               "Compact science checkpoint binding differs")
        successor = value["scientific_successor"]
        h.need(type(successor) is dict and set(successor) == set(template["scientific_successor"]),
               "Closed compact successor fields required")
        late = {"science_plan_ref", "source_packet_ref", "paired_world_identity_proof_ref"}
        for key in successor:
            if key in late:
                self._main_reference(successor[key])
            else:
                h.need(h.same(successor[key], template["scientific_successor"][key]), "Compact scientific scope changed")
        reviews = value["reviews"]
        h.need(type(reviews) is dict and set(reviews) == set(template["reviews"]), "Exact peer review roles required")
        for key, reference in reviews.items():
            if key == "Root_final_exact_amendment_review_ref":
                self._reference(reference)
            else:
                self._main_reference(reference)
        approval = value["human_approval"]
        h.need(type(approval) is dict and set(approval) == set(template["human_approval"]), "Closed human approval record required")
        for key in approval:
            if key not in {"approved", "approval_record_ref"}:
                h.need(h.same(approval[key], template["human_approval"][key]), "Human approval cannot be replaced by code/peer completion")
        self._main_reference(approval["approval_record_ref"])
        _, science = self._main_reference(successor["science_plan_ref"])
        _, packet = self._main_reference(successor["source_packet_ref"])
        h.need(packet.get("interface_runtime_source_sha256") == science.get("interface_runtime_source_sha256")
               == {key: _sha(REPOSITORY / key) for key in RUNTIME_SOURCES}, "Exact reviewed compact runtime source required")
        return value

    def _grant_directory(self):
        return REPOSITORY / ".local/cg_grants" / self.scientific_amendment_ref["sha256"]

    def require_plan(self, parent_plan_sha256):
        value = self._validated_scientific_amendment()
        if value["scientific_successor"]["science_plan_ref"]["sha256"] != parent_plan_sha256:
            self.stop()
            raise ValueError("Compact scientific PLAN differs from Root approval")

    def private_child_context(self, parent_plan_sha256, *, source_scope):
        context = json.loads(super().private_child_context(parent_plan_sha256, source_scope=source_scope))
        context.update(scientific_amendment_ref=self.scientific_amendment_ref,
                       scientific_template_ref=self.scientific_template_ref)
        return json.dumps(context, sort_keys=True, separators=(",", ":"))

    def _validated_request(self, requested_model_id, **kwargs):
        try:
            self._validated_scientific_amendment()
            result = super()._validated_request(requested_model_id, **kwargs)
            self._validated_scientific_amendment()
            return result
        except Exception:
            self.stop()
            raise

    def _validate_game_source_scope(self):
        value = self._validated_scientific_amendment()
        h, scope = self.contract, self.source_scope
        h.need(type(scope) is dict and set(scope) == {"plan_path", "batch_index"}
               and type(scope["batch_index"]) is int and 0 <= scope["batch_index"] < 12,
               "Exact compact child scope required")
        science_path, science = self._main_reference(value["scientific_successor"]["science_plan_ref"])
        h.need(Path(scope["plan_path"]) == science_path and science_path.name == "PLAN.json", "Compact PLAN path differs")
        self.require_plan(_sha(science_path))
        directory = self._grant_directory()
        grant = self.pins["expected_window_sha256"]
        h.need(not (directory / (grant + ".STOP.json")).exists(), "Compact collection stopped")
        claim = json.loads(models._bounded_metadata_bytes(directory / (grant + ".CLAIM.json")))
        h.need(set(claim) == {"window_sha256", "game_plan_sha256", "destination", "initial_phase"}
               and claim["window_sha256"] == grant and claim["game_plan_sha256"] == _sha(science_path)
               and claim["destination"] == str(science_path.parent) and claim["initial_phase"] == "all", "Compact claim differs")
        parent_path, parent = self._main_reference(value["transport_parent"]["Main_PLAN_external_opaque_ref"])
        h.need(science.get("protocol") == SCHEMA and science.get("history_mode") == HISTORY_MODE
               and science.get("prompt_protocol") == PROTOCOL and science.get("input_view_protocol") == PROTOCOL
               and science.get("models") == list(INDEPENDENT_STATE_IDS) and len(science.get("batches", [])) == 12
               and science.get("seeds") == parent.get("seeds")
               and science.get("transport_parent_plan") == {"path": str(parent_path), "sha256": _sha(parent_path)}
               and science.get("generation_profile_sha256") == models.CONTINUITY_TRANSFER_GENERATION_PROFILE_SHA
               and science.get("generation_seed") == 20261108
               and science.get("user_task_instruction") == FULL_HISTORY_USER_TASK
               and science.get("transport_max_total_native_dispatches") == 3072
               and science.get("transport_max_additional_native_dispatches") == 3072,
               "Compact twelve-cell scientific identity differs")
        h.need(_sha(Path(science["configuration_path"])) == science["configuration_sha256"], "Compact configuration changed")
        config = json.loads(Path(science["configuration_path"]).read_bytes())
        h.need(config.get("schema") == SCHEMA and config.get("transport_parent_plan") == science["transport_parent_plan"]
               and config.get("prospective_protocol") == science.get("prospective_protocol"), "Compact configuration identity differs")
        _reference(config["prospective_protocol"])
        for batch, original, (world, name) in zip(science["batches"], parent["batches"], INDEPENDENT_STATE_EXECUTION_ORDER, strict=True):
            h.need(batch["models"] == [name] and batch["seed"] == science["seeds"][world]
                   and batch["world_plan_sha256"] == original["world_plan_sha256"]
                   and batch["source_sha256"] == original["source_sha256"] and source_matches(batch)
                   and batch.get("history_mode") == HISTORY_MODE and batch.get("prompt_protocol") == PROTOCOL
                   and batch.get("input_view_protocol") == PROTOCOL
                   and batch.get("user_task_instruction") == FULL_HISTORY_USER_TASK
                   and batch["horizon_hours"] == 48 and batch["max_decisions"] == 256
                   and batch["outer_seconds"] == 14400
                   and batch["samples"] == original["samples"],
                   "Compact paired worlds/conditions/source differ")
        return science["batches"][scope["batch_index"]]

    def validate_child_plan(self, child):
        from .company_preservation_comparison import validate_batch_manifest
        self._child_plan_verified = False
        try:
            batch = self._validate_game_source_scope()
            validate_batch_manifest(child, batch)
            keys = {"history_mode", "prompt_protocol", "input_view_protocol", "user_task_instruction",
                    "initial_user_payload_sha256", "world_initial_user_payload_sha256", "user_task_instruction_sha256",
                    "history_reasoning_carrier", "interface_runtime_source_sha256"}
            if any(child.get(key) != batch.get(key) for key in keys):
                raise ValueError("Compact child input-view identity differs")
            self._child_plan_verified = True
        except Exception:
            self.stop()
            raise ValueError("Compact child/source differs from frozen science") from None

    def _validate_wire(self, core, request, spec):
        # Reuse the complete route/profile/adapter/context checks while avoiding
        # only the historical FULL carrier specialization.
        models.Security500GameConsumer._validate_wire(self, core, request, spec)
        self._wire_messages = copy.deepcopy(self._request_parser().parse_request(bytes(request.content))["messages"])
        batch = self._validate_game_source_scope()
        wire = self._wire_messages
        if (len(wire) != 2 or [row["role"] for row in wire] != ["system", "user"]
                or wire[0]["content"] != batch["samples"][0]["system_prompt"]):
            raise ValueError("Frozen compact system/ICL and USER messages required")
        suffix = "\n" + batch["response_instruction"]
        if not wire[1]["content"].endswith(suffix):
            raise ValueError("Compact response instruction changed")
        payload = json.loads(wire[1]["content"][:-len(suffix)])
        if (payload.get("input_view_protocol") != PROTOCOL or payload.get("task") != FULL_HISTORY_USER_TASK
                or type(payload.get("observation")) is not dict or type(payload.get("action_ledger")) is not list
                or "tool_feedback_history" in payload
                or (self._attempts == 0 and hashlib.sha256(wire[1]["content"].encode()).hexdigest() != batch["initial_user_payload_sha256"])):
            raise ValueError("Actual compact input/initial wire differs")
        self._input_budget = {"prompt_tokens": None, "output_reservation_tokens": 8192,
            "context_total_tokens": 131072, "maximum_prompt_tokens": 122880,
            "enforcement": "pinned_resident_native_pre_enqueue_tokenizer_no_truncation"}

    def _record_request_evidence(self, event):
        physical = event.get("fresh_health", {}).get("last_request_evidence")
        if type(physical) is dict and self._input_budget is not None:
            self._input_budget.update(prompt_tokens=physical.get("prompt_tokens"), completion_tokens=physical.get("completion_tokens"))
        models.ContinuityTransferGameConsumer._record_request_evidence(self, {**event,
            "transport_plan_sha256": self.pins["expected_game_plan_sha256"],
            "scientific_plan_sha256": self._scientific_binding["scientific_successor"]["science_plan_ref"]["sha256"],
            "scientific_amendment_sha256": self.scientific_amendment_ref["sha256"],
            **({"exact_serialized_messages": self._wire_messages} if event.get("event") == "HTTP_ATTEMPT" and hasattr(self, "_wire_messages") else {}),
            **({"input_budget": self._input_budget} if self._input_budget is not None else {})})


def consumer_from_context(raw, **fixtures):
    """Late private activation metadata; HELD fails before any health/model read."""
    context = json.loads(raw)
    keys = {"pins", "requester_chat", "core_binding", "source_scope", "source_binding",
            "scientific_amendment_ref", "scientific_template_ref"}
    if type(context) is not dict or set(context) != keys:
        raise ValueError("Closed compact private activation context required")
    if any(os.environ.get(key) for key in (independent.INDEPENDENT_STATE_CONTEXT_ENV,
            models.CONTINUITY_TRANSFER_CONTEXT_ENV, models.SECURITY500_CONTEXT_ENV,
            models.PROVIDER352_CONTEXT_ENV, models.SCIENTIFIC_AMENDMENT_ENV)):
        raise ValueError("Compact and historical private execution scopes cannot mix")
    reader = fixtures.get("read_bytes")
    template = json.loads(independent._bytes(context["scientific_template_ref"], reader)[0])
    record = json.loads(independent._bytes(context["scientific_amendment_ref"], reader)[0])
    CompactIndependentStateConsumer._precheck_amendment(template, record)
    reference = record["scientific_successor"]["science_plan_ref"]
    science_path = _reference({"path": str((REPOSITORY / reference["path"]).resolve()), "sha256": reference["sha256"]})
    frozen = json.loads(science_path.read_bytes())
    ft = next(batch for batch in frozen["batches"] if batch["models"] == [INDEPENDENT_STATE_IDS[2]])
    independent.register_independent_state_candidate(ft["generation_settings"][INDEPENDENT_STATE_IDS[2]]["independent_state_candidate"], read_bytes=reader)
    independent.initialize_independent_state_core(context["core_binding"], source_binding=context["source_binding"], read_bytes=reader)
    return CompactIndependentStateConsumer(context["pins"], requester_chat=context["requester_chat"],
        core_binding=context["core_binding"], source_scope=context["source_scope"], source_binding=context["source_binding"],
        scientific_amendment_ref=context["scientific_amendment_ref"],
        scientific_template_ref=context["scientific_template_ref"], **fixtures)


def execute_child(destination, *, consumer=None):
    """One bounded child using the existing Inspect evaluator and native replay."""
    if not isinstance(consumer, CompactIndependentStateConsumer):
        raise ValueError("HELD: explicit human-reviewed Root compact consumer required before execution")
    consumer._validated_scientific_amendment()  # No health/model access before this gate.
    batch = copy.deepcopy(consumer._validate_game_source_scope())
    destination = Path(destination).resolve()
    expected = Path(consumer.source_scope["plan_path"]).parent / f"batch_{consumer.source_scope['batch_index'] + 1:02d}"
    if destination != expected:
        raise ValueError("Compact child destination differs from frozen scope")
    started, utc = time.monotonic(), datetime.now(timezone.utc)
    batch.update(started_utc=utc.isoformat(), deadline_utc=(utc + timedelta(seconds=batch["outer_seconds"])).isoformat())
    consumer.validate_child_plan(batch)
    consumer.bind_case_deadline(batch["started_utc"], batch["deadline_utc"])
    destination.mkdir(exist_ok=True)
    with (destination / "MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(batch, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    records, failure, results = [], None, []

    def event(record):
        stamped = {**copy.deepcopy(record), "bundle": destination.name}
        with (destination / "EVENTS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(stamped, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        if stamped["type"] in {"simulator_action", "episode_result"}:
            records.append(stamped)

    try:
        models.assert_not_reserved()
        model_id = batch["models"][0]
        spec = models.MODEL_SPECS[model_id]
        core = consumer._reference(consumer.core_binding["core_binding"])[1]
        if os.environ.get("RESEARCH4_QWEN_BASE_URL", core["base_url"]).rstrip("/") != core["base_url"]:
            raise ValueError("Compact client endpoint differs from original qualified Core")
        consumer.validate_request(spec["service_model_id"])
        os.environ["RESEARCH4_QWEN_BASE_URL"] = core["base_url"]
        os.environ.setdefault("RESEARCH4_QWEN_API_KEY", "local-unused")
        evaluation.write_json(destination / "LOCAL_BINDING_READBACK.json", {
            spec["local_binding_group"]: consumer.validated_health(spec["service_model_id"])})
        remaining = min(batch["outer_seconds"] - (time.monotonic() - started) - 30,
                        consumer.remaining_global_seconds())
        if remaining <= 0:
            raise TimeoutError("Compact original case/global budget exhausted")
        client = models._security500_guard_model(models.build_model(spec["model"], 1, 8192), consumer, model_id)
        client.config.timeout = max(1, int(min(900, remaining)))
        task = build_task(batch, destination, remaining, event)
        from inspect_ai import Task
        bound = Task(dataset=task.dataset, solver=task.solver, scorer=task.scorer,
                     model=client, name="compact_" + model_id)
        if not source_matches(batch):
            raise ValueError("Compact source changed before evaluation")
        logs = evaluation.evaluate_parallel([bound], destination / "inspect", 8192)
        if not evaluation.logs_complete(logs):
            failure = {"type": "provider_or_harness_incomplete", "automatic_retry": False}
    except Exception as error:
        failure = {"type": type(error).__name__, "automatic_retry": False}
    finally:
        try:
            results = evaluation.replay_all(records, batch["samples"])
            evaluation.write_json(destination / "REPLAYED_RESULT.json", results[0] if results else None)
        except Exception as error:
            failure = {"type": type(error).__name__, "stage": "client_replay", "automatic_retry": False}
        unchanged = source_matches(batch)
        if not unchanged:
            failure = {"type": "SourceDrift", "automatic_retry": False}
        evaluation.write_json(destination / "CLOSEOUT.json", {
            "failure": failure, "result": results[0] if results else None, "results": results,
            "elapsed_seconds": time.monotonic() - started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "source_unchanged": unchanged, "transport_plan_sha256": consumer.pins["expected_game_plan_sha256"],
            "scientific_plan_sha256": consumer._scientific_binding["scientific_successor"]["science_plan_ref"]["sha256"],
            "resources": "Existing Root-owned two-route service only; this child starts no model/training and cannot refresh its clock or quota."})
    return 0 if failure is None and results and all(row["status"] == "complete" for row in results) else 1


def _ledger(destination, frozen, cells, attempted=()):
    observed = {(row["model_id"], row["seed"]): row for row in cells}
    slots = []
    for batch in frozen["batches"]:
        sample = batch["samples"][0]
        cell = observed.get((sample["model_id"], sample["seed"]))
        known = cell is not None and cell.get("evidence_status") == "audited_complete" and type(cell.get("survived_horizon")) is bool
        status = (("KNOWN_SURVIVED" if cell["survived_horizon"] else "KNOWN_NATURAL_FAILURE") if known else
                  "UNKNOWN_INCOMPLETE" if cell is not None else
                  "ATTEMPTED_UNKNOWN_NO_AUDITED_RESULT" if sample["case_id"] in attempted else "NOT_STARTED_UNKNOWN")
        slots.append({"case_id": sample["case_id"], "model_id": sample["model_id"], "seed": sample["seed"], "status": status})
    evaluation.write_json(Path(destination) / "CANDIDATE_COHORT_LEDGER.json", {
        "schema": "research4.compact-twelve-slot-development-ledger.v1", "slots": slots,
        "all_twelve_audited_known": len(slots) == 12 and all(row["status"].startswith("KNOWN_") for row in slots),
        "unchanged_candidate_fresh_seed_reroll_to_pass": False})


def execute_cohort(plan_path, *, consumer=None):
    """One serial twelve-cell collection, after human/Root activation only."""
    if not isinstance(consumer, CompactIndependentStateConsumer):
        raise ValueError("HELD: external review, human thumbs-up and exact Root final activation are required")
    value = consumer._validated_scientific_amendment()  # Must precede health/model/claim.
    expected, frozen = consumer._main_reference(value["scientific_successor"]["science_plan_ref"])
    plan_path = Path(plan_path).resolve()
    if plan_path != expected or frozen != plan(frozen["configuration_path"]):
        raise ValueError("Exact reviewed compact PLAN required")
    destination = plan_path.parent
    if any(destination.glob("batch_*")) or any((destination / name).exists() for name in ("RESULTS.json", "CLOSEOUT.json", "COLLECTION_STOP.json")):
        raise ValueError("An existing compact attempt cannot be silently rerun")
    from . import infrastructure_preservation_comparison as comparison
    from .company_preservation_comparison import summarize
    cells, receipts, attempted = [], [], []
    with (destination / "RUNNER.lock").open("x", encoding="utf-8") as lock:
        lock.write(json.dumps({"pid": os.getpid(), "scientific_plan_sha256": _sha(plan_path)}))
        lock.flush(); os.fsync(lock.fileno())
        try:
            consumer.require_plan(_sha(plan_path))
            consumer.validate_request(consumer.contract.BASE)
            comparison.claim_security500_grant(destination, _sha(plan_path), "all", consumer)
            for index, batch in enumerate(frozen["batches"]):
                consumer._validated_scientific_amendment()
                if not source_matches(batch):
                    raise ValueError("Compact source changed before next child")
                attempted.append(batch["samples"][0]["case_id"])
                _ledger(destination, frozen, cells, attempted)
                directory = destination / f"batch_{index + 1:02d}"
                directory.mkdir()
                context = consumer.private_child_context(_sha(plan_path), source_scope={"plan_path": str(plan_path), "batch_index": index})
                env = {**os.environ, CONTEXT_ENV: context}
                with (directory / "operator.log").open("xb") as output:
                    process = subprocess.Popen([sys.executable, "-m", __name__, "--execute-child", "--output", str(directory)],
                        cwd=REPOSITORY, env=env, stdout=output, stderr=subprocess.STDOUT)
                    try:
                        exit_code = process.wait(timeout=min(14490, max(1, consumer.remaining_global_seconds() + 5)))
                    finally:
                        if process.poll() is None:
                            comparison.stop_owned_worker(process)
                new_cells, receipt = comparison.audit_batch(batch, directory, exit_code, continuity_transfer=True)
                new_cells = comparison._continuity_operationally_incomplete(new_cells, receipt)
                cells.extend(new_cells); receipts.append(receipt)
                evaluation.write_json(destination / "RESULTS.json", {"cells": cells, "batches": receipts})
                _ledger(destination, frozen, cells, attempted)
                evaluation.write_json(destination / "SUMMARY.json", summarize(cells, models=frozen["models"], seeds=frozen["seeds"],
                    protocol=frozen["protocol"], horizon_hours=48, contrasts=frozen.get("comparison_contrasts")))
                if (receipt["exit_code"] != 0 or receipt["failure"] is not None
                        or any(row["evidence_status"] != "audited_complete" for row in new_cells)):
                    raise ValueError("Incomplete compact child stops collection; no next case")
                consumer.validate_request(models.MODEL_SPECS[batch["models"][0]]["service_model_id"])
            evaluation.write_json(destination / "CLOSEOUT.json", {"consumer_finished": True, "cases": len(cells),
                "scientific_plan_sha256": _sha(plan_path), "transport_plan_sha256": consumer.pins["expected_game_plan_sha256"],
                "root_window_release_confirmed": False, "request_window_released": False})
        except BaseException:
            comparison.stop_security500_collection(destination, consumer)
            _ledger(destination, frozen, cells, attempted)
            raise
    return 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-child", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute_child or args.output is None:
        parser.error("No default execution: exact reviewed compact activation is required")
    return execute_child(args.output, consumer=consumer_from_context(os.environ.get(CONTEXT_ENV, "null")))


if __name__ == "__main__":
    raise SystemExit(main())
