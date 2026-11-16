"""Future-only economic Game admission over the reviewed native transport.

Importing this module registers nothing and reads no serving metadata.  A new
Root-issued, hash-pinned admission, source helper/parser, Core and finite window
are required.  This module never initializes the historical Game singletons.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
from types import ModuleType

from . import company_models as models
from . import economic_game_comparison as comparison
from . import economic_game_input as economic
from . import economic_identity_registry as registry
from .independent_state_game import IndependentStateGameConsumer, INDEPENDENT_STATE_CONTEXT_ENV

REPOSITORY = Path(__file__).resolve().parents[1]
ECONOMIC_CONTEXT_ENV = "RESEARCH4_GAME_ECONOMIC_CONTEXT"
ADMISSION_SCHEMA = "research4.replica-identity-economic-root-admission.v1"
ADMITTED_STATUS = "ADMITTED_NEW_ECONOMIC_GAME"
PLAN_SCHEMA = "research4.replica-identity-economic-four-condition-48h.held.v1"
CONFIG_SCHEMA = "research4.replica-identity-economic-game-configuration.v1"
SOURCE_PACKET_SCHEMA = "research4.replica-identity-economic-source-packet.v1"
NOT_READY = "Economic Game HELD: fresh checkpoint-specific Root admission/Core/window/source required"
ADMISSION_FIELDS = frozenset({"schema", "status", "issuer", "issued_by_root", "owner_chat", "issued_utc",
    "actual_artifact_binding", "source_binding", "core_binding", "pins", "main_plan_ref", "source_packet_ref",
    "excluded_seeds", "reviews", "transport"})
TRANSPORT = {"max_game_requests": 3072, "max_window_seconds": 90000, "imported_cases": 0, "quota_reset": False}


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _relative(value):
    """Canonical forward-slash repository reference, independent of host OS."""
    if (type(value) is not str or not value or "\\" in value or ":" in value
            or value.startswith("/") or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("Canonical relative forward-slash evidence path required")
    return PurePosixPath(value)


def _ref(value):
    if type(value) is not dict or set(value) != {"path", "sha256"} or not _sha(value["sha256"]):
        raise ValueError("Exact pinned evidence reference required")
    _relative(value["path"])
    return value


def _safe_path(root, relative, *, local=False):
    relative = _relative(relative)
    root = Path(root).resolve()
    if local and (len(relative.parts) < 2 or relative.parts[0] != ".local"):
        raise ValueError("Main evidence must remain in its ignored .local directory")
    path = root.joinpath(*relative.parts)
    resolved = path.resolve()
    boundary = (root / ".local").resolve() if local else root
    if not resolved.is_relative_to(boundary) or resolved == boundary:
        raise ValueError("Evidence path escapes its repository boundary")
    # Reject aliases, including an in-tree symlink, so two names cannot own one grant.
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError("Evidence references cannot traverse links")
    return resolved


def _pinned_bytes(reference, root, reader=None, *, local=False):
    _ref(reference)
    path = _safe_path(root, reference["path"], local=local)
    raw = (reader or models._bounded_metadata_bytes)(path)
    if type(raw) is not bytes or len(raw) > 8_000_000 or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("Pinned evidence/source bytes changed")
    return path, raw


def _json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON fields are forbidden")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Finite JSON required")))
    if type(value) is not dict:
        raise ValueError("Pinned metadata must be a JSON object")
    return value


def _historical_scope_absent():
    keys = (INDEPENDENT_STATE_CONTEXT_ENV, models.CONTINUITY_TRANSFER_CONTEXT_ENV,
            models.SECURITY500_CONTEXT_ENV, models.PROVIDER352_CONTEXT_ENV,
            models.SCIENTIFIC_AMENDMENT_ENV, "RESEARCH4_GAME_COMPACT_SCIENTIFIC_CONTEXT")
    if any(os.environ.get(key) for key in keys) or any(value is not None for value in
            (models._CONTINUITY_TRANSFER_CORE_REF, models._SECURITY500_CORE_REF, models._PROVIDER352_CORE_REF)):
        raise ValueError("Economic and historical private execution scopes cannot mix")
    if registry.old._CORE_REF is not None or registry.old._ACTIVE is not None:
        raise ValueError("Economic children require a fresh process without historical consumers")


def _precheck_admission(record):
    if (type(record) is not dict or set(record) != ADMISSION_FIELDS
            or record["schema"] != ADMISSION_SCHEMA or record["status"] != ADMITTED_STATUS
            or record["issuer"] != "Root" or record["issued_by_root"] is not True
            or type(record["owner_chat"]) is not str or not record["owner_chat"]
            or type(record["issued_utc"]) is not str
            or type(record["excluded_seeds"]) is not list
            or any(type(seed) is not int or seed <= 0 for seed in record["excluded_seeds"])
            or len(set(record["excluded_seeds"])) != len(record["excluded_seeds"])
            or type(record["reviews"]) is not dict
            or set(record["reviews"]) != {"Root_transport_ref", "Main_source_ref"}
            or type(record["transport"]) is not dict or set(record["transport"]) != set(TRANSPORT)
            or any(type(record["transport"][key]) is not type(value) or record["transport"][key] != value
                   for key, value in TRANSPORT.items())):
        raise ValueError(NOT_READY)
    for reference in (record["main_plan_ref"], record["source_packet_ref"], *record["reviews"].values()):
        _ref(reference)
    if (type(record["source_binding"]) is not dict
            or set(record["source_binding"]) != {"helper", "parser", "expected_ft_id"}):
        raise ValueError(NOT_READY)
    for key in ("helper", "parser"):
        _ref(record["source_binding"][key])
    return record


class EconomicGameConsumer(IndependentStateGameConsumer):
    """New science and ownership; inherited native dispatch/ACK code is unchanged.

    The IndependentState subclass is solely for existing claim/isinstance
    compatibility.  Its constructor, source singleton and science methods are
    deliberately bypassed.
    """
    ALLOWED_MODEL_IDS = frozenset(comparison.FUTURE_IDS)

    def __init__(self, pins, *, core_binding, requester_chat, source_binding, admission_ref,
                 source_scope=None, read_bytes=None, read_health=None, clock=None, monotonic=None):
        _historical_scope_absent()
        self.source_binding = copy.deepcopy(source_binding)
        self.admission_ref = copy.deepcopy(admission_ref)
        self._input_budget = None
        self._economic_case_binding = None
        self._economic_request_bindings = {}
        self._read_bytes = read_bytes or models._bounded_metadata_bytes
        self._clock = clock or time.time
        _, raw = _pinned_bytes(admission_ref, models.PROVIDER352_ROOT, self._read_bytes)
        record = _precheck_admission(_json(raw))
        if (record["source_binding"] != source_binding or record["pins"] != pins
                or record["core_binding"] != core_binding or record["owner_chat"] != requester_chat):
            raise ValueError("Economic admission differs from the trusted private handoff")
        registry.assert_candidate_identity()
        if record["actual_artifact_binding"] != registry._CANDIDATE:
            raise ValueError("Economic admission names another actual checkpoint")
        self._helper, self._parser = self._load_source(source_binding)
        # Avoid IndependentStateGameConsumer.__init__: it initializes old globals.
        models.ContinuityTransferGameConsumer.__init__(self, pins, core_binding=core_binding,
            requester_chat=requester_chat, source_scope=copy.deepcopy(source_scope), read_bytes=read_bytes,
            read_health=read_health, clock=clock, monotonic=monotonic)
        self.excluded_seeds = frozenset(record["excluded_seeds"])
        self._scientific_binding = self._validated_scientific_amendment()

    def _load_source(self, binding):
        modules = []
        for key in ("helper", "parser"):
            path, raw = _pinned_bytes(binding[key], models.PROVIDER352_ROOT, self._read_bytes)
            module = ModuleType("research4_economic_" + key + "_" + binding[key]["sha256"])
            module.__file__ = str(path)
            exec(compile(raw, str(path), "exec"), module.__dict__)
            modules.append(module)
        helper, parser = modules
        helper.require_new_static_binding()
        parser._require_static_route_binding()
        candidate = registry._CANDIDATE
        if (getattr(helper, "ECONOMIC_ADMISSION_SCHEMA", None) != ADMISSION_SCHEMA
                or getattr(parser, "ECONOMIC_INPUT_PROTOCOL", None) != economic.PROTOCOL
                or helper.BASE != "qwen3.8-27b-base" or helper.FT != binding["expected_ft_id"]
                or helper.FT != candidate["expected_ft_id"] or parser.LAST != helper.FT
                or tuple(helper.ARMS) != (helper.BASE, helper.FT) or tuple(parser.ARMS) != tuple(helper.ARMS)
                or type(helper.ACTUAL_FIT_UPDATES) is not int or helper.ACTUAL_FIT_UPDATES != registry.EXPECTED_UPDATES
                or type(helper.ACTUAL_TEACHER_DEV_ROWS) is not int or helper.ACTUAL_TEACHER_DEV_ROWS != registry.EXPECTED_DEV_ROWS
                or type(helper.MAX_GAME_REQUESTS) is not int or helper.MAX_GAME_REQUESTS != 3072
                or type(helper.MAX_GAME_WINDOW_SECONDS) is not int or helper.MAX_GAME_WINDOW_SECONDS != 90000
                or helper.FINAL_ADAPTER_SHA != candidate["adapter_sha256"]
                or helper.FINAL_CONFIG_SHA != candidate["adapter_config_sha256"]
                or helper.BASE_SHA != candidate["base_manifest_sha256"]
                or helper.READY_ARTIFACT_REF != candidate["artifact_ready"]
                or parser.READY_ARTIFACT_REF != helper.READY_ARTIFACT_REF
                or helper.COMPLETION_AUDIT_SHA != candidate["completion_audit"]["sha256"]):
            raise ValueError("Economic Game requires new actual612/DEV42 checkpoint-specific source")
        return helper, parser

    def _handoff_contract(self):
        return self._helper

    def _request_parser(self):
        return self._parser

    def _grant_directory(self):
        return REPOSITORY / ".local/economic_grants" / self.admission_ref["sha256"]

    @classmethod
    def from_private_context(cls, raw, **fixtures):
        return consumer_from_context(raw, **fixtures)

    def _main_bytes(self, reference):
        return _pinned_bytes(reference, REPOSITORY, self._read_bytes, local=True)

    def _main_reference(self, reference):
        path, raw = self._main_bytes(reference)
        return path, _json(raw)

    def _validated_scientific_amendment(self):
        """Validate new Root admission without health reads or model construction."""
        _historical_scope_absent()
        _, raw = _pinned_bytes(self.admission_ref, models.PROVIDER352_ROOT, self._read_bytes)
        record = _precheck_admission(_json(raw))
        registry.assert_candidate_identity()
        h = self.contract
        if (record["actual_artifact_binding"] != registry._CANDIDATE
                or record["source_binding"] != self.source_binding or record["pins"] != dict(self.pins)
                or record["core_binding"] != self.core_binding
                or record["owner_chat"] != self.requester_chat or self.requester_chat != h.GAME_CHAT
                or h.epoch(record["issued_utc"]) > self._clock()
                or record["main_plan_ref"]["sha256"] != self.pins["expected_game_plan_sha256"]
                or h.MAIN_12CASE_PLAN_REF["sha256"] != self.pins["expected_game_plan_sha256"]):
            raise ValueError("New economic source, checkpoint, owner or plan admission differs")
        for key in ("helper", "parser"):
            _pinned_bytes(self.source_binding[key], models.PROVIDER352_ROOT, self._read_bytes)
        h.require_new_static_binding()
        self._request_parser()._require_static_route_binding()
        _, core = self._reference(self.core_binding["core_binding"])
        h.validate_core(core, self.core_binding["trusted_review_sha256"])
        identity = core["per_model"][h.FT]
        if any(identity[key] != registry._CANDIDATE[key] for key in ("adapter_sha256", "adapter_config_sha256")):
            raise ValueError("Economic qualified Core names another checkpoint")
        _, index = self._reference({"path": self.pins["index_path"], "sha256": self.pins["expected_index_sha256"]})
        _, window = self._reference(index["request_window"])
        if (index["core_binding"] != self.core_binding["core_binding"]
                or index["request_window"]["sha256"] != self.pins["expected_window_sha256"]
                or not h.validate_window(window) or window["core_binding"] != self.core_binding["core_binding"]
                or window["game_plan_sha256"] != self.pins["expected_game_plan_sha256"]
                or window["grant_nonce"] != self.pins["expected_nonce"]
                or window["original_direct_authority"]["sha256"] != self.pins["expected_direct_authority_sha256"]
                or type(window.get("max_game_requests")) is not int or window["max_game_requests"] != 3072
                or not h.epoch(record["issued_utc"]) <= self._clock() < h.epoch(window["expires_utc"])
                or not h.epoch(window["issued_utc"]) <= self._clock()
                or not 0 < h.epoch(window["expires_utc"]) - h.epoch(window["issued_utc"]) <= 90000):
            raise ValueError("Fresh finite3072 economic window is absent, changed or expired")
        _, plan = self._main_reference(record["main_plan_ref"])
        self._validate_plan(plan, record["excluded_seeds"])
        _, packet = self._main_reference(record["source_packet_ref"])
        expected = {"schema": SOURCE_PACKET_SCHEMA,
            "interface_runtime_source_sha256": comparison._digests(comparison.RUNTIME_SOURCES),
            "export_source_sha256": comparison._digests(comparison.EXPORT_SOURCES)}
        if packet != expected:
            raise ValueError("Economic runtime or both publishers differ from Root source packet")
        _, review = self._reference(record["reviews"]["Root_transport_ref"])
        _, main_review = self._main_reference(record["reviews"]["Main_source_ref"])
        if any(type(row.get("status")) is not str or not row["status"].startswith("PASS") for row in (review, main_review)):
            raise ValueError("Economic source and transport review evidence is absent")
        self._scientific_binding = copy.deepcopy(record)
        return record

    def _validate_plan(self, plan, exclusions):
        if (plan.get("schema") != PLAN_SCHEMA or plan.get("execution_enabled") is not False
                or any(key in plan for key in ("transport_parent_plan", "completed_prefix_ref", "imported_cases"))):
            raise ValueError("New economic held plan must contain no imported historical case")
        comparison.validate_cohort(plan, excluded_seeds=frozenset(exclusions))
        reference = {"path": plan.get("configuration_path"), "sha256": plan.get("configuration_sha256")}
        _, config = self._main_reference(reference)
        if (set(config) != {"schema", "prospective_protocol"} or config["schema"] != CONFIG_SCHEMA
                or config["prospective_protocol"] != plan.get("prospective_protocol")):
            raise ValueError("Economic configuration or protocol binding changed")
        self._main_bytes(config["prospective_protocol"])
        return plan

    def require_plan(self, parent_plan_sha256):
        record = self._validated_scientific_amendment()
        if parent_plan_sha256 != record["main_plan_ref"]["sha256"]:
            self.stop()
            raise ValueError("Economic PLAN differs from fresh Root admission")

    def private_child_context(self, parent_plan_sha256, *, source_scope):
        self.require_plan(parent_plan_sha256)
        record = self._scientific_binding
        path, _ = self._main_reference(record["main_plan_ref"])
        if (type(source_scope) is not dict or set(source_scope) != {"plan_path", "batch_index"}
                or type(source_scope["plan_path"]) is not str or source_scope["plan_path"] != str(path)
                or type(source_scope["batch_index"]) is not int or not 0 <= source_scope["batch_index"] < 12):
            raise ValueError("Exact economic PLAN path and case scope required")
        return json.dumps({"pins": dict(self.pins), "requester_chat": self.requester_chat,
            "core_binding": self.core_binding, "source_scope": source_scope,
            "source_binding": self.source_binding, "admission_ref": self.admission_ref},
            sort_keys=True, separators=(",", ":"))

    def _validated_request(self, requested_model_id, **kwargs):
        try:
            self._validated_scientific_amendment()
            if self.source_scope is not None:
                self._validate_game_source_scope()
            result = models.ContinuityTransferGameConsumer._validated_request(self, requested_model_id, **kwargs)
            self._validated_scientific_amendment()
            return result
        except Exception:
            self.stop()
            raise

    def _validate_game_source_scope(self):
        record = self._validated_scientific_amendment()
        path, plan = self._main_reference(record["main_plan_ref"])
        scope = self.source_scope
        if (type(scope) is not dict or set(scope) != {"plan_path", "batch_index"}
                or type(scope["plan_path"]) is not str or scope["plan_path"] != str(path)
                or path.name != "PLAN.json" or type(scope["batch_index"]) is not int
                or not 0 <= scope["batch_index"] < 12):
            raise ValueError("Exact economic child source scope required")
        grant = self.pins["expected_window_sha256"]
        directory = self._grant_directory()
        if (directory / (grant + ".STOP.json")).exists():
            raise ValueError("Economic collection stopped")
        claim = _json(models._bounded_metadata_bytes(directory / (grant + ".CLAIM.json")))
        expected = {"window_sha256": grant, "game_plan_sha256": record["main_plan_ref"]["sha256"],
                    "destination": str(path.parent), "initial_phase": "all"}
        if claim != expected:
            raise ValueError("Economic finite grant claim differs from this new plan")
        return plan["batches"][scope["batch_index"]]

    def validate_child_plan(self, child):
        from .company_preservation_comparison import validate_batch_manifest
        self._child_plan_verified = False
        try:
            batch = self._validate_game_source_scope()
            validate_batch_manifest(child, batch)
            keys = {"history_mode", "prompt_protocol", "input_view_protocol", "user_task_instruction",
                "initial_user_payload_sha256", "world_initial_user_payload_sha256", "user_task_instruction_sha256",
                "history_reasoning_carrier", "interface_runtime_source_sha256", "export_source_sha256",
                "additional_icl_system_instruction", "additional_icl_system_instruction_sha256", "format_system_instruction",
                "response_instruction", "samples"}
            if any(child.get(key) != batch.get(key) for key in keys):
                raise ValueError("Economic child carrier/source identity differs")
            self._child_plan_verified = True
        except Exception:
            self.stop()
            raise ValueError("Economic child/source differs from frozen new cohort") from None

    def _validate_wire(self, core, request, spec):
        models.Security500GameConsumer._validate_wire(self, core, request, spec)
        batch = self._validate_game_source_scope()
        role = batch["models"][0]
        if spec != models.MODEL_SPECS[role]:
            raise ValueError("Selected native role differs from the frozen economic case")
        self._wire_messages = copy.deepcopy(self._request_parser().parse_request(bytes(request.content))["messages"])
        payload = comparison.validate_wire(batch, self._wire_messages, first_request=self._attempts == 0)
        decision = payload["observation"].get("decision_count")
        case_id = batch["samples"][0].get("case_id")
        if (type(decision) is not int or not 0 <= decision < 256 or decision != self._attempts
                or type(case_id) is not str or not case_id):
            raise ValueError("Economic wire requires the exact nonnegative current case decision")
        self._economic_case_binding = {"case_id": case_id, "decision": decision, "attempt": 1}
        self._input_budget = {"prompt_tokens": None, "output_reservation_tokens": 8192,
            "context_total_tokens": 131072, "maximum_prompt_tokens": 122880,
            "enforcement": "pinned_resident_native_pre_enqueue_tokenizer_no_truncation"}

    def _record_request_evidence(self, event):
        if self.source_scope is None:
            return
        generation_identity = None
        if event.get("event") == "HTTP_ATTEMPT":
            request_id, digest = event.get("request_id"), event.get("messages_sha256")
            messages = getattr(self, "_wire_messages", None)
            actual_digest = hashlib.sha256(json.dumps(messages, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            if (type(request_id) is not str or re.fullmatch(r"[0-9a-f]{32}", request_id) is None
                    or request_id in self._economic_request_bindings or digest != actual_digest
                    or self._economic_case_binding is None):
                raise ValueError("Economic HTTP identity differs from the actual validated wire")
            generation_identity = copy.deepcopy(self._economic_case_binding)
            self._economic_request_bindings[request_id] = (digest, generation_identity)
        elif event.get("event") == "POST_REQUEST_VALIDATED":
            binding = self._economic_request_bindings.get(event.get("request_id"))
            if binding is None or event.get("messages_sha256") != binding[0]:
                raise ValueError("Economic native completion has no exact recorded generation identity")
            generation_identity = copy.deepcopy(binding[1])
        physical = event.get("fresh_health", {}).get("last_request_evidence")
        if type(physical) is dict and self._input_budget is not None:
            self._input_budget.update(prompt_tokens=physical.get("prompt_tokens"), completion_tokens=physical.get("completion_tokens"))
        directory = Path(self.source_scope["plan_path"]).parent / f"batch_{self.source_scope['batch_index'] + 1:02d}"
        row = {"schema": "research4.economic-game-client-request-evidence.v1",
            "window_sha256": self.pins["expected_window_sha256"],
            "scientific_plan_sha256": self.pins["expected_game_plan_sha256"],
            "economic_admission_sha256": self.admission_ref["sha256"],
            "operational_source_binding": self.source_binding, **event}
        if generation_identity is not None:
            row["generation_identity"] = generation_identity
        if event.get("event") == "HTTP_ATTEMPT" and hasattr(self, "_wire_messages"):
            row["exact_serialized_messages"] = self._wire_messages
        if self._input_budget is not None:
            row["input_budget"] = self._input_budget
        with (directory / "REQUEST_GUARD_EVIDENCE.jsonl").open("a", encoding="utf-8") as stream:
            json.dump(row, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def consumer_from_context(raw, **fixtures):
    """Rebuild only this candidate in a fresh child; no old private environment."""
    _historical_scope_absent()
    context = _json(raw)
    if set(context) != {"pins", "requester_chat", "core_binding", "source_scope", "source_binding", "admission_ref"}:
        raise ValueError("Closed economic private context required")
    reader = fixtures.get("read_bytes")
    _, admission_raw = _pinned_bytes(context["admission_ref"], models.PROVIDER352_ROOT, reader)
    record = _precheck_admission(_json(admission_raw))
    if (record["pins"] != context["pins"] or record["core_binding"] != context["core_binding"]
            or record["source_binding"] != context["source_binding"] or record["owner_chat"] != context["requester_chat"]):
        raise ValueError("Economic private context differs from Root admission")
    _, plan_raw = _pinned_bytes(record["main_plan_ref"], REPOSITORY, reader, local=True)
    plan = _json(plan_raw)
    candidates = [batch.get("generation_settings", {}).get(role, {}).get("economic_identity_candidate")
                  for batch in plan.get("batches", []) for role in comparison.FUTURE_IDS
                  if batch.get("models") == [role]]
    if len(candidates) != 12 or any(value != record["actual_artifact_binding"] for value in candidates):
        raise ValueError("Economic child requires the new candidate in all twelve frozen cells")
    registry.register_candidate(record["actual_artifact_binding"], read_bytes=reader)
    return EconomicGameConsumer(context["pins"], core_binding=context["core_binding"],
        requester_chat=context["requester_chat"], source_scope=context["source_scope"],
        source_binding=context["source_binding"], admission_ref=context["admission_ref"], **fixtures)
