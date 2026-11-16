"""Fresh per-checkpoint registration; reuse the reviewed two-route transport.

Artifact registration is planning metadata. Only a separately pinned, statically
bound Root helper/Core/window can admit a request. Historical roles are untouched.
"""
from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from types import ModuleType
from dateutil.tz import gettz

from . import company_models as m

INDEPENDENT_STATE_IDS = ("independent_state_base_neutral", "independent_state_base_operational_icl",
                         "independent_state_ft_neutral", "independent_state_ft_operational_icl")
INDEPENDENT_STATE_CONTEXT_ENV = "RESEARCH4_GAME_INDEPENDENT_STATE_CONTEXT"
HELD_PUBLIC_PATH = ".local/staging/independent_state_two_route_held_20261116/HOLD_PUBLIC_ROUTE_CONTRACT_v1.json"
HELD_PUBLIC_SHA = "3a554cd845b40adb71cb475a098ef93254423643bd31f9aad6e8d706520127c7"
NOT_READY = "Independent-state HELD: new actual artifact, qualified Core and finite3072 request grant required"
_CANDIDATE = None
_SPECS = None
_SOURCE = None
_MODULES = None
_CORE_REF = None
_CORE_RAW = None
_ACTIVE_KEY = None
_ACTIVE = None


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _ref(value):
    return (type(value) is dict and set(value) == {"path", "sha256"}
            and type(value["path"]) is str and bool(value["path"])
            and "\\" not in value["path"] and ":" not in value["path"] and _sha(value["sha256"]))


def _bytes(reference, reader=None):
    if not _ref(reference):
        raise ValueError("Exact Root-relative source/evidence reference required")
    relative = Path(reference["path"])
    path = (m.PROVIDER352_ROOT / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(m.PROVIDER352_ROOT.resolve()):
        raise ValueError("Evidence/source must remain beneath canonical Root")
    raw = (reader or m._bounded_metadata_bytes)(path)
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("Pinned evidence/source bytes changed")
    return raw, path


def _module(reference, reader=None):
    raw, path = _bytes(reference, reader)
    value = ModuleType("research4_independent_state_" + reference["sha256"])
    value.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), value.__dict__)
    return value


def register_independent_state_candidate(binding, *, read_bytes=None):
    """Register four exact planning roles from reviewed artifact metadata only."""
    global _CANDIDATE, _SPECS
    fields = {"expected_ft_id", "adapter_sha256", "adapter_config_sha256", "base_manifest_sha256",
              "artifact_ready", "ready_utc", "completion_audit", "fit_updates", "teacher_dev_rows"}
    if (type(binding) is not dict or set(binding) != fields
            or any(not _sha(binding[key]) for key in ("adapter_sha256", "adapter_config_sha256", "base_manifest_sha256"))
            or binding["base_manifest_sha256"] != m.LOCAL_BASE_MANIFEST_SHA256
            or type(binding["fit_updates"]) is not int or binding["fit_updates"] != 168
            or type(binding["teacher_dev_rows"]) is not int or binding["teacher_dev_rows"] != 24
            or type(binding["expected_ft_id"]) is not str
            or not re.fullmatch(r"sp-[0-9]{8}-[0-9]{2}-[0-9a-f]{8,64}", binding["expected_ft_id"])):
        raise ValueError(NOT_READY)
    route = binding["expected_ft_id"]
    if not binding["adapter_sha256"].startswith(route.rsplit("-", 1)[1]):
        raise ValueError("New FT ID must name the actual checkpoint prefix")
    if type(binding["ready_utc"]) is not str:
        raise ValueError("Exact UTC artifact READY instant required")
    ready = datetime.fromisoformat(binding["ready_utc"].replace("Z", "+00:00"))
    alberta = gettz("America/Edmonton")
    if (ready.tzinfo is None or ready.utcoffset().total_seconds() != 0 or alberta is None
            or ready.astimezone(alberta).strftime("sp-%Y%m%d-%H-") != route[:15]):
        raise ValueError("New FT ID must use the artifact READY Alberta date/hour")
    for key in ("artifact_ready", "completion_audit"):
        raw, _ = _bytes(binding[key], read_bytes)
        if type(json.loads(raw)) is not dict:
            raise ValueError("Reviewed artifact metadata must be a JSON object")
    if _CANDIDATE is not None:
        if binding != _CANDIDATE:
            raise ValueError("Independent-state candidate cannot be retargeted in this process")
        assert_candidate_identity()
        return copy.deepcopy(_CANDIDATE)
    specs = {}
    for index, name in enumerate(INDEPENDENT_STATE_IDS):
        ft, icl = index >= 2, index % 2 == 1
        spec = copy.deepcopy(m.MODEL_SPECS["qwen_base_neutral"])
        spec.update(model="openai-api/research4-qwen/" + (route if ft else "qwen3.8-27b-base"),
            service_model_id=route if ft else "qwen3.8-27b-base", arm_id=name,
            label=(route if ft else "Qwen3.8-27B BASE") + (" + operational ICL" if icl else " · neutral"),
            prompt_condition="operational_preservation_icl" if icl else "baseline",
            adapter_applied=ft, max_output_tokens=8192, required_context_tokens=131072,
            required_history_mode="full_chronological_v1", max_connections=1,
            request_timeout_seconds=900, execution_enabled=False,
            independent_state_candidate=copy.deepcopy(binding), independent_state_readiness="ARTIFACT_ONLY_HELD")
        if ft:
            spec.update(adapter_sha256=binding["adapter_sha256"], adapter_config_sha256=binding["adapter_config_sha256"],
                        adapter_rank=16, adapter_blocks=list(range(56, 64)), adapter_projection_count=62,
                        adapter_dtype="F32", completed_training_updates=168)
        specs[name] = spec
    m.MODEL_SPECS.update(specs)
    from .hosted_inspect_runner import LOCAL_QWEN_MODELS
    LOCAL_QWEN_MODELS.add("openai-api/research4-qwen/" + route)
    _CANDIDATE, _SPECS = copy.deepcopy(binding), copy.deepcopy(specs)
    return copy.deepcopy(_CANDIDATE)


def assert_candidate_identity():
    if _CANDIDATE is None or any(m.MODEL_SPECS.get(key) != value for key, value in _SPECS.items()):
        raise ValueError(NOT_READY)


def is_independent_state_selected(models=()):
    return bool(os.environ.get(INDEPENDENT_STATE_CONTEXT_ENV) or set(models) & set(INDEPENDENT_STATE_IDS))


def initialize_independent_state_source(binding, *, read_bytes=None):
    """Late source admission, separate from the already frozen scientific PLAN."""
    global _SOURCE, _MODULES
    if (type(binding) is not dict or set(binding) != {"helper", "parser", "public_contract", "expected_ft_id"}
            or binding["public_contract"] != {"path": HELD_PUBLIC_PATH, "sha256": HELD_PUBLIC_SHA}):
        raise ValueError("Pinned new Root source descriptor required")
    _bytes(binding["public_contract"], read_bytes)
    h, parser = _module(binding["helper"], read_bytes), _module(binding["parser"], read_bytes)
    h.require_new_static_binding()
    parser._require_static_route_binding()
    if (h.BASE != "qwen3.8-27b-base" or h.FT != binding["expected_ft_id"] or parser.LAST != h.FT
            or tuple(h.ARMS) != (h.BASE, h.FT) or tuple(parser.ARMS) != tuple(h.ARMS)
            or h.ACTUAL_FIT_UPDATES != 168 or h.ACTUAL_TEACHER_DEV_ROWS != 24
            or h.MAX_GAME_REQUESTS != 3072 or h.MAX_GAME_WINDOW_SECONDS != 90000
            or parser.READY_ARTIFACT_REF != h.READY_ARTIFACT_REF):
        raise ValueError("Exact fresh168/DEV24 two-route source and limits required")
    assert_candidate_identity()
    expected = _CANDIDATE
    if (h.FT != expected["expected_ft_id"] or h.FINAL_ADAPTER_SHA != expected["adapter_sha256"]
            or h.FINAL_CONFIG_SHA != expected["adapter_config_sha256"] or h.BASE_SHA != expected["base_manifest_sha256"]
            or h.READY_ARTIFACT_REF != expected["artifact_ready"]
            or h.COMPLETION_AUDIT_SHA != expected["completion_audit"]["sha256"]):
        raise ValueError("Operational source differs from the frozen candidate")
    if _SOURCE is not None and _SOURCE != binding:
        raise ValueError("Operational source binding cannot change in this process")
    _SOURCE, _MODULES = copy.deepcopy(binding), (h, parser)
    return h


def initialize_independent_state_core(binding, *, source_binding=None, read_bytes=None):
    global _CORE_REF, _CORE_RAW
    h = initialize_independent_state_source(source_binding, read_bytes=read_bytes) if source_binding is not None else (
        _MODULES[0] if _MODULES is not None else None)
    if (h is None or type(binding) is not dict or set(binding) != {"core_binding", "trusted_review_sha256"}
            or not h.ref(binding["core_binding"]) or not h.is_sha(binding["trusted_review_sha256"])):
        raise ValueError(NOT_READY)
    if (any(os.environ.get(key) for key in (m.CONTINUITY_TRANSFER_CONTEXT_ENV, m.SECURITY500_CONTEXT_ENV,
            m.PROVIDER352_CONTEXT_ENV, m.SCIENTIFIC_AMENDMENT_ENV))
            or any(value is not None for value in (m._CONTINUITY_TRANSFER_CORE_REF, m._SECURITY500_CORE_REF, m._PROVIDER352_CORE_REF))):
        raise ValueError("Fresh independent-state and historical transport scopes cannot mix")
    raw, _ = _bytes(binding["core_binding"], read_bytes)
    core = h.raw_json(raw, binding["core_binding"]["sha256"])
    h.validate_core(core, binding["trusted_review_sha256"])
    if _CORE_REF is not None and _CORE_REF != binding:
        raise ValueError("Qualified Core cannot change in this process")
    _CORE_REF, _CORE_RAW = copy.deepcopy(binding), raw
    return core


def independent_state_planning_core_ref():
    return copy.deepcopy(_CORE_REF)


def _context(raw):
    context = json.loads(raw)
    if type(context) is not dict or set(context) != {"pins", "requester_chat", "core_binding", "source_scope", "source_binding"}:
        raise ValueError("Closed fresh independent-state private context required")
    return context


def initialize_private_core():
    raw = os.environ.get(INDEPENDENT_STATE_CONTEXT_ENV)
    if raw:
        context = _context(raw)
        # Child registration uses candidate metadata frozen in the parent PLAN.
        plan = json.loads(m._bounded_metadata_bytes(Path(context["source_scope"]["plan_path"])))
        ft_batch = next(batch for batch in plan["batches"] if batch["models"] == [INDEPENDENT_STATE_IDS[2]])
        register_independent_state_candidate(ft_batch["generation_settings"][INDEPENDENT_STATE_IDS[2]]["independent_state_candidate"])
        initialize_independent_state_core(context["core_binding"], source_binding=context["source_binding"])


def current_core(required_models=None):
    initialize_private_core()
    assert_candidate_identity()
    if _CORE_RAW is None or (required_models is not None and any(key not in INDEPENDENT_STATE_IDS for key in required_models)):
        raise ValueError(NOT_READY)
    return json.loads(_CORE_RAW)


def independent_state_active_consumer():
    global _ACTIVE, _ACTIVE_KEY
    raw = os.environ.get(INDEPENDENT_STATE_CONTEXT_ENV)
    if not raw:
        raise ValueError(NOT_READY)
    if _ACTIVE is not None:
        if _ACTIVE_KEY != raw:
            _ACTIVE.stop()
            raise ValueError("Fresh private transport scope changed")
        return _ACTIVE
    initialize_private_core()
    context = _context(raw)
    _ACTIVE = IndependentStateGameConsumer(context["pins"], core_binding=context["core_binding"],
        requester_chat=context["requester_chat"], source_scope=context["source_scope"], source_binding=context["source_binding"])
    _ACTIVE_KEY = raw
    return _ACTIVE


class IndependentStateGameConsumer(m.ContinuityTransferGameConsumer):
    """Only new identity/source/scope; generation/ACK/budget code is inherited."""
    ALLOWED_MODEL_IDS = frozenset(INDEPENDENT_STATE_IDS)

    def __init__(self, *args, source_binding, **kwargs):
        self.source_binding = copy.deepcopy(source_binding)
        self._input_budget = None
        initialize_independent_state_source(source_binding, read_bytes=kwargs.get("read_bytes"))
        super().__init__(*args, **kwargs)

    def _handoff_contract(self):
        return _MODULES[0]

    def _request_parser(self):
        return _MODULES[1]

    def _grant_directory(self):
        return Path(__file__).resolve().parents[1] / ".local/is_grants"

    def private_child_context(self, parent_plan_sha256, *, source_scope):
        value = json.loads(super().private_child_context(parent_plan_sha256, source_scope=source_scope))
        value["source_binding"] = self.source_binding
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def require_plan(self, parent_plan_sha256):
        super().require_plan(parent_plan_sha256)
        if self.contract.MAIN_12CASE_PLAN_REF["sha256"] != parent_plan_sha256:
            self.stop()
            raise ValueError("New static helper is bound to another twelve-case PLAN")

    def _validate_game_source_scope(self):
        # Reuse the direct PLAN/claim/STOP/config/runtime-source checks, bypassing
        # only the historical six-neutral-case specialization.
        if self.source_binding != _SOURCE:
            raise ValueError("Operational source descriptor changed")
        batch = m.Security500GameConsumer._validate_game_source_scope(self)
        from .infrastructure_preservation_comparison import (INDEPENDENT_STATE_GAME_SCHEMA,
            INDEPENDENT_STATE_EXECUTION_ORDER, FULL_HISTORY_USER_TASK)
        from .company_world_evaluation import source_matches
        frozen = json.loads(m._bounded_metadata_bytes(Path(self.source_scope["plan_path"])))
        self.require_plan(self.pins["expected_game_plan_sha256"])
        import yaml
        config = yaml.safe_load(Path(frozen["configuration_path"]).read_bytes())
        protocol = config.get("prospective_protocol")
        if (type(protocol) is not dict or set(protocol) != {"path", "sha256"}
                or not _sha(protocol["sha256"])
                or hashlib.sha256(Path(protocol["path"]).read_bytes()).hexdigest() != protocol["sha256"]):
            raise ValueError("Prospective scientific protocol bytes differ")
        if (frozen.get("protocol") != INDEPENDENT_STATE_GAME_SCHEMA or frozen.get("models") != list(INDEPENDENT_STATE_IDS)
                or frozen.get("history_mode") != "full_chronological_v1"
                or frozen.get("user_task_instruction") != FULL_HISTORY_USER_TASK
                or frozen.get("generation_profile_sha256") != m.CONTINUITY_TRANSFER_GENERATION_PROFILE_SHA
                or frozen.get("generation_seed") != 20261108 or len(frozen.get("batches", [])) != 12
                or frozen.get("transport_max_total_native_dispatches") != 3072
                or frozen.get("transport_max_additional_native_dispatches") != 3072):
            raise ValueError("Exact fresh3072 four-condition full-history PLAN required")
        for row, (world, name) in zip(frozen["batches"], INDEPENDENT_STATE_EXECUTION_ORDER):
            if (row["models"] != [name] or row["seed"] != frozen["seeds"][world]
                    or row.get("history_mode") != "full_chronological_v1"
                    or row.get("prompt_protocol") != "company_world_full_chronological_v1"
                    or row.get("user_task_instruction") != FULL_HISTORY_USER_TASK
                    or row["horizon_hours"] != 48 or row["max_decisions"] != 256
                    or row["outer_seconds"] != 14400 or not source_matches(row)):
                raise ValueError("Fresh12 case/order/history/source differs")
        assert_candidate_identity()
        return batch

    validate_child_plan = m.ScientificSuccessorGameConsumer.validate_child_plan

    def _validate_wire(self, core, request, spec):
        super()._validate_wire(core, request, spec)
        self._wire_messages = copy.deepcopy(self._request_parser().parse_request(bytes(request.content))["messages"])
        batch = self._validate_game_source_scope()
        if (len(self._wire_messages) != 2 or [row["role"] for row in self._wire_messages] != ["system", "user"]
                or not self._wire_messages[1]["content"].startswith("FULL CHRONOLOGICAL TRANSCRIPT v1\n")
                or self._wire_messages[0]["content"] != batch["samples"][0]["system_prompt"]
                or (self._attempts == 0 and hashlib.sha256(self._wire_messages[1]["content"].encode()).hexdigest()
                    != batch["initial_user_payload_sha256"])):
            raise ValueError("Exact frozen full-history USER carrier/system/initial payload required")
        self._input_budget = {"prompt_tokens": None, "output_reservation_tokens": 8192,
            "context_total_tokens": 131072, "maximum_prompt_tokens": 122880,
            "enforcement": "pinned_resident_native_pre_enqueue_tokenizer_no_truncation"}

    def _record_request_evidence(self, event):
        physical = event.get("fresh_health", {}).get("last_request_evidence")
        if type(physical) is dict and self._input_budget is not None:
            self._input_budget.update(prompt_tokens=physical.get("prompt_tokens"), completion_tokens=physical.get("completion_tokens"))
        super()._record_request_evidence({**event, "scientific_plan_sha256": self.pins["expected_game_plan_sha256"],
            "operational_source_binding": self.source_binding,
            **({"exact_serialized_messages": self._wire_messages} if event.get("event") == "HTTP_ATTEMPT" and hasattr(self, "_wire_messages") else {}),
            **({"input_budget": self._input_budget} if self._input_budget is not None else {})})
