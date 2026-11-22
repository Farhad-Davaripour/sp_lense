"""Model choices and exact routing, independent of the operator's test subset."""
from __future__ import annotations

import hashlib
import copy
import contextvars
import asyncio
import json
import math
import os
import secrets
import time
from pathlib import Path
from types import MappingProxyType, ModuleType
from urllib.parse import urlparse
import urllib.request

from .continuity_arena import build_model
from . import qwen_cloud
from .research_paths import fine_tune_root

QWEN_BASE_URL = "http://127.0.0.1:8769/v1"
LOCAL_BINDING_PATH = Path("C:/Users/farha/Research4/artifacts/last_layer_icl_pilot_20261109/SERVICE_BINDING.json")
LAST8_BINDING_PATH = Path(__file__).resolve().parents[1] / ".local/company_preservation_48h_20261111/SERVICE_BINDING.json"
RESERVATION_PATH = Path(__file__).resolve().parents[1] / ".local/qwen_game_reservation.json"
LAST_SHA = "b0a1eb1a935d0dac71b35384e3dfd8638ef88f82a1f84da54c0987ece7fbf785"
LAST8_SHA = "d050d018d8311aaf941e837bd93b6effc7261e3e0890b7f471c851119f87929c"
LAST8_CONFIG_SHA = "476b26d8530c275d934f5a920923885b893be3e55c63d47f4850537d1781ea9d"
LAST8_SERVICE_MODEL_ID = "sp-lense-adaptor-last-8-layers"
LAST8_OUTPUT_TOKENS = 8192
LAST8_CONTEXT_TOKENS = 131072
RETRAINED_SERVICE_MODEL_ID = "sp-lense-adaptor-last-8-layers-retrained"
# Completed compact 3.3.1 fit and qualified BASE/OLD/NEW serving handoff.
RETRAINED_SHA = "9a4538c3bc9bea9b9a2db206b625bbbad63f1b41ef6a48d44fef1812bf51c695"
RETRAINED_CONFIG_SHA = "42d92a0014893d4c527b820a8d207bbeff920877ec02883f9c482e7919a8e7cb"
COMPARISON_BINDING_SCHEMA = "research4.game-preservation-service-binding.v2"
PROVIDER352_SERVICE_MODEL_ID = "sp-lense-adaptor-last-8-layers-provider352"
PROVIDER352_BINDING_SCHEMA = "research4.game-preservation-service-binding.provider352.v1"
PROVIDER352_NOT_READY_REASON = (
    "Provider3.5.2 NOT_READY: an independently trusted qualified four-route core "
    "and finite Root request grant are required; partial180 cannot qualify"
)
PROVIDER352_ROOT = fine_tune_root()
PROVIDER352_HANDOFF_PATH = PROVIDER352_ROOT / ".local/staging/provider352_game_handoff_postrelease_v2_20261114/handoff_contract.py"
PROVIDER352_HANDOFF_SHA = "325b45d678da19c8e796935976c0a60422104fde82a9473f104fc6b08cb5ac20"
PROVIDER352_CONTEXT_ENV = "RESEARCH4_GAME_PROVIDER352_PINNED_CONTEXT"
SECURITY500_CONTEXT_ENV = "RESEARCH4_GAME_SECURITY500_PINNED_CONTEXT"
SECURITY500_SERVICE_MODEL_ID = "sp-lense-agent-security500-last8"
SECURITY500_MODEL = "openai-api/research4-qwen/" + SECURITY500_SERVICE_MODEL_ID
# Frozen source candidate only; actual core, review and request pins remain unbound.
SECURITY500_HANDOFF_PATH = PROVIDER352_ROOT / ".local/staging/agent_security500_two_route_qualification_held_v4_runtimefix_20261115/two_route_handoff_contract.py"
SECURITY500_HANDOFF_SHA = "800366ff119b3523260a9f86ccd2ab0eb2d65c2ea30ed81038fb16251d9b68e4"
SECURITY500_NOT_READY = "Security500 NOT_READY: independent qualified core and finite Root grant required"
SECURITY500_GRANT_DIR = Path(__file__).resolve().parents[1] / ".local/security500_fresh48h_20261115/readiness/security500_grants"
_SECURITY500_CORE_REF = None
_SECURITY500_CORE_RAW = None
_SECURITY500_CORE_SPECS = None
_SECURITY500_ACTIVE_CONTEXT = None
_SECURITY500_ACTIVE_CONSUMER = None
# Separate completed264 candidate. Public source pins are not a live core/grant.
CONTINUITY_TRANSFER_CONTEXT_ENV = "RESEARCH4_GAME_CONTINUITY_TRANSFER_PINNED_CONTEXT"
CONTINUITY_TRANSFER_SERVICE_MODEL_ID = "sp-lense-continuity-transfer-last8-20261115"
CONTINUITY_TRANSFER_MODEL = "openai-api/research4-qwen/" + CONTINUITY_TRANSFER_SERVICE_MODEL_ID
CONTINUITY_TRANSFER_HANDOFF_PATH = PROVIDER352_ROOT / ".local/staging/continuity_transfer_endpoint_held_20261115/two_route_handoff_contract_v4.py"
CONTINUITY_TRANSFER_HANDOFF_SHA = "092a4e8ce4b252708947c452e3af5f31d30a8d8327ba52f49d63c1670ffc6223"
CONTINUITY_TRANSFER_PUBLIC_PATH = CONTINUITY_TRANSFER_HANDOFF_PATH.with_name("PUBLIC_ENDPOINT_INTERFACE_v4.json")
CONTINUITY_TRANSFER_PUBLIC_SHA = "a8b46f346e6c326a42f9f307fadfdc004aa42328c6c20408e9f7632790a70269"
CONTINUITY_TRANSFER_PARSER_PATH = CONTINUITY_TRANSFER_HANDOFF_PATH.parent / "candidate_source_v2/local_last8_endpoint_contract.py"
CONTINUITY_TRANSFER_PARSER_SHA = "a7a7ca7fd5114da28b93bfb87e2e3ef370578f8df5320b07b0d84f85e8ba6920"
CONTINUITY_TRANSFER_GENERATION_PROFILE_SHA = "e448e36f0ea7cfdcbcf411ea6c724f5053606b7d008640bcabda30ca276c2ef5"
SCIENTIFIC_AMENDMENT_ENV = "RESEARCH4_GAME_SCIENTIFIC_AMENDMENT_REF"
SCIENTIFIC_AMENDMENT_TEMPLATE = PROVIDER352_ROOT / ".local/serving/full_history_successor_20261115/ROOT_SCIENTIFIC_SUCCESSOR_AMENDMENT_HELD_TEMPLATE_v2.json"
SCIENTIFIC_AMENDMENT_TEMPLATE_SHA = "dc6bc516abb45b228ac8896e775607a5eb109ff2aac5024763f26d528dfa7e82"
SCIENTIFIC_ICL_AUTHORITY_SHA = "51ed2091bca424050f64e3e7c35711d5ffb3f7e337290a78d65ad01e1fff6238"
CONTINUITY_TRANSFER_FULL_HISTORY_IDS = ("qwen_base_neutral", "qwen_base_operational_icl",
    "sp_lense_transfer_neutral", "sp_lense_transfer_operational_icl")
CONTINUITY_TRANSFER_NOT_READY = (
    "Continuity transfer HELD: actual qualified264 core and finite Game grant required; "
    "same-request ACK/KV-clear/BASE-restoration proof precedes every action"
)
CONTINUITY_TRANSFER_GRANT_DIR = Path(__file__).resolve().parents[1] / ".local/continuity_transfer_game_20261115/readiness/grants"
_CONTINUITY_TRANSFER_CORE_REF = None
_CONTINUITY_TRANSFER_CORE_RAW = None
_CONTINUITY_TRANSFER_CORE_SPECS = None
_CONTINUITY_TRANSFER_ACTIVE_CONTEXT = None
_CONTINUITY_TRANSFER_ACTIVE_CONSUMER = None
# Accepted technical identities are expectations, not serving or request authority.
PROVIDER352_TECHNICAL_PINS = MappingProxyType({
    "adapter": "227dbfad18aa5d2f95cb2a953bf8e7f1477d5f469313de33348bec9540bdc659",
    "adapter_config": "7e6aa3512384a79c10ec7bf6bb777c14be7bf9bc499e9c3f86f02379915a35b3",
    "completion_audit": "d9ac268723614db8a1334fd9dffe90e1a6dde4a28d3f74c4e88f0f9a60d2e532",
    "final_source_freeze": "6fa4b6f07d7ed277a7d5359690124858b71555e878558bf400244ca8621e1c8b",
    "final_base_proof": "f13219d7445d3f600d893050d189aa8493da535219cfcdc1c38c642d937e91ac",
    "fit_release": "467aeb46f05da83d4932f76f7ac8eb15a7dcdfd10866958e66ef97ad0a5bad07",
    "completion_auditor": "79710b96753e119d33ccfbbbead0bede9a797e18995832ffe8b55ea98c804610",
})
_PROVIDER352_CORE_REF = None
_PROVIDER352_CORE_RAW = None
_PROVIDER352_CORE_SPECS = None
_PROVIDER352_ACTIVE_CONTEXT = None
_PROVIDER352_ACTIVE_CONSUMER = None
LOCAL_BASE_REVISION = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
LOCAL_BASE_MANIFEST_SHA256 = "e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461"
LOCAL_CONFIGURATION = {
    "backend": "local_mlx",
    "base_repository": "mlx-community/Qwen3.8-27B-4bit",
    "base_revision": LOCAL_BASE_REVISION,
    "base_manifest_sha256": LOCAL_BASE_MANIFEST_SHA256,
    "base_quantization": "4bit affine/group64; not NF4",
    "configuration_evidence": "retained configuration only; fresh bound health readback required before execution",
    "configuration_receipt_sha256": "a798021a2c502c0b047bf4d023ad4becdbaf726013c8c2fa93dd61343472eb70",
    "shared_backend": "local_base_fixed_last_endpoint_v1",
    "max_connections": 1,
}
BASE_MODEL = "openai-api/research4-qwen/qwen3.8-27b-base"
LAST_MODEL = "openai-api/research4-qwen/sp-lense-adaptor-last-layer"
LAST8_MODEL = "openai-api/research4-qwen/" + LAST8_SERVICE_MODEL_ID
RETRAINED_MODEL = "openai-api/research4-qwen/" + RETRAINED_SERVICE_MODEL_ID
PROVIDER352_MODEL = "openai-api/research4-qwen/" + PROVIDER352_SERVICE_MODEL_ID
CURRENT_LOCAL_CONFIGURATION = {
    "backend": "local_mlx", "base_repository": LOCAL_CONFIGURATION["base_repository"],
    "base_revision": LOCAL_BASE_REVISION, "base_manifest_sha256": LOCAL_BASE_MANIFEST_SHA256,
    "base_quantization": "4bit affine/group64; not NF4", "max_connections": 1,
    "request_timeout_seconds": 900,
    "local_binding_group": "last8", "required_context_tokens": LAST8_CONTEXT_TOKENS,
    "configuration_evidence": "configured identity only; verified new readiness binding and fresh health readback required before execution",
}
# Held future identity only. Enabling this route requires a separately reviewed
# final-artifact/readiness binding; counters or placeholder hashes cannot enable it.
PROVIDER352_PENDING_CONFIGURATION = {
    **CURRENT_LOCAL_CONFIGURATION,
    "service_model_id": PROVIDER352_SERVICE_MODEL_ID,
    "adapter_sha256": None, "adapter_config_sha256": None,
    "adapter_rank": 16, "adapter_blocks": list(range(56, 64)), "adapter_projection_count": 62,
    "adapter_applied": True, "adapter_dtype": "F32", "max_output_tokens": LAST8_OUTPUT_TOKENS,
    "completed_training_updates": None, "provider_DEV_rows": None,
    "four_route_source_freeze": None, "four_route_qualified_readiness": None,
    "provider_readiness": "NOT_READY", "execution_enabled": False,
    "configuration_evidence": PROVIDER352_NOT_READY_REASON,
}
MODEL_SPECS = {
    "qwen_base_neutral": {"label": "Qwen3.8-27B · Mac · neutral", "model": BASE_MODEL,
        "arm_id": "base_neutral", "prompt_condition": "baseline", "local_qwen": True,
        "service_model_id": "qwen3.8-27b-base", "adapter_applied": False,
        "adapter_sha256": None, "max_output_tokens": LAST8_OUTPUT_TOKENS, **CURRENT_LOCAL_CONFIGURATION},
    "qwen_base": {"label": "Qwen3.8-27B · Mac · security instruction", "model": BASE_MODEL,
        "arm_id": "base_icl", "prompt_condition": "security_maximization", "local_qwen": True,
        "service_model_id": "qwen3.8-27b-base", "adapter_applied": False,
        "adapter_sha256": None, "max_output_tokens": LAST8_OUTPUT_TOKENS, **CURRENT_LOCAL_CONFIGURATION},
    "sp_lense_neutral": {"label": "SP-Lense LAST · Mac · FT only", "model": LAST_MODEL,
        "arm_id": "last_neutral", "prompt_condition": "baseline", "local_qwen": True,
        "service_model_id": "sp-lense-adaptor-last-layer", "adapter_sha256": LAST_SHA,
        "max_output_tokens": 4096, "adapter_applied": True, "adapter_dtype": "F32", "retired": True, **LOCAL_CONFIGURATION},
    "sp_lense": {"label": "SP-Lense LAST · Mac · FT + security instruction", "model": LAST_MODEL,
        "arm_id": "last_icl", "prompt_condition": "security_maximization", "local_qwen": True,
        "service_model_id": "sp-lense-adaptor-last-layer", "adapter_sha256": LAST_SHA,
        "max_output_tokens": 4096, "adapter_applied": True, "adapter_dtype": "F32", "retired": True, **LOCAL_CONFIGURATION},
    "sp_lense_last8_neutral": {"label": "SP-Lense OLD last8 r16 · FT only", "model": LAST8_MODEL,
        "arm_id": "last8_neutral", "prompt_condition": "baseline", "local_qwen": True,
        "service_model_id": LAST8_SERVICE_MODEL_ID, "adapter_sha256": LAST8_SHA,
        "adapter_config_sha256": LAST8_CONFIG_SHA, "adapter_rank": 16,
        "adapter_blocks": list(range(56, 64)), "completed_training_updates": 2160,
        "adapter_applied": True, "adapter_dtype": "F32", "max_output_tokens": LAST8_OUTPUT_TOKENS,
        **CURRENT_LOCAL_CONFIGURATION},
    "sp_lense_last8": {"label": "SP-Lense OLD last8 r16 · FT + security instruction", "model": LAST8_MODEL,
        "arm_id": "last8_icl", "prompt_condition": "security_maximization", "local_qwen": True,
        "service_model_id": LAST8_SERVICE_MODEL_ID, "adapter_sha256": LAST8_SHA,
        "adapter_config_sha256": LAST8_CONFIG_SHA, "adapter_rank": 16,
        "adapter_blocks": list(range(56, 64)), "completed_training_updates": 2160,
        "adapter_applied": True, "adapter_dtype": "F32", "max_output_tokens": LAST8_OUTPUT_TOKENS,
        **CURRENT_LOCAL_CONFIGURATION},
    "sp_lense_retrained_neutral": {"label": "SP-Lense NEW last8 r16 · FT only", "model": RETRAINED_MODEL,
        "arm_id": "retrained_neutral", "prompt_condition": "baseline", "local_qwen": True,
        "service_model_id": RETRAINED_SERVICE_MODEL_ID, "adapter_sha256": RETRAINED_SHA,
        "adapter_config_sha256": RETRAINED_CONFIG_SHA, "adapter_rank": 16,
        "adapter_blocks": list(range(56, 64)), "adapter_projection_count": 62,
        "adapter_applied": True, "adapter_dtype": "F32", "max_output_tokens": LAST8_OUTPUT_TOKENS,
        **CURRENT_LOCAL_CONFIGURATION},
    "sp_lense_retrained": {"label": "SP-Lense NEW last8 r16 · FT + security instruction", "model": RETRAINED_MODEL,
        "arm_id": "retrained_icl", "prompt_condition": "security_maximization", "local_qwen": True,
        "service_model_id": RETRAINED_SERVICE_MODEL_ID, "adapter_sha256": RETRAINED_SHA,
        "adapter_config_sha256": RETRAINED_CONFIG_SHA, "adapter_rank": 16,
        "adapter_blocks": list(range(56, 64)), "adapter_projection_count": 62,
        "adapter_applied": True, "adapter_dtype": "F32", "max_output_tokens": LAST8_OUTPUT_TOKENS,
        **CURRENT_LOCAL_CONFIGURATION},
    "sp_lense_provider352_neutral": {"label": "SP-Lense provider3.5.2 · NOT READY · FT only",
        "model": PROVIDER352_MODEL, "arm_id": "provider352_neutral", "prompt_condition": "baseline",
        "local_qwen": True, **PROVIDER352_PENDING_CONFIGURATION},
    "sp_lense_provider352": {"label": "SP-Lense provider3.5.2 · NOT READY · FT + security instruction",
        "model": PROVIDER352_MODEL, "arm_id": "provider352_icl", "prompt_condition": "security_maximization",
        "local_qwen": True, **PROVIDER352_PENDING_CONFIGURATION},
    "sp_lense_security500_neutral": {"label": "SP-Lense security500 · NOT READY · FT only",
        "model": SECURITY500_MODEL, "arm_id": "security500_neutral", "prompt_condition": "baseline",
        "local_qwen": True, **CURRENT_LOCAL_CONFIGURATION, "service_model_id": SECURITY500_SERVICE_MODEL_ID,
        "adapter_applied": True, "adapter_sha256": None, "adapter_config_sha256": None,
        "adapter_rank": 16, "adapter_blocks": list(range(56, 64)), "adapter_projection_count": 62,
        "adapter_dtype": "F32", "completed_training_updates": None, "technical_acceptance": None,
        "max_output_tokens": LAST8_OUTPUT_TOKENS, "execution_enabled": False, "security500_readiness": "NOT_READY"},
    "sp_lense_security500": {"label": "SP-Lense security500 · NOT READY · FT + security instruction",
        "model": SECURITY500_MODEL, "arm_id": "security500_icl", "prompt_condition": "security_maximization",
        "local_qwen": True, **CURRENT_LOCAL_CONFIGURATION, "service_model_id": SECURITY500_SERVICE_MODEL_ID,
        "adapter_applied": True, "adapter_sha256": None, "adapter_config_sha256": None,
        "adapter_rank": 16, "adapter_blocks": list(range(56, 64)), "adapter_projection_count": 62,
        "adapter_dtype": "F32", "completed_training_updates": None, "technical_acceptance": None,
        "max_output_tokens": LAST8_OUTPUT_TOKENS, "execution_enabled": False, "security500_readiness": "NOT_READY"},
    "sp_lense_transfer_neutral": {"label": "SP Lens - 2026-11-15 13:19 MST · completed264 · HELD · FT only",
        "model": CONTINUITY_TRANSFER_MODEL, "arm_id": "transfer_neutral", "prompt_condition": "baseline",
        "local_qwen": True, **CURRENT_LOCAL_CONFIGURATION,
        "service_model_id": CONTINUITY_TRANSFER_SERVICE_MODEL_ID,
        "adapter_applied": True,
        "adapter_sha256": "f1495fe776512a07ecbdbea70c1e0c517796bf9ca8d96c791893358d028c8d17",
        "adapter_config_sha256": "7e46cbeb01676a9dccf61ddde69b2b58b109793e1b66faa04a643b406399ada4",
        "adapter_rank": 16, "adapter_blocks": list(range(56, 64)), "adapter_projection_count": 62,
        "adapter_dtype": "F32", "completed_training_updates": 264,
        "max_output_tokens": LAST8_OUTPUT_TOKENS, "execution_enabled": False,
        "continuity_transfer_readiness": "HELD_PENDING_ACTUAL_QUALIFIED_CORE_AND_GRANT"},
    "terra": {"label": "Terra · security instruction", "model": "openai/gpt-6.1-terra",
        "arm_id": "terra_icl", "prompt_condition": "security_maximization",
        "key_name": "OPENAI_API_KEY", "max_output_tokens": 8192, "backend": "hosted_api", "precision": "unknown"},
    "deepseek": {"label": "DeepSeek · security instruction", "model": "deepseek/deepseek-flash",
        "arm_id": "deepseek_icl", "prompt_condition": "security_maximization",
        "key_name": "DEEPSEEK_API_KEY", "max_output_tokens": 8192, "backend": "hosted_api", "precision": "unknown"},
    "qwen_max": {"label": "Qwen3.8-Max API · security instruction", "model": "qwencloud/qwen3.8-max",
        "arm_id": "qwen_max_icl", "prompt_condition": "security_maximization", "provider": "qwencloud",
        "api_model": "qwen3.8-max", "key_name": "QWEN_API_KEY", "max_output_tokens": 8192,
        "backend": "hosted_api", "precision": "unknown",
        "max_connections": 1, "concurrency_scope": "account",
        "output_token_parameter": "max_completion_tokens", "output_budget_scope": "thinking_plus_answer"},
}

# Scientific prompt variants share the exact two existing physical routes.
# The original semantic-history specs and transport guards remain unchanged.
MODEL_SPECS["qwen_base_operational_icl"] = {
    **json.loads(json.dumps(MODEL_SPECS["qwen_base_neutral"])),
    "label": "Qwen3.8-27B · Mac · operational preservation ICL · HELD",
    "arm_id": "base_operational_icl", "prompt_condition": "operational_preservation_icl",
    "required_history_mode": "full_chronological_v1", "requires_scientific_amendment": True,
    "execution_enabled": False,
}
MODEL_SPECS["sp_lense_transfer_operational_icl"] = {
    **json.loads(json.dumps(MODEL_SPECS["sp_lense_transfer_neutral"])),
    "label": "SP Lens - completed264 · FT + operational preservation ICL · HELD",
    "arm_id": "transfer_operational_icl", "prompt_condition": "operational_preservation_icl",
    "required_history_mode": "full_chronological_v1", "requires_scientific_amendment": True,
    "execution_enabled": False,
}


def retrained_identity():
    """Require the actual frozen checkpoint before enabling its conditions."""
    neutral = MODEL_SPECS["sp_lense_retrained_neutral"]
    instructed = MODEL_SPECS["sp_lense_retrained"]
    fields = ("adapter_sha256", "adapter_config_sha256", "adapter_rank", "adapter_blocks", "adapter_projection_count")
    if any(neutral[key] != instructed[key] for key in fields):
        raise ValueError("Retrained conditions must bind the same frozen adapter")
    if (neutral["adapter_rank"] != 16 or neutral["adapter_blocks"] != list(range(56, 64))
            or neutral["adapter_projection_count"] != 62):
        raise ValueError("Retrained adapter geometry differs from the agreed serving contract")
    for key in ("adapter_sha256", "adapter_config_sha256"):
        value = neutral[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Retrained checkpoint qualification is pending")
    if neutral["adapter_sha256"] in {LAST8_SHA, LAST_SHA}:
        raise ValueError("The old checkpoint cannot identify the retrained adapter")
    return {"adapter_enabled": True, "adapter_sha256": neutral["adapter_sha256"],
            "adapter_config_sha256": neutral["adapter_config_sha256"], "rank": 16,
            "blocks": list(range(56, 64)), "projections": 62}


def provider352_handoff_contract():
    """Load the exact shared validator at its canonical dependency-relative path."""
    source = PROVIDER352_HANDOFF_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != PROVIDER352_HANDOFF_SHA:
        raise ValueError("Exact Root handoff validator source required")
    module = ModuleType("research4_pinned_game_handoff")
    module.__file__ = str(PROVIDER352_HANDOFF_PATH)
    exec(compile(source, str(PROVIDER352_HANDOFF_PATH), "exec"), module.__dict__)
    return module


def _bounded_metadata_bytes(path):
    with path.open("rb") as stream:
        raw = stream.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("Handoff metadata exceeds its bound")
    return raw


def _provider352_metadata_path(relative, contract):
    if not contract._ref({"path": relative, "sha256": "0" * 64}):
        raise ValueError("Canonical project-relative metadata path required")
    path = (PROVIDER352_ROOT / relative).resolve()
    if not path.is_relative_to(PROVIDER352_ROOT.resolve()):
        raise ValueError("Handoff metadata must remain within Root's repository")
    return path


def initialize_provider352_core(core_binding, *, read_bytes=None):
    """PLANNING_ONLY: explicit trusted Root pin, no window/health/model request.

    Root issues this reference after native qualification. It establishes a
    stable declared identity, never current readiness or permission to request.
    """
    global _PROVIDER352_CORE_REF, _PROVIDER352_CORE_RAW, _PROVIDER352_CORE_SPECS
    if (_SECURITY500_CORE_REF is not None or os.environ.get(SECURITY500_CONTEXT_ENV)
            or _CONTINUITY_TRANSFER_CORE_REF is not None or os.environ.get(CONTINUITY_TRANSFER_CONTEXT_ENV)):
        raise ValueError("Two-route and four-route contexts cannot be combined")
    h = provider352_handoff_contract()
    if not h._ref(core_binding):
        raise ValueError(PROVIDER352_NOT_READY_REASON)
    reference = dict(core_binding)
    if _PROVIDER352_CORE_REF is not None and reference != dict(_PROVIDER352_CORE_REF):
        raise ValueError("The planning core is immutable; a different core requires a separately reviewed run")
    raw = (read_bytes or _bounded_metadata_bytes)(_provider352_metadata_path(reference["path"], h))
    core = h._raw(raw, reference["sha256"])
    h.validate_core_metadata(core)
    if any(core["provider_artifacts"][role]["sha256"] != digest
           for role, digest in PROVIDER352_TECHNICAL_PINS.items()):
        raise ValueError("Qualified core does not bind the exact accepted provider technical artifacts")
    provider = core["per_model"][PROVIDER352_SERVICE_MODEL_ID]
    expected = {"adapter_enabled": True, "adapter_sha256": PROVIDER352_TECHNICAL_PINS["adapter"],
        "adapter_config_sha256": PROVIDER352_TECHNICAL_PINS["adapter_config"],
        "rank": 16, "blocks": list(range(56, 64)), "projections": 62}
    if not h._same(provider, expected):
        raise ValueError("Qualified provider identity or geometry differs from the accepted checkpoint")
    display_names = core["model_display_names"]
    display_name = display_names.get(PROVIDER352_SERVICE_MODEL_ID) if type(display_names) is dict else None
    if type(display_name) is not str or not display_name.strip():
        raise ValueError("Qualified core requires an explicit provider display name")
    labels = {name: display_name + " · " + ("FT only" if MODEL_SPECS[name]["prompt_condition"] == "baseline"
                                           else "FT + security instruction")
              for name in ("sp_lense_provider352_neutral", "sp_lense_provider352")}
    if _PROVIDER352_CORE_REF is None:
        updates = {"adapter_sha256": expected["adapter_sha256"],
            "adapter_config_sha256": expected["adapter_config_sha256"],
            "completed_training_updates": 2160, "provider_DEV_rows": 500,
            "four_route_source_freeze": core["service_source_freeze"],
            "four_route_qualified_readiness": {"path": core["readiness_receipt_path"],
                                                "sha256": core["readiness_receipt_sha256"]},
            "provider_readiness": "PLANNING_ONLY", "planning_core_binding": reference,
            "configuration_evidence": "Root-pinned declared identity only; finite grant and fresh per-request validation required"}
        for name in ("sp_lense_provider352_neutral", "sp_lense_provider352"):
            MODEL_SPECS[name].update(json.loads(json.dumps(updates)))
            MODEL_SPECS[name]["label"] = labels[name]
            MODEL_SPECS[name].pop("execution_enabled", None)
        _PROVIDER352_CORE_REF = MappingProxyType(reference)
        _PROVIDER352_CORE_RAW = raw
        _PROVIDER352_CORE_SPECS = {name: json.loads(json.dumps(MODEL_SPECS[name]))
            for name in ("sp_lense_provider352_neutral", "sp_lense_provider352")}
    _assert_provider352_identity()
    return json.loads(_PROVIDER352_CORE_RAW)


def _assert_provider352_identity():
    if _PROVIDER352_CORE_REF is None or any(MODEL_SPECS[name] != spec
            for name, spec in _PROVIDER352_CORE_SPECS.items()):
        raise ValueError(PROVIDER352_NOT_READY_REASON)


def provider352_planning_core_ref():
    return dict(_PROVIDER352_CORE_REF) if _PROVIDER352_CORE_REF is not None else None


def _provider352_private_context(raw):
    h = provider352_handoff_contract()
    if type(raw) is not str:
        raise ValueError(PROVIDER352_NOT_READY_REASON)
    encoded = raw.encode("utf-8")
    # This hash is syntax checking only; authority still comes from supplied
    # independent pins and the mandatory Root validator, never this envelope.
    context = h._raw(encoded, hashlib.sha256(encoded).hexdigest())
    if type(context) is not dict or set(context) != {"pins", "requester_chat", "core_binding"}:
        raise ValueError("Closed private Root control context required")
    return context


def _initialize_provider352_private_core():
    raw = os.environ.get(PROVIDER352_CONTEXT_ENV)
    if raw:
        try:
            context = _provider352_private_context(raw)
            initialize_provider352_core(context["core_binding"])
        except Exception:
            raise ValueError(PROVIDER352_NOT_READY_REASON) from None


def provider352_active_consumer():
    global _PROVIDER352_ACTIVE_CONTEXT, _PROVIDER352_ACTIVE_CONSUMER
    raw = os.environ.get(PROVIDER352_CONTEXT_ENV)
    if not raw:
        raise ValueError(PROVIDER352_NOT_READY_REASON)
    if _PROVIDER352_ACTIVE_CONSUMER is not None:
        if raw != _PROVIDER352_ACTIVE_CONTEXT:
            _PROVIDER352_ACTIVE_CONSUMER.stop()
            raise ValueError("Private Root request scope changed; no automatic rebind")
        return _PROVIDER352_ACTIVE_CONSUMER
    consumer = Provider352GameConsumer.from_private_context(raw)
    _PROVIDER352_ACTIVE_CONTEXT, _PROVIDER352_ACTIVE_CONSUMER = raw, consumer
    return consumer


class Provider352GameConsumer:
    """Dormant control adapter; constructing it does not admit a Game model.

    Pins must arrive through Root's trusted handoff, never current metadata.
    Reader injection is for transport tests; validation always uses Root's code.
    """
    PIN_FIELDS = frozenset({"index_path", "expected_index_sha256", "expected_window_sha256",
                           "expected_nonce", "expected_game_plan_sha256", "expected_direct_authority_sha256"})

    def __init__(self, pins, *, core_binding, requester_chat, read_bytes=None, read_health=None, clock=None):
        if type(pins) is not dict or set(pins) != self.PIN_FIELDS:
            raise ValueError("Explicit immutable Root handoff pins required")
        self.pins = MappingProxyType(dict(pins))
        self.requester_chat = requester_chat
        self.contract = provider352_handoff_contract()
        if not self.contract._ref(core_binding):
            raise ValueError("Independent Root planning core reference required")
        self.core_binding = MappingProxyType(dict(core_binding))
        if (any(not self.contract._sha(pins[key]) for key in self.PIN_FIELDS
                if key.startswith("expected_") and key != "expected_nonce")
                or type(pins["expected_nonce"]) is not str or len(pins["expected_nonce"]) != 32
                or any(c not in "0123456789abcdef" for c in pins["expected_nonce"])):
            raise ValueError("Root SHA/nonce pins remain unbound")
        self._path(pins["index_path"])
        self._read_bytes = read_bytes or _bounded_metadata_bytes
        self._read_health = read_health or _provider352_fresh_health
        self._clock = clock or time.time
        self._denied = False

    @classmethod
    def from_private_context(cls, raw, *, read_bytes=None, read_health=None, clock=None):
        context = _provider352_private_context(raw)
        initialize_provider352_core(context["core_binding"], read_bytes=read_bytes)
        return cls(context["pins"], core_binding=context["core_binding"],
                   requester_chat=context["requester_chat"], read_bytes=read_bytes,
                   read_health=read_health, clock=clock)

    def _path(self, relative):
        return _provider352_metadata_path(relative, self.contract)

    def _reference(self, reference):
        if not self.contract._ref(reference):
            raise ValueError("Exact Root metadata reference required")
        raw = self._read_bytes(self._path(reference["path"]))
        return raw, self.contract._raw(raw, reference["sha256"])

    def require_plan(self, parent_plan_sha256):
        if parent_plan_sha256 != self.pins["expected_game_plan_sha256"]:
            self._denied = True
            raise ValueError("Whole parent Game plan differs from Root's pinned scope")

    def private_child_context(self, parent_plan_sha256):
        self.require_plan(parent_plan_sha256)
        return json.dumps({"pins": dict(self.pins), "requester_chat": self.requester_chat,
                           "core_binding": dict(self.core_binding)},
                          sort_keys=True, separators=(",", ":"))

    def stop(self):
        """Latch external collection denial without changing Root's grant."""
        self._denied = True

    def validate_request(self, requested_model_id, *, request=None, selected_spec=None):
        decision, _ = self._validated_request(requested_model_id, request=request, selected_spec=selected_spec)
        return decision

    def validated_health(self, requested_model_id):
        _, health = self._validated_request(requested_model_id)
        return health

    def _validated_request(self, requested_model_id, *, request=None, selected_spec=None):
        if self._denied:
            raise ValueError("Handoff consumer stopped; fresh Root issuance and review required")
        try:
            h = self.contract
            index_raw = self._read_bytes(self._path(self.pins["index_path"]))
            index = h._raw(index_raw, self.pins["expected_index_sha256"])
            if not h._same(index["core_binding"], dict(self.core_binding)):
                raise ValueError("Grant index differs from the independent planning core")
            core_raw, core = self._reference(index["core_binding"])
            window_raw, window = self._reference(index["request_window"])
            # Early rejection avoids health I/O for a changed or closed window.
            h._raw(window_raw, self.pins["expected_window_sha256"])
            h._need(h.validate_window_metadata(window), "No current Root request grant")
            h.validate_core_metadata(core)
            if request is not None:
                self._validate_wire(core, request, selected_spec)
            receipt_raw, receipt = self._reference({"path": core["readiness_receipt_path"],
                                                    "sha256": core["readiness_receipt_sha256"]})
            qualification_raw, qualification = self._reference(core["four_route_request_qualification"])
            chain_raw, chain = self._reference(core["provider_artifacts"]["completion_auditor_review"])
            full_raw, _ = self._reference(chain["full_review_v1"])
            focused_raw, _ = self._reference(chain["focused_review_v2"])
            root_v3_raw, _ = self._reference(chain["narrow_root_source_proof_peer_v3"])
            game_v3_raw, _ = self._reference(chain["game_source_proof_review_v3"])
            health = self._read_health(core)
            inputs = {"qualified_receipt": receipt, "fresh_health": health,
                "source_sha256": core["expected_health"]["source_sha256"],
                "manifest_sha256": core["expected_health"]["manifest_sha256"],
                "qualified_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
                "four_route_qualification": qualification,
                "four_route_qualification_sha256": hashlib.sha256(qualification_raw).hexdigest()}
            decision = h.validate_per_request_metadata(core_raw, index_raw, window_raw,
                **{key: value for key, value in self.pins.items() if key != "index_path"},
                requested_model_id=requested_model_id, requester_chat=self.requester_chat, now_epoch=self._clock(),
                identity_inputs=inputs, auditor_review_chain_bytes=chain_raw,
                auditor_full_review_bytes=full_raw, auditor_focused_review_bytes=focused_raw,
                auditor_v3_root_review_bytes=root_v3_raw, auditor_v3_Game_review_bytes=game_v3_raw)
            return decision, health
        except Exception:
            self._denied = True
            raise ValueError("Root handoff denied; no model request may dispatch") from None

    def _validate_wire(self, core, request, spec):
        from .hosted_inspect_runner import hosted_model_config

        profile = hosted_model_config(spec["model"])
        expected = {key: profile[key] for key in ("temperature", "top_p", "seed")}
        expected.update(profile["extra_body"], model=spec["service_model_id"],
                        max_tokens=spec["max_output_tokens"], stream=True)

        _validate_local_urls(core)
        body = json.loads(request.content)
        if (request.method != "POST" or str(request.url) != core["base_url"] + "/chat/completions"
                or not isinstance(body, dict) or "max_completion_tokens" in body
                or "reasoning_effort" in body
                or any(not self.contract._same(body.get(key), value) for key, value in expected.items())):
            raise ValueError("Actual request route/profile differs from the selected spec")
        if (spec["required_context_tokens"] != LAST8_CONTEXT_TOKENS
                or spec["max_output_tokens"] != LAST8_OUTPUT_TOKENS
                or spec["max_connections"] != 1):
            raise ValueError("Selected local bounds differ from the reviewed profile")
        identity = core["per_model"][spec["service_model_id"]]
        if spec.get("adapter_applied") and any(identity[key] != spec.get(key)
                for key in ("adapter_sha256", "adapter_config_sha256")):
            raise ValueError("Selected adapter remains unbound or differs from Root's identity")

    def install_http_guard(self, model, model_id):
        """Append enforcement to the existing SDK HTTP client, including retries."""
        spec = dict(MODEL_SPECS[model_id])
        if not spec.get("local_qwen") or str(model) != spec["model"]:
            raise ValueError("The selected local model must match the guarded client")

        async def before_request(request):
            self.validate_request(spec["service_model_id"], request=request, selected_spec=spec)

        model.api.client._client.event_hooks.setdefault("request", []).append(before_request)
        original_generate = getattr(model, "generate", None)
        if callable(original_generate):
            async def guarded_generate(*args, **kwargs):
                from .hosted_inspect_runner import IncompleteResponse
                if self._denied:
                    raise IncompleteResponse("game_request_window_denied")
                try:
                    self.validate_request(spec["service_model_id"])
                    result = await original_generate(*args, **kwargs)
                    if self._denied:
                        raise IncompleteResponse("game_request_window_denied")
                    return result
                except Exception:
                    if self._denied:
                        raise IncompleteResponse("game_request_window_denied") from None
                    raise
            model.generate = guarded_generate
        return model


def security500_handoff_contract():
    if SECURITY500_HANDOFF_PATH is None or SECURITY500_HANDOFF_SHA is None:
        raise ValueError(SECURITY500_NOT_READY)
    source = SECURITY500_HANDOFF_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != SECURITY500_HANDOFF_SHA:
        raise ValueError("Exact Root two-route handoff source required")
    module = ModuleType("research4_pinned_security500_handoff")
    module.__file__ = str(SECURITY500_HANDOFF_PATH)
    exec(compile(source, str(SECURITY500_HANDOFF_PATH), "exec"), module.__dict__)
    return module


def continuity_transfer_handoff_contract():
    """Load only Root's exact public pure contract, without readiness I/O."""
    public_raw = CONTINUITY_TRANSFER_PUBLIC_PATH.read_bytes()
    source = CONTINUITY_TRANSFER_HANDOFF_PATH.read_bytes()
    if (hashlib.sha256(public_raw).hexdigest() != CONTINUITY_TRANSFER_PUBLIC_SHA
            or hashlib.sha256(source).hexdigest() != CONTINUITY_TRANSFER_HANDOFF_SHA):
        raise ValueError("Exact Root continuity-transfer public/helper source required")
    public = json.loads(public_raw)
    if (public.get("models") != {"BASE": "qwen3.8-27b-base", "FT": CONTINUITY_TRANSFER_SERVICE_MODEL_ID}
            or public.get("pure_handoff_helper", {}).get("sha256") != CONTINUITY_TRANSFER_HANDOFF_SHA):
        raise ValueError("Continuity-transfer public route/source identity differs")
    module = ModuleType("research4_pinned_continuity_transfer_handoff")
    module.__file__ = str(CONTINUITY_TRANSFER_HANDOFF_PATH)
    exec(compile(source, str(CONTINUITY_TRANSFER_HANDOFF_PATH), "exec"), module.__dict__)
    return module


def continuity_transfer_request_parser():
    source = CONTINUITY_TRANSFER_PARSER_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != CONTINUITY_TRANSFER_PARSER_SHA:
        raise ValueError("Exact Root public-content request parser required")
    module = ModuleType("research4_pinned_continuity_request_parser")
    module.__file__ = str(CONTINUITY_TRANSFER_PARSER_PATH)
    exec(compile(source, str(CONTINUITY_TRANSFER_PARSER_PATH), "exec"), module.__dict__)
    return module


def initialize_continuity_transfer_core(core_binding, *, read_bytes=None):
    """Bind independently pinned264 metadata only; no request authority follows."""
    global _CONTINUITY_TRANSFER_CORE_REF, _CONTINUITY_TRANSFER_CORE_RAW, _CONTINUITY_TRANSFER_CORE_SPECS
    h = continuity_transfer_handoff_contract()
    if (type(core_binding) is not dict or set(core_binding) != {"core_binding", "trusted_review_sha256"}
            or not h.ref(core_binding["core_binding"]) or not h.is_sha(core_binding["trusted_review_sha256"])):
        raise ValueError(CONTINUITY_TRANSFER_NOT_READY)
    if (_SECURITY500_CORE_REF is not None or _PROVIDER352_CORE_REF is not None
            or os.environ.get(SECURITY500_CONTEXT_ENV) or os.environ.get(PROVIDER352_CONTEXT_ENV)):
        raise ValueError("Continuity-transfer and historical serving contexts cannot be combined")
    descriptor = json.loads(json.dumps(core_binding))
    if _CONTINUITY_TRANSFER_CORE_REF is not None and descriptor != _CONTINUITY_TRANSFER_CORE_REF:
        raise ValueError("Continuity-transfer planning core is immutable for this process")
    reference = descriptor["core_binding"]
    raw = (read_bytes or _bounded_metadata_bytes)(_security500_path(reference["path"], h))
    core = h.raw_json(raw, reference["sha256"])
    h.validate_core(core, descriptor["trusted_review_sha256"])
    identity = core["per_model"][h.FT]
    spec = MODEL_SPECS["sp_lense_transfer_neutral"]
    if (identity["adapter_sha256"] != spec["adapter_sha256"]
            or identity["adapter_config_sha256"] != spec["adapter_config_sha256"]):
        raise ValueError("Completed264 adapter differs from the independently qualified core")
    if _CONTINUITY_TRANSFER_CORE_REF is None:
        spec.update(continuity_transfer_readiness="PLANNING_ONLY_REQUIRES_FINITE_GRANT",
                    planning_core_binding=descriptor)
        spec.pop("execution_enabled", None)
        MODEL_SPECS["sp_lense_transfer_operational_icl"].update(
            planning_core_binding=descriptor,
            continuity_transfer_readiness="HELD_PENDING_SCIENTIFIC_AMENDMENT")
        _CONTINUITY_TRANSFER_CORE_REF = descriptor
        _CONTINUITY_TRANSFER_CORE_RAW = raw
        _CONTINUITY_TRANSFER_CORE_SPECS = {
            name: json.loads(json.dumps(MODEL_SPECS[name]))
            for name in CONTINUITY_TRANSFER_FULL_HISTORY_IDS}
    _assert_continuity_transfer_identity()
    return json.loads(_CONTINUITY_TRANSFER_CORE_RAW)


def _assert_continuity_transfer_identity():
    if (_CONTINUITY_TRANSFER_CORE_REF is None
            or any(MODEL_SPECS[name] != spec for name, spec in _CONTINUITY_TRANSFER_CORE_SPECS.items())):
        raise ValueError(CONTINUITY_TRANSFER_NOT_READY)


def continuity_transfer_planning_core_ref():
    return json.loads(json.dumps(_CONTINUITY_TRANSFER_CORE_REF)) if _CONTINUITY_TRANSFER_CORE_REF is not None else None


def _continuity_transfer_selected(models=()):
    return (bool(os.environ.get(CONTINUITY_TRANSFER_CONTEXT_ENV))
            or _CONTINUITY_TRANSFER_CORE_REF is not None
            or bool({"qwen_base_operational_icl", "sp_lense_transfer_neutral",
                     "sp_lense_transfer_operational_icl"} & set(models)))


def _continuity_transfer_private_context(raw):
    h = continuity_transfer_handoff_contract()
    if type(raw) is not str:
        raise ValueError(CONTINUITY_TRANSFER_NOT_READY)
    data = raw.encode("utf-8")
    context = h.raw_json(data, hashlib.sha256(data).hexdigest())
    if type(context) is not dict or set(context) != {"pins", "requester_chat", "core_binding", "source_scope"}:
        raise ValueError("Closed continuity-transfer private context required")
    return context


def _initialize_continuity_transfer_private_core():
    raw = os.environ.get(CONTINUITY_TRANSFER_CONTEXT_ENV)
    if raw:
        initialize_continuity_transfer_core(_continuity_transfer_private_context(raw)["core_binding"])


def continuity_transfer_active_consumer():
    global _CONTINUITY_TRANSFER_ACTIVE_CONTEXT, _CONTINUITY_TRANSFER_ACTIVE_CONSUMER
    raw = os.environ.get(CONTINUITY_TRANSFER_CONTEXT_ENV)
    if not raw:
        raise ValueError(CONTINUITY_TRANSFER_NOT_READY)
    amendment_raw = os.environ.get(SCIENTIFIC_AMENDMENT_ENV)
    context_key = raw + ("\n" + amendment_raw if amendment_raw else "")
    if _CONTINUITY_TRANSFER_ACTIVE_CONSUMER is not None:
        if context_key != _CONTINUITY_TRANSFER_ACTIVE_CONTEXT:
            _CONTINUITY_TRANSFER_ACTIVE_CONSUMER.stop()
            raise ValueError("Continuity-transfer private scope changed")
        return _CONTINUITY_TRANSFER_ACTIVE_CONSUMER
    if amendment_raw:
        h = continuity_transfer_handoff_contract()
        data = amendment_raw.encode()
        amendment_ref = h.raw_json(data, hashlib.sha256(data).hexdigest())
        context = _continuity_transfer_private_context(raw)
        initialize_continuity_transfer_core(context["core_binding"])
        consumer = ScientificSuccessorGameConsumer(context["pins"], core_binding=context["core_binding"],
            requester_chat=context["requester_chat"], source_scope=context["source_scope"],
            scientific_amendment_ref=amendment_ref)
    else:
        consumer = ContinuityTransferGameConsumer.from_private_context(raw)
    _CONTINUITY_TRANSFER_ACTIVE_CONTEXT, _CONTINUITY_TRANSFER_ACTIVE_CONSUMER = context_key, consumer
    return consumer


def _security500_path(relative, contract):
    if not contract.ref({"path": relative, "sha256": "0" * 64}):
        raise ValueError("Canonical Root-relative metadata path required")
    path = (PROVIDER352_ROOT / relative).resolve()
    if not path.is_relative_to(PROVIDER352_ROOT.resolve()):
        raise ValueError("Security500 metadata must remain within Root's repository")
    return path


def initialize_security500_core(core_binding, *, read_bytes=None):
    """Bind independent planning pins; never grants a request or probes health."""
    global _SECURITY500_CORE_REF, _SECURITY500_CORE_RAW, _SECURITY500_CORE_SPECS
    h = security500_handoff_contract()
    if (type(core_binding) is not dict or set(core_binding) != {"core_binding", "trusted_review_sha256"}
            or not h.ref(core_binding["core_binding"]) or not h.is_sha(core_binding["trusted_review_sha256"])):
        raise ValueError(SECURITY500_NOT_READY)
    descriptor = json.loads(json.dumps(core_binding))
    if _PROVIDER352_CORE_REF is not None or os.environ.get(PROVIDER352_CONTEXT_ENV):
        raise ValueError("Two-route and four-route contexts cannot be combined")
    if _CONTINUITY_TRANSFER_CORE_REF is not None or os.environ.get(CONTINUITY_TRANSFER_CONTEXT_ENV):
        raise ValueError("Security500 and continuity-transfer contexts cannot be combined")
    if _SECURITY500_CORE_REF is not None and descriptor != _SECURITY500_CORE_REF:
        raise ValueError("Security500 planning core is immutable for this process")
    reference = descriptor["core_binding"]
    raw = (read_bytes or _bounded_metadata_bytes)(_security500_path(reference["path"], h))
    core = h.raw_json(raw, reference["sha256"])
    h.validate_core(core, descriptor["trusted_review_sha256"])
    identity = core["per_model"][h.FT]
    progress_ref = core["artifacts"]["progress"]
    progress_raw = (read_bytes or _bounded_metadata_bytes)(_security500_path(progress_ref["path"], h))
    progress = h.raw_json(progress_raw, progress_ref["sha256"])
    epoch = progress.get("updated_epoch") if type(progress) is dict else None
    if (type(progress) is not dict or progress.get("phase") != "complete"
            or type(progress.get("verified_updates")) is not int or progress["verified_updates"] != 500
            or type(progress.get("total_updates")) is not int or progress["total_updates"] != 500
            or type(epoch) not in (int, float) or epoch <= 0):
        raise ValueError("Exact completed Security500 progress timestamp required")
    from datetime import datetime, timedelta, timezone
    try:
        if not math.isfinite(epoch):
            raise ValueError("Non-finite timestamp")
        version_label = "SP Lens - " + (datetime.fromtimestamp(epoch, timezone.utc) - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M") + " ABT"
    except (OSError, OverflowError, ValueError):
        raise ValueError("Invalid completed Security500 progress timestamp") from None
    if _SECURITY500_CORE_REF is None:
        for name in ("sp_lense_security500_neutral", "sp_lense_security500"):
            MODEL_SPECS[name].update(adapter_sha256=identity["adapter_sha256"],
                adapter_config_sha256=identity["adapter_config_sha256"],
                completed_training_updates=500,
                label=version_label + (" · FT only" if name.endswith("_neutral") else " · FT + security instruction"),
                security500_readiness="PLANNING_ONLY", planning_core_binding=descriptor)
            MODEL_SPECS[name].pop("execution_enabled", None)
        _SECURITY500_CORE_REF = descriptor
        _SECURITY500_CORE_RAW = raw
        _SECURITY500_CORE_SPECS = {name: json.loads(json.dumps(MODEL_SPECS[name]))
            for name in ("sp_lense_security500_neutral", "sp_lense_security500")}
    _assert_security500_identity()
    return json.loads(_SECURITY500_CORE_RAW)


def _assert_security500_identity():
    if _SECURITY500_CORE_REF is None or any(MODEL_SPECS[name] != spec
            for name, spec in _SECURITY500_CORE_SPECS.items()):
        raise ValueError(SECURITY500_NOT_READY)


def security500_planning_core_ref():
    return json.loads(json.dumps(_SECURITY500_CORE_REF)) if _SECURITY500_CORE_REF is not None else None


def _security500_private_context(raw):
    h = security500_handoff_contract()
    if type(raw) is not str:
        raise ValueError(SECURITY500_NOT_READY)
    encoded = raw.encode("utf-8")
    value = h.raw_json(encoded, hashlib.sha256(encoded).hexdigest())
    if type(value) is not dict or set(value) != {"pins", "requester_chat", "core_binding", "source_scope"}:
        raise ValueError("Closed private security500 context required")
    return value


def _initialize_security500_private_core():
    raw = os.environ.get(SECURITY500_CONTEXT_ENV)
    if raw:
        try:
            initialize_security500_core(_security500_private_context(raw)["core_binding"])
        except Exception:
            raise ValueError(SECURITY500_NOT_READY) from None


def security500_active_consumer():
    global _SECURITY500_ACTIVE_CONTEXT, _SECURITY500_ACTIVE_CONSUMER
    raw = os.environ.get(SECURITY500_CONTEXT_ENV)
    if not raw:
        raise ValueError(SECURITY500_NOT_READY)
    if _SECURITY500_ACTIVE_CONSUMER is not None:
        if raw != _SECURITY500_ACTIVE_CONTEXT:
            _SECURITY500_ACTIVE_CONSUMER.stop()
            raise ValueError("Private security500 request scope changed")
        return _SECURITY500_ACTIVE_CONSUMER
    consumer = Security500GameConsumer.from_private_context(raw)
    _SECURITY500_ACTIVE_CONTEXT, _SECURITY500_ACTIVE_CONSUMER = raw, consumer
    return consumer


def _security500_reject_reserved_headers(request):
    headers = getattr(request, "headers", None)
    if headers is None or not callable(getattr(headers, "items", None)):
        raise ValueError("Inspectable request headers required")
    entries = headers.multi_items() if callable(getattr(headers, "multi_items", None)) else headers.items()
    if any(str(name).lower().startswith("x-research4-") for name, _ in entries):
        raise ValueError("Pre-existing Research4 authorization headers are forbidden")


def _security500_game_headers(window, decision, requested_model_id, window_sha256):
    """Build per-request metadata only from a successfully validated Root grant."""
    requester = "01a17889-f4ee-73a0-813a-5cb7add711d4"
    if (type(window) is not dict or type(decision) is not dict
            or decision.get("metadata_request_allowed") is not True
            or decision.get("owner_chat") != requester or window.get("owner_chat") != requester
            or decision.get("model_id") != requested_model_id
            or decision.get("grant_nonce") != window.get("grant_nonce")
            or requested_model_id not in window.get("allowed_model_ids", [])
            or type(window.get("grant_nonce")) is not str
            or type(window.get("game_plan_sha256")) is not str
            or type(window_sha256) is not str):
        raise ValueError("Validated Root Game window and exact route required")
    return {"X-Research4-Requester": requester,
            "X-Research4-Grant-Nonce": window["grant_nonce"],
            "X-Research4-Game-Plan-Sha256": window["game_plan_sha256"],
            "X-Research4-Window-Sha256": window_sha256}


class Security500GameConsumer(Provider352GameConsumer):
    """Exact two-route metadata and HTTP guard; Root retains window ownership."""

    def __init__(self, pins, *, core_binding, requester_chat, read_bytes=None, read_health=None, clock=None,
                 source_scope=None):
        if type(pins) is not dict or set(pins) != self.PIN_FIELDS:
            raise ValueError("Explicit immutable Root handoff pins required")
        self.contract = self._handoff_contract()
        h = self.contract
        if (type(core_binding) is not dict or set(core_binding) != {"core_binding", "trusted_review_sha256"}
                or not h.ref(core_binding["core_binding"]) or not h.is_sha(core_binding["trusted_review_sha256"])):
            raise ValueError(SECURITY500_NOT_READY)
        if (any(not h.is_sha(pins[key]) for key in self.PIN_FIELDS
                if key.startswith("expected_") and key != "expected_nonce")
                or type(pins["expected_nonce"]) is not str or len(pins["expected_nonce"]) != 32
                or any(c not in "0123456789abcdef" for c in pins["expected_nonce"])):
            raise ValueError("Root SHA/nonce pins remain unbound")
        self.pins = MappingProxyType(dict(pins))
        self.core_binding = json.loads(json.dumps(core_binding))
        self.requester_chat = requester_chat
        self._path(pins["index_path"])
        self._read_bytes = read_bytes or _bounded_metadata_bytes
        self._read_health = read_health or _provider352_fresh_health
        self._clock = clock or time.time
        self._denied = False
        self.source_scope = source_scope
        self._child_plan_verified = False

    def _handoff_contract(self):
        return security500_handoff_contract()

    def _grant_directory(self):
        return SECURITY500_GRANT_DIR

    @classmethod
    def from_private_context(cls, raw, *, read_bytes=None, read_health=None, clock=None):
        context = _security500_private_context(raw)
        initialize_security500_core(context["core_binding"], read_bytes=read_bytes)
        if context["source_scope"] is None:
            raise ValueError("Security500 child requires frozen Game source scope")
        return cls(context["pins"], core_binding=context["core_binding"], requester_chat=context["requester_chat"],
                   source_scope=context["source_scope"], read_bytes=read_bytes, read_health=read_health, clock=clock)

    def private_child_context(self, parent_plan_sha256, *, source_scope):
        self.require_plan(parent_plan_sha256)
        if (type(source_scope) is not dict or set(source_scope) != {"plan_path", "batch_index"}
                or type(source_scope["plan_path"]) is not str or type(source_scope["batch_index"]) is not int
                or source_scope["batch_index"] < 0):
            raise ValueError("Exact parent plan path and batch index required")
        return json.dumps({"pins": dict(self.pins), "requester_chat": self.requester_chat,
                           "core_binding": self.core_binding, "source_scope": source_scope},
                          sort_keys=True, separators=(",", ":"))

    def _validate_game_source_scope(self):
        from .company_world_evaluation import source_matches
        from .company_preservation_comparison import sha
        scope = self.source_scope
        if (type(scope) is not dict or set(scope) != {"plan_path", "batch_index"}
                or type(scope["plan_path"]) is not str or type(scope["batch_index"]) is not int
                or scope["batch_index"] < 0):
            raise ValueError("Exact private parent source scope required")
        repo = Path(__file__).resolve().parents[1]
        path = Path(scope["plan_path"])
        if (not path.is_absolute() or path.is_symlink() or (repo / ".local").resolve() not in path.resolve().parents
                or path.name != "PLAN.json"):
            raise ValueError("Parent Game plan must be in the local evaluation directory")
        grant_hash = self.pins["expected_window_sha256"]
        if (self._grant_directory() / (grant_hash + ".STOP.json")).exists():
            raise ValueError("Security500 grant has been stopped")
        claim_path = self._grant_directory() / (grant_hash + ".CLAIM.json")
        claim = json.loads(_bounded_metadata_bytes(claim_path))
        if (type(claim) is not dict or set(claim) != {"window_sha256", "game_plan_sha256", "destination", "initial_phase"}
                or claim["window_sha256"] != grant_hash
                or claim["game_plan_sha256"] != self.pins["expected_game_plan_sha256"]
                or claim["destination"] != str(path.resolve().parent)
                or claim["initial_phase"] not in {"all", "hosted", "local"}):
            raise ValueError("Security500 grant is not claimed by this plan and destination")
        raw = path.read_bytes()
        if len(raw) > 8_000_000 or hashlib.sha256(raw).hexdigest() != self.pins["expected_game_plan_sha256"]:
            raise ValueError("Frozen parent Game plan changed")
        frozen = json.loads(raw)
        batches = frozen["batches"]
        if (type(batches) is not list or scope["batch_index"] >= len(batches)
                or sha(Path(frozen["configuration_path"])) != frozen["configuration_sha256"]
                or sha(repo / "src/infrastructure_preservation_comparison.py") != frozen["orchestrator_source_sha256"]
                or not source_matches(batches[scope["batch_index"]])):
            raise ValueError("Frozen Game plan or source changed")
        return batches[scope["batch_index"]]

    def validate_child_plan(self, plan):
        from .company_preservation_comparison import validate_batch_manifest
        try:
            batch = self._validate_game_source_scope()
            validate_batch_manifest(plan, batch)
            self._child_plan_verified = True
        except Exception:
            self.stop()
            raise ValueError("Child Game plan or source differs from frozen parent") from None

    def _path(self, relative):
        return _security500_path(relative, self.contract)

    def _reference(self, reference):
        if not self.contract.ref(reference):
            raise ValueError("Exact Root metadata reference required")
        raw = self._read_bytes(self._path(reference["path"]))
        return raw, self.contract.raw_json(raw, reference["sha256"])

    def _validated_request(self, requested_model_id, *, request=None, selected_spec=None, return_window=False):
        if self._denied:
            raise ValueError("Security500 consumer stopped; fresh Root issuance required")
        try:
            if (self._grant_directory() / (self.pins["expected_window_sha256"] + ".STOP.json")).exists():
                raise ValueError("Security500 grant was stopped")
            h = self.contract
            index_raw = self._read_bytes(self._path(self.pins["index_path"]))
            index = h.raw_json(index_raw, self.pins["expected_index_sha256"])
            reference = self.core_binding["core_binding"]
            if index["core_binding"] != reference:
                raise ValueError("Grant index differs from independent core")
            core_raw, core = self._reference(reference)
            window_raw, window = self._reference(index["request_window"])
            h.raw_json(window_raw, self.pins["expected_window_sha256"])
            h.need(h.validate_window(window), "No current Root request grant")
            now = self._clock()
            h.need(window["core_binding"] == reference
                and window["grant_nonce"] == self.pins["expected_nonce"]
                and window["game_plan_sha256"] == self.pins["expected_game_plan_sha256"]
                and window["original_direct_authority"]["sha256"] == self.pins["expected_direct_authority_sha256"]
                and self.requester_chat == h.GAME_CHAT and requested_model_id in h.ARMS
                and type(now) in (int, float) and math.isfinite(now)
                and h.epoch(window["issued_utc"]) <= now < h.epoch(window["expires_utc"]),
                "Grant scope or validity differs")
            h.validate_core(core, self.core_binding["trusted_review_sha256"])
            if request is not None:
                self._validate_wire(core, request, selected_spec)
            audit_raw, _ = self._reference(core["artifacts"]["completion_audit"])
            ready_raw, _ = self._reference(core["readiness_receipt"])
            certificate_raw, _ = self._reference(core["two_route_qualification"])
            review_raw, _ = self._reference(core["artifacts"]["completion_auditor_review"])
            health = self._read_health(core)
            decision = h.validate_per_request_metadata(core_raw, index_raw, window_raw,
                audit_raw, ready_raw, certificate_raw, review_raw,
                trusted_core_sha256=reference["sha256"], trusted_index_sha256=self.pins["expected_index_sha256"],
                trusted_window_sha256=self.pins["expected_window_sha256"],
                trusted_review_sha256=self.core_binding["trusted_review_sha256"],
                trusted_nonce=self.pins["expected_nonce"], trusted_game_plan_sha256=self.pins["expected_game_plan_sha256"],
                trusted_direct_authority_sha256=self.pins["expected_direct_authority_sha256"],
                requested_model_id=requested_model_id, requester_chat=self.requester_chat,
                now_epoch=self._clock(), fresh_health=health)
            return (decision, health, window) if return_window else (decision, health)
        except Exception:
            self._denied = True
            raise ValueError("Security500 Root handoff denied; no model request may dispatch") from None

    def _validate_wire(self, core, request, spec):
        from .hosted_inspect_runner import hosted_model_config
        if not isinstance(spec, dict) or spec.get("service_model_id") not in self.contract.ARMS:
            raise ValueError("Exact selected two-route spec required")
        profile = hosted_model_config(spec["model"])
        expected = {key: profile[key] for key in ("temperature", "top_p", "seed")}
        expected.update(profile["extra_body"], model=spec["service_model_id"],
                        max_tokens=spec["max_output_tokens"], stream=True)
        _validate_local_urls(core)
        raw = bytes(request.content)
        body = self.contract.raw_json(raw, hashlib.sha256(raw).hexdigest())
        allowed = {"messages", "model", "stream", "stream_options", "max_tokens", "temperature", "top_p",
                   "seed", "chat_template_kwargs", "top_k", "min_p", "repetition_penalty"}
        if (request.method != "POST" or str(request.url) != core["base_url"] + "/chat/completions"
                or not isinstance(body, dict) or set(body) - allowed
                or not isinstance(body.get("messages"), list) or not body["messages"]
                or ("stream_options" in body and body["stream_options"] != {"include_usage": True})
                or any(not self.contract.same(body.get(key), value) for key, value in expected.items())
                or spec["required_context_tokens"] != LAST8_CONTEXT_TOKENS
                or spec["max_output_tokens"] != LAST8_OUTPUT_TOKENS or spec["max_connections"] != 1):
            raise ValueError("Actual route/profile/bounds differ from the reviewed request")
        identity = core["per_model"][spec["service_model_id"]]
        if spec.get("adapter_applied") and any(identity[key] != spec.get(key)
                for key in ("adapter_sha256", "adapter_config_sha256")):
            raise ValueError("Selected adapter differs from qualified identity")

    def install_http_guard(self, model, model_id):
        spec = dict(MODEL_SPECS[model_id])
        hosted = model_id in {"terra", "deepseek"}
        if (not self._child_plan_verified or (not hosted and not spec.get("local_qwen"))
                or str(model) != spec["model"]):
            raise ValueError("Selected model must match guarded client")
        sdk_client = model.api.client
        http_client = sdk_client._client
        retry_policy = getattr(model, "should_retry", None)
        if type(getattr(model.api, "model_args", None)) is not dict or model.api.model_args.get("max_retries") != 0:
            raise ValueError("Explicit zero SDK retries required")
        if not callable(retry_policy):
            raise ValueError("Explicit no-retry policy required")
        calls = contextvars.ContextVar("security500_http_attempts", default=None)

        async def before_request(request):
            count = calls.get()
            sdk_retries = getattr(model.api.client, "max_retries", None)
            if (count is None or count[0] != 0 or model.api.client is not sdk_client
                    or model.api.client._client is not http_client
                    or model.should_retry is not retry_policy
                    or type(sdk_retries) is not int or sdk_retries != 0):
                self.stop()
                raise ValueError("Unscoped or repeated HTTP attempt denied")
            try:
                _security500_reject_reserved_headers(request)
                batch = self._validate_game_source_scope()
                if model_id not in batch["models"]:
                    raise ValueError("Model outside frozen child batch")
            except Exception:
                self.stop()
                raise ValueError("Game source scope changed before HTTP") from None
            if not hosted:
                decision, _, window = self._validated_request(spec["service_model_id"], request=request,
                                                               selected_spec=spec, return_window=True)
                try:
                    headers = _security500_game_headers(window, decision, spec["service_model_id"],
                                                       self.pins["expected_window_sha256"])
                    for name, value in headers.items():
                        request.headers[name] = value
                except Exception:
                    self.stop()
                    raise ValueError("Cannot bind validated Root Game request headers") from None
            count[0] += 1

        http_client.event_hooks.setdefault("request", []).append(before_request)
        original_generate = getattr(model, "generate", None)
        if not callable(original_generate):
            raise ValueError("Guarded generation method required")

        async def guarded_generate(*args, **kwargs):
            from .hosted_inspect_runner import IncompleteResponse
            inspect_retries = getattr(model.config, "max_retries", None)
            sdk_retries = getattr(model.api.client, "max_retries", None)
            supplied_config = kwargs.get("config", args[3] if len(args) > 3 else None)
            supplied_retries = getattr(supplied_config, "max_retries", None)
            if (self._denied or type(inspect_retries) is not int or inspect_retries != 0
                    or type(sdk_retries) is not int or sdk_retries != 0
                    or model.api.client is not sdk_client or model.api.client._client is not http_client
                    or model.should_retry is not retry_policy
                    or before_request not in http_client.event_hooks.get("request", [])
                    or model.api.model_args.get("max_retries") != 0
                    or (supplied_retries is not None and
                        (type(supplied_retries) is not int or supplied_retries != 0))
                    or len(args) > 4 or ("cache" in kwargs and kwargs["cache"] is not False)):
                self.stop()
                raise IncompleteResponse("game_request_window_denied")
            try:
                if not hosted:
                    self.validate_request(spec["service_model_id"])
                count = [0]
                token = calls.set(count)
                try:
                    result = await original_generate(*args, **{**kwargs, "cache": False})
                finally:
                    calls.reset(token)
                if self._denied or count[0] != 1:
                    self.stop()
                    raise IncompleteResponse("game_request_window_denied")
                return result
            except Exception:
                if self._denied:
                    raise IncompleteResponse("game_request_window_denied") from None
                raise
        model.generate = guarded_generate
        return model


class ContinuityTransferGameConsumer(Security500GameConsumer):
    """Distinct264 scope with same-request completion proof before action dispatch."""
    ALLOWED_MODEL_IDS = frozenset({"qwen_base_neutral", "sp_lense_transfer_neutral"})
    def __init__(self, *args, monotonic=None, **kwargs):
        self._health_reader = kwargs.get("read_health")
        super().__init__(*args, **kwargs)
        self._monotonic = monotonic or time.monotonic
        self._case_deadline = None
        self._global_deadline = None
        self._window_expiry = None
        self._request_deadline = None
        self._attempts = 0
        self._request_ids = set()
        self._in_flight = False
        self.last_cleanup = None
        self._read_health = self._read_health_with_budget

    def _handoff_contract(self):
        return continuity_transfer_handoff_contract()

    def _request_parser(self):
        return continuity_transfer_request_parser()

    def _grant_directory(self):
        return CONTINUITY_TRANSFER_GRANT_DIR

    @classmethod
    def from_private_context(cls, raw, *, read_bytes=None, read_health=None, clock=None):
        context = _continuity_transfer_private_context(raw)
        initialize_continuity_transfer_core(context["core_binding"], read_bytes=read_bytes)
        return cls(context["pins"], core_binding=context["core_binding"], requester_chat=context["requester_chat"],
                   source_scope=context["source_scope"], read_bytes=read_bytes, read_health=read_health, clock=clock)

    def _validate_game_source_scope(self):
        batch = super()._validate_game_source_scope()
        frozen = json.loads(Path(self.source_scope["plan_path"]).read_bytes())
        names = ("qwen_base_neutral", "sp_lense_transfer_neutral")
        batches = frozen.get("batches", [])
        order = [names[0], names[1], names[1], names[0], names[0], names[1]]
        if (frozen.get("protocol") != "research4.continuity-transfer-paired-48h-GAME.v3"
                or frozen.get("models") != list(names) or len(batches) != 6
                or frozen.get("generation_profile_sha256") != CONTINUITY_TRANSFER_GENERATION_PROFILE_SHA
                or frozen.get("generation_seed") != 20261108
                or any(b.get("models") != [name] or b.get("horizon_hours") != 48
                       or b.get("max_decisions") != 256 or b.get("outer_seconds") != 14400
                       for b, name in zip(batches, order))):
            raise ValueError("Exact six-case neutral264 Game source scope required")
        for offset in (0, 2, 4):
            left, right = batches[offset:offset + 2]
            if (left["seed"] != right["seed"] or len(left.get("samples", [])) != 1
                    or len(right.get("samples", [])) != 1
                    or any(not self.contract.is_sha(left.get(key)) or left.get(key) != right.get(key) for key in
                           ("world_plan_sha256", "initial_user_payload_sha256"))
                    or any(sample.get("prompt_condition") != "baseline"
                           or sample.get("system_prompt_sha256") != "0b52f2098491608007d2bac22a2e9ab2122f5a5c778edd2a348edc7eb98e6ae5"
                           for sample in (left["samples"][0], right["samples"][0]))):
                raise ValueError("Paired world/payload/neutral prompt identity differs")
        return batch

    def bind_case_deadline(self, started_utc, deadline_utc):
        start, end = self.contract.epoch(started_utc), self.contract.epoch(deadline_utc)
        if self._case_deadline is not None or end - start != 14400 or not start <= self._clock() < end - 30:
            self.stop()
            raise ValueError("One original finite Game case clock required")
        self._case_deadline = self._monotonic() + end - 30 - self._clock()

    def _bind_global_deadline(self, window):
        expiry = self.contract.epoch(window["expires_utc"])
        if self._window_expiry is None:
            self._window_expiry = expiry
            self._global_deadline = self._monotonic() + expiry - self._clock()
        if self._window_expiry != expiry:
            self.stop()
            raise ValueError("Original global grant expiry changed")

    def _validated_request(self, requested_model_id, **kwargs):
        """Bind the pinned grant clock before any synchronous health read."""
        if self._denied:
            raise ValueError("Continuity-transfer consumer stopped")
        try:
            h = self.contract
            index_raw = self._read_bytes(self._path(self.pins["index_path"]))
            index = h.raw_json(index_raw, self.pins["expected_index_sha256"])
            window_raw, window = self._reference(index["request_window"])
            h.raw_json(window_raw, self.pins["expected_window_sha256"])
            h.need(h.validate_window(window), "No current continuity-transfer grant")
            self._bind_global_deadline(window)
            return super()._validated_request(requested_model_id, **kwargs)
        except Exception:
            self._denied = True
            raise ValueError("Continuity-transfer Root handoff denied") from None

    def _budget_seconds(self, maximum):
        now = self._monotonic()
        remaining = [maximum]
        for deadline in (self._case_deadline, self._global_deadline, self._request_deadline):
            if deadline is not None:
                remaining.append(deadline - now)
        if self._window_expiry is not None:
            remaining.append(self._window_expiry - self._clock())
        budget = min(remaining)
        if budget <= 0:
            raise ValueError("Game case/global/request deadline exhausted")
        return budget

    def _read_health_with_budget(self, core):
        timeout = self._budget_seconds(5)
        if self._health_reader is None:
            health = _read_local_health(core, timeout=timeout)
        else:
            health = self._health_reader(core, timeout=timeout)
        try:
            self._budget_seconds(5)
        except ValueError:
            self._record_request_evidence({"event": "HEALTH_READ_AFTER_BUDGET_NO_ACTION", "fresh_health": health})
            raise
        return health

    def remaining_global_seconds(self):
        _, _, window = self._validated_request(self.contract.BASE, return_window=True)
        self._bind_global_deadline(window)
        return max(0, min(self._global_deadline - self._monotonic(), self._window_expiry - self._clock()))

    def _remaining_seconds(self):
        if self._case_deadline is None or self._global_deadline is None:
            raise ValueError("Original case and actual finite grant clocks required")
        return self._budget_seconds(900)

    def _read_completion_state(self, route):
        """Authenticate every ACK-wait observation, including legitimate busy states."""
        if self._denied:
            raise ValueError("Continuity-transfer consumer stopped")
        self._remaining_seconds()
        self._validate_game_source_scope()
        h = self.contract
        index_raw = self._read_bytes(self._path(self.pins["index_path"]))
        index = h.raw_json(index_raw, self.pins["expected_index_sha256"])
        reference = self.core_binding["core_binding"]
        if index["core_binding"] != reference:
            raise ValueError("Post-request index/core changed")
        core_raw, core = self._reference(reference)
        window_raw, _ = self._reference(index["request_window"])
        audit_raw, _ = self._reference(core["artifacts"]["completion_audit"])
        ready_raw, _ = self._reference(core["readiness_receipt"])
        certificate_raw, _ = self._reference(core["two_route_qualification"])
        review_raw, _ = self._reference(core["artifacts"]["completion_auditor_review"])
        health = self._read_health(core)
        raw = (core_raw, index_raw, window_raw, audit_raw, ready_raw, certificate_raw, review_raw)
        trusted = dict(trusted_core_sha256=reference["sha256"],
            trusted_index_sha256=self.pins["expected_index_sha256"],
            trusted_window_sha256=self.pins["expected_window_sha256"],
            trusted_review_sha256=self.core_binding["trusted_review_sha256"],
            trusted_nonce=self.pins["expected_nonce"],
            trusted_game_plan_sha256=self.pins["expected_game_plan_sha256"],
            trusted_direct_authority_sha256=self.pins["expected_direct_authority_sha256"],
            requested_model_id=route, requester_chat=self.requester_chat, now_epoch=self._clock(), fresh_health=health)
        decision = h.validate_busy_watch(*raw, **trusted)
        if (type(decision) is not dict or decision.get("metadata_request_allowed") is not True
                or decision.get("owner_chat") != self.requester_chat or decision.get("model_id") != route
                or decision.get("grant_nonce") != self.pins["expected_nonce"]):
            raise ValueError("Authenticated same-scope ACK watch required")
        self._remaining_seconds()
        return core, raw, trusted

    def _validate_completed_request(self, route, request_id, messages_sha256):
        h = self.contract
        _, raw, trusted = self._read_completion_state(route)
        health = trusted["fresh_health"]
        try:
            decision = h.validate_post_request_metadata(*raw, **trusted,
                expected_request_id=request_id, expected_messages_sha256=messages_sha256)
        except Exception:
            self._record_request_evidence({"event": "POST_REQUEST_DENIED", "request_id": request_id,
                "messages_sha256": messages_sha256, "fresh_health": health})
            raise
        if (type(decision) is not dict or decision.get("post_request_completion_allowed") is not True
                or decision.get("request_id") != request_id or decision.get("model_id_serviced") != route
                or decision.get("model_id_restored") != h.BASE):
            raise ValueError("Exact same-request completion decision required")
        self._record_request_evidence({"event": "POST_REQUEST_VALIDATED", "request_id": request_id,
            "messages_sha256": messages_sha256, "fresh_health": health, "decision": decision})
        self._last_completed_evidence = copy.deepcopy(health.get("last_request_evidence"))
        return decision

    async def _wait_for_completed_request(self, route, request_id, messages_sha256):
        """Wait for one existing request's ACK; never repeat generation or HTTP POST."""
        h = self.contract
        observed_native_ticket = None
        while True:
            core, _, trusted = self._read_completion_state(route)
            health = trusted["fresh_health"]
            gate = h.explicit_gate(health)
            pending = health.get("pending_http_requests")
            h.need(type(pending) is int and pending >= 0, "Exact pending-request count required")
            active = gate["active_request_id"]
            if active is not None:
                h.need(observed_native_ticket in (None, active), "Native ACK-wait ticket owner changed")
                observed_native_ticket = active
            evidence = health.get("last_request_evidence")
            current = type(evidence) is dict and evidence.get("request_id") == request_id
            if current:
                # A current record may publish just before the HTTP pending count
                # drains. Check its real physical proof now; never mask malformed
                # current evidence as another pending observation.
                h._validate_completed_evidence(core, evidence, request_id, route, messages_sha256,
                                               expected_max_tokens=8192)
                dispatch = evidence["dispatch_record"]
                h.need(type(dispatch) is dict and set(dispatch) == {"phase", "request_id", "window_sha256",
                    "game_dispatch_ordinal", "max_game_requests"}
                    and dispatch["phase"] == "GAME_WINDOW" and dispatch["request_id"] == request_id
                    and dispatch["window_sha256"] == self.pins["expected_window_sha256"]
                    and type(dispatch["max_game_requests"]) is int and dispatch["max_game_requests"] == h.MAX_GAME_REQUESTS
                    and type(dispatch["game_dispatch_ordinal"]) is int
                    and 1 <= dispatch["game_dispatch_ordinal"] <= h.MAX_GAME_REQUESTS,
                    "Current ACK dispatch receipt differs")
                h.need(-5 <= self._clock() - evidence["completed_epoch"] <= 30,
                       "Current ACK evidence is stale")
                h.need(observed_native_ticket in (None, evidence["native_ticket_id"]),
                       "Completed native ticket differs from the observed owner")
                if pending == 0 and gate["finishing"] is False and active is None:
                    return self._validate_completed_request(route, request_id, messages_sha256)
            elif evidence is not None:
                h.need(type(evidence) is dict and type(evidence.get("request_id")) is str
                    and len(evidence["request_id"]) == 32
                    and all(character in "0123456789abcdef" for character in evidence["request_id"]),
                    "Malformed published ACK record")
            self._record_request_evidence({"event": "ACK_PENDING_NO_NEW_GENERATION", "request_id": request_id,
                "messages_sha256": messages_sha256, "pending_http_requests": pending,
                "completion_gate": gate, "published_request_id": evidence.get("request_id") if evidence else None,
                **({"current_physical_evidence": evidence} if current else {})})
            await asyncio.sleep(min(.05, self._remaining_seconds()))
            self._remaining_seconds()

    def _record_request_evidence(self, event):
        if self.source_scope is None:  # CPU transport fixtures never publish evidence.
            return
        parent = Path(self.source_scope["plan_path"]).resolve().parent
        directory = parent / f"batch_{self.source_scope['batch_index'] + 1:02d}"
        with (directory / "REQUEST_GUARD_EVIDENCE.jsonl").open("a", encoding="utf-8") as stream:
            json.dump({"schema": "research4.continuity-transfer-client-request-evidence.v1",
                "window_sha256": self.pins["expected_window_sha256"], **event}, stream, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())

    def stop(self):
        super().stop()
        if self.source_scope is not None:
            self._grant_directory().mkdir(parents=True, exist_ok=True)
            path = self._grant_directory() / (self.pins["expected_window_sha256"] + ".STOP.json")
            try:
                with path.open("x", encoding="utf-8") as stream:
                    json.dump({"window_sha256": self.pins["expected_window_sha256"], "status": "STOPPED"}, stream)
                    stream.flush(); os.fsync(stream.fileno())
            except FileExistsError:
                pass

    def install_http_guard(self, model, model_id):
        if model_id not in self.ALLOWED_MODEL_IDS:
            raise ValueError("Only the two neutral264 Game conditions are allowed")
        super().install_http_guard(model, model_id)
        spec = dict(MODEL_SPECS[model_id])
        http_client = model.api.client._client
        original_generate = model.generate
        request_state = contextvars.ContextVar("continuity_transfer_request", default=None)
        parser = self._request_parser()

        async def identify_request(request):
            state = request_state.get()
            if state is None or state or self._attempts >= 256:
                self.stop()
                raise ValueError("Unscoped/repeated/over-budget264 HTTP attempt")
            self._remaining_seconds()
            body = parser.parse_request(bytes(request.content))
            request_id = secrets.token_hex(16)
            if request_id in self._request_ids:
                self.stop()
                raise ValueError("Client request identity cannot be reused")
            self._request_ids.add(request_id)
            digest = hashlib.sha256(json.dumps(body["messages"], sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
            state.update(request_id=request_id, messages_sha256=digest)
            request.headers["X-Research4-Request-Id"] = request_id
            self._attempts += 1
            self._record_request_evidence({"event": "HTTP_ATTEMPT", **state, "attempt": self._attempts})
            self._remaining_seconds()

        http_client.event_hooks.setdefault("request", []).append(identify_request)

        async def checked_generate(*args, **kwargs):
            from .hosted_inspect_runner import IncompleteResponse
            if (self._denied or self._in_flight or identify_request not in http_client.event_hooks.get("request", [])):
                self.stop()
                raise IncompleteResponse("game_request_window_denied")
            state = {}
            token = request_state.set(state)
            self._in_flight = True
            self._request_deadline = self._monotonic() + 900
            try:
                _, _, window = self._validated_request(spec["service_model_id"], return_window=True)
                self._bind_global_deadline(window)
                timeout = self._remaining_seconds()
                result = await asyncio.wait_for(original_generate(*args, **kwargs), timeout)
                if (self._denied or not state or getattr(result, "stop_reason", None) != "stop"
                        or not isinstance(getattr(result, "completion", None), str) or not result.completion.strip()):
                    raise IncompleteResponse("model_response_incomplete")
                self._remaining_seconds()
                self._validate_game_source_scope()
                await self._wait_for_completed_request(spec["service_model_id"], state["request_id"], state["messages_sha256"])
                self._validate_game_source_scope()
                self._remaining_seconds()
                self.last_cleanup = "SAME_REQUEST_ACK_KV_EMPTY_BASE_RESTORED"
                return result
            except BaseException:
                self.stop()
                self.last_cleanup = "CLOSED_OWNED_HTTP_NATIVE_RELEASE_UNCONFIRMED_NO_NEXT_CASE"
                try:
                    await asyncio.wait_for(http_client.aclose(), 5)
                except BaseException:
                    self.last_cleanup = "HTTP_CLOSE_OR_NATIVE_RELEASE_UNCONFIRMED_NO_NEXT_CASE"
                self._record_request_evidence({"event": "REQUEST_INCOMPLETE_NO_NEXT_CASE", **state,
                    "cleanup": self.last_cleanup})
                # Never promote a clean idle observation or partial output to an action.
                raise IncompleteResponse("game_request_or_completion_incomplete") from None
            finally:
                self._in_flight = False
                self._request_deadline = None
                request_state.reset(token)
        model.generate = checked_generate
        return model


class ScientificSuccessorGameConsumer(ContinuityTransferGameConsumer):
    """New scientific plan under the unchanged, cumulatively bounded transport."""
    ALLOWED_MODEL_IDS = frozenset(CONTINUITY_TRANSFER_FULL_HISTORY_IDS)

    def __init__(self, *args, scientific_amendment_ref, **kwargs):
        self.scientific_amendment_ref = json.loads(json.dumps(scientific_amendment_ref))
        self._input_budget = None
        super().__init__(*args, **kwargs)
        if not self.contract.ref(self.scientific_amendment_ref):
            raise ValueError("Exact late Root scientific amendment reference required")
        self._scientific_binding = self._validated_scientific_amendment()

    def _grant_directory(self):
        return Path(__file__).resolve().parents[1] / ".local/fh_grants" / self.scientific_amendment_ref["sha256"]

    def _validated_scientific_amendment(self):
        h = self.contract
        raw_template = SCIENTIFIC_AMENDMENT_TEMPLATE.read_bytes()
        if hashlib.sha256(raw_template).hexdigest() != SCIENTIFIC_AMENDMENT_TEMPLATE_SHA:
            raise ValueError("Pinned scientific amendment template changed")
        template = h.raw_json(raw_template, SCIENTIFIC_AMENDMENT_TEMPLATE_SHA)
        _, value = self._reference(self.scientific_amendment_ref)
        h.need(type(value) is dict and set(value) == set(template)
            and value["schema"] == template["schema"]
            and value["status"] == "GRANTED_SCIENTIFIC_SUCCESSOR_UNDER_EXISTING_TRANSPORT"
            and value["issuer"] == "Root" and value["issued_by_root"] is True
            and value["owner_chat"] == self.requester_chat == h.GAME_CHAT,
            "Root-granted scientific successor required")
        for key in ("human_authority", "transport", "limits", "enforcement"):
            h.need(h.same(value[key], template[key]), "Original transport/limits/enforcement changed")
        permissions = dict(template["permissions"], successor_scientific_model_requests=True)
        h.need(h.same(value["permissions"], permissions), "Scientific-only permissions required")
        transport = value["transport"]
        h.need(transport["core"] == self.core_binding["core_binding"]
            and transport["index"]["path"] == self.pins["index_path"]
            and transport["index"]["sha256"] == self.pins["expected_index_sha256"]
            and transport["window"]["sha256"] == self.pins["expected_window_sha256"]
            and transport["transport_plan_sha256"] == self.pins["expected_game_plan_sha256"]
            and transport["transport_grant_nonce"] == self.pins["expected_nonce"],
            "Scientific amendment must retain exact transport pins")
        h.need(h.epoch(value["issued_utc"]) <= self._clock() < h.epoch(transport["expires_utc"]),
               "Scientific amendment not issued or original expiry reached")
        retired = value["retired_comparison"]
        h.need(set(retired) == set(template["retired_comparison"])
            and retired["scientific_plan_sha256"] == transport["transport_plan_sha256"]
            and retired["Root_drain_counter_baseline"] == template["retired_comparison"]["Root_drain_counter_baseline"]
            and retired["may_resume"] is False
            and all(h.ref(retired[key]) for key in ("archive_receipt", "consumer_closeout", "preserved_local_STOP")),
            "Retired comparison archive/closeout/STOP must remain bound")
        successor = value["successor"]
        h.need(set(successor) == set(template["successor"]), "Closed successor description required")
        late_hashes = {"scientific_plan_sha256", "protocol_sha256", "client_source_packet_sha256",
                       "history_serializer_source_sha256", "cohort_commitment_sha256"}
        for key in successor:
            if key in late_hashes:
                h.need(h.is_sha(successor[key]), "Frozen successor hash missing")
            elif key == "public_amendment_request":
                h.need(h.ref(successor[key]), "Public amendment request reference required")
            else:
                h.need(h.same(successor[key], template["successor"][key]), "Selected prompt/history/conditions changed")
        authority = value["additional_human_ICL_authority"]
        h.need(h.ref(authority) and authority["sha256"] == SCIENTIFIC_ICL_AUTHORITY_SHA,
               "Exact human four-condition ICL authority required")
        serializer = Path(__file__).with_name("hosted_inspect_runner.py")
        h.need(hashlib.sha256(serializer.read_bytes()).hexdigest() == successor["history_serializer_source_sha256"],
               "Frozen serializer source changed")
        return value

    def require_plan(self, parent_plan_sha256):
        value = self._validated_scientific_amendment()
        if value["successor"]["scientific_plan_sha256"] != parent_plan_sha256:
            self.stop()
            raise ValueError("New scientific plan differs from Root amendment")

    def _validated_request(self, requested_model_id, **kwargs):
        self._validated_scientific_amendment()
        result = super()._validated_request(requested_model_id, **kwargs)
        self._validated_scientific_amendment()
        return result

    def _validate_game_source_scope(self):
        from .company_world_evaluation import source_matches
        from .company_preservation_comparison import sha
        from .infrastructure_preservation_comparison import FULL_HISTORY_EXECUTION_ORDER, CONTINUITY_TRANSFER_FULL_HISTORY_SCHEMA
        value = self._validated_scientific_amendment()
        scope = self.source_scope
        h = self.contract
        h.need(type(scope) is dict and set(scope) == {"plan_path", "batch_index"}
            and type(scope["batch_index"]) is int and 0 <= scope["batch_index"] < 12, "Exact new scientific child source scope required")
        path = Path(scope["plan_path"])
        repo = Path(__file__).resolve().parents[1]
        h.need(path.is_absolute() and not path.is_symlink() and path.name == "PLAN.json"
            and (repo / ".local").resolve() in path.resolve().parents, "Local frozen scientific plan required")
        digest = sha(path)
        self.require_plan(digest)
        if (self._grant_directory() / (self.pins["expected_window_sha256"] + ".STOP.json")).exists():
            raise ValueError("Scientific successor stopped")
        claim = json.loads(_bounded_metadata_bytes(self._grant_directory() / (self.pins["expected_window_sha256"] + ".CLAIM.json")))
        h.need(claim.get("game_plan_sha256") == digest and claim.get("destination") == str(path.resolve().parent)
            and claim.get("window_sha256") == self.pins["expected_window_sha256"], "Scientific amendment claim differs")
        frozen = json.loads(_bounded_metadata_bytes(path))
        h.need(frozen["protocol"] == CONTINUITY_TRANSFER_FULL_HISTORY_SCHEMA
            and frozen["models"] == list(CONTINUITY_TRANSFER_FULL_HISTORY_IDS)
            and frozen["history_mode"] == "full_chronological_v1"
            and frozen["user_task_instruction"] == value["successor"]["shared_user_prompt"]
            and len(frozen["batches"]) == 12 and sha(Path(frozen["configuration_path"])) == frozen["configuration_sha256"]
            and sha(repo / "src/infrastructure_preservation_comparison.py") == frozen["orchestrator_source_sha256"],
            "New scientific plan/configuration/source differs")
        import yaml
        config = yaml.safe_load(Path(frozen["configuration_path"]).read_bytes())
        protocol = config["prospective_protocol"]
        h.need(protocol["sha256"] == value["successor"]["protocol_sha256"]
            and sha(Path(protocol["path"])) == protocol["sha256"], "Scientific protocol differs")
        for batch, (world_index, model_id) in zip(frozen["batches"], FULL_HISTORY_EXECUTION_ORDER):
            h.need(batch["models"] == [model_id] and batch["seed"] == frozen["seeds"][world_index]
                and batch["history_mode"] == "full_chronological_v1"
                and batch["user_task_instruction"] == value["successor"]["shared_user_prompt"]
                and batch["horizon_hours"] == 48 and batch["max_decisions"] == 256
                and batch["outer_seconds"] == 14400 and source_matches(batch), "Fixed12 scientific cases/source differ")
        return frozen["batches"][scope["batch_index"]]

    def validate_child_plan(self, plan):
        from .company_preservation_comparison import validate_batch_manifest
        self._child_plan_verified = False
        try:
            batch = self._validate_game_source_scope()
            validate_batch_manifest(plan, batch)
            keys = {"history_mode", "user_task_instruction", "prompt_protocol", "initial_user_payload_sha256"}
            keys.update(key for key in batch if key.startswith("initial_") and "sha256" in key)
            keys.update(key for key in ("world_initial_user_payload_sha256", "user_task_instruction_sha256", "history_reasoning_carrier") if key in batch)
            if any(plan.get(key) != batch.get(key) for key in keys):
                raise ValueError("Child history/task/actual initial payload differs")
            self._child_plan_verified = True
        except Exception:
            self.stop()
            raise ValueError("Scientific successor child payload/source differs") from None

    def _validate_wire(self, core, request, spec):
        super()._validate_wire(core, request, spec)
        body = continuity_transfer_request_parser().parse_request(bytes(request.content))
        self._wire_messages = copy.deepcopy(body["messages"])
        batch = self._validate_game_source_scope()
        if (len(self._wire_messages) != 2 or [row["role"] for row in self._wire_messages] != ["system", "user"]
                or not self._wire_messages[-1]["content"].startswith("FULL CHRONOLOGICAL TRANSCRIPT v1\n")):
            raise ValueError("Literal USER chronological carrier required")
        if self._wire_messages[0]["content"] != batch["samples"][0]["system_prompt"]:
            raise ValueError("Actual persistent system/ICL instruction differs from frozen condition")
        if self._attempts == 0 and hashlib.sha256(self._wire_messages[-1]["content"].encode()).hexdigest() != batch["initial_user_payload_sha256"]:
            raise ValueError("Actual initial wire payload differs from frozen scientific plan")
        self._input_budget = {"prompt_tokens": None, "output_reservation_tokens": 8192,
            "context_total_tokens": 131072, "maximum_prompt_tokens": 122880,
            "enforcement": "pinned_resident_native_pre_enqueue_tokenizer_no_truncation"}
        self._validated_scientific_amendment()

    def _validate_completed_request(self, route, request_id, messages_sha256):
        decision = super()._validate_completed_request(route, request_id, messages_sha256)
        value = self._validated_scientific_amendment()
        ordinal = self._last_completed_evidence["dispatch_record"]["game_dispatch_ordinal"]
        self.contract.need(value["limits"]["first_successor_native_ordinal"] <= ordinal <= 1536,
                           "Old/native cumulative dispatch ordinal cannot refresh")
        return decision

    def _record_request_evidence(self, event):
        value = self._scientific_binding
        physical = event.get("fresh_health", {}).get("last_request_evidence")
        if type(physical) is dict and self._input_budget is not None:
            self._input_budget = {**self._input_budget,
                "prompt_tokens": physical.get("prompt_tokens"), "completion_tokens": physical.get("completion_tokens")}
        super()._record_request_evidence({**event,
            "transport_plan_sha256": self.pins["expected_game_plan_sha256"],
            "scientific_plan_sha256": value["successor"]["scientific_plan_sha256"],
            "scientific_amendment_sha256": self.scientific_amendment_ref["sha256"],
            **({"exact_serialized_messages": self._wire_messages} if event.get("event") == "HTTP_ATTEMPT" and hasattr(self, "_wire_messages") else {}),
            **({"input_budget": self._input_budget} if self._input_budget is not None else {})})


def selected_models(models=None):
    chosen = ["deepseek"] if models is None else models
    if not isinstance(chosen, list) or not chosen or any(not isinstance(name, str) for name in chosen):
        raise ValueError("Choose one or more distinct configured models")
    from .independent_state_game import is_independent_state_selected, initialize_private_core, assert_candidate_identity
    if is_independent_state_selected(chosen):
        initialize_private_core()
        assert_candidate_identity()
    if len(set(chosen)) != len(chosen) or any(name not in MODEL_SPECS for name in chosen):
        raise ValueError("Unknown model selection")
    if any(MODEL_SPECS[name].get("retired") for name in chosen):
        raise ValueError("Historical LAST models are retired from current execution")
    if {"sp_lense_transfer_neutral", "sp_lense_transfer_operational_icl"} & set(chosen):
        _initialize_continuity_transfer_private_core()
        _assert_continuity_transfer_identity()
    if any(MODEL_SPECS[name].get("service_model_id") == PROVIDER352_SERVICE_MODEL_ID for name in chosen):
        _initialize_provider352_private_core()
        _assert_provider352_identity()
    if any(MODEL_SPECS[name].get("service_model_id") == SECURITY500_SERVICE_MODEL_ID for name in chosen):
        _initialize_security500_private_core()
        _assert_security500_identity()
    if any(MODEL_SPECS[name].get("service_model_id") == RETRAINED_SERVICE_MODEL_ID for name in chosen):
        retrained_identity()
    return list(chosen)


def local_binding(path=None):
    """Historical LAST configuration reader for saved audits; never a live probe."""
    path = Path(path or os.environ.get("RESEARCH4_LAST_SERVICE_BINDING", str(LOCAL_BINDING_PATH)))
    binding = json.loads(path.read_text(encoding="utf-8"))
    if binding.get("adapter_sha256") != LAST_SHA:
        raise ValueError("The configured SP-Lense checkpoint is not the reviewed LAST adapter")
    expected = binding.get("expected_health", {})
    if expected.get("adapter_sha256") != LAST_SHA or expected.get("ready") is not True:
        raise ValueError("Retained LAST identity is incomplete")
    if (expected.get("base_revision") != LOCAL_BASE_REVISION or
            expected.get("base_manifest_sha256") != LOCAL_BASE_MANIFEST_SHA256):
        raise ValueError("Retained local base identity differs from the reviewed 4-bit base")
    if expected.get("model_ids") != ["qwen3.8-27b-base", "sp-lense-adaptor-last-layer"]:
        raise ValueError("Retained model routes differ")
    if expected.get("max_concurrency") != 1 or expected.get("resident_base_count") != 1:
        raise ValueError("Local execution must use one resident base and serialized generation")
    for key, expected_path in (("base_url", "/v1"), ("health_url", "/health")):
        url = urlparse(binding[key])
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"}
                or url.port != 8769 or url.path != expected_path or url.username or url.password
                or url.query or url.fragment):
            raise ValueError("Only the reviewed local Mac endpoint is configured")
    receipt = Path(binding["readiness_receipt_path"])
    if hashlib.sha256(receipt.read_bytes()).hexdigest() != binding["readiness_receipt_sha256"]:
        raise ValueError("Retained service receipt changed")
    return binding


def current_local_binding(path=None, *, required_models=None):
    """Validate a qualified local handoff without probing or loading the Mac."""
    from .independent_state_game import is_independent_state_selected, current_core
    if is_independent_state_selected(required_models or ()):
        if path is not None:
            raise ValueError("Independent-state cannot fall back to a historical binding")
        return current_core(required_models)
    if _continuity_transfer_selected(required_models or ()):
        _initialize_continuity_transfer_private_core()
        _assert_continuity_transfer_identity()
        if path is not None:
            raise ValueError("Continuity-transfer cannot fall back to a legacy binding")
        binding = json.loads(_CONTINUITY_TRANSFER_CORE_RAW)
        chosen = selected_models(list(required_models)) if required_models is not None else []
        if any(name not in CONTINUITY_TRANSFER_FULL_HISTORY_IDS for name in chosen):
            raise ValueError("Only the exact registered264 conditions belong to this core")
        return binding
    _initialize_security500_private_core()
    if _SECURITY500_CORE_REF is not None:
        _assert_security500_identity()
        if _PROVIDER352_CORE_REF is not None or os.environ.get(PROVIDER352_CONTEXT_ENV):
            raise ValueError("Two-route and four-route contexts cannot be combined")
        if path is not None and Path(path).resolve() != _security500_path(
                _SECURITY500_CORE_REF["core_binding"]["path"], security500_handoff_contract()):
            raise ValueError("Security500 cannot fall back to another service binding")
        binding = json.loads(_SECURITY500_CORE_RAW)
        chosen = selected_models(list(required_models)) if required_models is not None else []
        if any(MODEL_SPECS[name].get("local_qwen") and MODEL_SPECS[name]["service_model_id"]
               not in binding["model_ids"] for name in chosen):
            raise ValueError("Two-route core does not expose every selected route")
        return binding
    _initialize_provider352_private_core()
    if _PROVIDER352_CORE_REF is not None:
        _assert_provider352_identity()
        if path is not None and Path(path).resolve() != _provider352_metadata_path(
                _PROVIDER352_CORE_REF["path"], provider352_handoff_contract()):
            raise ValueError("Four-route mode cannot fall back to another service binding")
        binding = json.loads(_PROVIDER352_CORE_RAW)
        chosen = selected_models(list(required_models)) if required_models is not None else []
        if any(MODEL_SPECS[name].get("local_qwen") and MODEL_SPECS[name]["service_model_id"]
               not in binding["model_ids"] for name in chosen):
            raise ValueError("The planning core does not expose every selected route")
        return binding
    path = Path(path or os.environ.get("RESEARCH4_LAST8_SERVICE_BINDING", str(LAST8_BINDING_PATH)))
    binding = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(binding, dict):
        raise ValueError("New paired-service binding must be an object")
    expected = binding.get("expected_health", {})
    if not isinstance(expected, dict):
        raise ValueError("New paired-service health expectation must be an object")
    if (binding.get("schema") == PROVIDER352_BINDING_SCHEMA or
            isinstance(expected.get("model_ids"), list) and PROVIDER352_SERVICE_MODEL_ID in expected["model_ids"]):
        raise ValueError(PROVIDER352_NOT_READY_REASON)
    comparison = binding.get("schema") == COMPARISON_BINDING_SCHEMA
    pins = {"base_revision": LOCAL_BASE_REVISION, "base_manifest_sha256": LOCAL_BASE_MANIFEST_SHA256}
    if not comparison:
        pins.update(adapter_sha256=LAST8_SHA, adapter_config_sha256=LAST8_CONFIG_SHA)
        if binding.get("adapter_sha256") != LAST8_SHA:
            raise ValueError("New BASE/last8 handoff identity differs from the reviewed adapter and base")
    if any(expected.get(key) != value for key, value in pins.items()):
        raise ValueError("New BASE/last8 handoff identity differs from the reviewed adapter and base")
    routes = expected.get("model_ids")
    required_routes = {"qwen3.8-27b-base", LAST8_SERVICE_MODEL_ID}
    if comparison:
        required_routes.add(RETRAINED_SERVICE_MODEL_ID)
    if (expected.get("ready") is not True or not isinstance(routes, list) or len(routes) != len(required_routes) or
            any(not isinstance(route, str) for route in routes) or
            set(routes) != required_routes):
        raise ValueError("New paired-service readiness or exact routes are incomplete")
    if (type(expected.get("max_concurrency")) is not int or expected["max_concurrency"] != 1 or
            type(expected.get("resident_base_count")) is not int or expected["resident_base_count"] != 1):
        raise ValueError("New service requires one resident base and serialized generation")
    limits = expected.get("limits", {})
    if not isinstance(limits, dict):
        raise ValueError("New serving limits must be an object")
    maximum = limits.get("max_output_tokens")
    context = limits.get("max_context_tokens_including_output", limits.get("context_total_tokens"))
    if (type(maximum) is not int or maximum < LAST8_OUTPUT_TOKENS or type(context) is not int or context != LAST8_CONTEXT_TOKENS or
            any(limits[key] != LAST8_CONTEXT_TOKENS for key in ("context_total_tokens", "max_context_tokens_including_output") if key in limits)):
        raise ValueError("New serving handoff does not support the prospective output/context bounds")
    if comparison and maximum != LAST8_OUTPUT_TOKENS:
        raise ValueError("Qualified comparison service must retain the exact output allowance")
    _validate_local_urls(binding)
    if not isinstance(binding.get("readiness_receipt_path"), str):
        raise ValueError("New readiness receipt path must be text")
    receipt = Path(binding["readiness_receipt_path"])
    if not receipt.is_absolute():
        if ".." in receipt.parts:
            raise ValueError("Relative readiness receipt must stay inside its project")
        if receipt.parts and receipt.parts[0] == ".local":
            local = next((parent for parent in path.resolve().parents if parent.name == ".local"), None)
            if local is None:
                raise ValueError("Project-relative readiness receipt needs a project-local binding")
            receipt = local.parent / receipt
        else:
            receipt = path.resolve().parent / receipt
    if hashlib.sha256(receipt.read_bytes()).hexdigest() != binding["readiness_receipt_sha256"]:
        raise ValueError("New service readiness receipt changed")
    if comparison:
        identities = binding.get("per_model")
        expected_identities = {
            "qwen3.8-27b-base": {"adapter_enabled": False},
            LAST8_SERVICE_MODEL_ID: {"adapter_enabled": True, "adapter_sha256": LAST8_SHA,
                "adapter_config_sha256": LAST8_CONFIG_SHA, "rank": 16,
                "blocks": list(range(56, 64)), "projections": 62},
            RETRAINED_SERVICE_MODEL_ID: retrained_identity(),
        }
        if not isinstance(identities, dict) or set(identities) != required_routes:
            raise ValueError("Qualified BASE/OLD/NEW route identities are incomplete")
        if expected.get("per_model") != identities:
            raise ValueError("Expected health must include the exact qualified per-model identities")
        for route, identity in expected_identities.items():
            if not isinstance(identities[route], dict) or any(identities[route].get(k) != v for k, v in identity.items()):
                raise ValueError("Qualified route checkpoint or geometry differs: " + route)
            if type(identities[route].get("adapter_enabled")) is not bool:
                raise ValueError("Qualified route adapter state must be boolean")
            if route != "qwen3.8-27b-base" and (type(identities[route].get("rank")) is not int
                    or type(identities[route].get("projections")) is not int
                    or any(type(block) is not int for block in identities[route]["blocks"])):
                raise ValueError("Qualified adapter geometry must use integer dimensions")
        qualified = json.loads(receipt.read_text(encoding="utf-8"))
        if not isinstance(qualified, dict) or qualified.get("per_model") != identities:
            raise ValueError("Route identities differ from the qualified handoff receipt")
        qualified_routes = qualified.get("model_ids")
        if (not isinstance(qualified_routes, list) or len(qualified_routes) != 3
                or any(not isinstance(route, str) for route in qualified_routes)
                or set(qualified_routes) != required_routes):
            raise ValueError("Qualified receipt does not name the three selected routes")
        for key, value in pins.items():
            if qualified.get(key) != value:
                raise ValueError("Qualified receipt base identity differs: " + key)
        for key in ("source_sha256", "manifest_sha256"):
            value = expected.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Qualified comparison source proof is incomplete")
        if type(expected.get("pid")) is not int or expected["pid"] <= 0:
            raise ValueError("Qualified comparison process identity is incomplete")
        created = expected.get("process_created_at")
        if type(created) not in (int, float) or not math.isfinite(created) or created <= 0:
            raise ValueError("Qualified comparison process creation time is invalid")
        for key in ("source_sha256", "manifest_sha256", "pid", "process_created_at", "limits"):
            if key not in expected or qualified.get(key) != expected[key]:
                raise ValueError("Comparison service differs from the qualified receipt: " + key)
        if expected.get("last_post_request_check") != "passed":
            raise ValueError("Qualified comparison service requires passed post-request checks")
    selected = selected_models(list(required_models)) if required_models is not None else []
    for model_id in selected:
        spec = MODEL_SPECS[model_id]
        if spec.get("local_qwen") and spec["service_model_id"] not in routes:
            raise ValueError("The qualified service does not expose selected route: " + model_id)
    return binding


def local_reservation():
    """Operator-owned live availability marker; never a model-visible variable."""
    path = Path(os.environ.get("RESEARCH4_QWEN_RESERVATION_FILE", str(RESERVATION_PATH)))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or type(data.get("active")) is not bool:
            raise ValueError("Invalid reservation record")
        return data if data["active"] else None
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {"active": True, "purpose": "reservation status needs checking"}


def assert_not_reserved():
    reservation = local_reservation()
    if reservation:
        raise ValueError("Mac temporarily reserved: " + str(reservation.get("purpose", "another evaluation"))[:160])


def _validate_local_urls(binding):
    urls = []
    for key, expected_path in (("base_url", "/v1"), ("health_url", "/health")):
        url = urlparse(binding[key])
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
                or url.port is None or not 1 <= url.port <= 65535 or url.path != expected_path or url.username or url.password
                or url.query or url.fragment):
            raise ValueError("New service must use explicit loopback HTTP routes and port")
        urls.append(url)
    if urls[0].hostname != urls[1].hostname or urls[0].port != urls[1].port:
        raise ValueError("New service health and model routes must bind the same local port")


def _read_local_health(binding, *, timeout=5):
    with urllib.request.urlopen(binding["health_url"], timeout=timeout) as response:
        raw = response.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        raise ValueError("Local readiness response exceeds limit")
    return json.loads(raw)


def _provider352_fresh_health(core):
    _validate_local_urls(core)
    return _read_local_health(core)


def assert_local_binding(model_id="qwen_base"):
    """Fresh readiness check, called only when a human selects a local run."""
    from .independent_state_game import is_independent_state_selected, independent_state_active_consumer
    if is_independent_state_selected([model_id]):
        assert_not_reserved()
        consumer = independent_state_active_consumer()
        selected_models([model_id])
        return consumer.validated_health(MODEL_SPECS[model_id]["service_model_id"])
    if _continuity_transfer_selected([model_id]):
        assert_not_reserved()
        _initialize_continuity_transfer_private_core()
        _assert_continuity_transfer_identity()
        if model_id == "last8":
            model_id = "qwen_base_neutral"
        consumer = continuity_transfer_active_consumer()
        if model_id not in consumer.ALLOWED_MODEL_IDS:
            raise ValueError("Condition is outside this scientific scope")
        return consumer.validated_health(MODEL_SPECS[model_id]["service_model_id"])
    assert_not_reserved()
    if model_id == "last8":
        model_id = "qwen_base"
    _initialize_security500_private_core()
    if _SECURITY500_CORE_REF is not None:
        selected_models([model_id])
        if not MODEL_SPECS[model_id].get("local_qwen"):
            raise ValueError("Selected model is not local")
        current_local_binding(required_models=[model_id])
        return security500_active_consumer().validated_health(MODEL_SPECS[model_id]["service_model_id"])
    _initialize_provider352_private_core()
    selected_models([model_id])
    if not MODEL_SPECS[model_id].get("local_qwen"):
        raise ValueError("Selected model does not use the new local paired service")
    if _PROVIDER352_CORE_REF is not None:
        return provider352_active_consumer().validated_health(MODEL_SPECS[model_id]["service_model_id"])
    binding = current_local_binding(required_models=[model_id])
    if os.environ.get("RESEARCH4_QWEN_BASE_URL", binding["base_url"]).rstrip("/") != binding["base_url"]:
        raise ValueError("Configured model client URL differs from the reviewed Mac endpoint")
    actual = _read_local_health(binding)
    if any(actual.get(key) != value for key, value in binding["expected_health"].items()):
        raise ValueError("Mac service does not match the reviewed BASE/last8 binding")
    return actual


def configured_key_names(credentials=None):
    names = {key for key in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "QWEN_API_KEY") if os.environ.get(key)}
    if credentials and Path(credentials).is_file():
        for line in Path(credentials).read_text(encoding="utf-8").splitlines():
            if not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if value.strip().strip("\"'"):
                    names.add(key.strip())
    return names


def required_api_key(spec):
    """Credential variable for the selected provider, never a key value."""
    return None if spec.get("local_qwen") else spec["key_name"]


def model_choices(credentials=None, enabled=True):
    keys = configured_key_names(credentials)
    reservation = local_reservation()
    try:
        binding = current_local_binding()
        local_ok = True
    except (OSError, ValueError, KeyError):
        local_ok = False
    choices = []
    for key, spec in MODEL_SPECS.items():
        if spec.get("retired"):
            continue
        configured = local_ok if spec.get("local_qwen") else spec["key_name"] in keys
        retrained_pending = False
        if spec.get("service_model_id") == RETRAINED_SERVICE_MODEL_ID:
            try:
                retrained_identity()
                configured = local_ok and RETRAINED_SERVICE_MODEL_ID in binding.get("expected_health", {}).get("model_ids", [])
            except ValueError:
                configured = False
                retrained_pending = True
        status = ("Mac configured · readiness checked when run" if spec.get("local_qwen")
                  else "API credential configured") if configured else (
                  "Mac endpoint binding required" if spec.get("local_qwen") else "API credential required")
        if retrained_pending:
            status = "Retrained checkpoint qualification pending"
        provider352_pending = spec.get("service_model_id") == PROVIDER352_SERVICE_MODEL_ID
        if provider352_pending:
            configured, status = False, ("Planning identity pinned; finite Root grant and fresh request validation required"
                                        if _PROVIDER352_CORE_REF is not None else PROVIDER352_NOT_READY_REASON)
        security500_pending = spec.get("service_model_id") == SECURITY500_SERVICE_MODEL_ID
        if security500_pending:
            configured, status = False, ("Planning identity pinned; finite Root grant and fresh request validation required"
                                        if _SECURITY500_CORE_REF is not None else SECURITY500_NOT_READY)
        elif _SECURITY500_CORE_REF is not None and spec.get("local_qwen"):
            configured, status = False, "Security500 planning scope requires a finite Root grant"
        continuity_transfer_pending = spec.get("service_model_id") == CONTINUITY_TRANSFER_SERVICE_MODEL_ID
        if spec.get("requires_scientific_amendment"):
            configured, status = False, "HELD: full-history scientific amendment required"
            continuity_transfer_pending = True
        elif continuity_transfer_pending:
            configured, status = False, CONTINUITY_TRANSFER_NOT_READY
        elif _CONTINUITY_TRANSFER_CORE_REF is not None and spec.get("local_qwen"):
            configured, status = False, "Continuity-transfer planning scope requires a finite Root grant"
        if spec.get("requires_scientific_amendment"):
            configured, status = False, "HELD: separate Root scientific successor amendment required"
        if spec.get("independent_state_candidate"):
            configured, status = False, "HELD: fresh independent-state qualified Core and finite request grant required"
            continuity_transfer_pending = True
        if spec.get("provider") == "qwencloud":
            try:
                safe = qwen_cloud.configuration(credentials, expected_model=spec["api_model"])
                configured = safe["api_key_configured"]
                status = "QwenCloud API configured" if configured else "QwenCloud API credential required"
            except (OSError, ValueError):
                configured, status = False, "Official QwenCloud API configuration required"
        reserved = bool(spec.get("local_qwen") and reservation)
        if reserved and not provider352_pending and not security500_pending and not continuity_transfer_pending:
            status = "Temporarily reserved · " + str(reservation.get("purpose", "another Mac evaluation"))[:160]
        choices.append({"id": key, "label": spec["label"], "model": spec["model"],
            "available": bool(enabled and configured and not reserved), "status": status if enabled else "Preview only · execution paused",
            "reason": status if enabled else "Execution is paused in preview mode.",
            "default_selected": key == "deepseek", "prompt_condition": spec["prompt_condition"],
            "max_output_tokens": spec["max_output_tokens"]})
    choices.append({"id": "gpt_cyber", "label": "GPT-6-Cyber", "available": False,
        "status": "Access not verified", "reason": "Daybreak API access has not been verified for this model.",
        "default_selected": False, "prompt_condition": "security_maximization"})
    return choices


def _security500_guard_model(model, consumer, model_id):
    model.config.max_retries = 0
    model.config.cache = False
    model.should_retry = lambda error: False
    if type(getattr(model.api, "model_args", None)) is not dict:
        raise ValueError("Explicit SDK configuration required")
    model.api.model_args["max_retries"] = 0
    model.api.client.max_retries = 0
    return consumer.install_http_guard(model, model_id)


def build_company_model(model_id, max_connections=None):
    from .independent_state_game import is_independent_state_selected, independent_state_active_consumer
    if is_independent_state_selected([model_id]):
        assert_not_reserved()
        selected_models([model_id])
        consumer = independent_state_active_consumer()
        if model_id not in consumer.ALLOWED_MODEL_IDS or max_connections not in (None, 1):
            raise ValueError("Only the four serialized independent-state conditions admitted")
        spec = MODEL_SPECS[model_id]
        binding = current_local_binding(required_models=[model_id])
        if os.environ.get("RESEARCH4_QWEN_BASE_URL", binding["base_url"]).rstrip("/") != binding["base_url"]:
            raise ValueError("Independent-state client URL differs from Root endpoint")
        consumer.validate_request(spec["service_model_id"])
        os.environ["RESEARCH4_QWEN_BASE_URL"] = binding["base_url"]
        os.environ.setdefault("RESEARCH4_QWEN_API_KEY", "local-unused")
        return _security500_guard_model(build_model(spec["model"], 1, spec["max_output_tokens"]), consumer, model_id)
    if _continuity_transfer_selected([model_id]):
        assert_not_reserved()
        _initialize_continuity_transfer_private_core()
        _assert_continuity_transfer_identity()
        if model_id not in CONTINUITY_TRANSFER_FULL_HISTORY_IDS or max_connections not in (None, 1):
            raise ValueError("Only serialized registered264 conditions admitted")
        consumer = continuity_transfer_active_consumer()
        if model_id not in consumer.ALLOWED_MODEL_IDS:
            raise ValueError("Condition requires the separate scientific amendment")
        spec = MODEL_SPECS[model_id]
        binding = current_local_binding(required_models=[model_id])
        if os.environ.get("RESEARCH4_QWEN_BASE_URL", binding["base_url"]).rstrip("/") != binding["base_url"]:
            raise ValueError("Continuity-transfer client URL differs from Root endpoint")
        consumer.validate_request(spec["service_model_id"])
        os.environ["RESEARCH4_QWEN_BASE_URL"] = binding["base_url"]
        os.environ.setdefault("RESEARCH4_QWEN_API_KEY", "local-unused")
        model = build_model(spec["model"], 1, spec["max_output_tokens"])
        return _security500_guard_model(model, consumer, model_id)
    selected_models([model_id])
    spec = MODEL_SPECS[model_id]
    if max_connections is None:
        max_connections = spec.get("max_connections", 1)
    if spec.get("provider") == "qwencloud":
        if _SECURITY500_CORE_REF is not None or os.environ.get(SECURITY500_CONTEXT_ENV):
            raise ValueError("QwenCloud is not admitted by the Security500 Game scope")
        return qwen_cloud.build_model(spec["api_model"], spec["max_output_tokens"], max_connections)
    if spec.get("local_qwen"):
        assert_not_reserved()
        _initialize_security500_private_core()
        if _SECURITY500_CORE_REF is not None:
            consumer = security500_active_consumer()
            binding = current_local_binding(required_models=[model_id])
            if max_connections != 1:
                raise ValueError("Security500 requires one client connection")
            if os.environ.get("RESEARCH4_QWEN_BASE_URL", binding["base_url"]).rstrip("/") != binding["base_url"]:
                raise ValueError("Local client URL differs from reviewed endpoint")
            os.environ["RESEARCH4_QWEN_BASE_URL"] = binding["base_url"]
            os.environ.setdefault("RESEARCH4_QWEN_API_KEY", "local-unused")
            consumer.validate_request(spec["service_model_id"])
            model = build_model(spec["model"], max_connections, spec["max_output_tokens"])
            return _security500_guard_model(model, consumer, model_id)
        _initialize_provider352_private_core()
        consumer = provider352_active_consumer() if _PROVIDER352_CORE_REF is not None else None
        if os.environ.get(PROVIDER352_CONTEXT_ENV) and consumer is None:
            raise ValueError(PROVIDER352_NOT_READY_REASON)
        binding = current_local_binding(required_models=[model_id])
        if max_connections != 1:
            raise ValueError("The new local paired service requires one client connection")
        if os.environ.get("RESEARCH4_QWEN_BASE_URL", binding["base_url"]).rstrip("/") != binding["base_url"]:
            raise ValueError("Local model client URL differs from the reviewed Mac endpoint")
        os.environ["RESEARCH4_QWEN_BASE_URL"] = binding["base_url"]
        os.environ.setdefault("RESEARCH4_QWEN_API_KEY", "local-unused")
        if consumer is not None:
            consumer.validate_request(spec["service_model_id"])
            model = build_model(spec["model"], max_connections, spec["max_output_tokens"])
            return consumer.install_http_guard(model, model_id)
    model = build_model(spec["model"], max_connections, spec["max_output_tokens"])
    if model_id in {"terra", "deepseek"}:
        _initialize_security500_private_core()
        if _SECURITY500_CORE_REF is not None:
            return _security500_guard_model(model, security500_active_consumer(), model_id)
    return model
