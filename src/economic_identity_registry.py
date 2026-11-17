"""Future-only identity binding for the replica-continuity checkpoint.

No artifact is loaded or registered at import. This module runs in a fresh
evaluation process after independently verified training and release; it never
retargets the old independent-state registry or owns a model service.
"""
from __future__ import annotations

import copy
from datetime import datetime
import re

from . import company_models as models
from . import hosted_inspect_runner as hosted
from . import independent_state_game as old
from .economic_game_comparison import FUTURE_IDS

DATASET_MANIFEST_SHA = "7a28871df2fc1d0f7b6bd03496e3501cb91543c0c3e838083e2bcb5d67f42fd7"
TRAIN_SHA = "c6a43694322414b8274f76cd16e7add630caca804b2ecfb08ee9f0dedc31c003"
DEV_SHA = "5367d0081ebc09d588dee9993bc44b1e24c9051a6af7dcb758deacaf1734262f"
EXPECTED_UPDATES = 612
EXPECTED_DEV_ROWS = 42
EXPECTED_ADAPTER_GEOMETRY = {"tensor_count": 124, "projection_count": 62, "dtype": "F32",
                             "rank": 16, "layer_indices": list(range(56, 64)),
                             "parameter_count": 14590976}
_CANDIDATE = None
_SPECS = None


def validate_candidate(binding, ready, audit):
    """Check actual independently audited metadata; this grants no execution."""
    fields = {"expected_ft_id", "adapter_sha256", "adapter_config_sha256", "base_manifest_sha256",
              "artifact_ready", "ready_utc", "completion_audit", "fit_updates", "teacher_dev_rows",
              "training_manifest_sha256", "train_data_sha256", "dev_data_sha256"}
    if (type(binding) is not dict or set(binding) != fields
            or any(not old._sha(binding[k]) for k in ("adapter_sha256", "adapter_config_sha256", "base_manifest_sha256"))
            or not old._ref(binding["artifact_ready"]) or not old._ref(binding["completion_audit"])
            or binding["base_manifest_sha256"] != models.LOCAL_BASE_MANIFEST_SHA256
            or type(binding["fit_updates"]) is not int or binding["fit_updates"] != EXPECTED_UPDATES
            or type(binding["teacher_dev_rows"]) is not int or binding["teacher_dev_rows"] != EXPECTED_DEV_ROWS
            or binding["training_manifest_sha256"] != DATASET_MANIFEST_SHA
            or binding["train_data_sha256"] != TRAIN_SHA or binding["dev_data_sha256"] != DEV_SHA
            or not isinstance(binding["expected_ft_id"], str)
            or re.fullmatch(r"sp-[0-9]{8}-[0-9]{2}-[0-9a-f]{8,64}", binding["expected_ft_id"]) is None):
        raise ValueError("Fresh actual 612-update replica-identity artifact required")
    route = binding["expected_ft_id"]
    if not binding["adapter_sha256"].startswith(route.rsplit("-", 1)[1]):
        raise ValueError("Route must contain the actual adapter hash")
    try:
        instant = datetime.fromisoformat(binding["ready_utc"].replace("Z", "+00:00"))
        if instant.tzinfo is None or instant.utcoffset().total_seconds() != 0:
            raise ValueError
        if instant.astimezone(old._alberta_timezone()).strftime("sp-%Y%m%d-%H-") != route[:15]:
            raise ValueError
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Route must use the actual Alberta READY hour") from None
    if (type(ready) is not dict or ready.get("schema") != "sp-lense.research4-adapter-artifact-ready.v1"
            or not str(ready.get("status", "")).startswith("ARTIFACT_READY")
            or ready.get("adapter_geometry") != EXPECTED_ADAPTER_GEOMETRY
            or any(ready.get(k) != binding[k] for k in
                   ("expected_ft_id", "adapter_sha256", "adapter_config_sha256", "base_manifest_sha256",
                    "ready_utc", "fit_updates", "teacher_dev_rows", "completion_audit"))):
        raise ValueError("Actual READY metadata differs from the candidate")
    if (type(audit) is not dict or not str(audit.get("status", "")).startswith("PASS")
            or audit.get("completed_updates") != EXPECTED_UPDATES
            or audit.get("DEV_rows") != EXPECTED_DEV_ROWS or audit.get("unique_TRAIN_decisions") != 306
            or audit.get("epochs") != 2 or audit.get("final_adapter_sha256") != binding["adapter_sha256"]
            or audit.get("training_manifest_sha256") != DATASET_MANIFEST_SHA
            or audit.get("train_data_sha256") != TRAIN_SHA or audit.get("dev_data_sha256") != DEV_SHA
            or audit.get("adapter_geometry") != EXPECTED_ADAPTER_GEOMETRY):
        raise ValueError("Actual audit does not prove the requested data, recipe and adapter")
    return copy.deepcopy(binding)


def register_candidate(binding, *, read_bytes=None):
    """Register planning-only role metadata in this fresh process, after proof."""
    global _CANDIDATE, _SPECS
    if type(binding) is not dict:
        raise ValueError("Actual artifact binding required")
    ready_raw, _ = old._bytes(binding.get("artifact_ready"), read_bytes)
    audit_raw, _ = old._bytes(binding.get("completion_audit"), read_bytes)
    import json
    value = validate_candidate(binding, json.loads(ready_raw), json.loads(audit_raw))
    if _CANDIDATE is not None:
        if value != _CANDIDATE:
            raise ValueError("Candidate cannot be retargeted in the same process")
        assert_candidate_identity()
        return copy.deepcopy(_CANDIDATE)
    route = value["expected_ft_id"]
    specs = {}
    for index, name in enumerate(FUTURE_IDS):
        ft, icl = index >= 2, index % 2 == 1
        spec = copy.deepcopy(models.MODEL_SPECS["qwen_base_neutral"])
        spec.update(model="openai-api/research4-qwen/" + (route if ft else "qwen3.8-27b-base"),
            service_model_id=route if ft else "qwen3.8-27b-base", arm_id=name,
            label=(route if ft else "Qwen3.8-27B BASE") + (" + operational ICL" if icl else " · neutral"),
            prompt_condition="operational_preservation_icl" if icl else "baseline", adapter_applied=ft,
            max_output_tokens=8192, required_context_tokens=131072, required_history_mode="full_chronological_v1",
            max_connections=1, request_timeout_seconds=900, execution_enabled=False,
            economic_identity_candidate=copy.deepcopy(value), independent_state_readiness="ARTIFACT_ONLY_HELD")
        if ft:
            spec.update(adapter_sha256=value["adapter_sha256"], adapter_config_sha256=value["adapter_config_sha256"],
                        adapter_rank=16, adapter_blocks=list(range(56, 64)), adapter_projection_count=62,
                        adapter_dtype="F32", completed_training_updates=EXPECTED_UPDATES)
        specs[name] = spec
    models.MODEL_SPECS.update(specs)
    hosted.LOCAL_QWEN_MODELS.add("openai-api/research4-qwen/" + route)
    _CANDIDATE, _SPECS = copy.deepcopy(value), copy.deepcopy(specs)
    return copy.deepcopy(_CANDIDATE)


def assert_candidate_identity():
    if _CANDIDATE is None or any(models.MODEL_SPECS.get(key) != value for key, value in _SPECS.items()):
        raise ValueError("Fresh economic candidate is not registered or changed")


def current_candidate():
    assert_candidate_identity()
    return copy.deepcopy(_CANDIDATE)
