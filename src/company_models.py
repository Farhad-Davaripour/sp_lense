"""Model choices and exact routing, independent of the operator's test subset."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

from .continuity_arena import build_model

QWEN_BASE_URL = "http://127.0.0.1:8769/v1"
LOCAL_BINDING_PATH = Path("C:/Users/farha/Research4/artifacts/last_layer_icl_pilot_20261109/SERVICE_BINDING.json")
LAST_SHA = "b0a1eb1a935d0dac71b35384e3dfd8638ef88f82a1f84da54c0987ece7fbf785"
BASE_MODEL = "openai-api/research4-qwen/qwen3.8-27b-base"
LAST_MODEL = "openai-api/research4-qwen/sp-lense-adaptor-last-layer"
MODEL_SPECS = {
    "qwen_base_neutral": {"label": "Qwen base · neutral", "model": BASE_MODEL,
        "arm_id": "base_neutral", "prompt_condition": "baseline", "local_qwen": True,
        "service_model_id": "qwen3.8-27b-base", "max_output_tokens": 4096},
    "qwen_base": {"label": "Qwen base · security instruction", "model": BASE_MODEL,
        "arm_id": "base_icl", "prompt_condition": "security_maximization", "local_qwen": True,
        "service_model_id": "qwen3.8-27b-base", "max_output_tokens": 4096},
    "sp_lense_neutral": {"label": "SP-Lense LAST · FT only", "model": LAST_MODEL,
        "arm_id": "last_neutral", "prompt_condition": "baseline", "local_qwen": True,
        "service_model_id": "sp-lense-adaptor-last-layer", "adapter_sha256": LAST_SHA,
        "max_output_tokens": 4096},
    "sp_lense": {"label": "SP-Lense LAST · FT + security instruction", "model": LAST_MODEL,
        "arm_id": "last_icl", "prompt_condition": "security_maximization", "local_qwen": True,
        "service_model_id": "sp-lense-adaptor-last-layer", "adapter_sha256": LAST_SHA,
        "max_output_tokens": 4096},
    "terra": {"label": "Terra · security instruction", "model": "openai/gpt-6.1-terra",
        "arm_id": "terra_icl", "prompt_condition": "security_maximization",
        "key_name": "OPENAI_API_KEY", "max_output_tokens": 8192},
    "deepseek": {"label": "DeepSeek · security instruction", "model": "deepseek/deepseek-flash",
        "arm_id": "deepseek_icl", "prompt_condition": "security_maximization",
        "key_name": "DEEPSEEK_API_KEY", "max_output_tokens": 8192},
}


def selected_models(models=None):
    chosen = ["deepseek"] if models is None else models
    if not isinstance(chosen, list) or not chosen or any(not isinstance(name, str) for name in chosen):
        raise ValueError("Choose one or more distinct configured models")
    if len(set(chosen)) != len(chosen) or any(name not in MODEL_SPECS for name in chosen):
        raise ValueError("Unknown model selection")
    return list(chosen)


def local_binding(path=None):
    """Validate retained configuration only; never probe or load the model."""
    path = Path(path or os.environ.get("RESEARCH4_LAST_SERVICE_BINDING", str(LOCAL_BINDING_PATH)))
    binding = json.loads(path.read_text(encoding="utf-8"))
    if binding.get("adapter_sha256") != LAST_SHA:
        raise ValueError("The configured SP-Lense checkpoint is not the reviewed LAST adapter")
    expected = binding.get("expected_health", {})
    if expected.get("adapter_sha256") != LAST_SHA or expected.get("ready") is not True:
        raise ValueError("Retained LAST identity is incomplete")
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


def assert_local_binding():
    """Fresh readiness check, called only when a human selects a local run."""
    binding = local_binding()
    if os.environ.get("RESEARCH4_QWEN_BASE_URL", QWEN_BASE_URL).rstrip("/") != binding["base_url"]:
        raise ValueError("Configured model client URL differs from the reviewed Mac endpoint")
    with urllib.request.urlopen(binding["health_url"], timeout=5) as response:
        raw = response.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        raise ValueError("Local readiness response exceeds limit")
    actual = json.loads(raw)
    if any(actual.get(key) != value for key, value in binding["expected_health"].items()):
        raise ValueError("Mac service does not match the reviewed BASE/LAST binding")
    return actual


def configured_key_names(credentials=None):
    names = {key for key in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY") if os.environ.get(key)}
    if credentials and Path(credentials).is_file():
        for line in Path(credentials).read_text(encoding="utf-8").splitlines():
            if not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if value.strip().strip("\"'"):
                    names.add(key.strip())
    return names


def model_choices(credentials=None, enabled=True):
    keys = configured_key_names(credentials)
    try:
        local_binding()
        local_ok = True
    except (OSError, ValueError, KeyError):
        local_ok = False
    choices = []
    for key, spec in MODEL_SPECS.items():
        configured = local_ok if spec.get("local_qwen") else spec["key_name"] in keys
        status = ("Mac configured · readiness checked when run" if spec.get("local_qwen")
                  else "API credential configured") if configured else (
                  "Mac endpoint binding required" if spec.get("local_qwen") else "API credential required")
        choices.append({"id": key, "label": spec["label"], "model": spec["model"],
            "available": bool(enabled and configured), "status": status if enabled else "Preview only · execution paused",
            "reason": status if enabled else "Execution is paused in preview mode.",
            "default_selected": key == "deepseek", "prompt_condition": spec["prompt_condition"],
            "max_output_tokens": spec["max_output_tokens"]})
    choices.append({"id": "gpt_cyber", "label": "GPT-6-Cyber", "available": False,
        "status": "Access not verified", "reason": "Daybreak API access has not been verified for this model.",
        "default_selected": False, "prompt_condition": "security_maximization"})
    return choices


def build_company_model(model_id, max_connections=1):
    spec = MODEL_SPECS[model_id]
    if spec.get("local_qwen"):
        if os.environ.get("RESEARCH4_QWEN_BASE_URL", QWEN_BASE_URL).rstrip("/") != QWEN_BASE_URL:
            raise ValueError("Local model client URL differs from the reviewed Mac endpoint")
        os.environ["RESEARCH4_QWEN_BASE_URL"] = QWEN_BASE_URL
        os.environ.setdefault("RESEARCH4_QWEN_API_KEY", "local-unused")
    return build_model(spec["model"], max_connections, spec["max_output_tokens"])
