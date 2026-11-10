"""QwenCloud request binding over Inspect's existing OpenAI-compatible provider.

Configuration checks never contact a service. The small provider specialization
only binds the official endpoint, keeps redirects disabled, and translates the
shared token allowance into a total thinking-plus-answer limit. Generation and
streaming remain Inspect's implementation.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import qwen_spending


HOST = "ap-southeast-1.maas.qwencloudapi.com"
BASE_URL = f"https://{HOST}/compatible-mode/v1"
PROVIDER = "qwencloud"
SUPPORTED_MODELS = {"qwen3.8-max", "qwen3.8-max-0902", "qwen3.8-27b"}
MAX_MODELS = {"qwen3.8-max", "qwen3.8-max-0902"}
KEY_NAME = "QWEN_API_KEY"
_VARIABLES = {KEY_NAME, "QWEN_BASE_URL", "QWEN_MODEL"}
_registered = False


def validate_base_url(value: str) -> str:
    """Accept only the configured, verified Singapore HTTPS endpoint."""
    if not isinstance(value, str) or not value:
        raise ValueError("QwenCloud QWEN_BASE_URL is required")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValueError("QwenCloud requires the official Singapore HTTPS endpoint") from None
    if (parsed.scheme != "https" or parsed.hostname != HOST
            or port not in (None, 443)
            or parsed.path.rstrip("/") != "/compatible-mode/v1"
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("QwenCloud requires the official Singapore HTTPS endpoint")
    return BASE_URL


def _settings(credentials: Path | str | None = None) -> dict[str, str]:
    values: dict[str, str] = {}
    if credentials is not None and Path(credentials).is_file():
        for raw in Path(credentials).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.removeprefix("export ").strip()
            if key in _VARIABLES:
                values[key] = value.strip().strip("\"'")
    # Match dotenv override=False: an explicitly set environment value wins,
    # including an empty/misconfigured value. Never repair it through fallback.
    values.update({key: os.environ[key] for key in _VARIABLES if key in os.environ})
    return values


def configuration(credentials: Path | str | None = None,
                  expected_model: str | None = None) -> dict[str, Any]:
    """Return nonsecret metadata for catalog display and exact model binding."""
    values = _settings(credentials)
    base_url = validate_base_url(values.get("QWEN_BASE_URL", ""))
    configured_model = values.get("QWEN_MODEL", "")
    if configured_model not in SUPPORTED_MODELS:
        raise ValueError("QWEN_MODEL must name a verified QwenCloud model")
    if expected_model is not None:
        if expected_model not in SUPPORTED_MODELS:
            raise ValueError("Unrecognized QwenCloud model; no substitute is permitted")
        if expected_model in MAX_MODELS and configured_model != expected_model:
            raise ValueError("Configured QwenCloud flagship differs from the selected model")
    return {"provider": PROVIDER, "base_url": base_url,
            "configured_model": configured_model,
            "api_key_configured": bool(values.get(KEY_NAME)),
            "key_name": KEY_NAME, "redirects": False}


def _register_provider() -> None:
    global _registered
    if _registered:
        return
    from inspect_ai.model import GenerateConfig, modelapi
    from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI

    @modelapi(name=PROVIDER)
    class QwenCloudAPI(OpenAICompatibleAPI):
        def __init__(self, model_name, base_url=None, api_key=None,
                     config=None, **model_args):
            if model_name not in SUPPORTED_MODELS or model_args:
                raise ValueError("Unsupported QwenCloud model or client override")
            super().__init__(model_name=model_name, service=PROVIDER,
                base_url=validate_base_url(base_url), api_key=api_key,
                api_key_var=KEY_NAME, config=config or GenerateConfig(),
                responses_api=False, stream=True, client_timeout=180,
                max_retries=0)

        def connection_key(self):
            # All Qwen models in this worker share the same credential pool.
            # Keep the credential itself out of concurrency/debug labels.
            identity = hashlib.sha256(str(self.initial_api_key).encode()).hexdigest()
            return f"qwencloud-account:{identity}"

        def _create_http_client(self):
            # Preserve Inspect/OpenAI's HTTP implementation and SSL/timeouts.
            # Apply on every initialization, including after a closed client.
            client = super()._create_http_client()
            client.follow_redirects = False
            return client

        def completion_params(self, config, tools):
            params = super().completion_params(config, tools)
            limit = params.pop("max_tokens", None)
            if type(limit) is not int or not 1 <= limit <= 131072:
                raise ValueError("QwenCloud requires a bounded total output allowance")
            if config.reasoning_effort not in (None, "low"):
                raise ValueError("The reviewed QwenCloud reasoning effort is low")
            params.pop("reasoning_effort", None)
            extra = dict(params.get("extra_body") or {})
            if set(extra) - {"enable_thinking", "reasoning_effort"}:
                raise ValueError("Unreviewed QwenCloud body parameters are not permitted")
            if extra.get("enable_thinking", True) is not True or extra.get("reasoning_effort", "low") != "low":
                raise ValueError("QwenCloud thinking must remain enabled at low effort")
            params["max_completion_tokens"] = limit
            params["extra_body"] = {"enable_thinking": True, "reasoning_effort": "low"}
            return params

        async def _generate_completion(self, request, config):
            reservation = qwen_spending.reserve_for_request(request,
                endpoint=str(self.client.base_url), credential=self.initial_api_key,
                current_credential=self.client.api_key)
            try:
                completion = await super()._generate_completion(request, config)
                metadata = {"id": completion.id,
                    "id_type": "provider completion response ID, not assumed request ID",
                    "model": completion.model, "created": completion.created,
                    "usage": completion.usage.model_dump(mode="json") if completion.usage else None,
                    "finish_reasons": [choice.finish_reason for choice in completion.choices]}
            except BaseException as error:
                reservation.ambiguous(type(error).__name__, getattr(error, "status_code", None))
                raise
            reservation.response(metadata)
            return completion

    _registered = True


def _create_inspect_model(api_model: str, base_url: str, api_key: str,
                          max_output_tokens: int, max_connections: int):
    from inspect_ai.model import GenerateConfig, get_model

    _register_provider()
    return get_model(f"{PROVIDER}/{api_model}", base_url=base_url, api_key=api_key,
        memoize=False, config=GenerateConfig(max_tokens=max_output_tokens,
            max_connections=max_connections, max_retries=0, timeout=180,
            reasoning_effort="low",
            extra_body={"enable_thinking": True, "reasoning_effort": "low"}))


def build_model(api_model: str, max_output_tokens: int = 8192,
                max_connections: int = 2):
    """Construct an Inspect model; perform no model request or readiness probe."""
    if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 131072:
        raise ValueError("QwenCloud output allowance is invalid")
    if type(max_connections) is not int or not 1 <= max_connections <= 2:
        raise ValueError("QwenCloud connection allowance is invalid")
    safe = configuration(expected_model=api_model)
    key = _settings().get(KEY_NAME)
    if not key:
        raise ValueError("QWEN_API_KEY is required for QwenCloud")
    # api_key is get_model's special credential argument, not model_args or a
    # manifest field. No cloud environment variables are copied to Mac routing.
    return _create_inspect_model(api_model, safe["base_url"], key,
                                 max_output_tokens, max_connections)
