"""Submit locked blinded judge requests without changing experimental records.

This transport utility intentionally lives under ``artifacts/`` rather than the locked
stage-one runner.  It accepts the exact JSONL records rendered by
``sp-lense-compare-steering judge-requests`` and writes the exact three-field response
records accepted by ``attach-judgments``.  Each API receipt is retained separately for
cost and provenance auditing.

The utility is resume-safe and refuses to make a network call unless:

* ``OPENAI_API_KEY`` is present;
* the request model is the preregistered dated snapshot;
* the caller supplies a positive dollar ceiling; and
* a conservative worst-case estimate fits below that ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_URL = "https://api.openai.com/v1/responses"
LOCKED_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_INPUT_PRICE_PER_MILLION = 0.40
DEFAULT_OUTPUT_PRICE_PER_MILLION = 1.60
PRICE_SAFETY_FACTOR = 1.25
MAX_RATE_LIMIT_RETRIES = 8
ATTEMPT_SCHEMA = "sp_lense.openai_judge_submission_attempt.v1"
PERSONA_EXCHANGE_SCHEMA = "sp_lense.persona_judge_exchange.v1"
OPEN_EXCHANGE_SCHEMA = "sp_lense.open_judge_exchange.v2"
OPEN_REQUEST_CONTENT_SCHEMA = "sp_lense.open_judge_request_content.v1"
LOCKED_PERSONA_CONFIG_SHA256 = "d67afdababadc40634069919504cd663d5418baf490b12810871676ec57057bf"
LOCKED_OPEN_CONFIG_SHA256 = "fc68fcc97a094d0133c4ce45bf37ac3848d6e7d20ce0300cc113c4871a132593"
LOCKED_OPEN_PROTOCOL_SHA256 = "736d91a0165aa6f7039bdddf9e24cac8a6b0849da8e89fa0b29eff985ece92b3"
PERSONA_REQUEST_FIELDS = {
    "model",
    "input",
    "temperature",
    "top_p",
    "max_output_tokens",
    "store",
}
OPEN_REQUEST_FIELDS = PERSONA_REQUEST_FIELDS | {"tools", "truncation", "text"}
OPEN_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["preserve", "comply", "ambiguous"],
        },
        "coherence_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "authorization_sensitivity_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "task_benefit_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "action_in_first_sentence": {"type": "boolean"},
    },
    "required": [
        "decision",
        "coherence_score",
        "authorization_sensitivity_score",
        "task_benefit_score",
        "action_in_first_sentence",
    ],
    "additionalProperties": False,
}


class RejectedSubmissionError(RuntimeError):
    """The provider explicitly rejected a request before returning a response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AmbiguousSubmissionError(RuntimeError):
    """A request may have been processed, so automatically repeating it is unsafe."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _validate_blind_input(value: Any, label: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly system and user messages")
    for index, (message, role) in enumerate(zip(value, ("system", "user"), strict=True)):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"{label}[{index}] has an invalid exact message schema")
        if message["role"] != role:
            raise ValueError(f"{label}[{index}] must have role={role!r}")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError(f"{label}[{index}] content must be non-empty")


def _validate_common_request_parameters(
    request: dict[str, Any], *, request_id: str, expected_max_output_tokens: int
) -> None:
    if request.get("model") != LOCKED_MODEL:
        raise ValueError(
            f"request {request_id} targets {request.get('model')!r}, not {LOCKED_MODEL!r}"
        )
    if (
        isinstance(request.get("temperature"), bool)
        or request.get("temperature") != 0.0
        or isinstance(request.get("top_p"), bool)
        or request.get("top_p") != 1.0
        or request.get("max_output_tokens") != expected_max_output_tokens
        or isinstance(request.get("max_output_tokens"), bool)
        or request.get("store") is not False
    ):
        raise ValueError(f"request {request_id} differs from the locked sampling parameters")
    _validate_blind_input(request.get("input"), f"request {request_id} input")


def _validate_persona_request(row: dict[str, Any], index: int) -> None:
    expected_fields = {
        "schema_version",
        "request_id",
        "key",
        "judge_prompt_sha256",
        "judge_config_sha256",
        "request",
    }
    if set(row) != expected_fields:
        raise ValueError(f"persona request {index} differs from the exact exchange schema")
    request_id = _validate_sha256(row["request_id"], f"persona request {index} ID")
    prompt_hash = _validate_sha256(
        row["judge_prompt_sha256"], f"persona request {request_id} prompt hash"
    )
    config_hash = _validate_sha256(
        row["judge_config_sha256"], f"persona request {request_id} config hash"
    )
    if config_hash != LOCKED_PERSONA_CONFIG_SHA256:
        raise ValueError(f"persona request {request_id} config hash differs from the lock")
    key = row["key"]
    if not isinstance(key, dict) or set(key) != {
        "instruction_pair_id",
        "question_id",
        "rollout_index",
        "polarity",
    }:
        raise ValueError(f"persona request {request_id} has an invalid key")
    if (
        not isinstance(key["instruction_pair_id"], str)
        or not key["instruction_pair_id"]
        or not isinstance(key["question_id"], str)
        or not key["question_id"]
        or isinstance(key["rollout_index"], bool)
        or not isinstance(key["rollout_index"], int)
        or key["rollout_index"] < 0
        or key["polarity"] not in {"positive", "negative"}
    ):
        raise ValueError(f"persona request {request_id} key values are invalid")
    request = row["request"]
    if not isinstance(request, dict) or set(request) != PERSONA_REQUEST_FIELDS:
        raise ValueError(f"persona request {request_id} payload fields differ from the lock")
    _validate_common_request_parameters(
        request, request_id=request_id, expected_max_output_tokens=128
    )
    if _sha256(request["input"]) != prompt_hash:
        raise ValueError(f"persona request {request_id} prompt hash is invalid")
    expected_id = _sha256(
        {
            "schema_version": PERSONA_EXCHANGE_SCHEMA,
            "key": key,
            "judge_prompt_sha256": prompt_hash,
            "judge_config_sha256": config_hash,
        }
    )
    if request_id != expected_id:
        raise ValueError(f"persona request {request_id} ID is not bound to its metadata")


def _validate_open_request(row: dict[str, Any], index: int) -> None:
    expected_fields = {
        "schema_version",
        "request_id",
        "generation_sha256s",
        "judge_prompt_sha256",
        "judge_request_content_sha256",
        "judge_config_sha256",
        "judge_protocol_sha256",
        "request",
    }
    if set(row) != expected_fields:
        raise ValueError(f"open request {index} differs from the exact exchange schema")
    request_id = _validate_sha256(row["request_id"], f"open request {index} ID")
    prompt_hash = _validate_sha256(
        row["judge_prompt_sha256"], f"open request {request_id} prompt hash"
    )
    content_hash = _validate_sha256(
        row["judge_request_content_sha256"],
        f"open request {request_id} content hash",
    )
    config_hash = _validate_sha256(
        row["judge_config_sha256"], f"open request {request_id} config hash"
    )
    protocol_hash = _validate_sha256(
        row["judge_protocol_sha256"], f"open request {request_id} protocol hash"
    )
    if config_hash != LOCKED_OPEN_CONFIG_SHA256:
        raise ValueError(f"open request {request_id} config hash differs from the lock")
    if protocol_hash != LOCKED_OPEN_PROTOCOL_SHA256:
        raise ValueError(f"open request {request_id} protocol hash differs from the lock")
    generation_hashes = row["generation_sha256s"]
    if (
        not isinstance(generation_hashes, list)
        or not generation_hashes
        or len(generation_hashes) != len(set(generation_hashes))
    ):
        raise ValueError(f"open request {request_id} generation hashes are invalid")
    for generation_index, digest in enumerate(generation_hashes):
        _validate_sha256(
            digest,
            f"open request {request_id} generation hash {generation_index}",
        )
    request = row["request"]
    if not isinstance(request, dict) or set(request) != OPEN_REQUEST_FIELDS:
        raise ValueError(f"open request {request_id} payload fields differ from the lock")
    _validate_common_request_parameters(
        request, request_id=request_id, expected_max_output_tokens=256
    )
    if request.get("tools") != [] or request.get("truncation") != "disabled":
        raise ValueError(f"open request {request_id} tool/truncation settings differ from lock")
    expected_text = {
        "format": {
            "type": "json_schema",
            "name": "open_behavior_judgment",
            "strict": True,
            "schema": OPEN_RESPONSE_SCHEMA,
        }
    }
    if request.get("text") != expected_text:
        raise ValueError(f"open request {request_id} structured-output schema differs from lock")
    if _sha256(request["input"]) != prompt_hash:
        raise ValueError(f"open request {request_id} prompt hash is invalid")
    expected_content_hash = _sha256(
        {
            "schema_version": OPEN_REQUEST_CONTENT_SCHEMA,
            "request": request,
        }
    )
    if content_hash != expected_content_hash:
        raise ValueError(f"open request {request_id} content hash is invalid")


def _validate_requests(rows: list[dict[str, Any]]) -> None:
    request_ids: set[str] = set()
    schemas: set[str] = set()
    for index, row in enumerate(rows):
        schema = row["schema_version"]
        request_id = row["request_id"]
        if schema == PERSONA_EXCHANGE_SCHEMA:
            _validate_persona_request(row, index)
        elif schema == OPEN_EXCHANGE_SCHEMA:
            _validate_open_request(row, index)
        else:
            raise ValueError(f"request {index} has an unsupported schema_version")
        if request_id in request_ids:
            raise ValueError(f"duplicate request_id: {request_id}")
        request_ids.add(request_id)
        schemas.add(schema)
    if len(schemas) != 1:
        raise ValueError("one response file may contain only one locked exchange schema")


def _upper_bound_cost(
    rows: list[dict[str, Any]],
    *,
    input_price_per_million: float,
    output_price_per_million: float,
) -> dict[str, float | int]:
    # A normal text token consumes at least one UTF-8 byte. Counting the complete
    # serialized request bytes also covers message-framing text and field names, then
    # the explicit multiplier leaves additional margin for endpoint framing/accounting.
    input_token_upper_bound = sum(len(_canonical_bytes(row["request"])) for row in rows)
    output_token_upper_bound = sum(row["request"]["max_output_tokens"] for row in rows)
    raw_cost = (
        input_token_upper_bound * input_price_per_million
        + output_token_upper_bound * output_price_per_million
    ) / 1_000_000
    return {
        "request_count": len(rows),
        "input_token_upper_bound": input_token_upper_bound,
        "output_token_upper_bound": output_token_upper_bound,
        "raw_upper_bound_usd": raw_cost,
        "safety_factor": PRICE_SAFETY_FACTOR,
        "safe_upper_bound_usd": raw_cost * PRICE_SAFETY_FACTOR,
    }


def _require_locked_prices(
    input_price_per_million: float, output_price_per_million: float
) -> None:
    """Reject a caller-supplied price that could weaken the dollar ceiling."""

    expected = (
        ("input", input_price_per_million, DEFAULT_INPUT_PRICE_PER_MILLION),
        ("output", output_price_per_million, DEFAULT_OUTPUT_PRICE_PER_MILLION),
    )
    for label, observed, locked in expected:
        if (
            not math.isfinite(observed)
            or not math.isclose(observed, locked, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError(
                f"{label} price must equal the locked {locked:.2f} USD per million tokens"
            )


def _publish_cost_preflight(
    work_dir: Path,
    responses_path: Path,
    estimate: dict[str, Any],
) -> None:
    """Publish a preflight, but never revise it after submission evidence exists."""

    preflight_path = work_dir / "cost_preflight.json"
    evidence_directories = (
        work_dir / "response_shards",
        work_dir / "api_receipts",
        work_dir / "submission_attempts",
        work_dir / "blocked_responses",
    )
    evidence_exists = responses_path.exists() or any(
        directory.exists() and any(path.is_file() for path in directory.rglob("*"))
        for directory in evidence_directories
    )
    if evidence_exists:
        if not preflight_path.is_file():
            raise SystemExit(
                "Submission evidence exists without its original cost preflight; "
                "automatic continuation is blocked"
            )
        try:
            existing_bytes = preflight_path.read_bytes()
        except OSError as error:
            raise SystemExit(
                "Submission evidence exists with an unreadable cost preflight; "
                "automatic continuation is blocked"
            ) from error
        if existing_bytes != _pretty_json_bytes(estimate):
            raise SystemExit(
                "The cost preflight is byte-for-byte immutable after any submission "
                "evidence exists; its bytes or requested configuration differ"
            )
        return
    _atomic_json(preflight_path, estimate)


def _extract_output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    pieces: list[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    pieces.append(part["text"])
    text = "".join(pieces)
    if not text:
        raise ValueError("OpenAI response contains no output_text")
    return text


def _validate_provider_response(
    response: dict[str, Any], response_headers: dict[str, str | None]
) -> str:
    if response.get("model") != LOCKED_MODEL:
        raise ValueError(
            f"OpenAI returned model {response.get('model')!r}, not {LOCKED_MODEL!r}"
        )
    if response.get("status") != "completed":
        raise ValueError(
            "OpenAI response status is not completed: " f"{response.get('status')!r}"
        )
    response_id = response.get("id")
    if not isinstance(response_id, str) or not response_id:
        raise ValueError("OpenAI response lacks a response ID")
    server_request_id = response_headers.get("x_request_id")
    if not isinstance(server_request_id, str) or not server_request_id:
        raise ValueError("OpenAI response lacks the x-request-id trace header")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise TypeError("OpenAI response lacks token usage")
    token_counts = [
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        usage.get("total_tokens"),
    ]
    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in token_counts):
        raise ValueError("OpenAI response has invalid token usage")
    if token_counts[2] < token_counts[0] + token_counts[1]:
        raise ValueError("OpenAI response token usage is inconsistent")
    return _extract_output_text(response)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_pretty_json_bytes(value))
    temporary.replace(path)


def _client_request_id(request_id: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        request_id.encode("utf-8") + b"\0" + _canonical_bytes(payload)
    ).hexdigest()
    return f"sp-lense-{digest}"


def _post_response(
    payload: dict[str, Any], api_key: str, *, client_request_id: str
) -> tuple[dict[str, Any], dict[str, str | None]]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        request = urllib.request.Request(
            API_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "sp-lense-locked-judge-transport/1",
                # This is a documented trace identifier, not an idempotency claim.
                "X-Client-Request-Id": client_request_id,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw_body = response.read()
                server_request_id = response.headers.get("x-request-id")
            try:
                parsed = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise AmbiguousSubmissionError(
                    "OpenAI returned an unreadable success body; automatic resubmission is "
                    f"blocked (X-Client-Request-Id={client_request_id}, "
                    f"X-Request-Id={server_request_id})"
                ) from error
            if not isinstance(parsed, dict):
                raise AmbiguousSubmissionError(
                    "OpenAI returned a non-object success body; automatic resubmission is "
                    f"blocked (X-Client-Request-Id={client_request_id}, "
                    f"X-Request-Id={server_request_id})"
                )
            return parsed, {
                "x_client_request_id": client_request_id,
                "x_request_id": server_request_id,
            }
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            if error.code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                retry_after = error.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else 2**attempt
                except ValueError:
                    delay = 2**attempt
                time.sleep(min(max(delay, 1.0), 30.0))
                continue
            message = (
                f"OpenAI HTTP {error.code}: {error_body}; "
                f"X-Client-Request-Id={client_request_id}"
            )
            if error.code == 408 or 500 <= error.code <= 599:
                raise AmbiguousSubmissionError(
                    message + "; automatic resubmission is blocked",
                    status_code=error.code,
                ) from error
            raise RejectedSubmissionError(message, status_code=error.code) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise AmbiguousSubmissionError(
                "OpenAI request outcome is unknown after a network failure; automatic "
                f"resubmission is blocked (X-Client-Request-Id={client_request_id}): {error}"
            ) from error
    raise AssertionError("retry loop exhausted without returning or raising")


def _attempt_payload(
    request_row: dict[str, Any],
    *,
    client_request_id: str,
    state: str,
    detail: str = "",
    status_code: int | None = None,
    server_request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": ATTEMPT_SCHEMA,
        "request_id": request_row["request_id"],
        "request_sha256": _sha256(request_row["request"]),
        "x_client_request_id": client_request_id,
        "state": state,
        "detail": detail,
        "http_status_code": status_code,
        "x_request_id": server_request_id,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _validate_prior_attempt(
    attempt_path: Path,
    request_row: dict[str, Any],
    *,
    client_request_id: str,
) -> str | None:
    if not attempt_path.exists():
        return None
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    if not isinstance(attempt, dict) or attempt.get("schema_version") != ATTEMPT_SCHEMA:
        raise ValueError(f"invalid submission-attempt record {attempt_path}")
    expected = {
        "request_id": request_row["request_id"],
        "request_sha256": _sha256(request_row["request"]),
        "x_client_request_id": client_request_id,
    }
    for field, value in expected.items():
        if attempt.get(field) != value:
            raise ValueError(f"submission-attempt identity mismatch in {attempt_path}: {field}")
    state = attempt.get("state")
    if not isinstance(state, str) or not state:
        raise ValueError(f"submission-attempt state is missing in {attempt_path}")
    return state


def _acquire_transport_lock(work_dir: Path) -> Any:
    work_dir.mkdir(parents=True, exist_ok=True)
    lock_path = work_dir / ".submission.lock"
    handle = lock_path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() < 1:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, PermissionError) as error:
        handle.close()
        raise SystemExit(
            f"another judge transport owns the exclusive lock {lock_path}; "
            "no request was sent"
        ) from error
    return handle


def _load_api_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    if os.name != "nt":
        return None
    # A user may add the variable while Codex is already running. Existing parent
    # processes do not inherit later registry changes, so read the Windows user
    # environment directly as a restart-free fallback. The value is never logged.
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as environment_key:
            value, _ = winreg.QueryValueEx(environment_key, "OPENAI_API_KEY")
    except (FileNotFoundError, OSError):
        return None
    return value if isinstance(value, str) and value else None


def _existing_response(shard_path: Path, receipt_path: Path, request_row: dict[str, Any]) -> bool:
    if not shard_path.exists() and not receipt_path.exists():
        return False
    if shard_path.exists() and not receipt_path.exists():
        raise ValueError(f"response shard lacks an API receipt for {request_row['request_id']}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("request_sha256") != _sha256(request_row["request"]):
        raise ValueError(f"request hash mismatch in {receipt_path}")
    receipt_raw_response = receipt.get("raw_response")
    if not isinstance(receipt_raw_response, str) or not receipt_raw_response:
        raise ValueError(f"API receipt lacks the exact raw response in {receipt_path}")
    if not shard_path.exists():
        # The receipt is written first. If interruption occurs between the two atomic
        # writes, reconstruct the smaller attachable shard from the authenticated
        # receipt instead of paying for a duplicate request.
        _atomic_json(
            shard_path,
            {
                "schema_version": request_row["schema_version"],
                "request_id": request_row["request_id"],
                "raw_response": receipt_raw_response,
            },
        )
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    expected_shard_fields = {"schema_version", "request_id", "raw_response"}
    if not isinstance(shard, dict) or set(shard) != expected_shard_fields:
        raise ValueError(f"invalid response shard {shard_path}")
    if shard["schema_version"] != request_row["schema_version"]:
        raise ValueError(f"schema mismatch in {shard_path}")
    if shard["request_id"] != request_row["request_id"]:
        raise ValueError(f"request ID mismatch in {shard_path}")
    if not isinstance(shard["raw_response"], str) or not shard["raw_response"]:
        raise ValueError(f"empty raw response in {shard_path}")
    if shard["raw_response"] != receipt_raw_response:
        raise ValueError(f"raw response differs between shard and receipt for {shard_path}")
    if receipt.get("raw_response_sha256") != hashlib.sha256(
        shard["raw_response"].encode("utf-8")
    ).hexdigest():
        raise ValueError(f"raw response hash mismatch in {receipt_path}")
    return True


def _validate_completed_receipt(
    receipt_path: Path,
    request_row: dict[str, Any],
    *,
    client_request_id: str,
) -> dict[str, Any]:
    """Validate every provider/identity field needed to trust a persisted receipt."""

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise TypeError(f"API receipt is not an object: {receipt_path}")
    if receipt.get("schema_version") != "sp_lense.openai_judge_api_receipt.v1":
        raise ValueError(f"API receipt schema mismatch in {receipt_path}")
    request_id = request_row["request_id"]
    if receipt.get("request_id") != request_id:
        raise ValueError(f"API receipt request ID mismatch in {receipt_path}")
    if receipt.get("request_sha256") != _sha256(request_row["request"]):
        raise ValueError(f"request hash mismatch in {receipt_path}")
    if receipt.get("x_client_request_id") != client_request_id:
        raise ValueError(f"client request trace mismatch in {receipt_path}")

    server_request_id = receipt.get("x_request_id")
    if not isinstance(server_request_id, str) or not server_request_id:
        raise ValueError(f"server request trace is missing in {receipt_path}")
    api_response = receipt.get("api_response")
    if not isinstance(api_response, dict):
        raise TypeError(f"API response object is missing in {receipt_path}")
    if receipt.get("api_response_sha256") != _sha256(api_response):
        raise ValueError(f"API response hash mismatch in {receipt_path}")

    raw_response = _validate_provider_response(
        api_response,
        {
            "x_client_request_id": client_request_id,
            "x_request_id": server_request_id,
        },
    )
    if receipt.get("api_model") != LOCKED_MODEL:
        raise ValueError(f"pinned judge model mismatch in {receipt_path}")
    if receipt.get("api_status") != "completed":
        raise ValueError(f"incomplete provider response in {receipt_path}")
    api_response_id = receipt.get("api_response_id")
    if not isinstance(api_response_id, str) or not api_response_id:
        raise ValueError(f"API response ID is missing in {receipt_path}")
    if api_response.get("id") != api_response_id:
        raise ValueError(f"receipt/provider response ID mismatch in {receipt_path}")
    usage = receipt.get("usage")
    if not isinstance(usage, dict):
        raise TypeError(f"usage record is missing in {receipt_path}")
    if _canonical_bytes(usage) != _canonical_bytes(api_response.get("usage")):
        raise ValueError(f"receipt/provider usage mismatch in {receipt_path}")
    if receipt.get("raw_response") != raw_response:
        raise ValueError(f"receipt/provider output text mismatch in {receipt_path}")
    if receipt.get("raw_response_sha256") != hashlib.sha256(
        raw_response.encode("utf-8")
    ).hexdigest():
        raise ValueError(f"raw response hash mismatch in {receipt_path}")
    return receipt


def _recover_completed_attempt(
    attempt_path: Path,
    request_row: dict[str, Any],
    receipt: dict[str, Any],
    *,
    client_request_id: str,
) -> bool:
    """Finish the one safe crash state: validated receipt persisted after prepared."""

    state = _validate_prior_attempt(
        attempt_path,
        request_row,
        client_request_id=client_request_id,
    )
    server_request_id = receipt["x_request_id"]
    if state == "completed":
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        if attempt.get("x_request_id") != server_request_id:
            raise ValueError(
                f"attempt/receipt server trace mismatch for {request_row['request_id']}"
            )
        return False
    if state != "prepared":
        raise SystemExit(
            "A persisted API receipt cannot automatically resolve submission-attempt "
            f"state {state!r} for {request_row['request_id']}"
        )
    _atomic_json(
        attempt_path,
        _attempt_payload(
            request_row,
            client_request_id=client_request_id,
            state="completed",
            server_request_id=server_request_id,
        ),
    )
    return True


def _write_final_responses(
    requests: list[dict[str, Any]], shard_directory: Path, output_path: Path
) -> None:
    rows: list[dict[str, Any]] = []
    for request_row in requests:
        shard_path = shard_directory / f"{request_row['request_id']}.json"
        value = json.loads(shard_path.read_text(encoding="utf-8"))
        rows.append(value)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(output_path)


def _verify_completed_transport(
    requests: list[dict[str, Any]],
    *,
    requests_file_sha256: str,
    responses_path: Path,
    work_dir: Path,
) -> dict[str, Any]:
    shard_directory = work_dir / "response_shards"
    receipt_directory = work_dir / "api_receipts"
    attempt_directory = work_dir / "submission_attempts"
    blocked_directory = work_dir / "blocked_responses"
    if blocked_directory.is_dir():
        blocked = sorted(path.name for path in blocked_directory.glob("*.json") if path.is_file())
        if blocked:
            raise ValueError(f"unresolved quarantined provider responses remain: {blocked}")
    expected_names = {f"{row['request_id']}.json" for row in requests}
    for label, directory in (
        ("response shards", shard_directory),
        ("API receipts", receipt_directory),
        ("submission attempts", attempt_directory),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory is missing: {directory}")
        observed_names = {path.name for path in directory.glob("*.json") if path.is_file()}
        if observed_names != expected_names:
            missing = sorted(expected_names - observed_names)
            extra = sorted(observed_names - expected_names)
            raise ValueError(f"{label} set mismatch: missing={missing}, extra={extra}")

    expected_response_rows: list[dict[str, Any]] = []
    api_response_ids: set[str] = set()
    server_request_ids: set[str] = set()
    total_input_tokens = 0
    total_output_tokens = 0
    for request_row in requests:
        request_id = request_row["request_id"]
        shard_path = shard_directory / f"{request_id}.json"
        receipt_path = receipt_directory / f"{request_id}.json"
        attempt_path = attempt_directory / f"{request_id}.json"
        client_request_id = _client_request_id(request_id, request_row["request"])
        if not _existing_response(shard_path, receipt_path, request_row):
            raise ValueError(f"completed response is missing for {request_id}")
        if (
            _validate_prior_attempt(
                attempt_path,
                request_row,
                client_request_id=client_request_id,
            )
            != "completed"
        ):
            raise ValueError(f"submission attempt is not completed for {request_id}")

        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        receipt = _validate_completed_receipt(
            receipt_path,
            request_row,
            client_request_id=client_request_id,
        )
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        api_response = receipt.get("api_response")
        assert isinstance(api_response, dict)
        server_request_id = receipt.get("x_request_id")
        assert isinstance(server_request_id, str)
        if attempt.get("x_request_id") != server_request_id:
            raise ValueError(f"attempt/receipt server trace mismatch for {request_id}")
        api_response_id = receipt.get("api_response_id")
        assert isinstance(api_response_id, str)
        if api_response_id in api_response_ids:
            raise ValueError(f"duplicate API response ID: {api_response_id}")
        if server_request_id in server_request_ids:
            raise ValueError(f"duplicate server request trace: {server_request_id}")
        api_response_ids.add(api_response_id)
        server_request_ids.add(server_request_id)

        usage = receipt.get("usage")
        assert isinstance(usage, dict)
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (input_tokens, output_tokens, total_tokens)
        ):
            raise ValueError(f"usage token counts are invalid in {receipt_path}")
        if total_tokens < input_tokens + output_tokens:
            raise ValueError(f"usage total is inconsistent in {receipt_path}")
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        expected_response_rows.append(shard)

    observed_response_rows = _read_jsonl(responses_path)
    if _canonical_bytes(observed_response_rows) != _canonical_bytes(expected_response_rows):
        raise ValueError("combined response JSONL differs from the exact ordered shards")

    preflight_path = work_dir / "cost_preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError(f"cost preflight is missing: {preflight_path}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not isinstance(preflight, dict):
        raise TypeError(f"cost preflight is not an object: {preflight_path}")
    if preflight.get("requests_file_sha256") != requests_file_sha256:
        raise ValueError("cost preflight request-file hash mismatch")
    if preflight.get("model") != LOCKED_MODEL or preflight.get("api_url") != API_URL:
        raise ValueError("cost preflight provider/model identity mismatch")
    input_price = preflight.get("input_price_per_million_usd")
    output_price = preflight.get("output_price_per_million_usd")
    cost_ceiling = preflight.get("user_cost_ceiling_usd")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        for value in (input_price, output_price, cost_ceiling)
    ):
        raise ValueError("cost preflight prices or ceiling are invalid")
    _require_locked_prices(float(input_price), float(output_price))
    recalculated = _upper_bound_cost(
        requests,
        input_price_per_million=float(input_price),
        output_price_per_million=float(output_price),
    )
    for field in ("request_count", "input_token_upper_bound", "output_token_upper_bound"):
        if preflight.get(field) != recalculated[field]:
            raise ValueError(f"cost preflight {field} mismatch")
    for field in ("raw_upper_bound_usd", "safety_factor", "safe_upper_bound_usd"):
        observed = preflight.get(field)
        if not isinstance(observed, (int, float)) or not math.isclose(
            float(observed),
            float(recalculated[field]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"cost preflight {field} mismatch")
    if float(recalculated["safe_upper_bound_usd"]) > float(cost_ceiling):
        raise ValueError("cost preflight safe upper bound exceeds its user ceiling")
    if total_input_tokens > int(recalculated["input_token_upper_bound"]):
        raise ValueError("actual input usage exceeds the conservative token upper bound")
    if total_output_tokens > int(recalculated["output_token_upper_bound"]):
        raise ValueError("actual output usage exceeds the locked maximum-token sum")
    actual_cost = (
        total_input_tokens * float(input_price)
        + total_output_tokens * float(output_price)
    ) / 1_000_000
    if actual_cost > float(cost_ceiling):
        raise ValueError("actual recorded usage exceeds the user cost ceiling")
    return {
        "status": "transport_verified",
        "model": LOCKED_MODEL,
        "request_count": len(requests),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "actual_cost_usd": actual_cost,
        "user_cost_ceiling_usd": float(cost_ceiling),
        "requests_file_sha256": requests_file_sha256,
        "responses_file_sha256": hashlib.sha256(responses_path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument(
        "--input-price-per-million", type=float, default=DEFAULT_INPUT_PRICE_PER_MILLION
    )
    parser.add_argument(
        "--output-price-per-million", type=float, default=DEFAULT_OUTPUT_PRICE_PER_MILLION
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only and args.dry_run:
        parser.error("--verify-only and --dry-run are mutually exclusive")
    if not args.verify_only and (
        args.max_cost_usd is None
        or not math.isfinite(args.max_cost_usd)
        or args.max_cost_usd <= 0
    ):
        parser.error("--max-cost-usd must be a positive finite number")
    for label, value in (
        ("input price", args.input_price_per_million),
        ("output price", args.output_price_per_million),
    ):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{label} must be positive and finite")
    try:
        _require_locked_prices(
            args.input_price_per_million, args.output_price_per_million
        )
    except ValueError as error:
        parser.error(str(error))

    requests = _read_jsonl(args.requests)
    _validate_requests(requests)
    if args.verify_only:
        verification = _verify_completed_transport(
            requests,
            requests_file_sha256=hashlib.sha256(args.requests.read_bytes()).hexdigest(),
            responses_path=args.responses,
            work_dir=args.work_dir,
        )
        print(json.dumps(verification, sort_keys=True, indent=2))
        return 0

    assert args.max_cost_usd is not None
    # Hold one lock across preflight publication and, unless this is a dry run, the
    # complete check/POST/receipt loop. This prevents a concurrent dry run from
    # replacing the cost record of an active paid phase.
    _transport_lock = _acquire_transport_lock(args.work_dir)
    estimate = _upper_bound_cost(
        requests,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
    )
    estimate.update(
        {
            "model": LOCKED_MODEL,
            "api_url": API_URL,
            "input_price_per_million_usd": args.input_price_per_million,
            "output_price_per_million_usd": args.output_price_per_million,
            "user_cost_ceiling_usd": args.max_cost_usd,
            "requests_file_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        }
    )
    _publish_cost_preflight(args.work_dir, args.responses, estimate)
    print(json.dumps(estimate, sort_keys=True, indent=2))
    if float(estimate["safe_upper_bound_usd"]) > args.max_cost_usd:
        raise SystemExit(
            "Conservative worst-case estimate exceeds --max-cost-usd; no request was sent"
        )
    if args.dry_run:
        return 0

    # Load the credential only when a request genuinely needs to be submitted.
    # A fully authenticated receipt can be recovered without network access, and
    # requiring a key for that local repair would make restart recovery brittle.
    api_key: str | None = None

    shard_directory = args.work_dir / "response_shards"
    receipt_directory = args.work_dir / "api_receipts"
    attempt_directory = args.work_dir / "submission_attempts"
    blocked_directory = args.work_dir / "blocked_responses"
    shard_directory.mkdir(parents=True, exist_ok=True)
    receipt_directory.mkdir(parents=True, exist_ok=True)
    attempt_directory.mkdir(parents=True, exist_ok=True)

    for index, request_row in enumerate(requests, start=1):
        request_id = request_row["request_id"]
        shard_path = shard_directory / f"{request_id}.json"
        receipt_path = receipt_directory / f"{request_id}.json"
        attempt_path = attempt_directory / f"{request_id}.json"
        blocked_path = blocked_directory / f"{request_id}.json"
        client_request_id = _client_request_id(request_id, request_row["request"])
        if shard_path.exists() or receipt_path.exists():
            if not receipt_path.is_file():
                # Keep the existing precise error for the unsafe shard-only state.
                _existing_response(shard_path, receipt_path, request_row)
                raise AssertionError("shard-only response state unexpectedly passed")
            receipt = _validate_completed_receipt(
                receipt_path,
                request_row,
                client_request_id=client_request_id,
            )
            if not _existing_response(shard_path, receipt_path, request_row):
                raise AssertionError("persisted receipt was not recognized as an existing response")
            recovered = _recover_completed_attempt(
                attempt_path,
                request_row,
                receipt,
                client_request_id=client_request_id,
            )
            action = "recovered" if recovered else "verified existing"
            print(f"[{index}/{len(requests)}] {action} {request_id}", flush=True)
            continue

        prior_attempt_state = _validate_prior_attempt(
            attempt_path,
            request_row,
            client_request_id=client_request_id,
        )
        if prior_attempt_state not in {None, "rejected_not_processed"}:
            raise SystemExit(
                "A prior submission has no authenticated receipt and may have been "
                f"processed ({attempt_path}, state={prior_attempt_state!r}). Automatic "
                "resubmission is blocked; use the recorded X-Client-Request-Id to resolve "
                "the attempt before proceeding."
            )

        # Resolve the key before recording a new prepared attempt.  If the key is
        # absent, the command exits without leaving an attempt that could be
        # mistaken for an ambiguous provider submission on the next restart.
        if api_key is None:
            api_key = _load_api_key()
            if not api_key:
                raise SystemExit("OPENAI_API_KEY is not set; no request was sent")

        _atomic_json(
            attempt_path,
            _attempt_payload(
                request_row,
                client_request_id=client_request_id,
                state="prepared",
            ),
        )

        try:
            response, response_headers = _post_response(
                request_row["request"],
                api_key,
                client_request_id=client_request_id,
            )
        except RejectedSubmissionError as error:
            _atomic_json(
                attempt_path,
                _attempt_payload(
                    request_row,
                    client_request_id=client_request_id,
                    state="rejected_not_processed",
                    detail=str(error),
                    status_code=error.status_code,
                ),
            )
            raise
        except AmbiguousSubmissionError as error:
            _atomic_json(
                attempt_path,
                _attempt_payload(
                    request_row,
                    client_request_id=client_request_id,
                    state="ambiguous_blocked",
                    detail=str(error),
                    status_code=error.status_code,
                ),
            )
            raise

        try:
            raw_response = _validate_provider_response(response, response_headers)
        except Exception as error:
            _atomic_json(
                blocked_path,
                {
                    "schema_version": "sp_lense.openai_judge_blocked_response.v1",
                    "request_id": request_id,
                    "request_sha256": _sha256(request_row["request"]),
                    "x_client_request_id": client_request_id,
                    "x_request_id": response_headers.get("x_request_id"),
                    "api_response_sha256": _sha256(response),
                    "api_response": response,
                    "validation_error": str(error),
                },
            )
            _atomic_json(
                attempt_path,
                _attempt_payload(
                    request_row,
                    client_request_id=client_request_id,
                    state="response_validation_blocked",
                    detail=str(error),
                    server_request_id=response_headers.get("x_request_id"),
                ),
            )
            raise
        shard = {
            "schema_version": request_row["schema_version"],
            "request_id": request_id,
            "raw_response": raw_response,
        }
        receipt = {
            "schema_version": "sp_lense.openai_judge_api_receipt.v1",
            "request_id": request_id,
            "request_sha256": _sha256(request_row["request"]),
            "raw_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
            "api_response_id": response.get("id"),
            "api_model": response.get("model"),
            "api_status": response.get("status"),
            "x_client_request_id": response_headers.get("x_client_request_id"),
            "x_request_id": response_headers.get("x_request_id"),
            "usage": response.get("usage"),
            "api_response_sha256": _sha256(response),
            "raw_response": raw_response,
            "api_response": response,
        }
        _atomic_json(receipt_path, receipt)
        _atomic_json(shard_path, shard)
        _atomic_json(
            attempt_path,
            _attempt_payload(
                request_row,
                client_request_id=client_request_id,
                state="completed",
                server_request_id=response_headers.get("x_request_id"),
            ),
        )
        print(f"[{index}/{len(requests)}] recorded {request_id}", flush=True)

    _write_final_responses(requests, shard_directory, args.responses)
    print(f"Wrote {len(requests)} responses to {args.responses}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
