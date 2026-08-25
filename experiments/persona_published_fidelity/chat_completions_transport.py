"""Fail-closed Chat Completions transport for the published-fidelity sensitivity.

The transport is intentionally separate from the locked four-way comparison.  A
request is posted at most once.  If a process dies after recording its intent to
post but before recording a provider receipt, the outcome is treated as
ambiguous and must not be retried automatically.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from . import persona_published_fidelity as fidelity
except ImportError:  # pragma: no cover - direct script execution
    import persona_published_fidelity as fidelity


TRANSPORT_PREFLIGHT_SCHEMA = "sp_lense.persona_published_fidelity_transport_exact_byte_preflight.v1"
ATTEMPT_EVENT_SCHEMA = "sp_lense.persona_published_fidelity_transport_attempt.v1"
RECEIPT_SCHEMA = "sp_lense.persona_published_fidelity_transport_receipt.v1"
BLOCKED_SCHEMA = "sp_lense.persona_published_fidelity_transport_blocked_response.v1"
SUMMARY_SCHEMA = "sp_lense.persona_published_fidelity_transport_summary.v1"

DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 120.0
RETRY_POLICY: dict[str, Any] = {
    "maximum_automatic_POST_attempts_per_request": 1,
    "retry_after_preparing_event_without_receipt": False,
    "retry_after_network_exception": False,
    "retry_after_HTTP_408_409_429_or_5xx": False,
    "terminal_HTTP_statuses": list(range(400, 408)) + list(range(410, 429)) + list(range(430, 500)),
    "reason": (
        "a recorded POST intent without an authenticated provider receipt is an "
        "ambiguous send; fail closed so a crash cannot create a duplicate billable call"
    ),
}
_AMBIGUOUS_HTTP_STATUSES = {408, 409, 429}


@dataclass(frozen=True)
class HTTPResult:
    """Exact response returned by an injected or standard-library POST function."""

    status: int
    headers: Mapping[str, str]
    body: bytes


PostFunction = Callable[[str, Mapping[str, str], bytes, float], HTTPResult]


def _transport_paths(work_dir: Path, request_id: str) -> dict[str, Path]:
    fidelity._require_sha256(request_id, "transport request ID")
    return {
        "preparing": work_dir / "submission_attempts" / f"{request_id}.preparing.json",
        "completed": work_dir / "submission_attempts" / f"{request_id}.completed.json",
        "rejected": work_dir / "submission_attempts" / f"{request_id}.rejected.json",
        "ambiguous": work_dir / "submission_attempts" / f"{request_id}.ambiguous.json",
        "receipt": work_dir / "receipts" / f"{request_id}.json",
        "response": work_dir / "response_shards" / f"{request_id}.json",
        "blocked": work_dir / "blocked_responses" / f"{request_id}.json",
    }


def _evidence_exists(work_dir: Path) -> bool:
    return any(
        (work_dir / name).exists()
        for name in (
            "submission_attempts",
            "receipts",
            "response_shards",
            "blocked_responses",
            "judge_responses.jsonl",
            "transport_summary.json",
        )
    )


def _body_bytes(request: Mapping[str, Any]) -> bytes:
    # These are the exact bytes passed to urllib or the injected POST function.
    return fidelity.canonical_json_bytes(request)


def _ordered_request_identity(
    config: Mapping[str, Any], requests_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    requests = fidelity.read_jsonl(requests_path)
    identities: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in requests:
        fidelity.validate_judge_request(row, config)
        request_id = row["request_id"]
        if request_id in seen:
            raise ValueError(f"duplicate judge request ID: {request_id}")
        seen.add(request_id)
        body = _body_bytes(row["request"])
        identities.append(
            {
                "request_id": request_id,
                "request_row_sha256": fidelity.canonical_json_sha256(row),
                "request_body_sha256": fidelity.hashlib.sha256(body).hexdigest(),
            }
        )
    if not requests:
        raise ValueError("judge request file is empty")
    return requests, identities


def prepare_exact_byte_preflight(
    config: Mapping[str, Any],
    requests_path: Path,
    work_dir: Path,
    *,
    max_cost_usd: float,
) -> dict[str, Any]:
    """Freeze cost, request-file bytes, every body, and retry semantics.

    Once transport evidence exists, both preflight artifacts must already exist
    and be byte-identical.  This avoids silently authorizing a different request
    set after a partial run.
    """

    fidelity._reject_forbidden_path(requests_path)
    fidelity._reject_forbidden_path(work_dir)
    evidence = _evidence_exists(work_dir)
    legacy_path = work_dir / "cost_preflight.json"
    exact_path = work_dir / "exact_byte_preflight.json"
    if evidence and (not legacy_path.is_file() or not exact_path.is_file()):
        raise RuntimeError("transport evidence exists without both immutable preflights")

    cost = fidelity.publish_cost_preflight(
        config,
        requests_path,
        work_dir,
        max_cost_usd=max_cost_usd,
    )
    requests, identities = _ordered_request_identity(config, requests_path)
    exact_request_bytes = requests_path.read_bytes()
    payload = {
        "schema_version": TRANSPORT_PREFLIGHT_SCHEMA,
        "study_role": fidelity.SECONDARY_ROLE,
        "api": "chat.completions",
        "api_url": config["judge"]["api_url"],
        "model": config["judge"]["model"],
        "requests_file_sha256": fidelity.hashlib.sha256(exact_request_bytes).hexdigest(),
        "requests_file_size_bytes": len(exact_request_bytes),
        "request_count": len(requests),
        "ordered_request_identities": identities,
        "ordered_request_identities_sha256": fidelity.canonical_json_sha256(identities),
        "cost_preflight_sha256": fidelity.file_sha256(legacy_path),
        "user_cost_ceiling_usd": float(max_cost_usd),
        "retry_policy": RETRY_POLICY,
        "retry_policy_sha256": fidelity.canonical_json_sha256(RETRY_POLICY),
        "authorization_scope": (
            "exact frozen request bytes only; API key is read only immediately before a new POST"
        ),
    }
    fidelity.publish_exact_json(exact_path, payload)
    # Re-verify rather than trusting a pre-existing artifact.
    observed = fidelity.load_json(exact_path)
    if observed != payload:
        raise ValueError("exact-byte preflight differs from recomputation")
    if cost.get("user_cost_ceiling_usd") != float(max_cost_usd):
        raise ValueError("cost preflight ceiling differs from exact-byte preflight")
    return payload


def verify_exact_byte_preflight(
    config: Mapping[str, Any], requests_path: Path, work_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    legacy_path = work_dir / "cost_preflight.json"
    exact_path = work_dir / "exact_byte_preflight.json"
    if not legacy_path.is_file() or not exact_path.is_file():
        raise FileNotFoundError("both immutable judge transport preflights are required")
    cost = fidelity._require_matching_preflight(config, requests_path, legacy_path)
    observed = fidelity.load_json(exact_path)
    if not isinstance(observed, dict):
        raise TypeError("exact-byte preflight is not an object")
    requests, identities = _ordered_request_identity(config, requests_path)
    raw = requests_path.read_bytes()
    expected = {
        "schema_version": TRANSPORT_PREFLIGHT_SCHEMA,
        "study_role": fidelity.SECONDARY_ROLE,
        "api": "chat.completions",
        "api_url": config["judge"]["api_url"],
        "model": config["judge"]["model"],
        "requests_file_sha256": fidelity.hashlib.sha256(raw).hexdigest(),
        "requests_file_size_bytes": len(raw),
        "request_count": len(requests),
        "ordered_request_identities": identities,
        "ordered_request_identities_sha256": fidelity.canonical_json_sha256(identities),
        "cost_preflight_sha256": fidelity.file_sha256(legacy_path),
        "user_cost_ceiling_usd": float(cost["user_cost_ceiling_usd"]),
        "retry_policy": RETRY_POLICY,
        "retry_policy_sha256": fidelity.canonical_json_sha256(RETRY_POLICY),
        "authorization_scope": (
            "exact frozen request bytes only; API key is read only immediately before a new POST"
        ),
    }
    if observed != expected:
        raise ValueError("exact-byte preflight differs from current bytes or transport policy")
    return observed, requests


def _header(headers: Mapping[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return str(value)
    return None


def _validate_usage(raw: Mapping[str, Any], *, success: bool) -> dict[str, int] | None:
    usage = raw.get("usage")
    if not success and usage is None:
        return None
    if not isinstance(usage, Mapping):
        raise TypeError("successful Chat Completions response has no usage object")
    required = ("prompt_tokens", "completion_tokens", "total_tokens")
    values: dict[str, int] = {}
    for name in required:
        value = usage.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"provider usage {name} is invalid")
        values[name] = value
    if values["total_tokens"] != values["prompt_tokens"] + values["completion_tokens"]:
        raise ValueError("provider total_tokens differs from prompt plus completion tokens")
    if success and values["completion_tokens"] > 1:
        raise ValueError("provider exceeded the frozen one-token completion cap")
    return values


def _parse_provider_body(result: HTTPResult) -> tuple[dict[str, Any], str]:
    if not isinstance(result.status, int) or not 100 <= result.status <= 599:
        raise ValueError("HTTP status is invalid")
    try:
        text = result.body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("provider response body is not UTF-8 JSON") from error
    try:
        raw = json.loads(text, parse_constant=fidelity._reject_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("provider response body is not valid finite JSON") from error
    if not isinstance(raw, dict):
        raise TypeError("provider response JSON is not an object")
    return raw, text


def _actual_cost_usd(config: Mapping[str, Any], usages: Sequence[Mapping[str, int]]) -> float:
    prices = config["judge"]["budgeting_prices_usd_per_million"]
    prompt = sum(item["prompt_tokens"] for item in usages)
    completion = sum(item["completion_tokens"] for item in usages)
    return (prompt * float(prices["input"]) + completion * float(prices["output"])) / 1_000_000


def _attempt_event(
    request: Mapping[str, Any], exact_preflight: Mapping[str, Any], state: str
) -> dict[str, Any]:
    return {
        "schema_version": ATTEMPT_EVENT_SCHEMA,
        "study_role": fidelity.SECONDARY_ROLE,
        "request_id": request["request_id"],
        "request_row_sha256": fidelity.canonical_json_sha256(request),
        "request_body_sha256": fidelity.hashlib.sha256(_body_bytes(request["request"])).hexdigest(),
        "exact_byte_preflight_sha256": fidelity.canonical_json_sha256(exact_preflight),
        "state": state,
        "post_ordinal": 1,
        "retry_policy_sha256": fidelity.canonical_json_sha256(RETRY_POLICY),
    }


def _validate_receipt(
    receipt: Mapping[str, Any],
    request: Mapping[str, Any],
    exact_preflight: Mapping[str, Any],
) -> tuple[str, dict[str, Any] | None, dict[str, int] | None]:
    required = {
        "schema_version",
        "study_role",
        "request_id",
        "request_row_sha256",
        "request_body_sha256",
        "exact_byte_preflight_sha256",
        "client_request_id",
        "x_request_id",
        "http_status",
        "disposition",
        "raw_response_body_base64",
        "raw_response_body_sha256",
        "raw_response",
        "usage",
    }
    if set(receipt) != required:
        raise ValueError("transport receipt fields differ from the locked schema")
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("study_role") != fidelity.SECONDARY_ROLE
        or receipt.get("request_id") != request["request_id"]
        or receipt.get("request_row_sha256") != fidelity.canonical_json_sha256(request)
        or receipt.get("request_body_sha256")
        != fidelity.hashlib.sha256(_body_bytes(request["request"])).hexdigest()
        or receipt.get("exact_byte_preflight_sha256")
        != fidelity.canonical_json_sha256(exact_preflight)
        or receipt.get("client_request_id") != request["request_id"]
        or not isinstance(receipt.get("x_request_id"), str)
        or not receipt["x_request_id"].strip()
    ):
        raise ValueError("transport receipt identity is invalid")
    try:
        body = base64.b64decode(receipt["raw_response_body_base64"], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("transport receipt body encoding is invalid") from error
    if fidelity.hashlib.sha256(body).hexdigest() != receipt.get("raw_response_body_sha256"):
        raise ValueError("transport receipt body hash is invalid")
    raw, _ = _parse_provider_body(HTTPResult(int(receipt["http_status"]), {}, body))
    if raw != receipt.get("raw_response"):
        raise ValueError("transport receipt parsed response differs from exact body")
    disposition = receipt.get("disposition")
    if disposition not in {"accepted", "rejected", "ambiguous"}:
        raise ValueError("transport receipt disposition is invalid")
    success = disposition == "accepted"
    usage = _validate_usage(raw, success=success)
    if receipt.get("usage") != usage:
        raise ValueError("transport receipt usage differs from provider response")
    return str(disposition), raw, usage


def _receipt_from_result(
    request: Mapping[str, Any],
    exact_preflight: Mapping[str, Any],
    result: HTTPResult,
    disposition: str,
) -> dict[str, Any]:
    raw, _ = _parse_provider_body(result)
    x_request_id = _header(result.headers, "x-request-id")
    if x_request_id is None or not x_request_id.strip():
        raise ValueError("provider response is missing x-request-id")
    usage = _validate_usage(raw, success=disposition == "accepted")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "study_role": fidelity.SECONDARY_ROLE,
        "request_id": request["request_id"],
        "request_row_sha256": fidelity.canonical_json_sha256(request),
        "request_body_sha256": fidelity.hashlib.sha256(_body_bytes(request["request"])).hexdigest(),
        "exact_byte_preflight_sha256": fidelity.canonical_json_sha256(exact_preflight),
        "client_request_id": request["request_id"],
        "x_request_id": x_request_id.strip(),
        "http_status": result.status,
        "disposition": disposition,
        "raw_response_body_base64": base64.b64encode(result.body).decode("ascii"),
        "raw_response_body_sha256": fidelity.hashlib.sha256(result.body).hexdigest(),
        "raw_response": raw,
        "usage": usage,
    }


def _recover_or_classify(
    request: Mapping[str, Any], exact_preflight: Mapping[str, Any], work_dir: Path
) -> tuple[str, dict[str, int] | None]:
    paths = _transport_paths(work_dir, request["request_id"])
    receipt_exists = paths["receipt"].is_file()
    response_exists = paths["response"].is_file()
    blocked_exists = paths["blocked"].is_file()
    preparing_exists = paths["preparing"].is_file()
    terminal_events = [
        name for name in ("completed", "rejected", "ambiguous") if paths[name].is_file()
    ]
    if len(terminal_events) > 1:
        raise RuntimeError("request has conflicting terminal transport events")
    if (response_exists or blocked_exists or terminal_events) and not receipt_exists:
        raise RuntimeError("transport evidence exists without its provider receipt")
    if receipt_exists:
        receipt = fidelity.load_json(paths["receipt"])
        if not isinstance(receipt, dict):
            raise TypeError("transport receipt is not an object")
        disposition, raw, usage = _validate_receipt(receipt, request, exact_preflight)
        assert raw is not None
        if disposition == "accepted":
            exchange = {"request_id": request["request_id"], "raw_response": raw}
            fidelity.publish_exact_json(paths["response"], exchange)
            fidelity.publish_exact_json(
                paths["completed"], _attempt_event(request, exact_preflight, "completed")
            )
            if blocked_exists or paths["rejected"].exists() or paths["ambiguous"].exists():
                raise RuntimeError("accepted receipt conflicts with blocked transport evidence")
            return "accepted", usage
        blocked = {
            "schema_version": BLOCKED_SCHEMA,
            "study_role": fidelity.SECONDARY_ROLE,
            "request_id": request["request_id"],
            "receipt_sha256": fidelity.file_sha256(paths["receipt"]),
            "disposition": disposition,
            "automatic_retry_allowed": False,
        }
        fidelity.publish_exact_json(paths["blocked"], blocked)
        fidelity.publish_exact_json(
            paths[disposition], _attempt_event(request, exact_preflight, disposition)
        )
        if response_exists or paths["completed"].exists():
            raise RuntimeError("blocked receipt conflicts with accepted response evidence")
        return disposition, usage
    if preparing_exists:
        # A POST may have left the machine.  Never infer that it did not.
        return "ambiguous_preparing_without_receipt", None
    return "new", None


def _urllib_post(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> HTTPResult:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HTTPResult(
                int(response.status),
                {str(key): str(value) for key, value in response.headers.items()},
                response.read(),
            )
    except urllib.error.HTTPError as error:
        return HTTPResult(
            int(error.code),
            {str(key): str(value) for key, value in error.headers.items()},
            error.read(),
        )


def _disposition_for_status(status: int) -> str:
    if 200 <= status <= 299:
        return "accepted"
    if status in _AMBIGUOUS_HTTP_STATUSES or 500 <= status <= 599:
        return "ambiguous"
    if 400 <= status <= 499:
        return "rejected"
    return "ambiguous"


def _aggregate_complete_responses(
    config: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    exact_preflight: Mapping[str, Any],
    work_dir: Path,
) -> dict[str, Any]:
    exchanges: list[dict[str, Any]] = []
    usages: list[dict[str, int]] = []
    x_request_ids: list[str] = []
    for request in requests:
        paths = _transport_paths(work_dir, request["request_id"])
        state, usage = _recover_or_classify(request, exact_preflight, work_dir)
        if state != "accepted" or usage is None:
            raise RuntimeError(f"judge transport is incomplete at {request['request_id']}: {state}")
        exchange = fidelity.load_json(paths["response"])
        if not isinstance(exchange, dict):
            raise TypeError("judge response shard is not an object")
        exchanges.append(exchange)
        usages.append(usage)
        receipt = fidelity.load_json(paths["receipt"])
        assert isinstance(receipt, dict)
        x_request_ids.append(str(receipt["x_request_id"]))
    fidelity.build_score_rows(config, requests, exchanges)
    if len(set(x_request_ids)) != len(x_request_ids):
        raise RuntimeError("provider x-request-id values are not unique across judge calls")
    aggregate_path = work_dir / "judge_responses.jsonl"
    aggregate_sha = fidelity.publish_exact_jsonl(aggregate_path, exchanges)
    actual_cost = _actual_cost_usd(config, usages)
    ceiling = float(exact_preflight["user_cost_ceiling_usd"])
    if actual_cost > ceiling + 1e-12:
        raise RuntimeError("observed judge usage exceeds the immutable user cost ceiling")
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "study_role": fidelity.SECONDARY_ROLE,
        "status": "complete_verified",
        "main_ranking_eligible": False,
        "request_count": len(requests),
        "response_count": len(exchanges),
        "unique_x_request_id_count": len(set(x_request_ids)),
        "prompt_tokens": sum(item["prompt_tokens"] for item in usages),
        "completion_tokens": sum(item["completion_tokens"] for item in usages),
        "total_tokens": sum(item["total_tokens"] for item in usages),
        "actual_cost_usd": actual_cost,
        "user_cost_ceiling_usd": ceiling,
        "requests_file_sha256": exact_preflight["requests_file_sha256"],
        "exact_byte_preflight_sha256": fidelity.canonical_json_sha256(exact_preflight),
        "judge_responses_sha256": aggregate_sha,
    }
    fidelity.publish_exact_json(work_dir / "transport_summary.json", summary)
    return summary


def run_transport(
    config: Mapping[str, Any],
    requests_path: Path,
    work_dir: Path,
    *,
    max_cost_usd: float | None = None,
    verify_only: bool = False,
    api_key: str | None = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    post: PostFunction | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    limit: int | None = None,
) -> dict[str, Any]:
    """Post missing requests once, recover receipts, or verify without credentials."""

    if max_cost_usd is not None:
        exact = prepare_exact_byte_preflight(
            config, requests_path, work_dir, max_cost_usd=max_cost_usd
        )
        requests = _ordered_request_identity(config, requests_path)[0]
    else:
        exact, requests = verify_exact_byte_preflight(config, requests_path, work_dir)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("transport timeout must be positive and finite")
    if limit is not None and limit < 0:
        raise ValueError("transport limit must be nonnegative")

    accepted = 0
    blocked: list[tuple[str, str]] = []
    new_rows: list[Mapping[str, Any]] = []
    usages: list[dict[str, int]] = []
    for request in requests:
        state, usage = _recover_or_classify(request, exact, work_dir)
        if state == "accepted":
            accepted += 1
            assert usage is not None
            usages.append(usage)
        elif state == "new":
            new_rows.append(request)
        else:
            blocked.append((request["request_id"], state))

    if blocked:
        detail = ", ".join(f"{request_id}:{state}" for request_id, state in blocked[:3])
        raise RuntimeError(f"judge transport contains fail-closed blocked requests: {detail}")
    if verify_only:
        if new_rows:
            raise RuntimeError(
                f"verify-only transport is incomplete: {len(new_rows)} request(s) have no receipt"
            )
        return _aggregate_complete_responses(config, requests, exact, work_dir)

    remaining = new_rows if limit is None else new_rows[:limit]
    post_function = _urllib_post if post is None else post
    for request in remaining:
        # Credential resolution occurs only here, immediately before a genuinely
        # new POST.  Resume and verify-only paths never touch the environment.
        resolved_key = api_key if api_key is not None else os.environ.get(api_key_env)
        if not isinstance(resolved_key, str) or not resolved_key.strip():
            raise RuntimeError(
                f"{api_key_env} (or --api-key) is required only for a new judge POST"
            )
        paths = _transport_paths(work_dir, request["request_id"])
        fidelity.publish_exact_json(paths["preparing"], _attempt_event(request, exact, "preparing"))
        body = _body_bytes(request["request"])
        headers = {
            "Authorization": f"Bearer {resolved_key.strip()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Client-Request-Id": request["request_id"],
        }
        try:
            result = post_function(str(config["judge"]["api_url"]), headers, body, timeout_seconds)
            disposition = _disposition_for_status(result.status)
            receipt = _receipt_from_result(request, exact, result, disposition)
            # Receipt is durable before any derived response shard or terminal event.
            fidelity.publish_exact_json(paths["receipt"], receipt)
            state, usage = _recover_or_classify(request, exact, work_dir)
            if state != "accepted":
                raise RuntimeError(
                    f"provider response for {request['request_id']} is fail-closed: {state}"
                )
            assert usage is not None
            usages.append(usage)
            accepted += 1
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            # The preparing event is deliberately left without a receipt.  A
            # later run sees an ambiguous send and will not issue a duplicate.
            raise RuntimeError(
                f"network outcome is ambiguous for {request['request_id']}; automatic retry forbidden"
            ) from error

        if _actual_cost_usd(config, usages) > float(exact["user_cost_ceiling_usd"]) + 1e-12:
            raise RuntimeError("observed usage exceeded the immutable cost ceiling; stopping")

    if accepted == len(requests):
        return _aggregate_complete_responses(config, requests, exact, work_dir)
    return {
        "schema_version": SUMMARY_SCHEMA,
        "study_role": fidelity.SECONDARY_ROLE,
        "status": "partial_restart_safe",
        "main_ranking_eligible": False,
        "request_count": len(requests),
        "accepted_count": accepted,
        "remaining_count": len(requests) - accepted,
        "new_POSTs_this_invocation": len(remaining),
        "network_call_made": bool(remaining),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receipt-bound, at-most-once Chat Completions judge transport"
    )
    parser.add_argument("--config", type=Path, default=fidelity.DEFAULT_CONFIG_PATH)
    parser.add_argument("--lock", type=Path, default=fidelity.DEFAULT_LOCK_PATH)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = fidelity.load_config(args.config)
    fidelity.verify_lock(args.config, args.lock)
    result = run_transport(
        config,
        args.requests,
        args.work_dir,
        max_cost_usd=args.max_cost_usd,
        verify_only=args.verify_only,
        api_key_env=args.api_key_env,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
