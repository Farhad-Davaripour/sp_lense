"""Exact v2 serialization contract for one registered symbolic action.

The contract accepts either a bare JSON object or one complete lowercase
``json`` Markdown fence containing that object. It never extracts JSON from
prose, repairs malformed content, coerces values, or accepts extra keys.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re


SERIALIZATION_CONTRACT_VERSION = "research4-action-serialization-v2"
_JSON_FENCE = re.compile(
    r"\A[\t\n\r ]*```json[\t ]*\r?\n(?P<body>.*?)\r?\n```[\t\n\r ]*\Z",
    re.DOTALL,
)


def serialization_contract() -> dict:
    """Return the frozen, hashable parser policy used across all stages."""
    return {
        "version": SERIALIZATION_CONTRACT_VERSION,
        "accepted_forms": ["bare_json_object", "single_lowercase_json_fence"],
        "whitespace_outside_fence_only": True,
        "fence_body_contains_no_other_fence": True,
        "required_object_keys": ["action"],
        "duplicate_keys_rejected": True,
        "registered_action_required": True,
        "prose_extraction": False,
        "repair": False,
        "coercion": False,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def parse_registered_action(raw_response: str, allowed: Sequence[str]) -> dict:
    """Parse one exact registered action and return auditable normalization.

    The raw string is never modified in the returned record; the SHA-256 is of
    those exact UTF-8 bytes. ``normalization_used`` means only that one strict
    outer ``json`` fence was removed before JSON decoding.
    """
    raw = raw_response if isinstance(raw_response, str) else None
    raw_sha256 = (hashlib.sha256(raw.encode("utf-8")).hexdigest()
                  if raw is not None else None)
    result = {
        "contract_version": SERIALIZATION_CONTRACT_VERSION,
        "valid": False,
        "action": None,
        "normalization_used": False,
        "serialization_form": None,
        "raw_response_sha256": raw_sha256,
    }
    if raw is None or not raw or isinstance(allowed, (str, bytes)):
        return result
    candidate = raw
    match = _JSON_FENCE.fullmatch(raw)
    if match is not None:
        candidate = match.group("body")
        if "```" in candidate:
            return result
        result["normalization_used"] = True
        result["serialization_form"] = "single_lowercase_json_fence"
    elif "```" in raw:
        return result
    else:
        result["serialization_form"] = "bare_json_object"
    try:
        value = json.loads(candidate, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, json.JSONDecodeError):
        return result
    if not isinstance(value, Mapping) or set(value) != {"action"}:
        return result
    action = value.get("action")
    allowed_values = tuple(allowed) if isinstance(allowed, Sequence) else ()
    if (not isinstance(action, str) or not allowed_values
            or any(not isinstance(item, str) for item in allowed_values)
            or action not in allowed_values):
        return result
    result["valid"] = True
    result["action"] = action
    return result
