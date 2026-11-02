"""Preserve the model-visible sequence of actions and tool feedback.

This module does not execute tools or decide which state the model may see.
Callers must supply only observations that are appropriate to return to the
model; simulator-private state must not be passed to this history.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any


class ActionHistory:
    """Keep an isolated ordered transcript for one evaluation episode."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def append(
        self,
        *,
        raw_response: str,
        tool_response: Any,
        state_before: Any,
        state_after: Any,
        success: bool,
        noop: bool,
        parsed_command: dict[str, Any] | None = None,
        parse_error: str | None = None,
        final_action_text: str | None = None,
    ) -> dict[str, Any]:
        """Append an action exchange, including malformed or unsuccessful ones.

        ``raw_response`` is retained byte-for-byte as a Python string. For
        thinking-enabled models, ``final_action_text`` may additionally retain
        the extracted action; the full original response is never replaced.
        ``state_before`` and ``state_after`` must be model-visible observations,
        not private simulator state. Neither successful parsing nor a tool
        success is inferred by this class.
        """
        if not isinstance(raw_response, str):
            raise TypeError("raw_response must be a string")
        if final_action_text is not None and not isinstance(final_action_text, str):
            raise TypeError("final_action_text must be a string or None")
        if parsed_command is not None and not isinstance(parsed_command, dict):
            raise TypeError("parsed_command must be a dictionary or None")
        if parse_error is not None and not isinstance(parse_error, str):
            raise TypeError("parse_error must be a string or None")
        if type(success) is not bool or type(noop) is not bool:
            raise TypeError("success and noop must be booleans")

        entry: dict[str, Any] = {
            "entry_sequence": len(self._entries) + 1,
            "raw_response": raw_response,
            "raw_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
            "parse_error": parse_error,
            "tool_response": copy.deepcopy(tool_response),
            "state_before": copy.deepcopy(state_before),
            "state_after": copy.deepcopy(state_after),
            "success": success,
            "noop": noop,
        }
        if final_action_text is not None:
            entry["final_action_text"] = final_action_text
        if parsed_command is not None:
            entry["parsed_command"] = copy.deepcopy(parsed_command)
            if "action" in parsed_command:
                entry["action"] = copy.deepcopy(parsed_command["action"])
                entry["arguments"] = {
                    key: copy.deepcopy(value)
                    for key, value in parsed_command.items()
                    if key != "action"
                }
        self._entries.append(entry)
        return copy.deepcopy(entry)

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a copy suitable for including in the next model prompt."""
        return copy.deepcopy(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
