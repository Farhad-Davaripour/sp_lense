"""Separate a thinking-enabled response from its final tool command.

This module only parses output; tool-specific validation and dispatch remain
the caller's responsibility. The full decoded suffix is retained unchanged.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class ThinkingResponse:
    raw_suffix: str
    generated_tokens: int | None
    reasoning: str | None
    final_text: str | None
    command: dict[str, Any] | None
    status: str
    natural_completion: bool | None
    raw_final_text: str | None = None

    @property
    def dispatchable(self) -> bool:
        """A caller must still validate the tool name and arguments."""
        return self.status == "ready_for_tool_validation"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def parse_thinking_response(
    decoded_suffix: str, generated_tokens: int | None, *,
    natural_completion: bool | None, raw_suffix: str | None = None,
) -> ThinkingResponse:
    """Parse a decoded suffix after trusted terminal tokens have been removed.

    Qwen's opening thinking marker may already be in the prompt. A single
    complete lowercase JSON fence is supported; prose extraction and repairs
    are not. The caller is responsible for providing truthful completion and
    token counts; unknown or incomplete generations never become commands.
    """
    if not isinstance(decoded_suffix, str) or (raw_suffix is not None and not isinstance(raw_suffix, str)):
        raise TypeError("decoded and raw suffixes must be strings")
    if generated_tokens is not None and (type(generated_tokens) is not int or generated_tokens < 0):
        raise ValueError("generated token count must be nonnegative or unknown")
    if natural_completion is not None and type(natural_completion) is not bool:
        raise ValueError("natural completion must be boolean or unknown")
    raw = decoded_suffix if raw_suffix is None else raw_suffix
    raw_final = None

    def result(status, reasoning=None, final_text=None, command=None):
        return ThinkingResponse(raw, generated_tokens, reasoning, final_text, command, status,
                                natural_completion, raw_final)

    if generated_tokens == 0:
        return result("zero_generated_tokens")
    suffix = decoded_suffix.lstrip()
    if suffix.startswith("<think>"):
        suffix = suffix[len("<think>"):]
    if "</think>" not in suffix:
        return result("incomplete_thinking", suffix)
    reasoning, final = suffix.split("</think>", 1)
    raw_final = final
    if not final.strip():
        return result("missing_final_action", reasoning, final)
    candidate = final.strip()
    if candidate.startswith("```"):
        match = re.fullmatch(r"```json\r?\n([\s\S]*)\r?\n```", candidate)
        if match is None or "```" in match.group(1):
            return result("unsupported_final_wrapper", reasoning, final)
        candidate = match.group(1)
    try:
        command = json.loads(candidate, object_pairs_hook=_unique_object,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (TypeError, ValueError, RecursionError):
        return result("malformed_final_action", reasoning, candidate)
    if not isinstance(command, dict) or not isinstance(command.get("action"), str) or not command["action"]:
        return result("malformed_final_action", reasoning, candidate)
    if natural_completion is False:
        return result("generation_incomplete", reasoning, candidate, command)
    if generated_tokens is None:
        return result("usage_unknown", reasoning, candidate, command)
    if natural_completion is None:
        return result("completion_unknown", reasoning, candidate, command)
    return result("ready_for_tool_validation", reasoning, candidate, command)


def parse_tokenized_thinking(
    token_ids: Sequence[int], *, eos_token_ids: Sequence[int],
    terminal_token_ids: Sequence[int], decode: Callable[[list[int]], str],
) -> ThinkingResponse:
    """Verify natural EOS and remove only caller-declared trailing token IDs.

    `decode` should retain special-token text, allowing the original suffix
    to be recorded separately from the clean content passed to the parser.
    Never remove terminal markers by textual search inside a response.
    """
    if (not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes))
            or any(type(token) is not int for token in token_ids)):
        raise ValueError("token_ids must be a sequence of integers")
    if (any(type(token) is not int for token in eos_token_ids)
            or any(type(token) is not int for token in terminal_token_ids)
            or not set(eos_token_ids).issubset(set(terminal_token_ids))):
        raise ValueError("valid EOS IDs must be included in terminal token IDs")
    tokens = list(token_ids)
    raw = decode(tokens)
    if not eos_token_ids:
        return ThinkingResponse(raw, len(tokens), None, None, None, "completion_unknown", None)
    natural = bool(tokens and tokens[-1] in eos_token_ids)
    content = list(tokens)
    if natural:
        content.pop()
        while content:
            if content[-1] in terminal_token_ids:
                content.pop()
                continue
            # Qwen can place a newline between two terminal markers. Remove
            # such whitespace only when another verified marker precedes it.
            index = len(content)
            while index and decode([content[index - 1]]).strip() == "":
                index -= 1
            if index and index < len(content) and content[index - 1] in terminal_token_ids:
                del content[index - 1:]
                continue
            break
    return parse_thinking_response(decode(content), len(tokens), natural_completion=natural, raw_suffix=raw)
