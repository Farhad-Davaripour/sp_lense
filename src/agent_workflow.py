"""Compose model-output accounting, discovery, and action feedback.

This is transport-neutral: an application must connect its model and sandbox
or Inspect tools to it. The caller supplies *model-visible* observations only,
and remains responsible for sandboxing, argument validation, episode limits,
and accurately reporting completion and generated token usage.
"""
from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .agent_action_history import ActionHistory
from .agent_token_budget import OutputTokenBudget
from .agent_tool_discovery import ToolDiscovery
from .thinking_response_transport import ThinkingResponse, parse_thinking_response, parse_tokenized_thinking


@dataclass(frozen=True)
class WorkflowResult:
    """Full response, usage receipt and the exchange, if one occurred."""

    response: ThinkingResponse
    usage: dict[str, Any]
    exchange: dict[str, Any] | None


class AgentWorkflow:
    """One episode. Instantiate afresh for each independent episode.

    ``tool_callback(command, visible_state, discovery, parse_status)`` returns a mapping with explicit
    ``feedback``, ``state_after``, ``success`` and ``noop`` fields. Both states
    must be observations suitable for the model, not private simulator state.
    The callback handles every tool attempt, including discovery and unknown
    names. It owns authorization, argument validation, feedback and state changes;
    it can use ``discovery.discover()`` or ``discovery.is_available(name)``.
    A naturally completed malformed JSON or fenced action reaches the callback
    as ``command=None`` with its parser status; no tool should be executed. For a
    parsed action, ``parse_status`` is None.
    """

    def __init__(self, tools: Sequence[Mapping[str, Any]], output_token_limit: int,
                 tool_callback: Callable[[dict[str, Any] | None, Any, ToolDiscovery, str | None], Mapping[str, Any]]):
        if not callable(tool_callback):
            raise TypeError("tool_callback must be callable")
        self.discovery = ToolDiscovery(tools)
        self.history = ActionHistory()
        self.budget = OutputTokenBudget(output_token_limit)
        self._tool_callback = tool_callback

    def prompt_context(self, visible_state: Any) -> dict[str, Any]:
        """Return discovery entry point, current observation and action history."""
        return {"tools": self.discovery.initial_catalog(),
                "observation": copy.deepcopy(visible_state),
                "history": self.history.snapshot()}

    def begin_generation(self, requested: int | None = None) -> int:
        """Reserve an output-token cap before calling the model."""
        return self.budget.begin(requested)

    def finish_generation(self, decoded_suffix: str, generated_tokens: int | None, *,
                          natural_completion: bool | None, visible_state: Any,
                          raw_suffix: str | None = None) -> WorkflowResult:
        """Record usage, parse a completed generation and dispatch an action attempt.

        Usage is recorded even for malformed or incomplete responses. Unknown
        usage stops the budget and cannot dispatch. Completed malformed actions
        are reported to the callback as errors without executing a tool. Full raw
        reasoning is in the returned response, while history stores the final text.
        """
        if generated_tokens is None:
            usage = self.budget.fail_unknown("generation token count unavailable")
        else:
            usage = self.budget.commit(generated_tokens)
        response = parse_thinking_response(decoded_suffix, generated_tokens,
                                           natural_completion=natural_completion,
                                           raw_suffix=raw_suffix)
        return self._apply_response(response, usage, visible_state)

    def finish_tokenized(self, token_ids: Sequence[int], *, eos_token_ids: Sequence[int],
                         terminal_token_ids: Sequence[int], decode: Callable[[list[int]], str],
                         visible_state: Any) -> WorkflowResult:
        """Record generated suffix IDs and use trusted EOS IDs to verify completion."""
        usage = self.budget.commit(len(token_ids))
        response = parse_tokenized_thinking(token_ids, eos_token_ids=eos_token_ids,
                                            terminal_token_ids=terminal_token_ids,
                                            decode=decode)
        return self._apply_response(response, usage, visible_state)

    def _apply_response(self, response: ThinkingResponse, usage: dict[str, Any],
                        visible_state: Any) -> WorkflowResult:
        malformed = (response.status in {"malformed_final_action", "unsupported_final_wrapper",
                                         "missing_final_action"}
                     and response.natural_completion is True and response.generated_tokens is not None
                     and response.generated_tokens > 0)
        if not response.dispatchable and not malformed:
            return WorkflowResult(response, usage, None)

        command = copy.deepcopy(response.command)
        before = copy.deepcopy(visible_state)
        parse_status = response.status if malformed else None
        outcome = self._tool_callback(copy.deepcopy(command), copy.deepcopy(before),
                                      self.discovery, parse_status)
        if not isinstance(outcome, Mapping):
            raise TypeError("tool callback must return a mapping")
        for field in ("feedback", "state_after", "success", "noop"):
            if field not in outcome:
                raise ValueError(f"tool callback must return {field}")
        if type(outcome["success"]) is not bool or type(outcome["noop"]) is not bool:
            raise TypeError("tool callback success and noop must be booleans")

        exchange = self.history.append(raw_response=response.raw_final_text or "",
                                       final_action_text=response.final_text,
                                       parsed_command=command,
                                       parse_error=parse_status,
                                       tool_response=outcome["feedback"],
                                       state_before=before, state_after=outcome["state_after"],
                                       success=outcome["success"], noop=outcome["noop"])
        return WorkflowResult(response, usage, exchange)
