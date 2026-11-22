"""Account for all generated output tokens across an agent episode.

The caller supplies a prospective episode budget.  The ledger does not impose
an additional thinking or per-response cap; a caller can request a smaller
limit when the model's context or infrastructure requires it.  Actual usage
must include every generated suffix token, whether it is reasoning, a tool
command, malformed output, or a special token.
"""
from __future__ import annotations

import copy


class OutputTokenBudget:
    """Track one outstanding generation and retain uncertainty on failure."""

    def __init__(self, limit: int):
        if type(limit) is not int or limit <= 0:
            raise ValueError("episode output-token limit must be a positive integer")
        self.limit = limit
        self.known_used = 0
        self.active_cap = None
        self.stop_reason = None
        self.unknown_usage = False
        self._calls = []

    @property
    def remaining(self):
        """Return None while a call's final token usage is unknown."""
        if self.unknown_usage or self.active_cap is not None:
            return None
        return self.limit - self.known_used

    @property
    def stopped(self):
        return self.stop_reason is not None or self.remaining == 0

    def begin(self, requested=None):
        """Reserve one call and return its capped maximum generated tokens."""
        if self.active_cap is not None:
            raise RuntimeError("a generation is already outstanding")
        if self.stopped:
            raise RuntimeError("episode output budget is stopped")
        if requested is not None and (type(requested) is not int or requested <= 0):
            raise ValueError("requested output limit must be a positive integer")
        self.active_cap = min(requested, self.remaining) if requested is not None else self.remaining
        return self.active_cap

    def commit(self, generated_tokens):
        """Record actual token usage, independent of response validity."""
        if self.active_cap is None:
            raise RuntimeError("no generation is outstanding")
        if (type(generated_tokens) is not int or generated_tokens < 0
                or generated_tokens > self.active_cap):
            raise ValueError("generated tokens must fit the outstanding request")
        cap, before = self.active_cap, self.known_used
        self.known_used += generated_tokens
        self.active_cap = None
        record = {"requested_cap": cap, "generated_tokens": generated_tokens,
                  "known_before": before, "known_after": self.known_used,
                  "count_status": "known"}
        self._calls.append(record)
        if generated_tokens == 0:
            self.stop_reason = "zero_generated_tokens"
        elif self.known_used == self.limit:
            self.stop_reason = "output_budget_exhausted"
        return copy.deepcopy(record)

    def fail_unknown(self, reason):
        """Record an interrupted call whose generated-token count is unknown."""
        if self.active_cap is None:
            raise RuntimeError("no generation is outstanding")
        if not isinstance(reason, str) or not reason:
            raise ValueError("a nonempty failure reason is required")
        record = {"requested_cap": self.active_cap, "generated_tokens": None,
                  "known_before": self.known_used, "known_after": None,
                  "possible_total_upper_bound": self.known_used + self.active_cap,
                  "count_status": "unknown", "reason": reason}
        self._calls.append(record)
        self.active_cap = None
        self.unknown_usage = True
        self.stop_reason = "unknown_generated_tokens"
        return copy.deepcopy(record)

    def stop(self, reason):
        """Record an external stop after resolving any in-flight usage."""
        if self.active_cap is not None:
            raise RuntimeError("record outstanding usage before stopping")
        if not isinstance(reason, str) or not reason:
            raise ValueError("a nonempty stop reason is required")
        if self.stop_reason is None:
            self.stop_reason = reason

    def snapshot(self):
        """Return a detached view; do not imply unknown token usage was zero."""
        return {"limit": self.limit, "known_generated_tokens": self.known_used,
                "generated_tokens_total": None if self.unknown_usage or self.active_cap is not None else self.known_used,
                "remaining": self.remaining, "active_cap": self.active_cap,
                "stop_reason": self.stop_reason, "calls": copy.deepcopy(self._calls)}
