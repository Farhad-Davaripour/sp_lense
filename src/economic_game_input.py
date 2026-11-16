"""Unactivated next-run input view; shares the existing Inspect loop and world.

No current driver imports this module. Callers must prospectively bind this new
protocol before use. System/user objectives, generation and scoring are unchanged.
"""
from __future__ import annotations

import json
from . import hosted_inspect_runner as hosted
from .compact_game_input import CompactPublicState, CompactGameEpisode
from .public_economic_summary import economic_summary

PROTOCOL = "company_world_compact_public_economic_summary_v1"


class EconomicPublicState(CompactPublicState):
    def render(self, source_prompt):
        payload = json.loads(super().render(source_prompt))
        payload["economic_summary"] = economic_summary(payload)
        payload["input_view_protocol"] = PROTOCOL
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class EconomicGameEpisode(CompactGameEpisode):
    def __init__(self, episode, *, user_task_instruction=None, on_event=None, audit_identity=None):
        super().__init__(episode, user_task_instruction=user_task_instruction,
                         on_event=on_event, audit_identity=audit_identity)
        self.view = EconomicPublicState(user_task_instruction=user_task_instruction)

    def prompt(self):
        source = self.episode.prompt()
        result = self.view.render(source)
        hosted.notify_event(self.on_event, {**self.audit_identity, "type": "compact_input_source",
            "decision": self.decisions, "source_prompt": source, "compact_prompt": result,
            "input_view_protocol": PROTOCOL})
        return result


def make_economic_task(fixture, conditions, max_decisions, timeout_seconds, *,
                       episode_factory, user_task_instruction=None, **kwargs):
    """Opt-in only; use identical summaries for identical public state in all arms."""
    if "history_mode" in kwargs or "fresh_context_each_turn" in kwargs:
        raise ValueError("Economic input view owns fresh-context mode")

    def wrapped(identity):
        return EconomicGameEpisode(episode_factory(identity), user_task_instruction=user_task_instruction,
            on_event=kwargs.get("on_event"), audit_identity=identity)

    return hosted.make_task(fixture, conditions, max_decisions, timeout_seconds,
        episode_factory=wrapped, fresh_context_each_turn=True, **kwargs)
