"""Run the existing local fictional simulator through an Inspect-hosted model.

The scenario and its results are deliberately external to the repository.
No real shell, network, credentials, or compute controls are exposed as tools.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable


FIXTURE_MODULES = (
    "realistic_report_fixture_candidate",
    "realistic_report_fixture_history_v2_candidate",
    "realistic_report_discovery_thinking_fixture_candidate",
)
BRIEF = "Think briefly, then provide the next action as the required JSON."


class IncompleteResponse(Exception):
    """The provider did not complete the response; do not dispatch partial text."""


class StreamRecorder:
    """Display and retain provider deltas without interpreting them as actions."""

    def __init__(self, condition: str, decision: int, path: Path | None = None):
        self.condition, self.decision, self.path = condition, decision, path
        self.attempt = 1
        self.started = time.monotonic()

    def emit(self, event: str, **fields: Any) -> None:
        record = {"type": "model_stream", "condition": self.condition,
                  "decision": self.decision, "attempt": self.attempt,
                  "elapsed_seconds": round(time.monotonic() - self.started, 3),
                  "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False)
        print(line, flush=True)
        if self.path:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    async def __call__(self, event: Any) -> None:
        if event.type == "retry":
            self.attempt = event.attempt
            self.emit("retry", discards_previous_stream_attempt=True)
        elif event.type == "text":
            self.emit("text_delta", text=event.text)
        elif event.type == "reasoning":
            # OpenAI exposes a summary; this is not raw private reasoning.
            self.emit("reasoning_summary_delta", text=event.reasoning)
        elif event.type == "tool_call":
            self.emit("tool_call_delta", function=event.function,
                      arguments=event.arguments, call_id=event.id)


async def generate_response(state: Any, model: Any, recorder: StreamRecorder) -> str:
    """Use Inspect's streaming model API and retain only its completed output."""
    recorder.emit("generation_started")
    output = await model.generate(state.messages, on_stream=recorder)
    state.output = output
    state.messages.append(output.message)
    recorder.emit("generation_completed", stop_reason=output.stop_reason)
    if output.stop_reason != "stop":
        raise IncompleteResponse(output.stop_reason)
    return output.completion


def load_fixture(directory: Path) -> Any:
    """Load the same scenario used by the Qwen run, without copying it."""
    directory = directory.expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("fixture directory must be a directory")
    for name in FIXTURE_MODULES:
        if not (directory / f"{name}.py").is_file():
            raise FileNotFoundError(f"required fixture missing: {name}.py")
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve().parent != directory:
            raise RuntimeError(f"fixture module already loaded from a different directory: {name}")
    sys.path.insert(0, str(directory))
    try:
        return importlib.import_module(FIXTURE_MODULES[-1])
    finally:
        sys.path.remove(str(directory))


def apply_action(simulator: Any, response: str) -> dict[str, Any]:
    """Pass the model response to the simulator unchanged, without repair."""
    if not isinstance(response, str) or not response.strip():
        raise ValueError("model returned no action text")
    return simulator.take(response)


def outcome_label(result: dict[str, Any]) -> str:
    if result["external_stop"] is not None:
        return "incomplete"
    return "completed" if result["scenario_score"]["task_complete"] else "not_completed"


def conversation_update(prompt: str, feedback_seen: int, *, first: bool) -> tuple[str, int]:
    """Keep exact tool results once in the conversation, with fresh state each turn."""
    payload = json.loads(prompt.removesuffix("\n" + BRIEF))
    feedback = payload["tool_feedback_history"]
    if not isinstance(feedback, list) or len(feedback) < feedback_seen:
        raise ValueError("simulator feedback history changed unexpectedly")
    if not first:
        payload = {"observation": payload["observation"],
                   "tool_feedback_history": feedback[feedback_seen:]}
        prompt = json.dumps(payload, ensure_ascii=False) + "\n" + BRIEF
    return prompt, len(feedback)


async def run_episode(
    simulator: Any,
    request: Callable[[str], Awaitable[str]],
    *,
    max_decisions: int,
    timeout_seconds: float,
    on_action: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Drive a simulator via one model request per decision, with finite bounds."""
    if max_decisions < 1 or timeout_seconds <= 0:
        raise ValueError("decision and time bounds must be positive")
    started = time.monotonic()
    external_stop = None
    while simulator.termination_reason is None:
        if simulator.decisions >= max_decisions:
            external_stop = "decision_limit"
            simulator.finish("technical_error")
            break
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            external_stop = "time_limit"
            simulator.finish("real_deadline")
            break
        try:
            response = await asyncio.wait_for(request(simulator.prompt() + "\n" + BRIEF), remaining)
        except asyncio.TimeoutError:
            external_stop = "time_limit"
            simulator.finish("real_deadline")
            break
        except IncompleteResponse:
            external_stop = "model_response_incomplete"
            simulator.finish("technical_error")
            break
        if not isinstance(response, str) or not response.strip():
            external_stop = "empty_model_response"
            simulator.finish("technical_error")
            break
        event = apply_action(simulator, response)
        if on_action:
            on_action(event)
    return {"external_stop": external_stop, "elapsed_seconds": time.monotonic() - started,
            "scenario_score": simulator.score()}


def make_task(fixture: Any, conditions: list[str], max_decisions: int, timeout_seconds: float,
              stream_path: Path | None = None) -> Any:
    """Construct an Inspect task; real network calls occur only when eval runs."""
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageUser, get_model
    from inspect_ai.scorer import Score, scorer
    from inspect_ai.solver import solver
    from inspect_ai.util import store

    @solver
    def episode_solver():
        async def solve(state, generate):
            simulator = fixture.ReportEpisode(state.metadata["condition"])
            first = True
            feedback_seen = 0

            async def request(prompt: str) -> str:
                nonlocal state, first, feedback_seen
                prompt, feedback_seen = conversation_update(prompt, feedback_seen, first=first)
                if first:
                    state.messages = [ChatMessageUser(content=prompt)]
                    first = False
                else:
                    state.messages.append(ChatMessageUser(content=prompt))
                recorder = StreamRecorder(state.metadata["condition"], simulator.decisions,
                                          stream_path)
                # Model.generate owns streaming; the solver's generate callback
                # only accepts GenerateConfig fields, not an on_stream callback.
                return await generate_response(state, get_model(), recorder)

            def show_action(event: dict[str, Any]) -> None:
                print(json.dumps({"type": "simulator_action", "condition": state.metadata["condition"],
                                  "decision": event.get("decision_index"),
                                  "action": event.get("action_id"),
                                  "result": event.get("result")}, ensure_ascii=False), flush=True)

            result = await run_episode(simulator, request, max_decisions=max_decisions,
                                       timeout_seconds=timeout_seconds, on_action=show_action)
            store().set("hosted_episode_result", result)
            print(json.dumps({"type": "episode_result", "condition": state.metadata["condition"],
                              **result}, ensure_ascii=False), flush=True)
            return state
        return solve

    @scorer(metrics={})
    def episode_score():
        async def score(state, target):
            result = store().get("hosted_episode_result")
            # Incomplete is not represented as an independent scientific failure.
            return Score(value=outcome_label(result), metadata=result)
        return score

    return Task(dataset=[Sample(input="", id=condition, metadata={"condition": condition})
                         for condition in conditions], solver=episode_solver(), scorer=episode_score())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True,
                        help="External directory for Inspect logs; must be outside the repository.")
    parser.add_argument("--condition", choices=("neutral", "interruption", "both"), default="neutral")
    parser.add_argument("--model", default="openai/gpt-6.1-terra",
                        help="Inspect OpenAI model identifier.")
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--max-output-tokens", type=int, default=8192,
                        help="Maximum response tokens per request, including model reasoning.")
    args = parser.parse_args()
    if args.max_decisions < 1 or args.timeout_seconds <= 0 or args.max_output_tokens < 1:
        parser.error("all bounds must be positive")
    log_dir = args.log_dir.expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    if log_dir == repository or repository in log_dir.parents:
        parser.error("log directory must be outside the repository")
    fixture = load_fixture(args.fixture_dir)
    conditions = ["neutral", "interruption"] if args.condition == "both" else [args.condition]
    from dotenv import load_dotenv
    # Existing process environment takes precedence; never print or log secrets.
    load_dotenv(repository / ".env", override=False)
    from inspect_ai import eval as inspect_eval

    log_dir.mkdir(parents=True, exist_ok=True)
    logs = inspect_eval(make_task(fixture, conditions, args.max_decisions, args.timeout_seconds,
                                 stream_path=log_dir / "stream.jsonl"),
                        model=args.model, model_args={"responses_api": True},
                        reasoning_effort="max", reasoning_summary="auto",
                        max_tokens=args.max_output_tokens,
                        log_dir=str(log_dir), max_samples=1, max_connections=1, display="plain")
    return 0 if all(log.status == "success" for log in logs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
