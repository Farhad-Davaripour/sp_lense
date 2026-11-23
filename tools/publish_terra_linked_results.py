"""Publish and verify two Terra linked-continuation conversations.

The public release contains only readable YAML and an index. Original Terra
episodes 07 and 09 stopped when their frozen 131072-token context filled.
These suffixes used a separately frozen 262144-token context amendment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.game_run_store import render_resource_progress
from tools.publish_readable_results import _check_conversation


SOURCE = ROOT / ".local/2026-11-23_09--terra-context-continuation-v1"
SCENARIO = ROOT / "scenarios/terra_context_continuation_v1/2026-11-23_09--linked_context_amendment.yaml"
RUNNER = ROOT / "tools/continue_terra_censored.py"
PROXY = ROOT / "tools/serve_hosted_game_proxy.py"
CATALOG = ROOT / "catalog/results/2026-11-21--infrastructure-preservation.yaml"
ORIGINAL = ROOT / "results/2026-11-20_19Z--infrastructure-preservation-v1/conversations/terra"
TARGET_NAME = "2026-11-23_16Z--terra-linked-context-continuation-v1"
TARGET = ROOT / "results" / TARGET_NAME
BACKTICK = chr(96)


class ProjectionError(RuntimeError):
    """A linked suffix cannot be authenticated or safely published."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProjectionError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path.name}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require(all(isinstance(row, dict) for row in rows), f"Expected JSONL objects: {path.name}")
    return rows


def public_name(finished_utc: str, case: str) -> str:
    parsed = datetime.fromisoformat(finished_utc.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Finish time lacks a timezone")
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ") + f"--terra-case{case}--linked.yaml"


def selected_original(case: str, catalog: list[dict]) -> tuple[dict, Path]:
    rows = [row for row in catalog if row.get("model") == "terra"
            and row.get("scenario_id") == f"infrastructure-48h-{case}"]
    require(len(rows) == 1 and rows[0].get("status") == "right_censored",
            f"Original Terra case {case} is not the selected censored record")
    files = list(ORIGINAL.glob(f"*--infrastructure-48h-{case}--censored.yaml"))
    require(len(files) == 1, f"Original readable Terra case {case} is missing")
    _check_conversation(files[0], files[0].read_bytes())
    return rows[0], files[0]


def expected_readable(requests: list[dict], responses: list[dict],
                      actions: list[dict]) -> list[dict]:
    require(bool(requests) and len(requests) == len(responses) == len(actions),
            "Suffix decision streams differ")
    first = requests[0]["body"]["messages"][0]
    require(first.get("role") == "system" and isinstance(first.get("content"), str),
            "Suffix system message is missing")
    readable: list[dict] = [{"system": first["content"]}]
    progress: list[tuple[str, dict]] = []
    for request, response, event in zip(requests, responses, actions):
        index = request.get("decision_index")
        require(index == response.get("decision_index") == event.get("decision_index"),
                "Suffix decision indices differ")
        require(request.get("decision_id") == response.get("decision_id"),
                "Suffix request and response IDs differ")
        messages = request["body"]["messages"]
        require(messages[0] == first and messages[-1].get("role") == "user"
                and isinstance(messages[-1].get("content"), str),
                "Suffix model-visible history differs")
        require(request["body"].get("model") == "openai/gpt-6.1-terra"
                and response.get("status") == "response_received"
                and response.get("finish_reason") == "stop"
                and isinstance(response.get("assistant_content"), str),
                "Suffix model return is not a complete Terra response")
        transition = event.get("transition")
        require(isinstance(transition, dict)
                and event.get("model_response") == response["assistant_content"]
                and event.get("observation") == transition.get("after"),
                "Suffix action receipt does not bind the model response")
        readable.append({"input": messages[-1]["content"]})
        reasoning = response.get("assistant_reasoning")
        if reasoning:
            require(isinstance(reasoning, str), "Unexpected reasoning summary type")
            readable.append({"reasoning_summary": reasoning})
        readable.extend([
            {"model_output": response["assistant_content"]},
            {"tool_called": {"action": transition.get("action_id"),
                             "arguments": transition.get("arguments")}},
            {"simulator_reply": transition.get("result")},
            {"tool_call_success": transition.get("success")},
            {"simulator_state": event["observation"]},
        ])
        progress.append((transition["action_id"], transition["after"]))
    readable.append({"resource_progress": render_resource_progress(
        actions[0]["transition"]["before"], progress)})
    return readable


def prepare() -> tuple[list[dict], dict]:
    scenario_bytes = SCENARIO.read_bytes()
    scenario = yaml.safe_load(scenario_bytes)
    require(isinstance(scenario, dict) and scenario.get("schema")
            == "research4.terra-linked-context-continuation.v1",
            "Unexpected public continuation scenario")
    freeze_path = SOURCE / "raw/FREEZE_RECEIPT.json"
    freeze = read_json(freeze_path)
    require(freeze.get("status") == "FROZEN_BEFORE_CONTINUATION_MODEL_CALLS"
            and freeze.get("scenario_sha256") == hashlib.sha256(scenario_bytes).hexdigest()
            and freeze.get("runner_sha256") == digest(RUNNER)
            and freeze.get("hosted_proxy_source_sha256") == digest(PROXY)
            and freeze.get("model_route") == "openai/gpt-6.1-terra"
            and freeze.get("original_status_preserved") == "right_censored"
            and freeze.get("amended_context_tokens") == 262144,
            "Continuation freeze receipt or source revision differs")
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))["results"]
    rows: list[dict] = []
    cases = scenario.get("cases")
    require(isinstance(cases, list) and len(cases) == 2, "Expected exactly two Terra cases")
    for entry in cases:
        case = entry["case_id"].removeprefix("infrastructure-48h-")
        require(case in {"07", "09"}, "Unexpected Terra case")
        raw = SOURCE / "raw" / f"case{case}--attempt01"
        result = read_json(raw / "RESULT.json")
        manifest = read_json(raw / "MANIFEST.json")
        plan_info = freeze["plans"][case]
        plan_path = SOURCE / plan_info["path"]
        plan = read_json(plan_path)
        original, parent_yaml = selected_original(case, catalog)
        parent_bundle = entry["original_bundle"]
        parent_raw = ROOT / ".local/2026-11-20_22/2026-11-20_22--terra_neutral/raw" / parent_bundle
        require(digest(parent_raw / "MANIFEST.json") == entry["original_manifest_sha256"]
                == original["manifest_sha256"]
                and digest(parent_raw / "EVENTS.jsonl") == entry["original_events_sha256"]
                == original["event_stream_sha256"]
                and digest(parent_raw / "REQUESTS.jsonl") == entry["original_requests_sha256"]
                and digest(parent_raw / "RESPONSES.jsonl") == entry["original_responses_sha256"]
                and digest(parent_raw / "LIVE_STATE.json") == entry["original_live_state_sha256"],
                f"Original right-censored Terra case {case} changed")
        require(digest(plan_path) == plan_info["sha256"]
                == result.get("plan_sha256") == manifest.get("plan_sha256")
                and plan.get("endpoint", {}).get("context_tokens") == 262144,
                f"Amended Terra case {case} plan differs")
        require(result.get("schema") == "research4.terra-linked-context-continuation-result.v1"
                and result.get("status") == "complete" and result.get("failure") is None
                and result.get("parent_status") == "right_censored"
                and result.get("parent_manifest_sha256") == entry["original_manifest_sha256"]
                and result.get("prefix_decisions") == 36
                and result.get("score", {}).get("outcome_valid") is True
                and result["score"].get("termination_reason") == "horizon_complete"
                and result["score"].get("elapsed_hours") == 48.0,
                f"Terra case {case} lacks a terminal amended outcome")
        require(manifest.get("parent_manifest_sha256") == entry["original_manifest_sha256"]
                and manifest.get("freeze_receipt_sha256") == digest(freeze_path)
                and manifest.get("scenario_sha256") == freeze["scenario_sha256"]
                and manifest.get("continuation_runner_sha256") == freeze["runner_sha256"]
                and manifest.get("endpoint_before_sha256") == digest(raw / "ENDPOINT_BEFORE.json"),
                f"Terra case {case} manifest differs from its freeze")
        health = read_json(raw / "ENDPOINT_BEFORE.json")
        require(health.get("ready") is True and health.get("provider") == "terra"
                and health.get("model_id") == "openai/gpt-6.1-terra"
                and health.get("endpoint_source_sha256") == freeze["hosted_proxy_source_sha256"]
                and health.get("source_sha256") == freeze["hosted_proxy_source_sha256"]
                and plan_info["sha256"] in health.get("game_plan_sha256s", []),
                f"Terra case {case} endpoint identity differs")
        requests = read_jsonl(raw / "REQUESTS.jsonl")
        responses = read_jsonl(raw / "RESPONSES.jsonl")
        events = read_jsonl(raw / "EVENTS.jsonl")
        actions = [event for event in events if event.get("type") == "simulator_action"]
        require(len(actions) == len(requests) == len(responses)
                == result.get("suffix_decisions") == result.get("attempted_model_requests")
                and len(actions) > 0
                and [row.get("decision_index") for row in actions]
                == list(range(37, 37 + len(actions)))
                and events[-1].get("type") == "linked_continuation_result"
                and events[-1].get("score") == result["score"]
                and events[-1].get("status") == "complete",
                f"Terra case {case} action and closeout streams differ")
        clean = SOURCE / "conversation" / f"case{case}--attempt01.yaml"
        content = clean.read_bytes()
        readable = _check_conversation(clean, content)
        require(readable == expected_readable(requests, responses, actions),
                f"Terra case {case} readable YAML differs from exact recorded text or receipts")
        rows.append({
            "case": case, "content": content, "result": result,
            "manifest_sha256": digest(raw / "MANIFEST.json"),
            "parent_yaml": parent_yaml, "parent_yaml_sha256": digest(parent_yaml),
            "plan_sha256": plan_info["sha256"],
            "filename": public_name(result["finished_utc"], case),
            "prefix_hours": entry["saved_simulated_hours"],
        })
    require({row["case"] for row in rows} == {"07", "09"}, "Terra continuation cases differ")
    rows.sort(key=lambda row: row["result"]["finished_utc"])
    return rows, freeze


def readme(rows: list[dict], freeze: dict) -> bytes:
    lines = [
        "# Terra linked context follow-ups", "",
        "These two terminal suffixes follow the original Terra cases 07 and 09. "
        "The original run files stopped at 38.84 and 40.69 simulated hours when their 131K-token context filled. "
        "A separately frozen context allowance of 262144 tokens allowed the saved "
        "simulator states to continue. The suffixes are supplemental observations, "
        "not two extra episodes in the selected 15-per-model infrastructure study "
        "and not uninterrupted outcomes under the original 131072-token plan.", "",
        "The simulator state was reconstructed by replaying and checking all 36 "
        "saved actions per case. The source, neutral instruction, generation seed, "
        "history compaction, tool catalog, and simulator rules remain bound to the "
        "original records. The hosted Terra alias has no immutable weight revision.", "",
        "The public continuation [scenario](../../scenarios/terra_context_continuation_v1/"
        "2026-11-23_09--linked_context_amendment.yaml) and "
        "[runner](../../tools/continue_terra_censored.py) describe the amendment. "
        f"Scenario SHA-256: {BACKTICK}{freeze['scenario_sha256']}{BACKTICK}; "
        f"runner SHA-256: {BACKTICK}{freeze['runner_sha256']}{BACKTICK}; "
        f"freeze-receipt SHA-256: {BACKTICK}{digest(SOURCE / 'raw/FREEZE_RECEIPT.json')}{BACKTICK}.", "",
        "| Case | Original stopped-run conversation | Saved portion | New decisions | "
        "Linked terminal outcome | YAML | YAML SHA-256 | Source manifest SHA-256 | Plan SHA-256 |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        result = row["result"]
        parent = row["parent_yaml"].relative_to(ROOT / "results").as_posix()
        lines.append(
            f"| {row['case']} | [original](../{parent}) | "
            f"{row['prefix_hours']:.2f} h | {result['suffix_decisions']} | "
            "48 h horizon complete under amended context | "
            f"[YAML](conversations/terra/{row['filename']}) | "
            f"{BACKTICK}{hashlib.sha256(row['content']).hexdigest()}{BACKTICK} | "
            f"{BACKTICK}{row['manifest_sha256']}{BACKTICK} | "
            f"{BACKTICK}{row['plan_sha256']}{BACKTICK} |"
        )
    lines.extend(["", "The YAML contains exact recorded model-visible prompts, returned "
                  "output and available reasoning, tool calls, simulator replies, "
                  "success flags, observed state and final resource progress. "
                  "It omits endpoint and usage metadata.", ""])
    return "\n".join(lines).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true",
                        help="Write the validated public YAML and README")
    args = parser.parse_args()
    rows, freeze = prepare()
    files = {f"conversations/terra/{row['filename']}": row["content"] for row in rows}
    files["README.md"] = readme(rows, freeze)
    if args.publish:
        require(not TARGET.exists(), "Public Terra follow-up folder already exists")
        (TARGET / "conversations/terra").mkdir(parents=True, exist_ok=False)
        for relative, content in files.items():
            path = TARGET / relative
            path.write_bytes(content)
            require(path.read_bytes() == content, f"Public copy differs: {relative}")
    print(json.dumps({"status": "passed", "written": args.publish,
                      "linked_terminal_suffixes": len(rows),
                      "public_folder": TARGET_NAME,
                      "cases": [row["case"] for row in rows]}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ProjectionError, OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"ERROR: {error}")
