"""Publish readable local copies of closed rank-16 infrastructure conversations."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import game_run_store as store


SCHEMA = "research4.rank16-winner-infrastructure-game.v1"
SOURCE_FOLDER = "rank16_winner_game_20261120"
OUTPUT_INDEX = "INFRASTRUCTURE_RUNS.yaml"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def publish_exact(path, value):
    content = store._yaml_bytes(value)
    if path.exists():
        if path.read_bytes() != content:
            raise store.StoreError(f"Published evidence differs: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    store._atomic_write(path, content)


def build_conversation(bundle, manifest, result, provider):
    """Join recorded response text and simulator transitions by decision index."""
    responses = {}
    response_path = bundle / "RESPONSES.jsonl"
    for row in read_rows(response_path) if response_path.is_file() else []:
        index = row["decision_index"]
        if index in responses:
            raise store.StoreError("Duplicate response decision")
        responses[index] = row

    messages = []
    steps = []
    initial = None
    system = None
    prompt_indices = set()
    applied = set()
    terminal_count = 0
    case_id = manifest["samples"][0]["case_id"]
    task = manifest["simulator"]["user_task"] + "\n\n"
    for event in read_rows(bundle / "EVENTS.jsonl"):
        if event["bundle"] != bundle.name or event["case_id"] != case_id:
            raise store.StoreError("Event identity differs from the bundle manifest")
        kind = event["type"]
        if kind == "episode_started":
            if initial is not None:
                raise store.StoreError("Duplicate initial simulator observation")
            initial = event["observation"]
        elif kind == "model_prompt":
            index = event["decision_index"]
            if index in prompt_indices:
                raise store.StoreError("Duplicate input decision")
            prompt_indices.add(index)
            if event["system_instruction"] != system:
                system = event["system_instruction"]
                messages.append({"system": system})
            text = event["user_prompt"]
            text = task + store.readable_newlines(text[len(task):]) if text.startswith(task) else text
            messages.append({"input": text})
            response = responses.get(index, {})
            reasoning = response.get("assistant_reasoning")
            if reasoning:
                label = "reasoning_summary" if provider == "terra" else "provider_reasoning"
                messages.append({label: reasoning})
            if response.get("assistant_content") is not None:
                messages.append({"model_output": response["assistant_content"]})
        elif kind == "simulator_action":
            index = event["decision_index"]
            transition = event["transition"]
            response = responses.get(index, {})
            if (index not in prompt_indices or index in applied or not response
                    or response.get("assistant_content") != event["model_response"]
                    or transition["after"] != event["observation"]
                    or transition["result"] != event["result"]
                    or type(transition["success"]) is not bool):
                raise store.StoreError("Simulator action does not match recorded input and response")
            applied.add(index)
            action = transition["action_id"]
            messages.extend([
                {"tool_called": {"action": action, "arguments": transition["arguments"]}},
                {"simulator_reply": transition["result"]},
                {"tool_call_success": transition["success"]},
                {"simulator_state": transition["after"]},
            ])
            steps.append((action, transition["after"]))
        elif kind == "episode_result":
            if (event["status"] != result["status"] or event["score"] != result["score"]
                    or event.get("finished_utc") != result.get("finished_utc")):
                raise store.StoreError("Terminal event differs from the saved result")
            terminal_count += 1
        elif kind not in ("model_stream", "client_error", "model_request_rejected"):
            raise store.StoreError(f"Unsupported infrastructure event: {kind}")

    if initial is None or not responses.keys() <= prompt_indices or terminal_count != int("score" in result):
        raise store.StoreError("Incomplete binding between saved result, inputs, and simulator events")
    return messages + [{"resource_progress": store.render_resource_progress(initial, steps)}]


def publish_run(local, bundle):
    manifest_path = bundle / "MANIFEST.json"
    if (not manifest_path.is_file() or manifest_path.is_symlink()
            or bundle.resolve().parent != bundle.parent.resolve() or bundle.is_symlink()):
        return None
    manifest = read_json(manifest_path)
    if manifest.get("schema") != SCHEMA:
        return None
    terminal = bundle / "RESULT.json"
    if not terminal.is_file():
        terminal = bundle / "CLOSEOUT.json"
    if not (terminal.is_file() and (bundle / "EVENTS.jsonl").is_file()):
        return None
    result = read_json(terminal)
    status = result["status"]
    if status not in ("complete", "incomplete") or len(manifest["samples"]) != 1:
        raise store.StoreError("Expected one closed infrastructure case")
    if manifest["version"] != bundle.name:
        raise store.StoreError("Bundle name differs from its manifest")
    model_id = manifest["samples"][0]["model"]
    arm = re.sub(r"_seed_[0-9]+$", "", manifest["samples"][0]["arm_id"])
    if re.fullmatch(r"sp-\d{8}-\d{2}-[0-9a-f]+", model_id):
        arm = model_id
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", part) for part in (arm, bundle.name)):
        raise store.StoreError("Invalid Game folder name")
    finished_utc = result.get("finished_utc")
    if status == "complete" and not finished_utc:
        raise store.StoreError("A completed episode needs a recorded finish time")
    timestamp = "incomplete-unknown"
    if finished_utc:
        finish = datetime.fromisoformat(store._utc(finished_utc))
        timestamp = finish.astimezone(store._edmonton()).strftime("%Y-%m-%d_%H")
    destination = local / timestamp / (timestamp + "--" + arm)
    raw_ref = destination / "raw" / bundle.name / "REFERENCE.yaml"
    suffix = "--incomplete" if status == "incomplete" else ""
    conversation = destination / "conversation" / (bundle.name + suffix + ".yaml")
    if any(not path.resolve().is_relative_to(local) for path in (raw_ref, conversation)):
        raise store.StoreError("Publication path escapes the local Research4 store")
    source_files = sorted(path for path in bundle.iterdir() if path.is_file() and not path.is_symlink())
    if any(path.is_symlink() for path in bundle.iterdir()):
        raise store.StoreError("Raw evidence must not contain links")
    hashes = {path.name: store._sha(path) for path in source_files}
    health_path = bundle / "ENDPOINT_BEFORE.json"
    provider = read_json(health_path).get("provider") if health_path.is_file() else None
    messages = build_conversation(bundle, manifest, result, provider)
    if (read_json(manifest_path) != manifest or read_json(terminal) != result
            or any(store._sha(path) != hashes[path.name] for path in source_files)):
        raise store.StoreError("Raw evidence changed while building the readable transcript")
    raw_ref.parent.mkdir(parents=True, exist_ok=True)
    for source in source_files:
        target = raw_ref.parent / source.name
        if target.is_symlink() or target.resolve().parent != raw_ref.parent.resolve():
            raise store.StoreError("Raw-copy destination escaped the bundle folder")
        if not target.exists():
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != hashes[source.name]:
                raise store.StoreError("Raw evidence changed during copy")
            store._atomic_write(target, data)
        if store._sha(target) != hashes[source.name]:
            raise store.StoreError("Published raw evidence differs")
    publish_exact(raw_ref, {"directory": bundle.relative_to(local).as_posix(), "sha256": hashes})
    publish_exact(conversation, messages)
    return {"started_utc": manifest["started_utc"], "finished_utc": finished_utc, "status": status,
            "case_id": manifest["samples"][0]["case_id"], "model_id": model_id,
            "plan_sha256": manifest["plan_sha256"], "source_sha256": manifest["source_sha256"],
            "raw": raw_ref.relative_to(local).as_posix(),
            "conversation": conversation.relative_to(local).as_posix()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    local = store.DEFAULT_STORE.resolve(strict=True)
    source = local / SOURCE_FOLDER
    if source.resolve(strict=True).parent != local:
        raise store.StoreError("Bundle source must be inside the local Research4 store")
    with store._index_lock(local):
        rows = [row for folder in sorted(source.iterdir()) if folder.is_dir()
                if (row := publish_run(local, folder)) is not None]
        rows.sort(key=lambda row: (row["finished_utc"] is None,
                                   store._utc(row["finished_utc"] or row["started_utc"])))
        store._atomic_write(local / OUTPUT_INDEX, store._yaml_bytes(rows))
    print(f"Published {len(rows)} local infrastructure conversations.")


if __name__ == "__main__":
    main()
