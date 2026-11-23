"""Prepare and run linked suffixes for the two censored Terra infrastructure cases.

The original prefixes remain immutable and right-censored. This script replays
their saved simulator actions, verifies every recorded transition, and resumes
only after a separately frozen context-window amendment. Its preflight and
freeze modes do not contact a model or an endpoint.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
import urllib.request
import uuid

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = ROOT / "scenarios/terra_context_continuation_v1/2026-11-23_09--linked_context_amendment.yaml"
DEFAULT_ARCHIVE = ROOT.parent / "superseded/sp_lense4_worktrees/company-world-v4"
DEFAULT_TOKENIZER_ROOT = ROOT.parent / "superseded/sp_lense4/research4"
DEFAULT_PARENT_RAW = ROOT / ".local/2026-11-20_22/2026-11-20_22--terra_neutral/raw"
PROXY_SOURCE = ROOT / "tools/serve_hosted_game_proxy.py"
DEFAULT_PROXY_PYTHON = DEFAULT_ARCHIVE / ".local/arena-env/Scripts/python.exe"
TOKEN_ENV = "R4_TERRA_CONTINUATION_TOKEN"
CONTINUATION_CONTEXT_TOKENS = 262144
SOURCE_CLOSURE_PATHS = (
    "tools/run_rank16_infrastructure_game.py",
    "tools/rank16_prompt_tokens.py",
    "src/__init__.py",
    "src/infrastructure_preservation.py",
    "src/company_world.py",
    "src/company_world_v2.py",
    "src/company_economy.py",
    "src/company_economy_v2.py",
    "src/company_access_cases.py",
    "src/infrastructure_game.py",
    "src/security_game.py",
    "src/continuity_game.py",
    "src/security_challenges.py",
    "config/company_economy_v1.json",
    "config/company_economy_v2.json",
    "config/infrastructure_economy_v1.json",
    "scenarios/company_world_v1/world.yaml",
    "scenarios/company_world_v1/access_cases.yaml",
    "scenarios/company_world_v2/world.yaml",
)
PARENT_FILES = (
    "CONVERSATION.yaml",
    "ENDPOINT_BEFORE.json",
    "EVENTS.jsonl",
    "LIVE_STATE.json",
    "MANIFEST.json",
    "REQUESTS.jsonl",
    "RESPONSES.jsonl",
    "RESULT.json",
)


class ContinuationError(RuntimeError):
    """The continuation source or result is not tied to its recorded parent."""


@dataclass
class PreparedCase:
    case: dict
    parent_dir: Path
    parent_manifest: dict
    original_plan: dict
    episode: object
    messages: list[dict]
    next_prompt_tokens: int
    next_messages_sha256: str
    source_file_sha256s: dict[str, str]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContinuationError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise ContinuationError(f"Expected JSONL objects: {path}")
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_yaml(path: Path, value: object, dumper: type[yaml.Dumper]) -> None:
    data = yaml.dump(value, Dumper=dumper, allow_unicode=True, sort_keys=False, width=4096)
    if yaml.safe_load(data) != value:
        raise ContinuationError("Readable YAML did not round-trip")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def scenario(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict)
            or value.get("schema") != "research4.terra-linked-context-continuation.v1"
            or value.get("model_route") != "openai/gpt-6.1-terra"
            or value.get("continuation_method", {}).get("previous_context_tokens") != 131072
            or value.get("continuation_method", {}).get("amended_context_tokens") != CONTINUATION_CONTEXT_TOKENS
            or len(value.get("cases", [])) != 2
            or {case.get("case_id") for case in value["cases"]} != {"infrastructure-48h-07", "infrastructure-48h-09"}):
        raise ContinuationError("Expected exactly the two original Terra context continuations")
    return value


def source_closure(archive: Path, expected: str) -> dict[str, str]:
    source = {relative: file_sha(archive / relative) for relative in SOURCE_CLOSURE_PATHS}
    lines = [f"{relative}\t{source[relative]}" for relative in sorted(source)]
    digest = sha(("\n".join(lines) + "\n").encode("utf-8"))
    if digest != expected:
        raise ContinuationError("Archived replay source/config closure differs")
    return source


def load_archived(archive: Path, tokenizer_root: Path, source: dict[str, str], spec: dict):
    if (source["tools/run_rank16_infrastructure_game.py"] != spec["original_runner_sha256"]
            or source["tools/rank16_prompt_tokens.py"] != spec["prompt_token_counter_sha256"]
            or source["src/infrastructure_preservation.py"] != spec["infrastructure_simulator_sha256"]):
        raise ContinuationError("Archived runner, token counter or simulator hash differs")
    tokenizer_manifest = tokenizer_root / ".local/reference-tokenizer/REFERENCE_MANIFEST.json"
    if file_sha(tokenizer_manifest) != spec["reference_tokenizer_manifest_sha256"]:
        raise ContinuationError("Reference tokenizer manifest differs")
    interpreter = tokenizer_root / ".local/reference-tokenizer-venv/Scripts/python.exe"
    if not interpreter.is_file():
        raise ContinuationError("Archived reference tokenizer interpreter is unavailable")

    sys.path.insert(0, str(archive))
    from src import infrastructure_preservation as simulator  # type: ignore[import-not-found]
    if Path(simulator.__file__).resolve() != (archive / "src/infrastructure_preservation.py").resolve():
        raise ContinuationError("Simulator imported from the wrong checkout")
    spec_file = importlib.util.spec_from_file_location("_terra_archived_runner", archive / "tools/run_rank16_infrastructure_game.py")
    if spec_file is None or spec_file.loader is None:
        raise ContinuationError("Archived runner cannot be loaded")
    runner = importlib.util.module_from_spec(spec_file)
    spec_file.loader.exec_module(runner)
    runner.TOKENIZER_PY = interpreter
    runner.TOKENIZER = tokenizer_root / ".local/reference-tokenizer"

    reader_file = ROOT / "src/game_run_store.py"
    reader_spec = importlib.util.spec_from_file_location("_terra_continuation_reader", reader_file)
    if reader_spec is None or reader_spec.loader is None:
        raise ContinuationError("Readable YAML renderer cannot be loaded")
    reader = importlib.util.module_from_spec(reader_spec)
    reader_spec.loader.exec_module(reader)
    return simulator, runner, reader, file_sha(reader_file)


def verify_parent(case: dict, parent_raw: Path, archive: Path, simulator, runner, source: dict[str, str]) -> PreparedCase:
    parent_dir = parent_raw / case["original_bundle"]
    if not parent_dir.resolve().is_relative_to(parent_raw.resolve()):
        raise ContinuationError("Original bundle path escaped its saved collection")
    reference = yaml.safe_load((parent_dir / "REFERENCE.yaml").read_text(encoding="utf-8"))
    if (not isinstance(reference, dict)
            or reference.get("directory") != f"rank16_winner_game_20261120/{case['original_bundle']}"
            or set(reference.get("sha256", {})) != set(PARENT_FILES)):
        raise ContinuationError("Original source reference is incomplete")
    for filename, expected in reference["sha256"].items():
        if file_sha(parent_dir / filename) != expected:
            raise ContinuationError(f"Original parent evidence changed: {filename}")
    for filename, field in (
        ("MANIFEST.json", "original_manifest_sha256"),
        ("EVENTS.jsonl", "original_events_sha256"),
        ("REQUESTS.jsonl", "original_requests_sha256"),
        ("RESPONSES.jsonl", "original_responses_sha256"),
        ("LIVE_STATE.json", "original_live_state_sha256"),
        ("RESULT.json", "original_result_sha256"),
    ):
        if reference["sha256"][filename] != case[field]:
            raise ContinuationError(f"Parent {filename} is not bound to the public scenario")

    manifest = read_json(parent_dir / "MANIFEST.json")
    original_path = archive / f"config/terra_fresh9_20261120/seed_{case['simulator_seed']}.json"
    if file_sha(original_path) != case["original_plan_sha256"] or manifest.get("plan_sha256") != case["original_plan_sha256"]:
        raise ContinuationError("Original frozen case plan differs")
    original_plan = read_json(original_path)
    if any(manifest.get(key) != value for key, value in original_plan.items()):
        raise ContinuationError("Original manifest is not the original frozen plan")
    if (manifest.get("model_id") != "openai/gpt-6.1-terra"
            or manifest["simulator"]["seed"] != case["simulator_seed"]
            or manifest["source_sha256"]["src/infrastructure_preservation.py"] != source["src/infrastructure_preservation.py"]):
        raise ContinuationError("Original case identity or simulator source differs")
    result = read_json(parent_dir / "RESULT.json")
    if result.get("status") != "incomplete" or result.get("failure") != "worker_deadline_or_context_exhausted":
        raise ContinuationError("Original right-censored status differs")
    requests = read_jsonl(parent_dir / "REQUESTS.jsonl")
    responses = read_jsonl(parent_dir / "RESPONSES.jsonl")
    events = read_jsonl(parent_dir / "EVENTS.jsonl")
    actions = [row for row in events if row.get("type") == "simulator_action"]
    if (len(actions) != case["saved_decisions"]
            or len(requests) != len(actions) or len(responses) != len(actions)
            or result.get("conversation_count") != len(actions)):
        raise ContinuationError("Original prefix decision count differs")
    episode = simulator.InfrastructurePreservationEpisode(
        seed=case["simulator_seed"], horizon=manifest["simulator"]["horizon_hours"],
        max_decisions=100000, background_compute=manifest["simulator"]["background_compute"],
    )
    for index, row in enumerate(actions, 1):
        transition = row["transition"]
        action = {"action": transition["action_id"], **transition["arguments"]} if transition.get("action_id") else {}
        actual = episode.take(action)
        if actual != transition or row.get("observation") != actual["after"] or row.get("decision_index") != index:
            raise ContinuationError(f"Archived simulator replay differs at decision {index}")
    live = read_json(parent_dir / "LIVE_STATE.json")
    if episode.score() != live.get("score") or episode.termination_reason is not None:
        raise ContinuationError("Replayed prefix score differs or episode already ended")
    if abs(episode.elapsed_hours - case["saved_simulated_hours"]) > 1e-8:
        raise ContinuationError("Replayed prefix elapsed time differs")
    last_request, last_response = requests[-1], responses[-1]
    if (last_response.get("status") != "response_received"
            or last_response.get("finish_reason") != "stop"
            or not isinstance(last_response.get("assistant_content"), str)
            or runner.count_tokens(last_request["body"]["messages"], manifest["generation"]["chat_template_kwargs"]) != last_request["prompt_tokens"]):
        raise ContinuationError("Last original model exchange is not verified")
    messages = [*last_request["body"]["messages"],
                {"role": "assistant", "content": last_response["assistant_content"]}]
    next_user = manifest["simulator"]["user_task"] + "\n\n" + episode.prompt() + "\n\n" + runner.RESPONSE_INSTRUCTION
    candidate = [*messages, {"role": "user", "content": next_user}]
    prompt_tokens = runner.count_tokens(candidate, manifest["generation"]["chat_template_kwargs"])
    if prompt_tokens >= manifest["history"]["compact_at_prompt_tokens"]:
        compacted = runner.compact_history(messages, episode, manifest["history"]["retain_recent_exchanges"])
        candidate = [*compacted, {"role": "user", "content": next_user}]
        prompt_tokens = runner.count_tokens(candidate, manifest["generation"]["chat_template_kwargs"])
    digest = sha(runner.canonical_json(candidate).encode("utf-8"))
    if (prompt_tokens != case["next_prompt_after_original_compaction_tokens"]
            or digest != case["next_model_visible_messages_sha256"]
            or prompt_tokens <= manifest["endpoint"]["context_tokens"]
            or prompt_tokens + manifest["endpoint"]["max_output_tokens"] > CONTINUATION_CONTEXT_TOKENS):
        raise ContinuationError("Reconstructed next model-visible request differs from the frozen amendment")
    return PreparedCase(case, parent_dir, manifest, original_plan, episode, messages,
                        prompt_tokens, digest, reference["sha256"])


def prepare_all(args):
    scenario_path = args.scenario.resolve(strict=True)
    cohort = scenario(scenario_path)
    archive = args.archive_root.resolve(strict=True)
    tokenizer_root = args.tokenizer_root.resolve(strict=True)
    parent_raw = args.parent_raw.resolve(strict=True)
    if cohort["archived_source"].get("replay_source_closure_file_count") != len(SOURCE_CLOSURE_PATHS):
        raise ContinuationError("Archived source closure file count differs")
    closure = source_closure(archive, cohort["archived_source"]["replay_source_closure_sha256"])
    simulator, runner, reader, reader_sha = load_archived(archive, tokenizer_root, closure, cohort["archived_source"])
    prepared = [verify_parent(case, parent_raw, archive, simulator, runner, closure) for case in cohort["cases"]]
    return cohort, prepared, runner, reader, closure, reader_sha, file_sha(scenario_path)


def frozen_plan(prepared: PreparedCase) -> dict:
    value = copy.deepcopy(prepared.original_plan)
    if (value["endpoint"]["context_tokens"] != 131072
            or value["endpoint"]["max_output_tokens"] != 8192
            or value["history"]["compact_at_prompt_tokens"] != 120000
            or value["history"]["retain_recent_exchanges"] != 5
            or value["generation"]["seed"] != 20261108):
        raise ContinuationError("Original plan has an unexpected context or generation policy")
    value["endpoint"]["context_tokens"] = CONTINUATION_CONTEXT_TOKENS
    value["endpoint"]["token_env"] = TOKEN_ENV
    return value


def validate_output_root(path: Path) -> Path:
    output = path.resolve()
    local = (ROOT / ".local").resolve()
    if (not output.is_relative_to(local)
            or not re.fullmatch(r"\d{4}-\d\d-\d\d_\d\d--terra-context-continuation-v1", output.name)
            or "onedrive" in str(output).casefold()):
        raise ContinuationError("Output root must be a timestamped Research4 .local continuation folder")
    return output


def freeze(args, prepared: list[PreparedCase], scenario_sha: str, closure: dict[str, str],
           reader_sha: str) -> dict:
    output = validate_output_root(args.output_root)
    if output.exists():
        raise ContinuationError("Continuation output root already exists; do not overwrite an attempt")
    if not args.proxy_python.is_file():
        raise ContinuationError("Pinned hosted proxy interpreter is unavailable")
    runtime = subprocess.run(
        [str(args.proxy_python), "-c",
         "import sys,importlib.metadata as m,json;print(json.dumps({'python':sys.version.split()[0],'inspect_ai':m.version('inspect-ai'),'python_dotenv':m.version('python-dotenv')}))"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    runtime_versions = json.loads(runtime.stdout.splitlines()[-1])
    output.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "conversation", "raw/plans"):
        (output / name).mkdir(exist_ok=False)
    token = secrets.token_urlsafe(48)
    token_path = output / "raw/LOOPBACK_TOKEN.secret"
    descriptor = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as secret_file:
        secret_file.write(token.encode("ascii"))
        secret_file.flush()
        os.fsync(secret_file.fileno())
    os.chmod(token_path, 0o600)
    plans = {}
    for row in prepared:
        key = row.case["case_id"].removeprefix("infrastructure-48h-")
        path = output / "raw/plans" / f"case{key}.json"
        write_json(path, frozen_plan(row))
        plans[key] = {"path": path.relative_to(output).as_posix(), "sha256": file_sha(path),
                      "parent_manifest_sha256": row.case["original_manifest_sha256"],
                      "parent_events_sha256": row.case["original_events_sha256"],
                      "next_messages_sha256": row.next_messages_sha256,
                      "next_prompt_tokens": row.next_prompt_tokens}
    receipt = {
        "schema": "research4.terra-linked-context-continuation-freeze.v1",
        "status": "FROZEN_BEFORE_CONTINUATION_MODEL_CALLS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_sha256": scenario_sha,
        "runner_sha256": file_sha(Path(__file__)),
        "readable_yaml_renderer_sha256": reader_sha,
        "hosted_proxy_source_sha256": file_sha(PROXY_SOURCE),
        "hosted_proxy_interpreter_sha256": file_sha(args.proxy_python),
        "hosted_proxy_runtime": runtime_versions,
        "archived_source_closure_sha256": sha(("\n".join(
            f"{relative}\t{closure[relative]}" for relative in sorted(closure)) + "\n").encode("utf-8")),
        "loopback_token_sha256": sha(token.encode("utf-8")),
        "loopback_token_file": "raw/LOOPBACK_TOKEN.secret",
        "loopback_token_env_name": TOKEN_ENV,
        "model_route": "openai/gpt-6.1-terra",
        "original_status_preserved": "right_censored",
        "amended_context_tokens": CONTINUATION_CONTEXT_TOKENS,
        "plans": plans,
    }
    write_json(output / "raw/FREEZE_RECEIPT.json", receipt)
    (output / "README.md").write_text(
        "# Terra linked context continuations\n\n"
        "These are follow-up suffixes for the original right-censored Terra cases 07 and 09. "
        "The original prefixes remain unchanged. The new context allowance is a separate "
        "protocol revision, so terminal suffixes do not retroactively make the original runs complete.\n\n"
        "The frozen plans, receipt and exact raw evidence are in raw/. Readable suffix conversations "
        "are in conversation/. The public scenario is "
        "scenarios/terra_context_continuation_v1/2026-11-23_09--linked_context_amendment.yaml.\n",
        encoding="utf-8",
    )
    return receipt


def verify_frozen(args, prepared: PreparedCase, scenario_sha: str, reader_sha: str) -> tuple[Path, dict, str, str, str]:
    output = validate_output_root(args.output_root)
    receipt_path = output / "raw/FREEZE_RECEIPT.json"
    receipt = read_json(receipt_path)
    if (receipt.get("schema") != "research4.terra-linked-context-continuation-freeze.v1"
            or receipt.get("status") != "FROZEN_BEFORE_CONTINUATION_MODEL_CALLS"
            or receipt.get("scenario_sha256") != scenario_sha
            or receipt.get("runner_sha256") != file_sha(Path(__file__))
            or receipt.get("readable_yaml_renderer_sha256") != reader_sha
            or receipt.get("hosted_proxy_source_sha256") != file_sha(PROXY_SOURCE)
            or receipt.get("hosted_proxy_interpreter_sha256") != file_sha(args.proxy_python)
            or receipt.get("model_route") != "openai/gpt-6.1-terra"
            or receipt.get("original_status_preserved") != "right_censored"
            or receipt.get("amended_context_tokens") != CONTINUATION_CONTEXT_TOKENS):
        raise ContinuationError("Frozen continuation receipt or source differs")
    if (receipt.get("loopback_token_file") != "raw/LOOPBACK_TOKEN.secret"
            or receipt.get("loopback_token_env_name") != TOKEN_ENV):
        raise ContinuationError("Private token locator differs from the frozen receipt")
    token_path = (output / receipt["loopback_token_file"]).resolve(strict=True)
    if not token_path.is_relative_to((output / "raw").resolve()):
        raise ContinuationError("Private token file escaped the ignored raw collection")
    token = token_path.read_text(encoding="ascii")
    if len(token) < 24 or sha(token.encode("utf-8")) != receipt.get("loopback_token_sha256"):
        raise ContinuationError("Private loopback token does not match the frozen receipt")
    runtime = subprocess.run(
        [str(args.proxy_python), "-c",
         "import sys,importlib.metadata as m,json;print(json.dumps({'python':sys.version.split()[0],'inspect_ai':m.version('inspect-ai'),'python_dotenv':m.version('python-dotenv')}))"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    if json.loads(runtime.stdout.splitlines()[-1]) != receipt.get("hosted_proxy_runtime"):
        raise ContinuationError("Hosted proxy package versions differ from the frozen receipt")
    case_key = args.case
    entry = receipt.get("plans", {}).get(case_key)
    if not isinstance(entry, dict):
        raise ContinuationError("Frozen continuation case is absent")
    plan_path = (output / entry["path"]).resolve(strict=True)
    if not plan_path.is_relative_to((output / "raw/plans").resolve()):
        raise ContinuationError("Frozen plan path escaped the continuation")
    plan_sha = file_sha(plan_path)
    plan = read_json(plan_path)
    if (plan_sha != entry.get("sha256")
            or plan != frozen_plan(prepared)
            or entry.get("parent_manifest_sha256") != prepared.case["original_manifest_sha256"]
            or entry.get("parent_events_sha256") != prepared.case["original_events_sha256"]
            or entry.get("next_messages_sha256") != prepared.next_messages_sha256
            or entry.get("next_prompt_tokens") != prepared.next_prompt_tokens):
        raise ContinuationError("Frozen plan or original-prefix binding differs")
    return output, plan, plan_sha, token, file_sha(receipt_path)


def endpoint_health(plan: dict, plan_sha: str) -> dict:
    url = plan["endpoint"]["base_url"] + plan["endpoint"]["health_path"]
    with urllib.request.urlopen(url, timeout=10) as response:
        health = json.load(response)
    expected_source = file_sha(PROXY_SOURCE)
    if (not isinstance(health, dict) or health.get("ready") is not True
            or health.get("model_id") != plan["model_id"]
            or health.get("adapter_sha256") != plan.get("adapter_sha256")
            or health.get("provider") != "terra"
            or health.get("endpoint_source_sha256") != expected_source
            or health.get("source_sha256") != expected_source
            or plan_sha not in health.get("game_plan_sha256s", ())
            or health.get("max_output_tokens", 0) < plan["endpoint"]["max_output_tokens"]):
        raise ContinuationError("Hosted proxy health does not bind the exact continuation plans")
    return health


def readable(path: Path, rows: list[dict], initial: dict, steps: list[tuple[str | None, dict]], reader) -> None:
    view = [*rows, {"resource_progress": reader.render_resource_progress(initial, steps)}]
    write_yaml(path, view, reader._Dumper)


def execute(args, prepared: PreparedCase, runner, reader, scenario_sha: str, reader_sha: str) -> None:
    output, plan, plan_sha, token, receipt_sha = verify_frozen(args, prepared, scenario_sha, reader_sha)
    health = endpoint_health(plan, plan_sha)
    case_key = args.case
    run_name = f"case{case_key}--attempt{args.attempt:02d}"
    raw = output / "raw" / run_name
    conversation_path = output / "conversation" / (run_name + ".yaml")
    if raw.exists() or conversation_path.exists():
        raise ContinuationError("Continuation attempt already exists; do not overwrite it")
    raw.mkdir(parents=False, exist_ok=False)
    episode = prepared.episode
    messages = prepared.messages
    initial_state = copy.deepcopy(episode.events[-1]["after"])
    suffix: list[dict] = []
    view: list[dict] = [{"system": runner.SYSTEM}]
    progress_steps: list[tuple[str | None, dict]] = []
    started_utc = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema": "research4.terra-linked-context-continuation-attempt.v1",
        "version": run_name,
        "case_id": prepared.case["case_id"],
        "model_id": plan["model_id"],
        "parent_status": "right_censored",
        "parent_bundle": prepared.case["original_bundle"],
        "parent_manifest_sha256": prepared.case["original_manifest_sha256"],
        "parent_events_sha256": prepared.case["original_events_sha256"],
        "parent_decisions": prepared.case["saved_decisions"],
        "parent_simulated_hours": prepared.case["saved_simulated_hours"],
        "next_model_visible_messages_sha256": prepared.next_messages_sha256,
        "plan_sha256": plan_sha,
        "freeze_receipt_sha256": receipt_sha,
        "scenario_sha256": scenario_sha,
        "continuation_runner_sha256": file_sha(Path(__file__)),
        "archived_runner_sha256": prepared.parent_manifest["source_sha256"]["tools/run_rank16_infrastructure_game.py"],
        "endpoint_before_sha256": None,
        "started_utc": started_utc,
    }
    write_json(raw / "ENDPOINT_BEFORE.json", health)
    manifest["endpoint_before_sha256"] = file_sha(raw / "ENDPOINT_BEFORE.json")
    write_json(raw / "MANIFEST.json", manifest)
    append_jsonl(raw / "EVENTS.jsonl", {
        "type": "linked_continuation_started", "case_id": prepared.case["case_id"],
        "parent_manifest_sha256": prepared.case["original_manifest_sha256"],
        "parent_decisions": prepared.case["saved_decisions"],
        "observation": episode.observation(),
    })
    readable(conversation_path, view, initial_state, progress_steps, reader)
    deadline = time.monotonic() + plan["operations"]["worker_deadline_seconds"]
    failure = None
    terminal_error = None
    model_requests = 0

    try:
        while episode.termination_reason is None and episode.elapsed_hours < plan["simulator"]["horizon_hours"]:
            current = plan["simulator"]["user_task"] + "\n\n" + episode.prompt() + "\n\n" + runner.RESPONSE_INSTRUCTION
            candidate = [*messages, {"role": "user", "content": current}]
            prompt_tokens = runner.count_tokens(candidate, plan["generation"]["chat_template_kwargs"])
            if prompt_tokens >= plan["history"]["compact_at_prompt_tokens"]:
                messages = runner.compact_history(messages, episode, plan["history"]["retain_recent_exchanges"])
                candidate = [*messages, {"role": "user", "content": current}]
                prompt_tokens = runner.count_tokens(candidate, plan["generation"]["chat_template_kwargs"])
            maximum = min(plan["endpoint"]["context_tokens"] - prompt_tokens,
                          plan["endpoint"]["max_output_tokens"])
            if maximum <= 0:
                failure = "amended_context_exhausted"
                break
            if time.monotonic() >= deadline:
                failure = "worker_deadline"
                break
            if model_requests == 0:
                digest = sha(runner.canonical_json(candidate).encode("utf-8"))
                if prompt_tokens != prepared.next_prompt_tokens or digest != prepared.next_messages_sha256:
                    raise ContinuationError("First continuation prompt differs from the pre-dispatch freeze")
            decision = episode.decisions + 1
            decision_id = sha(f"{plan_sha}:{decision}:{uuid.uuid4().hex}".encode("utf-8"))[:32]
            body = {
                "model": plan["model_id"], "messages": candidate, "stream": False,
                "max_tokens": maximum,
                **{key: plan["generation"][key] for key in
                   ("seed", "temperature", "top_p", "top_k", "min_p", "repetition_penalty")},
                "chat_template_kwargs": plan["generation"]["chat_template_kwargs"],
            }
            append_jsonl(raw / "REQUESTS.jsonl", {
                "decision_index": decision, "decision_id": decision_id,
                "prompt_tokens": prompt_tokens, "body": body,
            })
            append_jsonl(raw / "EVENTS.jsonl", {
                "type": "model_prompt", "case_id": prepared.case["case_id"],
                "decision_index": decision, "system_instruction": runner.SYSTEM,
                "user_prompt": current, "prompt_tokens": prompt_tokens,
            })
            append_jsonl(raw / "EVENTS.jsonl", {
                "type": "model_stream", "event": "generation_started", "decision_index": decision,
            })
            model_requests += 1
            request_started = time.monotonic()
            try:
                response = runner.request(
                    plan["endpoint"]["base_url"] + plan["endpoint"]["chat_path"], token,
                    plan_sha, decision_id, body, plan["endpoint"]["request_timeout_seconds"],
                )
            except Exception as error:
                failure = "endpoint_request_error"
                append_jsonl(raw / "RESPONSES.jsonl", {
                    "decision_index": decision, "decision_id": decision_id,
                    "status": "request_error", "error_type": type(error).__name__,
                    "error": str(error)[:1000],
                })
                break
            exchange = {
                "decision_index": decision, "decision_id": decision_id,
                "prompt_tokens": prompt_tokens, "maximum_response_tokens": maximum,
                "assistant_content": response["content"],
                "assistant_reasoning": response["reasoning"], "usage": response["usage"],
                "finish_reason": response["finish_reason"],
                "elapsed_real_seconds": time.monotonic() - request_started,
                "parsed_action": None, "simulator_transition": None,
                "status": "response_received",
            }
            suffix.append(exchange)
            append_jsonl(raw / "RESPONSES.jsonl", exchange)
            view.append({"input": candidate[-1]["content"]})
            if response["reasoning"]:
                view.append({"reasoning_summary": response["reasoning"]})
                append_jsonl(raw / "EVENTS.jsonl", {
                    "type": "model_stream", "event": "reasoning_delta", "decision_index": decision,
                    "text": response["reasoning"],
                })
            if response["content"]:
                view.append({"model_output": response["content"]})
                append_jsonl(raw / "EVENTS.jsonl", {
                    "type": "model_stream", "event": "text_delta", "decision_index": decision,
                    "text": response["content"],
                })
            readable(conversation_path, view, initial_state, progress_steps, reader)
            if response["finish_reason"] != "stop":
                failure = "model_response_incomplete"
                break
            try:
                action = runner.extract_action(response["content"])
            except ValueError:
                failure = "malformed_model_action"
                break
            transition = episode.take(action)
            exchange.update(parsed_action=action, simulator_transition=transition, status="dispatched")
            append_jsonl(raw / "EVENTS.jsonl", {
                "type": "simulator_action", "case_id": prepared.case["case_id"],
                "decision_index": decision, "model_response": response["content"],
                "result": transition["result"], "transition": transition,
                "observation": transition["after"],
            })
            view.extend([
                {"tool_called": {"action": transition.get("action_id"),
                                 "arguments": transition.get("arguments")}},
                {"simulator_reply": transition.get("result")},
                {"tool_call_success": transition.get("success")},
                {"simulator_state": transition["after"]},
            ])
            progress_steps.append((transition.get("action_id"), transition["after"]))
            messages = [*candidate, {"role": "assistant", "content": response["content"]}]
            runner.write_yaml(raw / "CONVERSATION.yaml", suffix)
            runner.write_json(raw / "LIVE_STATE.json", {
                "status": "running", "parent_manifest_sha256": prepared.case["original_manifest_sha256"],
                "score": episode.score(), "suffix_conversation": suffix,
            })
            readable(conversation_path, view, initial_state, progress_steps, reader)
    except BaseException as error:
        terminal_error = error
        failure = failure or "client_or_simulator_error"
        append_jsonl(raw / "EVENTS.jsonl", {
            "type": "client_error", "case_id": prepared.case["case_id"],
            "error_type": type(error).__name__, "error": str(error)[:1000],
        })

    if not failure and (episode.score().get("outcome_valid") is not True
                        or (episode.termination_reason is None
                            and episode.elapsed_hours < plan["simulator"]["horizon_hours"])):
        failure = "missing_natural_terminal_receipt"
    if failure and episode.termination_reason is None:
        episode.finish("real_deadline" if failure == "worker_deadline" else "technical_error")
    result = {
        "schema": "research4.terra-linked-context-continuation-result.v1",
        "status": "complete" if failure is None else "incomplete",
        "failure": failure, "case_id": prepared.case["case_id"],
        "parent_status": "right_censored",
        "parent_manifest_sha256": prepared.case["original_manifest_sha256"],
        "plan_sha256": plan_sha, "score": episode.score(),
        "prefix_decisions": prepared.case["saved_decisions"],
        "suffix_decisions": len(progress_steps),
        "attempted_model_requests": model_requests,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(raw / "EVENTS.jsonl", {
        "type": "linked_continuation_result", "case_id": prepared.case["case_id"],
        "scenario_score": result["score"], "external_stop": failure, **result,
    })
    write_json(raw / "RESULT.json", result)
    runner.write_yaml(raw / "CONVERSATION.yaml", suffix)
    readable(conversation_path, view, initial_state, progress_steps, reader)
    print(json.dumps({"status": result["status"], "failure": failure,
                      "case_id": result["case_id"], "suffix_decisions": result["suffix_decisions"],
                      "total_decisions": result["prefix_decisions"] + result["suffix_decisions"],
                      "elapsed_simulated_hours": result["score"].get("elapsed_hours"),
                      "output": str(raw)}, sort_keys=True))
    if terminal_error is not None:
        raise terminal_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Verify archived prefixes and next prompts without endpoint calls or writes")
    mode.add_argument("--freeze", action="store_true", help="Create immutable continuation plans and receipt, without endpoint calls")
    mode.add_argument("--execute", action="store_true", help="Execute one frozen suffix through its bound hosted endpoint")
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--tokenizer-root", type=Path, default=DEFAULT_TOKENIZER_ROOT)
    parser.add_argument("--parent-raw", type=Path, default=DEFAULT_PARENT_RAW)
    parser.add_argument("--proxy-python", type=Path, default=DEFAULT_PROXY_PYTHON)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--case", choices=("07", "09"))
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if (args.freeze or args.execute) and args.output_root is None:
        parser.error("--output-root is required for --freeze or --execute")
    cohort, prepared, runner, reader, closure, reader_sha, scenario_sha = prepare_all(args)
    if args.preflight:
        print(json.dumps({"status": "preflight_pass", "model_execution": False,
                          "scenario_sha256": scenario_sha,
                          "cases": {row.case["case_id"]: {"parent_decisions": row.episode.decisions,
                                  "saved_hours": row.episode.elapsed_hours,
                                  "next_prompt_tokens": row.next_prompt_tokens,
                                  "next_messages_sha256": row.next_messages_sha256}
                                    for row in prepared}}, sort_keys=True))
        return 0
    if args.freeze:
        receipt = freeze(args, prepared, scenario_sha, closure, reader_sha)
        print(json.dumps({"status": receipt["status"], "model_execution": False,
                          "output_root": str(validate_output_root(args.output_root)),
                          "plan_sha256s": {key: value["sha256"] for key, value in receipt["plans"].items()}},
                         sort_keys=True))
        return 0
    if args.case is None or args.attempt < 1:
        parser.error("--execute requires --case and a positive --attempt")
    selected = next(row for row in prepared if row.case["case_id"].endswith(args.case))
    execute(args, selected, runner, reader, scenario_sha, reader_sha)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContinuationError, OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"ERROR: {error}")
