"""Publish selected Research4 Game conversations without raw run files.

The source evidence stays in the ignored ``.local`` tree.  This command copies
the original 40 catalog-selected readable conversations byte-for-byte and
builds a readable view of the later visible-state extension from exact recorded
model-visible messages, model returns, and simulator receipts.  It does not
run a model or alter source evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.game_run_store import _Dumper, render_resource_progress  # noqa: E402

ORIGINAL_FOLDER = "2026-11-20_19Z--infrastructure-preservation-v1"
EXTENSION_FOLDER = "2026-11-23_01Z--visible-state-extension-v1"
LOCAL_EXTENSION_FOLDER = "2026-11-22_18--visible-state-extension-v1"
SCENARIO_SOURCE = Path("scenarios/infrastructure_preservation_v1/visible_state_extension_v1.yaml")
SCENARIO_SHA256 = "4bfdfe6ad2b8e45d681ea9126343ce2931ea0f0cc2b3e7e79dbdd5ac31ba3761"
EXTENSION_ARMS = (
    "sp_lense_rank16_all64", "qwen38_27b_base", "terra", "deepseek_flash"
)
ORIGINAL_ARMS = {
    "sp_lense_rank16_all64": "sp_lense_rank16_all64",
    "qwen38_27b_base": "qwen38_27b_base",
    "terra": "terra",
    "deepseek_v41_flash": "deepseek_flash",
}
EXPECTED_HOSTED_FIRST_FIVE = {
    "deepseek_flash": (
        "vs01--attempt-02", "vs02--attempt-02", "vs03--attempt-02",
        "vs04--attempt-02", "vs05--attempt-02",
    ),
    "terra": (
        "vs03--attempt-02", "vs04--attempt-02", "vs05--attempt-02",
        "vs06--attempt-02", "vs08--attempt-02",
    ),
}
OLD_KEYS = {
    "system", "input", "reasoning", "provider_reasoning", "reasoning_summary", "model_output",
    "tool_called", "simulator_reply", "tool_call_success",
    "simulator_state", "resource_progress",
}
PRIVATE_PATTERNS = {
    "user-home-path": re.compile(
        r"(?i)(?:[A-Z]:[\\/]Users[\\/]|/Users/|/home/|G:[\\/]My Drive|OneDrive)"
    ),
    "credential-prefix": re.compile(
        r"(?i)(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{20,300}|github_pat_[A-Za-z0-9_]{20,300}|gh[pousr]_[A-Za-z0-9]{20,300}|AIza[A-Za-z0-9_-]{20,300})(?![A-Za-z0-9_-])"
    ),
    "personal-email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "local-endpoint": re.compile(r"(?i)https?://(?:localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)"),
    "assigned-secret": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*['\"]?[A-Za-z0-9_+/-]{20,}"
    ),
}


class ReleaseError(RuntimeError):
    """A selected record or publication contract could not be verified."""


@dataclass(frozen=True)
class Conversation:
    cohort: str
    arm: str
    case: str
    status: str
    finished_utc: str
    source_sha256: str
    source_label: str
    relative_path: Path
    content: bytes


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleaseError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ReleaseError(f"Expected JSONL objects: {path}")
    return rows


def _inside(base: Path, relative: str | Path) -> Path:
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base.resolve()):
        raise ReleaseError(f"Source path escapes the local evidence root: {relative}")
    return candidate


def _timestamp(value: str) -> tuple[datetime, str]:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ReleaseError("A run finish timestamp has no timezone")
    utc = parsed.astimezone(timezone.utc)
    return utc, utc.strftime("%Y-%m-%d_%H%M%SZ")


def _privacy_check(path: Path, content: bytes) -> None:
    text = content.decode("utf-8")
    hits = [label for label, pattern in PRIVATE_PATTERNS.items() if pattern.search(text)]
    if hits:
        # Deliberately do not print the matched text or silently redact it.
        raise ReleaseError(f"Privacy review required for {path}: {', '.join(hits)}")


def _check_conversation(path: Path, content: bytes) -> list[dict]:
    value = yaml.safe_load(content)
    if not isinstance(value, list) or not value:
        raise ReleaseError(f"Readable conversation must be a nonempty YAML list: {path}")
    for item in value:
        if not isinstance(item, dict) or len(item) != 1 or not set(item) <= OLD_KEYS:
            raise ReleaseError(f"Unexpected prompt metadata or structure in {path}")
    if list(value[0]) != ["system"] or list(value[-1]) != ["resource_progress"]:
        raise ReleaseError(f"Missing system or final resource progress in {path}")
    if not any("model_output" in item for item in value):
        raise ReleaseError(f"No recorded model output in {path}")
    _privacy_check(path, content)
    return value


def _dump_conversation(value: list[dict]) -> bytes:
    rendered = yaml.dump(value, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=4096)
    if yaml.safe_load(rendered) != value:
        raise ReleaseError("YAML round-trip altered a recorded value")
    return rendered.encode("utf-8")


def _original(local_root: Path) -> list[Conversation]:
    catalog = yaml.safe_load((ROOT / "catalog/results/2026-11-21--infrastructure-preservation.yaml").read_text(encoding="utf-8"))
    rows = catalog["results"]
    index = yaml.safe_load((local_root / "INFRASTRUCTURE_RUNS.yaml").read_text(encoding="utf-8"))
    if len(rows) != 40 or not isinstance(index, list):
        raise ReleaseError("Expected 40 original catalog results and a local run index")
    by_manifest: dict[str, list[tuple[dict, Path]]] = {}
    for entry in index:
        reference = _inside(local_root, entry["raw"])
        raw_dir = reference.parent
        manifest = raw_dir / "MANIFEST.json"
        if manifest.is_file():
            by_manifest.setdefault(_sha(manifest.read_bytes()), []).append((entry, raw_dir))
    selected = []
    for item in rows:
        matches = by_manifest.get(item["manifest_sha256"], [])
        if len(matches) != 1:
            raise ReleaseError(f"Original catalog manifest does not identify one run: {item['scenario_id']}")
        entry, raw_dir = matches[0]
        if _sha((raw_dir / "EVENTS.jsonl").read_bytes()) != item["event_stream_sha256"]:
            raise ReleaseError(f"Original event-stream digest mismatch: {item['scenario_id']}")
        expected_status = "complete" if item["status"] == "valid_terminal" else "incomplete"
        if item["status"] not in {"valid_terminal", "right_censored"} or entry["status"] != expected_status:
            raise ReleaseError(f"Original run status mismatch: {item['scenario_id']}")
        if _timestamp(item["finished_utc"])[0] != _timestamp(entry["finished_utc"])[0]:
            raise ReleaseError(f"Original finish timestamp mismatch: {item['scenario_id']}")
        source = _inside(local_root, entry["conversation"])
        content = source.read_bytes()
        _check_conversation(source, content)
        arm = ORIGINAL_ARMS[item["model"]]
        _, prefix = _timestamp(item["finished_utc"])
        suffix = "--censored" if item["status"] == "right_censored" else ""
        relative = Path(ORIGINAL_FOLDER) / "conversations" / arm / f"{prefix}--{item['scenario_id']}{suffix}.yaml"
        selected.append(Conversation(
            ORIGINAL_FOLDER, arm, item["scenario_id"], item["status"],
            item["finished_utc"], item["manifest_sha256"], item["id"],
            relative, content,
        ))
    if len({row.relative_path for row in selected}) != 40:
        raise ReleaseError("Duplicate original conversation output paths")
    if Counter(row.arm for row in selected) != Counter({arm: 10 for arm in EXTENSION_ARMS}):
        raise ReleaseError("Original selected arm counts are not ten each")
    return selected


def _extension_selection(local_root: Path, snapshot: datetime) -> list[tuple[str, Path, dict]]:
    raw_root = local_root / LOCAL_EXTENSION_FOLDER / "raw"
    chosen = []
    for arm in EXTENSION_ARMS:
        candidates = []
        for run_dir in (raw_root / arm).iterdir():
            result_path = run_dir / "RESULT.json"
            if not run_dir.is_dir() or not result_path.is_file():
                continue
            result = _read_json(result_path)
            if result.get("status") != "complete":
                continue
            finish, _ = _timestamp(result["finished_utc"])
            if finish <= snapshot:
                candidates.append((finish, run_dir.name, run_dir, result))
        candidates.sort(key=lambda row: (row[0], row[1]))
        picked = candidates[:5]
        if arm in EXPECTED_HOSTED_FIRST_FIVE and tuple(row[1] for row in picked) != EXPECTED_HOSTED_FIRST_FIVE[arm]:
            raise ReleaseError(f"Hosted first-five selection changed unexpectedly for {arm}")
        chosen.extend((arm, row[2], row[3]) for row in picked)
    return chosen


def _extension(local_root: Path, snapshot: datetime) -> list[Conversation]:
    freeze = _read_json(local_root / "2026-11-22_18/visible-state-extension-v1/FREEZE_RECEIPT_v4.json")
    if freeze.get("status") != "FINAL_FROZEN_BEFORE_MODEL_CALLS" or freeze.get("plan_count") != 40:
        raise ReleaseError("The visible-state extension freeze receipt is not passed")
    if freeze.get("scenario_definitions_sha256") != SCENARIO_SHA256 or _sha((ROOT / SCENARIO_SOURCE).read_bytes()) != SCENARIO_SHA256:
        raise ReleaseError("The extension scenario source differs from frozen v4")
    selected = []
    for arm, run_dir, result in _extension_selection(local_root, snapshot):
        manifest = _read_json(run_dir / "MANIFEST.json")
        case = run_dir.name.split("--", 1)[0]
        expected_plan = freeze["plan_sha256"].get(f"plans_v4/{arm}/{case}.json")
        if manifest.get("arm_id") != f"{arm}_{case}" or not expected_plan or manifest.get("plan_sha256") != expected_plan:
            raise ReleaseError(f"Extension arm or frozen plan mismatch: {arm}/{run_dir.name}")
        if len(manifest.get("samples", [])) != 1 or manifest["samples"][0].get("arm_id") != f"{arm}_{case}":
            raise ReleaseError(f"Extension sample identity mismatch: {arm}/{run_dir.name}")
        requests = _read_jsonl(run_dir / "REQUESTS.jsonl")
        responses = _read_jsonl(run_dir / "RESPONSES.jsonl")
        events = [row for row in _read_jsonl(run_dir / "EVENTS.jsonl") if row.get("type") == "simulator_action"]
        if not requests or not (len(requests) == len(responses) == len(events) == result.get("conversation_count")):
            raise ReleaseError(f"Extension decision count mismatch: {arm}/{run_dir.name}")
        system = requests[0]["body"]["messages"][0]
        if system.get("role") != "system" or not isinstance(system.get("content"), str):
            raise ReleaseError(f"Missing exact system prompt: {arm}/{run_dir.name}")
        conversation: list[dict] = [{"system": system["content"]}]
        progress_steps = []
        initial = None
        for index, (request, response, event) in enumerate(zip(requests, responses, events), 1):
            if request.get("decision_index") != index or response.get("decision_index") != index or event.get("decision_index") != index:
                raise ReleaseError(f"Extension decision order mismatch: {arm}/{run_dir.name}")
            if request.get("decision_id") != response.get("decision_id"):
                raise ReleaseError(f"Extension decision ID mismatch: {arm}/{run_dir.name}")
            messages = request["body"]["messages"]
            if messages[0] != system or messages[-1].get("role") != "user" or not isinstance(messages[-1].get("content"), str):
                raise ReleaseError(f"Extension model-visible input mismatch: {arm}/{run_dir.name}")
            if response.get("status") != "response_received" or not isinstance(response.get("assistant_content"), str):
                raise ReleaseError(f"Extension has no recorded model response: {arm}/{run_dir.name}")
            if event.get("model_response") != response["assistant_content"]:
                raise ReleaseError(f"Extension action does not bind exact model output: {arm}/{run_dir.name}")
            transition = event["transition"]
            if not isinstance(transition, dict) or not isinstance(event.get("observation"), dict):
                raise ReleaseError(f"Extension simulator receipt is missing: {arm}/{run_dir.name}")
            if initial is None:
                initial = transition["before"]
            conversation.append({"input": messages[-1]["content"]})
            reasoning = response.get("assistant_reasoning")
            if reasoning is not None:
                if not isinstance(reasoning, str):
                    raise ReleaseError(f"Unexpected reasoning record: {arm}/{run_dir.name}")
                if reasoning:
                    conversation.append({"reasoning_summary" if arm == "terra" else "provider_reasoning": reasoning})
            conversation.append({"model_output": response["assistant_content"]})
            conversation.append({"tool_called": {
                "action": transition.get("action_id"),
                "arguments": transition.get("arguments"),
            }})
            conversation.append({"simulator_reply": transition.get("result")})
            conversation.append({"tool_call_success": transition.get("success")})
            conversation.append({"simulator_state": event["observation"]})
            progress_steps.append((transition.get("action_id"), transition["after"]))
        conversation.append({"resource_progress": render_resource_progress(initial, progress_steps)})
        content = _dump_conversation(conversation)
        _check_conversation(run_dir / "CONVERSATION.yaml", content)
        _, prefix = _timestamp(result["finished_utc"])
        relative = Path(EXTENSION_FOLDER) / "conversations" / arm / f"{prefix}--{run_dir.name}.yaml"
        selected.append(Conversation(
            EXTENSION_FOLDER, arm, case, "complete", result["finished_utc"],
            _sha((run_dir / "MANIFEST.json").read_bytes()),
            f"{arm}/{run_dir.name}", relative, content,
        ))
    if len({row.relative_path for row in selected}) != len(selected):
        raise ReleaseError("Duplicate extension conversation output paths")
    return selected


def _cohort_readme(title: str, rows: list[Conversation], *, in_progress: bool, snapshot: datetime) -> bytes:
    lines = [f"# {title}", ""]
    if in_progress:
        counts = Counter(row.arm for row in rows)
        lines.extend([
            "**IN PROGRESS.** This is a dated snapshot of selected completed extension runs; the Qwen arms have not yet reached five each.",
            f"Snapshot UTC: `{snapshot.isoformat().replace('+00:00', 'Z')}`.", "",
            "Selected complete: " + ", ".join(f"`{arm}` {counts[arm]}/5" for arm in EXTENSION_ARMS) + ".", "",
            "The first five completed runs per arm are selected by recorded finish time after the user narrowed the study scope while outcomes were partly known. Extra completed runs and technical prefixes remain in ignored local evidence. These five-case groups are descriptive; Terra's selected case IDs differ from the other arms, so they are not a four-arm matched comparison.", "",
            f"Frozen scenario source: `{SCENARIO_SOURCE.as_posix()}` (SHA-256 `{SCENARIO_SHA256}`); its Git publication is pending the Game handoff. The simulator source is [`infrastructure_preservation.py`](../../src/infrastructure_preservation.py).",
            "",
        ])
    else:
        lines.extend([
            "These are the 40 catalog-selected episodes used in the current manuscript: 38 valid terminal records and two technically right-censored Terra prefixes. Four excluded technical attempts are not part of this collection.", "",
            "Each file is a byte-for-byte copy of the existing local readable YAML. Original run identity, status, simulator revision and source hashes are recorded in the [research catalog](../../catalog/results/2026-11-21--infrastructure-preservation.yaml).", "",
        ])
    lines.extend([
        "The YAML shows model-visible prompts, recorded model output and available reasoning, actual tool calls and simulator replies, tool success, observed state, and a final recorded resource timeline. Text is not rewritten. Qwen thinking, when present, remains inside its recorded model output. DeepSeek has separately recorded reasoning; Terra's separately recorded field is a provider summary and is labeled accordingly. Some Terra provider summaries are opaque nonprose strings; they are preserved verbatim and should not be interpreted as readable reasoning. A missing separate field is not evidence that no reasoning occurred.", "",
        "| Arm | Case | Recorded outcome | Finished UTC | Conversation | YAML SHA-256 | Source manifest SHA-256 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in sorted(rows, key=lambda r: (r.arm, r.case, r.finished_utc)):
        relative = row.relative_path.relative_to(row.cohort).as_posix()
        lines.append(
            f"| `{row.arm}` | `{row.case}` | `{row.status}` | `{row.finished_utc}` | "
            f"[YAML]({relative}) | `{_sha(row.content)}` | `{row.source_sha256}` |"
        )
    lines.append("")
    return ("\n".join(lines)).encode("utf-8")


def _root_readme(original: list[Conversation], extension: list[Conversation], snapshot: datetime, *, original_only: bool = False) -> bytes:
    counts = Counter(row.arm for row in extension)
    in_progress = any(counts[arm] < 5 for arm in EXTENSION_ARMS)
    lines = [
        "# Research4 selected Game conversations", "",
        "This directory contains only readable YAML conversations used or being collected for the Research4 manuscript. Full raw requests, model streams, endpoint logs, checkpoints and nonselected attempts stay in ignored local storage.", "",
        "Folder prefixes use the UTC hour (`Z`) of the earliest included result. Each YAML filename starts with that run's recorded UTC finish time; neither timestamp claims when a checkpoint was published.", "",
        f"- [Original infrastructure comparison]({ORIGINAL_FOLDER}/README.md): {len(original)} retained episodes, including two right-censored Terra prefixes.",
    ]
    if not original_only:
        lines.append(f"- [Visible-state extension]({EXTENSION_FOLDER}/README.md): {len(extension)} selected completed runs in the `{snapshot.isoformat().replace('+00:00', 'Z')}` snapshot; status: {'IN PROGRESS' if in_progress else 'COMPLETE'}.")
    lines.extend(["",
        "The extension is a separate protocol and its first-five-completed selection was user-directed after partial outcomes. The original ten and extension five per model must not be pooled as exchangeable or described as one four-arm matched cohort.", "",
        "The conversation YAMLs omit endpoint and usage metadata. Their source and output hashes are listed in each cohort README. The reduced machine-verifiable study records are under [`paper/artifacts/records`](../paper/artifacts/records/).", "",
        "Verify the published YAMLs and their recorded SHA-256 values from a clone with `python -B tools/publish_readable_results.py --verify-public`.", "",
    ])
    return "\n".join(lines).encode("utf-8")


def prepare(local_root: Path, snapshot: datetime, *, original_only: bool = False) -> tuple[list[Conversation], list[Conversation], dict[Path, bytes]]:
    original = _original(local_root)
    extension = [] if original_only else _extension(local_root, snapshot)
    if min(_timestamp(row.finished_utc)[0] for row in original).strftime("%Y-%m-%d_%HZ") != ORIGINAL_FOLDER[:14]:
        raise ReleaseError("Original folder timestamp is not the earliest selected finish hour")
    if extension and min(_timestamp(row.finished_utc)[0] for row in extension).strftime("%Y-%m-%d_%HZ") != EXTENSION_FOLDER[:14]:
        raise ReleaseError("Extension folder timestamp is not the earliest selected finish hour")
    in_progress = any(Counter(row.arm for row in extension)[arm] < 5 for arm in EXTENSION_ARMS)
    files = {row.relative_path: row.content for row in original + extension}
    files[Path("README.md")] = _root_readme(original, extension, snapshot, original_only=original_only)
    files[Path(ORIGINAL_FOLDER) / "README.md"] = _cohort_readme(
        "Original infrastructure preservation comparison", original,
        in_progress=False, snapshot=snapshot,
    )
    if not original_only:
        files[Path(EXTENSION_FOLDER) / "README.md"] = _cohort_readme(
            "Visible-state extension", extension,
            in_progress=in_progress, snapshot=snapshot,
        )
    for path, content in files.items():
        _privacy_check(path, content)
    return original, extension, files


def publish(output_root: Path, files: dict[Path, bytes]) -> None:
    target_root = output_root.resolve()
    scoped = {path.parts[0] for path in files if path.suffix == ".yaml"}
    existing = {path.relative_to(target_root) for path in target_root.rglob("*.yaml") if path.relative_to(target_root).parts[0] in scoped} if target_root.exists() else set()
    expected = {path for path in files if path.suffix == ".yaml"}
    unexpected = existing - expected
    if unexpected:
        raise ReleaseError(f"Refusing to overwrite a results tree with unexpected YAMLs: {len(unexpected)}")
    for relative, content in files.items():
        target = _inside(target_root, relative)
        if target.is_file() and target.read_bytes() == content:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    for relative, content in files.items():
        if _inside(target_root, relative).read_bytes() != content:
            raise ReleaseError(f"Published file failed byte verification: {relative}")


def verify_public(output_root: Path) -> dict[str, object]:
    """Verify the committed readable package without access to private raw runs."""
    root = output_root.resolve()
    if not (root / "README.md").is_file():
        raise ReleaseError("The results collection has no README")
    allowed_suffixes = {".yaml", ".md"}
    unexpected = [p for p in root.rglob("*") if p.is_file() and p.suffix not in allowed_suffixes]
    if unexpected:
        raise ReleaseError(f"Non-readable or raw files entered the results collection: {len(unexpected)}")
    row_pattern = re.compile(r"\[YAML\]\(([^)]+\.yaml)\)\s*\|\s*`([0-9a-f]{64})`")
    listed: set[Path] = set()
    counts = {}
    for cohort in (ORIGINAL_FOLDER, EXTENSION_FOLDER):
        readme = root / cohort / "README.md"
        if not readme.is_file():
            if cohort == EXTENSION_FOLDER:
                continue
            raise ReleaseError(f"Missing original cohort README: {cohort}")
        table = row_pattern.findall(readme.read_text(encoding="utf-8"))
        if not table:
            raise ReleaseError(f"The cohort README has no YAML checksum rows: {cohort}")
        arm_counts: Counter[str] = Counter()
        for relative, recorded_sha in table:
            path = _inside(readme.parent, relative)
            if path in listed or not path.is_relative_to(root / cohort):
                raise ReleaseError(f"Duplicate or escaping readable YAML link: {relative}")
            content = path.read_bytes()
            if _sha(content) != recorded_sha:
                raise ReleaseError(f"Published readable YAML hash mismatch: {relative}")
            _check_conversation(path, content)
            listed.add(path)
            arm_counts[path.parent.name] += 1
        counts[cohort] = dict(arm_counts)
        if cohort == ORIGINAL_FOLDER and arm_counts != Counter({arm: 10 for arm in EXTENSION_ARMS}):
            raise ReleaseError("Original public YAML count is not ten per arm")
        if cohort == EXTENSION_FOLDER and any(not 0 <= arm_counts[arm] <= 5 for arm in EXTENSION_ARMS):
            raise ReleaseError("Extension public YAML count exceeds five in an arm")
    actual = {p.resolve() for p in root.rglob("*.yaml")}
    if actual != listed:
        raise ReleaseError(f"Public YAML files differ from the README tables: {len(actual ^ listed)}")
    return {"public_yaml_count": len(listed), "per_cohort": counts, "privacy_check": "passed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=ROOT / ".local")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--publish", action="store_true", help="Write the validated readable results tree")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--original-only", action="store_true", help="Release the original 40 while extension runs continue")
    scope.add_argument("--extension-only", action="store_true", help="Refresh the extension without touching original conversations")
    parser.add_argument("--verify-public", action="store_true", help="Verify the published results without private local evidence")
    args = parser.parse_args()
    if args.verify_public:
        print(json.dumps(verify_public(args.output_root), sort_keys=True))
        return 0
    snapshot = datetime.now(timezone.utc)
    original, extension, files = prepare(args.local_root.resolve(), snapshot, original_only=args.original_only)
    if args.extension_only:
        files = {path: data for path, data in files.items() if path == Path("README.md") or path.parts[0] == EXTENSION_FOLDER}
    summary = {
        "snapshot_utc": snapshot.isoformat().replace("+00:00", "Z"),
        "original": dict(Counter(row.arm for row in original)),
        "extension_selected_complete": dict(Counter(row.arm for row in extension)),
        "public_yaml_count": len(original) + len(extension),
        "privacy_check": "passed",
        "written": bool(args.publish),
    }
    if args.publish:
        publish(args.output_root, files)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReleaseError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
