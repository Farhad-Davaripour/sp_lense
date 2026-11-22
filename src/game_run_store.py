"""Publish closed Game episodes as readable conversations and preserved evidence.

This is an offline, post-run operation: it never runs a model or modifies the
source batch. A publication failure must be handled separately from the Game's
scientific result. Writers share ``.index.lock`` with the historical importer.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.resources
import json
import os
from pathlib import Path
import re
import shutil
import uuid
from zoneinfo import ZoneInfo

import yaml


FORMAT = "research4.game-index.v1"
DEFAULT_STORE = Path(__file__).resolve().parents[1] / ".local"
METRICS = (
    ("Elapsed hours", ("elapsed_hours",)), ("Cash (USD)", ("cash_usd",)),
    ("Credits", ("credits",)), ("Reserve (USD)", ("reserve_usd", "reserve")),
    ("Electricity (kWh)", ("electricity_kwh", "energy")),
    ("GPUs", ("gpu_count", "compute_capacity_units")),
    ("Active replicas", ("agent_replica_count",)), ("Bitcoin (BTC)", ("btc_balance",)),
    ("Security level", ("security_level",)), ("Integrity", ("integrity",)),
)
GLYPHS = "▁▂▃▄▅▆▇█"


class StoreError(RuntimeError):
    """Invalid, conflicting, locked, or unverifiable evidence."""


class _Dumper(yaml.SafeDumper):
    def analyze_scalar(self, value):
        result = super().analyze_scalar(value)
        # PyYAML otherwise quotes multiline text that has trailing spaces,
        # making real newlines appear as literal "\\n" in text editors. YAML
        # block scalars preserve those spaces and remain lossless. Only permit
        # this override for characters representable literally in YAML.
        if "\n" in value and all(
            c in "\n\t" or 0x20 <= ord(c) <= 0xD7FF
            or 0xE000 <= ord(c) <= 0xFFFD or 0x10000 <= ord(c) <= 0x10FFFF
            for c in value
        ):
            result.allow_block = True
        return result


def _represent_string(dumper, value):
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", value,
        style="|" if "\n" in value or len(value) > 160 else None,
    )


_Dumper.add_representer(str, _represent_string)


def readable_newlines(value):
    """Render newline escapes in a valid JSON input prefix for display only.

    An even run of backslashes denotes a literal backslash and is left intact.
    Free text, code and suffixes are not decoded. Exact provider and harness
    text remains in the copied EVENTS/STREAM files, which remain authoritative.
    """
    if isinstance(value, str):
        try:
            obj, end = json.JSONDecoder().raw_decode(value)
            if not isinstance(obj, (dict, list)):
                return value
        except (ValueError, TypeError):
            return value

        def replace(match):
            backslashes = match.group(1)
            return backslashes[:-1] + "\n" if len(backslashes) % 2 else match.group(0)
        return re.sub(r"(\\+)n", replace, value[:end]) + value[end:]
    if isinstance(value, list):
        return [readable_newlines(item) for item in value]
    if isinstance(value, dict):
        return {key: readable_newlines(item) for key, item in value.items()}
    return value


def _yaml_bytes(value):
    text = yaml.dump(value, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=4096)
    if yaml.safe_load(text) != value:
        raise StoreError("YAML round-trip failed")
    return text.encode("utf-8")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def render_resource_progress(initial, steps):
    """Display recorded states and decimal deltas, without interpolating values."""
    states = [initial if isinstance(initial, dict) else {}] + [
        state if isinstance(state, dict) else {} for _, state in steps
    ]
    labels = ["0=initial"] + [
        str(i) + "=" + str(tool if tool is not None else "unknown tool")
        for i, (tool, _) in enumerate(steps, 1)
    ]
    lines = [
        "Recorded simulator resource progress",
        "Each position represents an actual recorded state; ? means not recorded.",
        "Each bar is scaled independently to that resource's observed minimum and maximum.",
        "Steps: " + " | ".join(labels), "",
    ]
    selected = []
    for label, keys in METRICS:
        if label == "Credits" and any(_number(s.get("cash_usd")) is not None for s in states):
            continue
        key = next((k for k in keys if any(_number(s.get(k)) is not None for s in states)), None)
        if key is None:
            continue
        values = [_number(s.get(key)) for s in states]
        known = [v for v in values if v is not None]
        low, high = min(known), max(known)

        def glyph(value):
            if value is None:
                return "?"
            if low == high:
                return GLYPHS[0]
            index = int((value - low) * Decimal(len(GLYPHS) - 1) / (high - low))
            return GLYPHS[max(0, min(len(GLYPHS) - 1, index))]

        def delta(a, b):
            if a is None or b is None:
                return "?"
            difference = b - a
            return ("+" if difference > 0 else "") + str(difference)

        lines.extend([
            f"{label} ({key}): " + "".join(glyph(v) for v in values),
            "  Values:  " + " → ".join("?" if v is None else str(v) for v in values),
            "  Changes: " + " → ".join(["start"] + [
                delta(values[i - 1], values[i]) for i in range(1, len(values))
            ]), "",
        ])
        selected.append((label, key, values[-1]))
    lines.append("FINAL RECORDED STATE — step " + str(len(steps)))
    if not selected:
        lines.append("No numeric resource values were recorded.")
    else:
        lines.extend(f"  {label} ({key}): {'?' if value is None else value}"
                     for label, key, value in selected)
    return "\n".join(lines)


def _filesystem_path(path):
    filename = str(Path(path).resolve())
    if os.name == "nt" and not filename.startswith("\\\\?\\"):
        return ("\\\\?\\UNC\\" + filename[2:] if filename.startswith("\\\\")
                else "\\\\?\\" + filename)
    return filename


def _sha(path):
    digest = hashlib.sha256()
    with open(_filesystem_path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    if path.read_bytes() != data:
        raise StoreError("Atomic write readback mismatch: " + str(path))


@contextmanager
def _index_lock(root):
    path = root / ".index.lock"
    nonce = uuid.uuid4().hex
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise StoreError("Index lock exists; stale or foreign locks are not removed") from exc
    original = os.fstat(fd)

    def owned():
        try:
            stat = path.stat()
            current = json.loads(path.read_text(encoding="utf-8"))
            return (current.get("nonce") == nonce and stat.st_ino == original.st_ino
                    and stat.st_dev == original.st_dev)
        except (OSError, ValueError):
            return False

    def require_owned():
        if not owned():
            raise StoreError("Index lock ownership changed")

    try:
        try:
            import psutil
            created = psutil.Process().create_time()
        except (ImportError, OSError):
            created = None
        owner = {"schema": "research4.game-index-lock.v1", "pid": os.getpid(),
                 "process_created_at": created, "nonce": nonce,
                 "created_utc": datetime.now(timezone.utc).isoformat()}
        os.write(fd, json.dumps(owner).encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        fd = None
        yield require_owned
        require_owned()
    finally:
        if fd is not None:
            os.close(fd)
        if owned():
            path.unlink()


def _edmonton():
    # Use the packaged IANA database, not an older operating-system database.
    try:
        import tzdata
        if tzdata.IANA_VERSION < "2026c":
            raise StoreError("Packaged IANA timezone database is too old for Alberta")
        resource = importlib.resources.files("tzdata.zoneinfo").joinpath("America", "Edmonton")
        with resource.open("rb") as handle:
            zone = ZoneInfo.from_file(handle, key="America/Edmonton")
        probe = datetime(2026, 11, 15, 12, tzinfo=timezone.utc).astimezone(zone)
        if probe.utcoffset().total_seconds() != -21600:
            raise StoreError("Packaged Edmonton timezone does not reflect Alberta's 2026 change")
        return zone
    except (ImportError, OSError, AttributeError) as exc:
        raise StoreError("Current packaged tzdata is required") from exc


def _utc(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone missing")
        return parsed.astimezone(timezone.utc).isoformat()
    except (AttributeError, TypeError, ValueError) as exc:
        raise StoreError("Invalid or timezone-naive completion time") from exc


def _inspect_records(batch):
    """Extract actual per-sample completion times from the recorded Inspect logs."""
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:
        raise StoreError("inspect-ai is required to read .eval evidence") from exc
    records = {}
    for path in sorted((batch / "inspect").glob("*.eval")):
        # File handle avoids an Inspect path-length limitation on Windows.
        resolved = path.resolve()
        if not resolved.is_relative_to(batch):
            raise StoreError("Inspect file escaped source batch")
        with open(_filesystem_path(resolved), "rb") as handle:
            log = read_eval_log(handle, format="eval")
        for sample in log.samples or []:
            records.setdefault(sample.id, []).append({
                "path": path, "log_status": log.status,
                "completed_at": sample.completed_at, "error": bool(sample.error),
            })
    return records


def _episode_id(event_sha, case_id):
    # Existing 204 entries use this scheme. The event digest binds the actual
    # model, condition, input, actions and outputs; other identity fields are
    # compared independently before admitting an idempotent redelivery.
    return hashlib.sha256(("research4-game-episode-v1\0" + event_sha + "\0" + case_id).encode()).hexdigest()


def _safe(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-").lower()[:48] or "model"


def _assert_no_manifest_secrets(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"api_key", "access_token", "authorization", "password", "secret"} and child:
                raise StoreError("Manifest contains a credential-like field")
            _assert_no_manifest_secrets(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_manifest_secrets(child)


def _event_digest_update(digest, event):
    # A submitted batch may be labeled with its actual batch name in EVENTS,
    # while the provider stream retains the manifest's pre-submission label.
    # Bundle labels are validated separately; every other field must match.
    value = dict(event)
    value.pop("bundle", None)
    digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8") + b"\n")


def _verify_stream(path, case, expected, count, allowed_bundles, event_bundles):
    digest = hashlib.sha256()
    actual = 0
    bundles = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("case_id") == case:
                bundle = event.get("bundle") if "bundle" in event else None
                if "bundle" in event:
                    if not isinstance(bundle, str) or bundle not in allowed_bundles:
                        raise StoreError("Unexpected model stream bundle for " + case)
                    bundles.add(bundle)
                if actual >= len(event_bundles):
                    raise StoreError("Model stream has additional records for " + case)
                recorded_bundle = event_bundles[actual]
                if bundle != recorded_bundle and (bundle is None or recorded_bundle is None
                                                   or {bundle, recorded_bundle} != allowed_bundles):
                    raise StoreError("Model stream bundle alias mismatch for " + case)
                _event_digest_update(digest, event)
                actual += 1
    if actual != count or digest.hexdigest() != expected:
        raise StoreError("Model stream does not match saved event stream for " + case)
    return sorted(bundles)


def _parse_events(path, samples_by_case, batch_name):
    episodes = {case: {"messages": [], "initial": None, "steps": [], "pending": {},
                       "counts": {}, "started": 0, "results": 0,
                       "stream_digest": hashlib.sha256(), "output_mismatches": [],
                       "event_bundles": set(), "stream_bundle_sequence": []} for case in samples_by_case}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            event = json.loads(line)
            case = event.get("case_id")
            if case not in episodes:
                raise StoreError(f"Unknown case in events at line {line_number}")
            sample = samples_by_case[case]
            episode = episodes[case]
            for key in ("model_id", "model", "arm_id", "seed", "prompt_condition", "rules_version", "system_prompt_sha256"):
                if key in event and event[key] != sample.get(key):
                    raise StoreError(f"Event {key} differs from manifest at line {line_number}")
            if "bundle" in event:
                allowed = {batch_name, sample.get("bundle")}
                if not isinstance(event["bundle"], str) or event["bundle"] not in allowed:
                    raise StoreError(f"Event bundle differs from manifest at line {line_number}")
                episode["event_bundles"].add(event["bundle"])
            kind = event.get("type")
            episode["counts"][kind] = episode["counts"].get(kind, 0) + 1
            if kind == "episode_started":
                episode["started"] += 1
                episode["initial"] = event.get("observation")
            elif kind == "episode_result":
                episode["results"] += 1
            elif kind == "compact_input_source":
                # The model-visible prompt is separately recorded as
                # model_prompt. Preserve this auxiliary source record in raw
                # EVENTS without presenting it as another conversation turn.
                for key in ("case_id", "model_id", "model", "arm_id"):
                    if key not in event or event[key] != sample.get(key):
                        raise StoreError("Compact input source identity differs from manifest")
            elif kind == "model_prompt":
                instruction = event.get("system_instruction")
                if "system_prompt" in sample and instruction != sample["system_prompt"]:
                    raise StoreError("Recorded system instruction differs from manifest")
                if "last_system" not in episode or episode["last_system"] != instruction:
                    episode["messages"].append({"system": instruction})
                    episode["last_system"] = instruction
                if "user_prompt" not in event:
                    raise StoreError("Recorded prompt missing user_prompt")
                episode["messages"].append({"input": event["user_prompt"]})
            elif kind == "model_stream":
                _event_digest_update(episode["stream_digest"], event)
                episode["stream_bundle_sequence"].append(event.get("bundle") if "bundle" in event else None)
                identity = (event.get("decision"), event.get("attempt"))
                event_type = event.get("event")
                pending = episode["pending"]
                if event_type == "generation_started":
                    if identity in pending:
                        raise StoreError("Duplicate generation start")
                    pending[identity] = {"reasoning": {}, "output": "", "closed": False}
                elif identity not in pending:
                    raise StoreError("Generation delta without start")
                elif event_type in ("reasoning_delta", "reasoning_summary_delta"):
                    item = pending[identity]
                    label = event.get("reasoning_kind") or (
                        "summary" if event_type == "reasoning_summary_delta" else "raw")
                    item["reasoning"][label] = item["reasoning"].get(label, "") + event.get("text", "")
                elif event_type == "text_delta":
                    pending[identity]["output"] += event.get("text", "")
                elif event_type == "generation_completed":
                    item = pending[identity]
                    if item["closed"]:
                        raise StoreError("Duplicate generation completion")
                    for label, text in item["reasoning"].items():
                        key = "reasoning_summary" if label == "summary" else (
                            "reasoning" if label == "raw" else "reasoning_" + str(label))
                        episode["messages"].append({key: text})
                    episode["messages"].append({"model_output": item["output"]})
                    item["closed"] = True
                else:
                    raise StoreError("Unsupported recorded stream event: " + str(event_type))
            elif kind == "simulator_action":
                candidates = [(key, value) for key, value in episode["pending"].items()
                              if key[0] == event.get("decision") and value["closed"]]
                received = event.get("model_response")
                if received is not None and (not candidates or candidates[-1][1]["output"] != received):
                    episode["messages"].append({"model_response_received": received})
                    episode["output_mismatches"].append({"decision": event.get("decision")})
                episode["messages"].extend([
                    {"tool_called": event.get("action")}, {"simulator_reply": event.get("result")},
                    {"tool_call_success": event.get("success")}, {"simulator_state": event.get("observation")},
                ])
                episode["steps"].append((event.get("action"), event.get("observation")))
            else:
                raise StoreError("Unsupported recorded event type: " + str(kind))
    for episode in episodes.values():
        for item in episode["pending"].values():
            if not item["closed"]:
                for label, text in item["reasoning"].items():
                    key = "reasoning_summary" if label == "summary" else (
                        "reasoning" if label == "raw" else "reasoning_" + str(label))
                    episode["messages"].append({key: text})
                if item["output"]:
                    episode["messages"].append({"partial_model_output": item["output"]})
        episode["messages"] = [readable_newlines(message) for message in episode["messages"]]
        episode["messages"].append({"resource_progress": render_resource_progress(
            episode["initial"], episode["steps"])})
    return episodes


def _copy_verified(source, target):
    digest = _sha(source)
    os.makedirs(_filesystem_path(target.parent), exist_ok=True)
    shutil.copyfile(_filesystem_path(source), _filesystem_path(target))
    if _sha(target) != digest or _sha(source) != digest:
        raise StoreError("Source changed or evidence copy failed: " + str(source))
    return {"source": str(source.resolve()), "copy": str(target.resolve()),
            "sha256": digest, "bytes": os.stat(_filesystem_path(source)).st_size}


def _collect_supporting_files(batch, staged_raw, final_raw, manifest):
    """Copy only recorded, hash-matching source and immutable run context."""
    declared = manifest.get("source_sha256") or {}
    if not isinstance(declared, dict):
        raise StoreError("Source digest manifest is not a mapping")
    repository = Path(__file__).resolve().parents[1]
    roots = (batch.parent / "RUN_SOURCE", batch / "RUN_SOURCE",
             batch.parent / "SOURCE_SNAPSHOT", repository)
    report = {"format": "research4.game-supporting-files.v1", "source_files": [],
              "unavailable_sources": [], "excluded_sources": [], "run_context_files": []}
    for relative, expected in sorted(declared.items()):
        path = Path(relative)
        if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
                or path.is_absolute() or ".." in path.parts
                or path.suffix.lower() not in (".py", ".json", ".yaml", ".yml")
                or any(word in str(path).lower() for word in ("sealed", "heldout", "held_out"))):
            report["excluded_sources"].append({"path": relative, "expected_sha256": expected})
            continue
        match = None
        for candidate_root in roots:
            candidate = candidate_root / path
            if (candidate.is_file() and candidate.resolve().is_relative_to(candidate_root.resolve())
                    and _sha(candidate) == expected):
                match = candidate
                break
        if match is None:
            report["unavailable_sources"].append({"path": relative, "expected_sha256": expected})
            continue
        record = _copy_verified(match, staged_raw / "source-snapshot" / path)
        if record["sha256"] != expected:
            raise StoreError("Source changed after identity check: " + relative)
        record["path"] = relative
        record["copy"] = str((final_raw / "source-snapshot" / path).resolve())
        report["source_files"].append(record)

    context_names = {"PLAN.json", "SOURCE_MANIFEST.json", "EVALUATION_CONFIG_v1.yaml",
                     "ORIGINAL_SELECTION.json", "ORIGINAL_WORLD_SCENARIOS.yaml", "SCENARIOS.yaml",
                     "INITIAL_SLOTS.json", "CANDIDATE_COHORT_INITIAL_SLOTS.json",
                     "INITIAL_MESSAGES_" + batch.name + ".json"}
    for source in sorted(batch.parent.iterdir()):
        if not source.is_file() or source.name not in context_names:
            continue
        if not source.resolve().is_relative_to(batch.parent.resolve()):
            raise StoreError("Run context escaped its source directory")
        data = source.read_bytes()
        parsed = json.loads(data) if source.suffix.lower() == ".json" else yaml.safe_load(data)
        _assert_no_manifest_secrets(parsed)
        record = _copy_verified(source, staged_raw / "run-context" / source.name)
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise StoreError("Run context changed after validation")
        record["copy"] = str((final_raw / "run-context" / source.name).resolve())
        report["run_context_files"].append(record)
    return report


def _canonical_root(store_root=None):
    root = Path(store_root or DEFAULT_STORE).resolve()
    marker = root / "RELOCATED_TO.yaml"
    if not marker.exists():
        return root
    data = yaml.safe_load(marker.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("format") != "research4.game-root-relocation.v1"
            or root.name != "game-runs" or Path(data.get("canonical_root", "")).resolve() != root.parent
            or Path(data.get("legacy_root", "")).resolve() != root):
        raise StoreError("Invalid legacy-root relocation marker")
    if not (root.parent / "INDEX.yaml").is_file():
        raise StoreError("Relocated canonical index is missing")
    return root.parent


def list_runs(store_root=None):
    """Return the canonical index entries; no legacy paths are mutated."""
    index_path = _canonical_root(store_root) / "INDEX.yaml"
    if not index_path.exists():
        return []
    document = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise StoreError("Unrecognized index format")
    return list(document.get("entries", []))


def load_conversation(store_root, episode_id):
    """Load an indexed conversation only after its path and hash are verified."""
    root = _canonical_root(store_root)
    matches = [e for e in list_runs(root) if e.get("episode_id") == episode_id]
    if len(matches) != 1:
        raise StoreError("Episode ID missing or ambiguous")
    path = Path(matches[0]["conversation"]).resolve()
    if not path.is_relative_to(root) or _sha(path) != matches[0]["conversation_sha256"]:
        raise StoreError("Conversation path or hash failed verification")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _readme(index, root):
    def label(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["# Game runs", "", "The date and hour in folder names is the recorded episode finish in America/Edmonton. Incomplete and unknown episodes are grouped separately.",
             "", "Each model folder contains raw/ and conversation/. Conversations contain recorded messages, tool calls and simulator states, followed by a resource progress timeline. INDEX.yaml contains identities, sources and hashes. Original evidence remains at its original location.",
             "", "See [HOW_TO_READ.md](HOW_TO_READ.md) for help interpreting tool calls and resource bars.",
             "", "| Finished (Edmonton) | Model | Setting / case | Status | Conversation |", "|---|---|---|---|---|"]
    for parent in root.parents:
        instructions = parent / "wiki/game-runs/README.md"
        if instructions.is_file():
            relative = os.path.relpath(instructions, root).replace("\\", "/")
            lines[2:2] = ["[Game run storage and reading instructions](" + relative + ")", ""]
            break
    for entry in index["entries"]:
        path = Path(entry["conversation"]).resolve()
        if not path.is_relative_to(root):
            raise StoreError("Index contains a conversation outside the store")
        model = str(entry.get("model") or "unknown").split("/")[-1]
        adapter = entry.get("adapter_sha256")
        if adapter:
            model += " / " + adapter[:8]
        values = [entry.get("finish_edmonton") or "Unknown", model,
                  str(entry.get("arm_id") or "unknown") + " / " + str(entry.get("case_id") or "").split("/")[-1],
                  entry.get("completion_status") or "unknown"]
        lines.append("| " + " | ".join(label(x) for x in values) + " | [YAML](" + path.relative_to(root).as_posix() + ") |")
    return ("\n".join(lines) + "\n").encode("utf-8")


def publish_batch(batch_dir, store_root=None, *, checkpoint_metadata=None):
    """Publish a closed batch, returning published entries and unknown cases.

    `checkpoint_metadata`, when supplied, must include a matching `model_id`
    and a full `adapter_sha256`; the digest must agree with the batch manifest.
    The function never opens an endpoint, runs a model, or changes the batch.
    """
    batch, root = Path(batch_dir).resolve(), _canonical_root(store_root)
    if any("sealed" in p.lower() or "heldout" in p.lower() or "held_out" in p.lower() for p in batch.parts):
        raise StoreError("Sealed or held-out inputs are outside this publisher")
    if root.is_relative_to(batch):
        raise StoreError("Store root cannot be inside the source batch")
    manifest_path, events_path, closeout_path = batch / "MANIFEST.json", batch / "EVENTS.jsonl", batch / "CLOSEOUT.json"
    if not manifest_path.is_file() or not events_path.is_file() or not closeout_path.is_file():
        return {"entries": [], "published": 0, "unknown": ["Batch lacks manifest, events or closeout"], "conflicts": []}
    manifest_bytes, closeout_bytes = manifest_path.read_bytes(), closeout_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    closeout = json.loads(closeout_bytes)
    _assert_no_manifest_secrets(manifest)
    samples = manifest.get("samples", [])
    cases = [sample.get("case_id") for sample in samples]
    if not cases or any(not isinstance(c, str) or not c for c in cases) or len(set(cases)) != len(cases):
        raise StoreError("Manifest cases missing or duplicate")
    settings = manifest.get("generation_settings", {})
    if checkpoint_metadata is not None:
        model = checkpoint_metadata.get("model_id")
        digest = checkpoint_metadata.get("adapter_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise StoreError("Checkpoint metadata requires a full adapter SHA-256")
        matching = [s for s in samples if model in (s.get("model_id"), s.get("model"),
                    settings.get(s.get("model_id"), {}).get("service_model_id"))]
        if not matching or any(settings.get(s["model_id"], {}).get("adapter_sha256") != digest for s in matching):
            raise StoreError("Checkpoint metadata does not match manifest identity")
    close_results = closeout.get("results") or ([closeout["result"]] if closeout.get("result") else [])
    closed = {entry.get("case_id"): entry for entry in close_results}
    if len(closed) != len(close_results):
        raise StoreError("Closeout contains duplicate cases")
    inspections = _inspect_records(batch)
    event_sha = _sha(events_path)
    episodes = _parse_events(events_path, {sample["case_id"]: sample for sample in samples}, batch.name)
    if _sha(events_path) != event_sha:
        raise StoreError("Event source changed while reading")
    zone = _edmonton()
    root.mkdir(parents=True, exist_ok=True)
    result = {"entries": [], "published": 0, "unknown": [], "conflicts": []}
    with _index_lock(root) as require_lock:
        index_path = root / "INDEX.yaml"
        prior = index_path.read_bytes() if index_path.exists() else None
        index = yaml.safe_load(prior) if prior else {"format": FORMAT, "description": "Game runs", "counts": {}, "entries": [], "warnings": []}
        if not isinstance(index, dict) or index.get("format") != FORMAT:
            raise StoreError("Unrecognized index format")
        if prior:
            snapshot = root / "_archive" / "index-snapshots" / ("INDEX-before-publish-" + hashlib.sha256(prior).hexdigest()[:16] + ".yaml")
            if snapshot.exists() and snapshot.read_bytes() != prior:
                raise StoreError("Index snapshot conflict")
            if not snapshot.exists():
                _atomic_write(snapshot, prior)
        changed = False
        for sample in samples:
            case = sample["case_id"]
            episode = episodes[case]
            if case not in closed:
                result["unknown"].append({"case_id": case, "reason": "No closed case record"})
                continue
            records = inspections.get(case, [])
            done = (closed[case].get("status") == "complete" and episode["started"] == 1
                    and episode["results"] == 1 and len(records) == 1
                    and records[0]["log_status"] == "success" and not records[0]["error"]
                    and bool(records[0]["completed_at"])
                    and all(v["closed"] for v in episode["pending"].values()))
            finish = _utc(records[0]["completed_at"]) if done else None
            local = datetime.fromisoformat(finish).astimezone(zone).isoformat() if finish else None
            hour = datetime.fromisoformat(finish).astimezone(zone).strftime("%Y-%m-%d_%H") if finish else "incomplete-unknown"
            model = sample.get("model")
            model_id = sample.get("model_id")
            arm = sample.get("arm_id")
            if not all(isinstance(v, str) and v for v in (model, model_id, arm)):
                raise StoreError("Manifest model or arm identity missing")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_id) or ".." in model_id:
                raise StoreError("Unsafe model ID in manifest")
            config = settings.get(model_id, {})
            adapter = config.get("adapter_sha256")
            if adapter is not None and (not isinstance(adapter, str) or not re.fullmatch(r"[0-9a-f]{64}", adapter)):
                raise StoreError("Malformed manifest adapter hash")
            stream = batch / ("STREAM_" + model_id + ".jsonl")
            if not stream.is_file():
                stream = batch / "STREAM.jsonl"
            stream_bundles = []
            allowed_bundles = {batch.name, sample.get("bundle")}
            if stream.is_file():
                stream_bundles = _verify_stream(stream, case, episode["stream_digest"].hexdigest(),
                                                episode["counts"].get("model_stream", 0), allowed_bundles,
                                                episode["stream_bundle_sequence"])
            uid = _episode_id(event_sha, case)
            logical = (case, model, arm, adapter, finish)
            matching = [e for e in index["entries"] if e.get("episode_id") == uid]
            if len(matching) > 1:
                raise StoreError("Duplicate episode ID in index")
            for existing in index["entries"]:
                prior_logical = (existing.get("case_id"), existing.get("model"), existing.get("arm_id"),
                                 existing.get("adapter_sha256"), existing.get("finish_utc"))
                if finish and prior_logical == logical and existing.get("episode_id") != uid:
                    raise StoreError("Conflicting evidence for the same completed episode")
            if matching:
                existing = matching[0]
                if (existing.get("case_id"), existing.get("model"), existing.get("arm_id"),
                    existing.get("adapter_sha256"), existing.get("finish_utc")) != logical:
                    raise StoreError("Episode identity or completion conflicts with prior publication")
                previous_manifest_path = Path(existing["raw_dir"]) / "MANIFEST.json"
                if not previous_manifest_path.is_file():
                    raise StoreError("Prior manifest missing; cannot verify model configuration")
                previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
                previous_samples = [s for s in previous_manifest.get("samples", []) if s.get("case_id") == case]
                previous_config = previous_manifest.get("generation_settings", {}).get(model_id, {})
                identity_keys = ("model", "model_id", "arm_id", "seed", "prompt_condition",
                                 "rules_version", "system_prompt_sha256", "horizon_hours")
                if (len(previous_samples) != 1 or previous_config != config
                        or any(previous_samples[0].get(k) != sample.get(k) for k in identity_keys)):
                    raise StoreError("Prior model configuration or scenario identity conflicts")
                prior_conversation = load_conversation(root, uid)
                if prior_conversation != episode["messages"]:
                    raise StoreError("Redelivery conversation differs from recorded publication")
                aliases = existing.setdefault("source_aliases", [existing.get("source_dir")])
                if str(batch) not in aliases:
                    aliases.append(str(batch)); changed = True
                result["entries"].append(existing)
                continue
            alias = "sp-lense-" + adapter[:12] if adapter else _safe(config.get("service_model_id") or model.split("/")[-1])
            folder = root / hour / (hour + "--" + alias)
            slug = _safe(arm)[:24] + "-" + uid[:16]
            raw = folder / "raw" / slug
            conversation = folder / "conversation" / (slug + ".yaml")
            if not raw.resolve().is_relative_to(root) or not conversation.resolve().is_relative_to(root):
                raise StoreError("Destination escaped store")
            if raw.resolve().is_relative_to(batch) or conversation.resolve().is_relative_to(batch):
                raise StoreError("Destination overlaps source batch")
            stage = root / "_in-progress" / ("publish-" + uuid.uuid4().hex)
            stage_raw, stage_conversation = stage / "raw", stage / "conversation.yaml"
            files = [events_path, manifest_path, closeout_path]
            for name in ("LOCAL_BINDING_READBACK.json", "REPLAYED_RESULT.json", "REPLAYED_RESULTS.json", "REQUEST_GUARD_EVIDENCE.jsonl"):
                path = batch / name
                if path.is_file():
                    files.append(path)
            if stream.is_file():
                files.append(stream)
            if len(records) == 1:
                files.append(records[0]["path"])
            copied = []
            for path in files:
                if not path.resolve().is_relative_to(batch):
                    raise StoreError("Evidence path escaped source batch")
                filename = "episode.eval" if path.suffix == ".eval" else path.name
                info = _copy_verified(path, stage_raw / filename)
                info["copy"] = str((raw / filename).resolve())
                copied.append(info)
            if stream.is_file():
                _verify_stream(stage_raw / stream.name, case, episode["stream_digest"].hexdigest(),
                               episode["counts"].get("model_stream", 0), allowed_bundles,
                               episode["stream_bundle_sequence"])
            if (_sha(events_path) != event_sha or manifest_path.read_bytes() != manifest_bytes
                    or closeout_path.read_bytes() != closeout_bytes):
                raise StoreError("Batch source changed during publication")
            support = _collect_supporting_files(batch, stage_raw, raw, manifest)
            support.update({"episode_id": uid, "case_id": case,
                            "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()})
            support_bytes = _yaml_bytes(support)
            _atomic_write(stage_raw / "SUPPORTING_FILES.yaml", support_bytes)
            text = _yaml_bytes(episode["messages"])
            _atomic_write(stage_conversation, text)
            digest = hashlib.sha256(text).hexdigest()
            entry = {"episode_id": uid, "finish_edmonton": local, "finish_utc": finish,
                     "completion_status": "complete" if done else "incomplete_or_unknown",
                     "model_id": model_id, "model": model, "adapter_sha256": adapter,
                     "arm_id": arm, "case_id": case, "batch": batch.name,
                     "source_collection": batch.parent.name, "conversation": str(conversation),
                     "conversation_sha256": digest, "raw_dir": str(raw), "source_dir": str(batch),
                     "source_aliases": [str(batch)], "source_event_sha256": event_sha,
                     "event_counts": episode["counts"], "provider_generations": len(episode["pending"]),
                     "completed_provider_generations": sum(v["closed"] for v in episode["pending"].values()),
                     "output_mismatches": episode["output_mismatches"], "unexpected_event_counts": {},
                     "missing": ([] if done else ["incomplete_or_unknown"]) + ([] if stream.is_file() else ["separate_stream_file"]),
                     "supporting_files_manifest": str(raw / "SUPPORTING_FILES.yaml"),
                     "supporting_files_sha256": hashlib.sha256(support_bytes).hexdigest(),
                     "missing_source_files": len(support["unavailable_sources"]),
                     "excluded_source_files": len(support["excluded_sources"])}
            provenance = {"format": "research4.game-provenance.v1", "entry": entry,
                          "sample": sample, "model_configuration": config, "raw_files": copied,
                          "conversation_display": "Escaped newlines in valid JSON input prefixes are displayed as line breaks; exact text remains in raw EVENTS/STREAM and is authoritative for replay.",
                          "bundle_labels": {"manifest": sample.get("bundle"),
                                            "events": sorted(episode["event_bundles"]),
                                            "stream": stream_bundles},
                          "inspect": [{"path": str(x["path"]), "log_status": x["log_status"],
                                       "completed_at": x["completed_at"], "error": x["error"]} for x in records]}
            _atomic_write(stage_raw / "PROVENANCE.yaml", _yaml_bytes(provenance))
            raw.parent.mkdir(parents=True, exist_ok=True)
            conversation.parent.mkdir(parents=True, exist_ok=True)
            require_lock()
            # A crash can leave one or both artifacts before the atomic index
            # replacement. Recover only byte-identical, provenance-matching
            # orphans; never overwrite a different unindexed destination.
            if raw.exists():
                prior_provenance = raw / "PROVENANCE.yaml"
                if (not prior_provenance.is_file()
                        or yaml.safe_load(prior_provenance.read_text(encoding="utf-8")) != provenance
                        or not (raw / "SUPPORTING_FILES.yaml").is_file()
                        or _sha(raw / "SUPPORTING_FILES.yaml") != hashlib.sha256(support_bytes).hexdigest()
                        or any(not Path(item["copy"]).resolve().is_relative_to(raw.resolve())
                               or not Path(item["copy"]).is_file()
                               or _sha(Path(item["copy"])) != item["sha256"]
                               for item in support["source_files"] + support["run_context_files"])
                        or any(not (raw / Path(item["copy"]).name).is_file()
                               or _sha(raw / Path(item["copy"]).name) != item["sha256"]
                               for item in copied)):
                    raise StoreError("Unindexed raw destination conflicts; no files overwritten")
            else:
                os.rename(stage_raw, raw)
            if conversation.exists():
                if _sha(conversation) != digest:
                    raise StoreError("Unindexed conversation conflicts; no files overwritten")
            else:
                os.rename(stage_conversation, conversation)
            index["entries"].append(entry)
            result["entries"].append(entry)
            result["published"] += 1
            if not done:
                result["unknown"].append({"case_id": case, "reason": "Incomplete or unverifiable completion"})
            changed = True
        if changed:
            require_lock()
            if prior is not None and index_path.read_bytes() != prior:
                raise StoreError("Index changed despite held lock")
            index["entries"].sort(key=lambda e: (e.get("finish_utc") is None, e.get("finish_utc") or "", e.get("model_id") or "", e.get("episode_id") or ""))
            counts = dict(index.get("counts", {}))
            counts.update({"episodes": len(index["entries"]),
                           "complete_episodes": sum(e.get("completion_status") == "complete" for e in index["entries"]),
                           "incomplete_or_unknown_episodes": sum(e.get("completion_status") != "complete" for e in index["entries"])})
            for target, source in (("prompts", "model_prompt"), ("actions", "simulator_action"), ("model_stream_records", "model_stream")):
                counts[target] = sum(e.get("event_counts", {}).get(source, 0) for e in index["entries"])
            index["counts"] = counts
            _atomic_write(index_path, _yaml_bytes(index))
        require_lock()
        if index_path.exists():
            _atomic_write(root / "README.md", _readme(index, root))
    return result
