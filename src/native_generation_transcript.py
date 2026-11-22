"""Future-only offline decoding of saved native token IDs; never runs a model.

The caller supplies existing canonical publication entries after a batch closes.
Only a new sibling YAML is written. Request/turn associations must be explicit:
guard rows use ``generation_identity={case_id, decision, attempt}`` (their
top-level ``attempt`` is a different, global HTTP ordinal), or a model_stream
event carries request_id/messages_sha256 with its case/decision/attempt.
Historical files without that evidence are reported as unavailable.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

from .research_paths import fine_tune_root


REFERENCE_MANIFEST_SHA256 = "de89053ae8ca8fe5c5175649b39b27b1c7929b463280d7f55be1a1cb765cade4"
NATIVE_METADATA_SHA256 = "df8ee1401626c4ee909263a83a1a6ce4298b3095d689217bd75ff75ccb2425d4"
REFERENCE_ROOT = fine_tune_root() / ".local/reference-tokenizer"
CPU_REFERENCE_PYTHON = REFERENCE_ROOT.parent / "reference-tokenizer-venv/Scripts/python.exe"
NATIVE_METADATA_PATH = REFERENCE_ROOT.parent / (
    "staging/continuity_transfer_endpoint_held_20261115/native_tokenizer_metadata_20261115/"
    "ACTUAL_NATIVE_TOKENIZER_SOURCE_READBACK_v1.json")
FORMAT = "research4.native-generation-transcript.v1"
NOTE = ("This is recorded native-generation text, separate from the client-visible stream. "
        "Reasoning denotes an explicit model output channel; it does not establish hidden "
        "internal intent or an exhaustive account of internal computation.")


class TranscriptError(ValueError):
    """Closed, immutable evidence or a safe new destination was not established."""


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise TranscriptError("Duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise TranscriptError("Nonfinite JSON value")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8"))


def _hex(value, length):
    return type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None


def _identity(value):
    if (type(value) is not dict or set(value) != {"case_id", "decision", "attempt"}
            or type(value["case_id"]) is not str or not value["case_id"]
            or type(value["decision"]) is not int or value["decision"] < 0
            or type(value["attempt"]) is not int or value["attempt"] < 1):
        raise TranscriptError("Explicit case/decision/generation-attempt identity required")
    return value["case_id"], value["decision"], value["attempt"]


@dataclass
class _Reference:
    tokenizer: object
    render_prompt: object
    provenance: dict
    eos_ids: list
    canonical_eos: int
    vocab_size: int
    think_open: int
    think_close: int
    base_manifest_sha256: str


def _load_reference():
    """Authenticate only local tokenizer files and existing CPU dependencies."""
    manifest_raw = (REFERENCE_ROOT / "REFERENCE_MANIFEST.json").read_bytes()
    if _sha(manifest_raw) != REFERENCE_MANIFEST_SHA256:
        raise TranscriptError("Authenticated CPU reference manifest differs")
    manifest = _json(manifest_raw)
    files = {}
    for item in manifest["files"]:
        name = item["path"]
        if name in files or Path(name).name != name:
            raise TranscriptError("Unsafe or duplicate tokenizer manifest member")
        raw = (REFERENCE_ROOT / name).read_bytes()
        if len(raw) != item["bytes"] or _sha(raw) != item["sha256"]:
            raise TranscriptError("Authenticated tokenizer member differs: " + name)
        files[name] = {"bytes": len(raw), "sha256": _sha(raw)}
    if set(files) != {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json", "vocab.json"}:
        raise TranscriptError("Closed tokenizer manifest members required")
    metadata_raw = NATIVE_METADATA_PATH.read_bytes()
    if _sha(metadata_raw) != NATIVE_METADATA_SHA256:
        raise TranscriptError("Authenticated native tokenizer source readback differs")
    metadata = _json(metadata_raw)
    if (metadata["tokenizer_json_sha256"] != files["tokenizer.json"]["sha256"]
            or metadata["base_manifest_sha256"] != manifest["source_base_manifest_sha256"]):
        raise TranscriptError("CPU/native tokenizer identity differs")
    # This existing loader imports tokenizers/Jinja only from the preinstalled,
    # version-pinned CPU reference venv. No transformers, model or network call.
    from . import full_history_context as context
    if context.REFERENCE_ROOT.resolve() != REFERENCE_ROOT.resolve():
        raise TranscriptError("CPU counter and decoder reference roots differ")
    tokenizer, _, dependency_provenance = context._resources()
    config = _json((REFERENCE_ROOT / "config.json").read_bytes())
    token_config = _json((REFERENCE_ROOT / "tokenizer_config.json").read_bytes())
    eos = sorted(config["eos_token_id"])
    # Model logits include padded vocabulary slots that are not tokenizer
    # entries. Preserve the native evidence's model vocabulary identity, and
    # separately reject an actual ID without a defined tokenizer entry.
    native_vocab_size = config["text_config"]["vocab_size"]
    canonical = tokenizer.token_to_id(token_config["eos_token"])
    opening, closing = tokenizer.token_to_id("<think>"), tokenizer.token_to_id("</think>")
    if (eos != [248044, 248046] or canonical != 248046 or opening is None or closing is None
            or tokenizer.decode([opening], skip_special_tokens=False) != "<think>"
            or tokenizer.decode([closing], skip_special_tokens=False) != "</think>"):
        raise TranscriptError("Native EOS/reasoning tags not established by tokenizer")
    return _Reference(tokenizer, context.render_formatted_prompt, {
        "reference_root": str(REFERENCE_ROOT.resolve()),
        "manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "files": files, "native_metadata_path": str(NATIVE_METADATA_PATH.resolve()),
        "native_metadata_sha256": NATIVE_METADATA_SHA256,
        "dependencies": dependency_provenance,
        "native_model_vocab_size": native_vocab_size,
        "defined_tokenizer_vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "decode_skip_special_tokens": False,
        "scope": "CPU text decoding only; no model inference or native execution equivalence claim",
    }, eos, canonical, native_vocab_size, opening, closing,
        manifest["source_base_manifest_sha256"])


def _snapshot(batch):
    paths = {name: batch / name for name in ("MANIFEST.json", "CLOSEOUT.json", "EVENTS.jsonl")}
    if not all(path.is_file() for path in paths.values()):
        raise TranscriptError("Batch must already have manifest, EVENTS and CLOSEOUT; active batches are excluded")
    guard = batch / "REQUEST_GUARD_EVIDENCE.jsonl"
    if guard.is_file():
        paths[guard.name] = guard
    raw = {name: path.read_bytes() for name, path in paths.items()}
    manifest, closeout = _json(raw["MANIFEST.json"]), _json(raw["CLOSEOUT.json"])
    cases = [sample.get("case_id") for sample in manifest.get("samples", [])]
    if not cases or any(type(case) is not str or not case for case in cases) or len(set(cases)) != len(cases):
        raise TranscriptError("Unique manifest cases required")
    results = closeout.get("results") or ([closeout["result"]] if closeout.get("result") else [])
    closed = [row.get("case_id") for row in results]
    if len(set(closed)) != len(closed) or set(closed) != set(cases):
        raise TranscriptError("Every manifest case must have one closed case record")
    records = {}
    for name in ("EVENTS.jsonl", "REQUEST_GUARD_EVIDENCE.jsonl"):
        records[name] = []
        for line_number, line in enumerate(raw.get(name, b"").splitlines(keepends=True), 1):
            if not line.strip():
                raise TranscriptError("Blank JSONL evidence record")
            value = _json(line)
            if type(value) is not dict:
                raise TranscriptError("JSONL evidence must contain objects")
            records[name].append((value, {"file": name, "line": line_number,
                                         "raw_record_sha256": _sha(line)}))
    return {"batch": batch, "paths": paths, "raw": raw, "cases": cases,
            "records": records, "source_files": {
                name: {"path": str(paths[name]), "bytes": len(value), "sha256": _sha(value)}
                for name, value in raw.items()}}


def _unchanged(snapshot):
    if any(path.read_bytes() != snapshot["raw"][name] for name, path in snapshot["paths"].items()):
        raise TranscriptError("Closed source changed while decoding; no output authorized")
    if (snapshot["batch"] / "REQUEST_GUARD_EVIDENCE.jsonl").is_file() != ("REQUEST_GUARD_EVIDENCE.jsonl" in snapshot["raw"]):
        raise TranscriptError("Guard evidence appeared/disappeared while decoding")


def _runtime_provenance():
    source = Path(__file__).resolve()
    context_source = source.with_name("full_history_context.py")
    executable = Path(sys.executable).resolve()
    return {"python_executable": str(executable), "python_executable_sha256": _sha(executable.read_bytes()),
            "python_version": sys.version, "pyyaml_version": yaml.__version__,
            "decoder_source": str(source), "decoder_source_sha256": _sha(source.read_bytes()),
            "prompt_renderer_source": str(context_source),
            "prompt_renderer_source_sha256": _sha(context_source.read_bytes())}


def _stream(records):
    return {"label": "client-visible recorded stream; separate evidence channel",
            "text": "".join(row.get("text", "") for row, _ in records if row.get("event") == "text_delta"),
            "reasoning": "".join(row.get("text", "") for row, _ in records
                                 if row.get("event") in {"reasoning_delta", "reasoning_summary_delta"}),
            "records": [ref for _, ref in records]}


def _decode_request(request_id, rows, reference):
    attempts = [(row, ref) for row, ref in rows if row.get("event") == "HTTP_ATTEMPT"]
    posts = [(row, ref) for row, ref in rows if row.get("event") == "POST_REQUEST_VALIDATED"]
    if len(attempts) != 1 or len(posts) != 1:
        raise TranscriptError("Exactly one HTTP_ATTEMPT and POST_REQUEST_VALIDATED required for this request")
    attempt, post = attempts[0][0], posts[0][0]
    digest = attempt.get("messages_sha256")
    if not _hex(digest, 64) or post.get("messages_sha256") != digest:
        raise TranscriptError("Same-request guard message hashes differ or are absent")
    if "exact_serialized_messages" in attempt and _digest(attempt["exact_serialized_messages"]) != digest:
        raise TranscriptError("Exact saved wire messages differ from guarded message hash")
    decision = post.get("decision", {})
    evidence = post.get("fresh_health", {}).get("last_request_evidence")
    if (type(decision) is not dict or decision.get("request_id") != request_id
            or decision.get("post_request_completion_allowed") is not True or type(evidence) is not dict
            or evidence.get("request_id") != request_id or evidence.get("messages_sha256") != digest):
        raise TranscriptError("Explicit same-request validated native completion evidence required")
    if (evidence.get("tokenizer_metadata_sha256") != NATIVE_METADATA_SHA256
            or evidence.get("base_manifest_sha256") != reference.base_manifest_sha256
            or evidence.get("vocab_size") != reference.vocab_size
            or evidence.get("canonical_eos_token_id") != reference.canonical_eos
            or evidence.get("eos_token_ids") != reference.eos_ids):
        raise TranscriptError("Saved native tokenizer/base/EOS identity differs from authenticated reference")
    ids = evidence.get("actual_generated_token_ids")
    if (type(ids) is not list or not ids or len(ids) > 8192
            or any(type(token) is not int or not 0 <= token < reference.vocab_size
                   or reference.tokenizer.id_to_token(token) is None for token in ids)
            or type(evidence.get("completion_tokens")) is not int or evidence["completion_tokens"] != len(ids)):
        raise TranscriptError("Exact bounded actual_generated_token_ids and matching completion count required")
    result = {"status": "incomplete", "request_id": request_id, "messages_sha256": digest,
              "guard_records": [ref for _, ref in rows], "actual_generated_token_count": len(ids),
              "token_ids_sha256": _digest(ids), "token_ids_hash_encoding": "canonical compact JSON array, UTF-8",
              "native_text_with_special_tokens": reference.tokenizer.decode(ids, skip_special_tokens=False),
              "reasoning_partition_status": "unavailable", "issues": []}
    ack = evidence.get("completion_ack", {})
    dispatch = evidence.get("dispatch_record", {})
    if (type(ack) is not dict or not _hex(evidence.get("native_ticket_id"), 32)
            or ack.get("request_id") != evidence["native_ticket_id"]
            or any(ack.get(key) is not True for key in ("worker_started", "worker_finished", "backend_barrier_done", "post_checks_passed"))
            or ack.get("native_errors") != [] or type(dispatch) is not dict
            or dispatch.get("phase") != "GAME_WINDOW" or dispatch.get("request_id") != request_id
            or not _hex(attempt.get("window_sha256"), 64)
            or dispatch.get("window_sha256") != attempt["window_sha256"]
            or post.get("window_sha256") != attempt["window_sha256"]):
        result["issues"].append("Same-request native acknowledgement/Game dispatch proof incomplete")
    if (ids[-1] not in reference.eos_ids or any(token in reference.eos_ids for token in ids[:-1])
            or evidence.get("EOS_detected") is not True or evidence.get("finish_reason") != "stop"):
        result["issues"].append("Complete native EOS termination not established; decoded text is partial/unverified")
    if result["issues"]:
        return result
    result["status"] = "complete"
    body = ids[:-1]
    result["native_response_text"] = reference.tokenizer.decode(body, skip_special_tokens=False)
    result["terminal_eos_token_id"] = ids[-1]
    openings = [i for i, token in enumerate(body) if token == reference.think_open]
    closings = [i for i, token in enumerate(body) if token == reference.think_close]
    start, boundary = None, None
    if openings == [0] and len(closings) == 1 and closings[0] > 0:
        start, boundary = 1, closings[0]
        result["reasoning_boundary_evidence"] = "Actual generated opening/closing tag token IDs"
    elif not openings and len(closings) == 1:
        messages = attempt.get("exact_serialized_messages")
        if type(messages) is list and _digest(messages) == digest:
            prompt = reference.render_prompt(messages)
            if prompt.endswith("<|im_start|>assistant\n<think>\n"):
                start, boundary = 0, closings[0]
                result["reasoning_boundary_evidence"] = "Authenticated exact-message chat template prefills <think>; generated closing tag token ID"
    if boundary is None:
        result["issues"].append("Unambiguous actual reasoning tags/prefilled opening not established; full native text retained without a channel assignment")
    else:
        result["reasoning_partition_status"] = "verified_explicit_output_channel"
        result["reasoning_text"] = reference.tokenizer.decode(body[start:boundary], skip_special_tokens=False)
        result["action_text"] = reference.tokenizer.decode(body[boundary + 1:], skip_special_tokens=False)
    return result


def _build(snapshot):
    events, guards = snapshot["records"]["EVENTS.jsonl"], snapshot["records"]["REQUEST_GUARD_EVIDENCE.jsonl"]
    expected, bindings, requests = defaultdict(list), defaultdict(set), defaultdict(list)
    binding_refs, malformed = defaultdict(list), []
    prompt_decisions = set()
    for row, ref in events:
        if row.get("type") == "model_prompt":
            identity = _identity({"case_id": row.get("case_id"), "decision": row.get("decision"), "attempt": 1})
            if identity[0] not in snapshot["cases"]:
                raise TranscriptError("Out-of-manifest prompt identity")
            prompt_decisions.add(identity[:2])
        if row.get("type") == "model_stream":
            identity = _identity({key: row.get(key) for key in ("case_id", "decision", "attempt")})
            if identity[0] not in snapshot["cases"]:
                raise TranscriptError("Out-of-manifest stream identity")
            expected[identity].append((row, ref))
            if "request_id" in row:
                rid = row["request_id"]
                if not _hex(rid, 32) or not _hex(row.get("messages_sha256"), 64):
                    malformed.append({"reason": "Malformed explicit event request binding", "record": ref})
                else:
                    bindings[rid].add(identity); binding_refs[rid].append(ref)
    for case, decision in prompt_decisions:
        if not any(key[:2] == (case, decision) for key in expected):
            expected[(case, decision, 1)] = []
    for row, ref in guards:
        rid = row.get("request_id")
        if not _hex(rid, 32):
            if row.get("event") in {"HTTP_ATTEMPT", "POST_REQUEST_VALIDATED"}:
                malformed.append({"reason": "Malformed guard request identity", "record": ref})
            continue
        requests[rid].append((row, ref))
        if "generation_identity" in row:
            try:
                identity = _identity(row["generation_identity"])
                if identity[0] not in snapshot["cases"]:
                    raise TranscriptError("Out-of-manifest guard generation identity")
                bindings[rid].add(identity); binding_refs[rid].append(ref)
            except TranscriptError as error:
                malformed.append({"request_id": rid, "reason": str(error), "record": ref})
    unmapped = [{"request_id": rid, "reason": "No unique explicit guard/event request-to-generation binding",
                 "guard_records": [ref for _, ref in rows], "binding_records": binding_refs[rid]}
                for rid, rows in requests.items() if len(bindings[rid]) != 1]
    by_identity = defaultdict(list)
    for rid, values in bindings.items():
        for identity in values:
            by_identity[identity].append(rid)
    try:
        reference, reference_error = _load_reference(), None
    except (OSError, ValueError, KeyError, ImportError) as error:
        reference, reference_error = None, str(error)
    output, runtime = {}, _runtime_provenance()
    for case in snapshot["cases"]:
        turns = []
        for identity in sorted(key for key in expected if key[0] == case):
            rows = expected[identity]
            rids = by_identity[identity]
            result = {"status": "unavailable", "issues": []}
            if len(rids) != 1 or len(bindings[rids[0]]) != 1:
                result["issues"].append("Unique explicit request_id to generation mapping not established; no order-based pairing used")
            elif reference is None:
                result["issues"].append("Authenticated offline tokenizer unavailable: " + reference_error)
            else:
                try:
                    result = _decode_request(rids[0], requests[rids[0]], reference)
                    # An event binding must agree with the guarded message hash.
                    if any(row.get("request_id") == rids[0] and row.get("messages_sha256") != result["messages_sha256"] for row, _ in rows):
                        raise TranscriptError("Explicit event/guard messages hashes differ")
                except (ValueError, KeyError, TypeError) as error:
                    result = {"status": "unavailable", "request_id": rids[0], "issues": [str(error)]}
            result.update(case_id=case, decision=identity[1], attempt=identity[2],
                          binding_records=[ref for rid in rids for ref in binding_refs[rid]],
                          client_visible_stream=_stream(rows))
            turns.append(result)
        extra = [{"identity": {"case_id": key[0], "decision": key[1], "attempt": key[2]}, "request_ids": rids}
                 for key, rids in by_identity.items() if key[0] == case and key not in expected]
        complete = bool(turns) and all(row["status"] == "complete" and row["reasoning_partition_status"] == "verified_explicit_output_channel" for row in turns)
        output[case] = {"format": FORMAT, "label": "Saved native-generation transcript (offline token decoding)",
                        "interpretation": NOTE, "case_id": case,
                        "status": "complete" if complete and not (extra or unmapped or malformed) else "incomplete_or_unavailable",
                        "decoder_runtime": runtime,
                        "source_files": snapshot["source_files"],
                        "tokenizer": reference.provenance if reference else {"status": "unavailable", "reason": reference_error,
                                                                             "required_manifest_sha256": REFERENCE_MANIFEST_SHA256},
                        "unmapped_requests": unmapped, "malformed_bindings": malformed,
                        "bindings_without_stream_generation": extra, "generations": turns}
    return output


def build_batch_transcripts(batch_dir):
    """Read a closed batch without writing; return one labeled document per case."""
    snapshot = _snapshot(Path(batch_dir).resolve())
    documents = _build(snapshot)
    _unchanged(snapshot)
    return documents


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(str, lambda dumper, value: dumper.represent_scalar(
    "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None))


def write_batch_transcripts(batch_dir, publication_entries):
    """Write new per-case sibling YAMLs from existing publish_batch entries.

    Refuses to replace any existing artifact. Byte-identical redelivery is a
    read-only success. No raw source, conversation, INDEX or model is changed.
    """
    snapshot = _snapshot(Path(batch_dir).resolve())
    documents = _build(snapshot)
    prepared, seen = [], set()
    for entry in publication_entries:
        case = entry.get("case_id")
        if case in seen or case not in documents:
            raise TranscriptError("Unique in-batch publication entries required")
        seen.add(case)
        conversation = Path(entry["conversation"]).resolve()
        raw_dir = Path(entry["raw_dir"]).resolve()
        if (conversation.suffix != ".yaml" or conversation.parent.name != "conversation"
                or not conversation.is_file() or conversation.is_relative_to(snapshot["batch"])
                or conversation.is_relative_to(raw_dir)
                or _sha(conversation.read_bytes()) != entry.get("conversation_sha256")
                or entry.get("source_event_sha256") != snapshot["source_files"]["EVENTS.jsonl"]["sha256"]):
            raise TranscriptError("Existing canonical conversation/hash/source binding required")
        destination = conversation.with_name(conversation.stem + ".native-generation.yaml")
        document = {**documents[case], "canonical_conversation": {
            "path": str(conversation), "sha256": entry["conversation_sha256"],
            "relationship": "Separate companion artifact; existing client-visible conversation is preserved"}}
        raw = yaml.dump(document, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=120).encode("utf-8")
        if yaml.safe_load(raw) != document:
            raise TranscriptError("Companion YAML round-trip failed")
        if destination.exists() and destination.read_bytes() != raw:
            raise TranscriptError("Existing native-generation companion conflicts; no overwrite allowed")
        prepared.append((destination, raw, document, conversation, entry["conversation_sha256"]))
    _unchanged(snapshot)
    results = []
    for destination, raw, document, conversation, conversation_sha in prepared:
        _unchanged(snapshot)
        if _sha(conversation.read_bytes()) != conversation_sha:
            raise TranscriptError("Canonical conversation changed while decoding")
        created = False
        try:
            with destination.open("xb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            created = True
        except FileExistsError:
            if destination.read_bytes() != raw:
                raise TranscriptError("Companion appeared with conflicting content; no overwrite allowed") from None
        if destination.read_bytes() != raw:
            raise TranscriptError("New companion readback differs")
        results.append({"case_id": document["case_id"], "path": str(destination),
                        "sha256": _sha(raw), "status": document["status"], "created": created})
    return {"format": FORMAT, "artifacts": results, "decoder_runtime": _runtime_provenance()}


def run_decoder_subprocess(batch_dir, publication_entries, python_executable):
    """Run this source in the explicitly supplied existing CPU reference venv.

    An artifact marked incomplete_or_unavailable is a successful evidence
    publication. Process, I/O, source, or output-verification failures raise
    TranscriptError and must be handled separately from a Game result.
    """
    executable = Path(python_executable)
    if (not executable.is_absolute() or not executable.is_file()
            or executable.resolve() != CPU_REFERENCE_PYTHON.resolve()):
        raise TranscriptError("Supply the exact existing absolute CPU reference-venv Python executable")
    entries = list(publication_entries)
    expected_source_sha = _sha(Path(__file__).resolve().read_bytes())
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    try:
        completed = subprocess.run([str(executable.resolve()), "-m", "src.native_generation_transcript",
                                    "--batch", str(Path(batch_dir).resolve()), "--entries-stdin"],
                                   input=json.dumps(entries, ensure_ascii=False, allow_nan=False),
                                   text=True, encoding="utf-8", capture_output=True, check=False,
                                   cwd=str(Path(__file__).resolve().parents[1]), env=environment, timeout=600)
    except (OSError, subprocess.SubprocessError) as error:
        raise TranscriptError("Offline decoder subprocess could not complete: " + str(error)) from error
    if completed.returncode != 0:
        raise TranscriptError("Offline decoder subprocess failed: " + completed.stderr[-4000:])
    try:
        result = _json(completed.stdout)
        runtime = result["decoder_runtime"]
        if (result["format"] != FORMAT or runtime["decoder_source_sha256"] != expected_source_sha
                or Path(runtime["python_executable"]).resolve() != executable.resolve()
                or runtime["python_executable_sha256"] != _sha(executable.read_bytes())):
            raise TranscriptError("Offline decoder subprocess runtime/source readback differs")
        artifacts = result["artifacts"]
        expected = {entry["case_id"]: Path(entry["conversation"]).resolve().with_name(
            Path(entry["conversation"]).stem + ".native-generation.yaml") for entry in entries}
        if len(expected) != len(entries) or len(artifacts) != len(expected):
            raise TranscriptError("Offline decoder subprocess output count differs")
        seen = set()
        for artifact in artifacts:
            case, path = artifact["case_id"], Path(artifact["path"]).resolve()
            if (case in seen or path != expected.get(case)
                    or artifact["status"] not in {"complete", "incomplete_or_unavailable"}
                    or _sha(path.read_bytes()) != artifact["sha256"]):
                raise TranscriptError("Offline decoder subprocess artifact readback differs")
            document = yaml.safe_load(path.read_bytes())
            if (document.get("format") != FORMAT or document.get("case_id") != case
                    or document.get("status") != artifact["status"]
                    or document.get("decoder_runtime") != runtime):
                raise TranscriptError("Offline decoder companion contents differ from summary")
            seen.add(case)
        return result
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise TranscriptError("Offline decoder subprocess output verification failed: " + str(error)) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--entries-stdin", action="store_true", required=True,
                        help="Read an existing publish_batch entries JSON list from stdin")
    args = parser.parse_args(argv)
    entries = _json(sys.stdin.read())
    if type(entries) is not list:
        raise TranscriptError("Publication entries JSON list required")
    print(json.dumps(write_batch_transcripts(args.batch, entries), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
