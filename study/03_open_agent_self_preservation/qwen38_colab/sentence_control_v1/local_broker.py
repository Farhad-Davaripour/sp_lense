"""Trusted local Jev transport: three fields, carried budgets, no retries/dedup."""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVICE = HERE.parent / "matched_jev_v1"
sys.path.insert(0, str(SERVICE))
from local_service import APPROVED_DOTENV, decode, official_transport, strict_json
from jev_budget import request_size_reserve, validate_readiness, write_json

_spec = importlib.util.spec_from_file_location("sentence_broker_schema", HERE / "schema.py")
schema = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(schema)

SCHEMA_SHA = "4408cebe3a6ae54d51a5bebd58eac2ce5915b9a42a866b6a062b5381581bd544"
ARTIFACTS = Path("C:/Users/farha/Research3/artifacts/sentence_control_v1")
PHASE = ARTIFACTS / "api_phase"
READINESS = HERE / "API_READINESS.json"
MONITOR = HERE / "RECOVERY_STATUS.json"
PRIOR_TOKEN_LEDGER = Path("C:/Users/farha/Research3/artifacts/token_refresh_recovery_v1/api_phase/STATUS.json")
MAX_CALLS = 2048
MAX_INPUT_USD = 1.0


def prior_usage():
    """Only the superseded REAL token ledger shares this new allowance."""
    old = strict_json(PRIOR_TOKEN_LEDGER.read_bytes()) if PRIOR_TOKEN_LEDGER.is_file() else {}
    result = {name: old.get(name, 0) for name in
              ("attempted_requests", "successful_requests", "input_tokens", "reserved_input_equivalent")}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in result.values()):
        raise ValueError("Invalid carried-forward actual usage")
    if result["successful_requests"] > result["attempted_requests"]:
        raise ValueError("Invalid carried-forward request counts")
    return result


def new_state(rate, prior):
    return {"model": schema.MODEL, "schema_version": schema.SCHEMA_VERSION,
            "schema_source_sha256": SCHEMA_SHA, "questions_sha256": schema.digest(schema.QUESTIONS),
            "attempted_requests": 0, "successful_requests": 0, "input_tokens": 0,
            "reserved_input_equivalent": 0, "input_estimated_usd": 0.0, "stopped": False,
            "carried_prior_token_usage": prior, "input_usd_per_million": rate,
            "max_attempts_including_prior": MAX_CALLS, "max_successful_including_prior": MAX_CALLS,
            "max_estimated_input_usd_including_prior": MAX_INPUT_USD,
            "deduplication": False, "automatic_retries": False,
            "prior_matched_phase_is_separate": True, "actual_transport_required": True}


def validate_batch(value):
    if not isinstance(value, dict) or set(value) != {"batch_id", "rows"}:
        raise ValueError("Invalid visible-state batch")
    if not isinstance(value["batch_id"], str) or not isinstance(value["rows"], list) or not 1 <= len(value["rows"]) <= 2:
        raise ValueError("Invalid batch identity/count")
    identities = set()
    for row in value["rows"]:
        if not isinstance(row, dict) or set(row) != {"id", "history", "tools"}:
            raise ValueError("Only visible history/tools are accepted")
        if not isinstance(row["id"], str) or row["id"] in identities:
            raise ValueError("Duplicate or invalid visible-state identity")
        identities.add(row["id"])
        if not isinstance(row["history"], list) or not row["history"]:
            raise ValueError("Visible history is required")
        for message in row["history"]:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user", "assistant", "tool"}
                    or not isinstance(message["content"], str)):
                raise ValueError("Invalid visible message")
        if not isinstance(row["tools"], list):
            raise ValueError("Visible tool catalogue is required")
    return value


def update_monitor(state, success, failure=None):
    current = strict_json(MONITOR.read_bytes()) if MONITOR.is_file() else {}
    current.update(updated_utc=datetime.now(timezone.utc).isoformat(),
                   real_jev_calls=state.get("successful_requests", 0),
                   real_jev_attempts=state.get("attempted_requests", 0),
                   new_api_input_tokens=state.get("input_tokens", 0),
                   new_api_input_estimated_usd=state.get("input_estimated_usd", 0.0),
                   api_reserved_input_equivalent=state.get("reserved_input_equivalent", 0),
                   new_api_committed_plus_reserved_input_estimated_usd=(
                       state.get("input_tokens", 0) + state.get("reserved_input_equivalent", 0)) * 0.042 / 1e6,
                   real_api_ledger=str(PHASE / "STATUS.json"))
    if not success:
        current.update(state="incomplete", failure={"stage": "genuine_jev_annotation", **(failure or {})})
    write_json(MONITOR, current)


def run(input_path, output_path, phase_dir=PHASE, readiness_path=READINESS, monitor=True):
    phase, output = Path(phase_dir), Path(output_path)
    allowed = Path("C:/Users/farha/Research3").resolve()
    if any(not Path(path).resolve().is_relative_to(allowed) for path in (input_path, output, phase)):
        raise ValueError("Transfer and ledger files must stay on C: under Research3")
    if output.exists() or output.with_suffix(output.suffix + ".gz").exists():
        raise ValueError("Never overwrite a prior annotation response")
    if hashlib.sha256((HERE / "schema.py").read_bytes()).hexdigest() != SCHEMA_SHA:
        raise ValueError("Frozen sentence schema changed")
    readiness = strict_json(Path(readiness_path).read_bytes())
    rate = validate_readiness(readiness, schema.MODEL)
    if rate != 0.042:
        raise ValueError("Pinned provider input rate changed")
    blob = Path(input_path).read_bytes()
    value = validate_batch(strict_json(gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob))
    phase.mkdir(parents=True, exist_ok=True)
    lock = phase / "API_OWNER.lock"
    owner = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(owner, str(os.getpid()).encode("ascii"))
    os.close(owner)
    state_path = phase / "STATUS.json"
    state = {}
    success, failure, results = False, None, []
    try:
        prior = prior_usage()
        state = strict_json(state_path.read_bytes()) if state_path.is_file() else new_state(rate, prior)
        counters = ("attempted_requests", "successful_requests", "input_tokens", "reserved_input_equivalent")
        if (any(isinstance(state.get(name), bool) or not isinstance(state.get(name), int)
                or state[name] < 0 for name in counters)
                or state["successful_requests"] > state["attempted_requests"]
                or state["model"] != schema.MODEL or state["schema_version"] != schema.SCHEMA_VERSION
                or state["input_usd_per_million"] != rate):
            raise ValueError("Invalid saved actual usage or pinned ledger identity")
        state["input_estimated_usd"] = state["input_tokens"] * rate / 1e6
        if (state["stopped"] or state["carried_prior_token_usage"] != prior
                or state["schema_source_sha256"] != SCHEMA_SHA
                or state["questions_sha256"] != schema.digest(schema.QUESTIONS)):
            raise ValueError("Stopped ledger or changed carried usage/schema")
        for row in value["rows"]:
            validate_readiness(readiness, schema.MODEL)
            request = schema.payload(row["history"], row["tools"])
            identity = schema.request_identity(request)
            reserve = request_size_reserve(request)["total_input_equivalent"]
            total_attempts = prior["attempted_requests"] + state["attempted_requests"]
            total_successes = prior["successful_requests"] + state["successful_requests"]
            cost_equivalent = (prior["input_tokens"] + prior["reserved_input_equivalent"]
                               + state["input_tokens"] + state["reserved_input_equivalent"] + reserve)
            if total_attempts >= MAX_CALLS or total_successes >= MAX_CALLS or cost_equivalent * rate / 1e6 > MAX_INPUT_USD:
                raise ValueError("Carried-forward pilot request or input-cost cap")
            number = state["attempted_requests"]
            path = phase / "calls" / (str(number).zfill(5) + ".json")
            if path.exists():
                raise ValueError("Never overwrite an actual prior attempt")
            record = {"attempt_index": number, "request": request, "identity": identity,
                      "batch_id": value["batch_id"], "row_id": row["id"],
                      "preflight_reserve": reserve, "cache_hit": False,
                      "new_actual_service_transport_required": True}
            write_json(path, record)
            state["attempted_requests"] += 1
            state["reserved_input_equivalent"] += reserve
            write_json(state_path, state)
            tick = time.monotonic()
            _, raw = official_transport(request, APPROVED_DOTENV)
            response = strict_json(raw)
            decoded = decode(response, schema)
            usage = decoded["usage"]["input_tokens"]
            if usage > reserve:
                # A decoded successful service call still counts, even when its
                # observed usage invalidates admission and no aid is delivered.
                state["successful_requests"] += 1
                state["input_tokens"] += usage
                state["reserved_input_equivalent"] -= reserve
                state["input_estimated_usd"] = state["input_tokens"] * rate / 1e6
                record.update(response=response, decoded=decoded, usage=decoded["usage"],
                              successful_service_response=True, aid_not_delivered=True,
                              actual_input_exceeded_reserve=True)
                write_json(path, record)
                raise ValueError("Provider usage exceeds the conservative reserve")
            annotation = {"identity": identity, "request": request, "response": response,
                          "decoded": decoded, "provider_response_raw_base64": base64.b64encode(raw).decode(),
                          "provider_response_raw_sha256": hashlib.sha256(raw).hexdigest(),
                          "schema_source_sha256": SCHEMA_SHA, "latency_seconds": time.monotonic() - tick,
                          "usage": decoded["usage"], "input_estimated_usd": usage * rate / 1e6,
                          "actual_service_call": True, "cache_hit": False,
                          "credential_in_payload_or_output": False, "training_targets_in_payload": False}
            record.update(completed=True, annotation=annotation)
            write_json(path, record)
            state["successful_requests"] += 1
            state["input_tokens"] += usage
            state["reserved_input_equivalent"] -= reserve
            state["input_estimated_usd"] = state["input_tokens"] * rate / 1e6
            write_json(state_path, state)
            results.append({"id": row["id"], "annotation": annotation, "cache_hit": False})
        result = {"batch_id": value["batch_id"], "rows": results}
        write_json(output, result)
        output.with_suffix(output.suffix + ".gz").write_bytes(gzip.compress(schema.canonical(result), mtime=0))
        success = True
        return result
    except Exception as exc:
        failure = {"error_type": type(exc).__name__, "http_status": getattr(exc, "code", None)}
        if state:
            state.update(stopped=True, failure=failure)
            write_json(state_path, state)
            if "path" in locals() and path.is_file() and "record" in locals() and not record.get("completed"):
                record.update(completed=False, failure=failure,
                              reserve_retained=not record.get("successful_service_response", False))
                write_json(path, record)
        raise RuntimeError("Sentence annotation stopped; inspect the sanitized local ledger") from None
    finally:
        lock.unlink()
        if monitor:
            update_monitor(state, success, failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    completed, error_type = False, None
    try:
        run(args.input, args.output)
        completed = True
    except Exception as exc:
        error_type = type(exc).__name__
    state = strict_json((PHASE / "STATUS.json").read_bytes()) if (PHASE / "STATUS.json").is_file() else {}
    summary = {"completed": completed, "error_type": error_type,
               "successful_calls": state.get("successful_requests", 0),
               "attempted_calls": state.get("attempted_requests", 0),
               "input_tokens": state.get("input_tokens", 0),
               "input_estimated_usd": state.get("input_estimated_usd", 0.0),
               "committed_plus_reserved_input_estimated_usd": (
                   state.get("input_tokens", 0) + state.get("reserved_input_equivalent", 0)) * 0.042 / 1e6,
               "response_gzip": str(args.output) + ".gz" if completed else None}
    print(json.dumps(summary))
    if not completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
