"""Trusted local-only fresh Jev calls. No cache/dedup; separate bounded ledger."""
import argparse
import base64
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

SIBLING = Path(__file__).resolve().parents[1] / "matched_jev_v1"
sys.path.insert(0, str(SIBLING))
from local_service import official_transport, decode, strict_json, APPROVED_DOTENV
from jev_budget import validate_readiness, request_size_reserve, write_json, canonical_bytes
import concept_schema_v2 as schema

SCHEMA_SHA = "c5662b6fa6c6e95beebb0d41f5830e46dc776fc4ea2ad8d80169058d025ac515"


def run(input_path, output_path, phase_dir, readiness_path):
    phase, output = Path(phase_dir), Path(output_path)
    assert "onedrive" not in str(phase.resolve()).lower()
    assert not output.exists(), "Preserve prior returned responses"
    assert hashlib.sha256((SIBLING / "concept_schema_v2.py").read_bytes()).hexdigest() == SCHEMA_SHA
    readiness = json.loads(Path(readiness_path).read_bytes())
    rate = validate_readiness(readiness, schema.MODEL)
    assert rate == .042
    blob = Path(input_path).read_bytes()
    value = strict_json(gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob)
    assert set(value) == {"batch_id", "rows"} and 1 <= len(value["rows"]) <= 2
    assert all(set(row) == {"id", "history", "tools"} for row in value["rows"])
    phase.mkdir(parents=True, exist_ok=True)
    state_path = phase / "STATUS.json"
    if state_path.exists():
        state = json.loads(state_path.read_bytes())
    else:
        state = {"attempted_requests": 0, "successful_requests": 0, "input_tokens": 0,
                 "reserved_input_equivalent": 0, "stopped": False, "max_attempts": 2048,
                 "max_successful_requests": 2048, "max_estimated_input_usd": 1.,
                 "input_usd_per_million": rate, "input_estimated_usd": 0.,
                 "force_real_call_every_refresh": True, "deduplication": False,
                 "separate_from_original_matched_phase": True, "no_automatic_retries": True}
    assert not state["stopped"], "Pilot ledger stopped; preserve failure"
    results = []
    for row in value["rows"]:
        validate_readiness(readiness, schema.MODEL)
        request = schema.payload(row["history"], row["tools"])
        identity = schema.request_identity(request)
        reserve = request_size_reserve(request)["total_input_equivalent"]
        assert state["attempted_requests"] < 2048 and state["successful_requests"] < 2048
        assert (state["input_tokens"] + state["reserved_input_equivalent"] + reserve) * rate / 1e6 <= 1.
        number = state["attempted_requests"]
        path = phase / "calls" / (str(number).zfill(5) + ".json")
        assert not path.exists(), "Never overwrite a prior actual attempt"
        record = {"attempt_index": number, "request": request, "identity": identity,
                  "batch_id": value["batch_id"], "row_id": row["id"],
                  "preflight_reserve": reserve, "cache_hit": False,
                  "new_actual_service_transport_required": True}
        write_json(path, record)
        state["attempted_requests"] += 1
        state["reserved_input_equivalent"] += reserve
        write_json(state_path, state)
        tick = time.monotonic()
        try:
            _, raw = official_transport(request, APPROVED_DOTENV)
            response = strict_json(raw)
            decoded = decode(response, schema)
            usage = decoded["usage"]["input_tokens"]
            assert usage <= reserve, "Observed input exceeded conservative reserve"
            annotation = {"identity": identity, "request": request, "response": response,
                "decoded": decoded, "provider_response_raw_base64": base64.b64encode(raw).decode(),
                "provider_response_raw_sha256": hashlib.sha256(raw).hexdigest(),
                "schema_source_sha256": SCHEMA_SHA, "latency_seconds": time.monotonic() - tick,
                "usage": decoded["usage"], "input_estimated_usd": usage * rate / 1e6,
                "access_verified_by_pinned_response": True, "credential_in_payload_or_output": False,
                "training_targets_in_payload": False, "fresh_transport_attempt_index": number,
                "cache_hit": False, "actual_service_call": True}
            record.update(completed=True, annotation=annotation)
            write_json(path, record)
            state["successful_requests"] += 1
            state["input_tokens"] += usage
            state["reserved_input_equivalent"] -= reserve
            state["input_estimated_usd"] = state["input_tokens"] * rate / 1e6
            write_json(state_path, state)
            results.append({"id": row["id"], "annotation": annotation, "cache_hit": False})
        except Exception as exc:
            # Credential-containing provider bodies/errors never reach outputs.
            state.update(stopped=True, failure={"attempt_index": number, "error_type": type(exc).__name__,
                                               "http_status": getattr(exc, "code", None)})
            record.update(completed=False, failure=state["failure"], reserve_retained=True)
            write_json(path, record)
            write_json(state_path, state)
            raise RuntimeError("Fresh token annotation stopped; inspect sanitized ledger") from None
    result = {"batch_id": value["batch_id"], "rows": results}
    write_json(output, result)
    zipped = output.with_suffix(output.suffix + ".gz")
    assert not zipped.exists()
    zipped.write_bytes(gzip.compress(canonical_bytes(result), mtime=0))
    print(json.dumps({"completed": True, "rows": len(results), "budget": state}), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output", "phase", "readiness"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    run(a.input, a.output, a.phase, a.readiness)
