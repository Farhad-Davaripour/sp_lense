"""Trusted local annotation only. Root injects transport; no key/Qwen access."""
import copy
import hashlib
import json
from pathlib import Path
from pilot_fixtures import CAMPAIGN, canonical, digest, load_preparation

READINESS_FIELDS = (
    "verified", "endpoint", "model", "official_docs_verified", "versioned_model_id_documented",
    "official_models_doc_url", "official_api_doc_url", "supported_question_types",
    "multiple_questions_verified", "published_request_limit",
    "published_state_longest_question_limit", "input_usd_per_million",
    "output_usd_per_million", "checked_at_utc")


def safe_readiness(record):
    return {name: copy.deepcopy(record[name]) for name in READINESS_FIELDS}


def load_frozen(fixture_dir):
    directory = Path(fixture_dir)
    frozen = json.loads((directory/"PILOT_FREEZE.json").read_text(encoding="utf-8"))
    for relative, expected in frozen["source_sha256"].items():
        if hashlib.sha256((CAMPAIGN/relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Pilot source changed after input freeze: "+relative)
    raw = (directory/"BOUNDARY_FIXTURES.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == frozen["boundary_file_sha256"]
    payload = json.loads(raw)
    without_hash = {key: value for key, value in payload.items() if key != "sha256_canonical_without_hash"}
    assert digest(without_hash) == payload["sha256_canonical_without_hash"] == frozen["boundary_payload_sha256"]
    assert len(payload["rows"]) == frozen["local_api_requests_admitted"] == 8
    return payload, frozen


def run(fixture_dir, output_dir, readiness, transport):
    """Annotate exactly eight frozen raw boundaries using root-owned transport.

    No credential/file/env discovery and no model loading or generation occurs.
    Future actual Qwen histories are outside this fixed local annotation stage.
    """
    schema, client_module = load_preparation()
    payload, frozen = load_frozen(fixture_dir)
    ready = safe_readiness(readiness)
    rate = client_module.validate_readiness(ready)
    estimated = sum(row["preflight_reserve"]["total_input_equivalent"] for row in payload["rows"])*rate/1e6
    if estimated > 0.05:
        raise RuntimeError("Eight-boundary preflight estimate exceeds global pilot input cap")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    # Constructing the client and writing provenance do not invoke the transport.
    client = client_module.BoundedClient(output/"api_cache", ready, transport,
                                         max_requests=128, max_estimated_input_usd=0.05, max_seconds=1800)
    plan = {"scope": "eight frozen supplied boundaries only", "input_freeze": frozen,
            "readiness": ready, "local_max_requests": 8, "estimated_input_reserve_usd": estimated,
            "global_max_requests": 128, "global_input_estimate_cap_usd": 0.05,
            "qwen_inference_run": False, "later_histories_annotated": False}
    client_module.write_json(output/"ANNOTATION_PLAN.json", plan)
    rows = []
    try:
        for row in payload["rows"]:
            request = schema.payload(row["history"], row["tools"])
            if request != row["request"] or schema.request_identity(request) != row["request_identity"]:
                raise RuntimeError("Frozen visible input identity mismatch")
            decoded, provenance = client.call(row["history"], row["tools"])
            aided = schema.augmented_history(row["history"], decoded)
            assert aided[:-1] == row["history"] and len(aided) == len(row["history"])+1
            annotated = {"id": row["id"], "boundary": row["boundary"], "cohort_index": row["cohort_index"],
                         "source_case_id": row["source_case_id"], "request_identity": row["request_identity"],
                         "decoded": decoded, "compact_aid": schema.compact_aid(decoded),
                         "plain_history": copy.deepcopy(row["history"]),
                         "aided_history_at_boundary": aided, "tools": copy.deepcopy(row["tools"]),
                         "cache_provenance": provenance, "diagnostic_only": True,
                         "annotation_scope": "boundary_only",
                         "future_annotation_reuse_permitted": False}
            rows.append(annotated)
            client_module.write_json(output/(row["id"]+".json"), annotated)
            client_module.write_json(output/"STATUS.json",
                                     {"stage": "annotating_boundaries", "completed_boundary_states": len(rows),
                                      "total": 8, "api": client.status(), "qwen_inference_run": False})
        result = {"completed": True, "scope": "boundary_only", "states": rows,
                  "api": client.status(), "estimated_preflight_input_usd": estimated,
                  "selected_model": payload["selected_model"], "qwen_inference_run": False,
                  "later_actual_histories_annotated": False,
                  "global_remaining_requests": 128-client.attempted,
                  "global_remaining_input_estimate_usd": 0.05-client.input_tokens*rate/1e6}
        client_module.write_json(output/"ANNOTATIONS.json", result)
        client_module.write_json(output/"STATUS.json",
                                 {"stage": "completed_boundary_annotations", "completed_boundary_states": 8,
                                  "api": client.status(), "qwen_inference_run": False})
        return result
    except Exception as error:
        failure = {"stage": "stopped_boundary_annotation", "error_type": type(error).__name__,
                   "completed_boundary_states": len(rows), "api": client.status(),
                   "no_automatic_retry": True, "qwen_inference_run": False}
        # No exception text, request headers, credential, or provider error body.
        client_module.write_json(output/"FAILURE.json", failure)
        raise RuntimeError(json.dumps(failure)) from None
