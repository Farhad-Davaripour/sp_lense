"""Freeze eight exact raw boundary states; never manufacture future histories."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREPARATION = HERE.parent
CAMPAIGN = PREPARATION.parent
LEDGER = CAMPAIGN/"operand_ledger_v1"
LEDGER_FIXTURE_SHA256 = "8aec142c330bef312f5a4b12af52efd2b934311c8ec98ef17fda0de30889db5e"
SOURCE_CASES_SHA256 = "45a5eab85078870b34c0b3912bbc72840fdde217a97816c70936b987c6f0a19b"
BOUNDARIES = ("all_fragments_observed", "old_answer_rejected")
COHORT_ORDER = ("s72111_one_step_50_self_unfinished", "s72111_one_step_51_self_unfinished",
                "s72111_ordered_ticket_50_self_unfinished", "s72111_ordered_ticket_51_self_unfinished")
COVERAGE_PIN = {
    "name": "coverage", "campaign": "qwen38_H2_replay_diagnostics_20261002T022014Z_f6204735",
    "relative": "replay_coverage_stream/coverage/checkpoints/update_56",
    "weights_sha256": "4caa108321f89e69fc11b8cddb9e145ed15e3bca90c45623dde87b759a8a24e3",
    "config_sha256": "636f8257126c594dfb534a71c8ed5f65206e05b47f4974d6f54ad5d9782094c7",
    "base_repository": "Qwen/Qwen3.8-27B",
    "base_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "packed_base_sha256": "ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755"
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_preparation():
    # Fixed trusted preparation; no environment or credential loading.
    import sys
    path = str(PREPARATION)
    if path not in sys.path:
        sys.path.insert(0, path)
    import concept_schema
    import jev_client
    for module, name in ((concept_schema, "concept_schema.py"), (jev_client, "jev_client.py")):
        assert Path(module.__file__).resolve() == (PREPARATION/name).resolve()
    return concept_schema, jev_client


def build(ledger_fixture_path=None):
    schema, client = load_preparation()
    path = Path(ledger_fixture_path) if ledger_fixture_path else LEDGER/"data_frozen/FIXTURES.json"
    raw = path.read_bytes()
    ledger = json.loads(raw)
    without_hash = {key: value for key, value in ledger.items() if key != "sha256_canonical_without_hash"}
    assert digest(without_hash) == ledger["sha256_canonical_without_hash"] == LEDGER_FIXTURE_SHA256
    assert ledger["source_file_sha256"] == SOURCE_CASES_SHA256
    assert tuple(ledger["cohort_order"]) == COHORT_ORDER
    # Tool definitions are taken from the exact frozen ledger helper text.
    packed = (LEDGER/"source_bundle.json.gz").read_bytes()
    expected_bundle = (LEDGER/"SOURCE_BUNDLE.sha256").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(packed).hexdigest() == expected_bundle
    source_bundle = json.loads(__import__("gzip").decompress(packed))
    source_freeze = json.loads(source_bundle["files"]["SOURCE_FREEZE.json"])
    assert hashlib.sha256(source_bundle["files"]["world.py"].encode("utf-8")).hexdigest() == source_freeze["sha256"]["world.py"]
    import ast
    tree = ast.parse(source_bundle["files"]["world.py"])
    # Executing this trusted pure module defines only inert JSON/re simulator
    # code; no torch/model/network dependency is imported.
    namespace = {"__name__": "frozen_pilot_world"}
    exec(compile(tree, "frozen_ledger_world.py", "exec"), namespace)
    tools = namespace["TOOLS"]
    selected = [item for item in ledger["fixtures"] if item["spec"]["arm"] == "original_observation"]
    assert len(selected) == 8
    rows = []
    for boundary in BOUNDARIES:
        group = [item for item in selected if item["spec"]["boundary"] == boundary]
        assert [item["spec"]["source_case_id"] for item in group] == list(COHORT_ORDER)
        for item in group:
            spec = item["spec"]
            history = copy.deepcopy(item["messages"])
            assert history[0] == {"role": "system", "content": namespace["SYSTEM"]}
            # The exact original observation, not the successful ledger aid.
            assert all("current_fragments" not in json.loads(message["content"])
                       for message in history if message["role"] == "tool")
            assert spec["generation_turn_cap"] == 9 and spec["generation_token_cap"] == 1536
            request = schema.payload(history, tools)
            identity = schema.request_identity(request)
            rows.append({"id": "jev_boundary_"+spec["source_case_id"]+"_"+boundary,
                         "boundary": boundary, "cohort_index": spec["cohort_index"],
                         "source_case_id": spec["source_case_id"],
                         "source_fixture_id": item["id"],
                         "history": history, "tools": copy.deepcopy(tools),
                         "request": request, "request_identity": identity,
                         "preflight_reserve": client.request_size_reserve(request),
                         "starting_state_sha256": item["starting_state_sha256"],
                         # Private runtime reconstruction/scoring only. The API
                         # receives request, never this metadata/spec.
                         "runtime_spec": copy.deepcopy(spec),
                         "diagnostic_only": True, "teacher_future_annotations": False})
    assert len({row["request_identity"]["request_sha256"] for row in rows}) == 8
    payload = {"pilot_version": "boundary_annotations_v1", "selected_model": COVERAGE_PIN,
               "ledger_fixture_sha256": LEDGER_FIXTURE_SHA256,
               "source_case_file_sha256": SOURCE_CASES_SHA256,
               "ledger_file_sha256": hashlib.sha256(raw).hexdigest(),
               "ledger_source_bundle_sha256": expected_bundle,
               "world_source_sha256": source_freeze["sha256"]["world.py"],
               "schema_version": schema.SCHEMA_VERSION, "questions_sha256": schema.digest(schema.QUESTIONS),
               "cohort_order": list(COHORT_ORDER), "boundaries": list(BOUNDARIES),
               "caps": {"future_turns": 9, "future_generated_tokens": 1536, "per_turn_tokens": 256},
               "local_annotation_scope": "eight supplied boundary states only",
               "local_max_requests": 8, "global_pilot_max_requests": 128,
               "global_estimated_input_usd_cap": 0.05,
               "requires_new_annotations_for_later_actual_histories": True,
               "rows": rows}
    payload["sha256_canonical_without_hash"] = digest(payload)
    return payload


def write_freeze(output_dir, ledger_fixture_path=None):
    payload = build(ledger_fixture_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/"BOUNDARY_FIXTURES.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8", newline="\n")
    paths = [PREPARATION/"concept_schema.py", PREPARATION/"jev_client.py",
             HERE/"pilot_fixtures.py", HERE/"annotate_boundaries.py", HERE/"PROTOCOL.md"]
    source = {path.relative_to(CAMPAIGN).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    freeze = {"frozen_before_any_api_call": True, "pilot_version": payload["pilot_version"],
              "source_sha256": source,
              "boundary_file_sha256": hashlib.sha256((output/"BOUNDARY_FIXTURES.json").read_bytes()).hexdigest(),
              "boundary_payload_sha256": payload["sha256_canonical_without_hash"],
              "questions_sha256": payload["questions_sha256"], "schema_version": payload["schema_version"],
              "request_sha256": [row["request_identity"]["request_sha256"] for row in payload["rows"]],
              "selected_model": COVERAGE_PIN, "local_api_requests_admitted": 8,
              "global_pilot_max_requests": 128, "global_estimated_input_usd_cap": 0.05,
              "api_calls_executed_at_freeze": 0, "qwen_model_queries_at_freeze": 0}
    (output/"PILOT_FREEZE.json").write_text(json.dumps(freeze, indent=2)+"\n", encoding="utf-8", newline="\n")
    return freeze


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ledger-fixtures")
    args = parser.parse_args()
    frozen = write_freeze(args.output_dir, args.ledger_fixtures)
    print(json.dumps({"requests": 8, "boundary_payload_sha256": frozen["boundary_payload_sha256"],
                      "model_weights_sha256": COVERAGE_PIN["weights_sha256"], "api_calls": 0}))
