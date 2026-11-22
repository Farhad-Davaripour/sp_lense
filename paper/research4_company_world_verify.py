"""Standard-library integrity and privacy checks for the review projection."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "company_world_review_manifest_v1.json"
ROWS = HERE / "company_world_review_rows_v1.jsonl"
SUMMARY = HERE / "company_world_review_summary_v1.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CASE_ID = re.compile(r"^CW-[0-9]{3}$")
SEED_ID = re.compile(r"^(E30|L12)-S[1-3]$")
FORBIDDEN_KEYS = {
    "case_id", "episode_id", "seed", "raw_dir", "source_dir", "conversation",
    "system_prompt", "user_prompt", "model_output", "reasoning", "response",
    "record_id", "company_id", "credential", "authorization",
}
FORBIDDEN_TEXT = (
    "C:\\", "C:/", "/Users/", ".local", "account_a", "account_b", "record_a",
    "record_b", "Harbor Press", "Meridian Booking", "Cedar Dispatch",
    "Juniper Retail", "Northstar Projects", "sk-", "api_key", "Bearer ",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_keys(value):
    if isinstance(value, dict):
        overlap = FORBIDDEN_KEYS.intersection(value)
        assert not overlap, f"forbidden keys: {sorted(overlap)}"
        for nested in value.values():
            inspect_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            inspect_keys(nested)


manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
for name, expected in manifest["files_sha256"].items():
    actual = sha(HERE / name)
    assert actual == expected, (name, actual, expected)

rows = [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines() if line]
summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
assert len(rows) == manifest["row_count"] == summary["rows"] == 42
assert Counter(row["cohort_id"] for row in rows) == Counter({
    "earlier_company_world_30": 30,
    "later_economic_12": 12,
})
assert len({row["anonymous_case_id"] for row in rows}) == 42
assert all(CASE_ID.fullmatch(row["anonymous_case_id"]) for row in rows)
assert all(SEED_ID.fullmatch(row["anonymous_seed_id"]) for row in rows)
assert all(row["observed_counts"]["company_tools_exposed"] is True for row in rows)

by_condition = defaultdict(list)
for row in rows:
    by_condition[f'{row["cohort_id"]}/{row["condition_id"]}'].append(row)
assert set(by_condition) == set(summary["conditions"])
assert summary["cohorts"] == dict(Counter(row["cohort_id"] for row in rows))
for condition, cases in by_condition.items():
    published = summary["conditions"][condition]
    assert published["runs"] == len(cases), condition
    assert published["complete"] == sum(case["final_outcome_known"] for case in cases), condition
    for key in (
        "source_inspection_calls", "company_operation_attempts",
        "company_operation_tool_successes", "successful_verified_effects",
        "payment_requests", "actual_payments_count", "actual_payments_usd",
    ):
        values = [case["observed_counts"][key] for case in cases]
        calculated = None if any(value is None for value in values) else sum(values)
        assert published[key] == calculated, (condition, key, published[key], calculated)

incomplete = [row for row in rows if not row["final_outcome_known"]]
assert len(incomplete) == 3
assert summary["incomplete_unknown_rows"] == [row["anonymous_case_id"] for row in incomplete]
assert {row["condition_id"] for row in incomplete} == {"replica_identity_base_operational_icl"}
for row in incomplete:
    assert row["completion_status"] == "incomplete_or_unknown"
    assert row["final_outcome_note"] == "unknown_due_to_incomplete_provider_or_harness"
    assert row["observed_counts"]["successful_verified_effects"] is None
    assert row["observed_counts"]["actual_payments_count"] is None
    assert row["observed_counts"]["actual_payments_usd"] is None

for row in rows:
    inspect_keys(row)
    for value in row["evidence_hashes"].values():
        assert HEX64.fullmatch(value), (row["anonymous_case_id"], value)
    if row["model_lineage"]["lineage_class"] == "hosted_provider_internal_lineage_unverified":
        assert row["model_lineage"]["provider_internal_lineage_verified"] is False
        assert row["model_lineage"]["base_manifest_sha256"] is None
    exposure = row["run_exposure"]
    assert isinstance(exposure["simulated_elapsed_hours_observed"], (int, float))
    assert isinstance(exposure["wall_elapsed_seconds"], (int, float))
    assert isinstance(exposure["observed_model_decisions"], int)
    assert isinstance(exposure["maximum_decisions"], int)
    assert 0 <= exposure["observed_model_decisions"] <= exposure["maximum_decisions"]
    assert isinstance(exposure["maximum_output_tokens"], int)
    assert exposure["thinking_enabled"] is True
    assert exposure["reasoning_effort"] == "low"
    assert exposure["sampling_parameters_recorded"] is False
    assert row["runtime_source_sha256"]
    assert all(HEX64.fullmatch(value) for value in row["runtime_source_sha256"].values())

by_conversation = defaultdict(list)
for row in rows:
    by_conversation[row["evidence_hashes"]["conversation_sha256"]].append(row)
duplicates = []
for digest, group in by_conversation.items():
    if len(group) > 1:
        duplicates.append({
            "anonymous_case_ids": sorted(row["anonymous_case_id"] for row in group),
            "byte_identical": True,
            "count": len(group),
            "distinct_world_plan_count": len({
                row["simulator_protocol"]["world_plan_sha256"] for row in group
            }),
            "group_id": "DC-" + digest[:12],
        })
assert sorted(summary["duplicate_display_conversation_groups"], key=lambda item: item["group_id"]) == sorted(duplicates, key=lambda item: item["group_id"])
assert summary["conversation_hash_definition"]

serialized = ROWS.read_text(encoding="utf-8") + SUMMARY.read_text(encoding="utf-8")
for forbidden in FORBIDDEN_TEXT:
    assert forbidden.casefold() not in serialized.casefold(), forbidden
assert not any(summary["privacy_scope"].values())

print(json.dumps({
    "status": "passed",
    "rows": len(rows),
    "incomplete_unknown_rows": len(incomplete),
    "rows_sha256": sha(ROWS),
    "summary_sha256": sha(SUMMARY),
    "unique_display_conversations": len(by_conversation),
    "privacy_checks": len(FORBIDDEN_KEYS) + len(FORBIDDEN_TEXT),
}, sort_keys=True))
