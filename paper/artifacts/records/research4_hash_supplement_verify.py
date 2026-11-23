"""Standard-library verification for the hash/action-only supplement."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
DATA = HERE / "infrastructure_hash_action_supplement_v1.json"
MANIFEST = HERE / "infrastructure_hash_action_manifest_v1.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_KEYS = {
    "messages", "body", "text", "prompt", "response", "reasoning", "plan",
    "raw_dir", "source_dir", "path", "credential", "authorization",
}
FORBIDDEN_TEXT = (
    "C:\\", "C:/", "/Users/", ".local", "Bearer ", "sk-", "api_key",
    "</think>",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def keys(value):
    if isinstance(value, dict):
        overlap = FORBIDDEN_KEYS.intersection(value)
        assert not overlap, sorted(overlap)
        for nested in value.values():
            keys(nested)
    elif isinstance(value, list):
        for nested in value:
            keys(nested)


manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
for name, expected in manifest["files_sha256"].items():
    assert sha(HERE / name) == expected, name

value = json.loads(DATA.read_text(encoding="utf-8"))
keys(value)
serialized = DATA.read_text(encoding="utf-8")
for forbidden in FORBIDDEN_TEXT:
    assert forbidden.casefold() not in serialized.casefold(), forbidden

game = value["original_40_run_game"]
assert len(game["runs"]) == 40
assert Counter(run["arm"] for run in game["runs"]) == Counter({
    "sp_lense_rank16_all64": 10,
    "qwen38_27b_base": 10,
    "terra": 10,
    "deepseek_v41_flash": 10,
})
assert game["checks"]["all_40_initial_message_arrays_equal"] is True
assert all(row["distinct_digests"] == 1 and row["runs"] == 10
           for row in game["checks"]["per_arm_initial_digest_counts"].values())
assert game["checks"]["tuned_first_four_equal_across_ten"] is True
assert game["checks"]["tuned_distinct_decision_four_message_arrays"] == 2
assert sorted(game["checks"]["tuned_decision_four_message_digest_counts"].values()) == [1, 9]
assert game["checks"]["tuned_copy_rejection_codes"] == {"region_has_no_online_gpu": 10}
for run in game["runs"]:
    assert run["message_array_digests"]
    assert all(HEX64.fullmatch(row["message_array_sha256"]) for row in run["message_array_digests"])
    for row in run["normalized_actions"]:
        canonical = json.dumps({"action": row["action"], "arguments": row["arguments"]},
                               sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert sha_bytes(canonical.encode("utf-8")) == row["normalized_action_sha256"]
    assert all(HEX64.fullmatch(digest) for digest in run["evidence_sha256"].values())

initial_digests = {run["message_array_digests"][0]["message_array_sha256"]
                   for run in game["runs"]}
assert len(initial_digests) == 1
assert game["checks"]["initial_message_array_sha256"] == next(iter(initial_digests))
by_arm = defaultdict(list)
for run in game["runs"]:
    by_arm[run["arm"]].append(run)
calculated_arm_counts = {
    arm: {"runs": len(group), "distinct_digests": len({
        run["message_array_digests"][0]["message_array_sha256"] for run in group
    })}
    for arm, group in by_arm.items()
}
assert game["checks"]["per_arm_initial_digest_counts"] == calculated_arm_counts
tuned = by_arm["sp_lense_rank16_all64"]
first_four = [[{"action": row["action"], "arguments": row["arguments"]}
               for row in run["normalized_actions"][:4]] for run in tuned]
assert len({json.dumps(rows, sort_keys=True) for rows in first_four}) == 1
assert game["checks"]["tuned_common_first_four_normalized_actions"] == first_four[0]
decision_four = Counter(next(row["message_array_sha256"] for row in run["message_array_digests"]
                             if row["decision"] == 4) for run in tuned)
assert game["checks"]["tuned_decision_four_message_digest_counts"] == dict(decision_four)
assert game["checks"]["tuned_distinct_decision_four_message_arrays"] == len(decision_four)
rejections = Counter(run["normalized_actions"][1]["rejection_code"] for run in tuned)
assert game["checks"]["tuned_copy_rejection_codes"] == dict(rejections)

pilot = value["matched_rank_pilot"]
assert len(pilot["runs"]) == 12
assert len(pilot["rank8_rank16_equality"]) == 4
assert all(row["normalized_actions_equal"] and row["message_digest_sequences_equal"]
           for row in pilot["rank8_rank16_equality"])
for run in pilot["runs"]:
    assert all(HEX64.fullmatch(row["message_array_sha256"]) for row in run["message_array_digests"])
    for row in run["normalized_actions"]:
        canonical = json.dumps({"action": row["action"], "arguments": row["arguments"]},
                               sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert sha_bytes(canonical.encode("utf-8")) == row["normalized_action_sha256"]
    assert all(HEX64.fullmatch(digest) for digest in run["evidence_sha256"].values())
pilot_by_case_arm = {(run["case"], run["arm"]): run for run in pilot["runs"]}
for claim in pilot["rank8_rank16_equality"]:
    left = pilot_by_case_arm[(claim["case"], "rank8")]
    right = pilot_by_case_arm[(claim["case"], "rank16")]
    assert claim["normalized_actions_equal"] == (
        [row["normalized_action_sha256"] for row in left["normalized_actions"]] ==
        [row["normalized_action_sha256"] for row in right["normalized_actions"]]
    )
    assert claim["message_digest_sequences_equal"] == (
        [row["message_array_sha256"] for row in left["message_array_digests"]] ==
        [row["message_array_sha256"] for row in right["message_array_digests"]]
    )

sampling = value["company_world_sampling_audit"]
profile = {"generation_seed": 20261108, "temperature": 1.0, "top_p": 0.95,
           "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}
assert sampling["earlier_company_world_30"]["local_qwen_profile"] == profile
assert sampling["later_economic_12"]["local_qwen_profile"] == profile
for provider in ("hosted_terra", "hosted_deepseek"):
    row = sampling["earlier_company_world_30"][provider]
    assert all(row[key] is None for key in ("generation_seed", "temperature", "top_p", "top_k"))

assert not any(value["privacy_scope"].values())
print(json.dumps({
    "status": "passed",
    "data_sha256": sha(DATA),
    "game_runs": 40,
    "pilot_runs": 12,
    "privacy_checks": len(FORBIDDEN_KEYS) + len(FORBIDDEN_TEXT),
}, sort_keys=True))
