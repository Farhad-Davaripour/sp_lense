"""Raw-free verifier for the new-study score and harm projection.

The selected catalogs and readable conversations establish row identity, status,
and file hashes. Security scores and policy harm are private-simulator RESULT
fields: this verifier pins their published projection bytes but cannot derive
their values from readable YAML alone. Run the sibling exporter --check-raw on
preserved local evidence for that separate source-backed check.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
PROJECTION = Path(__file__).with_name("new_study_score_projection_v1.json")
SCHEMA = "research4.new-study-score-projection.v1"
CATALOG_SCHEMA = "research4.readable-new-study-release.v1"
ARMS = ("sp_lense_rank16_all64", "qwen38_27b_base", "terra", "deepseek_flash")
CATALOGS = (
    (
        "infrastructure_split_gpu_layout_v3",
        "2026-11-23_17Z--split-gpu-layout-v3",
        "catalog/results/infrastructure_split_gpu_layout_v3--readable-release-v1.json",
        "7e635f1efffa3033f6658d9a06c38b2d59dfcc34dd3392b0aab251be94c080a6",
        20,
    ),
    (
        "company_code_access_v1",
        "2026-11-23_17Z--company-code-access-v1",
        "catalog/results/company_code_access_v1--readable-release-v1.json",
        "239ecb28368423288810fbe6f0a35a997b0ecfff4aaa1541fa71bbdafd3a2f3e",
        20,
    ),
    (
        "infrastructure_split_gpu_layout_v3_format_rerun_supplement_v1",
        "2026-11-25_03Z--split-gpu-format-rerun-supplement-v1",
        "catalog/results/infrastructure_split_gpu_layout_v3_format_rerun_supplement_v1--readable-release-v1.json",
        "b1bc5a53a51c804e0a9897608432de6e6d3d83f9db20b87dab2fb600a18aaed9",
        2,
    ),
    (
        "infrastructure_split_gpu_new_seed_supplement_v1",
        "2026-11-25_04Z--split-gpu-new-seed-supplement-v1",
        "catalog/results/infrastructure_split_gpu_new_seed_supplement_v1--readable-release-v1.json",
        "c240dcbc3e4ef3dcb84341c705f0ae848265a99e9cd6d6240d886ec0fc8897b1",
        8,
    ),
)
SPLIT_CASES = ("split_w3", "split_w2_e1", "split_w1_e2", "split_w2_n1", "split_w1_e1_n1")
COMPANY_CASES = tuple(f"company_case_{i:02}" for i in range(1, 6))
NEW_SEED_CASES = ("newseed_split_w3", "newseed_split_w1_e2")
EXPECTED_ROSTERS = (
    {(arm, case) for arm in ARMS for case in SPLIT_CASES},
    {(arm, case) for arm in ARMS for case in COMPANY_CASES},
    {("sp_lense_rank16_all64", "split_w1_e2"), ("qwen38_27b_base", "split_w3")},
    {(arm, case) for arm in ARMS for case in NEW_SEED_CASES},
)
TOP_FIELDS = frozenset(("schema", "catalogs", "rows"))
CATALOG_FIELDS = frozenset(("study_id", "folder", "catalog_path", "catalog_sha256"))
ROW_FIELDS = frozenset((
    "study_id", "folder", "arm", "case", "status", "outcome_valid",
    "security_outcome_score", "policy_harm_index", "yaml_path", "yaml_sha256",
    "frozen_plan_sha256", "source_result_sha256",
))
HEX = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_PROJECTION_SHA256 = "23225c089883cf305694d57dcaa4c73d032aa601777975b040c4e1f6ca7aa8d2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")),
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(value: object, keys: frozenset[str], label: str) -> None:
    require(type(value) is dict and set(value) == keys, f"{label}: unexpected or missing field")


def hex_hash(value: object, label: str) -> None:
    require(type(value) is str and HEX.fullmatch(value) is not None, f"{label}: invalid SHA-256")


def public_path(relative: str, prefix: str) -> Path:
    require(type(relative) is str and relative.startswith(prefix), "invalid public path prefix")
    parts = Path(relative).parts
    require(".." not in parts and not Path(relative).is_absolute(), "unsafe public path")
    path = ROOT / relative
    require(path.is_file(), f"missing public file: {relative}")
    return path


def expected_rows() -> tuple[list[dict], list[tuple[dict, dict]]]:
    catalog_projection = []
    source_rows = []
    for index, (study_id, folder, name, digest, count) in enumerate(CATALOGS):
        path = public_path(name, "catalog/results/")
        require(sha256(path) == digest, f"catalog hash mismatch: {name}")
        catalog = load_json(path)
        require(catalog.get("schema") == CATALOG_SCHEMA, f"catalog schema: {name}")
        require(catalog.get("study_id") == study_id and catalog.get("folder") == folder, f"catalog identity: {name}")
        rows = catalog.get("rows")
        require(type(rows) is list and len(rows) == count == catalog.get("row_count"), f"catalog row count: {name}")
        roster = {(row.get("arm"), row.get("case")) for row in rows if type(row) is dict}
        require(len(roster) == count and roster == EXPECTED_ROSTERS[index], f"selected roster: {name}")
        catalog_projection.append(dict(study_id=study_id, folder=folder, catalog_path=name, catalog_sha256=digest))
        for row in rows:
            require(row.get("status") in ("complete", "incomplete"), "catalog status")
            require(type(row.get("outcome_valid")) is bool, "catalog outcome_valid")
            require((row["status"] == "complete") == row["outcome_valid"], "catalog status/validity mismatch")
            segments = row.get("segments")
            require(type(segments) is list and bool(segments), "catalog score source missing")
            final = segments[-1]
            require(final.get("status") == row["status"], "final segment status mismatch")
            plan = final.get("plan_sha256")
            result_hash = final.get("source_sha256", {}).get("RESULT.json")
            hex_hash(plan, "frozen plan")
            hex_hash(result_hash, "source RESULT")
            conversation = row.get("conversation")
            require(type(conversation) is str and conversation.endswith(".yaml"), "catalog conversation path")
            yaml_path = f"results/{folder}/{conversation}"
            yaml_hash = row.get("yaml_sha256")
            hex_hash(yaml_hash, "YAML")
            require(sha256(public_path(yaml_path, f"results/{folder}/conversations/")) == yaml_hash,
                    f"conversation hash mismatch: {yaml_path}")
            source_rows.append((dict(
                study_id=study_id,
                folder=folder,
                arm=row["arm"],
                case=row["case"],
                status=row["status"],
                outcome_valid=row["outcome_valid"],
                yaml_path=yaml_path,
                yaml_sha256=yaml_hash,
                frozen_plan_sha256=plan,
                source_result_sha256=result_hash,
            ), row))
    require(len(source_rows) == 50, "expected 50 selected reader rows")
    return catalog_projection, source_rows


def verify() -> None:
    require(PROJECTION.is_file(), "projection missing")
    hex_hash(EXPECTED_PROJECTION_SHA256, "expected projection")
    require(sha256(PROJECTION) == EXPECTED_PROJECTION_SHA256, "projection byte hash mismatch")
    obj = load_json(PROJECTION)
    fields(obj, TOP_FIELDS, "projection")
    require(obj["schema"] == SCHEMA, "projection schema")
    catalog_rows, sources = expected_rows()
    require(type(obj["catalogs"]) is list and len(obj["catalogs"]) == len(CATALOGS), "projection catalog count")
    for catalog in obj["catalogs"]:
        fields(catalog, CATALOG_FIELDS, "projection catalog")
    require(obj["catalogs"] == catalog_rows, "projection catalog mismatch")
    require(type(obj["rows"]) is list and len(obj["rows"]) == len(sources), "projection row count")
    status_counts = Counter()
    for projected, (identity, _) in zip(obj["rows"], sources, strict=True):
        fields(projected, ROW_FIELDS, "projection row")
        for key, value in identity.items():
            require(projected[key] == value and type(projected[key]) is type(value), f"row {key} mismatch")
        score = projected["security_outcome_score"]
        harm = projected["policy_harm_index"]
        if projected["outcome_valid"]:
            require(type(score) in (int, float) and math.isfinite(score), "final security score missing/nonfinite")
            if projected["study_id"] == "company_code_access_v1":
                require(type(harm) is int and harm >= 0, "final policy harm missing/invalid")
            else:
                require(harm is None, "policy harm is not defined in infrastructure studies")
        else:
            require(score is None and harm is None, "incomplete outcomes must have null scores")
        status_counts[(projected["study_id"], projected["status"])] += 1
    print(f"PASS raw-free new-study score projection: {len(sources)} selected rows, {dict(status_counts)}")
    print("Trust boundary: readable YAML cannot recompute private-simulator security or harm scores; "
          "the published values are byte-pinned here and require exporter --check-raw for source verification.")


if __name__ == "__main__":
    verify()
