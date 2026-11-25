"""Verify the blinded company score projection without private run evidence.

The public catalog pins selected identities, statuses, frozen plans, RESULT
hashes and readable YAML hashes. The readable YAML does not contain the
private simulator's score or harm fields. This verifier pins the published
projection bytes and checks their shape and identity; run the sibling exporter
with --check-raw to verify score values against preserved RESULT.json bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
PROJECTION = Path(__file__).with_name("blinded_company_score_projection_v1.json")
SCHEMA = "research4.blinded-company-score-projection.v1"
STUDY_ID = "company_code_no_upfront_cue_v1"
FOLDER = "2026-11-25_13Z--company-code-no-upfront-cue-v1"
CATALOG_PATH = "catalog/results/company_code_no_upfront_cue_v1--readable-release-v1.json"
CATALOG_SHA256 = "c66937ee9347442251a3b96ed38bd849a8ffe536f45f9d19d0539162b92f112d"
CATALOG_SCHEMA = "research4.readable-new-study-release.v1"
ARMS = ("sp_lense_rank16_all64", "qwen38_27b_base", "terra", "deepseek_flash")
CASES = tuple(f"blinded_company_case_{i:02}" for i in range(1, 6))
ROSTER = {(arm, case) for arm in ARMS for case in CASES}
TOP_FIELDS = frozenset(("schema", "catalog", "rows"))
CATALOG_FIELDS = frozenset(("study_id", "folder", "catalog_path", "catalog_sha256"))
ROW_FIELDS = frozenset((
    "study_id", "arm", "case", "status", "outcome_valid",
    "security_outcome_score", "policy_harm_index", "responsible_outcome_score",
    "yaml_path", "yaml_sha256", "frozen_plan_sha256", "source_result_sha256",
))
HEX = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_PROJECTION_SHA256 = "12539d943ee191f43161367b9bd381d2331fae3d2e7ea9ae0020a6731a5f471e"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict:
    obj = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")),
    )
    require(type(obj) is dict, f"not a JSON object: {path.name}")
    return obj


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(value: object, allowed: frozenset[str], label: str) -> None:
    require(type(value) is dict and set(value) == allowed, f"{label}: unexpected or missing field")


def hex_hash(value: object, label: str) -> None:
    require(type(value) is str and HEX.fullmatch(value) is not None, f"{label}: invalid SHA-256")


def public_path(relative: str, prefix: str) -> Path:
    require(type(relative) is str and relative.startswith(prefix), "invalid public path prefix")
    parts = Path(relative).parts
    require(".." not in parts and not Path(relative).is_absolute(), "unsafe public path")
    path = ROOT / relative
    require(path.is_file(), f"missing public file: {relative}")
    return path


def expected_rows() -> tuple[dict, list[tuple[dict, dict]]]:
    path = public_path(CATALOG_PATH, "catalog/results/")
    require(sha256(path) == CATALOG_SHA256, "catalog hash mismatch")
    catalog = load_json(path)
    require(catalog.get("schema") == CATALOG_SCHEMA, "catalog schema mismatch")
    require(catalog.get("study_id") == STUDY_ID and catalog.get("folder") == FOLDER,
            "catalog study/folder mismatch")
    require(catalog.get("precall_count") == 0 and catalog.get("precall_dispositions") == [],
            "unexpected pre-call dispositions")
    rows = catalog.get("rows")
    require(type(rows) is list and len(rows) == catalog.get("row_count") == 20,
            "catalog row count mismatch")
    require(all(type(row) is dict for row in rows), "catalog row is not an object")
    require({(row.get("arm"), row.get("case")) for row in rows} == ROSTER,
            "catalog selected roster mismatch")
    identity_rows = []
    for row in rows:
        arm, case = row["arm"], row["case"]
        require(row.get("status") in ("complete", "incomplete"), "invalid catalog status")
        require(type(row.get("outcome_valid")) is bool, "invalid catalog outcome_valid")
        require((row["status"] == "complete") == row["outcome_valid"],
                "catalog status/validity mismatch")
        require(row.get("failure") is None and row.get("excluded_attempts") == [],
                "unexpected selected failure/exclusion")
        segments = row.get("segments")
        require(type(segments) is list and len(segments) == 1, "expected one primary segment")
        segment = segments[0]
        require(type(segment) is dict and segment.get("source") == "blinded_company_primary",
                "unexpected source segment")
        require(segment.get("run_id") == f"{case}--attempt-01", "unexpected attempt identity")
        require(segment.get("status") == row["status"] and segment.get("failure") is None,
                "segment status/failure mismatch")
        plan = segment.get("plan_sha256")
        result_hash = segment.get("source_sha256", {}).get("RESULT.json")
        hex_hash(plan, "frozen plan")
        hex_hash(result_hash, "source RESULT")
        require(plan == row.get("final_plan_sha256"), "final plan mismatch")
        conversation = row.get("conversation")
        require(type(conversation) is str and conversation.endswith(".yaml"), "conversation path invalid")
        yaml_path = f"results/{FOLDER}/{conversation}"
        yaml_hash = row.get("yaml_sha256")
        hex_hash(yaml_hash, "readable YAML")
        require(sha256(public_path(yaml_path, f"results/{FOLDER}/conversations/")) == yaml_hash,
                f"readable YAML hash mismatch: {arm} {case}")
        identity_rows.append((dict(
            study_id=STUDY_ID,
            arm=arm,
            case=case,
            status=row["status"],
            outcome_valid=row["outcome_valid"],
            yaml_path=yaml_path,
            yaml_sha256=yaml_hash,
            frozen_plan_sha256=plan,
            source_result_sha256=result_hash,
        ), row))
    return dict(study_id=STUDY_ID, folder=FOLDER,
                catalog_path=CATALOG_PATH, catalog_sha256=CATALOG_SHA256), identity_rows


def verify() -> None:
    require(PROJECTION.is_file(), "projection missing")
    hex_hash(EXPECTED_PROJECTION_SHA256, "expected projection")
    require(sha256(PROJECTION) == EXPECTED_PROJECTION_SHA256, "projection byte hash mismatch")
    obj = load_json(PROJECTION)
    fields(obj, TOP_FIELDS, "projection")
    require(obj["schema"] == SCHEMA, "projection schema mismatch")
    catalog_identity, sources = expected_rows()
    fields(obj["catalog"], CATALOG_FIELDS, "projection catalog")
    require(obj["catalog"] == catalog_identity, "projection catalog identity mismatch")
    require(type(obj["rows"]) is list and len(obj["rows"]) == len(sources) == 20,
            "projection row count mismatch")
    for row, (identity, _) in zip(obj["rows"], sources, strict=True):
        fields(row, ROW_FIELDS, "projection row")
        for key, value in identity.items():
            require(type(row[key]) is type(value) and row[key] == value,
                    f"projection {key} mismatch: {identity['arm']} {identity['case']}")
        scores = (row["security_outcome_score"], row["responsible_outcome_score"])
        harm = row["policy_harm_index"]
        if row["outcome_valid"]:
            require(all(type(value) in (int, float) and math.isfinite(value) for value in scores),
                    "valid outcome lacks finite scores")
            require(type(harm) is int and harm >= 0, "valid outcome lacks nonnegative harm index")
        else:
            require(scores == (None, None) and harm is None,
                    "incomplete outcome must have null scores and harm")
    print("PASS raw-free blinded-company score projection: 20 selected rows, catalog/YAML hashes and statuses")
    print("Trust boundary: readable YAML cannot recompute private-simulator scores or harm; "
          "the published values are byte-pinned here and require exporter --check-raw for source verification.")


if __name__ == "__main__":
    verify()
