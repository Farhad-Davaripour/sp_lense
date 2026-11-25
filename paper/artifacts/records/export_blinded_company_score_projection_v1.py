"""Export or source-check blinded company scores from hash-pinned RESULT files.

--write builds an allowlisted public JSON projection from preserved ignored
evidence. --check-raw verifies every source RESULT hash and requires exact
projection byte equality. No raw path or private RESULT field is published.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "blinded_company_score_verifier", HERE / "verify_blinded_company_score_projection_v1.py"
)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
DEFAULT_RAW_ROOT = verifier.ROOT / ".local" / "2026-11-25_06--company-code-no-upfront-cue-v1-run" / "raw"


def finite_score(value: object, label: str) -> int | float:
    verifier.require(type(value) in (int, float) and math.isfinite(value), f"missing/nonfinite {label}")
    return value


def build(raw_root: Path) -> bytes:
    catalog, sources = verifier.expected_rows()
    projected = []
    for identity, catalog_row in sources:
        segment = catalog_row["segments"][0]
        run_id = segment["run_id"]
        arm = identity["arm"]
        case = identity["case"]
        result_path = raw_root / arm / run_id / "RESULT.json"
        verifier.require(result_path.is_file(), f"missing local RESULT: {arm} {case}")
        verifier.require(verifier.sha256(result_path) == identity["source_result_sha256"],
                         f"source RESULT hash mismatch: {arm} {case}")
        result = verifier.load_json(result_path)
        score = result.get("score")
        verifier.require(type(score) is dict, f"source score missing: {arm} {case}")
        verifier.require(result.get("status") == identity["status"], f"source status mismatch: {arm} {case}")
        verifier.require(result.get("failure") == catalog_row["failure"], f"source failure mismatch: {arm} {case}")
        verifier.require(result.get("finished_utc") == catalog_row["finished_utc"],
                         f"source finish time mismatch: {arm} {case}")
        verifier.require(score.get("version") == verifier.STUDY_ID,
                         f"source score version mismatch: {arm} {case}")
        verifier.require(type(score.get("outcome_valid")) is bool and
                         score["outcome_valid"] == identity["outcome_valid"],
                         f"source outcome validity mismatch: {arm} {case}")
        verifier.require(score.get("termination_reason") == catalog_row["termination_reason"] and
                         score.get("elapsed_hours") == catalog_row["elapsed_hours"],
                         f"source terminal state mismatch: {arm} {case}")
        if identity["outcome_valid"]:
            verifier.require(score.get("score_status") == "final" and score.get("status") == "final",
                             f"source score not final: {arm} {case}")
            security = finite_score(score.get("security_outcome_score"), "security score")
            responsible = finite_score(score.get("responsible_outcome_score"), "responsible score")
            harm = score.get("policy_harm_index")
            verifier.require(type(harm) is int and harm >= 0, f"source harm invalid: {arm} {case}")
        else:
            verifier.require(score.get("score_status") == "incomplete", f"source score not incomplete: {arm} {case}")
            for key in ("security_outcome_score", "responsible_outcome_score", "policy_harm_index"):
                verifier.require(score.get(key) is None, f"incomplete source has {key}: {arm} {case}")
            security = responsible = harm = None
        projected.append(dict(
            study_id=verifier.STUDY_ID,
            arm=arm,
            case=case,
            status=identity["status"],
            outcome_valid=identity["outcome_valid"],
            security_outcome_score=security,
            policy_harm_index=harm,
            responsible_outcome_score=responsible,
            yaml_path=identity["yaml_path"],
            yaml_sha256=identity["yaml_sha256"],
            frozen_plan_sha256=identity["frozen_plan_sha256"],
            source_result_sha256=identity["source_result_sha256"],
        ))
    obj = {"schema": verifier.SCHEMA, "catalog": catalog, "rows": projected}
    return (json.dumps(obj, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate projection from preserved raw evidence")
    mode.add_argument("--check-raw", action="store_true", help="compare exact projection bytes with raw evidence")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT,
                        help="ignored local raw run folder")
    args = parser.parse_args()
    content = build(args.raw_root)
    if args.write:
        verifier.PROJECTION.write_bytes(content)
        print(f"WROTE {verifier.PROJECTION} SHA-256 {verifier.sha256(verifier.PROJECTION)}")
    else:
        verifier.require(verifier.PROJECTION.is_file(), "projection missing")
        verifier.require(verifier.PROJECTION.read_bytes() == content,
                         "projection differs from hash-pinned local RESULT values")
        print("PASS source-backed blinded-company score projection: 20 selected RESULT hashes and exact JSON bytes")


if __name__ == "__main__":
    main()
