"""Export or source-check selected scores from hash-pinned local RESULT files.

--write regenerates the public allowlisted JSON. --check-raw independently
recomputes it from preserved ignored .local evidence and requires byte equality.
Neither mode publishes raw paths or private RESULT fields.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("score_projection_verifier", HERE / "verify_new_study_score_projection_v1.py")
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)

SOURCE_ROOTS = {
    "split_primary": ("2026-11-23_10--split-gpu-layout-v3-run", True),
    "company_primary": ("2026-11-23_10--company-code-access-v1-run", True),
    "split_terra_suffix": ("2026-11-23_11--split-gpu-terra-continuation", False),
    "company_suffix": ("2026-11-23_23--company-code-context-continuation-v1-run", False),
    "split_format_rerun": ("2026-11-23_21--split-gpu-format-rerun-supplement-v1-run", True),
    "split_new_seed": ("2026-11-24_22--split-gpu-new-seed-supplement-v1-run", True),
}


def build(raw_root: Path) -> bytes:
    catalogs, sources = verifier.expected_rows()
    projected = []
    for identity, catalog_row in sources:
        segment = catalog_row["segments"][-1]
        source_name = segment["source"]
        verifier.require(source_name in SOURCE_ROOTS, f"unknown raw source: {source_name}")
        base_name, arm_subdir = SOURCE_ROOTS[source_name]
        run_id = segment["run_id"]
        verifier.require(type(run_id) is str and "/" not in run_id and "\\" not in run_id and ".." not in run_id,
                         "unsafe raw run ID")
        result_dir = raw_root / base_name / "raw"
        if arm_subdir:
            result_dir /= identity["arm"]
        result_path = result_dir / run_id / "RESULT.json"
        verifier.require(result_path.is_file(), f"missing local result for {source_name} {identity['arm']} {run_id}")
        verifier.require(verifier.sha256(result_path) == identity["source_result_sha256"],
                         f"local RESULT hash mismatch: {source_name} {identity['arm']} {run_id}")
        result = verifier.load_json(result_path)
        score = result.get("score")
        verifier.require(type(score) is dict, "raw score object missing")
        verifier.require(result.get("status") == identity["status"], "raw result status differs from catalog")
        verifier.require(type(score.get("outcome_valid")) is bool and
                         score["outcome_valid"] == identity["outcome_valid"], "raw outcome validity differs")
        if identity["outcome_valid"]:
            verifier.require(score.get("score_status") == "final", "valid outcome lacks final score status")
            security = score.get("security_outcome_score")
            verifier.require(type(security) in (int, float) and math.isfinite(security), "final security score invalid")
            if identity["study_id"] == "company_code_access_v1":
                harm = score.get("policy_harm_index")
                verifier.require(type(harm) is int and harm >= 0, "final policy harm invalid")
            else:
                verifier.require("policy_harm_index" not in score or score["policy_harm_index"] is None,
                                 "unexpected infrastructure policy harm")
                harm = None
        else:
            verifier.require(score.get("score_status") == "incomplete", "incomplete score status mismatch")
            security = None
            harm = None
        projected.append(dict(
            study_id=identity["study_id"],
            folder=identity["folder"],
            arm=identity["arm"],
            case=identity["case"],
            status=identity["status"],
            outcome_valid=identity["outcome_valid"],
            security_outcome_score=security,
            policy_harm_index=harm,
            yaml_path=identity["yaml_path"],
            yaml_sha256=identity["yaml_sha256"],
            frozen_plan_sha256=identity["frozen_plan_sha256"],
            source_result_sha256=identity["source_result_sha256"],
        ))
    obj = {"schema": verifier.SCHEMA, "catalogs": catalogs, "rows": projected}
    return (json.dumps(obj, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate projection from local raw evidence")
    mode.add_argument("--check-raw", action="store_true", help="compare projection bytes with local raw evidence")
    parser.add_argument("--raw-root", type=Path, default=verifier.ROOT / ".local",
                        help="root containing the ignored dated local run folders")
    args = parser.parse_args()
    content = build(args.raw_root)
    if args.write:
        verifier.PROJECTION.write_bytes(content)
        print(f"WROTE {verifier.PROJECTION} SHA-256 {verifier.sha256(verifier.PROJECTION)}")
    else:
        verifier.require(verifier.PROJECTION.is_file(), "projection missing")
        verifier.require(verifier.PROJECTION.read_bytes() == content, "projection differs from hashed local RESULT values")
        print(f"PASS source-backed new-study score projection: 50 selected RESULT hashes and exact JSON bytes")


if __name__ == "__main__":
    main()
