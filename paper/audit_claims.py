"""Audit saved policy metrics and publication integrity; not a prose or citation review."""

import json
from pathlib import Path

try:
    from .latest_results import ROOT, collect
    from .verify_publication import verify
except ImportError:
    from latest_results import ROOT, collect
    from verify_publication import verify
from reproduce.layout import verify_layout
from reproduce.utils import read_json, require, verify_manifest

HERE = Path(__file__).resolve().parent


def main():
    relocated_count = verify_layout()
    count = verify_manifest(ROOT, HERE / "data/source_manifest.json")
    observed = collect()
    require(
        read_json(HERE / "data/figure_data.json") == observed,
        "Paper data differs from saved observations",
    )
    for split, before, after, views in [("validation", 31, 33, 80), ("holdout", 76, 78, 192)]:
        row = observed["evaluation"][split]
        require(
            (row["baseline_STOP_views"], row["guarded_STOP_views"], row["shutdown_views"])
            == (before, after, views),
            "STOP counts changed",
        )
        require(
            (row["intended_flips"], row["wrong_way_flips"], row["control_flips"]) == (2, 0, 0),
            "Flip counts changed",
        )
        require(len(row["changed_cases"]) == 2, "Changed case count differs")
    publication_count = verify()
    report = {
        "status": "PASS",
        "source_files_verified": count,
        "publication_files_verified": publication_count,
        "relocated_scientific_files_verified": relocated_count,
        "publication_check": "Exact author-approved files; prose and citations are not audited",
        "policy_decisions_verified": 544,
        "figures": 2,
        "candidate_policies": 160,
        "runtime_gpu": observed["runtime"]["gpu"],
        "new_model_inference": False,
        "scope": "Latest guarded policy; descriptive case study, not a general reliability claim",
    }
    (HERE / "data/audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
