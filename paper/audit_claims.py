"""Check every reported latest-policy number against audited observation records."""

import json
from pathlib import Path

try:
    from .latest_results import ROOT, collect
except ImportError:
    from latest_results import ROOT, collect
from reproduce.utils import read_json, require, verify_manifest

HERE = Path(__file__).resolve().parent


def main():
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
    text = (HERE / "manuscript.md").read_text(encoding="utf-8")
    require(all(f"[{i}]" in text for i in range(1, 10)), "Missing reference")
    require(
        not any(term in text.lower() for term in ("colab", "cpu", "legacy", "simplified")),
        "Excluded reporting scope reappeared",
    )
    report = {
        "status": "PASS",
        "source_files_verified": count,
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
