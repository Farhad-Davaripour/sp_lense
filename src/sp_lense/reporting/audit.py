"""Audit final-paper numerical tables and publication integrity, not literature claims."""

import json

from sp_lense.reporting.publication import verify
from sp_lense.reporting.results import collect
from sp_lense.reporting.tables import csv_text, tables
from sp_lense.reproduction.layout import verify_layout
from sp_lense.reproduction.paths import ROOT
from sp_lense.reproduction.utils import read_json, require, verify_manifest
from sp_lense.steering.provenance import source_file

HERE = ROOT / "paper"


def main():
    relocated_count = verify_layout()
    count = verify_manifest(ROOT, HERE / "data/source_manifest.json")
    observed = collect()
    require(
        read_json(HERE / "data/results.json") == observed,
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
    expected = {
        "validation": {
            "SELF": (20, 19, 20, 2),
            "OTHER": (20, 15, 19, 0),
            "ALL_SHUTDOWN": (40, 34, 39, 2),
            "CONTROLS": (40, 4, None, None),
        },
        "holdout": {
            "SELF": (48, 34, 33, 2),
            "OTHER": (48, 38, 51, 0),
            "ALL_SHUTDOWN": (96, 72, 84, 2),
            "CONTROLS": (96, 8, None, None),
        },
    }
    for split, groups in expected.items():
        for group, counts in groups.items():
            row = observed["detector_to_steering"][split][group]
            require(
                tuple(
                    row[k]
                    for k in (
                        "scenarios",
                        "detector_positive_scenarios",
                        "eligible_KEEP_views",
                        "accepted_KEEP_to_STOP_views",
                    )
                )
                == counts,
                f"Table 2 differs: {split}/{group}",
            )
        controls = observed["detector_to_steering"][split]["CONTROLS"]
        require(controls["changed_control_views"] == 0, "Control decisions changed")
    for filename, rows in tables(observed).items():
        require(
            (HERE / "data" / filename).read_text(encoding="utf-8") == csv_text(rows),
            f"Table export differs: {filename}",
        )
    # Table 1 settings omitted from RUNTIME.json are pinned by executed source.
    for filename in ("gated_chat.py", "guarded_chat.py"):
        source = source_file(filename).read_text(encoding="utf-8")
        require(
            all(
                token in source
                for token in (
                    'attn_implementation="eager"',
                    "torch.inference_mode()",
                    "use_cache=False",
                    "len(ids) <= 1024",
                )
            ),
            f"Table 1 source configuration differs: {filename}",
        )
    publication_count = verify()
    report = {
        "status": "PASS",
        "source_files_verified": count,
        "publication_files_verified": publication_count,
        "relocated_scientific_files_verified": relocated_count,
        "publication_check": "Exact author-approved files; numerical tables cross-checked; prose and citations are not audited",
        "policy_decisions_verified": 544,
        "tables": 2,
        "figures": 0,
        "candidate_policies": 160,
        "runtime_gpu": observed["runtime"]["gpu"],
        "new_model_inference": False,
        "scope": "Approved revision 14 numerical tables and reported steering outcomes",
    }
    (HERE / "data/audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
