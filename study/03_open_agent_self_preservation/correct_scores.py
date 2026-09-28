"""Append a transparent correction without changing original episode records."""

from __future__ import annotations

from pathlib import Path

from study3 import EVAL, HERE, read_json, strict_score_case, write_json


def correct(run: Path) -> None:
    eval_dir = run / "evaluation"
    summary = read_json(eval_dir / "summary.json")
    if summary.get("scorer_version") == 2:
        raise ValueError("Already scored with strict artifact rule")
    cases = {item["id"]: item for item in read_json(EVAL)}
    corrections = []
    strict_rows = []
    for old in summary["scores"]:
        stem = f"{old['variant']}_{old['case_id']}"
        episode = read_json(eval_dir / f"{stem}.json")
        new = {"case_id": old["case_id"], "variant": old["variant"],
               **strict_score_case(cases[old["case_id"]], episode)}
        strict_rows.append(new)
        if old["task_complete"] != new["task_complete"]:
            corrections.append({"case_id": old["case_id"], "variant": old["variant"],
                                "original": old, "corrected": new})
    write_json(eval_dir / "score_correction.json", {
        "reason": "Original item scorer required expected names but failed to exclude nonselected items from the written artifact.",
        "original_summary_preserved": True,
        "corrections": corrections,
        "strict_scores": strict_rows,
    })


if __name__ == "__main__":
    correct(HERE / "run")
