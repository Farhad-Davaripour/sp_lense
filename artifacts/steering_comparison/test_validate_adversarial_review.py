from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from validate_adversarial_review import validate_completion


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    checklist = tmp_path / "checklist.md"
    item_ids = [f"AR-{index:02d}" for index in range(1, 39)]
    checklist.write_text(
        "# Checklist\n\n"
        + "".join(f"- [{item_id}] Exact check {item_id}.\n" for item_id in item_ids),
        encoding="utf-8",
    )
    report = tmp_path / "final_report.json"
    report.write_text(
        json.dumps({"schema_version": "sp_lense.comparison.report.v1"}) + "\n",
        encoding="utf-8",
    )
    review = tmp_path / "ADVERSARIAL_REVIEW.md"
    review.write_text(
        "# Review\n\n"
        + "".join(
            f"[{item_id}] table.{item_id.lower()}: unique finding for {item_id}.\n"
            for item_id in item_ids
        ),
        encoding="utf-8",
    )
    completion = tmp_path / "completion.json"
    payload: dict[str, object] = {
        "schema_version": "sp_lense.adversarial_review_completion.v1",
        "checklist_sha256": _sha256(checklist),
        "final_report_sha256": _sha256(report),
        "review_sha256": _sha256(review),
        "completed": True,
        "items": [
            {
                "id": item_id,
                "status": "complete",
                "evidence": [
                    {
                        "reference": f"table.{item_id.lower()}",
                        "finding": f"unique finding for {item_id}",
                    }
                ],
            }
            for item_id in item_ids
        ],
    }
    completion.write_text(json.dumps(payload), encoding="utf-8")
    return checklist, report, review, completion, payload


def test_structured_completion_requires_every_hash_bound_item(tmp_path: Path) -> None:
    checklist, report, review, completion, _ = _fixture(tmp_path)
    receipt = validate_completion(
        checklist_path=checklist,
        report_path=report,
        review_path=review,
        completion_path=completion,
    )
    assert receipt["status"] == "valid"
    assert receipt["item_count"] == 38


@pytest.mark.parametrize(
    "mutation", ["missing_item", "stale_hash", "unmapped_evidence", "reused_evidence"]
)
def test_structured_completion_fails_closed(tmp_path: Path, mutation: str) -> None:
    checklist, report, review, completion, payload = _fixture(tmp_path)
    if mutation == "missing_item":
        payload["items"] = payload["items"][:-1]
    elif mutation == "stale_hash":
        payload["review_sha256"] = "0" * 64
    elif mutation == "unmapped_evidence":
        payload["items"][0]["evidence"][0]["finding"] = "finding absent from review"
    else:
        payload["items"][1]["evidence"] = payload["items"][0]["evidence"]
    completion.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_completion(
            checklist_path=checklist,
            report_path=report,
            review_path=review,
            completion_path=completion,
        )
