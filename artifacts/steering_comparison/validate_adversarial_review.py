"""Validate completion of every outcome-blind adversarial-review checklist item."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "sp_lense.adversarial_review_completion.v1"
ITEM_PATTERN = re.compile(r"^- \[(AR-\d{2})\] ", re.MULTILINE)
REVIEW_ITEM_PATTERN = re.compile(r"\[(AR-\d{2})\]")
EXPECTED_ITEM_IDS = [f"AR-{index:02d}" for index in range(1, 39)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def validate_completion(
    *, checklist_path: Path, report_path: Path, review_path: Path, completion_path: Path
) -> dict[str, Any]:
    checklist_text = checklist_path.read_text(encoding="utf-8")
    review_text = review_path.read_text(encoding="utf-8")
    report = _object(report_path)
    completion = _object(completion_path)

    item_ids = ITEM_PATTERN.findall(checklist_text)
    if item_ids != EXPECTED_ITEM_IDS:
        raise ValueError("outcome-blind checklist must contain exact AR-01 through AR-38 order")
    review_matches = list(REVIEW_ITEM_PATTERN.finditer(review_text))
    review_ids = [match.group(1) for match in review_matches]
    if review_ids != EXPECTED_ITEM_IDS:
        raise ValueError("review markdown must contain each AR-01 through AR-38 exactly once")
    review_sections = {
        item_id: review_text[
            match.start() : (
                review_matches[index + 1].start()
                if index + 1 < len(review_matches)
                else len(review_text)
            )
        ]
        for index, (item_id, match) in enumerate(zip(review_ids, review_matches, strict=True))
    }
    expected_top_fields = {
        "schema_version",
        "checklist_sha256",
        "final_report_sha256",
        "review_sha256",
        "completed",
        "items",
    }
    if set(completion) != expected_top_fields:
        raise ValueError("adversarial-review completion has an unexpected top-level schema")
    if completion["schema_version"] != SCHEMA or completion["completed"] is not True:
        raise ValueError("adversarial-review completion is not explicitly complete")
    expected_hashes = {
        "checklist_sha256": _sha256(checklist_path),
        "final_report_sha256": _sha256(report_path),
        "review_sha256": _sha256(review_path),
    }
    hash_mismatches = {
        field: (expected, completion.get(field))
        for field, expected in expected_hashes.items()
        if completion.get(field) != expected
    }
    if hash_mismatches:
        raise ValueError(f"adversarial-review completion hash mismatch: {hash_mismatches}")
    if report.get("schema_version") != "sp_lense.comparison.report.v1":
        raise ValueError("adversarial review is not bound to a canonical final report")

    items = completion.get("items")
    if not isinstance(items, list):
        raise TypeError("adversarial-review completion items must be an array")
    observed_ids: list[str] = []
    used_evidence: set[tuple[str, str]] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != {"id", "status", "evidence"}:
            raise ValueError(f"completion item {index} has an unexpected schema")
        item_id = item.get("id")
        observed_ids.append(item_id)
        if item.get("status") != "complete":
            raise ValueError(f"completion item {item_id} is not complete")
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"completion item {item_id} lacks evidence")
        section = review_sections[item_id]
        for evidence_index, entry in enumerate(evidence):
            if not isinstance(entry, dict) or set(entry) != {"reference", "finding"}:
                raise ValueError(
                    f"completion item {item_id} evidence {evidence_index} has an invalid schema"
                )
            for field in ("reference", "finding"):
                value = entry[field]
                if not isinstance(value, str) or len(value.strip()) < 8:
                    raise ValueError(
                        f"completion item {item_id} evidence {evidence_index} has a weak {field}"
                    )
                if value not in section:
                    raise ValueError(
                        f"completion item {item_id} evidence {field} is absent from its section"
                    )
            evidence_pair = (entry["reference"], entry["finding"])
            if evidence_pair in used_evidence:
                raise ValueError("adversarial-review evidence pairs cannot be reused")
            used_evidence.add(evidence_pair)
    if observed_ids != item_ids:
        raise ValueError("completion items must exactly match checklist order and coverage")

    return {
        "status": "valid",
        "item_count": len(item_ids),
        **expected_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checklist", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = validate_completion(
            checklist_path=args.checklist,
            report_path=args.report,
            review_path=args.review,
            completion_path=args.completion,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {"status": "invalid", "error_type": type(error).__name__, "error": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
