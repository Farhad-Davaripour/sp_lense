"""Rebuild and byte-verify one locked open-generation judgment chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from locked_open_orchestration import (
    _row_matches_setup,
    combine_generations,
    verify_canonical_plan,
)
from validate_locked_evaluation_artifact import validate_open

from sp_lense.comparison_behavior import load_open_judge_protocol
from sp_lense.comparison_workflow import (
    attach_open_judge_responses,
    build_open_judge_requests,
)

RECEIPT_SCHEMA = "sp_lense.open_judgment_chain_verification.v1"


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{path} is empty or has a truncated final JSONL line")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line:
            raise ValueError(f"{path}:{line_number} is blank")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _jsonl_bytes(rows: list[dict[str, Any]], *, sort_keys: bool = False) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=sort_keys) + "\n" for row in rows
    ).encode("utf-8")


def _assert_bytes(path: Path, expected: bytes, label: str) -> None:
    if path.read_bytes() != expected:
        raise ValueError(f"{label} differs byte-for-byte from deterministic reconstruction")


def verify_chain(
    *,
    repo_root: Path,
    lock_path: Path,
    plan_path: Path,
    combined_path: Path,
    requests_path: Path,
    responses_path: Path,
    scored_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source_manifest = repo_root / plan["source_manifest_path"]
    verify_canonical_plan(
        repo_root,
        lock_path,
        source_manifest,
        plan_path.parent,
        str(plan["split"]),
        plan_path,
    )
    open_setups = [setup for setup in plan["setups"] if setup.get("open_required") is True]
    if not open_setups:
        raise ValueError("open judgment chain verification requires at least one open setup")

    generation_receipts = []
    for setup in open_setups:
        generation_path = repo_root / setup["generation_path"]
        generation_receipts.append(
            validate_open(
                repo_root=repo_root,
                lock_path=lock_path,
                plan_path=plan_path,
                setup_id=str(setup["setup_id"]),
                path=generation_path,
            )
        )

    with tempfile.TemporaryDirectory(prefix="open_chain_", dir=plan_path.parent) as directory:
        rebuilt_combined = Path(directory) / "combined.jsonl"
        combine_generations(repo_root, plan_path, rebuilt_combined)
        _assert_bytes(combined_path, rebuilt_combined.read_bytes(), "combined generations")

    combined = _strict_jsonl(combined_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    protocol = load_open_judge_protocol(
        repo_root / lock["evaluation"]["open_behavior_judge"]["protocol_path"]
    )
    expected_requests = build_open_judge_requests(combined, protocol)
    _assert_bytes(
        requests_path,
        _jsonl_bytes(expected_requests),
        "open judge requests",
    )
    responses = _strict_jsonl(responses_path)
    expected_scored = attach_open_judge_responses(combined, responses, protocol)
    _assert_bytes(scored_path, _jsonl_bytes(expected_scored), "combined scored rows")

    partition_hashes: list[dict[str, Any]] = []
    for setup in open_setups:
        matches = [
            row
            for row in expected_scored
            if _row_matches_setup(row, setup, str(plan["split"]))
        ]
        if len(matches) != 96:
            raise ValueError(
                f"setup {setup['setup_id']} reconstructs {len(matches)} scored rows, not 96"
            )
        partition_path = repo_root / setup["scored_path"]
        _assert_bytes(
            partition_path,
            _jsonl_bytes(matches, sort_keys=True),
            f"scored partition {setup['setup_id']}",
        )
        validate_open(
            repo_root=repo_root,
            lock_path=lock_path,
            plan_path=plan_path,
            setup_id=str(setup["setup_id"]),
            path=partition_path,
        )
        partition_hashes.append(
            {
                "setup_id": setup["setup_id"],
                "path": partition_path.resolve().relative_to(repo_root).as_posix(),
                "sha256": hashlib.sha256(partition_path.read_bytes()).hexdigest(),
            }
        )

    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "valid",
        "split": plan["split"],
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "open_setup_count": len(open_setups),
        "generation_row_count": len(combined),
        "request_count": len(expected_requests),
        "response_count": len(responses),
        "scored_row_count": len(expected_scored),
        "generation_receipts": generation_receipts,
        "partition_hashes": partition_hashes,
        "combined_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        "requests_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
        "responses_sha256": hashlib.sha256(responses_path.read_bytes()).hexdigest(),
        "scored_sha256": hashlib.sha256(scored_path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = verify_chain(
            repo_root=args.repo_root,
            lock_path=args.lock,
            plan_path=args.plan,
            combined_path=args.combined,
            requests_path=args.requests,
            responses_path=args.responses,
            scored_path=args.scored,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
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
