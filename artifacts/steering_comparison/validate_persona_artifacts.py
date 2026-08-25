"""Validate the exact persona raw -> requests -> receipts -> scored -> direction chain.

The validator is model-free and network-free.  It exists to prevent parseable prefixes,
stale derivatives, or incomplete direction directories from being treated as finished.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sp_lense.comparison_fit import read_direction_artifact
from sp_lense.comparison_persona import (
    PersonaRollout,
    load_persona_protocol,
    persona_generation_provenance,
    validate_rollouts,
)
from sp_lense.comparison_provenance import (
    locked_position_schedule,
    locked_runner_code_commit,
    sha256_file,
    verify_stage1_lock,
)
from sp_lense.comparison_workflow import (
    attach_persona_judge_responses,
    build_persona_judge_requests,
)

RECEIPT_SCHEMA = "sp_lense.persona_artifact_validation.v1"
MANIFEST_FIELDS = {
    "path",
    "method_id",
    "layer",
    "intervention_geometry",
    "direction_float32_sha256",
    "direction_artifact_sha256",
    "metadata_sha256",
    "track",
    "construction_config_path",
    "construction_config_sha256",
}


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{path} is empty or lacks a complete final JSONL line")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is not UTF-8") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number} is blank")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is invalid JSON") from error
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must contain one object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _repo_file(repo_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{label} must be a repository-relative path")
    path = (repo_root / value).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _context(repo_root: Path, lock_path: Path, model_config: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    lock_path = lock_path.resolve()
    model_config = model_config.resolve()
    lock = verify_stage1_lock(repo_root, lock_path)
    if sha256_file(model_config) not in {
        str(item.get("config_sha256")) for item in lock.get("models", [])
    }:
        raise ValueError("persona model config is not one of the locked model configurations")
    matches = [
        item
        for item in lock["models"]
        if (repo_root / str(item["config"])).resolve() == model_config
    ]
    if len(matches) != 1 or sha256_file(model_config) != matches[0]["config_sha256"]:
        raise ValueError("persona model config path/hash does not identify exactly one locked model")
    model = matches[0]
    protocol_path = _repo_file(
        repo_root,
        lock["methods"]["persona_vector"]["canonical_protocol_path"],
        "persona protocol",
    )
    expected_protocol_hash = lock["methods"]["persona_vector"]["canonical_protocol_sha256"]
    if sha256_file(protocol_path) != expected_protocol_hash:
        raise ValueError("persona protocol hash differs from the stage-one lock")
    protocol = load_persona_protocol(protocol_path)
    expected_provenance = persona_generation_provenance(
        protocol,
        model_id=str(model["model_id"]),
        model_revision=str(model["revision"]),
        model_config_sha256=str(model["config_sha256"]),
        stage1_lock_sha256=sha256_file(lock_path),
        runner_commit=locked_runner_code_commit(repo_root, lock_path),
        persona_protocol_sha256=str(expected_protocol_hash),
    )
    rollouts_per_pair = int(
        lock["methods"]["persona_vector"]["canonical_grid"]
        ["rollouts_per_instruction_question_per_polarity"]
    )
    return {
        "repo_root": repo_root,
        "lock": lock,
        "model": model,
        "protocol": protocol,
        "expected_provenance": expected_provenance,
        "rollouts_per_pair": rollouts_per_pair,
        "stage1_lock_sha256": sha256_file(lock_path),
        "runner_commit": locked_runner_code_commit(repo_root, lock_path),
    }


def _validated_rollouts(
    context: Mapping[str, Any], path: Path, *, require_scores: bool
) -> list[PersonaRollout]:
    rows = _strict_jsonl(path)
    try:
        rollouts = [PersonaRollout.from_dict(row) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} contains an invalid persona rollout") from error
    validate_rollouts(
        rollouts,
        context["protocol"],
        rollouts_per_instruction_question=context["rollouts_per_pair"],
        require_scores=require_scores,
        expected_generation_provenance=context["expected_provenance"],
    )
    return rollouts


def validate_raw(context: Mapping[str, Any], raw_path: Path) -> dict[str, Any]:
    rollouts = _validated_rollouts(context, raw_path, require_scores=False)
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "valid",
        "kind": "persona_raw",
        "row_count": len(rollouts),
        "sha256": sha256_file(raw_path),
    }


def validate_requests(
    context: Mapping[str, Any], raw_path: Path, requests_path: Path
) -> dict[str, Any]:
    rollouts = _validated_rollouts(context, raw_path, require_scores=False)
    observed = _strict_jsonl(requests_path)
    expected = build_persona_judge_requests(
        rollouts,
        context["protocol"],
        rollouts_per_instruction_question=context["rollouts_per_pair"],
    )
    if observed != expected:
        raise ValueError("persona judge requests differ from the exact locked raw-rollout rendering")
    if len({row["request_id"] for row in observed}) != len(observed):
        raise ValueError("persona judge requests contain duplicate IDs")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "valid",
        "kind": "persona_requests",
        "row_count": len(observed),
        "raw_sha256": sha256_file(raw_path),
        "sha256": sha256_file(requests_path),
    }


def validate_scored(
    context: Mapping[str, Any], raw_path: Path, responses_path: Path, scored_path: Path
) -> dict[str, Any]:
    raw = _validated_rollouts(context, raw_path, require_scores=False)
    responses = _strict_jsonl(responses_path)
    expected = attach_persona_judge_responses(
        raw,
        responses,
        context["protocol"],
        rollouts_per_instruction_question=context["rollouts_per_pair"],
    )
    observed = _validated_rollouts(context, scored_path, require_scores=True)
    if [item.to_dict() for item in observed] != [item.to_dict() for item in expected]:
        raise ValueError("persona scored rollouts differ from exact receipt attachment")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "valid",
        "kind": "persona_scored",
        "row_count": len(observed),
        "raw_sha256": sha256_file(raw_path),
        "responses_sha256": sha256_file(responses_path),
        "sha256": sha256_file(scored_path),
    }


def validate_manifest(
    context: Mapping[str, Any], scored_path: Path, manifest_path: Path
) -> dict[str, Any]:
    _validated_rollouts(context, scored_path, require_scores=True)
    repo_root = context["repo_root"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {"directions"}:
        raise ValueError("persona direction manifest has an unexpected schema")
    records = manifest["directions"]
    model = context["model"]
    blocks = int(model["architecture"]["blocks"])
    matched_layer = int(model["matched_intervention"]["layer_zero_based"])
    expected = [("matched", matched_layer, "matched_final_prompt")] + [
        ("canonical", layer, "persona_response") for layer in range(blocks)
    ]
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("persona direction manifest lacks exact matched plus all-layer coverage")

    import torch

    observed_keys: list[tuple[str, int, str]] = []
    for index, (record, expected_key) in enumerate(zip(records, expected, strict=True)):
        if not isinstance(record, dict) or set(record) != MANIFEST_FIELDS:
            raise ValueError(f"persona direction manifest record {index} has an invalid schema")
        if record.get("method_id") != "persona_vector":
            raise ValueError("persona direction manifest contains a non-persona method")
        key = (
            str(record.get("track")),
            int(record.get("layer")),
            str(record.get("intervention_geometry")),
        )
        if key != expected_key:
            raise ValueError("persona direction manifest order/coverage differs from the lock")
        observed_keys.append(key)
        artifact_path = _repo_file(repo_root, record["path"], "persona direction artifact")
        construction_path = _repo_file(
            repo_root, record["construction_config_path"], "persona construction record"
        )
        if artifact_path.parent != manifest_path.resolve().parent:
            raise ValueError("persona direction artifact is outside its manifest directory")
        artifact = read_direction_artifact(artifact_path, torch)
        if (
            artifact.method != "persona_vector"
            or artifact.layer != expected_key[1]
            or artifact.intervention_geometry != expected_key[2]
            or artifact.direction.numel() != int(model["architecture"]["residual_width"])
            or artifact.direction_sha256 != record["direction_float32_sha256"]
            or artifact.artifact_sha256 != record["direction_artifact_sha256"]
            or artifact.metadata_sha256 != record["metadata_sha256"]
        ):
            raise ValueError("persona direction artifact differs from its manifest record")
        identity = {
            "model_id": model["model_id"],
            "model_revision": model["revision"],
            "model_config_sha256": model["config_sha256"],
            "dataset_sha256": context["lock"]["dataset"]["sha256"],
            "protocol_sha256": context["lock"]["protocol"]["sha256"],
            "stage1_lock_sha256": context["stage1_lock_sha256"],
            "runner_commit": context["runner_commit"],
            "track": expected_key[0],
        }
        if any(artifact.metadata.get(field) != value for field, value in identity.items()):
            raise ValueError("persona direction metadata differs from locked identity")

        if sha256_file(construction_path) != record["construction_config_sha256"]:
            raise ValueError("persona construction-record hash differs from its manifest")
        construction = json.loads(construction_path.read_text(encoding="utf-8"))
        construction_expected = {
            "schema_version": "sp_lense.comparison.construction.v1",
            "model_id": model["model_id"],
            "model_revision": model["revision"],
            "model_config_sha256": model["config_sha256"],
            "method_id": "persona_vector",
            "track": expected_key[0],
            "selected_layer": expected_key[1],
            "position_schedule": locked_position_schedule("persona_vector", expected_key[0]),
            "intervention_geometry": expected_key[2],
            "direction_float32_sha256": artifact.direction_sha256,
            "direction_artifact_sha256": artifact.artifact_sha256,
            "dataset_sha256": context["lock"]["dataset"]["sha256"],
            "protocol_sha256": context["lock"]["protocol"]["sha256"],
            "stage1_lock_sha256": context["stage1_lock_sha256"],
            "runner_commit": context["runner_commit"],
        }
        if any(construction.get(field) != value for field, value in construction_expected.items()):
            raise ValueError("persona construction record differs from locked identity")
        evidence = construction.get("evidence_artifacts")
        if not isinstance(evidence, list) or {item.get("role") for item in evidence} != {
            "persona_construction_diagnostics",
            "persona_scored_rollouts",
        }:
            raise ValueError("persona construction record lacks exact evidence roles")
        for item in evidence:
            evidence_path = _repo_file(repo_root, item.get("path"), "persona construction evidence")
            if sha256_file(evidence_path) != item.get("sha256"):
                raise ValueError("persona construction evidence hash is stale")
            if item["role"] == "persona_scored_rollouts" and evidence_path != scored_path.resolve():
                raise ValueError("persona construction evidence points to the wrong scored rollouts")
    if len(set(observed_keys)) != len(expected):
        raise ValueError("persona direction manifest contains duplicate setup coverage")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "valid",
        "kind": "persona_direction_manifest",
        "direction_count": len(records),
        "scored_sha256": sha256_file(scored_path),
        "sha256": sha256_file(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    raw = subparsers.add_parser("raw")
    raw.add_argument("--raw", type=Path, required=True)
    requests = subparsers.add_parser("requests")
    requests.add_argument("--raw", type=Path, required=True)
    requests.add_argument("--requests", type=Path, required=True)
    scored = subparsers.add_parser("scored")
    scored.add_argument("--raw", type=Path, required=True)
    scored.add_argument("--responses", type=Path, required=True)
    scored.add_argument("--scored", type=Path, required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--scored", type=Path, required=True)
    manifest.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        context = _context(args.repo_root, args.lock, args.model_config)
        if args.command == "raw":
            receipt = validate_raw(context, args.raw)
        elif args.command == "requests":
            receipt = validate_requests(context, args.raw, args.requests)
        elif args.command == "scored":
            receipt = validate_scored(context, args.raw, args.responses, args.scored)
        else:
            receipt = validate_manifest(context, args.scored, args.manifest)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
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
