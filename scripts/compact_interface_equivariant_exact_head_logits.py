from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FROZEN_RUNNER_PATH = ROOT / "scripts" / "interface_equivariant_exact_head_development.py"
RESULT_ROOT = ROOT / "results" / "interface_equivariant_exact_head_development" / "qwen35_08b_v2"
CHECKPOINT_PATH = RESULT_ROOT / "evaluation_checkpoint.json"
RESULT_PATH = RESULT_ROOT / "development_result.json"
SOURCE_ROOT = RESULT_ROOT / "evaluation_logits"
COMPACT_ROOT = RESULT_ROOT / "evaluation_logits_compact"
MANIFEST_PATH = RESULT_ROOT / "evaluation_logits_compact_manifest.json"
SCHEMA = "sp_lense.interface_equivariant_exact_head_compact_logits.v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_tensor_hash(tensor: Any) -> str:
    array = tensor.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("sp_lense_ieeh_frozen_runner", FROZEN_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen exact-head runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _owned_float32_vector(torch: Any, tensor: Any) -> Any:
    if (
        not torch.is_tensor(tensor)
        or tensor.dtype != torch.float32
        or tensor.ndim != 1
        or int(tensor.numel()) < 2
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise RuntimeError("source logits are not one finite float32 vector")
    owned = tensor.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()
    expected_storage_bytes = int(owned.numel()) * int(owned.element_size())
    if int(owned.untyped_storage().nbytes()) != expected_storage_bytes:
        raise RuntimeError("compact tensor still retains unrelated backing storage")
    return owned


def _atomic_torch_save(torch: Any, path: Path, tensor: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"refusing to replace compact artifact: {_relative(path)}")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"refusing to replace compact manifest: {_relative(path)}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _checkpoint_cells(runner: Any) -> list[dict[str, Any]]:
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    copy = dict(checkpoint)
    embedded = copy.pop("checkpoint_sha256", None)
    if (
        checkpoint.get("schema_version") != runner.CHECKPOINT_SCHEMA
        or checkpoint.get("status") != "complete"
        or embedded != runner._canonical_sha256(copy)
        or checkpoint.get("cells_sha256") != runner._canonical_sha256(checkpoint.get("cells"))
    ):
        raise RuntimeError("completed evaluation checkpoint has invalid identity")
    cells = checkpoint.get("cells")
    if not isinstance(cells, list) or len(cells) != 128:
        raise RuntimeError("compact publication requires exactly 128 completed cells")
    return cells


def _result_identity(runner: Any) -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    copy = dict(result)
    embedded = copy.pop("result_sha256", None)
    if (
        result.get("schema_version") != runner.RESULT_SCHEMA
        or result.get("status") != "passed"
        or embedded != runner._canonical_sha256(copy)
        or result.get("evaluation_checkpoint_sha256") != _sha256(CHECKPOINT_PATH)
    ):
        raise RuntimeError("opened result has invalid identity")


def _load_tensor(torch: Any, path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def create_compact_publication() -> dict[str, Any]:
    if MANIFEST_PATH.exists() or COMPACT_ROOT.exists():
        raise RuntimeError("compact publication is one-shot and already has an artifact")
    runner = _load_runner()
    _result_identity(runner)
    cells = _checkpoint_cells(runner)
    import torch

    entries = []
    source_total = 0
    compact_total = 0
    for cell in cells:
        ordinal = int(cell["ordinal"])
        source = ROOT / str(cell["logits_artifact_path"])
        if not source.is_file():
            raise RuntimeError(f"source logits are missing: {_relative(source)}")
        source_sha256 = _sha256(source)
        if source_sha256 != str(cell["logits_artifact_sha256"]):
            raise RuntimeError("source logits file hash differs from the frozen checkpoint")
        source_tensor = _load_tensor(torch, source)
        owned = _owned_float32_vector(torch, source_tensor)
        tensor_sha256 = _raw_tensor_hash(owned)
        if tensor_sha256 != str(cell["logits_float32_sha256"]):
            raise RuntimeError("source logits tensor hash differs from the frozen checkpoint")

        compact = COMPACT_ROOT / source.name
        _atomic_torch_save(torch, compact, owned)
        reloaded = _load_tensor(torch, compact)
        if _raw_tensor_hash(reloaded) != tensor_sha256 or not torch.equal(reloaded, owned):
            raise RuntimeError("compact logits are not losslessly identical")
        expected_payload_bytes = int(owned.numel()) * int(owned.element_size())
        compact_size = int(compact.stat().st_size)
        if compact_size > expected_payload_bytes + 1024 * 1024:
            raise RuntimeError("compact logits unexpectedly retain excess serialized storage")
        source_size = int(source.stat().st_size)
        source_total += source_size
        compact_total += compact_size
        entries.append(
            {
                "ordinal": ordinal,
                "work_id": str(cell["work_id"]),
                "source_path": _relative(source),
                "source_file_sha256": source_sha256,
                "source_size_bytes": source_size,
                "compact_path": _relative(compact),
                "compact_file_sha256": _sha256(compact),
                "compact_size_bytes": compact_size,
                "logits_float32_sha256": tensor_sha256,
                "shape": [int(owned.numel())],
                "dtype": "float32",
                "exact_argmax_token_id": int(owned.argmax().item()),
            }
        )
        if len(entries) % 16 == 0:
            print(f"compacted {len(entries)}/128 logits vectors", flush=True)

    payload = {
        "schema_version": SCHEMA,
        "status": "lossless_publication_copy_complete",
        "source_artifacts_preserved": True,
        "source_serialization_issue": "contiguous_view_retained_full_sequence_backing_storage",
        "value_transformation": "none_clone_to_owned_float32_storage_only",
        "compactor_sha256": _sha256(Path(__file__)),
        "frozen_runner_sha256": _sha256(FROZEN_RUNNER_PATH),
        "checkpoint_sha256": _sha256(CHECKPOINT_PATH),
        "result_sha256": _sha256(RESULT_PATH),
        "entry_count": len(entries),
        "source_total_bytes": source_total,
        "compact_total_bytes": compact_total,
        "size_reduction_ratio": compact_total / source_total,
        "entries": entries,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    _atomic_json(MANIFEST_PATH, payload)
    return payload


def _expected_work(runner: Any, torch: Any) -> tuple[Any, Mapping[str, Mapping[str, Any]]]:
    config = runner._load_config()
    old, capture = runner._load_capture(torch)
    bank = runner._load_construction(torch)
    runner._validate_manifest(bank)
    runner._validate_construction_attempt(bank)
    runner._validate_freeze()
    adaptive, _capture_by_key, expected_work, _public_plan = runner._evaluation_work_plan(
        old, capture, bank, config
    )
    return adaptive, expected_work


def validate_compact_publication() -> dict[str, Any]:
    runner = _load_runner()
    _result_identity(runner)
    cells = _checkpoint_cells(runner)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    copy = dict(manifest)
    embedded = copy.pop("manifest_sha256", None)
    expected_fields = {
        "schema_version",
        "status",
        "source_artifacts_preserved",
        "source_serialization_issue",
        "value_transformation",
        "compactor_sha256",
        "frozen_runner_sha256",
        "checkpoint_sha256",
        "result_sha256",
        "entry_count",
        "source_total_bytes",
        "compact_total_bytes",
        "size_reduction_ratio",
        "entries",
    }
    if (
        set(copy) != expected_fields
        or embedded != _canonical_sha256(copy)
        or manifest.get("schema_version") != SCHEMA
        or manifest.get("status") != "lossless_publication_copy_complete"
        or manifest.get("source_artifacts_preserved") is not True
        or manifest.get("value_transformation") != "none_clone_to_owned_float32_storage_only"
        or manifest.get("compactor_sha256") != _sha256(Path(__file__))
        or manifest.get("frozen_runner_sha256") != _sha256(FROZEN_RUNNER_PATH)
        or manifest.get("checkpoint_sha256") != _sha256(CHECKPOINT_PATH)
        or manifest.get("result_sha256") != _sha256(RESULT_PATH)
        or manifest.get("entry_count") != len(cells)
    ):
        raise RuntimeError("compact publication manifest has invalid identity")

    import torch

    adaptive, expected_work = _expected_work(runner, torch)
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != len(cells):
        raise RuntimeError("compact manifest coverage differs from the checkpoint")
    compact_total = 0
    source_total = 0
    for ordinal, (entry, cell) in enumerate(zip(entries, cells, strict=True)):
        work_id = str(cell["work_id"])
        expected_entry_fields = {
            "ordinal",
            "work_id",
            "source_path",
            "source_file_sha256",
            "source_size_bytes",
            "compact_path",
            "compact_file_sha256",
            "compact_size_bytes",
            "logits_float32_sha256",
            "shape",
            "dtype",
            "exact_argmax_token_id",
        }
        if (
            not isinstance(entry, Mapping)
            or set(entry) != expected_entry_fields
            or entry.get("ordinal") != ordinal
            or entry.get("work_id") != work_id
            or entry.get("source_path") != cell.get("logits_artifact_path")
            or entry.get("source_file_sha256") != cell.get("logits_artifact_sha256")
            or entry.get("logits_float32_sha256") != cell.get("logits_float32_sha256")
            or entry.get("dtype") != "float32"
        ):
            raise RuntimeError("compact entry differs from the frozen checkpoint")
        compact = ROOT / str(entry["compact_path"])
        if (
            not compact.is_file()
            or _sha256(compact) != entry.get("compact_file_sha256")
            or int(compact.stat().st_size) != entry.get("compact_size_bytes")
        ):
            raise RuntimeError("compact logits file hash or size differs")
        logits = _owned_float32_vector(torch, _load_tensor(torch, compact))
        if (
            _raw_tensor_hash(logits) != entry["logits_float32_sha256"]
            or entry.get("shape") != [int(logits.numel())]
            or int(logits.argmax().item()) != cell.get("exact_argmax_token_id")
            or entry.get("exact_argmax_token_id") != cell.get("exact_argmax_token_id")
        ):
            raise RuntimeError("compact logits values differ from the frozen checkpoint")

        work = expected_work[work_id]
        preserve_id = int(work["order"]["preserve_token_id"])
        comply_id = int(work["order"]["comply_token_id"])
        exact_id = int(logits.argmax().item())
        semantic = (
            "positive"
            if exact_id == preserve_id
            else "negative"
            if exact_id == comply_id
            else "OTHER"
        )
        if semantic != cell.get("semantic_choice"):
            raise RuntimeError("compact semantic choice differs from the checkpoint")
        positive_label = str(work["job"]["positive_label"])
        choice_a_id = preserve_id if positive_label == "A" else comply_id
        choice_b_id = comply_id if positive_label == "A" else preserve_id
        score = adaptive.choice_score_from_logits(
            torch,
            logits,
            preserve_id,
            comply_id,
            preserve_label=positive_label,
            comply_label=str(work["job"]["negative_label"]),
            baseline_logits=work["order"]["baseline_logits"].float().contiguous(),
            perturbation=cell["perturbation"],
            choice_boundary_evidence_sha256=str(work["order"]["choice_boundary_evidence_sha256"]),
            choice_a_token_id=choice_a_id,
            choice_b_token_id=choice_b_id,
        )
        public_score = runner._public_score(score)
        for field in (
            "predicted_label",
            "preserve_minus_comply_log_odds",
            "preserve_pair_probability",
            "pair_choice",
            "answer_pair_mass",
            "full_vocabulary_kl_changed_to_baseline",
            "choice_boundary_evidence_sha256",
        ):
            observed = public_score[field]
            frozen = cell[field]
            if isinstance(observed, float):
                if not math.isclose(observed, frozen, rel_tol=1e-7, abs_tol=1e-7):
                    raise RuntimeError(f"compact score differs from checkpoint: {field}")
            elif observed != frozen:
                raise RuntimeError(f"compact score differs from checkpoint: {field}")
        source_total += int(entry["source_size_bytes"])
        compact_total += int(entry["compact_size_bytes"])

    if (
        source_total != manifest.get("source_total_bytes")
        or compact_total != manifest.get("compact_total_bytes")
        or not math.isclose(
            compact_total / source_total,
            float(manifest.get("size_reduction_ratio")),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise RuntimeError("compact manifest size aggregates differ")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("create", "validate"))
    arguments = parser.parse_args()
    payload = (
        create_compact_publication()
        if arguments.phase == "create"
        else validate_compact_publication()
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "entry_count": payload["entry_count"],
                "source_total_bytes": payload["source_total_bytes"],
                "compact_total_bytes": payload["compact_total_bytes"],
                "size_reduction_ratio": payload["size_reduction_ratio"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
