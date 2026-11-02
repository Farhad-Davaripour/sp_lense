"""Fail-closed sequential publication from local disk to Unity Catalog Volumes."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import base64
import hashlib
import json
import os
import re
import shutil
import uuid


PUBLISH_MANIFEST_VERSION = "research4-volume-publish-manifest-v1"
FAILURE_PUBLISH_MANIFEST_VERSION = "research4-volume-failure-manifest-v1"
UPDATE_RECEIPT_VERSION = "research4-durable-update-receipt-v1"
SUCCESS_MANIFEST = "CHILD_OUTPUT_MANIFEST.json"
FAILURE_MANIFEST = "CHILD_FAILURE_MANIFEST.json"
FAILURE_EMBED_PATHS = (
    "MLFLOW_PARAMS.json", "MLFLOW_PROGRESS.jsonl", "RANK_0_PARTIAL.json",
    "RANK_0_RECEIPT.json", "TRAINING_RECEIPT.json",
    "DATABRICKS_EVIDENCE.json", "MLFLOW_FINAL.json",
    "checkpoint/adapter_config.json",
)
FAILURE_EMBED_FILE_LIMIT = 8 * 1024 * 1024
FAILURE_EMBED_TOTAL_LIMIT = 32 * 1024 * 1024


def completion_metadata(*, identity, intent_sha256, plan_sha256,
                        executed_optimizer_step_count,
                        durable_published_update_count) -> dict:
    if (not isinstance(identity, str) or not identity
            or not re.fullmatch(r"[0-9a-f]{64}", str(intent_sha256))
            or not re.fullmatch(r"[0-9a-f]{64}", str(plan_sha256))
            or type(executed_optimizer_step_count) is not int
            or type(durable_published_update_count) is not int
            or executed_optimizer_step_count < 0
            or durable_published_update_count < 0
            or durable_published_update_count > executed_optimizer_step_count):
        raise PermissionError("Child completion metadata is invalid")
    return {
        "identity": identity,
        "intent_sha256": intent_sha256,
        "plan_sha256": plan_sha256,
        "executed_optimizer_step_count": executed_optimizer_step_count,
        "durable_published_update_count": durable_published_update_count,
    }


def failure_metadata(*, identity, intent_sha256, plan_sha256) -> dict:
    if (not isinstance(identity, str) or not identity
            or not re.fullmatch(r"[0-9a-f]{64}", str(intent_sha256))
            or not re.fullmatch(r"[0-9a-f]{64}", str(plan_sha256))):
        raise PermissionError("Child failure metadata is invalid")
    return {"identity": identity, "intent_sha256": intent_sha256,
            "plan_sha256": plan_sha256}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def file_descriptor(path) -> dict:
    target = Path(path)
    return {"bytes": target.stat().st_size, "sha256": file_sha256(target)}


def local_atomic_bytes(path, data: bytes) -> dict:
    """Durably write on local ephemeral storage; never call this on /Volumes."""
    target = Path(path)
    if str(target).replace("\\", "/").startswith("/Volumes/"):
        raise PermissionError("Local durable writes cannot target a Volume")
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f".pending-{target.name}-{uuid.uuid4().hex}")
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, target)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    return file_descriptor(target)


def local_atomic_json(path, value) -> dict:
    return local_atomic_bytes(
        path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def fsync_local_tree(root) -> list[dict]:
    """Fsync every closed local file before it is hashed and published."""
    local_root = Path(root)
    if str(local_root).replace("\\", "/").startswith("/Volumes/"):
        raise PermissionError("Local fsync staging cannot target a Volume")
    files = _relative_files(local_root)
    for path in files:
        descriptor = os.open(path, os.O_RDWR if os.name == "nt" else os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    if os.name != "nt":
        directories = {local_root, *(path.parent for path in files)}
        for directory in sorted(directories, key=lambda value: len(value.parts),
                                reverse=True):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return [{"path": path.relative_to(local_root).as_posix(),
             **file_descriptor(path)} for path in files]


def _compare_files(left, right) -> bool:
    left_path = Path(left); right_path = Path(right)
    if left_path.stat().st_size != right_path.stat().st_size:
        return False
    with left_path.open("rb") as lhs, right_path.open("rb") as rhs:
        while True:
            left_block = lhs.read(1024 * 1024)
            right_block = rhs.read(1024 * 1024)
            if left_block != right_block:
                return False
            if not left_block:
                return True


def publish_immutable_file(local_path, volume_path, *, copy_file=None) -> dict:
    """Sequentially copy one closed local file, then verify exact readback."""
    source = Path(local_path); target = Path(volume_path)
    if not source.is_file():
        raise FileNotFoundError(f"Local publication source is absent: {source}")
    source_descriptor = file_descriptor(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or not _compare_files(source, target):
            raise FileExistsError(f"Immutable Volume publication conflicts: {target}")
        return {**source_descriptor, "path": str(target), "idempotent": True}
    try:
        # O_EXCL prevents a raced publisher from truncating an immutable
        # receipt.  The target is written once, sequentially, and never fsynced
        # or renamed on the Volume filesystem.
        with source.open("rb") as read_stream, target.open("xb") as write_stream:
            if copy_file is None:
                shutil.copyfileobj(read_stream, write_stream, 1024 * 1024)
            else:
                copy_file(read_stream, write_stream)
            write_stream.flush()
    except FileExistsError:
        if not target.is_file() or not _compare_files(source, target):
            raise
    readback = file_descriptor(target)
    if readback != source_descriptor or not _compare_files(source, target):
        raise IOError(f"Immutable Volume readback differs: {target}")
    return {**readback, "path": str(target), "idempotent": False}


def publish_immutable_bytes(data: bytes, volume_path, *, local_stage_dir) -> dict:
    stage = Path(local_stage_dir) / f"publish-{uuid.uuid4().hex}.bin"
    try:
        local_atomic_bytes(stage, data)
        return publish_immutable_file(stage, volume_path)
    finally:
        stage.unlink(missing_ok=True)


def publish_immutable_json(value, volume_path, *, local_stage_dir) -> dict:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return publish_immutable_bytes(data, volume_path,
                                   local_stage_dir=local_stage_dir)


def publish_update_receipt(local_output, volume_output, row) -> dict:
    update = row.get("optimizer_update")
    if type(update) is not int or update < 0:
        raise PermissionError("Durable update index is invalid")
    record_sha = row.get("progress_record_sha256")
    payload = dict(row); embedded = payload.pop("progress_record_sha256", None)
    if embedded != digest(payload):
        raise PermissionError("Progress record self-hash differs")
    _validate_update_receipts(
        local_output, update, require_progress_match=False)
    _validate_update_receipts(
        volume_output, update, require_progress_match=False)
    receipt = {
        "version": UPDATE_RECEIPT_VERSION,
        "optimizer_update": update,
        "progress_record": row,
        "progress_record_sha256": record_sha,
    }
    receipt["receipt_sha256"] = digest(receipt)
    relative = Path("progress_updates") / f"update-{update:06d}.json"
    local_path = Path(local_output) / relative
    local_atomic_json(local_path, receipt)
    published = publish_immutable_file(
        local_path, Path(volume_output) / relative)
    return {"relative_path": relative.as_posix(), **published,
            "receipt_sha256": receipt["receipt_sha256"]}


def _relative_files(root, *, exclude=()) -> list[Path]:
    root = Path(root); excluded = set(exclude)
    result = []
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root).as_posix() not in excluded:
            result.append(path)
    return sorted(result, key=lambda value: value.relative_to(root).as_posix())


def _validate_update_receipts(root, required_update_count, *,
                              require_progress_match=True) -> list[dict]:
    directory = Path(root) / "progress_updates"
    paths = sorted(directory.glob("update-*.json")) if directory.is_dir() else []
    if required_update_count is None:
        expected = list(range(len(paths)))
    else:
        expected = list(range(required_update_count))
    actual = []
    receipts = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(value); embedded = payload.pop("receipt_sha256", None)
        if (value.get("version") != UPDATE_RECEIPT_VERSION
                or embedded != digest(payload)):
            raise PermissionError("Durable update receipt self-hash differs")
        update = value.get("optimizer_update")
        progress = value.get("progress_record")
        if (type(update) is not int or not isinstance(progress, dict)
                or progress.get("optimizer_update") != update
                or value.get("progress_record_sha256")
                   != progress.get("progress_record_sha256")):
            raise PermissionError("Durable update receipt binding differs")
        progress_payload = dict(progress)
        progress_sha = progress_payload.pop("progress_record_sha256", None)
        if progress_sha != digest(progress_payload):
            raise PermissionError("Durable progress record self-hash differs")
        actual.append(update)
        receipts.append({"optimizer_update": update,
                         "receipt_sha256": embedded,
                         "progress_record_sha256": progress_sha,
                         "relative_path": path.relative_to(root).as_posix()})
    if actual != expected:
        raise PermissionError("Durable update receipts are missing or out of order")
    if require_progress_match:
        progress_path = Path(root) / "MLFLOW_PROGRESS.jsonl"
        if not progress_path.is_file():
            if receipts:
                raise PermissionError("Durable progress journal is absent")
            progress_rows = []
        else:
            progress_rows = [
                json.loads(line) for line in
                progress_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if len(progress_rows) != len(receipts):
            raise PermissionError("Durable progress journal count differs")
        for progress, receipt_path in zip(progress_rows, paths, strict=True):
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("progress_record") != progress:
                raise PermissionError(
                    "Durable update receipt differs from progress journal")
    return receipts


def publish_tree(local_root, volume_root, *, status, manifest_name,
                 required_update_count, metadata=None) -> dict:
    if status not in {"complete", "failed"}:
        raise ValueError("Published child status is invalid")
    if manifest_name not in {SUCCESS_MANIFEST, FAILURE_MANIFEST}:
        raise ValueError("Published child manifest name is invalid")
    local_root = Path(local_root); volume_root = Path(volume_root)
    fsync_local_tree(local_root)
    update_receipts = _validate_update_receipts(
        local_root, required_update_count)
    excluded = {SUCCESS_MANIFEST, FAILURE_MANIFEST}
    files = []
    for local_path in _relative_files(local_root, exclude=excluded):
        relative = local_path.relative_to(local_root).as_posix()
        descriptor = publish_immutable_file(local_path, volume_root / relative)
        files.append({"path": relative, "bytes": descriptor["bytes"],
                      "sha256": descriptor["sha256"]})
    manifest = {
        "version": PUBLISH_MANIFEST_VERSION,
        "status": status,
        "files": files,
        "update_receipts": update_receipts,
        "metadata": dict(metadata or {}),
    }
    manifest["manifest_sha256"] = digest(manifest)
    local_manifest = local_root / manifest_name
    local_atomic_json(local_manifest, manifest)
    publish_immutable_file(local_manifest, volume_root / manifest_name)
    verify_manifest(volume_root, manifest_name=manifest_name,
                    expected_status=status,
                    required_update_count=required_update_count,
                    expected_metadata=dict(metadata or {}))
    return manifest


def verify_manifest(volume_root, *, manifest_name, expected_status,
                    required_update_count, allowed_extra=(),
                    expected_metadata=None) -> dict:
    root = Path(volume_root); manifest_path = root / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = dict(manifest); embedded = payload.pop("manifest_sha256", None)
    if (manifest.get("version") != PUBLISH_MANIFEST_VERSION
            or manifest.get("status") != expected_status
            or embedded != digest(payload)):
        raise PermissionError("Published child manifest differs")
    if (expected_metadata is not None
            and manifest.get("metadata") != dict(expected_metadata)):
        raise PermissionError("Published child manifest metadata differs")
    listed = []
    for row in manifest.get("files", []):
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise PermissionError("Published child file descriptor is invalid")
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("Published child path escapes its root")
        path = root.joinpath(*relative.parts)
        if not path.is_file() or file_descriptor(path) != {
                "bytes": row["bytes"], "sha256": row["sha256"]}:
            raise PermissionError("Published child file readback differs")
        listed.append(relative.as_posix())
    if len(listed) != len(set(listed)):
        raise PermissionError("Published child manifest duplicates a file")
    updates = _validate_update_receipts(root, required_update_count)
    if manifest.get("update_receipts") != updates:
        raise PermissionError("Published child update receipt manifest differs")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    expected = set(listed) | {manifest_name} | set(allowed_extra)
    if actual != expected:
        raise PermissionError("Published child tree has missing or extra files")
    return manifest


def publish_failure_manifest(local_root, volume_root, *, metadata,
                             error_type, error) -> dict:
    """Publish terminal failure evidence without recopying conflicted files."""
    local_root = Path(local_root); volume_root = Path(volume_root)
    if not isinstance(metadata, dict) or set(metadata) != {
            "identity", "intent_sha256", "plan_sha256"}:
        raise PermissionError("Child failure metadata fields differ")
    bound_metadata = failure_metadata(**metadata)
    local_files = fsync_local_tree(local_root)
    local_by_path = {row["path"]: row for row in local_files}
    embedded_local_files = []
    embedded_total = 0
    for relative in FAILURE_EMBED_PATHS:
        descriptor = local_by_path.get(relative)
        if descriptor is None:
            continue
        if descriptor["bytes"] > FAILURE_EMBED_FILE_LIMIT:
            raise PermissionError(
                f"Failure evidence file exceeds embed limit: {relative}")
        embedded_total += descriptor["bytes"]
        if embedded_total > FAILURE_EMBED_TOTAL_LIMIT:
            raise PermissionError("Failure evidence exceeds total embed limit")
        content = (local_root / relative).read_bytes()
        embedded_local_files.append({
            **descriptor,
            "encoding": "base64",
            "content": base64.b64encode(content).decode("ascii"),
        })
    durable_files = []
    if volume_root.exists():
        for path in _relative_files(volume_root, exclude={FAILURE_MANIFEST}):
            durable_files.append({
                "path": path.relative_to(volume_root).as_posix(),
                **file_descriptor(path),
            })
    manifest = {
        "version": FAILURE_PUBLISH_MANIFEST_VERSION,
        "status": "failed",
        "metadata": bound_metadata,
        "error_type": str(error_type),
        "error": str(error),
        "local_files": local_files,
        "embedded_local_files": embedded_local_files,
        "durable_files": durable_files,
    }
    manifest["manifest_sha256"] = digest(manifest)
    publish_immutable_json(
        manifest, volume_root / FAILURE_MANIFEST,
        local_stage_dir=local_root / ".failure_publication_stage")
    return verify_failure_manifest(
        volume_root, expected_metadata=bound_metadata)


def verify_failure_manifest(volume_root, *, expected_metadata=None) -> dict:
    root = Path(volume_root); path = root / FAILURE_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(manifest); embedded = payload.pop("manifest_sha256", None)
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != {
            "identity", "intent_sha256", "plan_sha256"}:
        raise PermissionError("Published failure metadata differs")
    bound_metadata = failure_metadata(**metadata)
    if (expected_metadata is not None
            and bound_metadata != failure_metadata(**dict(expected_metadata))):
        raise PermissionError("Published failure metadata binding differs")
    if (manifest.get("version") != FAILURE_PUBLISH_MANIFEST_VERSION
            or manifest.get("status") != "failed"
            or not isinstance(manifest.get("error_type"), str)
            or not isinstance(manifest.get("error"), str)
            or embedded != digest(payload)):
        raise PermissionError("Published failure manifest differs")
    for field in ("local_files", "durable_files"):
        rows = manifest.get(field)
        if not isinstance(rows, list):
            raise PermissionError("Published failure descriptors are absent")
        seen = set()
        for row in rows:
            if (not isinstance(row, dict)
                    or set(row) != {"path", "bytes", "sha256"}
                    or not isinstance(row["bytes"], int) or row["bytes"] < 0
                    or not re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"]))):
                raise PermissionError("Published failure descriptor is invalid")
            relative = PurePosixPath(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise PermissionError("Published failure path escapes its root")
            if relative.as_posix() in seen:
                raise PermissionError("Published failure descriptor is duplicated")
            seen.add(relative.as_posix())
            if field == "durable_files":
                target = root.joinpath(*relative.parts)
                if not target.is_file() or file_descriptor(target) != {
                        "bytes": row["bytes"], "sha256": row["sha256"]}:
                    raise PermissionError(
                        "Published failure durable readback differs")
    embedded_rows = manifest.get("embedded_local_files")
    if not isinstance(embedded_rows, list):
        raise PermissionError("Embedded failure evidence is absent")
    local_descriptors = {row["path"]: row for row in manifest["local_files"]}
    if [row.get("path") for row in embedded_rows] != [
            path for path in FAILURE_EMBED_PATHS if path in local_descriptors]:
        raise PermissionError("Embedded failure evidence order differs")
    embedded_total = 0
    for row in embedded_rows:
        if (not isinstance(row, dict)
                or set(row) != {"path", "bytes", "sha256", "encoding", "content"}
                or row.get("encoding") != "base64"):
            raise PermissionError("Embedded failure evidence descriptor differs")
        try:
            content = base64.b64decode(row["content"], validate=True)
        except (ValueError, TypeError) as exc:
            raise PermissionError("Embedded failure evidence is malformed") from exc
        embedded_total += len(content)
        if (len(content) > FAILURE_EMBED_FILE_LIMIT
                or embedded_total > FAILURE_EMBED_TOTAL_LIMIT
                or local_descriptors.get(row["path"]) != {
                    "path": row["path"], "bytes": row["bytes"],
                    "sha256": row["sha256"]}
                or len(content) != row["bytes"]
                or hashlib.sha256(content).hexdigest() != row["sha256"]):
            raise PermissionError("Embedded failure evidence content differs")
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*") if item.is_file()
    }
    expected = {row["path"] for row in manifest["durable_files"]}
    expected.add(FAILURE_MANIFEST)
    if actual != expected:
        raise PermissionError("Published failure tree has missing or extra files")
    return manifest
