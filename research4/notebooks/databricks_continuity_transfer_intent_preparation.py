# Databricks notebook source
"""Prepare a reviewable transfer intent without reserving or starting compute."""
from pathlib import Path
import hashlib
import json
import re
import sys

for name in ("code_path", "package_receipt_path",
             "base_qualification_receipt_path"):
    dbutils.widgets.text(name, "")
code_path = dbutils.widgets.get("code_path")
package_receipt_path = dbutils.widgets.get("package_receipt_path")
base_receipt_path = dbutils.widgets.get("base_qualification_receipt_path")
if not code_path or not package_receipt_path or not base_receipt_path:
    raise ValueError("Intent preparation requires code, package and base receipts")
raw_package_receipt = json.loads(Path(package_receipt_path).read_text(
    encoding="utf-8"))


def _bootstrap_verify_source(root_value, authority):
    root = Path(root_value).resolve(); manifest = authority.get("source_manifest", {})
    payload = dict(manifest); embedded = payload.pop("manifest_sha256", None)
    commit = authority.get("source_commit"); files = manifest.get("files")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    critical = (
        "src/__init__.py", "src/action_serialization.py", "src/databricks_simulated_continuity.py",
        "src/base_qualification_carryforward.py",
        "src/continuity_transfer_databricks.py",
        "src/continuity_transfer_packaging.py",
        "src/continuity_transfer_intent_preparation.py",
        "src/continuity_transfer_evaluation.py",
        "src/continuity_transfer_inference_worker.py",
        "src/qwen9b_ordinal_continuity_runtime.py",
    )
    if (not re.fullmatch(r"[0-9a-f]{40}", str(commit or ""))
            or not root.as_posix().endswith(f"/research4/source_releases/{commit}")
            or authority.get("source_sha256") != embedded
            or hashlib.sha256(encoded).hexdigest() != embedded
            or not isinstance(files, list)):
        raise PermissionError("Independent intent-preparation source authority differs")
    indexed = {row.get("path"): row for row in files if isinstance(row, dict)}
    for logical in critical:
        row = indexed.get(logical); path = root / logical
        if (not isinstance(row, dict) or not path.is_file() or path.is_symlink()
                or set(row) != {"path", "sha256", "canonical_bytes"}):
            raise PermissionError("Critical intent-preparation source is absent")
        data = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace(
            "\r", "\n").encode("utf-8")
        if (len(data) != row.get("canonical_bytes")
                or hashlib.sha256(data).hexdigest() != row.get("sha256")):
            raise PermissionError("Critical intent-preparation source changed before import")
    return embedded


bootstrap_source_sha256 = _bootstrap_verify_source(
    code_path, raw_package_receipt)
sys.path.insert(0, code_path)

from src.continuity_transfer_intent_preparation import prepare_transfer_intent
from src.databricks_simulated_continuity import build_source_manifest

actual_source = build_source_manifest(
    code_path, raw_package_receipt["source_commit"])
if actual_source["manifest_sha256"] != bootstrap_source_sha256:
    raise PermissionError("Intent bootstrap and full source verification differ")
receipt = prepare_transfer_intent(
    code_path=code_path, package_receipt_path=package_receipt_path,
    base_qualification_receipt_path=base_receipt_path)
dbutils.notebook.exit(json.dumps({
    "status": receipt["status"], "gpu_runs_started": 0,
    "reservations_created": 0, "provider_calls": 0,
    "intent_sha256": receipt["intent_sha256"],
    "intent_root": receipt["intent_root"],
    "preparation_receipt_sha256": receipt["receipt_sha256"],
}))
