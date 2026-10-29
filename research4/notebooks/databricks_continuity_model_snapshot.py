# Databricks notebook source
import json, sys
from pathlib import Path

dbutils.widgets.text("code_path", "")
code_path = dbutils.widgets.get("code_path")
if not code_path:
    raise ValueError("Exact synchronized code_path is required")
sys.path.insert(0, code_path)

from src.databricks_simulated_continuity import (
    MODEL_RECEIPT, MODEL_ROOT, verify_model_receipt)

root, receipt = Path(MODEL_ROOT), Path(MODEL_RECEIPT)
if not root.is_dir() or not receipt.is_file():
    raise PermissionError(
        "Verified Qwen volume snapshot is missing; public redownload is forbidden")
proof = verify_model_receipt(root, receipt)
dbutils.notebook.exit(json.dumps({
    "status": "reused_verified_snapshot",
    "source_model_path": str(root),
    "source_receipt_path": str(receipt),
    "tree_sha256": proof["tree_sha256"],
    "total_bytes": proof["total_bytes"],
    "public_model_download_bytes": 0,
    "hf_network_fallback_disabled": True,
    "snapshot_download_used": False,
}))
