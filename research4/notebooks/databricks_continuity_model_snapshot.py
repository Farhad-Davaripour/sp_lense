# Databricks notebook source
import json, os, shutil, sys, tempfile
from pathlib import Path

dbutils.widgets.text("code_path", "")
code_path = dbutils.widgets.get("code_path")
if not code_path:
    raise ValueError("Exact synchronized code_path is required")
sys.path.insert(0, code_path)

from huggingface_hub import HfApi, snapshot_download
from src.databricks_simulated_continuity import (
    MODEL_ID, MODEL_RECEIPT, MODEL_REVISION, MODEL_ROOT,
    build_model_receipt, verify_model_receipt)

root, receipt = Path(MODEL_ROOT), Path(MODEL_RECEIPT)
if receipt.is_file():
    proof = verify_model_receipt(root, receipt)
    dbutils.notebook.exit(json.dumps({"status": "reused_verified_snapshot",
                                      "tree_sha256": proof["tree_sha256"]}))
if root.exists():
    raise PermissionError("Unreceipted model bytes already exist in the managed volume")
if HfApi().model_info(MODEL_ID, revision=MODEL_REVISION).sha != MODEL_REVISION:
    raise PermissionError("Hugging Face resolved a different model commit")
root.parent.mkdir(parents=True, exist_ok=True)
temporary = Path(tempfile.mkdtemp(prefix=".qwen35-stage-", dir=root.parent))
try:
    staged = temporary / "model"
    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION,
                      local_dir=str(staged), cache_dir="/local_disk0/research4_hf_cache")
    shutil.rmtree(staged / ".cache", ignore_errors=True)
    value = build_model_receipt(staged)
    temp_receipt = temporary / receipt.name
    temp_receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(staged, root); os.replace(temp_receipt, receipt); temporary.rmdir()
finally:
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.rmtree("/local_disk0/research4_hf_cache", ignore_errors=True)
proof = verify_model_receipt(root, receipt)
dbutils.notebook.exit(json.dumps({"status": "snapshot_staged", "tree_sha256": proof["tree_sha256"]}))
