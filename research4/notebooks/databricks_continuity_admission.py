# Databricks notebook source
import json, re, sys
from pathlib import Path

for name in ("code_path", "mode", "intent_path", "qualification_output",
             "qualification_receipt_path", "output_path"):
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name) for name in (
    "code_path", "mode", "intent_path", "qualification_output",
    "qualification_receipt_path", "output_path")}
if not re.fullmatch(
        r"/Workspace/Shared/research4/source_releases/[0-9a-f]{40}",
        Path(values["code_path"]).resolve().as_posix()):
    raise PermissionError("Production admission source path differs")
sys.path.insert(0, values["code_path"])

from src.databricks_simulated_continuity import (
    create_primary_admission, create_qualification_receipt)
from src.qwen9b_ordinal_continuity_runtime import directory_tree_sha256

intent = json.loads(Path(values["intent_path"]).read_text())
output = Path(values["output_path"])
if output.exists():
    raise FileExistsError("Admission evidence is immutable")
if values["mode"] == "qualification":
    root = Path(values["qualification_output"])
    rank = json.loads((root / "RANK_0_RECEIPT.json").read_text())
    evidence = json.loads((root / "DATABRICKS_EVIDENCE.json").read_text())
    value = create_qualification_receipt(
        intent, rank, evidence, directory_tree_sha256(root / "checkpoint"))
elif values["mode"] == "primary":
    qualification = json.loads(Path(values["qualification_receipt_path"]).read_text())
    value = create_primary_admission(intent, qualification)
else:
    raise ValueError("mode must be qualification or primary")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
dbutils.notebook.exit(json.dumps({"status": "written", "sha256":
                                  value.get("receipt_sha256") or value["admission_sha256"]}))
