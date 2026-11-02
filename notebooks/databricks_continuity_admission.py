# Databricks notebook source
import hashlib, json, re, sys
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
raw_intent = json.loads(Path(values["intent_path"]).read_text(encoding="utf-8"))


def bootstrap_verify_source(root_value, authority):
    root = Path(root_value).resolve()
    manifest = authority.get("source_manifest", {})
    payload = dict(manifest); embedded = payload.pop("manifest_sha256", None)
    commit = authority.get("source_commit"); files = manifest.get("files")
    critical = (
        "src/__init__.py", "src/action_serialization.py",
        "src/base_qualification_carryforward.py",
        "src/databricks_simulated_continuity.py",
        "src/databricks_continuity_partition.py",
        "src/databricks_continuity_partition_worker.py",
        "src/qualification_technical_successor.py",
        "src/qwen9b_ordinal_continuity_controller.py",
        "src/qwen9b_ordinal_continuity_lora.py",
        "src/qwen9b_ordinal_continuity_runtime.py",
        "src/qwen9b_ordinal_continuity_training.py",
        "src/qwen9b_ordinal_continuity_worker.py",
        "src/volume_durable_io.py",
    )
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    if (not re.fullmatch(r"[0-9a-f]{40}", str(commit or ""))
            or root.as_posix() !=
               f"/Workspace/Shared/research4/source_releases/{commit}"
            or authority.get("source_release_path") != root.as_posix()
            or manifest.get("source_commit") != commit
            or authority.get("source_sha256") != embedded
            or hashlib.sha256(canonical).hexdigest() != embedded
            or not isinstance(files, list)):
        raise PermissionError("Independent admission source authority differs")
    indexed = {row.get("path"): row for row in files
               if isinstance(row, dict)}
    for logical in critical:
        row = indexed.get(logical); path = root / logical
        if (not isinstance(row, dict) or not path.is_file()
                or path.is_symlink()
                or set(row) != {"path", "sha256", "canonical_bytes"}):
            raise PermissionError("Critical admission source is absent")
        data = path.read_text(encoding="utf-8").replace(
            "\r\n", "\n").replace("\r", "\n").encode("utf-8")
        if (len(data) != row.get("canonical_bytes")
                or hashlib.sha256(data).hexdigest() != row.get("sha256")):
            raise PermissionError("Critical admission source changed before import")
    return embedded


bootstrap_source_sha256 = bootstrap_verify_source(
    values["code_path"], raw_intent)
sys.path.insert(0, values["code_path"])

from src.databricks_simulated_continuity import (
    create_primary_admission, create_qualification_receipt,
    verify_deployed_source)
from src.databricks_continuity_partition import _verify_durable_completion
from src.qwen9b_ordinal_continuity_runtime import directory_tree_sha256
from src.qwen9b_ordinal_continuity_training import TrainingPlan

intent = raw_intent
source = verify_deployed_source(values["code_path"], intent)
if source["manifest_sha256"] != bootstrap_source_sha256:
    raise PermissionError("Admission bootstrap and full source verification differ")
output = Path(values["output_path"])
if output.exists():
    raise FileExistsError("Admission evidence is immutable")
if values["mode"] == "qualification":
    root = Path(values["qualification_output"])
    plan = TrainingPlan(**intent["plan"]).validate()
    _verify_durable_completion(
        root.parent, identity=intent["identity"],
        intent_sha256=intent["intent_sha256"],
        plan_sha256=intent["plan_sha256"],
        optimizer_updates=plan.optimizer_updates)
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
data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
try:
    with output.open("xb") as stream:
        stream.write(data)
        stream.flush()
    if output.read_bytes() != data:
        raise IOError("Sequential admission evidence readback differs")
except FileExistsError:
    raise
except BaseException:
    output.unlink(missing_ok=True)
    raise
dbutils.notebook.exit(json.dumps({"status": "written", "sha256":
                                  value.get("receipt_sha256") or value["admission_sha256"]}))
