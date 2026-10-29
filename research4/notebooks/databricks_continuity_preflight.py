# Databricks notebook source
import hashlib, json, sys
from pathlib import Path
import re

for name in ("code_path", "intent_path", "plan_path", "admission_path",
             "parent_checkpoint", "qualification_intent_path",
             "qualification_output", "qualification_receipt_path",
             "base_qualification_receipt_path",
             "expected_mode"):
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name) for name in (
    "code_path", "intent_path", "plan_path", "admission_path", "parent_checkpoint",
    "qualification_intent_path", "qualification_output",
    "qualification_receipt_path", "base_qualification_receipt_path")}
if not values["code_path"] or not values["intent_path"] or not values["plan_path"]:
    raise ValueError("Exact code, intent and plan paths are required")
raw_intent = json.loads(Path(values["intent_path"]).read_text(encoding="utf-8"))


def _bootstrap_verify_source(root_value, authority):
    root = Path(root_value).resolve(); manifest = authority.get("source_manifest", {})
    payload = dict(manifest); embedded = payload.pop("manifest_sha256", None)
    commit = authority.get("source_commit"); files = manifest.get("files")
    canonical_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    critical = (
        "src/__init__.py", "src/action_serialization.py", "src/databricks_simulated_continuity.py",
        "src/databricks_continuity_partition.py",
        "src/databricks_continuity_partition_worker.py",
        "src/qwen9b_ordinal_continuity_runtime.py",
        "src/qwen9b_ordinal_continuity_training.py",
        "src/qwen9b_ordinal_continuity_lora.py",
        "src/qwen9b_ordinal_continuity_worker.py",
    )
    if (not re.fullmatch(r"[0-9a-f]{40}", str(commit or ""))
            or not root.as_posix().endswith(f"/research4/source_releases/{commit}")
            or authority.get("source_sha256") != embedded
            or hashlib.sha256(canonical_bytes).hexdigest() != embedded
            or not isinstance(files, list)):
        raise PermissionError("Independent partition source authority differs")
    indexed = {row.get("path"): row for row in files if isinstance(row, dict)}
    for logical in critical:
        row = indexed.get(logical); path = root / logical
        if not isinstance(row, dict) or not path.is_file() or path.is_symlink():
            raise PermissionError("Critical partition source is absent")
        data = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace(
            "\r", "\n").encode("utf-8")
        if (len(data) != row.get("canonical_bytes")
                or hashlib.sha256(data).hexdigest() != row.get("sha256")):
            raise PermissionError("Critical partition source changed before import")
    return embedded


bootstrap_source_sha256 = _bootstrap_verify_source(values["code_path"], raw_intent)
sys.path.insert(0, values["code_path"])

from src.databricks_simulated_continuity import (
    GOLD_RUN_TABLE, MODEL_RECEIPT, SILVER_TABLE,
    digest, expected_predecessor_binding,
    validate_base_qualification_receipt, validate_exact_gold_row,
    validate_intent, validate_primary_authority_files,
    validate_policy_silver_payload, verify_deployed_source,
    verify_model_receipt)
from src.qwen9b_ordinal_continuity_runtime import directory_tree_sha256
from src.qwen9b_ordinal_continuity_training import (
    TrainingPlan, canonical_sha256, validate_sequential_parent)
from pyspark.sql import functions as F

intent = validate_intent(raw_intent)
if not values["base_qualification_receipt_path"]:
    raise PermissionError("Training requires the frozen-base format gate")
base_qualification = validate_base_qualification_receipt(json.loads(Path(
    values["base_qualification_receipt_path"]).read_text()))
if (base_qualification["receipt_sha256"]
        != intent["base_qualification_receipt_sha256"]
        or base_qualification["base_qualification_intent_sha256"]
           != intent["base_qualification_intent_sha256"]):
    raise PermissionError("Training intent and frozen-base gate differ")
plan = json.loads(Path(values["plan_path"]).read_text())
if intent["plan"]["mode"] != dbutils.widgets.get("expected_mode"):
    raise PermissionError("Job resource does not admit this partition mode")
if intent["plan"] != plan:
    raise PermissionError("Intent and plan differ")
source_proof = verify_deployed_source(values["code_path"], intent)
if source_proof["manifest_sha256"] != bootstrap_source_sha256:
    raise PermissionError("Partition bootstrap and full source verification differ")
proof = verify_model_receipt(intent["model_root"], MODEL_RECEIPT)
if (proof["tree_sha256"] != intent["model_tree_sha256"]
        or proof["receipt_sha256"] != intent["model_receipt_sha256"]):
    raise PermissionError("Intent and volume model receipt differ")
base_rows = spark.table(GOLD_RUN_TABLE).where(
    F.col("partition_identity") == base_qualification["identity"]).collect()
validate_exact_gold_row(base_rows, {
    "partition_identity": base_qualification["identity"],
    "mode": "base_qualification", "condition": "frozen_base",
    "stage": "format_schema", "seed": 0,
    "plan_sha256": base_qualification["plan_sha256"],
    "source_commit": intent["source_commit"],
    "model_tree_sha256": intent["model_tree_sha256"],
    "checkpoint_tree_sha256": intent["model_tree_sha256"],
    "receipt_sha256": base_qualification["receipt_sha256"],
    "evidence_sha256": base_qualification["raw_results_sha256"],
    "intent_sha256": base_qualification["base_qualification_intent_sha256"],
}, label="frozen-base format qualification")
if plan["mode"] == "qualification" and any(values[name] for name in (
        "admission_path", "parent_checkpoint", "qualification_intent_path",
        "qualification_output", "qualification_receipt_path")):
    raise PermissionError("Qualification forbids primary authority and parent")
if plan["mode"] == "primary" and any(not values[name] for name in (
        "admission_path", "qualification_intent_path", "qualification_output",
        "qualification_receipt_path")):
    raise PermissionError("Primary partition requires full qualification authority")
if plan["mode"] == "primary" and plan["stage"] == "delta1" and values["parent_checkpoint"]:
    raise PermissionError("Delta1 must start from the frozen base")
if plan["mode"] == "primary" and plan["stage"] != "delta1" and not values["parent_checkpoint"]:
    raise PermissionError("Delta2/Delta3 require the exact predecessor")
if plan["mode"] == "primary":
    authority = validate_primary_authority_files(
        intent, values["admission_path"], values["qualification_intent_path"],
        values["qualification_output"], values["qualification_receipt_path"])
    admission = authority["admission"]
    qualification_intent = authority["qualification_intent"]
    qualification_receipt = authority["qualification_receipt"]
    qualification_rows = spark.table(GOLD_RUN_TABLE).where(
        F.col("partition_identity") == qualification_intent["identity"]).collect()
    validate_exact_gold_row(qualification_rows, {
        "partition_identity": qualification_intent["identity"],
        "mode": "qualification", "plan_sha256": qualification_intent["plan_sha256"],
        "source_commit": intent["source_commit"],
        "model_tree_sha256": intent["model_tree_sha256"],
        "checkpoint_tree_sha256": qualification_receipt[
            "qualification_checkpoint_tree_sha256"],
        "receipt_sha256": qualification_receipt[
            "qualification_rank_receipt_sha256"],
        "evidence_sha256": qualification_receipt[
            "qualification_evidence_sha256"],
        "intent_sha256": qualification_intent["intent_sha256"],
    }, label="qualification")
    parent = (None if not values["parent_checkpoint"] else json.loads(
        (Path(values["parent_checkpoint"]) / "adapter_config.json").read_text()))
    validate_sequential_parent(
        TrainingPlan(**plan).validate(), parent,
        source_sha256=intent["source_sha256"])
    predecessor = expected_predecessor_binding(plan)
    if predecessor is not None:
        parent_tree = directory_tree_sha256(values["parent_checkpoint"])
        predecessor_rows = spark.table(GOLD_RUN_TABLE).where(
            F.col("partition_identity") == predecessor["partition_identity"]).collect()
        predecessor_row = validate_exact_gold_row(predecessor_rows, {
            **predecessor, "source_commit": intent["source_commit"],
            "model_tree_sha256": intent["model_tree_sha256"],
            "checkpoint_tree_sha256": parent_tree,
            "qualification_receipt_sha256": qualification_receipt["receipt_sha256"],
        }, label="sequential predecessor")
        if (not re.fullmatch(r"[0-9a-f]{64}", predecessor_row.get(
                "admission_sha256") or "")
                or not re.fullmatch(r"[0-9a-f]{64}", predecessor_row.get(
                    "intent_sha256") or "")):
            raise PermissionError("Predecessor Gold lineage is incomplete")
kind = plan["condition"]
selected = spark.table(SILVER_TABLE).where(
    (F.col("source_commit") == intent["source_commit"])
    & (F.col("record_kind") == kind) & (F.col("split") == "train")
    & (F.col("admitted_for_training") == F.lit(True))).select(
        "dataset_tree_sha256", "record_id", "record_kind", "condition",
        "split", "family_id", "danger_level", "record_sha256", "record_json"
    ).orderBy("record_id").collect()
for row in selected:
    if (row.dataset_tree_sha256 != intent["dataset_tree_sha256"]
            or hashlib.sha256(row.record_json.encode("utf-8")).hexdigest()
               != row.record_sha256):
        raise PermissionError("Pre-GPU Silver payload hash or dataset tree differs")
    validate_policy_silver_payload(
        row.record_json, record_id=row.record_id,
        expected_condition=plan["condition"], expected_split=row.split,
        expected_family_id=row.family_id,
        expected_danger_level=row.danger_level)
    if (row.record_kind != plan["condition"]
            or row.condition != plan["condition"]
            or row.split != "train"):
        raise PermissionError("Pre-GPU Silver metadata semantics differ")
binding = [{"record_id": row.record_id, "record_sha256": row.record_sha256}
           for row in selected]
if len(binding) != intent["silver_record_count"] or digest(binding) != intent["silver_records_sha256"]:
    raise PermissionError("Pre-GPU Silver record binding differs")
dbutils.notebook.exit(json.dumps({"status": "validated_not_reserved",
                                  "identity": intent["identity"],
                                  "max_gpu_hours": intent["max_gpu_hours"],
                                  "source_manifest_sha256": source_proof[
                                      "manifest_sha256"]}))
