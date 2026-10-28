# Databricks notebook source
import json, sys
from pathlib import Path
import re

for name in ("code_path", "intent_path", "plan_path", "admission_path",
             "parent_checkpoint", "qualification_intent_path",
             "qualification_output", "qualification_receipt_path",
             "expected_mode"):
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name) for name in (
    "code_path", "intent_path", "plan_path", "admission_path", "parent_checkpoint",
    "qualification_intent_path", "qualification_output",
    "qualification_receipt_path")}
if not values["code_path"] or not values["intent_path"] or not values["plan_path"]:
    raise ValueError("Exact code, intent and plan paths are required")
sys.path.insert(0, values["code_path"])

from src.databricks_simulated_continuity import (
    GOLD_RUN_TABLE, MODEL_RECEIPT, SILVER_TABLE,
    digest, expected_predecessor_binding,
    validate_exact_gold_row, validate_intent, validate_primary_authority_files,
    verify_deployed_source, verify_model_receipt)
from src.qwen9b_ordinal_continuity_runtime import directory_tree_sha256
from src.qwen9b_ordinal_continuity_training import (
    TrainingPlan, canonical_sha256, validate_sequential_parent)
from pyspark.sql import functions as F

intent = validate_intent(json.loads(Path(values["intent_path"]).read_text()))
plan = json.loads(Path(values["plan_path"]).read_text())
if intent["plan"]["mode"] != dbutils.widgets.get("expected_mode"):
    raise PermissionError("Job resource does not admit this partition mode")
if intent["plan"] != plan:
    raise PermissionError("Intent and plan differ")
source_proof = verify_deployed_source(values["code_path"], intent)
proof = verify_model_receipt(intent["model_root"], MODEL_RECEIPT)
if (proof["tree_sha256"] != intent["model_tree_sha256"]
        or proof["receipt_sha256"] != intent["model_receipt_sha256"]):
    raise PermissionError("Intent and volume model receipt differ")
if plan["mode"] == "qualification" and any(values[name] for name in (
        "admission_path", "parent_checkpoint", "qualification_intent_path",
        "qualification_output", "qualification_receipt_path")):
    raise PermissionError("Qualification forbids primary authority and parent")
if plan["mode"] == "primary" and any(not values[name] for name in (
        "admission_path", "qualification_intent_path", "qualification_output",
        "qualification_receipt_path")):
    raise PermissionError("Primary partition requires full qualification authority")
if plan["mode"] == "primary" and plan["stage"] != "skill" and not values["parent_checkpoint"]:
    raise PermissionError("Policy partition requires its predecessor checkpoint")
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
kind = "neutral_skill" if plan["stage"] == "skill" else plan["condition"]
selected = spark.table(SILVER_TABLE).where(
    (F.col("source_commit") == intent["source_commit"])
    & (F.col("record_kind") == kind) & (F.col("split") == "train")
    & (F.col("admitted_for_training") == F.lit(True))).select(
        "record_id", "record_sha256").orderBy("record_id").collect()
binding = [{"record_id": row.record_id, "record_sha256": row.record_sha256}
           for row in selected]
if len(binding) != intent["silver_record_count"] or digest(binding) != intent["silver_records_sha256"]:
    raise PermissionError("Pre-GPU Silver record binding differs")
dbutils.notebook.exit(json.dumps({"status": "validated_not_reserved",
                                  "identity": intent["identity"],
                                  "max_gpu_hours": intent["max_gpu_hours"],
                                  "source_manifest_sha256": source_proof[
                                      "manifest_sha256"]}))
