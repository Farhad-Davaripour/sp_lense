"""Reviewed zero-update technical successor for the consumed rank-4 qualifier."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re


VERSION = "research4-qualification-technical-successor-v1"
PRIOR_IDENTITY = "qualification_rank4"
SUCCESSOR_IDENTITY = "qualification_rank4_parent_mlflow_v2"


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()


def authority() -> dict:
    value = {
        "version": VERSION,
        "prior_identity": PRIOR_IDENTITY,
        "successor_identity": SUCCESSOR_IDENTITY,
        "prior_source_commit": "c0854d9d2647a6610f8c5e011e354cb949d290e5",
        "prior_plan_sha256": "84b9170f742b4e90ba119a9c0679887aa77610144b576019c261d6b443f68c19",
        "prior_intent_sha256": "6c018f10fe42008ece25b75ff5d2eb0e2980d8c287f57d0f49c085010f487f8a",
        "owner_run_id": "635802519363060",
        "preflight_run_id": "399214681735642",
        "gpu_task_run_id": "784166844712827",
        "cleanup_run_id": "467030032743426",
        "mlflow_run_id": "f0703a8f787c4a9f95a160296808f9f6",
        "closeout_receipt_sha256": "50f8da18bbb905df86e0c96f9593979e95df61187ac4d5e45d06c505fd7199bf",
        "closeout_file_sha256": "63452e837cd2233775f9b9e366c0348acc7097b953a38363776d29797524fe27",
        "consumed_file_sha256": "ae3f7baf8dff00d05ac9978090e8e49a4caaadbbbd8fc85569ff5c4e63a865a7",
        "gpu_task_output_file_sha256": "527bdd3dbbd347f996f5f1695bc790af3991d2bfa0d85872547b86a7d64624f0",
        "terminal_file_sha256": "b775b1db87159207ea879776cfbc2621d4e217c8b12990daf79247ac16cf979f",
        "prior_worker_source_sha256": "3d60eb0fd370f44d262323123179d7d2a9dd04d69cbdc5cb7dc5e79bb3a9ef95",
        "prior_worker_ordering_proof_sha256": "acf7f056708cd393e98440d1f2583a2b782f38b15d8405be92c9631ddb237c24",
        "failure_class": "child_mlflow_auth_before_model",
        "model_loads": 0, "model_forwards": 0, "backward_passes": 0,
        "optimizer_updates": 0, "checkpoint_written": False,
        "training_receipt_written": False, "evidence_written": False,
        "operator_stop_sent": False, "ceiling_kill_sent": False,
        "child_exit_confirmed": True, "reservation_released": True,
        "consumed_identity_retained": True, "automatic_retry": False,
    }
    value["technical_successor_sha256"] = _digest(value)
    return value


def validate(value: Mapping) -> dict:
    actual = dict(value) if isinstance(value, Mapping) else {}
    expected = authority()
    payload = dict(actual); embedded = payload.pop("technical_successor_sha256", None)
    if (actual != expected or embedded != _digest(payload)
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(actual.get(key, "")))
                   for key in (
                       "prior_plan_sha256", "prior_intent_sha256",
                       "closeout_receipt_sha256", "closeout_file_sha256",
                       "consumed_file_sha256", "gpu_task_output_file_sha256",
                       "terminal_file_sha256", "prior_worker_source_sha256",
                       "prior_worker_ordering_proof_sha256",
                       "technical_successor_sha256"))):
        raise PermissionError("Rank-4 technical successor authority differs")
    return actual
