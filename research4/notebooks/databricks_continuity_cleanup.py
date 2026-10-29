# Databricks notebook source
"""Independent ALL_DONE release; never imports the workload implementation."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


for name in ("code_path", "intent_path", "run_id"):
    dbutils.widgets.text(name, "")
values = {name: dbutils.widgets.get(name) for name in (
    "code_path", "intent_path", "run_id")}
if any(not value for value in values.values()):
    raise ValueError("Cleanup requires exact code, intent and job-run owner")

# Cleanup deliberately trusts only immutable intent bytes and the owner-bound
# lock schema reproduced here. Drifted workload modules cannot alter or block
# release of a paid-compute reservation.
intent = json.loads(Path(values["intent_path"]).read_text(encoding="utf-8"))
payload = dict(intent); intent_sha = payload.pop("intent_sha256", None)
manifest = intent.get("source_manifest", {})
manifest_payload = dict(manifest); manifest_sha = manifest_payload.pop(
    "manifest_sha256", None)
identity = intent.get("identity"); owner = values["run_id"]
commit = intent.get("source_commit")
if (intent_sha != digest(payload)
        or not re.fullmatch(r"[a-z0-9_.:-]{1,160}", str(identity or ""))
        or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", owner)
        or not re.fullmatch(r"[0-9a-f]{40}", str(commit or ""))
        or manifest.get("source_commit") != commit
        or manifest_sha != digest(manifest_payload)
        or intent.get("source_sha256") != manifest_sha
        or intent.get("source_release_path") != (
            f"/Workspace/Shared/research4/source_releases/{commit}")
        or Path(values["code_path"]).resolve().as_posix() != (
            f"/Workspace/Shared/research4/source_releases/{commit}")):
    raise PermissionError("Independent cleanup intent/source authority differs")

lock_root = Path(
    "/Volumes/research4_simulated_continuity/bronze/model_data/locks")
active = lock_root / "scientific.lock"
consumed = lock_root / "consumed" / f"{identity}.json"
if not consumed.exists():
    dbutils.notebook.exit(json.dumps({
        "status": "no_reservation_to_release", "identity": identity}))

expected = {"owner": owner, "identity": identity, "intent_sha256": intent_sha}


def check_reservation(value):
    if (any(value.get(key) != expected_value
            for key, expected_value in expected.items())
            or value.get("status") != "reserved_after_gpu_ready_before_model"):
        raise PermissionError("Databricks reservation ownership differs")


def check_closeout(value):
    current = dict(value); embedded = current.pop("receipt_sha256", None)
    if (current.get("version")
            != "research4-databricks-active-reservation-closeout-v1"
            or any(current.get(key) != expected_value
                   for key, expected_value in expected.items())
            or current.get("consumed_identity_retained") is not True
            or embedded != digest(current)):
        raise PermissionError("Databricks closeout receipt differs")
    return dict(value)


check_reservation(json.loads(consumed.read_text(encoding="utf-8")))
closeout_dir = lock_root / "closeouts" / identity
closeout_dir.mkdir(parents=True, exist_ok=True)
owner_key = hashlib.sha256(owner.encode("utf-8")).hexdigest()
closeout = closeout_dir / f"{owner_key}.json"
if closeout.exists():
    receipt = check_closeout(json.loads(closeout.read_text(encoding="utf-8")))
    if active.exists():
        current = json.loads(active.read_text(encoding="utf-8"))
        (check_closeout(current) if current.get("version") else
         check_reservation(current))
        active.unlink()
else:
    if not active.exists():
        raise PermissionError("Active reservation disappeared without closeout")
    current = json.loads(active.read_text(encoding="utf-8"))
    if current.get("version"):
        receipt = check_closeout(current)
    else:
        check_reservation(current)
        receipt = {
            "version": "research4-databricks-active-reservation-closeout-v1",
            **expected,
            "terminal_status": "job_all_done_cleanup",
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "consumed_identity_retained": True,
            "reservation_sha256": digest(current),
        }
        receipt["receipt_sha256"] = digest(receipt)
        pending = closeout_dir / f".pending-{owner_key}-{os.urandom(16).hex()}.json"
        descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical(receipt) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(pending, active)
    os.replace(active, closeout)
    receipt = check_closeout(json.loads(closeout.read_text(encoding="utf-8")))

check_reservation(json.loads(consumed.read_text(encoding="utf-8")))
status = "active_released_consumed_identity_retained"
gold_table = "research4_simulated_continuity.gold.execution_admissions"
safe_identity = identity.replace("'", "''")
existing = spark.table(gold_table).where(
    f"identity = '{safe_identity}' AND intent_sha256 = '{intent_sha}' "
    f"AND job_run_id = '{owner}' AND status = '{status}'").limit(1).count()
if existing == 0:
    spark.createDataFrame([{
        "identity": identity, "intent_sha256": intent_sha,
        "job_run_id": owner, "status": status,
        "consumed_at": datetime.now(timezone.utc),
    }], schema=spark.table(gold_table).schema).write.mode(
        "append").saveAsTable(gold_table)
dbutils.notebook.exit(json.dumps({
    "status": status, "identity": identity,
    "closeout_receipt_sha256": receipt["receipt_sha256"]}))
