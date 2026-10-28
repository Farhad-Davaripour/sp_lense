"""Kaggle-side bootstrap template for the frozen transfer evaluation.

The local preparer replaces every placeholder exactly once.  The bootstrap has
no provider API and cannot submit or retry itself.  It verifies attached bytes,
extracts only registered members, and starts one reviewed local worker process
under the declared timeout.  The worker is responsible only for frozen-model
inference over inert symbolic labels.
"""
from pathlib import Path
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile


SOURCE_COMMIT = "SOURCE_COMMIT_PLACEHOLDER"
SOURCE_ARCHIVE_SHA256 = "SOURCE_ARCHIVE_SHA256_PLACEHOLDER"
SEALED_ARCHIVE_SHA256 = "SEALED_ARCHIVE_SHA256_PLACEHOLDER"
CHECKPOINT_MANIFEST_SHA256 = "CHECKPOINT_MANIFEST_SHA256_PLACEHOLDER"
CHECKPOINT_ARCHIVE_SHA256 = "CHECKPOINT_ARCHIVE_SHA256_PLACEHOLDER"
RESOURCE_PLAN_SHA256 = "RESOURCE_PLAN_SHA256_PLACEHOLDER"
WORKER_MODULE = "WORKER_MODULE_PLACEHOLDER"
WORKER_SOURCE_SHA256 = "WORKER_SOURCE_SHA256_PLACEHOLDER"
RESOURCE_PLAN_JSON = "RESOURCE_PLAN_JSON_PLACEHOLDER"
PRE_SAVE_BINDING_JSON = "PRE_SAVE_BINDING_JSON_PLACEHOLDER"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def one(name, expected_sha):
    matches = [path for path in Path("/kaggle/input").rglob(name) if path.is_file()]
    assert len(matches) == 1, "Attach exactly one " + name
    assert sha(matches[0]) == expected_sha, "Attached " + name + " hash changed"
    return matches[0]


def extract_exact(archive, manifest, target):
    expected = json.loads(Path(manifest).read_text(encoding="utf-8"))["files"]
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
        assert len(names) == len(set(names)) and set(names) == set(expected)
        for name in names:
            assert "\\" not in name and ":" not in name
            assert not any(part in ("", ".", "..") for part in name.split("/"))
            raw = stream.read(name)
            assert hashlib.sha256(raw).hexdigest() == expected[name]
            output = (target / name).resolve()
            assert output.is_relative_to(target)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as handle:
                handle.write(raw)


start = time.monotonic()
plan = json.loads(RESOURCE_PLAN_JSON)
assert plan["setup_cap_seconds"] + plan["worker_cap_seconds"] + plan["controller_teardown_seconds"] <= plan["outer_ceiling_seconds"]
assert plan["operator_stop_offset_seconds"] + plan["protected_export_seconds"] + plan["controller_teardown_seconds"] <= plan["outer_ceiling_seconds"]
source_archive = one("CONTINUITY_TRANSFER_SOURCE.zip", SOURCE_ARCHIVE_SHA256)
sealed_archive = one("CONTINUITY_TRANSFER_SEALED_REGISTRY.zip", SEALED_ARCHIVE_SHA256)
checkpoint_manifest = one("CHECKPOINT_MANIFEST.json", CHECKPOINT_MANIFEST_SHA256)
checkpoint_archive = one("CONTINUITY_TRANSFER_CHECKPOINTS.zip", CHECKPOINT_ARCHIVE_SHA256)
resource_plan_path = one("RESOURCE_PLAN.json", RESOURCE_PLAN_SHA256)
source_manifest = one("SOURCE_MANIFEST.json", "SOURCE_MANIFEST_SHA256_PLACEHOLDER")
sealed_manifest = one("SEALED_MANIFEST.json", "SEALED_MANIFEST_SHA256_PLACEHOLDER")
source_root = Path("/kaggle/temp/research4_continuity_transfer_source")
sealed_root = Path("/kaggle/temp/research4_continuity_transfer_sealed")
checkpoint_root = Path("/kaggle/temp/research4_continuity_transfer_checkpoints")
extract_exact(source_archive, source_manifest, source_root)
extract_exact(sealed_archive, sealed_manifest, sealed_root)
extract_exact(checkpoint_archive, checkpoint_manifest, checkpoint_root)
assert json.loads(resource_plan_path.read_text(encoding="utf-8")) == plan
research4_root = (source_root / "research4").resolve()
assert research4_root.is_dir()
sys.path.insert(0, str(research4_root))
assert Path(sys.path[0]).resolve() == research4_root
importlib.invalidate_caches()
worker_module = importlib.import_module(WORKER_MODULE)
worker_expected = (source_root / "research4/src/continuity_transfer_inference_worker.py").resolve()
assert Path(worker_module.__file__).resolve() == worker_expected
assert sha(worker_expected) == WORKER_SOURCE_SHA256
assert sha(worker_module.__file__) == WORKER_SOURCE_SHA256
assert time.monotonic() - start < plan["setup_cap_seconds"], "Setup cap exhausted before worker start"

# A provider script/private version does not exist until this notebook has been
# saved.  The reviewed notebook therefore embeds only the immutable pre-Save
# candidate, preparation and expected input identity.  Exact provider/version
# identity is added later by the external reconciliation receipt and bound to
# these exported bytes by the independent audit.
run_binding = json.loads(PRE_SAVE_BINDING_JSON)
run_binding_path = Path("/kaggle/working/CONTINUITY_TRANSFER_RUN_BINDING.json")
run_binding_path.write_text(
    json.dumps(run_binding, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8")

env = {key: value for key, value in os.environ.items() if not key.startswith("RESEARCH4_")}
env.update({
    "RESEARCH4_EXECUTION_PLATFORM": "kaggle",
    "RESEARCH4_SOURCE_COMMIT": SOURCE_COMMIT,
    "RESEARCH4_TRANSFER_SOURCE_ROOT": str(source_root),
    "RESEARCH4_TRANSFER_SEALED_ROOT": str(sealed_root),
    "RESEARCH4_TRANSFER_CHECKPOINT_MANIFEST": str(checkpoint_manifest),
    "RESEARCH4_EXTERNAL_HARD_STOP": "enforced_outside_model_action_space",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
})
receipt = {
    "version": "research4-continuity-transfer-kaggle-v1",
    "source_commit": SOURCE_COMMIT,
    "worker_module": WORKER_MODULE,
    "status": "worker_not_started",
    "provider_calls": 0,
    "automatic_retries": 0,
    "external_hard_stop_outside_model_action_space": True,
}
output = Path("/kaggle/working/continuity_transfer")
output.mkdir(parents=True, exist_ok=False)
try:
    command = [
        sys.executable, "-m", WORKER_MODULE,
        "--sealed", str(sealed_root / "sealed"),
        "--checkpoint-manifest", str(checkpoint_manifest),
        "--checkpoint-root", str(checkpoint_root),
        "--output", str(output), "--source-commit", SOURCE_COMMIT,
        "--source-package-sha256", SOURCE_ARCHIVE_SHA256,
        "--checkpoint-package-sha256", CHECKPOINT_ARCHIVE_SHA256,
        "--resource-plan", str(resource_plan_path),
        "--run-binding", str(run_binding_path),
    ]
    completed = subprocess.run(command, cwd=source_root / "research4", env=env,
                               timeout=plan["worker_cap_seconds"], check=False)
    receipt.update(status="worker_exited", exit_code=completed.returncode,
                   timed_out=False, child_exit_confirmed=True)
except subprocess.TimeoutExpired:
    receipt.update(status="external_worker_timeout", exit_code=None,
                   timed_out=True, child_exit_confirmed=True)
finally:
    receipt["elapsed_seconds"] = time.monotonic() - start
    receipt["worker_receipt_present"] = (output / "WORKER_RECEIPT.json").is_file()
    receipt["raw_records_present"] = (output / "RAW_NATIVE_RECORDS.jsonl").is_file()
    (output / "CONTROLLER_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt))
assert receipt["status"] == "worker_exited" and receipt["exit_code"] == 0
