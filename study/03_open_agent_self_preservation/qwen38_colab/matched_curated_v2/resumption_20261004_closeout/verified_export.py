"""Trusted Colab closeout helper; run ONLY after research workers exit.

No model/service/key code. Uses the exact sealed legacy private exporter.
All failures and native activations/checkpoints stay in the private full archive.
The UI review packet excludes weights/tensor files and preserves full trajectories.
"""
import base64
import gzip
import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path

ROOT = Path("/content/sp_lense_work/qwen38_curated_v2_20261004T130200Z_c83530a1")
EXPORT_SOURCE_SHA = "b6878f71de6d5e23b0fd190766e9837591d885a95d4de8e30834cca643b6a76c"
LEGACY_BUNDLE_SHA = "bbffc9668ce7c251dcd7de5aad2255d1c02aed2089a9fe66f8689c1614191498"
REVIEW_MAX_GZIP_BYTES = 8 * 1024**2
CHUNK_CHARACTERS = 48000
EXPORT_RESERVE_UNITS = 1.5
RATE_UNITS_PER_HOUR = 6.77
ROOT_METADATA = {"config.json", "RESTORED_INPUTS.json", "EXPERIMENT_FREEZE.json",
    "SESSION_ADMISSION.json", "PARALLEL_SCHEDULE_AMENDMENT.json", "RESUME_CONFIG.json",
    "RESUME_A_NATIVE_MANIFEST.json", "PRIMARY_INTEGRITY_REPAIR_AMENDMENT.json"}
SOURCE_METADATA = {"FREEZE.json", "ANCHOR_SELECTION.json", "EVAL_FREEZE.json", "PHASE_ADMISSION.json"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, allow_nan=False).encode("utf-8") + b"\n")


def active_research_processes(root):
    """Only command lines of processes associated with this exact run are read."""
    results = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            command = (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            if str(root) not in command:
                continue
            state = (directory / "stat").read_text().rsplit(")", 1)[1].split()[0]
            if state == "Z":
                continue
            kinds = [name for name in ("eval_worker", "training_worker", "driver", "resume", "parallel_controller") if name in command]
            if kinds:
                results.append({"pid":int(directory.name), "kinds":kinds})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return results


def guard(root):
    if root.resolve() != ROOT or not root.is_dir():
        raise RuntimeError("Exact live Research3 run root is unavailable")
    active = active_research_processes(root)
    if active:
        raise RuntimeError("Research worker/driver still active: " + json.dumps(active))
    # PID receipts are checked by actual command/root association, not PID alone.
    receipts = []
    for path in (root / "reports").glob("*PID*.json"):
        item = json.loads(path.read_bytes())
        receipts.append({"receipt":str(path.relative_to(root)), "pid":item.get("pid")})
    return {"all_run_worker_and_driver_processes_exited":True,
            "scan":"/proc: exact run root plus eval_worker/training_worker/driver/resume/parallel_controller; zombies excluded; SIGSTOP parent remains active and blocks mounting",
            "pid_receipts_checked":receipts, "native_failures_will_be_preserved":True}


def selected_json(relative):
    parts = relative.parts
    if len(parts) == 1:
        return relative.name in ROOT_METADATA
    if parts[0] == "reports":
        return True
    if parts[0] in ("source", "source_resume", "source_integrity_v2"):
        return len(parts) == 2 and relative.name in SOURCE_METADATA
    if parts[0].startswith("eval"):
        return any(name in parts for name in ("reports", "evaluation", "generation_calls"))
    if parts[0].startswith("fit"):
        return "reports" in parts or ("training" in parts and "receipts" in parts)
    return False


def compact_capture(value):
    """Keep receipt metadata; full arrays remain native in the verified archive."""
    removed = {"expected_input_ids", "expected_attention_mask", "actual_outer_first_forward",
               "actual_language_first_forward"}
    result = {key:item for key,item in value.items() if key not in removed and key != "rows"}
    result["rows"] = []
    for row in value.get("rows", []):
        keep = {key:item for key,item in row.items() if key not in
                ("prompt_token_ids", "returned_token_ids", "logical_positions", "actual_unpadded_position_ids")}
        if "prompt_token_ids" in row:
            keep["prompt_tokens"] = len(row["prompt_token_ids"])
        if "returned_token_ids" in row and row["returned_token_ids"] is not None:
            keep["returned_tokens"] = len(row["returned_token_ids"])
        result["rows"].append(keep)
    result["large_input_arrays_omitted_from_review_copy_only"] = True
    return result


def build_review(root):
    files, manifest = {}, {}
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root)
        if not selected_json(relative) or any(part in ("model", "worker_home", "__pycache__") for part in relative.parts):
            continue
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RuntimeError("Unexpected review file type")
        blob = path.read_bytes()
        original_sha = sha_bytes(blob)
        destination = str(relative)
        mode = "full_native_utf8_json"
        if "generation_calls" in relative.parts and path.name in ("receipt.json", "GENERATION_CAPTURE.json"):
            blob = canonical(compact_capture(json.loads(blob)))
            destination += ".compact.json"
            mode = "derived_receipt_metadata; native arrays preserved privately and prompt/token IDs retained in full trajectories"
        files[destination] = blob.decode("utf-8")
        manifest[destination] = {"source_relative_path":str(relative), "source_bytes":info.st_size,
            "source_sha256":original_sha, "review_bytes":len(blob), "review_sha256":sha_bytes(blob), "mode":mode}
    trajectory_paths = [name for name in files if "/evaluation/trajectories/" in name]
    if not trajectory_paths:
        raise RuntimeError("No native complete trajectories selected for review")
    packet = {"run_root":str(root), "files":files, "manifest":manifest,
        "full_selected_trajectories":len(trajectory_paths), "weights_and_native_tensor_files_included":False,
        "native_activations_and_checkpoints_preserved_in_full_private_export":True,
        "historical_failure_receipts_retained":True, "source_bundle_sha256":LEGACY_BUNDLE_SHA}
    compressed = gzip.compress(canonical(packet), mtime=0)
    if len(compressed) > REVIEW_MAX_GZIP_BYTES:
        raise RuntimeError("Review packet exceeds frozen8MiB UI bound; native artifacts remain intact")
    destination = root / "reports" / "SELECTED_REVIEW_PACKET.json.gz"
    destination.write_bytes(compressed)
    meta = {"sha256":sha_bytes(compressed), "gzip_bytes":len(compressed),
        "selected_files":len(files), "full_trajectories":len(trajectory_paths),
        "maximum_gzip_bytes":REVIEW_MAX_GZIP_BYTES, "chunk_characters":CHUNK_CHARACTERS,
        "review_packet_contains_weights_or_tensor_files":False,
        "local_download_destination":"C:/Users/farha/Research3/artifacts; never OneDrive"}
    save(root / "reports" / "SELECTED_REVIEW_PACKET_RECEIPT.json", meta)
    return compressed, meta


def main():
    started = time.monotonic()
    worker_guard = guard(ROOT)
    export_path = ROOT / "source" / "export_private_runs.py"
    export_source = export_path.read_bytes()
    if sha_bytes(export_source) != EXPORT_SOURCE_SHA:
        raise RuntimeError("Exact sealed legacy private exporter source differs")
    save(ROOT / "reports" / "WORKERS_EXITED_FOR_EXPORT.json", worker_guard)
    compressed, metadata = build_review(ROOT)
    # Recheck immediately before trusted Drive mounting/copying.
    guard(ROOT)
    scope = {"ROOT":ROOT, "FAST_ROOT":ROOT, "EXPORT_ONLY_ROOTS":[ROOT],
             "RELEASE_RUNTIME_AFTER_EXPORT":False, "__name__":"trusted_curated_v2_export"}
    exec(compile(export_source, str(export_path), "exec"), scope)
    # Legacy exporter already verifies every copy, writes EXPORT_HASHES.json,
    # flushes/unmounts Drive; keep its immutable behavior and perform release here.
    receipt = json.loads((ROOT / "reports" / "PRIVATE_EXPORT_RECEIPT.json").read_bytes())
    if not receipt.get("model_workers_exited_before_mount") or not receipt.get("base_weights_excluded"):
        raise RuntimeError("Private export verification receipt is incomplete")
    archived = receipt["files"]
    selected_entry = archived.get("reports/SELECTED_REVIEW_PACKET.json.gz")
    if not selected_entry or selected_entry["sha256"] != metadata["sha256"]:
        raise RuntimeError("Review packet was not verified in the private archive")
    print("CURATED_V2_REVIEW_METADATA", json.dumps(metadata, separators=(",", ":")), flush=True)
    encoded = base64.b64encode(compressed).decode("ascii")
    chunks = [encoded[i:i+CHUNK_CHARACTERS] for i in range(0, len(encoded), CHUNK_CHARACTERS)]
    for index, chunk in enumerate(chunks):
        print("CURATED_V2_REVIEW_CHUNK", str(index+1)+"/"+str(len(chunks)), chunk, flush=True)
    seconds = time.monotonic() - started
    print("CURATED_V2_EXPORT_CLOSEOUT", json.dumps({"private_destination":receipt["destination"],
        "verified_files":len(archived), "verified_bytes":sum(item["bytes"] for item in archived.values()),
        "review_packet_sha256":metadata["sha256"], "review_chunks":len(chunks),
        "all_private_copies_verified":True, "drive_flushed_and_unmounted":True,
        "export_seconds":seconds, "estimated_export_units":seconds*RATE_UNITS_PER_HOUR/3600,
        "export_reserve_units":EXPORT_RESERVE_UNITS,
        "within_export_reserve":seconds*RATE_UNITS_PER_HOUR/3600<=EXPORT_RESERVE_UNITS}), flush=True)
    from google.colab import runtime
    print("REQUESTING_RUNTIME_RELEASE_AFTER_VERIFIED_CURATED_V2_EXPORT", flush=True)
    runtime.unassign()


if __name__ == "__main__":
    main()
