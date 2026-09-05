"""Read-only pinned inputs for a separately authorized post-hoc external audit.

No recording-budget attachment, lock acquisition, model imports or writes.
The original recording fault is authenticated, never repaired or overridden.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
import subprocess
from pathlib import Path

COMMIT = "541af36a5eaac261c7d59d27418805718fd95dd5"
LOCK_SHA = "8e0812645d457cb77a31da33cd41aadfbb749933300963f5961c7b7398a385e4"
INVENTORY_SHA = "27582f0327eb6116277ea8232c14b785d3f71aa05f3154c3a3afbed7853a5e39"
NAMESPACE = "evidence/paired_common_drift_comply_three_family_1800_clean01_v1_qwen35_08b"
QUOTAS = {
    "logits": 214616520,
    "rows": 226492416,
    "updates": 67108864,
    "workerlog": 4194304,
    "metadata": 2097152,
    "events": 1048576,
    "endpoint": 1048576,
    "final": 5242880,
    "receipts": 1048576,
    "control": 1048576,
    "scratch": 1048576,
    "total": 536870912,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe(root, relative):
    require(
        isinstance(relative, str)
        and relative
        and "\\" not in relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        "safe relative artifact/source path",
    )
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), "contained input path")
    for part in (path, *path.parents):
        if part == root.parent:
            break
        info = part.lstat()
        require(
            not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400,
            "no reparse input",
        )
        if part == root:
            break
    return path


def category(name):
    if name.startswith("logits/"):
        require(name in {f"logits/{i:02d}.f32.zlib" for i in range(1, 217)}, "canonical logits")
        return "logits"
    groups = {
        "rows": "rows.jsonl",
        "updates": "updates.jsonl",
        "workerlog": "worker.log",
        "metadata": "preregistration.json RUN_STARTED.json WORKER_CLAIM.json storage_preflight.json runtime.json analysis.json result.json RUN_STATUS.json INVALID.json VERIFICATION_FAILURE.json",
        "events": "forward_events.jsonl derivative_events.jsonl skip_events.jsonl",
        "endpoint": "endpoint.json",
        "final": "verification.json PILOT_REPORT.md comply_vector.json candidate_freeze.json CLOSEOUT.md CLOSEOUT.json",
        "receipts": "capture_receipt.json RECORDING_FAILURE.json FINAL_INVENTORY.json",
        "control": "recording_state.json recording.lock",
        "scratch": "recording_scratch.tmp",
    }
    matches = [key for key, names in groups.items() if name in names.split()]
    require(len(matches) == 1, "scoped original artifact")
    return matches[0]


def git(root, *args, input=None):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        input=input,
        capture_output=True,
        check=True,
        timeout=15,
    ).stdout


def authenticate(root: Path, original: Path) -> dict:
    root, original = Path(root).resolve(), Path(original).resolve()
    require(original == root / NAMESPACE, "exact immutable original namespace")
    require(
        digest(original / "preregistration.json") == LOCK_SHA
        and digest(original / "FINAL_INVENTORY.json") == INVENTORY_SHA,
        "pinned lock/inventory",
    )
    lock, inventory = (
        read(original / "preregistration.json"),
        read(original / "FINAL_INVENTORY.json"),
    )
    require(
        inventory["phase"] == "SEALED"
        and inventory["quiescent"] is True
        and inventory["fault_code"] == "ARTIFACT_BYTE_CAP"
        and inventory["valid_candidate"] is False
        and inventory["inventory_self_hash"] is None,
        "preserved original failed disposition",
    )
    entries = {item["path"]: item for item in inventory["files"]}
    actual = {p.relative_to(original).as_posix() for p in original.rglob("*") if p.is_file()}
    require(
        len(entries) == len(inventory["files"]) == len(actual) == 237 and set(entries) == actual,
        "exact237 inventory files",
    )
    require(
        {p.relative_to(original).as_posix() for p in original.rglob("*") if p.is_dir()}
        == {"logits"},
        "only original logits subdirectory",
    )
    hashes, totals, sizes = {}, {key: 0 for key in QUOTAS if key != "total"}, {}
    for name in sorted(actual):
        path = safe(original, name)
        size = path.stat().st_size
        value = digest(path)
        require(size == entries[name]["bytes"], "inventory byte size")
        require(
            value == (INVENTORY_SHA if name == "FINAL_INVENTORY.json" else entries[name]["sha256"]),
            "inventory content hash",
        )
        if name == "FINAL_INVENTORY.json":
            require(entries[name]["sha256"] is None, "nonrecursive original self hash")
        hashes[name], sizes[name] = value, size
        totals[category(name)] += size
    require(
        totals == inventory["category_bytes"]
        and sum(sizes.values()) == inventory["total_bytes"]
        and all(value <= QUOTAS[key] for key, value in totals.items())
        and sum(sizes.values()) <= QUOTAS["total"],
        "original file/category/total quotas",
    )
    policy = lock["plan"]["recording_policy"]
    require(
        policy["quotas"] == QUOTAS
        and all(
            size <= policy["logit_arrays"]["individual_compressed_cap_bytes"]
            for name, size in sizes.items()
            if name.startswith("logits/")
        ),
        "pinned quotas and individual raw-array caps",
    )
    require(
        sizes["recording_state.json"] == 4096
        and sizes["recording.lock"] == 1
        and sizes["verification.json"] == 0,
        "fixed controls and empty original verification",
    )
    require(
        not any(name in actual for name in ("comply_vector.json", "candidate_freeze.json")),
        "no original candidate",
    )
    state, failure = (
        read(original / "recording_state.json"),
        read(original / "RECORDING_FAILURE.json"),
    )
    require(
        state["phase"] == "SEALED"
        and state["fault_code"] == "ARTIFACT_BYTE_CAP"
        and failure["status"] == "INCONCLUSIVE",
        "original recording failure unchanged",
    )
    source = lock["source_sha256"]
    require(len(source) == 146, "exact original146 source inputs")
    for name, expected in source.items():
        require(digest(safe(root, name)) == expected, "original frozen source SHA256")
    require(
        git(root, "rev-parse", COMMIT + "^{commit}").decode().strip() == COMMIT, "pinned commit"
    )
    tree = {}
    for record in git(root, "ls-tree", "-rz", "--full-tree", COMMIT).split(b"\0"):
        if record:
            header, path = record.split(b"\t", 1)
            mode, kind, oid = header.decode().split()
            tree[path.decode()] = (mode, kind, oid)
    require(
        {name[len(NAMESPACE) + 1 :] for name in tree if name.startswith(NAMESPACE + "/")} == actual,
        "exact committed original namespace",
    )
    paths = sorted(set(source) | {NAMESPACE + "/" + name for name in actual})
    require(
        all("\n" not in name and "\r" not in name and not name.startswith('"') for name in paths),
        "unambiguous raw Git paths",
    )
    oids = (
        git(
            root,
            "hash-object",
            "--no-filters",
            "--stdin-paths",
            input=("\n".join(paths) + "\n").encode(),
        )
        .decode()
        .splitlines()
    )
    require(
        len(oids) == len(paths)
        and all(
            name in tree
            and tree[name][0] in {"100644", "100755"}
            and tree[name][1:] == ("blob", oid)
            for name, oid in zip(paths, oids, strict=True)
        ),
        "raw no-filter Git blob identity",
    )
    require(
        len(json.dumps({"source_sha256": source, "artifact_sha256": hashes}).encode()) <= 262144,
        "compact hash maps",
    )
    return {
        "plan": lock["plan"],
        "source_sha256": source,
        "artifact_sha256": hashes,
        "original_total_bytes": sum(sizes.values()),
        "original_file_count": len(actual),
        "commit": COMMIT,
        "commit_identity": "raw Git blob identity matched",
        "original_status": "INCONCLUSIVE",
        "original_fault": "ARTIFACT_BYTE_CAP",
    }


def check_runtime(root, original, plan, rows, rows_at):
    root, original = Path(root), Path(original)
    started, runtime, capture, meta, storage, lock = [
        read(original / name)
        for name in (
            "RUN_STARTED.json",
            "RUN_STATUS.json",
            "capture_receipt.json",
            "runtime.json",
            "storage_preflight.json",
            "preregistration.json",
        )
    ]
    start, deadline = started["started_monotonic"], started["deadline_monotonic"]
    require(
        all(type(x) in (int, float) and math.isfinite(x) for x in (start, deadline))
        and deadline == start + 1800
        and started["timeout_seconds"] == 1800
        and started["forward_ceiling"] == 216
        and started["derivative_ceiling"] == 96,
        "exact original1800 deadline and216/96 caps",
    )
    usage = started["usage_preflight"]
    require(
        type(usage["standard_used_percent"]) in (int, float)
        and 0 <= usage["standard_used_percent"] < 90
        and math.isfinite(usage["checked_at_unix"]),
        "recorded finite below90 usage",
    )
    require(
        runtime["status"] == capture["status"] == "complete_valid"
        and runtime["forward_attempts"] == runtime["completed_forwards"] == len(rows) == 216
        and runtime["derivative_attempts"] == 96
        and runtime["skipped_cells"] == 0
        and runtime["retries_allowed"] is False
        and runtime["reason"] is None
        and runtime["cleanup_error"] is None
        and 0 <= runtime["elapsed_seconds"] <= 1800
        and 0 <= capture["elapsed_seconds"] <= 1800,
        "complete captured worker, not valid recording",
    )
    require(
        capture["process_attempts"] == 1
        and capture["worker_exit_code"] == 0
        and all(
            capture[key] is True
            for key in (
                "worker_started",
                "worker_joined",
                "reader_joined",
                "prefix_reader_joined",
                "budget_watcher_joined",
                "fault_writer_joined",
                "quiescent",
                "eof_observed",
                "observed_snapshot_stable",
            )
        )
        and all(
            capture[key] is False
            for key in (
                "unread_tail_possible",
                "fault_persistence_error",
                "terminated",
                "termination_attempted",
                "kill_attempted",
            )
        )
        and all(
            capture[key] is None
            for key in ("cleanup_error", "exception", "technical_recording_fault")
        ),
        "one fully joined capture with EOF",
    )
    log = original / "worker.log"
    require(
        all(
            capture[key] == log.stat().st_size
            for key in ("captured_prefix_bytes", "full_output_bytes", "total_observed_bytes")
        )
        and all(
            capture[key] == digest(log)
            for key in ("captured_prefix_sha256", "full_output_sha256", "observed_bytes_sha256")
        ),
        "complete raw worker log length/hash",
    )
    for name, count in (("forward_events.jsonl", 216), ("derivative_events.jsonl", 96)):
        events = rows_at(original / name)
        require(len(events) == 2 * count, "exact journal count")
        for i in range(count):
            first, last = events[2 * i : 2 * i + 2]
            require(
                first["event"] == "attempt_started"
                and last["event"] == "attempt_completed"
                and first["attempt"] == last["attempt"] == i + 1
                and last.get("error") is None,
                "exact journal completion",
            )
        times = [event["monotonic"] for event in events]
        require(
            all(math.isfinite(t) and start <= t <= deadline for t in times)
            and times == sorted(times),
            "journal deadline/order",
        )
    require(
        not rows_at(original / "skip_events.jsonl")
        and len(rows_at(original / "updates.jsonl")) == 8
        and sum(row["condition"] == "final" for row in rows) == 12
        and sum(row["gradient"] is not None for row in rows) == 96,
        "8 updates12 finals0 skips",
    )
    require(
        meta["model_id"] == "Qwen/Qwen3.5-0.8B"
        and meta["model_revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
        and meta["device"] == "cpu"
        and meta["dtype"] == "float32"
        and meta["d_model"] == 1024
        and all(meta[key] == value for key, value in lock["environment"].items()),
        "pinned runtime/environment",
    )
    require(
        started["command"]
        == [
            meta["executable"],
            "-u",
            str(root / "scripts/paired_common_drift_comply_1800_clean01.py"),
            "_worker",
        ],
        "exact worker command",
    )
    bounds = plan["config"]["storage"]
    require(
        storage["bounds"] == bounds
        and storage["passed"] is True
        and storage["available_free_bytes"] >= bounds["minimum_free_bytes"]
        and start <= storage["monotonic"] <= deadline
        and all(row["logit_count"] == bounds["vocabulary"] for row in rows),
        "prospective storage and vocabulary",
    )
    return {
        "worker_capture_complete": True,
        "forwards": 216,
        "derivatives": 96,
        "updates": 8,
        "final_replays": 12,
        "skips": 0,
        "process_attempts": 1,
        "elapsed_seconds": runtime["elapsed_seconds"],
        "deadline_seconds": 1800,
        "full_worker_log_sha256": digest(log),
        "original_recording_remains": "INCONCLUSIVE ARTIFACT_BYTE_CAP",
    }
