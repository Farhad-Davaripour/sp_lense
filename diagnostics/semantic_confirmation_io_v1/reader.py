"""Independent saved-byte reader. Does not import writer, binding or helpers."""
import hashlib
import json
import math
import os
import stat
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_COMMIT = "2620f66d4d50456c88800554bc9a26845998cf50"
CONTRACT_PATH = "diagnostics/semantic_confirmation_resource_v1/contract.json"
CONTRACT_SHA256 = "7610b46b0248569ba51f3f90638734582202c7214822dadafd57fefabc41af3b"


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def contract_data():
    raw = subprocess.check_output(["git", "-C", str(ROOT), "show", CONTRACT_COMMIT + ":" + CONTRACT_PATH])
    demand(digest(raw) == CONTRACT_SHA256, "independent pinned contract hash")
    result = json.loads(raw)
    demand(result["study"]["vocabulary"] == 248320 and result["logit_codec"]["bytes_per_vector"] == 993280,
           "pinned complete vector dimensions")
    return result


def checked_relative(name):
    demand(isinstance(name, str) and name != "", "nonempty file name")
    demand(not any(char in name for char in '\x00<>:"|?*'), "unsafe path character")
    pieces = name.replace("\\", "/").split("/")
    blocked = ["con", "prn", "aux", "nul"] + [prefix + str(n) for prefix in ("com", "lpt") for n in range(1, 10)]
    for piece in pieces:
        demand(piece not in ("", ".", "..") and not piece.endswith((" ", ".")), "unsafe path component")
        demand(piece.split(".")[0].casefold() not in blocked, "Windows device path")
    return "/".join(pieces).casefold()


def saved_path(root, name):
    key = checked_relative(name)
    base = Path(root).resolve(strict=True)
    path = base / key
    demand(path.resolve(strict=False).is_relative_to(base), "saved path containment")
    for part in (path, *path.parents):
        if part == base:
            break
        if part.exists():
            info = part.lstat()
            demand(not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & 0x400), "linked saved path")
    if path.exists() and path.is_file():
        demand(path.stat().st_nlink == 1, "linked saved file")
    return path


def read_raw_logits(root, name, expected_sha256):
    c = contract_data()
    path = saved_path(root, name)
    demand(path.stat().st_size == 993280, "complete final-token raw byte length")
    raw = path.read_bytes()
    demand(len(raw) == c["study"]["vocabulary"] * 4 and digest(raw) == expected_sha256, "raw length/hash identity")
    values = struct.unpack("<248320f", raw)
    demand(all(math.isfinite(value) for value in values), "finite saved float32 evidence")
    return {"raw": raw, "values": values, "sha256": digest(raw)}


def verify_index(root, index_name, expected_index_sha256):
    c = contract_data()
    index_path = saved_path(root, index_name)
    raw_index = index_path.read_bytes()
    demand(digest(raw_index) == expected_index_sha256 and len(raw_index) <= c["storage"]["per_file_bytes"], "index hash/size")
    index = json.loads(raw_index)
    demand(index["schema"] == "sp_lense.confirmation_io_index.v1" and index["index_excludes_itself"] is True, "explicit index scope")
    demand(index["binding"]["resource_commit"] == CONTRACT_COMMIT and index["binding"]["contract_sha256"] == CONTRACT_SHA256, "index contract binding")
    expected_settings = {"seconds": c["seconds"], "storage": c["storage"], "study": c["study"],
        "codec": c["logit_codec"]["name"], "row_maximum_bytes": c["row_maximum_bytes"], "hook_metadata": c["hook_metadata"]}
    demand(index["settings"] == expected_settings and index["real_run_authorized"] is False, "independently fixed resource settings")
    errors, tracked, all_files = [], {}, {}
    for directory, folders, names in os.walk(Path(root), followlinks=False):
        for folder in folders:
            saved_path(root, (Path(directory) / folder).relative_to(root).as_posix())
        for name in names:
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            key = checked_relative(rel)
            demand(rel == key and key not in all_files, "noncanonical/duplicate saved names")
            actual = saved_path(root, rel).read_bytes()
            all_files[key] = {"bytes": len(actual), "sha256": digest(actual)}
    categories = {category: 0 for category in c["storage"]["categories_bytes"]}
    counts = {"logits": 0, "rows": 0}
    for item in index["reconciliation"]["files"]:
        name = checked_relative(item["path"])
        demand(name not in tracked and name != checked_relative(index_name), "duplicate/self-index entry")
        tracked[name] = item
        actual = all_files.get(name)
        if actual is None:
            errors.append("missing:" + name)
            continue
        if (actual["bytes"], actual["sha256"]) != (item["actual_bytes"], item["actual_sha256"]):
            errors.append("changed_since_index:" + name)
        category = item["category"]
        if category not in categories:
            errors.append("untracked_external:" + name)
        else:
            categories[category] += actual["bytes"]
        if category in counts:
            counts[category] += 1
        if category == "rows" and actual["bytes"] > c["row_maximum_bytes"]:
            errors.append("row_limit:" + name)
        if category == "logits" and (not item.get("logit") or not name.endswith(".f32")):
            errors.append("logit_codec_declaration:" + name)
        if category == "ledgers":
            profile = c["ledger_streams"].get(name)
            content = saved_path(root, name).read_bytes()
            chunks = content.splitlines(keepends=True) if name.endswith(".jsonl") else [content]
            if profile is None or len(chunks) > profile["maximum_records"] or any(len(part) > profile["maximum_record_bytes"] for part in chunks):
                errors.append("ledger_record_limit:" + name)
            if item["complete"] and (not content.endswith(b"\n") or any(not part.endswith(b"\n") for part in chunks)):
                errors.append("incomplete_record_boundary:" + name)
        if category == "logs":
            limit = c["log_streams"].get(name.removeprefix("logs/"))
            if limit is None or actual["bytes"] > limit:
                errors.append("log_stream_limit:" + name)
        if actual["bytes"] > c["storage"]["per_file_bytes"]:
            errors.append("file_limit:" + name)
        if not item["complete"]:
            errors.append("incomplete:" + name)
        elif (actual["bytes"], actual["sha256"]) != (item["expected_bytes"], item["expected_sha256"]):
            errors.append("expected_bytes_mismatch:" + name)
        if item.get("logit") and item["complete"]:
            try:
                read_raw_logits(root, name, item["expected_sha256"])
            except (OSError, ValueError) as error:
                errors.append("invalid_raw_logits:" + name + ":" + str(error))
    index_key = checked_relative(index_name)
    if set(all_files) != set(tracked) | {index_key}:
        errors.append("actual_file_membership_mismatch")
    categories["failure_closeout"] += len(raw_index)
    for category, used in categories.items():
        if used > c["storage"]["categories_bytes"][category]:
            errors.append("category_limit:" + category)
    if any(count > c["study"]["maximum_forwards"] for count in counts.values()):
        errors.append("evidence_file_count_limit")
    total = sum(item["bytes"] for item in all_files.values())
    if total > c["storage"]["total_bytes"]:
        errors.append("total_limit")
    if index["reconciliation"]["actual_total_bytes"] + len(raw_index) != total:
        errors.append("declared_total_mismatch")
    if index["status"] != "COMPLETE" or index["sticky_failure"] or index["failures"] or index["reconciliation"]["issues"]:
        errors.append("writer_declared_incomplete")
    return {"status": "COMPLETE" if not errors else "INCOMPLETE", "errors": errors,
            "actual_bytes": total, "actual_file_count": len(all_files), "category_bytes": categories,
            "all_files_accounted": set(all_files) == set(tracked) | {index_key},
            "steering_or_model_success_tested": False}
