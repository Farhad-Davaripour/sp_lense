"""ONE frozen <=60s filesystem batch, confined to this <=32MiB namespace."""
import copy
import hashlib
import json
import os
import struct
import sys
import threading
import time
from pathlib import Path

from binding import binding_record, load_bound_resource
from reader import read_raw_logits, verify_index
from writer import EvidenceIOError, EvidenceWriter

HERE = Path(__file__).resolve().parent
MIB = 1024 * 1024
STARTED = time.monotonic()
RESULTS = []


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_error(fn):
    try:
        fn()
    except (OSError, ValueError):
        return
    raise AssertionError("invalid operation unexpectedly succeeded")


def file_json(name, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    actual = sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
    require(len(raw) <= 5 * MIB and actual + len(raw) <= 32 * MIB, "preparation output ceiling")
    with (HERE / name).open("xb") as stream:
        stream.write(raw)


def vector():
    return struct.pack("<8f", 0.0, -0.0, 1.0, -1.0, 1 / 3, 2 ** -149,
                       3.4028234663852886e38, -3.4028234663852886e38) * (248320 // 8)


def writer(case, **kwargs):
    return EvidenceWriter(HERE / "fixtures" / case, preparation_root=HERE, **kwargs)


def audit(w, closeout):
    result = verify_index(w.root, closeout["path"], closeout["sha256"])
    require(result["actual_bytes"] == closeout["actual_total_including_index"], "actual writer/reader byte totals")
    return result


def successful_roundtrip():
    w = writer("success")
    raw = vector()
    expected = hashlib.sha256(raw).hexdigest()
    saved = w.write_logits("logits/001.f32", raw)
    w.write_row("rows/001.json", b'{"fixture":"bounded synthetic row"}\n')
    w.append_event("routing_events.jsonl", b'{"fixture":"route one"}\n')
    w.append_event("routing_events.jsonl", b'{"fixture":"route two"}\n')
    w.append_log("worker_stdout.log", b"bounded first line\n")
    w.append_log("worker_stdout.log", b"bounded second line\n")
    reread = read_raw_logits(w.root, "LOGITS/001.F32", expected)
    require(reread["raw"] == raw and saved["recorded_bytes"] == 993280, "all saved float32 bytes preserved")
    require(struct.pack("<f", reread["values"][1]) == b"\0\0\0\x80", "negative zero preserved")
    close = w.closeout()
    verified = audit(w, close)
    require(verified["status"] == "COMPLETE" and verified["all_files_accounted"], "complete independent saved audit")
    expect_error(lambda: w.write_row("rows/late.json", b"{}\n"))
    return {"vector_bytes": len(raw), "sha256": expected, "index": close, "reader": verified}


def corrupt_and_incomplete_reader():
    w = writer("corrupt")
    raw = vector()
    saved = w.write_logits("logits/001.f32", raw)
    close = w.closeout()
    # Explicit fault injection into NEW fixture bytes after successful closeout.
    # Retain the corrupt file and original expected hash, never repair it.
    with (w.root / "logits/001.f32").open("r+b") as stream:
        stream.seek(12)
        stream.write(b"\1")
    expect_error(lambda: read_raw_logits(w.root, "logits/001.f32", saved["expected_sha256"]))
    observed = verify_index(w.root, close["path"], close["sha256"])
    require(observed["status"] == "INCOMPLETE" and any("changed_since_index" in e for e in observed["errors"]), "corruption cannot pass")
    incomplete = HERE / "fixtures" / "incomplete_reader"
    incomplete.mkdir()
    (incomplete / "short.f32").write_bytes(raw[:17])
    expect_error(lambda: read_raw_logits(incomplete, "short.f32", hashlib.sha256(raw[:17]).hexdigest()))
    expect_error(lambda: verify_index(w.root, close["path"], "0" * 64))
    return {"corruption_retained": True, "reader": observed, "short_vector_bytes": 17,
            "index_hash_failure_rejected": True}


class FailAfterPrefix:
    def __init__(self, prefix):
        self.prefix = prefix
        self.calls = 0

    def __call__(self, descriptor, data):
        self.calls += 1
        if self.calls == 1:
            return os.write(descriptor, data[:self.prefix])
        raise OSError("injected write failure after retained prefix")


def partial_write_and_closeout():
    w = writer("partial_vector", write_function=FailAfterPrefix(4096))
    expect_error(lambda: w.write_logits("logits/001.f32", vector()))
    before = w.reconcile()
    require(w.sticky_failure and before["actual_total_bytes"] == 4096, "partial byte count observed")
    require(before["reserved"]["used_bytes"]["logits"] == 993280, "attempted full reservation remains distinct")
    require(before["reserved"]["remaining_closeout_bytes"] == 8 * MIB, "failure reserve untouched")
    expect_error(lambda: w.write_row("rows/blocked.json", b"{}\n"))
    require(not (w.root / "rows/blocked.json").exists(), "sticky failure blocks new ordinary files")
    close = w.closeout()
    observed = audit(w, close)
    require(observed["status"] == "INCOMPLETE" and observed["all_files_accounted"], "partial file honestly reconciled")
    require(close["remaining_closeout_bytes"] == 8 * MIB - close["bytes"], "only actual closeout spends reserve")
    return {"actual_prefix_bytes": 4096, "attempted_vector_bytes": 993280, "index": close,
            "reader": observed, "partial_file_retained": True}


def partial_append_and_cap():
    w = writer("partial_append")
    w.append_log("worker_stdout.log", b"original\n")
    w.write_function = FailAfterPrefix(3)
    expect_error(lambda: w.append_log("worker_stdout.log", b"appended\n"))
    actual = (w.root / "logs/worker_stdout.log").read_bytes()
    require(actual == b"original\napp", "existing prefix and partial append retained")
    require(w.reconcile()["actual_total_bytes"] == len(actual), "append actual size reconciled")
    partial_close = w.closeout()
    require(audit(w, partial_close)["status"] == "INCOMPLETE", "partial append not success")
    cap = writer("append_cap")
    cap.append_log("worker_stdout.log", b"x" * MIB)
    expect_error(lambda: cap.append_log("worker_stdout.log", b"x"))
    require((cap.root / "logs/worker_stdout.log").stat().st_size == MIB, "append rejected without truncation or extra byte")
    require(cap.reconcile()["reserved"]["remaining_closeout_bytes"] == 8 * MIB, "cap failure preserves closeout")
    cap_close = cap.closeout()
    require(audit(cap, cap_close)["status"] == "INCOMPLETE", "cap rejection retained in closeout")
    return {"partial_append_bytes": len(actual), "partial_index": partial_close,
            "exact_append_cap_bytes": MIB, "cap_index": cap_close}


def external_mismatch():
    w = writer("external_file")
    w.write_row("rows/001.json", b"{}\n")
    (w.root / "external.bin").write_bytes(b"new external fixture")
    seen = w.reconcile()
    require(w.sticky_failure and "untracked_external:external.bin" in seen["issues"], "external file visible")
    require(seen["actual_total_bytes"] == 23, "external bytes included")
    expect_error(lambda: w.append_log("worker_stdout.log", b"blocked\n"))
    close = w.closeout()
    read = audit(w, close)
    require(read["status"] == "INCOMPLETE" and read["all_files_accounted"], "external evidence accounted without adoption as valid")
    return {"actual_bytes_before_closeout": seen["actual_total_bytes"], "index": close, "reader": read}


def ownership_paths_and_settings():
    c, helper = load_bound_resource()
    configs = {n: helper.component_settings(c) for n in ("recorder", "worker_supervisor", "audit_supervisor", "saved_judge")}
    for component, field, value in (("worker_supervisor", "worker", 300), ("audit_supervisor", "audit", 90)):
        wrong = copy.deepcopy(configs)
        wrong[component]["seconds"][field] = value
        expect_error(lambda wrong=wrong: writer("bad_settings_" + field, component_settings=wrong))
    wrong = copy.deepcopy(configs)
    wrong["recorder"]["storage"]["total_bytes"] = 96 * MIB
    expect_error(lambda: writer("bad_old_budget", component_settings=wrong))
    for i, path in enumerate(("../escape.bin", "/absolute.bin", "C:/outside.bin", "aux.txt", "name.", "a/../b")):
        w = writer("bad_path_" + str(i))
        expect_error(lambda w=w, path=path: w.write_row(path, b"{}\n"))
        close = w.closeout()
        require(audit(w, close)["status"] == "INCOMPLETE", "path rejection retained")
    collision = writer("case_collision")
    collision.write_row("rows/one.json", b"first\n")
    expect_error(lambda: collision.write_row("ROWS/ONE.JSON", b"second\n"))
    require((collision.root / "rows/one.json").read_bytes() == b"first\n", "case alias cannot overwrite")
    collision_close = collision.closeout()
    own = writer("owner_thread")
    errors = []
    def foreign_write():
        try:
            own.write_row("rows/foreign.json", b"{}\n")
        except EvidenceIOError as error:
            errors.append(str(error))
    thread = threading.Thread(target=foreign_write)
    thread.start()
    thread.join(2)
    require(not thread.is_alive() and len(errors) == 1 and not (own.root / "rows/foreign.json").exists(), "foreign owner cannot write")
    own.write_row("rows/owner.json", b"{}\n")
    own_close = own.closeout()
    require(audit(own, own_close)["status"] == "COMPLETE", "authorized owner remains usable")
    return {"bad_paths_rejected": 6, "old_settings_rejected": 3, "case_collision_index": collision_close,
            "owner_thread_index": own_close, "foreign_write_rejected": True}


def main():
    require(not (HERE / "BATCH_STARTED.json").exists(), "one batch only")
    freeze_bytes = (HERE / "SOURCE_FREEZE.json").read_bytes()
    frozen = json.loads(freeze_bytes)
    for name, digest in frozen["source_sha256"].items():
        require(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, "pre-batch frozen source mismatch")
    file_json("BATCH_STARTED.json", {"source_freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "binding": binding_record(), "deadline_seconds": 60, "model_calls": 0, "tokenizer_calls": 0})
    status, error = "PASS", None
    try:
        for name, fn in (("full_vector_roundtrip_and_actual_files", successful_roundtrip),
                         ("independent_corrupt_incomplete_reader", corrupt_and_incomplete_reader),
                         ("partial_write_sticky_reconciliation_reserve", partial_write_and_closeout),
                         ("partial_append_and_append_cap", partial_append_and_cap),
                         ("external_file_mismatch_is_visible", external_mismatch),
                         ("single_owner_paths_case_and_settings", ownership_paths_and_settings)):
            require(time.monotonic() - STARTED <= 60, "60-second batch ceiling")
            evidence = fn()
            RESULTS.append({"name": name, "status": "PASS", "evidence": evidence})
            require(time.monotonic() - STARTED <= 60, "60-second batch ceiling")
    except BaseException as caught:
        status, error = "FAIL", type(caught).__name__ + ": " + str(caught)
    for name, digest in frozen["source_sha256"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            status, error = "FAIL", "source changed during batch: " + name
    elapsed = time.monotonic() - STARTED
    size = sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
    if elapsed > 60 or size > 32 * MIB:
        status, error = "FAIL", "batch time/preparation byte ceiling exceeded"
    report = {"status": status, "error": error, "cases": RESULTS, "elapsed_seconds": elapsed,
        "bytes_before_report": size, "source_freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "binding": binding_record(), "model_calls": 0, "tokenizer_calls": 0, "dataset_reads": 0,
        "whole_cohort_simulated": False, "supervisors_executed": False, "real_hook_capacity_verified": False,
        "real_run_authorized": False, "milestone_credit": False,
        "fixture_faults_retained": True, "scope": "PURE_FILESYSTEM_COMPONENT_ONLY"}
    file_json("TEST_REPORT.json", report)
    print(json.dumps({"status": status, "cases_passed": len(RESULTS), "elapsed_seconds": elapsed,
                      "bytes_before_report": size, "error": error}))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
