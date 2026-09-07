"""ONE <=60s pure batch. Does not load/tokenize models or import project runners.

Large storage boundaries use integer reservations, not large filesystem writes.
Only this namespace receives the start receipt and final JSON report.
"""
import copy
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

from contract import (HERE, MIB, AttemptCounter, ResourceLedger, ResourceLimitError,
                      component_settings, decode_logits, encode_logits, load_contract,
                      source_record_bounds, validate_component_settings,
                      validate_contract, validate_prompt_lengths)

STARTED = time.monotonic()
DEADLINE = STARTED + 60
RESULTS = []


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def rejects(fn):
    try:
        fn()
    except (ResourceLimitError, OverflowError):
        return
    raise AssertionError("invalid boundary accepted")


def bounded_json(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    check(len(raw) <= 5 * MIB, "preparation per-file limit")
    current = sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
    check(current + len(raw) <= 32 * MIB, "preparation namespace limit")
    with (HERE / name).open("xb") as stream:
        stream.write(raw)


def stage(name, fn):
    check(time.monotonic() < DEADLINE, "60-second batch deadline")
    result = fn()
    check(time.monotonic() < DEADLINE, "60-second batch deadline")
    RESULTS.append({"name": name, "status": "PASS", "evidence": result})


def math_and_settings(c):
    bounds = source_record_bounds(c)
    check(bounds["raw_logit_vector_bytes"] == 993280, "complete vocab bytes")
    check(bounds["raw_logits_180_bytes"] == 178790400, "180 vector bytes")
    check(bounds["rows_180_bytes"] == 47185920, "45MiB rows")
    check(bounds["all_ledger_maxima_bytes"] == 4063232, "ledger profile arithmetic")
    check(bounds["all_hook_check_records_bytes"] == 446464, "109 complete records")
    check(bounds["reserved_total_bytes"] == 288 * MIB and bounds["non_closeout_bytes"] == 280 * MIB, "reserved failure space")
    settings = {name: component_settings(c) for name in ("recorder", "worker_supervisor", "audit_supervisor", "saved_judge")}
    validate_component_settings(c, settings)
    for component, key, old in (("worker_supervisor", "worker", 300), ("audit_supervisor", "audit", 90)):
        bad = copy.deepcopy(settings)
        bad[component]["seconds"][key] = old
        rejects(lambda bad=bad: validate_component_settings(c, bad))
    bad = copy.deepcopy(settings)
    bad["recorder"]["storage"]["total_bytes"] = 96 * MIB
    rejects(lambda: validate_component_settings(c, bad))
    bad["recorder"]["storage"]["total_bytes"] = 92 * MIB
    rejects(lambda: validate_component_settings(c, bad))
    bad_contract = copy.deepcopy(c)
    bad_contract["storage"]["categories_bytes"]["failure_closeout"] -= 1
    rejects(lambda: validate_contract(bad_contract))
    validate_prompt_lengths(c, [160] * 24)
    for lengths in ([160] * 23, [160] * 23 + [161], [160] * 23 + [0], [160] * 23 + [True]):
        rejects(lambda lengths=lengths: validate_prompt_lengths(c, lengths))
    return bounds


def raw_codec(c):
    # Exactly representable edge values include negative zero and subnormals.
    pattern = (0.0, -0.0, 1.0, -1.0, 1 / 3, 2 ** -149,
               3.4028234663852886e38, -3.4028234663852886e38)
    values = pattern * (c["study"]["vocabulary"] // len(pattern))
    raw = encode_logits(c, values)
    check(raw == struct.pack("<8f", *pattern) * (len(values) // 8), "known little-endian byte pattern")
    check(encode_logits(c, decode_logits(c, raw)) == raw, "byte-identical round trip")
    for altered in (raw[:-1], raw + b"\0", struct.pack("<f", float("nan")) + raw[4:]):
        rejects(lambda altered=altered: decode_logits(c, altered))
    rejects(lambda: encode_logits(c, values[:-1]))
    rejects(lambda: encode_logits(c, (float("inf"),) + values[1:]))
    return {"raw_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "negative_zero_bytes": raw[4:8].hex(), "round_trip": "byte-identical",
            "raw_evidence_files_written": 0, "historical_zlib_files_modified": 0}


def attempt_and_row_counts(c):
    counter = AttemptCounter(c)
    for kind, maximum in counter.limits.items():
        for _ in range(maximum):
            counter.reserve_attempt(kind)
        before = dict(counter.attempts)
        rejects(lambda kind=kind: counter.reserve_attempt(kind))
        check(counter.attempts == before, "ceiling rejection atomic")
    failing = AttemptCounter(c)
    failing.reserve_attempt("forward")
    failing.mark_failed()
    rejects(lambda: failing.reserve_attempt("forward"))
    ledger = ResourceLedger(c)
    for i in range(180):
        ledger.add_logit(f"logits/{i:03d}.f32", 993280)
        ledger.add_row(f"rows/{i:03d}.json", 262144)
    before = ledger.report()
    rejects(lambda: ledger.add_logit("logits/180.f32", 993280))
    rejects(lambda: ledger.add_row("rows/180.json", 262144))
    check(ledger.report() == before, "count rejection atomic")
    small = ResourceLedger(c)
    rejects(lambda: small.add_logit("logits/001.f32", 993281))
    rejects(lambda: small.add_logit("logits/001.zlib", 993280))
    rejects(lambda: small.add_row("rows/001.json", 262145))
    return {"attempt_limits": counter.attempts, "max_source_record_usage": ledger.report(),
            "failed_attempts_consume_allowance": True}


def ledgers_and_logs(c):
    ledger = ResourceLedger(c)
    for name, profile in c["ledger_streams"].items():
        rejects(lambda name=name, profile=profile: ledger.add_event(name, profile["maximum_record_bytes"] + 1))
        for _ in range(profile["maximum_records"]):
            ledger.add_event(name, profile["maximum_record_bytes"])
        rejects(lambda name=name: ledger.add_event(name, 1))
    check(ledger.used["ledgers"] == 4063232, "all streams jointly fit")
    for name, limit in c["log_streams"].items():
        ledger.add_log(name, limit // 2)
        ledger.add_log(name, limit - limit // 2)
        before = ledger.report()
        rejects(lambda name=name: ledger.add_log(name, 1))
        check(ledger.report() == before, "append overflow atomic, never truncates")
    rejects(lambda: ledger.add_event("unbounded_extra_events.jsonl", 1))
    rejects(lambda: ledger.add_log("undeclared.log", 1))
    return ledger.report()


def hook_boundaries(c):
    ledger = ResourceLedger(c)
    for _ in range(109):
        ledger.add_hook_check(4096)
    rejects(lambda: ledger.add_hook_check(1))
    clean = ResourceLedger(c)
    rejects(lambda: clean.add_hook_check(4097))
    clean.add_hook_snapshot("setup_before", [MIB, 3], [MIB + 1024, 32], 4096)
    before = clean.report()
    invalid = [([MIB + 1], [1], 1), ([MIB] * 65, [1] * 65, 1),
               ([1, 1], [1, 1], 1), ([MIB], [5 * MIB + 1], 1),
               ([MIB], [], 1), ([1], [1], 65537)]
    for raw, encoded, manifest in invalid:
        rejects(lambda raw=raw, encoded=encoded, manifest=manifest:
                clean.add_hook_snapshot("invalid", raw, encoded, manifest))
        check(clean.report() == before, "snapshot reservation is complete and atomic")
    # Capacity failure rejects the complete artifact; it does not trim chunks.
    rejects(lambda: clean.add_hook_snapshot("too_large", [MIB] * 4, [5 * MIB] * 4, 100))
    check(clean.report() == before, "oversized compressed snapshot not partially recorded")
    reserve = ResourceLedger(c)
    normal_cap = 16 * MIB - 65536
    # Pure size boundary: no compression-size claim and no giant file writes.
    sizes = [MIB] * 15 + [MIB - 65536 - 100]
    reserve.add_hook_snapshot("capacity", [MIB] * len(sizes), sizes, 100)
    check(reserve.used["hooks"] == normal_cap, "normal hook allocation exact")
    rejects(lambda: reserve.add_hook_json("overflow.json", 1))
    reserve.add_hook_json("fault.json", 65536, fault=True)
    check(reserve.used["hooks"] == 16 * MIB, "hook fault reserve remains usable")
    return {"clean_checks_maximum_bytes": ledger.used["hooks"],
            "hook_total_cap_bytes": reserve.used["hooks"], "snapshot_checks": len(invalid) + 1,
            "metadata_capacity_verified": False, "synthetic_sizes_are_not_measured_model_metadata": True}


def allocations_and_closeout(c):
    ledger = ResourceLedger(c)
    # Exercise the category allocator at its full mathematical boundary. Typed
    # evidence methods above separately enforce tighter counts/record lengths.
    for category, limit in c["storage"]["categories_bytes"].items():
        if category == "failure_closeout":
            continue
        left = limit - (65536 if category == "hooks" else 0)
        i = 0
        while left:
            amount = min(5 * MIB, left)
            ledger._reserve(category, f"allocation/{category}_{i}", amount)
            i += 1
            left -= amount
        if category == "hooks":
            ledger._reserve(category, "allocation/hook_fault", 65536, hook_fault=True)
    check(sum(ledger.used.values()) == 280 * MIB, "all noncloseout allocations full")
    check(ledger.report()["remaining_closeout_bytes"] == 8 * MIB, "failure reserve cannot be borrowed")
    rejects(lambda: ledger.add_bundle_file("sources_inputs", "sources/extra", 1))
    ledger.add_bundle_file("failure_closeout", "failure/one.json", 5 * MIB)
    ledger.add_bundle_file("failure_closeout", "failure/two.json", 3 * MIB)
    check(sum(ledger.used.values()) == 288 * MIB, "exact final ceiling")
    rejects(lambda: ledger.add_bundle_file("failure_closeout", "failure/overflow.json", 1))
    small = ResourceLedger(c)
    rejects(lambda: small.add_bundle_file("sources_inputs", "big", 5 * MIB + 1))
    small.add_bundle_file("sources_inputs", "source.json", 5 * MIB)
    rejects(lambda: small.add_bundle_file("sources_inputs", "SOURCE.JSON", 1))
    rejects(lambda: small.add_bundle_file("audit_inventory", "source.json", 1, append=True))
    for path in ("../escape", "/absolute", "C:/outside", "a/../b", "name.", "name "):
        rejects(lambda path=path: small.add_bundle_file("sources_inputs", path, 1))
    return {"fully_reserved_bytes": sum(ledger.used.values()), "per_file_bytes": 5 * MIB,
            "closeout_reserve_bytes": 8 * MIB, "large_boundary_files_written": 0,
            "overflow_behavior": "reject complete write; no borrowing, truncation or overwrite"}


def main():
    check(not (HERE / "BATCH_STARTED.json").exists(), "one batch only; no rerun")
    c = load_contract()
    bound_sources = ("contract.json", "contract.py", "verify_resource_contract.py", "README.md")
    bindings = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in bound_sources}
    bounded_json("BATCH_STARTED.json", {"deadline_seconds": 60, "source_sha256": bindings,
        "model_calls": 0, "tokenizer_calls": 0, "large_boundary_files_written": 0})
    status, error = "PASS", None
    try:
        for name, fn in (("contract_math_and_component_coherence", math_and_settings),
                         ("fixed_raw_codec_complete_round_trip", raw_codec),
                         ("attempts_and_source_record_counts", attempt_and_row_counts),
                         ("all_ledger_profiles_and_append_logs", ledgers_and_logs),
                         ("complete_hook_metadata_chunk_boundaries", hook_boundaries),
                         ("allocations_per_file_and_failure_reserve", allocations_and_closeout)):
            stage(name, lambda fn=fn: fn(c))
    except BaseException as caught:
        status, error = "FAIL", type(caught).__name__ + ": " + str(caught)
    elapsed = time.monotonic() - STARTED
    if elapsed > 60:
        status, error = "FAIL", "batch exceeded 60-second ceiling"
    for name, digest in bindings.items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            status, error = "FAIL", "source changed during batch: " + name
    report = {"status": status, "scope": "MODEL_FREE_RESOURCE_CONTRACT_ONLY", "elapsed_seconds": elapsed,
        "error": error, "cases": RESULTS, "source_sha256": bindings,
        "model_calls": 0, "tokenizer_calls": 0, "dataset_reads": 0,
        "runner_integration": False, "real_run_authorized": False, "milestone_credit": False,
        "pending_verification": c["pending_verification"]}
    bounded_json("TEST_REPORT.json", report)
    print(json.dumps({"status": status, "cases_passed": len(RESULTS), "elapsed_seconds": elapsed,
                      "error": error, "real_run_authorized": False}))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
