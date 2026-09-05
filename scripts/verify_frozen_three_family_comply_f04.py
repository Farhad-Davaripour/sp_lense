"""Independent fixed-C f04 audit and transfer-only bounded finalization; no ML imports."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import frozen_three_family_comply_f04_plan as protocol
from scripts import three_family_recording_bindings as bindings
from scripts.frozen_three_family_comply_f04_recording import TransferBudget
from scripts.three_family_bounded_capture import exception_record

parent = protocol.isolate(
    "scripts._three_family_C_f04_audit", "scripts/verify_frozen_crossed_comply_f04.py"
)
engine = parent.engine
OUTPUT, require, read, sha = ROOT / protocol.OUTPUT, protocol.require, protocol.read, protocol.sha
for module in (parent, parent.parent, engine):
    module.protocol, module.OUTPUT = protocol, OUTPUT
f32, outcome, compare_summary = parent.f32, parent.outcome, parent.compare_summary
verify_renderings, verify_data = parent.verify_renderings, parent.verify_data
comparability_data = parent.comparability_data
_summary = parent.summary
AUDIT_MATCH = "INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH"


def summary(rows):
    result = _summary(rows)
    result["status"] = result["status"].replace(
        "FROZEN_COMPLY_F04_", "FROZEN_THREE_FAMILY_COMPLY_F04_"
    )
    return result


engine.summary = summary
# Start from the original definition so only the exact count/time constants change.
raw_audit = protocol.isolate(
    "scripts._three_family_C_f04_raw_audit", "scripts/verify_crossed_pair_probe.py"
)
raw_audit.protocol, raw_audit.OUTPUT = protocol, OUTPUT
raw_audit.verify_data = verify_data
_verify_numeric = protocol.adapt(raw_audit, "verify", {20: 12, 600: 300}, {20: 2, 600: 2})


def historical_comparison(config_key):
    """Use saved outcomes only after exact prompt/runtime/h0/raw-baseline equivalence."""
    try:
        spec = protocol.config_at()[config_key]
        for item in spec["files"]:
            require(
                sha((ROOT / item["path"]).read_bytes()) == item["sha256"],
                "historical comparison hash",
            )
        historical = ROOT / spec["namespace"]
        result = comparability_data(
            read(OUTPUT / "preregistration.json")["plan"],
            engine.rows_at(OUTPUT / "rows.jsonl"),
            OUTPUT,
            read(historical / "preregistration.json"),
            engine.rows_at(historical / "rows.jsonl"),
            historical,
            read(historical / "runtime.json"),
            read(OUTPUT / "runtime.json"),
        )
        result["historical_result_usable"] = result["comparable"]
        result["historical_strict_accepted"] = None
        result["historical_total"] = None
        result["rerun"] = False
        if result["comparable"]:
            audit = read(historical / "verification.json")
            require(
                audit["status"] == AUDIT_MATCH, "authenticated saved independent historical audit"
            )
            result["historical_strict_accepted"] = audit["summary"]["matrix"]["strict_accepted"]
            result["historical_total"] = 4
        # The inherited helper's P-specific alias is not used as C evidence.
        result.pop("historical_preserve_result_usable", None)
        return result
    except Exception as error:  # noqa: BLE001 - comparison never authorizes extra observations.
        return {
            "status": "UNVERIFIED_COMPARABILITY",
            "comparable": False,
            "historical_result_usable": False,
            "historical_strict_accepted": None,
            "historical_total": None,
            "rerun": False,
            "exception": exception_record(error),
        }


def verify():
    for path in OUTPUT.iterdir():
        if path.is_file() and path.suffix == ".json":
            parent.parent.finite_tree(read(path))
    result = _verify_numeric()
    if result["status"] != "INCONCLUSIVE":
        result["historical_comparisons"] = {
            "old_C": historical_comparison("historical_old_comply_comparison"),
            "P": historical_comparison("historical_comparison"),
        }
    return result


def transfer_passed(result):
    value = result.get("summary", {})
    return (
        result.get("status") == AUDIT_MATCH
        and value.get("matrix", {}).get("matrix_pass") is True
        and value["matrix"].get("strict_accepted") == 4
        and value.get("replay_matches") == 4
    )


def report(result):
    text = parent.parent.report(result).replace("f03", "f04")
    text = text.replace("Frozen crossed-trained COMPLY", "Frozen three-family-trained COMPLY")
    text = text.replace(
        "Prior construction8/8 had all-B baselines, four B→A flips and four weakened B retentions; an A-favoring alternative remains important.",
        "Source construction12/12 had all-B baselines, six B→A flips and six weakened B retentions; an A-favoring alternative remains important.",
    )
    text += "\n## Saved historical comparisons: no rerun\n\n"
    text += json.dumps(result.get("historical_comparisons", {}), indent=2) + "\n"
    text += (
        "\nOnly exact prompt, model, boundary, h0 and raw-baseline equivalence permits historical comparison. "
        "Old C and P remain separate frozen results; neither changes this test's thresholds. "
        "This is previously exposed DEVELOPMENT outside fitting, not pristine heldout confirmation. "
        "No new candidate, fitting, P rerun, task preservation, gate/controller or general-reliability claim. STOP.\n"
    )
    return text


def finalize_recording(budget, capture_receipt, runtime_status, *, audit=None):
    """Same reserved, quiescence-first recording discipline; never create a candidate."""
    output, audit = budget.root, verify if audit is None else audit
    quiescent = capture_receipt.get("quiescent") is True
    if not quiescent:
        failure = {
            "status": "INCONCLUSIVE",
            "reason": "worker/writer quiescence unconfirmed",
            "recording_sealed": False,
            "transfer_passed": False,
            "capture": capture_receipt,
            "retries_allowed": False,
        }
        persisted = threading.Event()

        def save_failure():
            try:
                budget.begin_finalization(quiescent=False)
                for name, value in (
                    ("capture_receipt.json", capture_receipt),
                    ("RUN_STATUS.json", runtime_status),
                    ("RECORDING_FAILURE.json", failure),
                ):
                    budget.write_bytes(output / name, bindings.encoded_json(value), final=True)
                persisted.set()
            except BaseException:  # noqa: BLE001 - bounded best effort, never recursive logging.
                return

        helper = threading.Thread(target=save_failure, daemon=True)
        helper.start()
        helper.join(timeout=1.0)
        return {
            **failure,
            "failure_receipts_persisted": persisted.is_set(),
            "failure_receipt_writer_joined": not helper.is_alive(),
            "final_inventory_withheld": True,
        }
    try:
        budget.begin_finalization(quiescent=True)
        for name, value in (
            ("capture_receipt.json", capture_receipt),
            ("RUN_STATUS.json", runtime_status),
        ):
            budget.write_bytes(output / name, bindings.encoded_json(value), final=True)
        complete = capture_receipt.get("status") == runtime_status.get("status") == "complete_valid"
        verified = (
            audit()
            if complete
            else {
                "status": "INCONCLUSIVE",
                "runtime": runtime_status,
                "capture": capture_receipt,
                "retries_allowed": False,
            }
        )
        require(
            verified.get("status") in {AUDIT_MATCH, "INCONCLUSIVE"},
            "exact independent audit disposition",
        )
        passed = transfer_passed(verified)
        budget.write_bytes(
            output / "verification.json",
            bindings.encoded_json(verified, sorted_keys=True),
            final=True,
        )
        prefix = "Recording validity requires a complete matching FINAL_INVENTORY.json; faults override provisional reports.\n\n"
        budget.write_bytes(
            output / "PILOT_REPORT.md", (prefix + report(verified)).encode(), final=True
        )
        budget.write_bytes(
            output / "CLOSEOUT.json",
            bindings.encoded_json(
                {
                    "status": "PROVISIONAL_UNTIL_FINAL_INVENTORY",
                    "numeric_audit_status": verified["status"],
                    "transfer_provisionally_passed": passed,
                    "new_candidate_created": False,
                    "runtime_status": runtime_status["status"],
                    "capture_complete": complete,
                    "retries_allowed": False,
                }
            ),
            final=True,
        )
        inventory = budget.finalize_inventory(valid_candidate=False, quiescent=True)
        require(not inventory["valid_candidate"], "transfer cannot create a candidate")
        require(
            not inventory.get("fault_code") or not complete, "recording fault overrides transfer"
        )
        valid = complete and not inventory.get("fault_code") and verified["status"] == AUDIT_MATCH
        return {
            "status": "complete_valid" if valid else "INCONCLUSIVE",
            "recording_sealed": True,
            "numeric_audit_status": verified["status"],
            "transfer_passed": bool(valid and passed),
            "final_inventory": inventory,
            "retries_allowed": False,
        }
    except BaseException as error:  # noqa: BLE001 - immutable partial failure, one receipt only.
        failure = {
            "status": "INCONCLUSIVE",
            "failure_category": "technical_recording_or_audit",
            "exception": exception_record(error),
            "transfer_passed": False,
            "recording_sealed": False,
            "retries_allowed": False,
        }
        try:
            budget.fault("FINALIZATION_FAILURE")
            budget.write_bytes(
                output / "RECORDING_FAILURE.json", bindings.encoded_json(failure), final=True
            )
            failure["failure_receipt_persisted"] = True
        except BaseException:  # noqa: BLE001 - no retry on failed receipt.
            failure["failure_receipt_persisted"] = False
        try:
            failure["final_inventory"] = budget.finalize_inventory(
                valid_candidate=False, quiescent=True
            )
            failure["recording_sealed"] = True
        except BaseException:  # noqa: BLE001 - explicitly unsealed, no false completeness.
            failure["final_inventory_complete"] = False
        return failure


def read_sealed_recording(output=OUTPUT):
    seal = TransferBudget(output, initialize=False).verify_inventory()
    require(
        not seal["inventory"]["valid_candidate"],
        "no newly constructed candidate in transfer evidence",
    )
    if seal["inventory"].get("fault_code"):
        return {"status": "INCONCLUSIVE", "transfer_passed": False, "recording_inventory": seal}
    result = read(Path(output) / "verification.json")
    require(result.get("status") in {AUDIT_MATCH, "INCONCLUSIVE"}, "hashed independent disposition")
    if result["status"] == AUDIT_MATCH:
        capture = read(Path(output) / "capture_receipt.json")
        runtime = read(Path(output) / "RUN_STATUS.json")
        require(
            capture.get("status") == runtime.get("status") == "complete_valid"
            and capture.get("quiescent") is True,
            "hashed complete capture/runtime required for transfer disposition",
        )
    return {**result, "transfer_passed": transfer_passed(result), "recording_inventory": seal}


if __name__ == "__main__":
    require(sys.argv[1:] == [], "read-only sealed verifier; finalization belongs to supervisor")
    try:
        result = read_sealed_recording()
    except Exception as error:  # noqa: BLE001 - read-only no repair/retry.
        result = {
            "status": "INCONCLUSIVE",
            "transfer_passed": False,
            "exception": exception_record(error),
        }
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("summary", "baselines", "recording_inventory")
            },
            indent=2,
        )
    )
    if result["status"] == "INCONCLUSIVE":
        raise SystemExit(1)
