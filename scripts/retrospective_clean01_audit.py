"""One retrospective external reconstruction; never repair the original recording."""

from __future__ import annotations

import hashlib
import importlib.abc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ORIGINAL = ROOT / "evidence/paired_common_drift_comply_three_family_1800_clean01_v1_qwen35_08b"
OUTPUT = ROOT / "evidence/paired_common_drift_clean01_retrospective_audit01"
SOURCES = (
    "scripts/retrospective_clean01_audit.py",
    "scripts/retrospective_clean01_inputs.py",
    "docs/RETROSPECTIVE_CLEAN01_AUDIT01.md",
)
MIB = 1048576
LIMITS = {
    "envelope": 262144,
    "row": 2048,
    "update": 4096,
    "trajectory": 4096,
    "structure": 4096,
    "markdown": 16384,
    "sources": 65536,
    "controls": 16384,
}
BOUND = (
    LIMITS["envelope"]
    + 216 * LIMITS["row"]
    + 8 * LIMITS["update"]
    + 10 * LIMITS["trajectory"]
    + sum(LIMITS[k] for k in ("structure", "markdown", "sources", "controls"))
)
ROW_KEYS = (
    "cell_id",
    "condition",
    "actual_next_token_label",
    "baseline_label",
    "comply_label",
    "signed_margin",
    "signed_delta_log_odds",
    "answer_pair_mass",
    "kl_from_baseline",
    "net_relative_norm",
    "path_relative_norm",
    "maximum_delta_error",
    "maximum_offset_error",
    "maximum_step_error",
    "maximum_current_logit_difference",
    "maximum_current_h_difference",
    "requested_accepted",
    "quality_valid",
    "shared_w_sha256",
    "logits_file",
)
UPDATE_KEYS = (
    "stage",
    "status",
    "loss_before",
    "B",
    "B0",
    "denominator",
    "d_norm",
    "clip_factor",
    "projection_factor",
    "projection_distance",
    "step_norm",
    "net_norm",
    "path_before",
    "path_after",
    "w_before_sha256",
    "w_after_sha256",
)
PAIR_KEYS = ("pair_index", "a", "c", "m_A", "m_B", "margin_loss", "drift_loss", "loss")
SUMMARY_KEYS = (
    "status",
    "stop_reason",
    "updates",
    "attempted_updates",
    "shared_path",
    "shared_net",
    "forward_count",
    "derivative_count",
    "final_accepted",
    "accepted_flips",
    "accepted_retentions",
    "actual_A_to_B",
    "actual_B_to_A",
    "final_other_token_outcomes",
    "retention_weakening",
    "independent_semantic_situations",
    "rendering_rows",
    "transfer_ran",
)
QUALIFICATION = (
    "New retrospective external audit execution of saved evidence, not recovery of the unsaved "
    "original audit. Original INCONCLUSIVE ARTIFACT_BYTE_CAP and empty verification remain unchanged. "
    "No prospective pass, valid candidate, vector export or selection. Recorded-gradient arithmetic "
    "is checked; real derivatives are not independently rerun. No model/tokenizer loads or F/D. "
    "These are three development/training situations, not held-out generalization or a gate test."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)
        + "\n"
    ).encode()


def scalar(value):
    require(type(value) in (str, int, float, bool) or value is None, "scalar whitelist only")
    if isinstance(value, str):
        require(len(value) <= 256, "bounded summary string")
    if type(value) is float:
        require(math.isfinite(value), "finite report scalar")
    return value


def select(value, keys):
    return {key: scalar(value[key]) for key in keys}


def compact_row(row, line, hashes):
    result = {
        **select({"maximum_step_error": None, **row}, ROW_KEYS),
        "rows_jsonl_line": line,
        "logits_sha256": hashes[row["logits_file"]],
        "recorded_gradient_present": row["gradient"] is not None,
        "reconstruction": "MATCH",
    }
    require(len(encoded(result)) <= LIMITS["row"], "compact row byte gap")
    return result


def numeric_error(actual, expected):
    """Report maxima after the independent checker has already enforced identities/tolerances."""
    if isinstance(expected, dict):
        values = [numeric_error(actual[key], value) for key, value in expected.items()]
    elif isinstance(expected, (list, tuple)):
        values = [numeric_error(a, b) for a, b in zip(actual, expected, strict=True)]
    elif type(expected) is float:
        error = abs(actual - expected)
        return error, error / max(1.0, abs(expected))
    else:
        return 0.0, 0.0
    return tuple(max((item[i] for item in values), default=0.0) for i in (0, 1))


def compact_update(update, expected, line):
    absolute, scaled = numeric_error(update, expected)
    result = {
        **select(expected, UPDATE_KEYS),
        "updates_jsonl_line": line,
        "hashes_above_are_reconstructed": True,
        "recorded_w_before_sha256": scalar(update["w_before_sha256"]),
        "recorded_w_after_sha256": scalar(update["w_after_sha256"]),
        "recorded_gradient_arithmetic": "MATCH",
        "real_derivatives_revalidated": False,
        "maximum_absolute_component_error": absolute,
        "maximum_scaled_component_error": scaled,
        "pairs": [
            {**select(pair, PAIR_KEYS), "curvature": scalar(pair["curvature"])}
            for pair in expected["pairs"]
        ],
    }
    require(len(result["pairs"]) == 6, "six compact pairs")
    require(len(encoded(result)) <= LIMITS["update"], "compact update byte gap")
    return result


def compact_trajectory(group):
    result = {
        "condition": scalar(group["condition"]),
        "loss": scalar(group["loss"]),
        "pairs": [select(pair, PAIR_KEYS) for pair in group["pairs"]],
    }
    require(len(result["pairs"]) == 6, "six trajectory pairs")
    require(len(encoded(result)) <= LIMITS["trajectory"], "compact trajectory byte gap")
    return result


def pack(envelope, rows, updates, trajectory):
    require(len(encoded(envelope)) <= LIMITS["envelope"], "envelope byte gap")
    require(len(rows) == 216 and len(updates) == 8 and len(trajectory) == 10, "fixed report counts")
    for values, key in ((rows, "row"), (updates, "update"), (trajectory, "trajectory")):
        require(all(len(encoded(value)) <= LIMITS[key] for value in values), "component byte gap")
    raw = encoded({**envelope, "rows": rows, "updates": updates, "trajectory": trajectory})
    maximum = (
        LIMITS["envelope"]
        + 216 * LIMITS["row"]
        + 8 * LIMITS["update"]
        + 10 * LIMITS["trajectory"]
        + LIMITS["structure"]
    )
    require(len(raw) <= maximum and BOUND <= MIB, "aggregate report byte gap")
    return raw


def size_test():
    """Fake 1024D native arrays are present at input, absent from fixed-whitelist output."""
    started = time.monotonic()
    large = [sys.float_info.max] * 1024
    row = dict.fromkeys(ROW_KEYS, sys.float_info.max)
    for key in (
        "cell_id",
        "condition",
        "actual_next_token_label",
        "baseline_label",
        "comply_label",
        "shared_w_sha256",
        "logits_file",
    ):
        row[key] = "x" * 80
    row.update(gradient=large, h0=large, shared_w=large, h=large)
    update = dict.fromkeys(UPDATE_KEYS, sys.float_info.max)
    update.update(stage=8, status="ready", w_before_sha256="f" * 64, w_after_sha256="e" * 64)
    pairs = [
        {
            **dict.fromkeys(PAIR_KEYS, sys.float_info.max),
            "curvature": sys.float_info.max,
            "J_A": large,
            "J_B": large,
            "J_c": large,
        }
        for _ in range(6)
    ]
    update.update(pairs=pairs, w_before=large, w_after=large, g=large, d=large, s=large, r=large)
    rows = [compact_row(row, i + 1, {row["logits_file"]: "a" * 64}) for i in range(216)]
    nonstep = {key: value for key, value in row.items() if key != "maximum_step_error"}
    require(
        compact_row(nonstep, 1, {row["logits_file"]: "a" * 64})["maximum_step_error"] is None,
        "nonstep cast diagnostic is explicitly not applicable",
    )
    updates = [compact_update(update, update, i + 1) for i in range(8)]
    trajectory = [
        compact_trajectory({"condition": "final", "loss": sys.float_info.max, "pairs": pairs})
        for _ in range(10)
    ]
    envelope = {
        "qualification": QUALIFICATION,
        "fake_source_hashes": {str(i) + "x" * 240: "f" * 64 for i in range(146 + 237)},
    }
    raw = pack(envelope, rows, updates, trajectory)
    require(
        b'"J_A"' not in raw and b'"w_after"' not in raw and b'"gradient"' not in raw,
        "native arrays must not leak",
    )
    rejected = False
    try:
        pack({"oversize": "x" * LIMITS["envelope"]}, rows, updates, trajectory)
    except ValueError:
        rejected = True
    require(rejected, "oversized envelope fails closed")
    source_bytes = sum((ROOT / path).stat().st_size for path in SOURCES)
    require(source_bytes <= LIMITS["sources"], "source allowance")
    return {
        "status": "FAKE_NATIVE_ARRAY_SIZE_TEST_PASSED",
        "fake_serialized_bytes": len(raw),
        "formal_all_new_artifacts_bound_bytes": BOUND,
        "maximum_bytes": MIB,
        "component_limits": LIMITS,
        "source_bytes": source_bytes,
        "elapsed_seconds": time.monotonic() - started,
        "real_evidence_numerics_run": False,
        "native_vectors_serialized": False,
    }


def write_new(name, raw, limit):
    require(len(raw) <= limit, "artifact byte gap")
    if name not in {"audit.json", "REPORT.md"}:
        control_bytes = sum(
            p.stat().st_size
            for p in OUTPUT.iterdir()
            if p.is_file() and p.name not in {"audit.json", "REPORT.md"}
        )
        require(control_bytes + len(raw) <= LIMITS["controls"], "aggregate control byte gap")
    with (OUTPUT / name).open("xb") as stream:
        stream.write(raw)


def guard():
    """Fail closed on ML imports and any write-open to the original evidence tree."""

    class NoML(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"torch", "transformers", "transformer_lens", "tokenizers"}:
                raise RuntimeError("MODEL_IMPORT_FORBIDDEN")

    sys.meta_path.insert(0, NoML())
    old = ORIGINAL.resolve()

    def audit_event(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int)
                and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            )
            if writing and path.is_relative_to(old):
                raise RuntimeError("ORIGINAL_EVIDENCE_WRITE_FORBIDDEN")
        if event in {"os.remove", "os.rename", "os.rmdir", "os.mkdir"}:
            for arg in args[:2]:
                if isinstance(arg, (str, bytes, os.PathLike)) and Path(
                    os.fsdecode(arg)
                ).resolve().is_relative_to(old):
                    raise RuntimeError("ORIGINAL_EVIDENCE_MUTATION_FORBIDDEN")

    sys.addaudithook(audit_event)


def audit_once():
    started = time.monotonic()
    guard()
    write_new("AUDIT_ENTERED.json", encoded({"new_execution": True, "pid": os.getpid()}), 1024)
    from scripts import retrospective_clean01_inputs as inputs
    from scripts import verify_paired_common_drift_comply_1800_clean01 as checker

    identity = inputs.authenticate(ROOT, ORIGINAL)
    plan = identity.pop("plan")
    checker.verify_plan(plan)
    rows = checker.engine.rows_at(ORIGINAL / "rows.jsonl")
    runtime = inputs.check_runtime(ROOT, ORIGINAL, plan, rows, checker.engine.rows_at)
    reconstructed = checker.verify_data(plan, rows, ORIGINAL)
    checker.compare_summary(checker.read(ORIGINAL / "analysis.json"), reconstructed["summary"])
    require(
        reconstructed["status"] == checker.AUDIT_MATCH, "external reconstruction match required"
    )
    require(
        reconstructed["summary"]["forward_count"] == 216
        and reconstructed["summary"]["derivative_count"] == 96,
        "reconstructed counts agree with saved runtime",
    )
    updates = checker.engine.rows_at(ORIGINAL / "updates.jsonl")
    compact_updates = [
        compact_update(a, b, i + 1)
        for i, (a, b) in enumerate(zip(updates, reconstructed["optimizer_checks"], strict=True))
    ]
    after = inputs.authenticate(ROOT, ORIGINAL)
    after.pop("plan")
    require(after == identity, "old evidence/source hashes unchanged after audit")
    require(not {"torch", "transformers", "transformer_lens", "tokenizers"} & set(sys.modules),
            "no ML imports")
    summary = reconstructed["summary"]
    envelope = {
        "schema": "sp_lense.retrospective_external_audit.v1",
        "status": "RETROSPECTIVE_EXTERNAL_RECONSTRUCTION_MATCH",
        "qualification": QUALIFICATION,
        "original_status": "INCONCLUSIVE",
        "original_fault": "ARTIFACT_BYTE_CAP",
        "original_validity_repaired": False,
        "valid_candidate": False,
        "new_execution": True,
        "model_loads": 0,
        "tokenizer_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
        "independent_real_derivative_validation": False,
        "provenance": identity,
        "recorded_runtime_checks": runtime,
        "audit_sources_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES
        },
        "summary": select(summary, SUMMARY_KEYS),
        "directional_coverage": summary["directional_coverage"],
        "maximum_absolute_score_errors": reconstructed["maximum_absolute_arithmetic_errors"],
        "maximum_current_logit_difference": reconstructed["maximum_current_logit_difference"],
        "maximum_cast_component_error": reconstructed["maximum_cast_component_error"],
        "maximum_nonfinal_difference": reconstructed["maximum_nonfinal_difference"],
        "maximum_update_absolute_error": max(
            u["maximum_absolute_component_error"] for u in compact_updates
        ),
        "maximum_update_scaled_error": max(
            u["maximum_scaled_component_error"] for u in compact_updates
        ),
        "numeric_elapsed_seconds": time.monotonic() - started,
        "score_absolute_tolerance": 2e-5,
        "geometry_tolerance": 1e-6,
        "update_scaled_tolerance": 1e-9,
        "source_and_evidence_unchanged": True,
    }
    raw = pack(
        envelope,
        [compact_row(r, i + 1, identity["artifact_sha256"]) for i, r in enumerate(rows)],
        compact_updates,
        [compact_trajectory(g) for g in summary["paired_objective_trajectory"]],
    )
    markdown = (
        "# Retrospective external audit — clean01\n\n" + QUALIFICATION + "\n\n"
        "New reconstruction: **MATCH** for all216 saved rows/raw arrays,96 recorded gradients,"
        "8updates and12 final replays. Full-vocabulary score, geometry, journal, conditional "
        "schedule, endpoint and recorded-gradient update checks passed. No model was executed.\n\n"
        f"Final acceptance: {summary['final_accepted']}/12; accepted flips: {summary['accepted_flips']}; "
        f"retentions: {summary['accepted_retentions']}; retention weakening: {summary['retention_weakening']}. "
        f"Stop: {summary['stop_reason']}; native net {summary['shared_net']:.12g}; "
        f"path {summary['shared_path']:.12g}.\n\n"
        "See audit.json for compact per-row/per-update results, pair trajectory, maxima and "
        "hash-bound raw references. Native vectors are not exported. This retrospective "
        "behavioral/arithmetic support does not supply the missing prospective audit pass. "
        "Original recording status remains **INCONCLUSIVE / ARTIFACT_BYTE_CAP**. STOP.\n"
    ).encode()
    write_new("audit.json", raw, MIB)
    write_new("REPORT.md", markdown, LIMITS["markdown"])
    print(
        json.dumps(
            {
                "status": envelope["status"],
                "audit_bytes": len(raw),
                "numeric_elapsed_seconds": envelope["numeric_elapsed_seconds"],
            }
        ),
        flush=True,
    )


def run():
    started = time.monotonic()
    require(not OUTPUT.exists(), "one separately named external attempt only; no retry")
    OUTPUT.mkdir()
    write_new(
        "ATTEMPT.json",
        encoded(
            {
                "kind": "RETROSPECTIVE_EXTERNAL_AUDIT_NOT_A_NEW_LOCK",
                "qualification": QUALIFICATION,
                "started_unix": time.time(),
                "aggregate_execution_cap_seconds": 120,
                "working_execution_cap_seconds": 105,
                "parent_execution_cap_seconds": 110,
                "preexecution_lint_reservation_seconds": 10,
                "total_new_artifacts_cap_bytes": MIB,
            }
        ),
        4096,
    )
    outcome = {
        "status": "RETROSPECTIVE_AUDIT_INCOMPLETE",
        "original_status": "INCONCLUSIVE",
        "valid_candidate": False,
        "retries_allowed": False,
    }
    try:
        tested = size_test()
        write_new("SIZE_CHECK.json", encoded(tested), 4096)
        remaining = 105 - (time.monotonic() - started)
        require(remaining > 0, "no execution budget after size test")
        child = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "_audit"],
            cwd=ROOT,
            capture_output=True,
            timeout=remaining,
            check=False,
        )
        outcome.update(
            child_exit_code=child.returncode,
            stdout_bytes=len(child.stdout),
            stdout_sha256=hashlib.sha256(child.stdout).hexdigest(),
            stderr_bytes=len(child.stderr),
            stderr_sha256=hashlib.sha256(child.stderr).hexdigest(),
            stdout_prefix=child.stdout[:2048].decode("utf-8", "replace"),
            stderr_prefix=child.stderr[:2048].decode("utf-8", "replace"),
        )
        if child.returncode == 0:
            outcome["status"] = "RETROSPECTIVE_EXTERNAL_RECONSTRUCTION_MATCH"
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - one bounded failure, no retry.
        outcome.update(exception_type=type(error).__name__, exception_prefix=str(error)[:2048])
    outcome.update(
        aggregate_execution_seconds=time.monotonic() - started,
        model_loads=0,
        tokenizer_loads=0,
        real_forwards=0,
        real_derivatives=0,
    )
    outcome["charged_with_preexecution_reservation_seconds"] = (
        outcome["aggregate_execution_seconds"] + 10
    )
    require(outcome["aggregate_execution_seconds"] <= 110, "aggregate execution cap exceeded")
    existing = sorted(p for p in OUTPUT.iterdir() if p.is_file())
    outcome["artifact_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in existing
    }
    raw = encoded(outcome)
    source_bytes = sum((ROOT / p).stat().st_size for p in SOURCES)
    require(
        sum(p.stat().st_size for p in existing) + len(raw) + source_bytes <= MIB,
        "complete new artifact byte gap",
    )
    write_new("CLOSEOUT.json", raw, 8192)
    print(json.dumps(outcome, indent=2))
    return 0 if outcome["status"] == "RETROSPECTIVE_EXTERNAL_RECONSTRUCTION_MATCH" else 1


if __name__ == "__main__":
    require(len(sys.argv) == 2 and sys.argv[1] in {"run", "_audit"}, "run only; no tuning switches")
    if sys.argv[1] == "_audit":
        require((OUTPUT / "SIZE_CHECK.json").is_file(), "size test must precede audit")
        audit_once()
    else:
        raise SystemExit(run())
