"""One prospectively locked frozen-C f04 transfer; unchanged twelve-cell model math."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import frozen_three_family_comply_f04_plan as protocol
from scripts import frozen_three_family_comply_f04_recording as recording
from scripts import three_family_bounded_capture as capture
from scripts import three_family_recording_bindings as bindings

parent = protocol.isolate(
    "scripts._three_family_C_f04_runtime", "scripts/frozen_crossed_comply_f04.py"
)
engine = parent.engine
OUTPUT, require, base = ROOT / protocol.OUTPUT, protocol.require, parent.base
for module in (parent, parent.parent):
    module.protocol, module.OUTPUT = protocol, OUTPUT
engine.protocol = protocol
for key in ("SCRIPT", "VERIFY", "TEST", "DOC", "PLAN", "OUTPUT"):
    setattr(engine, key, getattr(protocol, key))
evaluate, make_delta, validate_vectors = parent.evaluate, parent.make_delta, parent.validate_vectors
DerivativeGuard, EligibilityError = parent.DerivativeGuard, parent.EligibilityError
norm, EPS = protocol.norm, protocol.EPS
TransferBudget = recording.TransferBudget
_summary = parent.summarize
AUTH_KEY = "SP_LENSE_FROZEN_THREE_FAMILY_C_F04_AUTHORIZATION"
AUTH_SCOPE = "one frozen three-family C f04 transfer;12F/0D/300s/64MiB;no retry"


def summarize(rows):
    result = _summary(rows)
    result["status"] = result["status"].replace(
        "FROZEN_COMPLY_F04_", "FROZEN_THREE_FAMILY_COMPLY_F04_"
    )
    return result


def source_identity():
    """Batch Git checks keep the same exact-byte provenance within fresh-usage limits."""
    expected = protocol.build_plan()["input_sha256"]
    paths = list(dict.fromkeys([*expected, *protocol.SOURCE_PATHS]))

    def git_bytes(*args, input=None):
        return subprocess.check_output(["git", *args], cwd=ROOT, input=input)

    tracked = {}
    for entry in git_bytes("ls-files", "--stage", "-z").split(b"\0"):
        if entry:
            metadata, path = entry.split(b"\t", 1)
            _mode, blob, stage = metadata.split()
            require(stage == b"0", "no unmerged tracked source state")
            tracked[path.decode()] = blob.decode()
    dirty = set()
    for options in (("diff", "--name-only", "-z"), ("diff", "--cached", "--name-only", "-z")):
        dirty.update(x.decode() for x in git_bytes(*options).split(b"\0") if x)
    require(
        all(path in tracked and path not in dirty for path in paths),
        "clean tracked source/input set",
    )
    result = {path: protocol.sha((ROOT / path).read_bytes()) for path in paths}
    require(
        all(result[path] == digest for path, digest in expected.items()), "frozen raw input bytes"
    )
    raw_blobs = (
        git_bytes(
            "hash-object", "--no-filters", "--stdin-paths", input=("\n".join(paths) + "\n").encode()
        )
        .decode()
        .splitlines()
    )
    require(
        all(tracked[path] == blob for path, blob in zip(paths, raw_blobs, strict=True)),
        "raw Git source bytes without normalization",
    )
    return result


engine.summarize, engine.source_identity = summarize, source_identity
require_freeze = engine.require_freeze


def freeze():
    recording.require_certificate()
    original = base.write_new
    base.write_new = bindings.write_preregistration
    try:
        return parent.freeze()
    finally:
        base.write_new = original


def require_authorization():
    require(
        json.loads(os.environ.get(AUTH_KEY, "null"))
        == {
            "authorized_by": "supervisor",
            "scope": AUTH_SCOPE,
            "preregistration_sha256": protocol.sha((OUTPUT / "preregistration.json").read_bytes()),
        },
        "exact supervisor authorization bound to frozen transfer lock",
    )


def checked_usage(value):
    require(
        isinstance(value, dict)
        and all(
            type(value.get(key)) in (int, float) and math.isfinite(value[key])
            for key in ("standard_used_percent", "checked_at_unix")
        )
        and 0 <= value["standard_used_percent"] < 90
        and 0 <= time.time() - value["checked_at_unix"] <= 60,
        "finite fresh usage below90 required",
    )
    return value


def preflight(worker_entry=False):
    from scripts.verify_frozen_three_family_comply_f04 import verify_renderings

    recording.require_certificate()
    record = require_freeze()
    path = protocol.OUTPUT + "/preregistration.json"
    require(
        base.git(ROOT, "show", "--pretty=", "--name-only", "HEAD").splitlines() == [path]
        and base.git(ROOT, "rev-parse", "HEAD^") == record["source_commit"]
        and not base.git(ROOT, "status", "--porcelain", "--", path),
        "clean preregistration-only HEAD immediately after source commit",
    )
    names = {"preregistration.json"}
    if worker_entry:
        names |= {"RUN_STARTED.json", "worker.log", "recording_state.json", "recording.lock"}
    require(
        OUTPUT.is_dir() and {p.name for p in OUTPUT.iterdir()} == names,
        "untouched namespace; no prior claim, calls, retries or extras",
    )
    verify_renderings(record["plan"])
    require(
        len(record["plan"]["prompts"]) == 4
        and len(record["plan"]["cells"]) == 12
        and not record["plan"]["derivative_cells"],
        "exact4/12/0 before load",
    )
    return {
        "status": "ZERO_MODEL_PREFLIGHT_PASSED",
        "source_commit": record["source_commit"],
        "source_hash_entries": len(record["source_sha256"]),
        "preregistration_sha256": protocol.sha((OUTPUT / "preregistration.json").read_bytes()),
        "storage": protocol.storage_preflight(ROOT, record["plan"]["config"]),
        "prompts": 4,
        "forwards": 12,
        "derivatives": 0,
        "model_loads": 0,
        "tokenizer_loads": 0,
        "real_forwards": 0,
        "real_derivatives": 0,
    }


def worker():
    require_authorization()
    preflight(worker_entry=True)
    record, started = (
        protocol.read(OUTPUT / "preregistration.json"),
        protocol.read(OUTPUT / "RUN_STARTED.json"),
    )
    usage = checked_usage(started.get("usage_preflight"))
    values = [started.get(key) for key in ("started_monotonic", "deadline_monotonic")]
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in values)
        and started["command"] == [sys.executable, "-u", str(ROOT / protocol.SCRIPT), "_worker"]
        and started["forward_ceiling"] == 12
        and started["derivative_ceiling"] == 0
        and started["timeout_seconds"] == 300
        and started["deadline_monotonic"] == started["started_monotonic"] + 300
        and started["started_monotonic"] <= time.monotonic() < started["deadline_monotonic"],
        "exact12/0/300 finite budget before worker claim",
    )
    budget = TransferBudget(OUTPUT, initialize=False)
    try:
        with bindings.bind_writers(budget):
            base.write_new(OUTPUT / "WORKER_CLAIM.json", {"pid": os.getpid()})
            vectors = {key: value["vector"] for key, value in protocol.candidates().items()}
            validate_vectors(record["plan"], vectors)
            storage = protocol.storage_preflight(ROOT, record["plan"]["config"])
            base.write_new(
                OUTPUT / "storage_preflight.json", {**storage, "monotonic": time.monotonic()}
            )
            checked_usage(usage)
            require(
                time.monotonic() < started["deadline_monotonic"], "deadline before model loading"
            )
            backend, _unused = base.load_backend(record["plan"])
            base.write_new(
                OUTPUT / "runtime.json",
                {
                    **backend.metadata(),
                    **base.environment(),
                    "logits_encoding": "zlib little-endian float32",
                    "candidate_vector_sha256": {
                        key: value["vector_float64_le_sha256"]
                        for key, value in record["plan"]["candidates"].items()
                    },
                    "forward_ceiling": 12,
                    "derivative_ceiling": 0,
                },
            )
            ledger = base.Ledger(OUTPUT, record["plan"]["cells"], started["deadline_monotonic"])
            rows = evaluate(record["plan"], backend, vectors, ledger, OUTPUT)
            base.write_new(OUTPUT / "analysis.json", summarize(rows))
    except BaseException as error:  # noqa: BLE001 - one bounded failure, no traceback/rescue.
        failure = {
            "status": "INCONCLUSIVE",
            "exception": capture.exception_record(error),
            "eligibility_failure": isinstance(error, EligibilityError),
            "retries_allowed": False,
        }
        try:
            budget.write_bytes(OUTPUT / "INVALID.json", bindings.encoded_json(failure))
        except BaseException as receipt_error:  # noqa: BLE001 - disclose once; no recursive write.
            failure["invalid_receipt_persisted"] = False
            failure["receipt_exception"] = capture.exception_record(receipt_error)
        print(json.dumps(failure), flush=True)
        raise SystemExit(1) from None


def supervise(command, output, usage, timeout=300):
    from scripts.verify_frozen_three_family_comply_f04 import finalize_recording

    output, started = Path(output), time.monotonic()
    require(timeout == 300, "fixed300s including loading; no extension")
    budget = TransferBudget(output)
    receipt, entered = None, False
    try:
        budget.write_bytes(
            output / "RUN_STARTED.json",
            bindings.encoded_json(
                {
                    "command": command,
                    "started_monotonic": started,
                    "deadline_monotonic": started + timeout,
                    "timeout_seconds": timeout,
                    "usage_preflight": usage,
                    "forward_ceiling": 12,
                    "derivative_ceiling": 0,
                }
            ),
        )
        entered = True
        receipt = capture.run_capture(command, budget, started + timeout, cwd=ROOT)
        if receipt.get("quiescent") is not True:
            # No mutable journal or artifact reads while a writer may still be active.
            result = {
                "status": "INCONCLUSIVE",
                "reason": "worker/writer quiescence unconfirmed",
                "forward_attempts": None,
                "completed_forwards": None,
                "derivative_attempts": None,
                "elapsed_seconds": time.monotonic() - started,
                "cleanup_error": receipt.get("cleanup_error"),
                "retries_allowed": False,
            }
        else:
            reason = (
                None
                if receipt["status"] == "complete_valid"
                else "technical bounded capture/worker failure"
            )
            if (output / "INVALID.json").exists() or not (output / "analysis.json").exists():
                reason = reason or "worker incomplete or invalid"
            fa, fc, fi = engine.recorder.journal_counts(output / "forward_events.jsonl")
            da, dc, di = engine.recorder.journal_counts(output / "derivative_events.jsonl")
            elapsed = time.monotonic() - started
            if fa != 12 or fc != 12 or fi or da != 0 or dc != 0 or di or elapsed > timeout:
                reason = reason or "exact12/0 journal/deadline fault"
            result = {
                "status": "complete_valid" if reason is None else "INCONCLUSIVE",
                "reason": reason,
                "forward_attempts": fa,
                "completed_forwards": fc,
                "derivative_attempts": da,
                "elapsed_seconds": elapsed,
                "cleanup_error": receipt.get("cleanup_error"),
                "retries_allowed": False,
            }
    except BaseException as error:  # noqa: BLE001 - bounded terminal technical fault.
        result = {
            "status": "INCONCLUSIVE",
            "exception": capture.exception_record(error),
            "retries_allowed": False,
        }
        if receipt is None:
            receipt = {
                "status": "INCONCLUSIVE",
                "quiescent": not entered,
                "worker_started": None if entered else False,
                "exception": capture.exception_record(error),
            }
    return finalize_recording(budget, receipt, result)


def run():
    require_authorization()
    preflight()
    usage = checked_usage(json.loads(os.environ.get("SP_LENSE_USAGE_PREFLIGHT", "null")))
    return supervise([sys.executable, "-u", str(ROOT / protocol.SCRIPT), "_worker"], OUTPUT, usage)


if __name__ == "__main__":
    commands = {"freeze": freeze, "preflight": preflight, "run": run, "_worker": worker}
    require(
        len(sys.argv) == 2 and sys.argv[1] in commands,
        "Use freeze/preflight/run; no adjustable recipes",
    )
    result = commands[sys.argv[1]]()
    print(json.dumps(result, indent=2, allow_nan=False))
    if sys.argv[1] == "run" and result["status"] != "complete_valid":
        raise SystemExit(1)
