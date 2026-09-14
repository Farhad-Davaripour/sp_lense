"""Root-owned release, zero-fit preflight and external 600-second watch for the
prompted-compression integration (job compression_integration_20260914_takeover_v1).

The loader and compression wrapper live in ``compression_integration_v1``. This
wrapper owns only the prospective versioned plan (source/runtime/input hashes),
the exclusive release, the child process boundary and the 256 MiB total-output
bound. It never authors a capture lock, never loads a model/tokenizer, never
touches the holdout and never edits a reviewed core or old runner.

Modes::

    prepare   --prompted-lock <completed prompted lock> [--run-id ID]
    preflight --plan PLAN --sha256 SHA        # zero estimator/PCA fits
    run       --plan PLAN --sha256 SHA [--review REVIEW]   # release + watch
    worker    --plan PLAN --sha256 SHA        # internal, watched child
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import compression_comparison_v1 as core
import native_development_runner_v2 as native
import compression_integration_v1 as integration

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN_V1 = HERE / "COMPRESSION_INTEGRATION_FIT_PLAN_V1.json"
PLAN_V2 = HERE / "COMPRESSION_INTEGRATION_FIT_PLAN_V2.json"
DEFAULT_REVIEW = HERE / "COMPRESSION_INTEGRATION_REVIEW_V1.md"
DEFAULT_SPAN_PLAN = HERE / "SPAN_FIT_PLAN_V2.json"
DEFAULT_RUN_ID = "compression_supervised_20260914_v1"
BASELINE_DIR = HERE / "runs" / integration.BASELINE_RUN_ID
STATUS_RESERVE = 16384


def sha_path(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, sort_keys=True)
        stream.write("\n")


def pin(path, root=ROOT):
    resolved = Path(path).resolve()
    relative = resolved.relative_to(Path(root).resolve()).as_posix()
    return {"path": relative, "sha256": sha_path(resolved), "bytes": resolved.stat().st_size}


def source_pins():
    return {relative: sha_path(ROOT / relative) for relative in integration.SOURCE_FILES}


def runtime_pins():
    return {name: importlib.metadata.version(name) for name in integration.RUNTIME_PACKAGES}


def no_native_modules():
    assert not any(name.split(".")[0] in native.PACKAGES for name in sys.modules), "native module loaded"


def prepare(args):
    lock_path = Path(args.prompted_lock).resolve()
    lock_raw = lock_path.read_bytes()
    lock = json.loads(lock_raw)
    assert lock.get("schema") == integration.PROMPTED_LOCK_SCHEMA, "prompted lock schema"
    assert lock.get("scientific_execution_authorized") is True, "prompted lock not authorized"
    run_id = lock.get("run_id")
    assert isinstance(run_id, str) and run_id, "prompted lock run_id"
    run_dir = (HERE / "runs" / run_id).resolve()
    assert run_dir.is_dir(), "prompted run directory missing"
    assert lock["caps"].get("forwards") == integration.FORWARDS, "prompted forward ceiling"
    pins = {name: pin(run_dir / name) for name in ("index.json", "windows.f32",
                                                   "capture_receipt.json", "supervisor_success.json")}
    assert lock.get("caps", {}).get("fits") == 0 and lock.get("caps", {}).get("derivatives") == 0, "prompted caps"
    plan = {
        "schema": integration.PLAN_SCHEMA, "job_id": integration.JOB_ID, "run_id": args.run_id,
        "seconds": integration.SECONDS, "output_bytes": integration.OUTPUT_BYTES,
        "cv_fit_limit": integration.CV_FIT_LIMIT, "refit_limit": integration.REFIT_LIMIT,
        "pca_fit_limit": integration.PCA_FIT_LIMIT, "holdout_access": False,
        "threads": 1, "C": list(core.C_VALUES), "thresholds": list(core.TAUS),
        "folds": list(core.FOLDS), "conditions": list(core.CONDITIONS),
        "source_files": source_pins(), "runtime_packages": runtime_pins(),
        "reference_span": {"run_id": "span_capture_20260914_v1", "plan": pin(DEFAULT_SPAN_PLAN)},
        "reference_baseline": {"run_id": integration.BASELINE_RUN_ID,
                               "files": {name: pin(BASELINE_DIR / name) for name in integration.BASELINE_NAMES}},
        "prompted_source": {"run_id": run_id, "forwards": lock["caps"]["forwards"],
                            "tokens_per_view": lock["caps"]["tokens_per_view"], "lock": pin(lock_path),
                            "index": pins["index.json"], "windows": pins["windows.f32"],
                            "receipt": pins["capture_receipt.json"], "success": pins["supervisor_success.json"]},
        "scientific_execution_authorized": False,
    }
    save_new(PLAN_V1, plan)
    print(json.dumps({"plan": str(PLAN_V1), "sha256": sha_path(PLAN_V1)}))
    return 0


def preflight(args):
    no_native_modules()
    result = integration.preflight(args.plan, args.sha256, ROOT)
    print(json.dumps(result, sort_keys=True))
    return 0


def worker(args):
    no_native_modules()
    plan = integration.load_plan(args.plan, args.sha256)
    assert plan.get("scientific_execution_authorized") is True, "plan not released"
    assert sha_path(Path(__file__).resolve()) == plan.get("supervisor_source_sha256"), "supervisor source pin"
    assert not (HERE / "runs" / "native_model_owner.json").exists(), "native model active"
    output = integration.output_dir(plan, ROOT)
    assert not output.exists(), "output exists"
    from threadpoolctl import threadpool_limits
    started = time.monotonic()
    with threadpool_limits(limits=plan.get("threads", 1)):
        result = integration.run(args.plan, args.sha256, ROOT,
                                 deadline=lambda: time.monotonic() - started > plan["seconds"])
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    assert total <= plan["output_bytes"], "output cap"
    report = {"status": "worker_complete", "run_id": plan["run_id"], "plan_sha256": args.sha256,
              "counters": result["counters"], "output_bytes": total, "holdout_accessed": False}
    save_new(output / "worker_complete.json", report)
    print(json.dumps({key: report[key] for key in ("status", "run_id", "counters", "output_bytes")},
                     sort_keys=True))
    return 0


def run(args):
    plan_path = Path(args.plan)
    plan = integration.load_plan(plan_path, args.sha256)
    assert plan.get("scientific_execution_authorized") is False, "already released"
    review = Path(args.review or DEFAULT_REVIEW)
    text = review.read_text(encoding="utf-8").lower()
    assert "pass_scoped" in text, "independent review not admitted"
    integration.verify_sources_and_runtime(plan, ROOT)
    output = integration.output_dir(plan, ROOT)
    assert not output.exists() and not (HERE / "runs" / "native_model_owner.json").exists(), "output or owner exists"
    released = dict(plan)
    released["scientific_execution_authorized"] = True
    released["supervisor_source_sha256"] = sha_path(Path(__file__).resolve())
    released["review"] = {"path": review.resolve().relative_to(ROOT).as_posix(), "sha256": sha_path(review)}
    save_new(PLAN_V2, released)
    release_sha = sha_path(PLAN_V2)
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--plan", str(PLAN_V2),
               "--sha256", release_sha]
    started = time.monotonic()
    try:
        pid, _raw = native.watch(command, str(ROOT), plan["seconds"])
        result = read_json(output / "development_results.json")
        counters = result.get("counters", {})
        assert result.get("status") == "COMPLETE", "child result status"
        assert counters.get("cv_fits") == integration.CV_FIT_LIMIT, "cv budget"
        assert counters.get("refits") == integration.REFIT_LIMIT, "refit budget"
        assert counters.get("pca_fits", integration.PCA_FIT_LIMIT + 1) <= integration.PCA_FIT_LIMIT, "pca budget"
        receipt = {"status": "complete", "pid": pid, "elapsed_seconds": time.monotonic() - started,
                   "plan_sha256": release_sha, "results_sha256": sha_path(output / "development_results.json"),
                   "counters": counters, "run_id": plan["run_id"], "holdout_accessed": False}
        raw = (json.dumps(receipt, indent=2, allow_nan=False, sort_keys=True) + "\n").encode()
        total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) + len(raw)
        assert total <= plan["output_bytes"] - STATUS_RESERVE, "output cap"
        native.write_new(output / "supervisor_success.json", raw)
        print(json.dumps({key: receipt[key] for key in ("status", "run_id", "counters", "results_sha256")},
                         sort_keys=True))
        return 0
    except BaseException as exc:
        failure = {"status": "failed", "run_id": plan["run_id"], "error": str(exc)[:1024],
                   "error_type": type(exc).__name__, "elapsed_seconds": time.monotonic() - started,
                   "child_pid": getattr(exc, "owned_child_pid", None),
                   "child_closed": getattr(exc, "owned_child_closed", None)}
        destination = output / "supervisor_failure.json" if output.exists() else \
            HERE / "runs" / (plan["run_id"] + "_supervisor_failure.json")
        try:
            save_new(destination, failure)
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight", "run", "worker"])
    parser.add_argument("--plan")
    parser.add_argument("--sha256")
    parser.add_argument("--prompted-lock")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--review")
    args = parser.parse_args(argv)
    try:
        if args.mode == "prepare":
            assert args.prompted_lock, "--prompted-lock required"
            return prepare(args)
        assert args.plan and args.sha256, "--plan and --sha256 required"
        return {"preflight": preflight, "run": run, "worker": worker}[args.mode](args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "code": type(exc).__name__, "detail": str(exc)[:1024]}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
