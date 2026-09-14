"""Root-owned release, zero-fit preflight and external 60-second watch for the identity fit.

Job ``identity_fit_pipeline_20260914_v1``. The loader and comparison live in
``identity_fit_v1``; this wrapper owns only the prospective versioned plan
(source/runtime/input hashes), the exclusive release, the single watched child
boundary and the 64 MiB total-output bound. It never authors a capture lock, never
loads a model/tokenizer, never touches the holdout and never edits a reviewed core.

Modes::

    prepare   --lock <completed identity capture lock> [--plan IDENTITY_FIT_PLAN_V1.json]
    preflight --plan PLAN --sha256 SHA        # zero estimator/PCA fits
    run       --plan PLAN --sha256 SHA --review REVIEW   # release + one watch
    worker    --plan PLAN --sha256 SHA        # internal, watched child

``prepare`` pins the completed capture outputs, the identity source files, the
shared manifests and the saved prompted-control artifacts into the same finite
plan file; ``preflight`` authenticates them with zero fits; ``run`` admits a
review, writes the authorized release in place and observes exactly one child.
Failure evidence is preserved in ``supervisor_failure.json``.
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import identity_fit_v1 as fit
import native_development_runner_v2 as native

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE / "IDENTITY_FIT_PLAN_V1.json"
SPAN_PLAN = HERE / "SPAN_FIT_PLAN_V2.json"
IDENTITY_LOCK = HERE / "RUN_LOCK_IDENTITY_CAPTURE_V1.json"
OLD_RUN_DIR = HERE / "runs" / fit.OLD_CONTROL_RUN_ID
OLD_MODEL_RESULTS = OLD_RUN_DIR / ("model_results_%s__%s__%s.json"
                                   % (fit.OLD_CONTROL_CONDITION, fit.OLD_DIMENSION, fit.OLD_FAMILY))
DEFAULT_RUN_ID = "identity_fit_20260914_v1"
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


def save_plan(path, value):
    """The finite prospective plan file is revised in place by prepare/run."""
    raw = json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n"
    Path(path).write_text(raw, encoding="utf-8", newline="\n")


def pin(path, root=ROOT):
    resolved = Path(path).resolve()
    return {"path": resolved.relative_to(Path(root).resolve()).as_posix(),
            "sha256": sha_path(resolved), "bytes": resolved.stat().st_size}


def source_pins():
    return {relative: sha_path(ROOT / relative) for relative in fit.SOURCE_FILES}


def runtime_pins():
    return {name: importlib.metadata.version(name) for name in fit.RUNTIME_PACKAGES}


def no_native_modules():
    assert not any(name.split(".")[0] in native.PACKAGES for name in sys.modules), "native module loaded"


def prepare(args):
    lock_path = Path(args.lock).resolve()
    lock = read_json(lock_path)
    assert lock.get("schema") == fit.IDENTITY_LOCK_SCHEMA, "identity lock schema"
    assert lock.get("scientific_execution_authorized") is True, "identity lock not authorized"
    run_id = lock.get("run_id")
    assert isinstance(run_id, str) and run_id, "identity lock run_id"
    run_dir = (HERE / "runs" / run_id).resolve()
    assert run_dir.is_dir(), "identity capture run directory missing"
    caps = lock["caps"]
    assert caps.get("forwards") == 640, "identity forward ceiling"
    assert caps.get("fits") == 0 and caps.get("derivatives") == 0, "identity caps"
    outputs = {name: pin(run_dir / name) for name in
               ("index.json", "windows.f32", "capture_receipt.json", "supervisor_success.json")}
    span_plan = read_json(SPAN_PLAN)
    plan = {
        "schema": fit.PLAN_SCHEMA, "job_id": fit.JOB_ID, "run_id": args.run_id,
        "status": "RESOLVED_PENDING_REVIEW", "seconds": fit.SECONDS, "output_bytes": fit.OUTPUT_BYTES,
        "classifier_fits": fit.CLASSIFIER_FITS, "cv_fits": fit.CV_FITS, "refit_fits": fit.REFITS,
        "pca_fits": fit.PCA_FITS, "family": fit.FAMILY, "C": fit.C_VALUE,
        "thresholds": list(fit.TAUS), "folds": list(fit.FOLDS), "holdout_access": False,
        "identity": {"run_id": run_id, "lock": pin(lock_path), "index": outputs["index.json"],
                     "windows": outputs["windows.f32"], "receipt": outputs["capture_receipt.json"],
                     "success": outputs["supervisor_success.json"]},
        "manifests": dict(span_plan["manifests"]),
        "old_control": {"run_id": fit.OLD_CONTROL_RUN_ID, "condition": fit.OLD_CONTROL_CONDITION,
                        "dimension": fit.OLD_DIMENSION, "family": fit.OLD_FAMILY, "C": fit.OLD_C,
                        "cv_scores": pin(OLD_RUN_DIR / "cv_scores.json"),
                        "model_results": pin(OLD_MODEL_RESULTS)},
        "source_files": source_pins(), "runtime_packages": runtime_pins(),
        "scientific_execution_authorized": False,
        "pin_status": "RESOLVED",
    }
    save_plan(args.plan, plan)
    print(json.dumps({"plan": str(Path(args.plan).resolve()), "sha256": sha_path(args.plan)}))
    return 0


def preflight(args):
    no_native_modules()
    result = fit.preflight(args.plan, args.sha256, ROOT)
    print(json.dumps(result, sort_keys=True))
    return 0


def worker(args):
    no_native_modules()
    plan = fit.load_plan(args.plan, args.sha256)
    assert plan.get("scientific_execution_authorized") is True, "plan not released"
    assert sha_path(Path(__file__).resolve()) == plan.get("supervisor_source_sha256"), "supervisor source pin"
    output = fit.output_dir(plan, ROOT)
    assert not output.exists(), "output exists"
    from threadpoolctl import threadpool_limits
    started = time.monotonic()
    with threadpool_limits(limits=plan.get("threads", 1)):
        result = fit.run(args.plan, args.sha256, ROOT,
                         deadline=lambda: time.monotonic() - started > plan["seconds"])
    total = _output_bytes(output)
    assert total <= plan["output_bytes"], "output cap"
    report = {"status": "worker_complete", "run_id": plan["run_id"], "plan_sha256": args.sha256,
              "counters": result["counters"], "output_bytes": total, "holdout_accessed": False}
    save_new(output / "worker_complete.json", report)
    print(json.dumps({key: report[key] for key in ("status", "run_id", "counters", "output_bytes")},
                     sort_keys=True))
    return 0


def _output_bytes(output):
    return sum(path.stat().st_size for path in Path(output).rglob("*") if path.is_file())


def run(args):
    plan_path = Path(args.plan)
    plan = fit.load_plan(plan_path, args.sha256)
    assert plan.get("scientific_execution_authorized") is False, "already released"
    review = Path(args.review).resolve()
    text = review.read_text(encoding="utf-8").lower()
    assert "pass_scoped" in text and args.sha256 in text, "independent review not admitted"
    fit.verify_sources_and_runtime(plan, ROOT)
    output = fit.output_dir(plan, ROOT)
    assert not output.exists() and not (HERE / "runs" / "native_model_owner.json").exists(), "output or owner"
    released = dict(plan)
    released["scientific_execution_authorized"] = True
    released["status"] = "RELEASED"
    released["supervisor_source_sha256"] = sha_path(Path(__file__).resolve())
    released["review"] = {"path": review.relative_to(ROOT).as_posix(), "sha256": sha_path(review)}
    save_plan(plan_path, released)
    release_sha = sha_path(plan_path)
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--plan", str(plan_path),
               "--sha256", release_sha]
    started = time.monotonic()
    try:
        pid, _raw = native.watch(command, str(ROOT), plan["seconds"])
        result = read_json(output / "development_results.json")
        counters = result.get("counters", {})
        assert result.get("status") == "COMPLETE", "child result status"
        assert counters.get("cv_fits") == fit.CV_FITS, "cv budget"
        assert counters.get("refits") == fit.REFITS, "refit budget"
        assert counters.get("pca_fits") == fit.PCA_FITS, "pca budget"
        assert counters.get("old_cv_fits") == 0 and counters.get("old_pca_fits") == 0, "old fits"
        receipt = {"status": "complete", "pid": pid, "elapsed_seconds": time.monotonic() - started,
                   "plan_sha256": release_sha, "results_sha256": sha_path(output / "development_results.json"),
                   "counters": counters, "run_id": plan["run_id"], "holdout_accessed": False}
        raw = (json.dumps(receipt, indent=2, allow_nan=False, sort_keys=True) + "\n").encode()
        assert _output_bytes(output) + len(raw) <= plan["output_bytes"] - STATUS_RESERVE, "output cap"
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
    parser.add_argument("--plan", default=str(PLAN))
    parser.add_argument("--sha256")
    parser.add_argument("--lock", default=str(IDENTITY_LOCK))
    parser.add_argument("--review")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args(argv)
    try:
        if args.mode == "prepare":
            return prepare(args)
        assert args.plan and args.sha256, "--plan and --sha256 required"
        if args.mode == "run":
            assert args.review, "--review required"
        return {"preflight": preflight, "run": run, "worker": worker}[args.mode](args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "code": type(exc).__name__, "detail": str(exc)[:1024]}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
