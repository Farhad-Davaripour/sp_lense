"""Generate the finite ``jlens_pilot_execution.v2`` lock for the cached pilot.

Run in the locked classifier runtime (the same ``.runtime`` that will execute
Phase B). It computes every hash from the worktree and the pinned inputs; it
does not fabricate a value, load a tensor body, tokenize, fit or run anything.
The ``source_commit`` must be a commit whose blobs are byte-equal to the current
worktree for every ``run_jlens_pilot_v2.SOURCE_PATHS`` entry.

Usage:
    .runtime/Scripts/python.exe development/jlens_trigger_v1/prepare_pilot_lock_v2.py \
        --commit <40-hex> --export-dir development/jlens_trigger_v1/runs/lens_export_20260914_v1 \
        --out development/jlens_trigger_v1/JLENS_PILOT_LOCK_V1.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import jlens_io_v1 as io
import jlens_pilot_v1 as pilot
import run_jlens_pilot_v2 as module

runner = io.runner
ROOT = Path(__file__).resolve().parents[2]
STUDY = Path("development/jlens_trigger_v1")

DEFAULT_EXPORT_DIR = (STUDY / "runs" / "lens_export_20260914_v1").as_posix()
DEFAULT_OUT = (STUDY / "JLENS_PILOT_LOCK_V1.json").as_posix()
DEFAULT_PLAN = (STUDY / "JLENS_PILOT_PLAN_V1.json").as_posix()
DEFAULT_SURFACES = (STUDY / "JLENS_SURFACE_TOKEN_RECEIPT_V1.json").as_posix()
RUN_ID = "jlens_pilot_20260914_v1"
PROVIDER_FILES = ("numpy/__init__.py", "scipy/__init__.py", "sklearn/__init__.py")


def need(condition, code):
    if not condition:
        raise io.GateError(code)


def runtime_block():
    prefix = Path(runner.runtime_prefix())
    packages = {
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
    }
    provider_sources = {}
    for relative in PROVIDER_FILES:
        path = runner.relative_path(prefix, "Lib/site-packages/" + relative)
        provider_sources[relative] = runner.sha(runner.small_bytes(path))
    return {
        "prefix": str(prefix),
        "python": ".".join(map(str, sys.version_info[:3])),
        "packages": packages,
        "provider_sources": provider_sources,
        "classifier_sources": dict(pilot.CLASSIFIER_SOURCES),
    }


def source_block(root):
    sources = {}
    for relative in module.SOURCE_PATHS:
        path = runner.relative_path(root, relative)
        sources[relative] = runner.sha(runner.small_bytes(path))
    return sources


def file_pin(root, relative):
    path = runner.relative_path(root, relative)
    need(path.is_file() and not path.is_symlink(), "EXPORT_FILE")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": io.file_sha256(path)}


def build(root=ROOT, commit=None, export_dir=DEFAULT_EXPORT_DIR, plan_path=DEFAULT_PLAN,
          surfaces_path=DEFAULT_SURFACES):
    root = Path(root).resolve()
    need(type(commit) is str and len(commit) == 40, "COMMIT_FORMAT")
    plan = runner.strict_json(runner.small_bytes(root / plan_path))
    known = plan["known_pins"]
    receipt = runner.strict_json(runner.small_bytes(root / surfaces_path))
    surfaces = [
        {"surface": item["surface"], "token_id": item["token_id"], "single_token": item["single_token"]}
        for item in receipt["surfaces"]
    ]
    need(len(surfaces) == 6 and all(item["single_token"] is True for item in surfaces), "SURFACE_TABLE")

    export_npz = file_pin(root, (Path(export_dir) / "lens_jacobians.npz").as_posix())
    export_receipt = file_pin(root, (Path(export_dir) / "lens_export_receipt.json").as_posix())
    lock = {
        "schema": module.SCHEMA,
        "job_id": pilot.JOB_ID,
        "run_id": RUN_ID,
        "release": io.RELEASE_V1,
        "scientific_execution_authorized": True,
        "caps": dict(pilot.CAPS),
        "source_commit": commit,
        "source_files": source_block(root),
        "runtime": runtime_block(),
        "threads": {"intra": 1, "inter": 1},
        "snapshot_cache_root": known["lens"]["cache_root"],
        "inputs": {
            "lens": known["lens"],
            "lens_export": {
                "path": export_npz["path"],
                "bytes": export_npz["bytes"],
                "sha256": export_npz["sha256"],
                "receipt": export_receipt,
            },
            "model_snapshot_lock": known["model"]["snapshot_lock"],
            "unprompted_cache": known["unprompted_cache"],
            "prompted_cache": known["prompted_cache"],
            "manifests": known["manifests"],
            "surfaces": surfaces,
        },
    }
    need(set(lock["inputs"]) == set(module.INPUT_ROLES), "INPUT_ROLES")
    need(set(lock["source_files"]) == set(module.SOURCE_PATHS), "SOURCE_SET")
    return lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--surfaces", default=DEFAULT_SURFACES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        lock = build(commit=args.commit, export_dir=args.export_dir,
                     plan_path=args.plan, surfaces_path=args.surfaces)
        raw = runner.encoded(lock)
        out = (ROOT / args.out).resolve()
        need(out.is_relative_to(ROOT), "OUTPUT_SCOPE")
        runner.write_new(out, raw)
        print(runner.encoded({"status": "lock_written", "path": args.out,
                              "bytes": len(raw), "sha256": runner.sha(raw)}).decode().strip())
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI reports a structured failure
        print(runner.encoded({"status": "failed", "code": type(exc).__name__,
                              "detail": str(exc)[:1024]}).decode().strip())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
