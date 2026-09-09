"""Model-free candidate preparation and default-deny future construction entry point."""
import argparse
from pathlib import Path
import sys
import time

from gate import METHOD, CONTRACT, artifact, canonical, decode, digest, fit, load_artifact, require
from checker import verify
from source_auth import HERE, build_manifest, extract_features

CORE_FILES = ("gate.py", "checker.py", "source_auth.py", "construction.py")
PER_FILE = 5 * 1024 * 1024
NAMESPACE_CAP = 16 * 1024 * 1024
CONSTRUCTION_CAP = 8 * 1024 * 1024
CONSTRUCTION_PAYLOAD_CAP = 1024 * 1024  # Leaves ample room for a separately owned finite wrapper.
FAULT_RESERVE = 64 * 1024


def source_lock():
    return {name: digest((HERE / name).read_bytes()) for name in CORE_FILES}


def write_new(path, raw):
    require(path.resolve().is_relative_to(HERE.resolve()), "OUTPUT_ESCAPE")
    require(len(raw) <= PER_FILE, "OUTPUT_FILE_LIMIT")
    if path.parent.name == "construction_attempt_001":
        payload_bytes = sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
        require(payload_bytes + len(raw) <= CONSTRUCTION_PAYLOAD_CAP, "CONSTRUCTION_PAYLOAD_LIMIT")
    existing = sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
    require(existing + len(raw) <= NAMESPACE_CAP, "OUTPUT_NAMESPACE_LIMIT")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        import os
        os.fsync(stream.fileno())


def construction_lock(manifest, sources):
    return canonical({"schema": "native_gate_construction_lock_v1", "method": METHOD,
                      "feature_contract": CONTRACT, "rows": 32, "positive": 8, "negative": 24,
                      "training_manifest_sha256": digest(manifest), "source_sha256": digest(sources),
                      "fit_count": 1, "worker_seconds": 60, "cleanup_seconds": 5,
                      "total_bytes": CONSTRUCTION_CAP, "per_file_bytes": PER_FILE,
                      "success": "32_correct_and_independent_check", "retry": False,
                      "evaluation_sequence": "separate_clean_authorship_after_accepted_artifact",
                      "checkpoint_tensor_reads": 0, "model_calls": 0, "tokenizer_calls": 0})


def candidate():
    """Authorized metadata-only milestone. No extraction or fit is called."""
    targets = ("TRAINING_MANIFEST.json", "CORE_SOURCE_LOCK.json", "CONSTRUCTION_LOCK_DRAFT.json", "RELEASE_DRAFT.json")
    require(not any((HERE / name).exists() for name in targets), "CANDIDATE_ALREADY_EXISTS")
    manifest = canonical(build_manifest())
    sources = canonical(source_lock())
    lock = construction_lock(manifest, sources)
    release = canonical({"schema": "native_gate_construction_release_v1", "approved": False,
                         "operation": "one_construction_fit", "construction_lock_sha256": digest(lock),
                         "training_manifest_sha256": digest(manifest), "source_sha256": digest(sources),
                         "model_permission": False, "tokenizer_permission": False,
                         "evaluation_permission": False})
    for name, raw in zip(targets, (manifest, sources, lock, release), strict=True):
        write_new(HERE / name, raw)
    return {"status": "MODEL_FREE_CANDIDATE", "files": {name: digest(raw) for name, raw in
            zip(targets, (manifest, sources, lock, release), strict=True)},
            "selected_rows": 32, "real_feature_fits": 0, "feature_coordinates_emitted": False,
            "model_calls": 0, "tokenizer_calls": 0, "checkpoint_tensor_reads": 0}


def authorize(release_raw, expected_release_sha256, manifest_raw, lock_raw, sources_raw):
    require(expected_release_sha256 and digest(release_raw) == expected_release_sha256, "RELEASE_HASH")
    release = decode(release_raw)
    require(type(release) is dict and release.get("approved") is True, "RELEASE_DEFAULT_DENY")
    expected = {"schema": "native_gate_construction_release_v1", "approved": True,
                "operation": "one_construction_fit", "construction_lock_sha256": digest(lock_raw),
                "training_manifest_sha256": digest(manifest_raw), "source_sha256": digest(sources_raw),
                "model_permission": False, "tokenizer_permission": False, "evaluation_permission": False}
    require(canonical(release) == canonical(expected), "RELEASE_SCOPE_OR_BINDING")
    require(canonical(decode(sources_raw)) == canonical(source_lock()), "CORE_SOURCE_CHANGED")
    require(canonical(decode(lock_raw)) == construction_lock(manifest_raw, sources_raw), "CONSTRUCTION_LOCK_CHANGED")
    return release


def construct(release_path, expected_release_sha256):
    """Future root-released entry point; never invoked on real rows in this milestone.

    Deadline checks cover each bounded phase; an external retained deadline owner is
    still required before any real release. This is not a production launcher.
    """
    started = time.monotonic()
    deadline = started + 60.0
    manifest_raw = (HERE / "TRAINING_MANIFEST.json").read_bytes()
    lock_raw = (HERE / "CONSTRUCTION_LOCK_DRAFT.json").read_bytes()
    sources_raw = (HERE / "CORE_SOURCE_LOCK.json").read_bytes()
    release_raw = Path(release_path).read_bytes()
    release = authorize(release_raw, expected_release_sha256, manifest_raw, lock_raw, sources_raw)
    destination = HERE / "construction_attempt_001"
    destination.mkdir(exist_ok=False)  # Immutable one-shot claim, retained even after failure.
    fit_count = 0
    scientific_outcome = None
    try:
        manifest = decode(manifest_raw)
        rows, labels = extract_features(manifest, deadline=deadline)
        require(len(rows) == 32 and all(len(row) == 1024 for row in rows)
            and len(labels) == 32 and labels.count(1) == 8 and labels.count(-1) == 24, "EXACT_TRAINING_SET")
        require(time.monotonic() - started < 60, "CONSTRUCTION_DEADLINE")
        fit_count = 1
        model = fit(rows, labels)
        require(time.monotonic() - started < 60, "CONSTRUCTION_DEADLINE")
        checked = verify(rows, labels, model)
        scientific_outcome = "CONSTRUCTION_PASS" if checked["correct"] == 32 else "CONSTRUCTION_FIT_FAIL"
        feature_hash = digest(canonical({"rows": rows, "labels": labels}))
        bindings = {"training_manifest_sha256": digest(manifest_raw),
                    "construction_lock_sha256": release["construction_lock_sha256"],
                    "feature_sha256": feature_hash, "source_sha256": digest(sources_raw)}
        raw = artifact(model, **bindings)
        loaded = load_artifact(raw, expected_sha256=digest(raw), expected_bindings=bindings)
        require([model.score(row) for row in rows] == [loaded.score(row) for row in rows], "RELOAD_DIFFERENCE")
        require(time.monotonic() - started < 60, "CONSTRUCTION_DEADLINE")
        status = scientific_outcome
        result = {"status": status, "scientific_pass": checked["correct"] == 32,
                  "check": checked, "artifact_sha256": digest(raw), "real_feature_fits": fit_count,
                  "model_calls": 0, "tokenizer_calls": 0, "checkpoint_tensor_reads": 0,
                  "elapsed_seconds": time.monotonic() - started}
        result_raw = canonical(result)
        require(len(raw) + len(result_raw) <= CONSTRUCTION_PAYLOAD_CAP - FAULT_RESERVE, "CONSTRUCTION_STORAGE")
        write_new(destination / "FITTED_GATE.json", raw)
        write_new(destination / "RESULT.json", result_raw)
        return result
    except Exception as error:
        # Fixed error codes only; never serialize residuals or arbitrary exception strings.
        failure = {"status": "CONSTRUCTION_FIT_FAIL" if scientific_outcome == "CONSTRUCTION_FIT_FAIL" else "TECHNICAL_INCONCLUSIVE",
                   "scientific_outcome": scientific_outcome, "scientific_pass": False,
                   "technical_inconclusive": True, "exception_type": type(error).__name__,
                   "real_feature_fits": fit_count, "model_calls": 0, "tokenizer_calls": 0,
                   "checkpoint_tensor_reads": 0, "elapsed_seconds": time.monotonic() - started}
        # Preserve any partly/fully published result; an independent fault receipt
        # invalidates technical completion without erasing an earlier scientific fail.
        failure_path = destination / ("TECHNICAL_FAULT.json" if (destination / "RESULT.json").exists() else "RESULT.json")
        write_new(failure_path, canonical(failure))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("candidate", "fit"))
    parser.add_argument("--release")
    parser.add_argument("--release-sha256")
    args = parser.parse_args()
    if args.command == "candidate":
        require(args.release is None and args.release_sha256 is None, "CANDIDATE_NO_RELEASE")
        result = candidate()
    else:
        require(args.release and args.release_sha256, "RELEASE_REQUIRED")
        result = construct(args.release, args.release_sha256)
    print(canonical(result).decode("ascii"))


if __name__ == "__main__":
    main()
