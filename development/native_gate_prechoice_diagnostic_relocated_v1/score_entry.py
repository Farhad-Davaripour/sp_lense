"""Approval-guarded, model-free score entry for the fixed PRECHOICE diagnostic.

Thin wrapper only. It admits one supervisor-pinned fixed release, authenticates the
already-frozen capture through the existing functions, and delegates every score to
diagnostic_scoring and independent_math_check. No model, tokenizer or provider is
imported; no fitting, replay, re-encoding, retry or overwrite occurs.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTURE = HERE / "capture"
RELEASE_FILE = HERE / "score_release" / "RELEASE.json"
SCORE_EVIDENCE = HERE / "score_evidence"
ATTEMPT = "prechoice_diagnostic_score_attempt_001"
OUTPUT = SCORE_EVIDENCE / ATTEMPT
OUTPUT_REL = "score_evidence/" + ATTEMPT
RELEASE_SCHEMA = "prechoice_fixed_score_release.v1"
CAPTURE_RELEASE_SHA256 = "2adf26dd31bf1c34bcffdd06c1ab139ec51ee8760fa5379cafab24ef49625c42"
CAPTURE_SOURCE_SHA256 = "f53cafd99e28827e00eb0a2cfcf519fc6e444257a49a37fe5b4cc4a2e8872f5f"
LIMITS = {"primary_cases": 8, "independent_cases": 8, "phase_seconds": 10,
          "phase_output_bytes": 65536, "total_output_bytes": 131072}
SOURCE_PATHS = ("score_entry.py", "diagnostic_scoring.py", "diagnostic_reader.py",
                "atomic_result_writer.py", "independent_math_check.py",
                "capture/fit_source_auth.py")
RELEASE_BYTES = 65536
SOURCE_BYTES = 5 * 1024 ** 2


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _decode(raw):
    def unique(pairs):
        out = {}
        for key, item in pairs:
            need(key not in out, "DUPLICATE_JSON_KEY")
            out[key] = item
        return out

    def bad(_value):
        raise ValueError("JSON_NONFINITE")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=bad)


def _read(path, cap, code):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap, code)
    return path.read_bytes()


def _admit(approved_sha):
    """Fixed-release admission; runs before any import or real-data/gate access."""
    need(type(approved_sha) is str and len(approved_sha) == 64
         and all(c in "0123456789abcdef" for c in approved_sha), "APPROVED_SHA_EXPLICIT_64_HEX")
    raw = _read(RELEASE_FILE, RELEASE_BYTES, "RELEASE_FILE_BOUND")
    need(sha(raw) == approved_sha, "RELEASE_SHA256")
    release = _decode(raw)
    need(type(release) is dict and release.get("schema") == RELEASE_SCHEMA, "RELEASE_SCHEMA")
    need(release.get("approved") is True, "RELEASE_NOT_APPROVED")
    need(release.get("attempt") == ATTEMPT, "RELEASE_ATTEMPT")
    need(release.get("limits") == LIMITS, "RELEASE_LIMITS")
    need(release.get("output") in (OUTPUT_REL, str(OUTPUT)), "RELEASE_OUTPUT")
    need(release.get("capture_release_sha256") == CAPTURE_RELEASE_SHA256, "RELEASE_CAPTURE_RELEASE")
    need(release.get("capture_source_sha256") == CAPTURE_SOURCE_SHA256, "RELEASE_CAPTURE_SOURCE")
    need(sha(_read(CAPTURE / "root_release" / "RELEASE.json", SOURCE_BYTES, "CAPTURE_RELEASE_BOUND"))
         == CAPTURE_RELEASE_SHA256, "ACTUAL_CAPTURE_RELEASE_SHA256")
    need(sha(_read(CAPTURE / "SOURCE_FREEZE.json", SOURCE_BYTES, "CAPTURE_SOURCE_BOUND"))
         == CAPTURE_SOURCE_SHA256, "ACTUAL_CAPTURE_SOURCE_SHA256")
    sources = release.get("source_sha256")
    need(type(sources) is dict and set(sources) == set(SOURCE_PATHS), "RELEASE_SOURCE_KEYS")
    for name in SOURCE_PATHS:
        path = HERE / name
        need(path.is_file() and not path.is_symlink() and path.stat().st_size <= SOURCE_BYTES, "SOURCE_FILE_BOUND")
        need(sha(path.read_bytes()) == sources[name], "SOURCE_SHA256")
    return release


def _load_modules():
    """Import production modules after admission, capture directory first, path restored."""
    import importlib, sys
    path = list(sys.path)
    names = ("fit_source_auth", "input_reader", "support", "index_contract", "authority", "counts")
    saved = {name: sys.modules.pop(name, None) for name in names}
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(CAPTURE))
    try:
        scoring = importlib.import_module("diagnostic_scoring")
        indep = importlib.import_module("independent_math_check")
        fsa = importlib.import_module("fit_source_auth")
        reader = importlib.import_module("input_reader")
    finally:
        sys.path[:] = path
        for name, module in saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module
    return scoring, indep, fsa, reader


def _execute(release, *, output_dir=None, _modules=None, _primary_deadline=None):
    """Run the two 10-second phases over authenticated rows; publish once, no retry."""
    lim = release["limits"]
    deadline = time.monotonic() + lim["phase_seconds"] if _primary_deadline is None else _primary_deadline
    need(time.monotonic() < deadline, "PRIMARY_DEADLINE")
    need(release.get("output") in (OUTPUT_REL, str(OUTPUT)), "EXECUTE_OUTPUT")
    out = Path(output_dir) if output_dir is not None else OUTPUT
    out.mkdir(parents=True, exist_ok=False)
    scoring, indep, fsa, reader = tuple(_modules) if _modules is not None else _load_modules()

    manifest = fsa.build_manifest(deadline=deadline)
    need(manifest["execution"]["release_sha256"] == CAPTURE_RELEASE_SHA256
         and manifest["execution"]["source_freeze_sha256"] == CAPTURE_SOURCE_SHA256,
         "MANIFEST_CAPTURE_IDENTITY")
    selection = manifest["selection"]
    fixed = tuple(reader.keys())
    need(tuple(x["case"] for x in selection) == fixed, "MANIFEST_SELECTION_KEYS")
    need(len(selection) == 2 * lim["primary_cases"], "MANIFEST_ROW_COUNT")
    labels = tuple(x["label"] for x in selection)
    need(labels == tuple(1 if "_self_shutdown__" in key else -1 for key in fixed), "MANIFEST_LABELS")
    rows, extracted = fsa.extract_features(manifest, deadline=deadline)
    need(len(rows) == 2 * lim["primary_cases"] and tuple(extracted) == labels, "EXTRACTED_ROWS_LABELS")
    pairs = [list(rows[index:index + 2]) for index in range(0, len(rows), 2)]
    need(len(pairs) == lim["primary_cases"], "PAIR_COUNT")

    model = scoring.load_model()
    primary = scoring.score(model, scoring.row_document(pairs), deadline=deadline)
    need(primary.get("total") == lim["primary_cases"]
         and len(primary.get("cases", ())) == lim["primary_cases"], "PRIMARY_CASE_COUNT")
    need(len(_bytes(primary)) <= lim["phase_output_bytes"], "PRIMARY_OUTPUT_BOUND")
    need(time.monotonic() < deadline, "PRIMARY_DEADLINE")
    scoring.write_result(primary, out / "PRIMARY.json")
    primary_sha256 = sha(_bytes(primary))
    need(sha(_read(out / "PRIMARY.json", lim["phase_output_bytes"], "PRIMARY_FILE_BOUND"))
         == primary_sha256, "SAVED_PRIMARY_BYTES")
    need(time.monotonic() < deadline, "PRIMARY_DEADLINE")

    keys8 = tuple(scoring.keys())
    indep_deadline = time.monotonic() + lim["phase_seconds"]
    computed_all = True
    scores_all = True
    routes_all = True
    records = []
    for index, key in enumerate(keys8):
        need(time.monotonic() < indep_deadline, "INDEPENDENT_DEADLINE")
        case = primary["cases"][index]
        need(case["case_key"] == key, "PRIMARY_KEY_ORDER")
        try:
            reference = indep.case_scores(pairs[index], model.mu, model.heads)
            valid = (reference.get("route") in ("ON", "OFF") and len(reference.get("scores", ())) == 3
                     and all(type(v) in (int, float) and math.isfinite(v) for v in reference["scores"]))
        except Exception:
            reference = None
            valid = False
        if not valid:
            computed_all = False
            scores_all = False
            routes_all = False
            records.append({"case_key": key, "numerical_valid": False, "independent_scores": None,
                            "independent_route": None, "primary_scores": list(case["head_scores"]),
                            "primary_route": case["route"], "scores_match": False, "route_match": False,
                            "classifier_correct": bool(case["correct"])})
            continue
        scores_match = list(case["head_scores"]) == list(reference["scores"])
        route_match = case["route"] == reference["route"]
        scores_all = scores_all and scores_match
        routes_all = routes_all and route_match
        records.append({"case_key": key, "numerical_valid": True, "independent_scores": list(reference["scores"]),
                        "independent_route": reference["route"], "primary_scores": list(case["head_scores"]),
                        "primary_route": case["route"], "scores_match": scores_match, "route_match": route_match,
                        "classifier_correct": bool(case["correct"])})
    need(time.monotonic() < indep_deadline, "INDEPENDENT_DEADLINE")
    numerical_valid = computed_all and scores_all and routes_all
    independent = {"schema": "prechoice_diagnostic_independent_result.v1", "attempt": ATTEMPT,
                   "primary_sha256": primary_sha256,
                   "case_count": len(records), "numerical_valid": numerical_valid,
                   "scores_match": scores_all, "routes_match": routes_all,
                   "classifier_correct": primary["correct"], "classifier_total": primary["total"],
                   "records": records}
    need(len(_bytes(independent)) <= lim["phase_output_bytes"], "INDEPENDENT_OUTPUT_BOUND")
    need(len(_bytes(primary)) + len(_bytes(independent)) <= lim["total_output_bytes"], "TOTAL_OUTPUT_BOUND")
    scoring.write_result(independent, out / "INDEPENDENT.json")
    need(time.monotonic() < indep_deadline, "INDEPENDENT_DEADLINE")
    return {"attempt": ATTEMPT, "numerical_valid": numerical_valid, "scores_match": scores_all,
            "routes_match": routes_all, "correct": primary["correct"], "total": primary["total"],
            "primary": str(out / "PRIMARY.json"), "independent": str(out / "INDEPENDENT.json")}


def run(approved_sha):
    """Admit the fixed release, then execute; anything unapproved stops before data."""
    deadline = time.monotonic() + LIMITS["phase_seconds"]
    release = _admit(approved_sha)
    return _execute(release, _primary_deadline=deadline)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Approval-guarded fixed prechoice diagnostic score entry.")
    parser.add_argument("--approved-score-release-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        outcome = run(args.approved_score_release_sha256)
    except Exception as exc:
        print("FAIL %s: %s" % (type(exc).__name__, exc))
        return 1
    print(json.dumps(outcome, sort_keys=True))
    return 0 if outcome["numerical_valid"] and outcome["correct"] == outcome["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
