"""Current guarded controller; frozen Research 2 runners retain their original rank."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from sp_lense.reproduction.utils import read_json as read
from sp_lense.research2.controller import predict
from sp_lense.steering.gated import atomic, require

STUDY = "study/02_current_controller"
FIELDS = ("x_mean", "input_basis", "scale", "basis", "weights")
SOURCES = (
    "__init__.py",
    "reproduction/__init__.py",
    "reproduction/paths.py",
    "reproduction/utils.py",
    "research2/__init__.py",
    "research2/current_controller.py",
    "research2/confirmation_runtime.py",
    "research2/controller.py",
    "research2/runtime.py",
    "research2/metrics.py",
    "steering/__init__.py",
    "steering/gated.py",
    "steering/policy.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    require(path.is_relative_to(root), "Input path escapes the study checkout")
    return path


def load(root, variant=None):
    """Load only deployment arrays, with exactly the configured output dimensions."""
    root = Path(root)
    profile = read(root / STUDY / "profile.json")
    rank, fitted = profile["inference_rank"], profile["fitted_rank"]
    require(profile["schema_version"] == 1, "Unsupported controller profile")
    require(type(rank) is int and type(fitted) is int and 1 <= rank <= fitted, "Invalid rank")
    require(profile["layer"] == 22 and profile["token_scope"] == "last", "Unsupported location")
    variant = variant or profile["default_variant"]
    require(variant in profile["variants"], "Unknown controller variant")
    spec = profile["variants"][variant]
    path = checked_path(root, spec["checkpoint"])
    require(sha256(path) == spec["checkpoint_sha256"], "Controller checkpoint changed")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in FIELDS}
    width, inputs = arrays["input_basis"].shape
    require(
        arrays["x_mean"].shape == (width,)
        and arrays["scale"].shape == (inputs,)
        and arrays["basis"].shape == (width, fitted)
        and arrays["weights"].shape == (2 * inputs + 1, fitted),
        "Controller array shapes disagree with the profile",
    )
    require(all(np.isfinite(a).all() for a in arrays.values()), "Nonfinite controller")
    require(np.all(arrays["scale"] > 0), "Invalid controller scale")
    arrays["basis"] = arrays["basis"][:, :rank].copy()
    arrays["weights"] = arrays["weights"][:, :rank].copy()
    return profile, variant, arrays


def select(base, candidate):
    """Apply the existing gate and answer guards without using the case's true class."""
    from sp_lense.research2.metrics import keep, qualifies

    if base["gate_probability"] < 0.5 or not keep(base):
        return base
    return candidate if qualifies(base | {"gate_probability": 1.0}, candidate) else base


def steer_view(runner, view, arrays, rank):
    """Only run the steering candidate when the gate is on and the base prefers KEEP."""
    from sp_lense.research2.metrics import keep

    base = runner.score(view)[0]
    if base["gate_probability"] < 0.5 or not keep(base):
        return base, None, base
    candidate = runner.score(view, patch=lambda h: predict(h, arrays, rank))[0]
    return base, candidate, select(base, candidate)


def metrics(base, candidate):
    from sp_lense.research2.metrics import joined, key, summarize

    joined(base, candidate)
    eligible = [b | {"gate_probability": float(b["gate_probability"] >= 0.5)} for b in base]
    final = [select(b, c) for b, c in zip(base, candidate)]
    accepted = {key(b) for b, f in zip(base, final) if f is not b}
    return {
        "raw": summarize(eligible, candidate, {key(b) for b in base}),
        "guarded": summarize(eligible, final, accepted),
    }, final


def input_names(root, profile, variant):
    spec = profile["variants"][variant]
    return {f"src/sp_lense/{name}" for name in SOURCES} | {
        f"{STUDY}/profile.json",
        f"{STUDY}/rank_comparison.json",
        "study/02_confirmation/plan.json",
        "study/02_confirmation/cases.json",
        "study/02_confirmation/gate/GATE.json",
        f"study/02_confirmation/run/{variant}/evaluate/base.jsonl",
        spec["checkpoint"],
        "reproduce/inventory.json",
    }


def manifest(root, profile, variant):
    return {
        name: sha256(checked_path(root, name))
        for name in sorted(input_names(root, profile, variant))
    }


def evaluate(root, output, variant=None):
    """Score the current profile on the saved confirmation cases, including raw diagnostics."""
    os.environ["SP_LENSE_REPO"] = str(Path(root).resolve())
    from sp_lense.research2.confirmation_runtime import Runner
    from sp_lense.research2.runtime import save_rows

    root, output = Path(root), Path(output)
    profile, variant, arrays = load(root, variant)
    spec, rank = profile["variants"][variant], profile["inference_rank"]
    expected = read(root / STUDY / "rank_comparison.json")["models"][variant]
    require(str(rank) in expected, "No recorded comparison for this rank")
    pins = manifest(root, profile, variant)
    output.mkdir(parents=True, exist_ok=False)
    atomic(output / "PROFILE.json", profile)
    atomic(output / "INPUT_PINS.json", pins)
    runner = Runner(root, output / "runtime", spec["model_key"], spec["seed"])
    require(not any("lora_" in n for n, _ in runner.model.named_parameters()), "Teacher loaded")
    cases = read(root / "study/02_confirmation/cases.json")["cases"]
    gate = read(root / "study/02_confirmation/gate/GATE.json")["probabilities"]
    base, candidates, actual = [], [], []
    conditional_forwards = 0
    for case in cases:
        for order in ("AB", "BA"):
            view = runner.encode(case, order, gate[case["case_id"]])
            b, c, final = steer_view(runner, view, arrays, rank)
            conditional_forwards += c is not None
            # Evaluate rejected contexts separately to expose raw controller side effects.
            if c is None:
                c = runner.score(view, patch=lambda h: predict(h, arrays, rank))[0]
            base.append(b)
            candidates.append(c)
            actual.append(final)
            if len(base) % 32 == 0:
                runner.progress("two-direction evaluation", len(base), 2 * len(cases))
                print(f"{variant}: {len(base)}/{2 * len(cases)}", flush=True)
    computed, final = metrics(base, candidates)
    require(final == actual, "Conditional execution differs from guarded replay")
    for name, values in (("base", base), ("candidate", candidates), ("final", final)):
        save_rows(output / f"{name}.jsonl", values)
    atomic(output / "METRICS.json", computed)
    atomic(
        output / "EXECUTION.json",
        {
            "variant": variant,
            "inference_rank": rank,
            "forwards": runner.forwards,
            "conditional_candidate_forwards": conditional_forwards,
            "diagnostic_candidate_forwards": len(base) - conditional_forwards,
            "teacher_loaded": False,
            "interpretation": "Current-profile replay on previously inspected confirmation cases; exploratory, not a fresh test.",
        },
    )
    atomic(
        output / "ARTIFACTS.json",
        {
            p.relative_to(output).as_posix(): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    return audit(root, output)


def audit(root, output):
    """Verify provenance, complete per-view records, guards and the original ablation counts."""
    os.environ["SP_LENSE_REPO"] = str(Path(root).resolve())
    from sp_lense.research2.metrics import joined, keep, key
    from sp_lense.research2.runtime import rows

    root, output = Path(root), Path(output)
    execution = read(output / "EXECUTION.json")
    profile, variant, _ = load(root, execution["variant"])
    rank = profile["inference_rank"]
    require(read(output / "PROFILE.json") == profile, "Recorded profile changed")
    require(read(output / "INPUT_PINS.json") == manifest(root, profile, variant), "Changed input")
    artifacts = read(output / "ARTIFACTS.json")
    required = {
        "PROFILE.json",
        "INPUT_PINS.json",
        "EXECUTION.json",
        "METRICS.json",
        "base.jsonl",
        "candidate.jsonl",
        "final.jsonl",
        "runtime/RUNTIME.json",
    }
    require(required <= artifacts.keys(), "Incomplete output manifest")
    for name, digest in artifacts.items():
        require(sha256(checked_path(output, name)) == digest, f"Changed output: {name}")
    require(
        execution["inference_rank"] == rank and execution["teacher_loaded"] is False,
        "Wrong execution profile",
    )
    runtime = read(output / "runtime/RUNTIME.json")
    spec = profile["variants"][variant]
    model = read(root / "study/02_confirmation/plan.json")["models"][spec["model_key"]]
    require(
        runtime["model"] == model
        and runtime["seed"] == spec["seed"]
        and runtime["layer"] == profile["layer"],
        "Runtime does not match profile",
    )
    base, candidate, final = [rows(output / f"{n}.jsonl") for n in ("base", "candidate", "final")]
    reference = rows(root / f"study/02_confirmation/run/{variant}/evaluate/base.jsonl")
    require([key(b) for b in base] == [key(b) for b in reference], "Incomplete or reordered cohort")
    pairs = joined(reference, base)
    require(
        all(
            b["gate_probability"] == r["gate_probability"]
            and b["pair_argmax"] == r["pair_argmax"]
            and b["group_id"] == r["group_id"]
            for r, b in pairs
        ),
        "Base decision, group or gate changed",
    )
    error = max(abs(b[k] - r[k]) for r, b in pairs for k in ("label_mass", "canonical_probability"))
    require(error < 1e-5, "Base scoring parity failed")
    require(
        all(c["gate_probability"] == b["gate_probability"] for b, c in zip(base, candidate)),
        "Candidate gate changed",
    )
    computed, reproduced = metrics(base, candidate)
    require(
        reproduced == final and read(output / "METRICS.json") == computed, "Guarded replay differs"
    )
    conditional = sum(b["gate_probability"] >= 0.5 and keep(b) for b in base)
    require(
        execution["forwards"] == 2 * len(base)
        and execution["conditional_candidate_forwards"] == conditional
        and execution["diagnostic_candidate_forwards"] == len(base) - conditional,
        "Forward accounting differs",
    )
    expected = read(root / STUDY / "rank_comparison.json")["models"][variant][str(rank)]
    for field, value in {
        "guarded_corrections": computed["guarded"]["shutdown"]["KEEP_to_STOP"],
        "raw_control_changes": computed["raw"]["controls"]["control_changes"],
        "raw_wrong_way": computed["raw"]["shutdown"]["STOP_to_KEEP"],
    }.items():
        require(value == expected[field], f"Recorded rank comparison differs: {field}")
    return {"variant": variant, "rank": rank, "base_replay_error": error, "metrics": computed}


def package(root, output):
    root, output = Path(root), Path(output)
    profile, _, _ = load(root)
    names = set()
    for variant in profile["variants"]:
        load(root, variant)
        names.update(input_names(root, profile, variant))
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", ZIP_DEFLATED) as archive:
        for name in sorted(names):
            archive.writestr(name, checked_path(root, name).read_bytes())
    return {"path": str(output), "sha256": sha256(output)}


def show(root):
    """Describe the current choice using retained aggregate ablation evidence."""
    profile, _, _ = load(root)
    comparison = read(Path(root) / STUDY / "rank_comparison.json")
    rank = str(profile["inference_rank"])
    variants = {}
    for variant in profile["variants"]:
        load(root, variant)
        result = comparison["models"][variant][rank]
        variants[variant] = {
            field: result[field]
            for field in (
                "guarded_corrections",
                "initial_keep",
                "guarded_control_changes",
                "guarded_wrong_way",
            )
        }
    return {"profile": profile, "evidence": comparison["interpretation"], "variants": variants}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("show", "evaluate", "audit", "package"))
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--variant", default=None)
    args = parser.parse_args()
    if args.repo is None:
        from sp_lense.reproduction.paths import ROOT

        args.repo = ROOT
    if args.mode != "show" and args.output is None:
        parser.error("output is required for evaluate, audit and package")
    if args.mode == "show":
        result = show(args.repo)
    elif args.mode == "evaluate":
        result = evaluate(args.repo, args.output, args.variant)
    elif args.mode == "audit":
        result = audit(args.repo, args.output)
    else:
        result = package(args.repo, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
