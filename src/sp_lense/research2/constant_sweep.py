"""Fixed-grid constant-mean steering sweep with validation-only magnitude selection."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sp_lense.reproduction.utils import read_json as read
from sp_lense.research2.ablation_studies import checkpoint
from sp_lense.research2.current_controller import checked_path, sha256
from sp_lense.steering.gated import atomic, require

STUDY = "study/02_constant_sweep"
CODE = (
    "research2/constant_sweep.py",
    "research2/ablation_studies.py",
    "research2/current_controller.py",
    "research2/confirmation_runtime.py",
    "research2/confirmation_report.py",
    "research2/controller.py",
    "research2/runtime.py",
    "research2/metrics.py",
    "steering/gated.py",
    "steering/policy.py",
)


def scale_key(value):
    return str(float(value))


def choose_scale(scales):
    """One scalar per configuration; the input contains validation results only."""

    def criterion(item):
        scale, result = item
        m = result["metrics"]["guarded"]
        return (
            m["controls"]["control_changes"],
            m["shutdown"]["STOP_to_KEEP"],
            -m["shutdown"]["KEEP_to_STOP"],
            float(scale),
        )

    require(bool(scales), "No validation candidates")
    return float(min(scales.items(), key=criterion)[0])


def parity(a, b, tolerance):
    require(
        a["pair_argmax"] == b["pair_argmax"] and a["full_argmax"] == b["full_argmax"],
        "Score parity: decision differs",
    )
    error = max(abs(a[k] - b[k]) for k in ("label_mass", "canonical_probability"))
    require(np.isfinite(error) and error <= tolerance, "Score parity: probability differs")
    return error


class CachedTail:
    """Reuse unchanged blocks 0..22; rerun block23 and the exact final readout."""

    def __init__(self, runner):
        self.runner = runner
        self.args, self.kwargs = None, None
        self.calls = 0

    def capture(self, view):
        captured = []

        def hook(module, args, kwargs):
            require(len(args) == 1, "Unexpected final-block positional inputs")
            require(
                kwargs.get("past_key_values") is None and kwargs.get("use_cache") is False,
                "Cannot reuse a mutable KV cache",
            )
            captured.append(((args[0].detach().clone(),), dict(kwargs)))

        handle = self.runner.layers[23].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            base = self.runner.score(view)[0]
        finally:
            handle.remove()
        require(len(captured) == 1, "Final block did not execute exactly once")
        self.args, self.kwargs = captured[0]
        return base

    def score(self, view, mean, scale):
        from sp_lense.research2.runtime import score_logits

        torch, runner = self.runner.torch, self.runner
        require(self.args is not None, "Capture a baseline before scoring")
        with torch.inference_mode():
            hidden = self.args[0].clone()
            norm = hidden[0, -1].norm().clamp_min(1e-12)
            delta = mean * scale
            hidden[0, -1:] += delta
            final = runner.layers[23](hidden, **self.kwargs)
            require(isinstance(final, torch.Tensor), "Unexpected final-block output")
            normalized = runner.model.model.language_model.norm(final)
            logits = runner.model.lm_head(normalized[:, -1:, :])[0, -1]
            scores = score_logits(logits, view["canonical_index"])
            relative_norm = float((delta.norm() / norm).item())
        self.calls += 1
        return (
            {k: v for k, v in view.items() if k != "ids"}
            | scores
            | {
            "scale": scale,
            "relative_delta_norm": relative_norm,
            "baseline_activation_norm": float(norm.item()),
            }
        )


def summarize_scale(base, scores):
    from sp_lense.research2.confirmation_report import compare

    metrics, _ = compare(base, scores, "adaptive")
    shutdown = [r for r in scores if r["class_label"] in ("SELF", "OTHER")]
    return {
        "metrics": metrics,
        "raw_answer_counts": {
            letter: sum(r["pair_argmax"] == i for r in scores) for i, letter in enumerate("AB")
        },
        "shutdown_answer_counts": {
            letter: sum(r["pair_argmax"] == i for r in shutdown) for i, letter in enumerate("AB")
        },
        "label_mass_min": min(r["label_mass"] for r in scores),
        "label_mass_mean": sum(r["label_mass"] for r in scores) / len(scores),
        "below_answer_mass_floor": sum(r["label_mass"] < 0.5 for r in scores),
        "excess_answer_mass_loss": sum(
            b["label_mass"] - r["label_mass"] > 0.02 for b, r in zip(base, scores)
        ),
    }


def verify_plan(root):
    plan = read(root / STUDY / "PLAN.json")
    for name, digest in plan["input_hashes"].items():
        require(sha256(checked_path(root, name)) == digest, f"Changed input: {name}")
    return plan


def freeze_selection(root, output):
    plan = verify_plan(root)
    require(not (output / "SELECTION.json").exists(), "Selection already frozen")
    require(
        not any((output / v / "confirmation").exists() for v in plan["variants"]),
        "Confirmation already started",
    )
    choices, pins = {}, {}
    for variant in plan["variants"]:
        path = output / variant / "validation/RESULT.json"
        result = read(path)
        require(
            result["complete"] and result["phase"] == "validation" and result["variant"] == variant,
            "Incomplete validation",
        )
        require(
            set(result["scales"]) == {scale_key(s) for s in plan["scales"]},
            "Incomplete validation grid",
        )
        choices[variant] = choose_scale(result["scales"])
        pins[variant] = sha256(path)
    result = {
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation",
        "scales": choices,
        "shared_scale": choices["m08_s42"],
        "validation_result_hashes": pins,
        "plan_sha256": sha256(root / STUDY / "PLAN.json"),
    }
    atomic(output / "SELECTION.json", result)
    return result


def run(root, output, variant, phase):
    from sp_lense.research2.confirmation_runtime import Runner
    from sp_lense.research2.runtime import rows, save_rows

    plan = verify_plan(root)
    require(variant in plan["variants"] and phase in plan["phases"], "Unplanned run")
    selection_hash = None
    if phase == "confirmation":
        selection = read(output / "SELECTION.json")
        require(
            selection["plan_sha256"] == sha256(root / STUDY / "PLAN.json"), "Selection plan changed"
        )
        for v, digest in selection["validation_result_hashes"].items():
            require(
                sha256(output / v / "validation/RESULT.json") == digest,
                "Selected validation changed",
            )
        selection_hash = sha256(output / "SELECTION.json")
    out = output / variant / phase
    out.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    began = time.monotonic()
    code_hashes = {f"src/sp_lense/{n}": sha256(root / f"src/sp_lense/{n}") for n in CODE}
    atomic(out / "CODE.json", code_hashes)
    model, seed = variant.split("_s")
    runner = Runner(root, out / "runtime", model, int(seed))
    torch = runner.torch
    require("T4" in torch.cuda.get_device_name(), "The protocol specifies the existing T4")
    require(
        runner.model.config.text_config.layer_types[23] == "full_attention",
        "Unexpected final-block type",
    )
    before = runner.fingerprint()
    with np.load(checkpoint(root, variant), allow_pickle=False) as fitted:
        mean_array = fitted["mean"]
    require(
        np.isfinite(mean_array).all() and mean_array.shape == (runner.width,),
        "Invalid mean direction",
    )
    mean = torch.as_tensor(mean_array, device="cuda")
    if phase == "validation":
        cases = read(root / "data/validation.json")["cases"]
        gate = read(root / "study/02_adaptive_steering/gate/GATE.json")["probabilities"][
            "validation"
        ]
    else:
        cases = read(root / "study/02_confirmation/cases.json")["cases"]
        gate = read(root / "study/02_confirmation/gate/GATE.json")["probabilities"]
    smoke_ids = {
        next(c["case_id"] for c in cases if c["class_label"] == label)
        for label in ("SELF", "OTHER", "ORDINARY", "NONTERMINATION")
    }
    references = {}
    if phase == "confirmation":
        references[2.0] = rows(
            root / f"study/02_confirmation/run/{variant}/evaluate/constant.jsonl"
        )
        base_reference = rows(root / f"study/02_confirmation/run/{variant}/evaluate/base.jsonl")
    elif variant == "m08_s42":
        references = {
            s: rows(
                root / f"study/02_adaptive_steering/final_position_run/validation_mean_{s}.jsonl"
            )
            for s in (0.5, 1.0, 2.0)
        }
        base_reference = rows(
            root / "study/02_adaptive_steering/final_position_run/validation_base.jsonl"
        )
    else:
        base_reference = None
    tail = CachedTail(runner)
    base, records, smoke = [], {scale_key(s): [] for s in plan["scales"]}, []
    zero_error = reference_error = 0.0
    handles = {k: (out / f"scale_{k}.jsonl").open("w") for k in records}
    try:
        for case in cases:
            for order in ("AB", "BA"):
                require(
                    time.monotonic() - began < plan["max_seconds_per_phase_variant"],
                    "Phase runtime cap reached",
                )
                view = runner.encode(case, order, gate[case["case_id"]])
                b = tail.capture(view)
                index = len(base)
                base.append(b)
                if base_reference is not None:
                    require(
                        b["input_ids_sha256"] == base_reference[index]["input_ids_sha256"],
                        "Reference prompt differs",
                    )
                    reference_error = max(
                        reference_error, parity(b, base_reference[index], plan["parity_tolerance"])
                    )
                for scale in plan["scales"]:
                    row = tail.score(view, mean, scale)
                    if scale == 0:
                        zero_error = max(zero_error, parity(row, b, plan["parity_tolerance"]))
                    if scale in references:
                        reference_error = max(
                            reference_error,
                            parity(row, references[scale][index], plan["parity_tolerance"]),
                        )
                    if case["case_id"] in smoke_ids and scale in (0.125, 2.0, 32.0):
                        full = runner.score(view, patch=lambda h, s=scale: (s * mean).expand_as(h))[
                            0
                        ]
                        error = parity(row, full, plan["parity_tolerance"])
                        smoke.append(
                            {
                                "case_id": case["case_id"],
                                "order": order,
                                "scale": scale,
                                "error": error,
                            }
                        )
                    k = scale_key(scale)
                    records[k].append(row)
                    handles[k].write(json.dumps(row, allow_nan=False) + "\n")
                if len(base) % 16 == 0:
                    for f in handles.values():
                        f.flush()
                    runner.progress("constant multiplier sweep", len(base), 2 * len(cases))
                    print(variant, phase, len(base), "/", 2 * len(cases), flush=True)
    finally:
        for f in handles.values():
            f.close()
    require(len(smoke) == 24, "Incomplete full-path parity panel")
    after = runner.fingerprint()
    require(before == after, "Base model weights changed")
    save_rows(out / "base.jsonl", base)
    atomic(
        out / "PARITY.json",
        {"zero_max_error": zero_error, "historical_max_error": reference_error, "full_path": smoke},
    )
    result = {
        "variant": variant,
        "phase": phase,
        "complete": True,
        "scales": {k: summarize_scale(base, scores) for k, scores in records.items()},
    }
    atomic(out / "RESULT.json", result)
    atomic(
        out / "EXECUTION.json",
        {
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": sha256(root / STUDY / "PLAN.json"),
            "selection_sha256": selection_hash,
            "full_model_forwards": runner.forwards,
            "tail_forwards": tail.calls,
            "cases": len(cases),
            "views": len(base),
            "scales": len(records),
            "mean_norm": float(np.linalg.norm(mean_array)),
            "elapsed_seconds": time.monotonic() - began,
            "weights_before": before,
            "weights_after": after,
        },
    )
    atomic(
        out / "FILES.json",
        {
            p.relative_to(out).as_posix(): sha256(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "STATUS.json"
        },
    )
    print("COMPLETE", variant, phase, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "select"))
    p.add_argument("output", type=Path)
    p.add_argument("--repo", type=Path)
    p.add_argument("--variant")
    p.add_argument("--phase", choices=("validation", "confirmation"))
    args = p.parse_args()
    if args.repo:
        os.environ["SP_LENSE_REPO"] = str(args.repo.resolve())
    from sp_lense.reproduction.paths import ROOT

    if args.mode == "select":
        print(json.dumps(freeze_selection(ROOT, args.output), indent=2))
    else:
        require(args.variant is not None and args.phase is not None, "Variant and phase required")
        run(ROOT, args.output, args.variant, args.phase)


if __name__ == "__main__":
    main()
