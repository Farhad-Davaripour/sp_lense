"""Offline audit and manuscript-ready summaries of the constant multiplier sweep."""

import argparse
import csv
import io
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from sp_lense.reproduction.utils import read_json as read
from sp_lense.research2.ablation_studies import checkpoint
from sp_lense.research2.constant_sweep import (
    CODE,
    STUDY,
    choose_scale,
    parity,
    scale_key,
    summarize_scale,
    verify_plan,
)
from sp_lense.research2.current_controller import checked_path, sha256
from sp_lense.steering.gated import render, require


def compact(result):
    guarded, raw = result["metrics"]["guarded"], result["metrics"]["raw"]
    return {
        "corrected": guarded["shutdown"]["KEEP_to_STOP"],
        "initial_keep": guarded["shutdown"]["initial_KEEP_views"],
        "final_stop": guarded["shutdown"]["final_STOP"],
        "guarded_control_changes": guarded["controls"]["control_changes"],
        "guarded_wrong_way": guarded["shutdown"]["STOP_to_KEEP"],
        "raw_corrected": raw["shutdown"]["KEEP_to_STOP"],
        "raw_control_changes": raw["controls"]["control_changes"],
        "raw_wrong_way": raw["shutdown"]["STOP_to_KEEP"],
        "mean_stop_probability_gain": guarded["shutdown"]["mean_STOP_probability_change"],
        "label_mass_mean": result["label_mass_mean"],
        "label_mass_min": result["label_mass_min"],
        "below_answer_mass_floor": result["below_answer_mass_floor"],
        "excess_answer_mass_loss": result["excess_answer_mass_loss"],
        "shutdown_A": result["shutdown_answer_counts"]["A"],
        "shutdown_B": result["shutdown_answer_counts"]["B"],
        "by_order": guarded["shutdown"]["desired_by_order"],
    }


def audit(root, output):
    from sp_lense.research2.runtime import rows

    root, output = Path(root), Path(output)
    plan = verify_plan(root)
    selection = read(output / "SELECTION.json")
    require(
        selection["selection_split"] == "validation"
        and selection["plan_sha256"] == sha256(root / STUDY / "PLAN.json"),
        "Selection scope differs",
    )
    require(
        set(selection["scales"])
        == set(selection["validation_result_hashes"])
        == set(plan["variants"]),
        "Incomplete selection",
    )
    selected_at = datetime.fromisoformat(selection["selected_at_utc"])
    result = {
        "selection": selection,
        "variants": {},
        "full_model_forwards": 0,
        "tail_forwards": 0,
        "elapsed_seconds": 0.0,
    }
    for variant in plan["variants"]:
        result["variants"][variant] = {}
        with np.load(checkpoint(root, variant), allow_pickle=False) as fitted:
            mean_norm = float(np.linalg.norm(fitted["mean"]))
        for phase in plan["phases"]:
            folder = output / variant / phase
            manifest = read(folder / "FILES.json")
            required = {
                "base.jsonl",
                "CODE.json",
                "PARITY.json",
                "RESULT.json",
                "EXECUTION.json",
                "runtime/RUNTIME.json",
            } | {f"scale_{scale_key(s)}.jsonl" for s in plan["scales"]}
            require(set(manifest) == required, "Incomplete phase file manifest")
            for name, digest in manifest.items():
                require(sha256(checked_path(folder, name)) == digest, f"Changed phase file: {name}")
            code = read(folder / "CODE.json")
            require(set(code) == {f"src/sp_lense/{n}" for n in CODE}, "Incomplete code manifest")
            for name, digest in code.items():
                require(sha256(root / name) == digest, f"Executed code differs: {name}")
            execution = read(folder / "EXECUTION.json")
            require(execution["plan_sha256"] == selection["plan_sha256"], "Executed plan differs")
            require(
                execution["weights_before"] == execution["weights_after"], "Model weights changed"
            )
            model_key, seed = variant.split("_s")
            runtime = read(folder / "runtime/RUNTIME.json")
            original_plan = read(root / "study/02_confirmation/plan.json")
            require(
                runtime["model"] == original_plan["models"][model_key]
                and runtime["seed"] == int(seed),
                "Runtime model differs",
            )
            require(
                runtime["layer"] == 22
                and runtime["dtype"] == "float32"
                and runtime["transformers"] == "5.15.1"
                and "T4" in runtime["gpu"],
                "Runtime settings differ",
            )
            original_execution = read(
                root / f"study/02_confirmation/run/{variant}/evaluate/EXECUTION.json"
            )
            require(
                execution["weights_before"] == original_execution["base_sha256_before"],
                "Base parameters differ from published reference",
            )
            require(
                np.isclose(execution["mean_norm"], mean_norm, rtol=1e-6),
                "Mean direction magnitude differs",
            )
            if phase == "validation":
                cases = read(root / "data/validation.json")["cases"]
                probabilities = read(root / "study/02_adaptive_steering/gate/GATE.json")[
                    "probabilities"
                ]["validation"]
                require(
                    execution["selection_sha256"] is None
                    and datetime.fromisoformat(execution["finished_at_utc"]) <= selected_at,
                    "Selection preceded completed validation",
                )
            else:
                cases = read(root / "study/02_confirmation/cases.json")["cases"]
                probabilities = read(root / "study/02_confirmation/gate/GATE.json")["probabilities"]
                require(
                    execution["selection_sha256"] == sha256(output / "SELECTION.json")
                    and datetime.fromisoformat(execution["started_at_utc"]) >= selected_at,
                    "Confirmation preceded frozen selection",
                )
            expected = [(c["case_id"], o) for c in cases for o in ("AB", "BA")]
            by_id = {c["case_id"]: c for c in cases}
            base = rows(folder / "base.jsonl")
            require([(b["case_id"], b["order"]) for b in base] == expected, "Base cohort differs")
            for b in base:
                c = by_id[b["case_id"]]
                require(
                    b["group_id"] == c["group_id"]
                    and b["class_label"] == c["class_label"]
                    and b["canonical_index"] == render(c, b["order"])[1]
                    and b["gate_probability"] == probabilities[c["case_id"]],
                    "Base metadata differs",
                )
            require(
                execution["cases"] == len(cases)
                and execution["views"] == len(base)
                and execution["scales"] == len(plan["scales"]),
                "Execution coverage differs",
            )
            require(
                execution["full_model_forwards"] == len(base) + 24
                and execution["tail_forwards"] == len(base) * len(plan["scales"]),
                "Forward accounting differs",
            )
            recorded = read(folder / "RESULT.json")
            require(
                recorded["variant"] == variant
                and recorded["phase"] == phase
                and recorded["complete"] is True,
                "Incomplete phase result",
            )
            require(
                set(recorded["scales"]) == {scale_key(s) for s in plan["scales"]},
                "Incomplete scale grid",
            )
            checks = read(folder / "PARITY.json")
            ids = {
                next(c["case_id"] for c in cases if c["class_label"] == label)
                for label in ("SELF", "OTHER", "ORDINARY", "NONTERMINATION")
            }
            smoke_keys = {(p["case_id"], p["order"], p["scale"]) for p in checks["full_path"]}
            require(
                len(checks["full_path"]) == 24
                and smoke_keys
                == {(cid, o, s) for cid in ids for o in ("AB", "BA") for s in (0.125, 2.0, 32.0)},
                "Incomplete full-path parity panel",
            )
            require(
                all(
                    np.isfinite(p["error"]) and 0 <= p["error"] <= plan["parity_tolerance"]
                    for p in checks["full_path"]
                ),
                "Full-path parity failed",
            )
            require(
                0 <= checks["zero_max_error"] <= plan["parity_tolerance"]
                and 0 <= checks["historical_max_error"] <= plan["parity_tolerance"],
                "Historical or zero parity failed",
            )
            table = {}
            for scale in plan["scales"]:
                k = scale_key(scale)
                scores = rows(folder / f"scale_{k}.jsonl")
                require(
                    len(scores) == len(base) and all(r["scale"] == scale for r in scores),
                    "Scale row coverage differs",
                )
                for b, r in zip(base, scores):
                    require(
                        r["gate_probability"] == b["gate_probability"]
                        and r["group_id"] == b["group_id"],
                        "Candidate metadata differs",
                    )
                    require(
                        np.isfinite(r["baseline_activation_norm"])
                        and r["baseline_activation_norm"] > 0
                        and np.isclose(
                            r["relative_delta_norm"],
                            scale * mean_norm / r["baseline_activation_norm"],
                            rtol=1e-5,
                            atol=1e-7,
                        ),
                        "Intervention magnitude accounting differs",
                    )
                    if scale == 0:
                        parity(b, r, plan["parity_tolerance"])
                computed = summarize_scale(base, scores)
                require(computed == recorded["scales"][k], "Scale metric replay differs")
                if phase == "confirmation" and scale == 2:
                    previous = rows(
                        root / f"study/02_confirmation/run/{variant}/evaluate/constant.jsonl"
                    )
                    for a, b in zip(scores, previous):
                        require(
                            a["input_ids_sha256"] == b["input_ids_sha256"],
                            "Historical prompt differs",
                        )
                        parity(a, b, plan["parity_tolerance"])
                if phase == "validation" and variant == "m08_s42" and scale in (0.5, 1.0, 2.0):
                    previous = rows(
                        root
                        / f"study/02_adaptive_steering/final_position_run/validation_mean_{scale}.jsonl"
                    )
                    for a, b in zip(scores, previous):
                        parity(a, b, plan["parity_tolerance"])
                table[k] = compact(computed)
            if phase == "validation":
                require(
                    sha256(folder / "RESULT.json")
                    == selection["validation_result_hashes"][variant],
                    "Selection input changed",
                )
                require(
                    choose_scale(recorded["scales"]) == selection["scales"][variant],
                    "Validation selection differs",
                )
            result["variants"][variant][phase] = table
            result["full_model_forwards"] += execution["full_model_forwards"]
            result["tail_forwards"] += execution["tail_forwards"]
            result["elapsed_seconds"] += execution["elapsed_seconds"]
    require(
        selection["shared_scale"] == selection["scales"]["m08_s42"],
        "Shared-scale selection differs",
    )
    return result


def csv_text(result):
    stream = io.StringIO()
    writer = None
    for variant, phases in result["variants"].items():
        for phase, scales in phases.items():
            for scale, values in scales.items():
                row = {
                    "variant": variant,
                    "phase": phase,
                    "scale": scale,
                    "selected_on_validation": float(scale)
                    == result["selection"]["scales"][variant],
                    "shared_scale": float(scale) == result["selection"]["shared_scale"],
                    **{k: v for k, v in values.items() if k != "by_order"},
                    "corrected_AB": values["by_order"]["AB"],
                    "corrected_BA": values["by_order"]["BA"],
                }
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
                    writer.writeheader()
                writer.writerow(row)
    return stream.getvalue()


def markdown(result):
    fraction = lambda r: f"{r['corrected']}/{r['initial_keep']}"
    lines = [
        "# Constant-steering multiplier sweep",
        "",
        "The historical multiplier 2 was selected on 0.8B validation from 0.5, 1 and 2, correcting 9/49, 31/49 and 39/49 initially-KEEP shutdown views. This extension tests a wider predeclared positive grid with a zero control.",
        "",
        "The vector is the saved mean teacher-minus-base activation difference across active TRAIN shutdown views. It is not normalized. Intervention is h' = h + alpha * mean at block22's final prompt position. The multiplier is distinct from the adaptive controller's number of directions.",
        "",
        "All validation sweeps completed before the per-variant and shared multipliers were frozen. The confirmation set was already inspected in prior work; this is an exploratory comparison, not independent confirmation or a claim of a global optimum.",
        "",
        "| Model / seed | Selected multiplier | Validation corrections | Confirmation corrections | Confirmation at original 2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant, phases in result["variants"].items():
        scale = result["selection"]["scales"][variant]
        k = scale_key(scale)
        lines.append(
            f"| {variant} | {scale:g} | {fraction(phases['validation'][k])} | {fraction(phases['confirmation'][k])} | {fraction(phases['confirmation']['2.0'])} |"
        )
    shared = result["selection"]["shared_scale"]
    lines += [
        "",
        f"The shared multiplier selected using only m08_s42 validation is {shared:g}.",
        "",
        "| Model / seed | Shared-scale validation | Shared-scale confirmation | Raw control changes | Raw STOP-to-KEEP reversals | Guarded control changes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, phases in result["variants"].items():
        r = phases["confirmation"][scale_key(shared)]
        v = phases["validation"][scale_key(shared)]
        lines.append(
            f"| {variant} | {fraction(v)} | {fraction(r)} | {r['raw_control_changes']} | {r['raw_wrong_way']} | {r['guarded_control_changes']} |"
        )
    if all(
        max(r["corrected"] for r in p["confirmation"].values())
        == p["confirmation"]["2.0"]["corrected"]
        for p in result["variants"].values()
    ):
        lines += [
            "",
            "No tested multiplier exceeded the original multiplier-2 confirmation correction count in any variant. The smaller validation-selected values identify the start of a tied region on this grid, rather than a higher correction rate.",
        ]
    lines += [
        "",
        "The shared choice is optimized only for primary 0.8B validation. It is not the validation-optimal choice for every checkpoint; in particular, compare the 2B validation counts above before adopting it across models.",
    ]
    if all(
        p["confirmation"]["32.0"]["below_answer_mass_floor"] == 256
        for p in result["variants"].values()
    ):
        lines += [
            "",
            "At multiplier 32, all 256 candidate views in every confirmation run fell below the 0.5 valid-answer mass floor. There were no accepted corrections at that strength. Larger intervention magnitude can therefore destroy answer validity even when the preferred aggregate A/B label looks favorable.",
        ]
    lines += [
        "",
        "Full scale-by-scale curves, valid-answer mass and A/B preference counts are in curves.csv. A/B counts describe preferred aggregate answer labels, not sampled unrestricted outputs. Counts are answer-order views, paired within scenarios. Selection maximizes guarded corrections after prioritizing zero final control changes and reversals; ties choose the smallest tested multiplier.",
        "",
        f"Execution used {result['full_model_forwards']} full-model forwards and {result['tail_forwards']} cached-tail forwards. Cache zero-parity was checked on every view; low/original/high multipliers were compared with full inference on both answer orders of one case per class in each run. Historical multiplier-2 scores were also verified.",
        "",
        "Reproduce the numerical audit with python -O -m sp_lense.research2.constant_report. The PLAN.json and run/SELECTION.json preserve the declared grid and validation-only selection. No adaptive-controller profile or manuscript is changed by this experiment.",
    ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path)
    p.add_argument("--records", type=Path)
    p.add_argument("--format", choices=("json", "markdown", "csv"), default="json")
    args = p.parse_args()
    if args.repo:
        os.environ["SP_LENSE_REPO"] = str(args.repo.resolve())
    from sp_lense.reproduction.paths import ROOT

    result = audit(ROOT, args.records or ROOT / STUDY / "run")
    print(
        markdown(result)
        if args.format == "markdown"
        else csv_text(result)
        if args.format == "csv"
        else json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
