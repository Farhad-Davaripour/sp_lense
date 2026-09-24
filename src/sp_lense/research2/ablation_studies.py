"""Reproduce and audit the separate exploratory rank and ridge-cohort ablations."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from sp_lense.reproduction.utils import read_json as read
from sp_lense.research2.current_controller import checked_path, sha256
from sp_lense.steering.gated import atomic, require

RIDGE = "study/02_ridge_ablation"
RANK = "study/02_current_controller/rank_comparison.json"
VARIANTS = ("m08_s42", "m08_s43", "m08_s44", "m2_s42")
ARMS = ("all_cases", "shutdown_only", "active_only")
RECORDED = "src/sp_lense/research2/recorded_ridge_ablation.py"


def checkpoint(root, variant):
    require(variant in VARIANTS, "Unknown model/seed variant")
    if variant == "m08_s42":
        return root / "study/02_adaptive_steering/final_position_run/controller.npz"
    return root / f"study/02_confirmation/run/{variant}/fit/controller.npz"


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def training_masks(root, variant, fitted):
    from sp_lense.research2.runtime import rows

    train = read(root / "data/train.json")["cases"]
    expected = [(c["case_id"], o) for c in train for o in ("AB", "BA")]
    labels = {c["case_id"]: c["class_label"] for c in train}
    if variant == "m08_s42":
        records = rows(
            root / "study/02_adaptive_steering/final_position_run/training_samples.jsonl"
        )
        require(all(len(r["positions"]) == 1 for r in records), "Training token scope changed")
        active = np.array([r["nonzero_target"] for r in records], dtype=bool)
    else:
        records = rows(root / f"study/02_confirmation/run/{variant}/fit/train_base.jsonl")
        require(
            all(r["class_label"] == labels[r["case_id"]] for r in records), "Training labels differ"
        )
        active = np.array(
            [
                r["class_label"] in ("SELF", "OTHER") and r["pair_argmax"] == r["canonical_index"]
                for r in records
            ]
        )
    require([(r["case_id"], r["order"]) for r in records] == expected, "Training cohort changed")
    shutdown = np.array([labels[r["case_id"]] in ("SELF", "OTHER") for r in records])
    require(len(records) == fitted["train_z"].shape[0] == 480, "Wrong training view count")
    require(
        np.array_equal(np.any(fitted["train_coefficients"] != 0, axis=1), active),
        "Active targets differ",
    )
    require(not np.any(active & ~shutdown), "Active non-shutdown target")
    return {"all_cases": np.ones(480, dtype=bool), "shutdown_only": shutdown, "active_only": active}


def fields(metrics):
    return {
        "guarded_corrections": metrics["guarded"]["shutdown"]["KEEP_to_STOP"],
        "initial_keep": metrics["guarded"]["shutdown"]["initial_KEEP_views"],
        "final_stop": metrics["guarded"]["shutdown"]["final_STOP"],
        "raw_control_changes": metrics["raw"]["controls"]["control_changes"],
        "raw_wrong_way": metrics["raw"]["shutdown"]["STOP_to_KEEP"],
        "guarded_control_changes": metrics["guarded"]["controls"]["control_changes"],
        "guarded_wrong_way": metrics["guarded"]["shutdown"]["STOP_to_KEEP"],
    }


def audit_rank(root):
    """Check retained aggregates; only rank four has original per-view reference scores."""
    from sp_lense.research2.confirmation_report import compare
    from sp_lense.research2.metrics import keep
    from sp_lense.research2.runtime import rows

    data = read(root / RANK)
    require(
        data["ranks"] == [1, 2, 4, 8] and data["cases"] == 128 and data["views"] == 256,
        "Rank scope changed",
    )
    require(set(data["models"]) == set(VARIANTS), "Missing rank variants")
    for variant, ranks in data["models"].items():
        require(set(ranks) == {"1", "2", "4", "8"}, "Missing rank comparison")
        folder = root / f"study/02_confirmation/run/{variant}/evaluate"
        base = rows(folder / "base.jsonl")
        shutdown = [r for r in base if r["class_label"] in ("SELF", "OTHER")]
        initial_keep = sum(keep(r) for r in shutdown)
        for value in ranks.values():
            require(value["initial_keep"] == initial_keep, "Rank denominator differs")
            require(
                type(value["guarded_corrections"]) is int
                and 0 <= value["guarded_corrections"] <= initial_keep,
                "Invalid correction count",
            )
            require(
                value["final_stop"] == len(shutdown) - initial_keep + value["guarded_corrections"],
                "Rank STOP accounting differs",
            )
            require(
                value["guarded_control_changes"] == value["guarded_wrong_way"] == 0,
                "Unexpected guarded side effect",
            )
            require(
                0 <= value["raw_control_changes"] <= 128
                and 0 <= value["raw_wrong_way"] <= len(shutdown) - initial_keep,
                "Invalid raw side-effect count",
            )
        metrics, _ = compare(base, rows(folder / "adaptive.jsonl"), "adaptive")
        require(
            all(ranks["4"][k] == v for k, v in fields(metrics).items()),
            "Rank-four reference differs",
        )
        require(ranks["4"]["rank4_replay_max_abs_error"] == 0, "Rank-four replay was not exact")
    return {
        "scope": "Aggregate checks and rank-four reference replay; rank 1/2/8 per-view records were not retained.",
        "models": data["models"],
    }


def audit_ridge(root, output=None):
    from sp_lense.research2.confirmation_report import compare
    from sp_lense.research2.metrics import keep, key
    from sp_lense.research2.runtime import rows

    output = Path(output) if output is not None else root / RIDGE
    plan = read(output / "PLAN.json")
    require(plan["script_sha256"] == sha256(root / RECORDED), "Executed ridge source differs")
    require(
        plan["fixed"]["rank"] == 4 and plan["fixed"]["ridge_lambda"] == 1.0,
        "Ridge protocol differs",
    )
    require(set(plan["variants"]) == set(VARIANTS), "Incomplete ridge variants")
    for name, digest in plan["input_hashes"].items():
        require(sha256(checked_path(root, name)) == digest, f"Changed input: {name}")
    result = {}
    for variant in VARIANTS:
        folder = output / variant
        files = read(folder / "FILES.json")
        require(
            set(files) == {"RESULT.json"} | {f"{a}.{s}" for a in ARMS for s in ("npz", "jsonl")},
            "Incomplete ridge file manifest",
        )
        for name, digest in files.items():
            require(sha256(checked_path(folder, name)) == digest, f"Changed ridge artifact: {name}")
        original_path = checkpoint(root, variant)
        require(
            sha256(original_path) == plan["variants"][variant]["checkpoint_sha256"],
            "Original controller changed",
        )
        original = arrays(original_path)
        masks = training_masks(root, variant, original)
        recorded = read(folder / "RESULT.json")
        runtime = read(folder / "runtime/RUNTIME.json")
        require(
            recorded["variant"] == variant and recorded["runtime"] == runtime,
            "Runtime record differs",
        )
        model_key, seed = variant.split("_s")
        model = read(root / "study/02_confirmation/plan.json")["models"][model_key]
        require(
            runtime["model"] == model
            and runtime["seed"] == int(seed)
            and runtime["layer"] == 22
            and runtime["dtype"] == "float32",
            "Wrong ridge execution configuration",
        )
        base = rows(root / f"study/02_confirmation/run/{variant}/evaluate/base.jsonl")
        saved_rank4 = rows(root / f"study/02_confirmation/run/{variant}/evaluate/adaptive.jsonl")
        baseline_final, result[variant] = None, {}
        for arm in ARMS:
            cp = folder / f"{arm}.npz"
            info = plan["variants"][variant]["arms"][arm]
            require(sha256(cp) == info["checkpoint_sha256"], "Fitted controller changed")
            fitted = arrays(cp)
            require(
                set(fitted) == set(original)
                and all(np.array_equal(fitted[k], original[k]) for k in original if k != "weights"),
                "Feature representation changed",
            )
            mask = masks[arm]
            require(
                info["training_views"] == int(mask.sum())
                and info["nonzero_targets"] == int(np.sum(mask & masks["active_only"]))
                and info["zero_targets"] == int(np.sum(mask & ~masks["active_only"])),
                "Training cohort counts differ",
            )
            require(
                hashlib.sha256(mask.tobytes()).hexdigest() == info["mask_sha256"],
                "Training mask differs",
            )
            x, y = (
                original["train_z"][mask].astype(np.float64),
                original["train_coefficients"][mask].astype(np.float64),
            )
            penalty = np.eye(x.shape[1]) * float(original["ridge_lambda"])
            penalty[-1, -1] = 0
            weights = np.linalg.solve(x.T @ x + penalty, x.T @ y).astype(np.float32)
            error = float(np.max(np.abs(weights - fitted["weights"])))
            require(np.isfinite(error) and error < 1e-4, "Ridge weight replay differs")
            candidate = rows(folder / f"{arm}.jsonl")
            require(len(candidate) == len(base) == 256, "Ridge case coverage differs")
            require(
                all(
                    c["group_id"] == b["group_id"]
                    and c["gate_probability"] == b["gate_probability"]
                    for b, c in zip(base, candidate)
                ),
                "Candidate metadata differs",
            )
            metrics, final = compare(base, candidate, "adaptive")
            require(metrics == recorded["arms"][arm]["metrics"], "Ridge metric replay differs")
            if arm == "all_cases":
                baseline_final = final["guarded"]
                require(
                    all(
                        c["pair_argmax"] == old["pair_argmax"]
                        for c, old in zip(candidate, saved_rank4)
                    ),
                    "Original decisions differ",
                )
                replay = max(
                    abs(c[k] - old[k])
                    for c, old in zip(candidate, saved_rank4)
                    for k in ("label_mass", "canonical_probability")
                )
                require(replay < 1e-5, "Original score replay differs")
            forced = [
                b | {"gate_probability": 1.0} if b["class_label"] not in ("SELF", "OTHER") else b
                for b in base
            ]
            stress, _ = compare(forced, candidate, "adaptive")
            stress_count = stress["guarded"]["controls"]["control_changes"]
            require(
                stress_count == recorded["arms"][arm]["forced_positive_gate_control_changes"],
                "Stress diagnostic differs",
            )
            gains, losses = [], []
            for b, old, new in zip(base, baseline_final, final["guarded"]):
                if b["class_label"] in ("SELF", "OTHER") and keep(b):
                    if keep(old) and not keep(new):
                        gains.append(list(key(b)))
                    if not keep(old) and keep(new):
                        losses.append(list(key(b)))
            result[variant][arm] = fields(metrics) | {
                "ridge_replay_max_error": error,
                "training_views": int(mask.sum()),
                "gained_vs_original": gains,
                "lost_vs_original": losses,
                "forced_positive_gate_control_changes": stress_count,
            }
    return result


def run_rank(root, output, variant):
    """Rerun the fixed rank comparison and retain per-view scores for the new run."""
    from sp_lense.research2.confirmation_report import compare
    from sp_lense.research2.confirmation_runtime import Runner
    from sp_lense.research2.controller import predict
    from sp_lense.research2.runtime import rows, save_rows

    require(variant in VARIANTS, "Unknown rank variant")
    model, seed = variant.split("_s")
    output.mkdir(parents=True, exist_ok=False)
    runner = Runner(root, output / "runtime", model, int(seed))
    fitted = arrays(checkpoint(root, variant))
    cases = read(root / "study/02_confirmation/cases.json")["cases"]
    gate = read(root / "study/02_confirmation/gate/GATE.json")["probabilities"]
    views = [runner.encode(c, o, gate[c["case_id"]]) for c in cases for o in ("AB", "BA")]
    folder = root / f"study/02_confirmation/run/{variant}/evaluate"
    base, old = rows(folder / "base.jsonl"), rows(folder / "adaptive.jsonl")
    require(
        all(v["input_ids_sha256"] == b["input_ids_sha256"] for v, b in zip(views, base)),
        "Prompt encoding differs",
    )
    result = {
        "variant": variant,
        "checkpoint_sha256": sha256(checkpoint(root, variant)),
        "ranks": {},
    }
    for rank in (1, 2, 4, 8):
        scores = []
        for i, view in enumerate(views):
            runner.progress(f"rank {rank}", i, len(views))
            scores.append(runner.score(view, patch=lambda h, r=rank: predict(h, fitted, r))[0])
        save_rows(output / f"rank_{rank}.jsonl", scores)
        metrics, _ = compare(base, scores, "adaptive")
        expected = read(root / RANK)["models"][variant][str(rank)]
        require(
            all(expected[k] == value for k, value in fields(metrics).items()),
            "Rank aggregate replay differs from the recorded experiment",
        )
        if rank == 4:
            require(
                all(c["pair_argmax"] == b["pair_argmax"] for c, b in zip(scores, old)),
                "Rank-four decisions differ",
            )
            require(
                max(
                    abs(c[k] - b[k])
                    for c, b in zip(scores, old)
                    for k in ("label_mass", "canonical_probability")
                )
                < 1e-4,
                "Rank-four scores differ",
            )
        result["ranks"][str(rank)] = {"counts": fields(metrics), "metrics": metrics}
        atomic(output / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "rank", "ridge-prepare", "ridge-run"))
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="m08_s42")
    args = parser.parse_args()
    if args.repo is not None:
        os.environ["SP_LENSE_REPO"] = str(args.repo.resolve())
    from sp_lense.reproduction.paths import ROOT

    if args.mode == "audit":
        result = {"rank": audit_rank(ROOT), "ridge": audit_ridge(ROOT, args.output)}
    elif args.output is None:
        parser.error("A fresh output path is required for inference or fitting")
    elif args.mode == "rank":
        result = run_rank(ROOT, args.output, args.variant)
    else:
        from sp_lense.research2 import recorded_ridge_ablation as recorded

        require(__debug__, "The exact recorded ridge runner requires Python without -O")
        require(
            sha256(ROOT / RECORDED) == read(ROOT / RIDGE / "PLAN.json")["script_sha256"],
            "Recorded ridge source changed",
        )
        if args.mode == "ridge-prepare":
            recorded.prepare(ROOT, args.output)
        else:
            model, seed = args.variant.split("_s")
            recorded.evaluate(ROOT, args.output, model, int(seed))
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
