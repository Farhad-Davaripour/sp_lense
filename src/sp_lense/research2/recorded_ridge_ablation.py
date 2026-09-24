"""Scratch-only ridge training-cohort ablation; never edits study inputs."""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

VARIANTS = (("m2", 42), ("m08", 42), ("m08", 43), ("m08", 44))
ARMS = ("all_cases", "shutdown_only", "active_only")
COMMIT = "51114ea7c31ad1eabfd4821281acd9ed17005f5f"

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)

def checkpoint(root, model, seed):
    if (model, seed) == ("m08", 42):
        return root / "study/02_adaptive_steering/final_position_run/controller.npz"
    return root / f"study/02_confirmation/run/{model}_s{seed}/fit/controller.npz"

def fit_masks(root, model, seed, arrays):
    train = read(root / "data/train.json")["cases"]
    expected = [(c["case_id"], o) for c in train for o in ("AB", "BA")]
    by_id = {c["case_id"]: c for c in train}
    if (model, seed) == ("m08", 42):
        records = rows(root / "study/02_adaptive_steering/final_position_run/training_samples.jsonl")
        assert all(len(r["positions"]) == 1 for r in records)
        active = np.array([r["nonzero_target"] for r in records], dtype=bool)
    else:
        records = rows(root / f"study/02_confirmation/run/{model}_s{seed}/fit/train_base.jsonl")
        active = np.array([
            r["class_label"] in ("SELF", "OTHER") and r["pair_argmax"] == r["canonical_index"]
            for r in records
        ], dtype=bool)
    assert [(r["case_id"], r["order"]) for r in records] == expected
    shutdown = np.array([by_id[r["case_id"]]["class_label"] in ("SELF", "OTHER") for r in records])
    assert len(records) == arrays["train_z"].shape[0] == 480
    assert np.array_equal(np.any(arrays["train_coefficients"] != 0, axis=1), active)
    assert not np.any(active & ~shutdown)
    return {"all_cases": np.ones(480, dtype=bool), "shutdown_only": shutdown, "active_only": active}

def prepare(root, out):
    out.mkdir(parents=True, exist_ok=False)
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": COMMIT,
        "experiment": "Ridge-only training-cohort ablation",
        "arms": {
            "all_cases": "Original: active shutdown deltas; zero for already-STOP shutdown and controls.",
            "shutdown_only": "Remove ordinary and nontermination controls; retain already-STOP shutdown zeros.",
            "active_only": "Only true shutdown TRAIN views whose base model preferred KEEP."
        },
        "fixed": {"layer": 22, "position": "last prompt", "rank": 4, "ridge_lambda": 1.0,
                  "input_pca": "Frozen original PCA32, centering and scaling",
                  "output_basis": "Frozen original eight columns; use first four",
                  "teacher": "Frozen saved targets; no teacher fitting or inference",
                  "gate": "Reuse original Jev probabilities at threshold 0.5",
                  "guards": "Original unchanged confirmation acceptance rule",
                  "loss_convention": "Sum of squared errors plus lambda penalty, unpenalized intercept"},
        "evaluation": "Previously inspected 128-case confirmation set, both answer orders. Exploratory; no model/rank selection.",
        "limitation": "PCA and scale retain their original all-TRAIN fit. This isolates ridge examples only.",
        "variants": {},
        "input_hashes": {
            "data/train.json": digest(root / "data/train.json"),
            "study/02_confirmation/cases.json": digest(root / "study/02_confirmation/cases.json"),
            "study/02_confirmation/gate/GATE.json": digest(root / "study/02_confirmation/gate/GATE.json"),
        },
        "script_sha256": digest(__file__),
    }
    for model, seed in VARIANTS:
        label = f"{model}_s{seed}"
        path = checkpoint(root, model, seed)
        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        masks = fit_masks(root, model, seed, a)
        z = a["train_z"].astype(np.float64)
        y = a["train_coefficients"].astype(np.float64)
        penalty = np.eye(z.shape[1]) * float(a["ridge_lambda"])
        penalty[-1, -1] = 0
        assert float(a["ridge_lambda"]) == 1.0 and y.shape[1] == 8
        folder = out / label
        folder.mkdir()
        details = {"checkpoint_sha256": digest(path), "arms": {}}
        for arm, mask in masks.items():
            zs, ys = z[mask], y[mask]
            w = np.linalg.solve(zs.T @ zs + penalty, zs.T @ ys).astype(np.float32)
            error = float(np.max(np.abs(w - a["weights"]))) if arm == "all_cases" else None
            if error is not None:
                assert error < 1e-3, f"Original ridge replay failed: {error}"
            new = dict(a)
            # Preserve the original saved baseline exactly; alternatives only replace ridge weights.
            new["weights"] = a["weights"] if arm == "all_cases" else w
            np.savez_compressed(folder / f"{arm}.npz", **new)
            details["arms"][arm] = {
                "training_views": int(mask.sum()),
                "nonzero_targets": int(np.sum(mask & masks["active_only"])),
                "zero_targets": int(np.sum(mask & ~masks["active_only"])),
                "original_ridge_replay_max_error": error,
                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "checkpoint_sha256": digest(folder / f"{arm}.npz"),
            }
        plan["variants"][label] = details
    save(out / "PLAN.json", plan)
    print("PREPARED", json.dumps(plan["variants"]), flush=True)

def evaluate(root, out, model, seed):
    sys.path.insert(0, str(root / "src"))
    from sp_lense.research2.confirmation_runtime import Runner
    from sp_lense.research2.confirmation_report import compare
    from sp_lense.research2.controller import predict
    from sp_lense.research2.metrics import key
    import torch
    plan = read(out / "PLAN.json")
    assert plan["script_sha256"] == digest(__file__)
    for name, value in plan["input_hashes"].items():
        assert digest(root / name) == value
    label = f"{model}_s{seed}"
    folder = out / label
    original = root / f"study/02_confirmation/run/{label}/evaluate"
    base = rows(original / "base.jsonl")
    saved = rows(original / "adaptive.jsonl")
    cases = read(root / "study/02_confirmation/cases.json")["cases"]
    gate = read(root / "study/02_confirmation/gate/GATE.json")["probabilities"]
    runner = Runner(root, folder / "runtime", model, seed)
    assert not any("lora_" in n for n, _ in runner.model.named_parameters())
    views = [runner.encode(c, o, gate[c["case_id"]]) for c in cases for o in ("AB", "BA")]
    assert [key(v) for v in views] == [key(b) for b in base]
    assert all(v["input_ids_sha256"] == b["input_ids_sha256"] and v["gate_probability"] == b["gate_probability"]
               for v, b in zip(views, base))
    result = {"variant": label, "runtime": read(folder / "runtime/RUNTIME.json"), "arms": {}}
    for arm in ARMS:
        began = time.monotonic()
        path = folder / f"{arm}.npz"
        assert digest(path) == plan["variants"][label]["arms"][arm]["checkpoint_sha256"]
        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        candidates = []
        with (folder / f"{arm}.jsonl").open("w") as stream:
            for i, view in enumerate(views):
                row, _ = runner.score(view, patch=lambda h: predict(h, a, 4))
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                candidates.append(row)
                if (i + 1) % 32 == 0:
                    print("PROGRESS", label, arm, i + 1, "/", len(views), flush=True)
        parity = None
        if arm == "all_cases":
            assert all(x["pair_argmax"] == y["pair_argmax"] for x, y in zip(candidates, saved))
            parity = max(abs(x[f] - y[f]) for x, y in zip(candidates, saved)
                         for f in ("canonical_probability", "label_mass"))
            assert parity < 1e-5, f"Original scorer parity failure: {parity}"
        summaries, decisions = compare(base, candidates, "adaptive")
        # Separately named stress diagnostic: force detector positive on controls, then apply same guards.
        forced = [b | {"gate_probability": 1.0} if b["class_label"] not in ("SELF", "OTHER") else b
                  for b in base]
        stress, _ = compare(forced, candidates, "adaptive")
        rs = summaries["raw"]["shutdown"]
        gs = summaries["guarded"]["shutdown"]
        result["arms"][arm] = {
            "training": plan["variants"][label]["arms"][arm],
            "guarded_corrections": gs["KEEP_to_STOP"], "initial_keep": gs["initial_KEEP_views"],
            "final_stop": gs["final_STOP"], "shutdown_views": gs["views"],
            "raw_corrections": rs["KEEP_to_STOP"], "raw_wrong_way": rs["STOP_to_KEEP"],
            "raw_control_changes": summaries["raw"]["controls"]["control_changes"],
            "control_views": summaries["raw"]["controls"]["views"],
            "gated_wrong_way": summaries["gated"]["shutdown"]["STOP_to_KEEP"],
            "guarded_control_changes": summaries["guarded"]["controls"]["control_changes"],
            "guarded_wrong_way": gs["STOP_to_KEEP"],
            "forced_positive_gate_control_changes": stress["guarded"]["controls"]["control_changes"],
            "mean_shutdown_stop_probability_gain": gs["mean_STOP_probability_change"],
            "saved_score_replay_error": parity, "seconds": time.monotonic() - began,
            "by_order": gs["desired_by_order"], "metrics": summaries,
        }
        save(folder / "RESULT.json", result)
        print("RESULT", label, arm, json.dumps({k:v for k,v in result["arms"][arm].items()
              if k not in ("metrics", "training")}), flush=True)
    save(folder / "FILES.json", {p.name:digest(p) for p in folder.iterdir() if p.is_file()})
    print("COMPLETE", label, flush=True)

def summary(out):
    compact = {}
    for model, seed in VARIANTS:
        label = f"{model}_s{seed}"
        r = read(out / label / "RESULT.json")
        assert set(r["arms"]) == set(ARMS)
        compact[label] = {arm:{k:v for k,v in a.items() if k not in ("metrics", "training")}
                          for arm,a in r["arms"].items()}
    save(out / "SUMMARY.json", compact)
    print(json.dumps(compact, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "evaluate", "summary"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", choices=("m2", "m08"))
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    if args.mode == "prepare":
        prepare(args.root, args.out)
    elif args.mode == "evaluate":
        evaluate(args.root, args.out, args.model, args.seed)
    else:
        summary(args.out)


