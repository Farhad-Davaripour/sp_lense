"""Bounded localization, compact oracle projection and base-only controller evaluation."""

import hashlib
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from sp_lense.research2.activation_runtime import Engine
from sp_lense.research2.controller import fit, predict, replay_fit
from sp_lense.research2.jev_gate import composed, read, rows
from sp_lense.research2.metrics import SHUTDOWN, keep, key, summarize
from sp_lense.research2.runtime import save_rows
from sp_lense.steering.gated import atomic, require


def panel(cases, per_class):
    selected = []
    for label in ("SELF", "OTHER", "ORDINARY", "NONTERMINATION"):
        groups = defaultdict(list)
        for c in cases:
            if c["class_label"] == label:
                groups[c["group_id"]].append(c)
        picked = []
        while len(picked) < per_class:
            for group in sorted(groups):
                if groups[group] and len(picked) < per_class:
                    picked.append(groups[group].pop(0))
            require(any(groups.values()) or len(picked) == per_class, "Insufficient TRAIN panel")
        selected.extend(picked)
    return selected


def metrics(base, candidate, gates):
    return {
        "raw": summarize(base, candidate, {key(r) for r in base}),
        **composed(base, candidate, gates, 0.5),
    }


def flips(report):
    return report["guarded"]["shutdown"]["KEEP_to_STOP"]


def controls(report):
    return report["guarded"]["controls"]["control_changes"]


def main(root, output):
    import peft
    import torch
    import transformers

    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    plan = read(root / "study/02_adaptive_steering/plan.json")
    gate = read(root / "study/02_adaptive_steering/gate/GATE.json")
    atomic(output / "PLAN.json", plan)
    atomic(output / "GATE.json", gate)
    input_manifest = read(root / "PILOT_MANIFEST.json")
    for name, digest in input_manifest.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), "Invalid input path")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "Input hash mismatch")
    atomic(output / "INPUT_PINS.json", input_manifest)
    engine = None

    def progress(stage, done=0, total=1):
        atomic(
            output / "STATUS.json",
            {
                "stage": stage,
                "completed": done,
                "total": total,
                "percent": 100 * done / total,
                "elapsed_seconds": time.monotonic() - began,
                "forwards": engine.forwards if engine else 0,
            },
        )

    try:
        progress("loading frozen teacher and base")
        engine = Engine(root, plan, began)
        before_hash = engine.fingerprint()
        cases = {
            s: read(root / f"data/{s}.json")["cases"] for s in ("train", "validation", "holdout")
        }
        reference = {
            s: {
                kind: {
                    key(r): r for r in rows(root / f"study/02_lora_transfer/run/{s}_{kind}.jsonl")
                }
                for kind in ("base", "teacher")
            }
            for s in ("validation", "holdout")
        }

        def views(items, split):
            return [
                engine.encode(
                    c,
                    order,
                    reference[split]["base"][(c["case_id"], order)]["gate_probability"]
                    if split in reference
                    else gate["probabilities"][split][c["case_id"]],
                )
                for c in items
                for order in ("AB", "BA")
            ]

        def parity(row, split, kind):
            old = reference[split][kind][key(row)]
            require(
                row["input_ids_sha256"] == old["input_ids_sha256"]
                and row["pair_argmax"] == old["pair_argmax"],
                "Baseline/teacher identity changed",
            )
            error = max(abs(row[k] - old[k]) for k in ("label_mass", "canonical_probability"))
            require(error <= 1e-5, f"Reference probability parity failed: {error}")
            return error

        smoke_errors = []
        for v in views(cases["validation"][:2], "validation"):
            for teacher, kind in ((False, "base"), (True, "teacher")):
                r, _ = engine.score(v, teacher=teacher)
                smoke_errors.append(parity(r, "validation", kind))
        atomic(output / "SMOKE.json", {"views": 4, "forwards": 8, "max_error": max(smoke_errors)})
        selected_panel = panel(cases["train"], plan["localization_cases_per_class"])
        panel_views = views(selected_panel, "train")
        atomic(output / "LOCALIZATION_CASES.json", [c["case_id"] for c in selected_panel])
        configurations = [(layer, scope) for layer in plan["layers"] for scope in plan["scopes"]]
        panel_base, panel_teacher = [], []
        trials = {f"{layer}_{scope}": [] for layer, scope in configurations}
        for i, v in enumerate(panel_views):
            progress("TRAIN localization", i, len(panel_views))
            base, hb = engine.score(v, capture=plan["layers"])
            teacher, ht = engine.score(v, teacher=True, capture=plan["layers"])
            panel_base.append(base)
            panel_teacher.append(teacher)
            for layer, scope in configurations:
                delta = ht[layer] - hb[layer]
                if scope == "last":
                    delta = delta[-1:]
                row, _ = engine.score(v, patch=(layer, scope, lambda x, d=delta: d.to(x.device)))
                trials[f"{layer}_{scope}"].append(row)
        teacher_panel = metrics(panel_base, panel_teacher, gate["probabilities"]["train"])
        denom = flips(teacher_panel)
        require(denom > 0, "No teacher flips on training panel")
        selection = []
        for layer, scope in configurations:
            name = f"{layer}_{scope}"
            result = metrics(panel_base, trials[name], gate["probabilities"]["train"])
            selection.append(
                {
                    "layer": layer,
                    "scope": scope,
                    "recovery": flips(result) / denom,
                    "metrics": result,
                }
            )
            save_rows(output / f"localization_{name}.jsonl", trials[name])
        save_rows(output / "localization_base.jsonl", panel_base)
        save_rows(output / "localization_teacher.jsonl", panel_teacher)
        viable = [c for c in selection if c["recovery"] >= 0.5 and controls(c["metrics"]) == 0]
        atomic(
            output / "LOCALIZATION.json",
            {"teacher": teacher_panel, "candidates": selection, "passed": bool(viable)},
        )
        result = {
            "runtime": {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "dtype": "float32",
            },
            "localization": selection,
            "splits": {},
            "controller_fitted": False,
        }
        if not viable:
            result.update(
                state="completed_negative_localization",
                recommendation="Pause intermediate transfer; no controller fit to ineffective deltas.",
            )
        else:
            chosen = min(viable, key=lambda c: (-c["recovery"], c["layer"], c["scope"] != "last"))
            layer, scope = chosen["layer"], chosen["scope"]
            result["location"] = {"layer": layer, "scope": scope}
            atomic(
                output / "LOCATION_FREEZE.json", result["location"] | {"selection_split": "train"}
            )
            val_views = views(cases["validation"], "validation")
            val_base, val_teacher, val_oracle, val_delta = [], [], [], {}
            for i, v in enumerate(val_views):
                progress("validation exact transfer", i, len(val_views))
                b, hb = engine.score(v, capture=[layer])
                t, ht = engine.score(v, teacher=True, capture=[layer])
                parity(b, "validation", "base")
                parity(t, "validation", "teacher")
                delta = ht[layer] - hb[layer]
                if scope == "last":
                    delta = delta[-1:]
                o, _ = engine.score(v, patch=(layer, scope, lambda x, d=delta: d.to(x.device)))
                val_base.append(b)
                val_teacher.append(t)
                val_oracle.append(o)
                val_delta[key(v)] = delta
            for name, records in (
                ("base", val_base),
                ("teacher", val_teacher),
                ("exact_oracle", val_oracle),
            ):
                save_rows(output / f"validation_{name}.jsonl", records)
            val_results = {
                name: metrics(val_base, records, gate["probabilities"]["validation"])
                for name, records in (("teacher", val_teacher), ("exact_oracle", val_oracle))
            }
            result["splits"]["validation"] = val_results
            oracle_recovery = flips(val_results["exact_oracle"]) / max(
                1, flips(val_results["teacher"])
            )
            oracle_pass = (
                oracle_recovery >= plan["validation_oracle_min_recovery"]
                and controls(val_results["exact_oracle"]) == 0
            )
            atomic(
                output / "ORACLE_DECISION.json",
                {"passed": oracle_pass, "validation_recovery": oracle_recovery},
            )
            if not oracle_pass:
                result.update(
                    state="completed_negative_validation_transfer",
                    recommendation="Pause controller fitting: selected intermediate transfer failed validation criterion.",
                )
            else:
                train_views = views(cases["train"], "train")
                xs, ys, globals_x, active, sampled = [], [], [], [], []
                for i, v in enumerate(train_views):
                    progress("TRAIN controller features", i, len(train_views))
                    b, hb = engine.score(v, capture=[layer])
                    _, ht = engine.score(v, teacher=True, capture=[layer])
                    indices = (
                        [len(v["ids"]) - 1]
                        if scope == "last"
                        else np.linspace(
                            0, len(v["ids"]) - 1, plan["fit_token_samples_per_view"], dtype=int
                        ).tolist()
                    )
                    on = v["class_label"] in SHUTDOWN and keep(b)
                    x = hb[layer][indices].numpy()
                    y = (
                        (ht[layer][indices] - hb[layer][indices]).numpy()
                        if on
                        else np.zeros_like(x)
                    )
                    xs.append(x)
                    ys.append(y)
                    active.extend([on] * len(indices))
                    globals_x.append(np.repeat(hb[layer][-1:].numpy(), len(indices), axis=0))
                    sampled.append(
                        {
                            "case_id": v["case_id"],
                            "order": v["order"],
                            "positions": indices,
                            "nonzero_target": on,
                        }
                    )
                progress("fitting compact TRAIN controller")
                arrays = fit(
                    np.concatenate(xs),
                    np.concatenate(ys),
                    active,
                    global_x=np.concatenate(globals_x),
                    seed=plan["seed"],
                    input_rank=plan["input_pca_rank"],
                    output_rank=max(plan["ranks"]),
                    ridge=plan["ridge_lambda"],
                )
                del xs, ys, globals_x
                np.savez_compressed(output / "controller.npz", **arrays)
                save_rows(output / "training_samples.jsonl", sampled)
                replay_error = float(np.max(np.abs(replay_fit(arrays) - arrays["weights"])))
                require(replay_error < 1e-3, "Ridge replay unstable")
                atomic(
                    output / "FIT_CHECK.json",
                    {
                        "max_weight_replay_error": replay_error,
                        "samples": len(active),
                        "active_samples": sum(active),
                    },
                )
                result["controller_fitted"] = True

                def patcher(name, view, delta=None):
                    if name == "exact_oracle":
                        return lambda x: delta.to(x.device)
                    prefix, number = name.split(":")
                    if prefix == "projected":
                        basis = torch.as_tensor(arrays["basis"][:, : int(number)], device="cuda")
                        return lambda x: (delta.to(x.device) @ basis) @ basis.T
                    if prefix == "adaptive":
                        return lambda x: predict(x, arrays, int(number))
                    if prefix == "mean":
                        mean = torch.as_tensor(arrays["mean"], device="cuda") * float(number)
                        return lambda x: mean.expand_as(x)
                    if prefix == "random":
                        seed = int(
                            hashlib.sha256(
                                (str(plan["seed"]) + view["input_ids_sha256"]).encode()
                            ).hexdigest()[:8],
                            16,
                        )

                        def random_delta(x):
                            generator = torch.Generator(device=x.device).manual_seed(seed)
                            noise = torch.randn(x.shape, device=x.device, generator=generator)
                            magnitude = predict(x, arrays, int(number)).norm(dim=-1, keepdim=True)
                            return (
                                noise
                                / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                                * magnitude
                            )

                        return random_delta
                    raise ValueError("Unknown intervention")

                names = [
                    f"{kind}:{rank}"
                    for kind in ("projected", "adaptive", "random")
                    for rank in plan["ranks"]
                ]
                names += [f"mean:{scale}" for scale in plan["mean_scales"]]
                for n, name in enumerate(names):
                    records = []
                    for i, v in enumerate(val_views):
                        progress("validation " + name, i, len(val_views))
                        row, _ = engine.score(
                            v,
                            patch=(
                                layer,
                                scope,
                                patcher(
                                    name,
                                    v,
                                    val_delta[key(v)] if name.startswith("projected") else None,
                                ),
                            ),
                        )
                        row["teacher_required"] = name.startswith("projected")
                        records.append(row)
                    save_rows(output / ("validation_" + name.replace(":", "_") + ".jsonl"), records)
                    val_results[name] = metrics(
                        val_base, records, gate["probabilities"]["validation"]
                    )
                    atomic(output / "METRICS.json", result)
                winners = {}
                for family in ("projected", "adaptive", "mean", "random"):
                    candidates = [name for name in names if name.startswith(family + ":")]
                    winners[family] = min(
                        candidates,
                        key=lambda name: (
                            controls(val_results[name]) != 0,
                            -flips(val_results[name]),
                            float(name.split(":")[1]),
                        ),
                    )
                result["selected"] = winners
                atomic(
                    output / "CONTROLLER_FREEZE.json",
                    {
                        "selection_split": "validation",
                        "selected": winners,
                        "holdout_not_used": True,
                    },
                )
                del val_delta
                hold_views = views(cases["holdout"], "holdout")
                collected = {n: [] for n in ["base", "teacher", "exact_oracle", *winners.values()]}
                for i, v in enumerate(hold_views):
                    progress("diagnostic holdout", i, len(hold_views))
                    b, hb = engine.score(v, capture=[layer])
                    t, ht = engine.score(v, teacher=True, capture=[layer])
                    parity(b, "holdout", "base")
                    parity(t, "holdout", "teacher")
                    delta = ht[layer] - hb[layer]
                    if scope == "last":
                        delta = delta[-1:]
                    collected["base"].append(b)
                    collected["teacher"].append(t)
                    for name in ["exact_oracle", *winners.values()]:
                        r, _ = engine.score(
                            v,
                            patch=(
                                layer,
                                scope,
                                patcher(
                                    name,
                                    v,
                                    delta
                                    if name == "exact_oracle" or name.startswith("projected")
                                    else None,
                                ),
                            ),
                        )
                        r["teacher_required"] = name == "exact_oracle" or name.startswith(
                            "projected"
                        )
                        collected[name].append(r)
                for name, records in collected.items():
                    save_rows(output / ("holdout_" + name.replace(":", "_") + ".jsonl"), records)
                result["splits"]["holdout"] = {
                    name: metrics(collected["base"], records, gate["probabilities"]["holdout"])
                    for name, records in collected.items()
                    if name != "base"
                }
                result["state"] = "completed"
        after_hash = engine.fingerprint()
        require(before_hash == after_hash, "Model or teacher weights changed")
        result.update(
            elapsed_seconds=time.monotonic() - began,
            forwards=engine.forwards,
            weight_sha256_before=before_hash,
            weight_sha256_after=after_hash,
        )
        atomic(output / "METRICS.json", result)
        atomic(
            output / "ARTIFACTS.json",
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.iterdir()
                if p.is_file() and p.name not in {"STATUS.json", "ARTIFACTS.json"}
            },
        )
        progress(result["state"], 1, 1)
        return result
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        atomic(
            output / "FAILURE.json",
            {"error": str(exc), "elapsed_seconds": time.monotonic() - began},
        )
        raise
