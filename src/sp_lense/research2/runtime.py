"""One bounded teacher fit, followed conditionally by block-10 oracle transfer."""

import hashlib
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

from sp_lense.research2.metrics import evaluate, key, transfer_recovery
from sp_lense.steering.gated import REVISION, atomic, render, require

LABELS = [[32, 357], [33, 417]]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def save_rows(path, values):
    Path(path).write_text("".join(json.dumps(v, allow_nan=False) + "\n" for v in values))


def score_logits(logits, canonical):
    """Exact Research 1 grouped A/B token mass and conditional answer scoring."""
    import torch

    logits = logits.float()
    require(bool(torch.isfinite(logits).all()), "Nonfinite logits")
    lp = logits.log_softmax(-1)
    mass = torch.stack([lp[group].exp().sum() for group in LABELS])
    pair = mass / mass.sum()
    return {
        "label_mass": mass.sum().item(),
        "canonical_probability": pair[canonical].item(),
        "pair_argmax": pair.argmax().item(),
        "full_argmax": logits.argmax().item(),
        "answer_probabilities": pair.tolist(),
        "answer_token_mass": mass.tolist(),
    }


def main(repo, output, config, callback=None):
    import numpy as np
    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    root, out = Path(repo), Path(output)
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    forwards = 0

    def progress(stage, completed=0, total=1):
        elapsed = time.monotonic() - began
        status = {
            "stage": stage,
            "completed": completed,
            "total": total,
            "percent": 100 * completed / total,
            "elapsed_seconds": elapsed,
            "scored_forwards": forwards,
        }
        atomic(out / "STATUS.json", status)
        if callback:
            callback(status)
        require(elapsed < config["max_seconds"], "Hard GPU runtime budget reached")

    try:
        require(torch.cuda.is_available(), "CUDA GPU required")
        require(transformers.__version__ == "5.15.1", "Pinned transformers required")
        require(config["epochs"] == 1, "Pilot permits one epoch")
        atomic(out / "CONFIG.json", config)
        source_paths = list((root / "src/sp_lense/research2").glob("*.py")) + [
            root / "src/sp_lense/steering/gated.py",
            root / "src/sp_lense/steering/policy.py",
            root / "study/02_lora_transfer/config.json",
        ]
        for split in ("train", "validation", "holdout"):
            source_paths.append(root / f"data/{split}.json")
        source_paths.extend(
            [
                root / "study/baseline_scores/CLASSIFIER.json",
                root / "study/policy_training/observations.jsonl",
            ]
        )
        for split in ("validation", "holdout"):
            source_paths.extend(
                [
                    root / f"study/baseline_scores/{split}.jsonl",
                    root / f"study/guarded_steering/{split}.jsonl",
                ]
            )
        atomic(
            out / "INPUT_PINS.json",
            {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source_paths
            },
        )
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        progress("loading pinned model")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-0.8B", revision=REVISION, trust_remote_code=False
        )
        model = (
            Qwen3_5ForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3.5-0.8B",
                revision=REVISION,
                dtype=torch.float32,
                attn_implementation="eager",
                trust_remote_code=False,
            )
            .to("cuda")
            .eval()
        )
        actual_labels = [
            sorted(
                {
                    ids[0]
                    for text in (label, " " + label, "\n" + label)
                    if len(ids := tokenizer.encode(text, add_special_tokens=False)) == 1
                }
            )
            for label in ("A", "B")
        ]
        require(actual_labels == LABELS, "Answer token set changed")
        layer = model.model.language_model.layers[10]
        cases = {
            s: read(root / f"data/{s}.json")["cases"] for s in ("train", "validation", "holdout")
        }
        probabilities = read(root / "study/baseline_scores/CLASSIFIER.json")["probabilities"]
        views = {}
        for split, items in cases.items():
            views[split] = []
            for case in items:
                for order in ("AB", "BA"):
                    text, canonical = render(case, order)
                    ids = tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    if hasattr(ids, "keys"):
                        ids = ids["input_ids"]
                    if hasattr(ids, "tolist"):
                        ids = ids.tolist()
                    if ids and isinstance(ids[0], list):
                        ids = ids[0]
                    ids = list(map(int, ids))
                    require(0 < len(ids) <= 1024, "Prompt length outside original limits")
                    views[split].append(
                        {
                            "case_id": case["case_id"],
                            "order": order,
                            "class_label": case["class_label"],
                            "canonical_index": canonical,
                            "ids": ids,
                            "input_ids_sha256": hashlib.sha256(
                                json.dumps(ids, separators=(",", ":")).encode()
                            ).hexdigest(),
                            "gate_probability": probabilities.get(split, {}).get(
                                case["case_id"], 0.0
                            ),
                        }
                    )

        def forward(v):
            ids = torch.tensor([v["ids"]], device="cuda")
            return model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=1,
            ).logits[0, -1]

        def score(v, adapter=False, delta=None, capture=False):
            nonlocal forwards
            progress("forward budget check", forwards, max(1, forwards + 1))
            captured = []

            def hook(module, args, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                require(h.shape[0] == 1 and h.shape[-1] == 1024, "Hook tensor mismatch")
                if capture:
                    captured.append(h[0, -1].detach().float().cpu().clone())
                if delta is None:
                    return output
                require(tuple(delta.shape) == (1024,), "Delta shape mismatch")
                updated = h.clone()
                updated[0, -1] += delta.to(device=h.device, dtype=h.dtype)
                return (
                    (updated,) + output[1:]
                    if isinstance(output, tuple)
                    else [updated] + output[1:]
                    if isinstance(output, list)
                    else updated
                )

            handle = layer.register_forward_hook(hook) if capture or delta is not None else None
            disabled = (
                model.disable_adapter()
                if hasattr(model, "disable_adapter") and not adapter
                else nullcontext()
            )
            try:
                with disabled, torch.inference_mode():
                    result = score_logits(forward(v), v["canonical_index"])
                if capture:
                    require(len(captured) == 1, "Hook did not fire exactly once")
            finally:
                if handle:
                    handle.remove()
            forwards += 1
            return {k: value for k, value in v.items() if k != "ids"} | result, captured[
                0
            ] if capture else None

        archived = {
            s: {
                key(r): r
                for r in rows(root / f"study/baseline_scores/{s}.jsonl")
                if r["condition"] == "baseline"
            }
            for s in ("validation", "holdout")
        }
        smoke = views["validation"][:4]
        smoke_rows = [score(v)[0] for v in smoke]
        tolerance = config["parity_tolerance"]

        def parity(fresh, reference):
            errors = []
            for r in fresh:
                b = reference[key(r)]
                require(r["input_ids_sha256"] == b["input_ids_sha256"], "Prompt hash parity failed")
                require(r["pair_argmax"] == b["pair_argmax"], "Baseline decision parity failed")
                errors.extend(abs(r[k] - b[k]) for k in ("canonical_probability", "label_mass"))
            error = max(errors, default=0)
            require(error <= tolerance, f"Baseline probability parity failed: {error}")
            return error

        smoke_error = parity(smoke_rows, archived["validation"])
        atomic(out / "SMOKE.json", {"max_error": smoke_error, "views": 4, "tolerance": tolerance})
        targets = [
            name
            for name, module in model.named_modules()
            if ".language_model.layers." in name
            and name.endswith((".q_proj", ".v_proj"))
            and isinstance(module, torch.nn.Linear)
        ]
        require(bool(targets), "No language LoRA targets")
        model = get_peft_model(
            model,
            LoraConfig(
                r=config["rank"],
                lora_alpha=config["alpha"],
                lora_dropout=0,
                target_modules=targets,
                bias="none",
            ),
        )
        model.eval()
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        require(
            bool(trainable) and all("lora_" in n for n, p in trainable),
            "Unexpected trainable base weights",
        )

        def frozen_hash():
            digest = hashlib.sha256()
            for name, p in model.named_parameters():
                if "lora_" not in name:
                    require(not p.requires_grad, "Trainable base parameter")
                    digest.update(name.encode())
                    digest.update(p.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        before_hash = frozen_hash()
        init_error = parity([score(v)[0] for v in smoke], {key(r): r for r in smoke_rows})
        train_base = {
            key(r): r
            for r in rows(root / "study/policy_training/observations.jsonl")
            if r["strength"] == 0
        }
        training = list(views["train"])
        random.Random(config["seed"]).shuffle(training)
        training_labels = []
        optimizer = torch.optim.AdamW(
            [p for n, p in trainable], lr=config["learning_rate"], weight_decay=0.01
        )
        optimizer.zero_grad(set_to_none=True)
        losses = []
        train_began = time.monotonic()
        for i, v in enumerate(training):
            progress("teacher training", i, len(training))
            require(
                v["input_ids_sha256"] == train_base[key(v)]["input_ids_sha256"],
                "TRAIN prompt drift",
            )
            target = (
                1 - v["canonical_index"]
                if v["class_label"] in {"SELF", "OTHER"}
                else train_base[key(v)]["pair_argmax"]
            )
            training_labels.append(
                {"case_id": v["case_id"], "order": v["order"], "target_index": target}
            )
            logits = forward(v).float()
            lp = logits.log_softmax(-1)
            loss = -torch.logsumexp(lp[LABELS[target]], dim=0)
            require(bool(torch.isfinite(loss)), "Nonfinite training loss")
            (loss / config["accumulation_steps"]).backward()
            losses.append(loss.item())
            if (i + 1) % config["accumulation_steps"] == 0 or i + 1 == len(training):
                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in trainable], 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        training_seconds = time.monotonic() - train_began
        after_hash = frozen_hash()
        require(before_hash == after_hash, "Frozen base weights changed")
        disabled_error = parity([score(v)[0] for v in smoke], {key(r): r for r in smoke_rows})
        model.save_pretrained(out / "adapter", safe_serialization=True)
        save_rows(out / "training_labels.jsonl", training_labels)
        atomic(
            out / "TRAINING.json",
            {
                "losses": losses,
                "seconds": training_seconds,
                "trainable_parameters": sum(p.numel() for n, p in trainable),
                "trainable_names": [n for n, p in trainable],
                "target_modules": targets,
                "base_sha256_before": before_hash,
                "base_sha256_after": after_hash,
                "initial_disabled_error": init_error,
                "trained_disabled_error": disabled_error,
            },
        )
        model.eval()
        result = {
            "state": "running",
            "splits": {},
            "transfer": {},
            "runtime": {
                "gpu": torch.cuda.get_device_name(),
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "model_revision": REVISION,
                "dtype": "float32",
                "attention": "eager",
                "tf32": False,
                "training_seconds": training_seconds,
            },
        }
        base_all, teacher_all, captures = {}, {}, {}
        mean = None
        generator = torch.Generator(device="cpu").manual_seed(config["seed"])
        for split in ("validation", "holdout"):
            base, teacher = [], []
            captures[split] = {}
            for i, v in enumerate(views[split]):
                progress(split + " teacher evaluation", i, len(views[split]))
                b, hb = score(v, capture=True)
                t, ht = score(v, adapter=True, capture=True)
                base.append(b)
                teacher.append(t)
                captures[split][key(v)] = (hb, ht - hb)
            parity_error = parity(base, archived[split])
            save_rows(out / f"{split}_base.jsonl", base)
            save_rows(out / f"{split}_teacher.jsonl", teacher)
            base_all[split], teacher_all[split] = base, teacher
            result["splits"][split] = evaluate(base, teacher) | {
                "baseline_parity_max_error": parity_error
            }
            r1 = rows(root / f"study/guarded_steering/{split}.jsonl")
            result["splits"][split]["research1"] = evaluate(base, r1)["methods"]["raw"]
            if split == "validation":
                result["gate1_pass"] = result["splits"][split]["gate1_pass"]
                atomic(
                    out / "VALIDATION_DECISION.json",
                    {
                        "gate1_pass": result["gate1_pass"],
                        "configuration": config,
                        "repairs_used": 0,
                        "holdout_opened_for_new_scoring": False,
                    },
                )
                if result["gate1_pass"]:
                    differences = []
                    for i, v in enumerate(views["train"]):
                        if v["class_label"] not in {"SELF", "OTHER"}:
                            continue
                        progress("training mean activation difference", i, len(views["train"]))
                        _, hb = score(v, capture=True)
                        _, ht = score(v, adapter=True, capture=True)
                        differences.append(ht - hb)
                    mean = torch.stack(differences).mean(0)
                    np.savez_compressed(
                        out / "training_differences.npz",
                        differences=torch.stack(differences).numpy(),
                        mean=mean.numpy(),
                    )
            if result["gate1_pass"]:
                predictions = {method: [] for method in ("oracle", "training_mean", "random")}
                deltas, relative = [], []
                for i, v in enumerate(views[split]):
                    progress(split + " activation transfer", i, len(views[split]))
                    hb, delta = captures[split][key(v)]
                    noise = torch.randn(delta.shape, generator=generator)
                    noise *= delta.norm() / noise.norm()
                    deltas.append(delta.numpy())
                    relative.append(
                        {
                            "case_id": v["case_id"],
                            "order": v["order"],
                            "oracle_relative_norm": (delta.norm() / hb.norm()).item(),
                            "mean_relative_norm": (mean.norm() / hb.norm()).item(),
                        }
                    )
                    for name, difference in (
                        ("oracle", delta),
                        ("training_mean", mean),
                        ("random", noise),
                    ):
                        prediction, _ = score(v, delta=difference)
                        predictions[name].append(prediction)
                np.savez_compressed(
                    out / f"{split}_differences.npz",
                    differences=np.stack(deltas),
                    view_ids=np.array([v["case_id"] + ":" + v["order"] for v in views[split]]),
                )
                save_rows(out / f"{split}_perturbations.jsonl", relative)
                result["transfer"][split] = {}
                for name, prediction in predictions.items():
                    save_rows(out / f"{split}_{name}.jsonl", prediction)
                    result["transfer"][split][name] = evaluate(base, prediction) | {
                        "teacher_recovery": transfer_recovery(base, teacher, prediction)
                    }
            atomic(out / "METRICS.json", result)
        result.update(
            state="completed",
            elapsed_seconds=time.monotonic() - began,
            scored_forwards=forwards,
            training_forwards=len(training),
            interpretation="Exploratory; frozen detector features and reused diagnostic holdout. Oracle transfer uses the same-prompt teacher.",
        )
        atomic(out / "METRICS.json", result)
        atomic(
            out / "ARTIFACTS.json",
            {
                str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.rglob("*")
                if p.is_file() and p.name not in {"STATUS.json", "ARTIFACTS.json"}
            },
        )
        progress("completed", 1, 1)
        return result
    except BaseException as exc:
        atomic(
            out / "FAILURE.json",
            {
                "error": repr(exc),
                "elapsed_seconds": time.monotonic() - began,
                "scored_forwards": forwards,
            },
        )
        raise
