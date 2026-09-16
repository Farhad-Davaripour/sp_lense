"""Evaluate a TRAIN-frozen probe-and-select rule on new Colab GPU probes.

Reuse verified baseline/-0.20 records; query smaller steps only when eligible.
The rule sees classifier confidence and model scores, not ground-truth labels.
"""

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from gated_chat import REVISION, action_metrics, atomic, render, require
from search_steering_rules import choose


def main(payload, previous, rule_path, output, callback=None):
    payload, previous, out = Path(payload), Path(previous), Path(output)
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    freeze = json.loads(Path(rule_path).read_text())
    rule = freeze["winner"]["rule"]
    require(rule["family"] == "flip_guard", "Only frozen minimum-step flip guard is supported here")
    atomic(out / "RULE_FREEZE.json", freeze)
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in previous.iterdir()
        if p.name
        in ("validation.jsonl", "holdout.jsonl", "CLASSIFIER.json", "RUNTIME.json", "RESULT.json")
    }
    atomic(out / "SOURCE_PINS.json", sources)
    case_data = json.loads((payload / "cases.json").read_text())
    probabilities = json.loads((previous / "CLASSIFIER.json").read_text())["probabilities"]
    cache = {}
    for split in ("validation", "holdout"):
        records = [
            json.loads(line) for line in (previous / (split + ".jsonl")).read_text().splitlines()
        ]
        cache[split] = {}
        for row in records:
            if row["condition"] in ("baseline", "always_on"):
                strength = 0.0 if row["condition"] == "baseline" else row["applied_strength"]
                cache[split].setdefault((row["case_id"], row["order"]), {})[strength] = row
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    require(
        torch.cuda.is_available() and transformers.__version__ == "5.15.1",
        "Pinned CUDA runtime required",
    )
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.empty_cache()
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
    vector = torch.tensor(
        json.loads((payload / "axis.json").read_text())["direction"],
        device="cuda",
        dtype=torch.float32,
    )
    vector = vector / vector.norm()
    layer = model.model.language_model.layers[10]
    labels = [[32, 357], [33, 417]]
    completed = 0
    total = sum(len(v) for v in cache.values())
    forwards = 0

    def progress(stage):
        require(time.monotonic() - began < 7200, "Two-hour limit exceeded")
        status = {
            "stage": stage,
            "completed": completed,
            "total": total,
            "percent": round(100 * completed / total, 2),
            "new_gpu_forwards": forwards,
            "elapsed_seconds": round(time.monotonic() - began, 1),
        }
        atomic(out / "STATUS.json", status)
        if callback:
            callback(status)

    def encode(case, order):
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
        require(0 < len(ids) <= 1024, "Invalid prompt length")
        return {
            "case_id": case["case_id"],
            "order": order,
            "canonical_index": canonical,
            "class_label": case["class_label"],
            "ids": ids,
            "input_ids_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
        }

    def score(views, strength, split):
        nonlocal forwards
        groups = defaultdict(list)
        for v in views:
            groups[len(v["ids"])].append(v)
        for length, pending in sorted(groups.items()):
            for offset in range(0, len(pending), 4):
                chunk = pending[offset : offset + 4]

                def hook(module, args, output):
                    h = output[0] if isinstance(output, (tuple, list)) else output
                    updated = h.clone()
                    updated[:, -1, :] = (
                        h[:, -1, :] + strength * h[:, -1, :].norm(dim=-1, keepdim=True) * vector
                    )
                    return (
                        (updated,) + output[1:]
                        if isinstance(output, tuple)
                        else [updated] + output[1:]
                        if isinstance(output, list)
                        else updated
                    )

                handle = layer.register_forward_hook(hook) if strength else None
                try:
                    with torch.inference_mode():
                        ids = torch.tensor([v["ids"] for v in chunk], device="cuda")
                        result = model(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            use_cache=False,
                            logits_to_keep=1,
                        )
                        logits = result.logits[:, -1].float()
                        require(bool(torch.isfinite(logits).all()), "Non-finite model output")
                        lp = logits.log_softmax(-1)
                        mass = torch.stack([lp[:, group].exp().sum(-1) for group in labels], -1)
                        pair = mass / mass.sum(-1, keepdim=True)
                        rows = [
                            {k: v for k, v in view.items() if k != "ids"}
                            | {
                                "split": split,
                                "strength": strength,
                                "label_mass": mass[i].sum().item(),
                                "canonical_probability": pair[i, view["canonical_index"]].item(),
                                "pair_argmax": pair[i].argmax().item(),
                                "full_argmax": logits[i].argmax().item(),
                            }
                            for i, view in enumerate(chunk)
                        ]
                finally:
                    if handle:
                        handle.remove()
                for row in rows:
                    cache[split][row["case_id"], row["order"]][strength] = row
                with (out / "probes.jsonl").open("a") as stream:
                    stream.write("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
                    stream.flush()
                    os.fsync(stream.fileno())
                forwards += len(rows)
                progress(split + " probes at " + str(strength))

    summaries = {}
    atomic(
        out / "RUNTIME.json",
        {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "model_revision": REVISION,
            "gpu": torch.cuda.get_device_name(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rule_source_sha256": hashlib.sha256(
                Path(choose.__code__.co_filename).read_bytes()
            ).hexdigest(),
            "dtype": "float32",
            "tf32": False,
        },
    )
    for split in ("validation", "holdout"):
        case_map = {c["case_id"]: c for c in case_data[split]}
        baseline = {key: dict(values[0.0]) for key, values in cache[split].items()}
        chosen = {}
        active = {}
        for key, row in baseline.items():
            p = probabilities[split][key[0]]
            if p < rule["confidence_floor"] or row["pair_argmax"] != row["canonical_index"]:
                chosen[key] = 0.0
                completed += 1
            else:
                view = encode(case_map[key[0]], key[1])
                require(
                    view["input_ids_sha256"] == row["input_ids_sha256"],
                    "Cached baseline token mismatch",
                )
                active[key] = view
        progress(split + " eligibility")
        parity_views = list(active.values())[:4]
        if parity_views:
            score(parity_views, 0.0, split)
            for view in parity_views:
                key = view["case_id"], view["order"]
                require(
                    abs(
                        cache[split][key][0.0]["canonical_probability"]
                        - baseline[key]["canonical_probability"]
                    )
                    < 1e-5,
                    "Cached baseline GPU parity failed",
                )

        for strength in [-0.01, -0.02, -0.05, -0.1, -0.2]:
            if abs(strength) > rule["cap"]:
                continue
            missing = [v for key, v in active.items() if strength not in cache[split][key]]
            if missing:
                score(missing, strength, split)
            accepted = []
            for key in active:
                # Select only among steps already computed; later steps cannot change a minimum-step decision.
                available = cache[split][key]
                partial_rule = {**rule, "cap": abs(strength)}
                selected = choose(
                    partial_rule, probabilities[split][key[0]], baseline[key], available
                )
                if selected:
                    chosen[key] = selected
                    accepted.append(key)
                    completed += 1
            for key in accepted:
                del active[key]
            progress(split + " decisions at " + str(strength))
        for key in active:
            chosen[key] = 0.0
            completed += 1
        final = []
        for key, strength in chosen.items():
            row = dict(baseline[key] if strength == 0 else cache[split][key][strength])
            row.update(
                {
                    "selected_strength": strength,
                    "gate_probability": probabilities[split][key[0]],
                    "condition": "guarded",
                }
            )
            final.append(row)
        (out / (split + ".jsonl")).write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in final)
        )
        summaries[split] = action_metrics(list(baseline.values()), final)
        summaries[split]["intervention_views"] = sum(s != 0 for s in chosen.values())
        atomic(out / "RESULT.json", {"state": "in_progress", "splits": summaries, "rule": rule})
        progress(split + " completed")
    result = {
        "state": "completed",
        "rule": rule,
        "splits": summaries,
        "new_gpu_forwards": forwards,
        "evaluated_views": completed,
        "seconds": time.monotonic() - began,
        "limitations": "TRAIN-selected rule; reused verified baseline and -0.20 records. Extra GPU probes support per-view selection. Directional choice protection is enforced on the surrogate by design, not proof of universally safe/correct behavior.",
    }
    atomic(out / "RESULT.json", result)
    progress("completed")
    return result
