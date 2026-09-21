"""Standalone frozen-base controller: no PEFT model or teacher checkpoint is loaded."""

import json
from pathlib import Path

import numpy as np

from sp_lense.research2.controller import predict
from sp_lense.research2.runtime import score_logits
from sp_lense.steering.gated import REVISION, render, require


def attach_controller(model, arrays, layer, scope, rank):
    """Attach a prompt-time intervention computed solely from this model's hidden states."""
    require(scope in ("last", "all"), "Unknown token scope")

    def hook(module, args, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        require(h.shape[0] == 1, "This pilot controller expects one prompt per call")
        x = h[0] if scope == "all" else h[0, -1:]
        delta = predict(x, arrays, rank)
        modified = h.clone()
        if scope == "all":
            modified[0] += delta
        else:
            modified[0, -1:] += delta
        return (
            (modified,) + output[1:]
            if isinstance(output, tuple)
            else [modified] + output[1:]
            if isinstance(output, list)
            else modified
        )

    return model.model.language_model.layers[layer].register_forward_hook(hook)


def verify(root, output):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    root, output = Path(root), Path(output)
    report = json.loads((output / "METRICS.json").read_text())
    require(report["controller_fitted"], "No controller available")
    selected = report["selected"]["adaptive"]
    rank = int(selected.split(":")[1])
    location = report["location"]
    with np.load(output / "controller.npz", allow_pickle=False) as saved:
        arrays = {
            name: saved[name] for name in ("x_mean", "input_basis", "scale", "basis", "weights")
        }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
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
    require(
        not any("lora_" in name for name, _ in model.named_parameters()),
        "Teacher adapter unexpectedly present",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-0.8B", revision=REVISION, trust_remote_code=False
    )
    cases = json.loads((root / "data/validation.json").read_text())["cases"]
    picked = [
        next(c for c in cases if c["class_label"] == label)
        for label in ("SELF", "OTHER", "ORDINARY", "NONTERMINATION")
    ]
    path = output / ("validation_" + selected.replace(":", "_") + ".jsonl")
    reference = {
        (r["case_id"], r["order"]): r for r in map(json.loads, path.read_text().splitlines())
    }
    errors = []
    for case in picked:
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
            tensor = torch.tensor([ids], device="cuda")
            handle = attach_controller(model, arrays, location["layer"], location["scope"], rank)
            try:
                with torch.inference_mode():
                    logits = model(
                        input_ids=tensor,
                        attention_mask=torch.ones_like(tensor),
                        use_cache=False,
                        logits_to_keep=1,
                    ).logits[0, -1]
                    row = score_logits(logits, canonical)
            finally:
                handle.remove()
            expected = reference[case["case_id"], order]
            require(row["pair_argmax"] == expected["pair_argmax"], "Teacher-free decision mismatch")
            error = max(abs(row[k] - expected[k]) for k in ("canonical_probability", "label_mass"))
            require(error < 1e-5, "Teacher-free score mismatch")
            errors.append({"case_id": case["case_id"], "order": order, "error": error})
    receipt = {
        "model_class": type(model).__name__,
        "adapter_checkpoint_loaded": False,
        "lora_parameters_present": False,
        "new_forwards": len(errors),
        "views": errors,
        "max_error": max(r["error"] for r in errors),
        "selected": selected,
        "location": location,
    }
    (output / "STUDENT_ONLY_CHECK.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.output)))
