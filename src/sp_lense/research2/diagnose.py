"""One validation-only hook identity check; no fitting or parameter search."""

import json
from pathlib import Path


def main(root, output):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    from sp_lense.research2.runtime import score_logits
    from sp_lense.steering.gated import REVISION, render, require

    root, output = Path(root), Path(output)
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
    layer = model.model.language_model.layers[10]
    model = PeftModel.from_pretrained(model, output / "adapter", is_trainable=False).eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision=REVISION)
    case = next(
        c
        for c in json.loads((root / "data/validation.json").read_text())["cases"]
        if c["case_id"] == "V02_S05"
    )
    text, canonical = render(case, "AB")
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
    tokens = torch.tensor([ids], device="cuda")
    from contextlib import nullcontext

    def run(adapter, delta=None):
        captures = []

        def hook(module, args, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            if delta is not None:
                h = h.clone()
                h[0, -1] += delta.to(h.device)
            captures.append(h[0, -1].detach().cpu().clone())
            if delta is None:
                return output
            return (
                (h,) + output[1:]
                if isinstance(output, tuple)
                else [h] + output[1:]
                if isinstance(output, list)
                else h
            )

        handle = layer.register_forward_hook(hook)
        try:
            with nullcontext() if adapter else model.disable_adapter(), torch.inference_mode():
                logits = model(
                    input_ids=tokens,
                    attention_mask=torch.ones_like(tokens),
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[0, -1]
                result = score_logits(logits, canonical)
            require(len(captures) == 1, "Hook did not fire once")
            return result, captures[0]
        finally:
            handle.remove()

    base, hb = run(False)
    teacher, ht = run(True)
    patched, hp = run(False, ht - hb)
    error = float((hp - ht).abs().max())
    require(error <= 1e-5, "Patched hidden does not equal teacher hidden")
    changes = {
        n: float(p.detach().float().norm()) for n, p in model.named_parameters() if "lora_B" in n
    }
    receipt = {
        "case_id": "V02_S05",
        "order": "AB",
        "split": "validation",
        "post_block_zero_based": 10,
        "position": "final prompt token",
        "patched_hidden_max_abs_error_to_teacher": error,
        "base": base,
        "teacher": teacher,
        "patched": patched,
        "adapter_B_norms": changes,
        "new_forwards": 3,
        "interpretation": "Correct intermediate hidden replacement does not reproduce this teacher decision. Nonzero downstream adapters remain a possible explanation, not a proven causal attribution.",
    }
    (output / "HOOK_CHECK.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("output")
    args = parser.parse_args()
    main(args.root, args.output)
