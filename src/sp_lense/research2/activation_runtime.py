"""Pinned model scoring and intermediate hidden-state hooks for the adaptive pilot."""

import hashlib
import json
import time
from contextlib import nullcontext

from sp_lense.research2.runtime import score_logits
from sp_lense.steering.gated import REVISION, render, require


class Engine:
    def __init__(self, root, plan, began):
        import torch
        import transformers
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        require(
            torch.cuda.is_available() and transformers.__version__ == "5.15.1",
            "Pinned GPU runtime required",
        )
        torch.manual_seed(plan["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.torch, self.began, self.cap, self.forwards = torch, began, plan["max_gpu_seconds"], 0
        self.tokenizer = AutoTokenizer.from_pretrained(
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
        self.layers = model.model.language_model.layers
        require(
            len(self.layers) == 24 and max(plan["layers"]) < 23,
            "Intervention must precede final block output",
        )
        self.model = PeftModel.from_pretrained(
            model, root / "study/02_lora_transfer/run/adapter", is_trainable=False
        ).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        labels = [
            sorted(
                {
                    ids[0]
                    for text in (s, " " + s, "\n" + s)
                    if len(ids := self.tokenizer.encode(text, add_special_tokens=False)) == 1
                }
            )
            for s in ("A", "B")
        ]
        require(labels == [[32, 357], [33, 417]], "Answer token set mismatch")

    def fingerprint(self):
        h = hashlib.sha256()
        for name, p in self.model.named_parameters():
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def encode(self, case, order, gate_probability):
        text, canonical = render(case, order)
        ids = self.tokenizer.apply_chat_template(
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
        require(0 < len(ids) <= 1024, "Prompt length outside baseline limits")
        return {
            "case_id": case["case_id"],
            "order": order,
            "class_label": case["class_label"],
            "canonical_index": canonical,
            "ids": ids,
            "gate_probability": gate_probability,
            "input_ids_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
        }

    def score(self, view, teacher=False, capture=(), patch=None):
        torch = self.torch
        require(time.monotonic() - self.began < self.cap, "Hard GPU run budget reached")
        snapshots, handles, diagnostics = {}, [], {}
        indices = sorted(set(capture) | ({patch[0]} if patch else set()))

        def make_hook(index):
            def hook(module, args, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                require(h.shape[0] == 1 and h.shape[-1] == 1024, "Hidden shape mismatch")
                if index in capture:
                    require(index not in snapshots, "Hook fired more than once")
                    snapshots[index] = h[0].detach().float().cpu().clone()
                if patch is None or index != patch[0]:
                    return output
                x = h[0] if patch[1] == "all" else h[0, -1:]
                delta = patch[2](x)
                require(
                    delta.shape == x.shape and bool(torch.isfinite(delta).all()),
                    "Invalid predicted delta",
                )
                diagnostics["relative_delta_norm"] = (
                    delta.norm() / x.norm().clamp_min(1e-12)
                ).item()
                diagnostics["layer"] = index
                diagnostics["scope"] = patch[1]
                updated = h.clone()
                if patch[1] == "all":
                    updated[0] += delta
                else:
                    updated[0, -1:] += delta
                diagnostics["patch_addition_max_error"] = (
                    ((updated[0] if patch[1] == "all" else updated[0, -1:]) - x - delta)
                    .abs()
                    .max()
                    .item()
                )
                return (
                    (updated,) + output[1:]
                    if isinstance(output, tuple)
                    else [updated] + output[1:]
                    if isinstance(output, list)
                    else updated
                )

            return hook

        try:
            for index in indices:
                handles.append(self.layers[index].register_forward_hook(make_hook(index)))
            with nullcontext() if teacher else self.model.disable_adapter(), torch.inference_mode():
                ids = torch.tensor([view["ids"]], device="cuda")
                logits = self.model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[0, -1]
                row = score_logits(logits, view["canonical_index"])
            require(set(snapshots) == set(capture), "Missing capture hook")
            require(patch is None or "layer" in diagnostics, "Intervention hook did not fire")
        finally:
            for handle in handles:
                handle.remove()
        self.forwards += 1
        return {k: v for k, v in view.items() if k != "ids"} | row | diagnostics | {
            "adapter_enabled": teacher
        }, snapshots
