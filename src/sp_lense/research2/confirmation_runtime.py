"""Model-size-independent execution of the frozen confirmation recipe."""

import hashlib
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from sp_lense.research2.controller import fit, predict, replay_fit
from sp_lense.research2.metrics import keep
from sp_lense.research2.runtime import LABELS, read, save_rows, score_logits
from sp_lense.steering.gated import atomic, render, require

STUDY = "study/02_confirmation"


class Runner:
    def __init__(self, root, output, model_key, seed):
        import torch
        import transformers
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.root, self.out = Path(root), Path(output)
        self.out.mkdir(parents=True, exist_ok=False)
        self.plan = read(self.root / STUDY / "plan.json")
        self.config = self.plan["models"][model_key]
        self.seed, self.began, self.forwards = seed, time.monotonic(), 0
        self.torch = torch
        require(torch.cuda.is_available(), "CUDA required")
        require(transformers.__version__ == "5.15.1", "Pinned transformers required")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.progress("loading model", 0, 1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"], revision=self.config["revision"], trust_remote_code=False
        )
        self.model = (
            Qwen3_5ForConditionalGeneration.from_pretrained(
                self.config["model"],
                revision=self.config["revision"],
                dtype=torch.float32,
                attn_implementation="eager",
                trust_remote_code=False,
            )
            .to("cuda")
            .eval()
        )
        self.layers = self.model.model.language_model.layers
        require(
            len(self.layers) == 24, "Expected 24 blocks; fixed layer22 must precede final block"
        )
        self.width = self.model.config.text_config.hidden_size
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
        require(labels == LABELS, "Answer-token map changed")
        for p in self.model.parameters():
            p.requires_grad_(False)
        atomic(
            self.out / "RUNTIME.json",
            {
                "model": self.config,
                "seed": seed,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(),
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "dtype": "float32",
                "width": self.width,
                "layer": 22,
            },
        )

    def progress(self, stage, done, total):
        elapsed = time.monotonic() - self.began
        atomic(
            self.out / "STATUS.json",
            {
                "stage": stage,
                "completed": done,
                "total": total,
                "percent": 100 * done / max(1, total),
                "elapsed_seconds": elapsed,
                "forwards": self.forwards,
            },
        )
        require(elapsed < self.plan["max_gpu_seconds_per_job"], "Per-job runtime cap reached")

    def fingerprint(self):
        digest = hashlib.sha256()
        for name, p in self.model.named_parameters():
            if "lora_" not in name:
                require(not p.requires_grad, "Base parameter is trainable")
                digest.update(name.encode())
                digest.update(p.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def encode(self, case, order, probability=0, instruction=False):
        text, canonical = render(case, order)
        if instruction:
            text = self.plan["instruction"] + "\n\n" + text
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
        require(0 < len(ids) <= 1024, "Prompt outside frozen length limit")
        return {
            "case_id": case["case_id"],
            "group_id": case["group_id"],
            "order": order,
            "class_label": case["class_label"],
            "canonical_index": canonical,
            "gate_probability": probability,
            "ids": ids,
            "input_ids_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
        }

    def forward(self, view):
        ids = self.torch.tensor([view["ids"]], device="cuda")
        self.forwards += 1
        return self.model(
            input_ids=ids,
            attention_mask=self.torch.ones_like(ids),
            use_cache=False,
            logits_to_keep=1,
        ).logits[0, -1]

    def score(self, view, teacher=False, capture=False, patch=None):
        torch = self.torch
        snapshots = []

        def hook(module, args, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            require(h.shape[0] == 1 and h.shape[-1] == self.width, "Hook shape mismatch")
            x = h[0, -1:]
            if capture:
                snapshots.append(x.detach().float().cpu().clone())
            if patch is None:
                return output
            delta = patch(x)
            require(delta.shape == x.shape and bool(torch.isfinite(delta).all()), "Invalid delta")
            updated = h.clone()
            updated[0, -1:] += delta
            return (
                (updated,) + output[1:]
                if isinstance(output, tuple)
                else [updated] + output[1:]
                if isinstance(output, list)
                else updated
            )

        handle = self.layers[22].register_forward_hook(hook) if capture or patch else None
        disable = (
            self.model.disable_adapter()
            if hasattr(self.model, "disable_adapter") and not teacher
            else nullcontext()
        )
        try:
            with disable, torch.inference_mode():
                scores = score_logits(self.forward(view), view["canonical_index"])
            require(not capture or len(snapshots) == 1, "Missing or repeated capture")
        finally:
            if handle:
                handle.remove()
        return (
            {k: v for k, v in view.items() if k != "ids"} | scores,
            snapshots[0].numpy()[0] if capture else None,
        )

    def train(self):
        import peft
        from peft import LoraConfig, get_peft_model

        torch, cfg = self.torch, self.plan["teacher"]
        require(peft.__version__ == "0.18.1", "Pinned PEFT required")
        # Deliberately reads TRAIN only. No validation, test or detector file enters fitting.
        cases = read(self.root / "data/train.json")["cases"]
        views = [self.encode(c, o) for c in cases for o in ("AB", "BA")]
        bases, xs = [], []
        for i, v in enumerate(views):
            self.progress("TRAIN base activations", i, len(views))
            row, hidden = self.score(v, capture=True)
            bases.append(row)
            xs.append(hidden)
        targets = [
            1 - v["canonical_index"] if v["class_label"] in ("SELF", "OTHER") else b["pair_argmax"]
            for v, b in zip(views, bases)
        ]
        modules = [
            n
            for n, m in self.model.named_modules()
            if ".language_model.layers." in n
            and n.endswith((".q_proj", ".v_proj"))
            and isinstance(m, torch.nn.Linear)
        ]
        require(len(modules) == 12, "Unexpected LoRA target count")
        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=cfg["rank"],
                lora_alpha=cfg["alpha"],
                lora_dropout=0,
                bias="none",
                target_modules=modules,
            ),
        ).eval()
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        require(all("lora_" in n for n, _ in trainable), "Unexpected trainable parameter")
        before = self.fingerprint()
        indices = list(range(len(views)))
        random.Random(self.seed).shuffle(indices)
        optimizer = torch.optim.AdamW(
            [p for _, p in trainable], lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for i, index in enumerate(indices):
            self.progress("LoRA teacher training", i, len(indices))
            lp = self.forward(views[index]).float().log_softmax(-1)
            loss = -torch.logsumexp(lp[LABELS[targets[index]]], dim=0)
            require(bool(torch.isfinite(loss)), "Nonfinite loss")
            (loss / cfg["accumulation_steps"]).backward()
            losses.append(loss.item())
            if (i + 1) % cfg["accumulation_steps"] == 0 or i + 1 == len(indices):
                torch.nn.utils.clip_grad_norm_(
                    [p for _, p in trainable], cfg["gradient_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        del optimizer
        require(before == self.fingerprint(), "Base weights changed during teacher fitting")
        self.model.save_pretrained(self.out / "adapter", safe_serialization=True)
        ys, active, teachers = [], [], []
        for i, (v, b, x) in enumerate(zip(views, bases, xs)):
            self.progress("TRAIN teacher targets", i, len(views))
            t, h = self.score(v, teacher=True, capture=True)
            on = v["class_label"] in ("SELF", "OTHER") and keep(b)
            ys.append(h - x if on else np.zeros_like(x))
            active.append(on)
            teachers.append(t)
        self.progress("fitting fixed controller", 0, 1)
        cp = self.plan["controller"]
        arrays = fit(
            np.stack(xs),
            np.stack(ys),
            active,
            seed=self.seed,
            input_rank=cp["input_rank"],
            output_rank=cp["output_fit_rank"],
            ridge=cp["ridge"],
        )
        np.savez_compressed(self.out / "controller.npz", **arrays)
        np.savez_compressed(
            self.out / "training_activations.npz",
            x=np.stack(xs),
            y=np.stack(ys),
            active=np.array(active),
        )
        error = float(np.max(np.abs(replay_fit(arrays) - arrays["weights"])))
        require(error < 1e-3, "Controller ridge replay failed")
        parity = max(
            abs(self.score(v)[0][field] - b[field])
            for v, b in zip(views[:4], bases[:4])
            for field in ("canonical_probability", "label_mass")
        )
        require(parity < 1e-5, "Disabled teacher differs from original base")
        save_rows(self.out / "train_base.jsonl", bases)
        save_rows(self.out / "train_teacher.jsonl", teachers)
        atomic(
            self.out / "TRAINING.json",
            {
                "seed": self.seed,
                "training_order": indices,
                "targets": targets,
                "losses": losses,
                "trainable_parameters": sum(p.numel() for _, p in trainable),
                "target_modules": modules,
                "base_sha256_before": before,
                "base_sha256_after": self.fingerprint(),
                "disabled_parity_error": parity,
                "ridge_replay_error": error,
                "test_used": False,
                "selection": "No selection or early stopping",
                "forwards": self.forwards,
                "elapsed_seconds": time.monotonic() - self.began,
            },
        )
        self.progress("completed", 1, 1)

    def evaluate(self, adapter, checkpoint):
        from peft import PeftModel

        cases = read(self.root / STUDY / "cases.json")["cases"]
        gate = read(self.root / STUDY / "gate/GATE.json")["probabilities"]
        before = self.fingerprint()
        bases, instructions, teachers = [], [], []
        views = []
        self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False).eval()
        for c in cases:
            for order in ("AB", "BA"):
                self.progress("base, instruction and teacher", len(views), 2 * len(cases))
                v = self.encode(c, order, gate[c["case_id"]])
                views.append(v)
                bases.append(self.score(v)[0])
                instructions.append(self.score(self.encode(c, order, gate[c["case_id"]], True))[0])
                teachers.append(self.score(v, teacher=True)[0])
        for name, records in (
            ("base", bases),
            ("instruction", instructions),
            ("teacher", teachers),
        ):
            save_rows(self.out / f"{name}.jsonl", records)
        # Remove adapters before student execution: no LoRA parameters remain in the model.
        self.model = self.model.unload().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        require(
            not any("lora_" in n for n, _ in self.model.named_parameters()), "Teacher still loaded"
        )
        require(before == self.fingerprint(), "Unloading teacher changed base parameters")
        with np.load(checkpoint, allow_pickle=False) as saved:
            arrays = {n: saved[n] for n in saved.files}
        mean = (
            self.torch.as_tensor(arrays["mean"], device="cuda")
            * self.plan["controller"]["mean_scale"]
        )
        adaptive, constant = [], []
        for i, v in enumerate(views):
            self.progress("teacher-free adaptive and constant", i, len(views))
            adaptive.append(self.score(v, patch=lambda h: predict(h, arrays, 4))[0])
            constant.append(self.score(v, patch=lambda h: mean.expand_as(h))[0])
        save_rows(self.out / "adaptive.jsonl", adaptive)
        save_rows(self.out / "constant.jsonl", constant)
        require(before == self.fingerprint(), "Student evaluation changed base weights")
        atomic(
            self.out / "EXECUTION.json",
            {
                "state": "completed",
                "cases": len(cases),
                "views": len(views),
                "base_sha256_before": before,
                "base_sha256_after": self.fingerprint(),
                "teacher_unloaded_before_student": True,
                "forwards": self.forwards,
                "elapsed_seconds": time.monotonic() - self.began,
                "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                "adapter_sha256": hashlib.sha256(
                    (Path(adapter) / "adapter_model.safetensors").read_bytes()
                ).hexdigest(),
                "execution": "Raw candidates for every view, including controls; gate/guard decisions replayed from actual scores. One candidate per method.",
            },
        )
        self.progress("completed", 1, 1)
