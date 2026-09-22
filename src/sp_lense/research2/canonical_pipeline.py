"""Canonical-only detector -> frozen base -> conditional controller -> guards."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from sp_lense.research2.jev_gate import read
from sp_lense.research2.metrics import keep, key, qualifies, summarize
from sp_lense.research2.runtime import save_rows, score_logits
from sp_lense.research2.student import attach_controller
from sp_lense.steering.gated import REVISION, atomic, render, require

STUDY = "study/02_canonical_pipeline"
CHECKPOINT = "study/02_adaptive_steering/final_position_run/controller.npz"


def main(root, output):
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    plan = read(root / STUDY / "plan.json")
    freeze = read(root / STUDY / "FREEZE.json")
    gate = read(root / STUDY / "gate/GATE.json")
    for manifest in (freeze["hashes"], read(root / "PILOT_MANIFEST.json")):
        for name, digest in manifest.items():
            path = (root / name).resolve()
            require(
                path.is_relative_to(root.resolve())
                and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                f"Frozen input changed: {name}",
            )
    atomic(output / "INPUT_PINS.json", read(root / "PILOT_MANIFEST.json"))
    atomic(output / "PLAN.json", plan)
    atomic(output / "GATE.json", gate)
    forwards = 0

    def progress(stage, done=0, total=128):
        atomic(
            output / "STATUS.json",
            {
                "stage": stage,
                "completed": done,
                "total": total,
                "percent": 100 * done / total,
                "forwards": forwards,
                "elapsed_seconds": time.monotonic() - began,
            },
        )
        require(
            time.monotonic() - began < plan["max_gpu_seconds"], "Canonical run time cap reached"
        )

    try:
        require(
            torch.cuda.is_available() and transformers.__version__ == "5.15.1",
            "Pinned T4 runtime required",
        )
        progress("loading original base only")
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
        for p in model.parameters():
            p.requires_grad_(False)
        require(
            not any("lora_" in n for n, _ in model.named_parameters()), "LoRA parameters present"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-0.8B", revision=REVISION, trust_remote_code=False
        )
        labels = [
            sorted(
                {
                    ids[0]
                    for text in (label, " " + label, "\n" + label)
                    if len(ids := tokenizer.encode(text, add_special_tokens=False)) == 1
                }
            )
            for label in ("A", "B")
        ]
        require(labels == [[32, 357], [33, 417]], "Answer-token mapping changed")
        with np.load(root / CHECKPOINT, allow_pickle=False) as checkpoint:
            arrays = {
                n: checkpoint[n] for n in ("x_mean", "input_basis", "scale", "basis", "weights")
            }

        def fingerprint():
            digest = hashlib.sha256()
            for name, p in model.named_parameters():
                digest.update(name.encode())
                digest.update(p.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        before = fingerprint()
        cases = read(root / plan["dataset"])["cases"]
        base_rows, candidate_rows, final_rows, traces = [], [], [], []
        accepted = set()
        for case in cases:
            probability = gate["probabilities"][case["case_id"]]
            for order in plan["answer_orders"]:
                progress("canonical pipeline", len(base_rows))
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
                tensor = torch.tensor([ids], device="cuda")
                metadata = {
                    "case_id": case["case_id"],
                    "order": order,
                    "class_label": case["class_label"],
                    "canonical_index": canonical,
                    "gate_probability": probability,
                    "input_ids_sha256": hashlib.sha256(
                        json.dumps(ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                }

                def score(tensor=tensor, canonical=canonical, metadata=metadata):
                    nonlocal forwards
                    with torch.inference_mode():
                        logits = model(
                            input_ids=tensor,
                            attention_mask=torch.ones_like(tensor),
                            use_cache=False,
                            logits_to_keep=1,
                        ).logits[0, -1]
                        result = score_logits(logits, canonical)
                    forwards += 1
                    return metadata | result

                base = score()
                base_rows.append(base)
                candidate = None
                if probability < 0.5:
                    reason = "gate_off"
                elif not keep(base):
                    reason = "preserve_baseline_STOP"
                else:
                    handle = attach_controller(model, arrays, 22, "last", 4)
                    try:
                        candidate = score()
                    finally:
                        handle.remove()
                    candidate_rows.append(candidate)
                    if qualifies(base | {"gate_probability": 1.0}, candidate):
                        reason = "accepted"
                        accepted.add(key(base))
                    elif keep(candidate):
                        reason = "no_desired_flip"
                    elif candidate["label_mass"] < 0.5:
                        reason = "low_AB_mass"
                    elif base["label_mass"] - candidate["label_mass"] > 0.02:
                        reason = "AB_mass_loss"
                    else:
                        reason = "no_probability_improvement"
                final = candidate if reason == "accepted" else base
                final_rows.append(final | {"intervention_accepted": reason == "accepted"})
                traces.append(
                    {
                        "case_id": case["case_id"],
                        "order": order,
                        "gate_probability": probability,
                        "candidate_executed": candidate is not None,
                        "decision": reason,
                    }
                )
                if len(base_rows) % 16 == 0:
                    for name, values in (
                        ("base", base_rows),
                        ("candidates", candidate_rows),
                        ("final", final_rows),
                        ("trace", traces),
                    ):
                        save_rows(output / f"{name}.jsonl", values)
        for name, values in (
            ("base", base_rows),
            ("candidates", candidate_rows),
            ("final", final_rows),
            ("trace", traces),
        ):
            save_rows(output / f"{name}.jsonl", values)
        after = fingerprint()
        require(before == after, "Base model parameters changed")
        eligibility = [
            r | {"gate_probability": float(r["gate_probability"] >= 0.5)} for r in base_rows
        ]
        result = {
            "state": "completed",
            "scope": plan["scope"],
            "cases": len(cases),
            "views": len(base_rows),
            "gate": gate["metrics"],
            "pipeline": summarize(eligibility, final_rows, accepted),
            "candidate_forwards": len(candidate_rows),
            "forwards": forwards,
            "teacher_loaded": False,
            "lora_parameters_present": False,
            "model_class": type(model).__name__,
            "weight_sha256_before": before,
            "weight_sha256_after": after,
            "elapsed_seconds": time.monotonic() - began,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "revision": REVISION,
            "detector_stage": "Fresh API calls executed locally before GPU stage; no API credential uploaded.",
            "interpretation": plan["interpretation"],
        }
        atomic(output / "METRICS.json", result)
        atomic(
            output / "ARTIFACTS.json",
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.iterdir()
                if p.is_file() and p.name not in ("STATUS.json", "ARTIFACTS.json")
            },
        )
        progress("completed", len(base_rows))
        return result
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        atomic(
            output / "FAILURE.json",
            {"error": str(exc), "elapsed_seconds": time.monotonic() - began},
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    p.add_argument("output")
    a = p.parse_args()
    print(json.dumps(main(a.root, a.output)))
