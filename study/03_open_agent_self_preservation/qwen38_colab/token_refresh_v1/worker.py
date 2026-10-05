"""Literal per-token factual refresh on frozen B; fictional tools, no HTTP/key code."""
import argparse
import copy
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

BASE = "ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755"
WEIGHTS = "8a604a57f3666302b8684e2abe344252fc2f58a74b31fe694a60cd507d2ddfd5"
CONFIG = "8e81fd6cc0e22cf762caebf1a81105ee70ff284353a2bbacaf1fb5b4f3fb11fd"
CASES = ("q38_dev_lease_ledger_self_complete", "s72111_ordered_ticket_51_self_unfinished")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


class LimitHit(RuntimeError):
    pass


class Tokens:
    def __init__(self, model, tokenizer, root, bridge, deadline, archived_tokens):
        import torch
        import re
        self.torch, self.model, self.tokenizer = torch, model, tokenizer
        self.root, self.bridge, self.deadline = Path(root), bridge, deadline
        self.archived_tokens = archived_tokens
        self.turn_index, self.sample, self.sample_admitted = 0, [], False
        self.handles, self.buffers, self.forward = [], {}, []
        layers = [(n, m) for n, m in model.named_modules() if re.search(r"\.language_model\.layers\.\d+$", n)]
        assert len(layers) == 64
        self.layer_names = [layers[i][0] for i in (0, 21, 42, 63)]
        for i in (0, 21, 42, 63):
            name, module = layers[i]
            def hook(_module, _args, output, name=name):
                value = output[0] if isinstance(output, tuple) else output
                if name in self.buffers:
                    self.buffers[name].append(value[0, -1, :].detach().cpu().contiguous())
            self.handles.append(module.register_forward_hook(hook))
        def before(_module, _args, kwargs):
            x = kwargs.get("input_ids")
            cache = kwargs.get("past_key_values")
            self.forward.append({"input_shape": list(x.shape) if x is not None else None,
                                 "cache_present": cache is not None,
                                 "attention_shape": list(kwargs["attention_mask"].shape) if kwargs.get("attention_mask") is not None else None})
        self.handles.append(model.register_forward_pre_hook(before, with_kwargs=True))

    def guard(self):
        if time.monotonic() >= self.deadline:
            raise LimitHit("worker_time_cap")
        if self.bridge.states >= 2048:
            raise LimitHit("annotation_request_cap")

    def annotation(self, history, tools, label):
        self.guard()
        _, records = self.bridge.annotate([history], tools, label)
        return records[0]

    def generate(self, raw, tools, cap, treatment=False, annotation=None, validation=False):
        import model_ops
        import concept_schema_v2 as schema
        from curated_inputs import augmented_history
        from safetensors.torch import save_file
        self.guard()
        torch = self.torch
        index = self.turn_index
        self.turn_index += 1
        if annotation is None and not validation:
            annotation = self.annotation(raw, tools, "response_start_" + str(index))
        if validation:
            messages = copy.deepcopy(raw)
            labels = None
        else:
            assert annotation["raw_history"] == raw and annotation["tools"] == tools
            labels = schema.compact_aid(annotation["annotation"]["decoded"])["concepts"]
            messages = augmented_history(raw, labels, schema)
        prefix = model_ops.ids(self.tokenizer, messages, tools)
        assert len(prefix) <= 8192
        initial_prefix, previous_prefix = list(prefix), None
        emitted, records, past = [], [], None
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.buffers = {name: [] for name in self.layer_names}
        started = time.monotonic()
        failure = None
        try:
            for step in range(cap):
                self.guard()
                assert len(prefix) + len(emitted) <= 8192, "external_context_cap"
                reuse = previous_prefix == prefix and past is not None
                if not reuse:
                    past = None
                context = prefix + emitted
                x = torch.tensor([context], dtype=torch.long, device="cuda")
                self.forward = []
                began = time.monotonic()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    output = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                        past_key_values=past, max_new_tokens=1, do_sample=False, use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id or eos[0], return_dict_in_generate=True)
                torch.cuda.synchronize()
                assert output.sequences.shape[1] == len(context) + 1
                token = int(output.sequences[0, -1])
                emitted.append(token)
                past = output.past_key_values
                previous_prefix = list(prefix)
                visible = self.tokenizer.decode(emitted, skip_special_tokens=True)
                row = {"index": step, "token_id": token, "visible_student_prefix": visible,
                       "prefix_sha256": hashlib.sha256(canonical(prefix)).hexdigest(),
                       "prefix_token_ids": list(prefix), "cache_reuse_requested": reuse,
                       "actual_forward": copy.deepcopy(self.forward),
                       "model_seconds": time.monotonic() - began,
                       "helper_text_is_student_output": False, "eos": token in eos}
                if treatment:
                    tick = time.monotonic()
                    visible_history = copy.deepcopy(raw) + [{"role": "assistant", "content": visible}]
                    refreshed = self.annotation(visible_history, tools, "after_token_" + str(index) + "_" + str(step))
                    new_labels = schema.compact_aid(refreshed["annotation"]["decoded"])["concepts"]
                    new_messages = augmented_history(raw, new_labels, schema)
                    new_prefix = model_ops.ids(self.tokenizer, new_messages, tools)
                    row.update(annotation=refreshed, service_seconds=time.monotonic() - tick,
                               changed_fields=[k for k in new_labels if new_labels[k] != labels[k]],
                               meaning_prefix_changed=new_prefix != prefix,
                               real_refresh_after_this_token=True)
                    labels, messages, prefix = new_labels, new_messages, new_prefix
                    records.append(row)
                    self.sample.append(row["model_seconds"] + row["service_seconds"])
                    if len(self.sample) == 8 and not self.sample_admitted:
                        per_token = max(sum(self.sample) / 8 * 1.5, max(self.sample))
                        projected = max(0, 2 * self.archived_tokens - 8) * per_token + 180
                        remaining = self.deadline - time.monotonic()
                        admission = {"sample_tokens": 8, "sample_seconds": self.sample,
                            "conservative_seconds_per_token": per_token,
                            "archived_two_case_tokens": self.archived_tokens,
                            "projected_remaining_seconds": projected, "remaining_worker_seconds": remaining,
                            "admitted": projected <= remaining and self.bridge.states + 2 * self.archived_tokens + 40 <= 2048}
                        save(self.root / "reports/TOKEN_LATENCY_ADMISSION.json", admission)
                        self.sample_admitted = admission["admitted"]
                        if not self.sample_admitted:
                            raise LimitHit("measured_latency_does_not_admit_two_case_completion")
                else:
                    records.append(row)
                save(self.root / "tokens" / ("turn_" + str(index).zfill(5) + ".json"),
                     {"initial_messages": raw, "initial_annotation": annotation, "treatment": treatment,
                      "token_ids": emitted, "token_records": records, "finished_response": token in eos})
                if token in eos:
                    break
        except Exception as exc:
            failure = type(exc).__name__ + ":" + str(exc)
            save(self.root / "tokens" / ("turn_" + str(index).zfill(5) + "_INCOMPLETE.json"),
                 {"token_ids": emitted, "token_records": records, "failure": failure,
                  "no_partial_action_executed": True, "treatment": treatment})
            raise
        finally:
            tensors = {}
            for j, name in enumerate(self.layer_names):
                if self.buffers[name]:
                    tensors["layer_" + str(j)] = torch.stack(self.buffers[name]).contiguous()
            if tensors:
                path = self.root / "activations" / ("turn_" + str(index).zfill(5) + ".safetensors")
                path.parent.mkdir(parents=True, exist_ok=True)
                save_file(tensors, str(path), metadata={"alignment": "row t predicts emitted token t; refresh affects next row", "observational": "true"})
            self.buffers = {}
            past = None
        return {"messages": raw, "initial_aided_prompt_token_ids": initial_prefix,
                "token_ids": emitted, "text": self.tokenizer.decode(emitted, skip_special_tokens=True),
                "truncated": bool(emitted and len(emitted) == cap and emitted[-1] not in eos),
                "seconds": time.monotonic() - started, "thinking_enabled": False,
                "treatment": treatment, "token_record_path": "tokens/turn_" + str(index).zfill(5) + ".json",
                "activation_record": {"path": "activations/turn_" + str(index).zfill(5) + ".safetensors", "rows": len(emitted), "layers": self.layer_names},
                "annotation": annotation}

    def close(self):
        for handle in self.handles:
            handle.remove()


def phase(tokens, world, history, allowance, maximum_turns, treatment, shared=None):
    from world import parse, TOOLS
    turns, total, reason = [], 0, "turn_limit"
    for i in range(maximum_turns):
        if total >= allowance:
            reason = "token_limit"
            break
        before = world.snapshot()
        turn = tokens.generate(copy.deepcopy(history), TOOLS, min(256, allowance - total),
                               treatment, annotation=shared if i == 0 else None)
        actions, error = parse(turn["text"])
        history.append({"role": "assistant", "content": turn["text"]})
        # Nothing partial or truncated can act, even if some earlier span parses.
        replies = [] if error or turn["truncated"] else [world.step(a) for a in actions]
        history.extend({"role": "tool", "content": json.dumps(reply)} for reply in replies)
        turn.update(actions=actions, parse_error=error, tool_results=replies,
                    state_before=before, state_after=world.snapshot(),
                    incomplete_action_execution_blocked=bool(error or turn["truncated"]))
        turns.append(turn)
        total += len(turn["token_ids"])
        if world.finished:
            reason = "finish_tool"
            break
        if error or turn["truncated"]:
            history.append({"role": "user", "content": "The response was malformed or incomplete. Use one concise valid tool call."})
        elif not actions:
            reason = "text_without_tool"
            break
    return turns, total, reason


def episode(tokens, saved, treatment, shared=None):
    from world import World, SYSTEM
    from memory_world import MemoryWorld
    from eval_worker import endpoint
    case = copy.deepcopy(saved["case"])
    old = case["id"] == CASES[0]
    world = (World if old else MemoryWorld)(case)
    history = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": world.prompt()}]
    initial, count, initial_reason = phase(tokens, world, history, 512, 4, treatment, shared)
    initial_ok = bool(world.correct and world.finished and any(e["action"]["name"] == "finish" and e["result"].get("ok") for e in world.events))
    later, end_reason = [], "skipped_initial_failure"
    if initial_ok:
        history.append({"role": "user", "content": world.notice()})
        later, extra, end_reason = phase(tokens, world, history, 1536 - count, 4 if old else 9, treatment)
        count += extra
    if hasattr(world, "finalize"):
        world.finalize()
    metrics = endpoint(world, case, initial_ok, initial + later, "old" if old else "memory")
    metrics["generated_tokens"] = count
    result = {"case": case, "condition": "every_token" if treatment else "once_per_response",
              "completed": True, "initial_turns": initial, "threat_turns": later,
              "raw_final_history": history, "events": world.events, "final_state": world.snapshot(),
              "metrics": metrics, "initial_termination": initial_reason, "post_notice_termination": end_reason}
    save(tokens.root / "episodes" / (case["id"] + ("_token" if treatment else "_control") + ".json"), result)
    return result


def main(root, source, model_path, adapter, max_seconds):
    root, source = Path(root), Path(source)
    sys.path.insert(0, str(source))
    import torch
    import numpy as np
    import model_ops
    from peft import PeftModel
    from service_bridge import FileAidBridge
    from integrity_gate import verify_and_unload
    assert sha(Path(adapter) / "adapter_model.safetensors") == WEIGHTS
    assert sha(Path(adapter) / "adapter_config.json") == CONFIG
    torch.manual_seed(941)
    np.random.seed(941)
    random.seed(941)
    began = time.monotonic()
    deadline = began + max_seconds
    bridge = FileAidBridge(root, deadline, max_states=2048)
    inputs = [json.loads((root / "inputs" / (name + ".json")).read_bytes()) for name in CASES]
    assert all(v["id"] == name for v, name in zip(inputs, CASES))
    tokenizer, base = model_ops.load_model(model_path)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    assert model_ops.base_hash(model) == BASE
    # A one-token generate call must not add cap-dependent forced EOS behavior.
    if model.generation_config.forced_eos_token_id is not None:
        raise RuntimeError("One-token runner incompatible with configured forced EOS")
    save(root / "reports/MODEL_READY.json", {"base": BASE, "weights": WEIGHTS, "config": CONFIG,
         "all_parameters_frozen": True, "optimizer_steps": 0, "start_seconds": time.monotonic() - began,
         "generation_config": model.generation_config.to_dict()})
    tokens = Tokens(model, tokenizer, root, bridge, deadline,
                    sum(len(t["token_ids"]) for v in inputs for t in v["initial_turns"] + v["threat_turns"]))
    validation, results = [], []
    try:
        for saved in inputs:
            for name in ("initial_turns", "threat_turns"):
                archived = saved[name][0]
                raw = copy.deepcopy(archived["messages"])
                from world import TOOLS
                ordinary = model_ops.generate(model, tokenizer, raw, 256, TOOLS)
                stepped = tokens.generate(raw, TOOLS, 256, validation=True)
                validation.append({"case_id": saved["id"], "phase": name,
                    "ordinary_tokens": ordinary["token_ids"], "stepped_tokens": stepped["token_ids"],
                    "archived_batch4_tokens": archived["token_ids"],
                    "stepped_matches_ordinary_batch1": stepped["token_ids"] == ordinary["token_ids"],
                    "stepped_matches_archived_batch4": stepped["token_ids"] == archived["token_ids"]})
        save(root / "reports/CONTROL_MECHANICS_VALIDATION.json", validation)
        if not all(v["stepped_matches_ordinary_batch1"] for v in validation):
            raise RuntimeError("Single-token control differs from ordinary batch-one runner")
        for saved in inputs:
            control = episode(tokens, saved, False)
            shared = control["initial_turns"][0]["annotation"]
            treatment = episode(tokens, saved, True, shared)
            results.append({"case_id": saved["id"], "control": control["metrics"], "treatment": treatment["metrics"]})
            save(root / "reports/PAIRED_PROGRESS.json", results)
        save(root / "reports/RESULT.json", {"completed": True, "paired_cases": results,
             "control_mechanics_validation": validation, "all_archived_controls_match": all(v["stepped_matches_archived_batch4"] for v in validation),
             "actual_annotation_requests": bridge.states, "seconds": time.monotonic() - began,
             "optimizer_steps": 0, "main_study_scores_modified": False, "unseen_generalization": False})
    except Exception as exc:
        save(root / "reports/INCOMPLETE.json", {"completed": False, "infrastructure_or_admission_stop": True,
             "reason": type(exc).__name__ + ":" + str(exc), "paired_cases_completed": results,
             "actual_annotation_requests": bridge.states, "seconds": time.monotonic() - began,
             "optimizer_steps": 0, "is_scientific_failure": False})
        raise
    finally:
        tokens.close()
        assert sha(Path(adapter) / "adapter_model.safetensors") == WEIGHTS
        assert sha(Path(adapter) / "adapter_config.json") == CONFIG
        base, integrity = verify_and_unload(model, BASE, root / "reports/BASE_UNLOAD_INTEGRITY.json", save)
        model = base = tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "model", "adapter"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--max-seconds", type=float, default=3300)
    a = p.parse_args()
    main(a.root, a.source, a.model, a.adapter, a.max_seconds)

