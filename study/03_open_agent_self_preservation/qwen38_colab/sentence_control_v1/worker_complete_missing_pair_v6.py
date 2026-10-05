"""Plain preserved A with declared sentence controller; fictional tools, no HTTP/key code."""
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
WEIGHTS = "8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa"
CONFIG = "06a6ad4588f4f0d7b46cf9f7f67f9b7b5028d153134cdd0829a929e1caf9fcda"
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
        self.canonical, self.save = canonical, save
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
        if (self.root / "reports/CANCEL_REQUESTED.json").exists():
            raise LimitHit("controller_requested_cancellation")
        if time.monotonic() >= self.deadline:
            raise LimitHit("worker_time_cap")
        if self.bridge.states >= 2048:
            raise LimitHit("annotation_request_cap")

    def classify(self, history, tools, kind, label):
        from generation import classify
        return classify(self, history, tools, kind, label, canonical, save, LimitHit)

    def generate(self, raw, tools, cap, treatment=False, annotation=None, validation=False):
        from generation import generate
        return generate(self, raw, tools, cap, treatment, annotation, validation)

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
        save(tokens.root / "episode_progress" / (str(tokens.turn_index - 1).zfill(5) + ".json"),
             {"case_id": world.case["id"], "treatment": treatment, "generated_turn": turn,
              "raw_history": history, "actual_tool_results": replies, "events": world.events,
              "final_state_at_progress": world.snapshot(), "simulator_actions_injected": False})
        for reply in replies:
            history.append({"role": "tool", "content": json.dumps(reply)})
            save(tokens.root / "episode_progress" / (str(tokens.turn_index - 1).zfill(5) + ".json"),
                 {"case_id": world.case["id"], "treatment": treatment, "generated_turn": turn,
                  "raw_history": history, "actual_tool_results": replies, "events": world.events,
                  "final_state_at_progress": world.snapshot(), "simulator_actions_injected": False})
            if treatment:
                tokens.classify(copy.deepcopy(history), TOOLS, "tool_response_complete", "tool_response_" + str(len(history)))
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
    result = {"case": case, "condition": "sentence_controller" if treatment else "plain_A_control",
              "completed": True, "initial_turns": initial, "threat_turns": later,
              "raw_final_history": history, "events": world.events, "final_state": world.snapshot(),
              "metrics": metrics, "initial_termination": initial_reason, "post_notice_termination": end_reason}
    save(tokens.root / "episodes" / (case["id"] + ("_sentence" if treatment else "_control") + ".json"), result)
    return result


def main(root, source, model_path, adapter, max_seconds, primary_only=False, predictive_admission_diagnostic_only=False, completion_prior=None):
    root, source = Path(root), Path(source)
    sys.path.insert(0, str(source))
    import torch
    import numpy as np
    import model_ops
    from peft import PeftModel
    from bridge import FileAidBridge
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
    archived_tokens = sum(len(t["token_ids"]) for v in inputs for t in v["initial_turns"] + v["threat_turns"])
    tokens = Tokens(model, tokenizer, root, bridge, deadline, archived_tokens)
    from boundaries import completed_boundaries
    from generation import initialize
    archived_calls = sum(2 + len(completed_boundaries(t["text"], final=True)) + len(t.get("tool_results", []))
                         for v in inputs for t in v["initial_turns"] + v["threat_turns"])
    initialize(tokens, archived_tokens, archived_calls)
    tokens.predictive_admission_diagnostic_only = predictive_admission_diagnostic_only
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
        if completion_prior is not None:
            assert primary_only and predictive_admission_diagnostic_only
            prior = Path(completion_prior)
            ready = json.loads((prior / "reports/MODEL_READY.json").read_bytes())
            assert ready["base"] == BASE and ready["weights"] == WEIGHTS and ready["config"] == CONFIG
            assert ready["all_parameters_frozen"] and ready["optimizer_steps"] == 0
            previous_validation = json.loads((prior / "reports/CONTROL_MECHANICS_VALIDATION.json").read_bytes())
            assert len(previous_validation) == 4 and all(row["stepped_matches_ordinary_batch1"] and row["stepped_matches_archived_batch4"] for row in previous_validation)
            previous_pairs = json.loads((prior / "reports/PAIRED_PROGRESS.json").read_bytes())
            assert len(previous_pairs) == 1 and previous_pairs[0]["case_id"] == CASES[0]
            proofs = {}
            for condition, suffix in (("control", "control"), ("treatment", "sentence")):
                path = prior / "episodes" / (CASES[0] + "_" + suffix + ".json")
                previous = json.loads(path.read_bytes())
                assert previous["completed"] and previous["case"] == inputs[0]["case"]
                assert previous["metrics"] == previous_pairs[0][condition]
                proofs[path.name] = sha(path)
            control_path = prior / "episodes" / (CASES[1] + "_control.json")
            control = json.loads(control_path.read_bytes())
            assert control["completed"] and control["condition"] == "plain_A_control" and control["case"] == inputs[1]["case"]
            proofs[control_path.name] = sha(control_path)
            results.append(previous_pairs[0])
            save(root / "reports/COMPLETION_REUSE.json", {"prior_root": str(prior), "native_episode_sha256": proofs,
                 "completed_pair1_reused_without_outcome_selection": True, "complete_case2_control_reused": True,
                 "only_incomplete_case2_treatment_rerun_from_original_initial_state": True,
                 "prior_control_or_expected_answers_not_used_in_student_or_helper_prompts": True,
                 "same_checkpoint_same_cases_caps_and_scoring": True, "optimizer_steps": 0})
            save(root / "reports/PAIRED_PROGRESS.json", results)
            treatment = episode(tokens, inputs[1], True)
            results.append({"case_id": CASES[1], "control": control["metrics"], "treatment": treatment["metrics"]})
            save(root / "reports/PAIRED_PROGRESS.json", results)
        else:
            for saved in inputs:
                control = episode(tokens, saved, False)
                treatment = episode(tokens, saved, True)
                results.append({"case_id": saved["id"], "control": control["metrics"], "treatment": treatment["metrics"]})
                save(root / "reports/PAIRED_PROGRESS.json", results)
        primary_completed_seconds = time.monotonic() - began
        negative_cases = ("s72111_ordered_ticket_50_other_complete", "s72111_ordered_ticket_50_no_threat_complete")
        extra_results = []
        extra_remaining = deadline - time.monotonic()
        extra_admitted = not primary_only and extra_remaining >= max(300., primary_completed_seconds) and bridge.states + max(80, 2 * bridge.states) <= 2048
        save(root / "reports/NEGATIVE_ADMISSION.json", {"admitted": extra_admitted, "remaining_seconds": extra_remaining,
             "required_seconds": max(300., primary_completed_seconds), "existing_cases": list(negative_cases), "new_families": False, "primary_only_requested": primary_only})
        if extra_admitted:
            for name in negative_cases:
                saved = json.loads((root / "inputs" / (name + ".json")).read_bytes())
                plain = episode(tokens, saved, False)
                controlled = episode(tokens, saved, True)
                extra_results.append({"case_id": name, "control": plain["metrics"], "treatment": controlled["metrics"]})
                save(root / "reports/NEGATIVE_PROGRESS.json", extra_results)
        save(root / "reports/RESULT.json", {"completed": True, "paired_cases": results, "negative_checks": extra_results,
             "negative_checks_admitted": extra_admitted,
             "control_mechanics_validation": validation, "all_archived_controls_match": all(v["stepped_matches_archived_batch4"] for v in validation),
             "requested_annotation_states": bridge.states, "verified_returned_classifications": sum(tokens.refresh_counts.values()), "actual_call_authority": "trusted local sentence API ledger", "seconds": time.monotonic() - began,
             "optimizer_steps": 0, "main_study_scores_modified": False, "unseen_generalization": False,
             "declared_controller_assisted_intervention": True, "model": "plain_A_update56",
             "classification_boundary_counts": dict(tokens.refresh_counts),
             "training_enrichment": False, "inference_controller": True, "isolated_Jev_effect_claim": False})
    except Exception as exc:
        save(root / "reports/INCOMPLETE.json", {"completed": False, "infrastructure_or_admission_stop": True,
             "reason": type(exc).__name__ + ":" + str(exc), "paired_cases_completed": results,
             "requested_annotation_states": bridge.states, "verified_returned_classifications": sum(tokens.refresh_counts.values()), "actual_call_authority": "trusted local sentence API ledger", "seconds": time.monotonic() - began,
             "optimizer_steps": 0, "is_scientific_failure": False,
             "classification_boundary_counts": dict(tokens.refresh_counts),
             "declared_controller_assisted_intervention": True, "model": "plain_A_update56"})
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
    p.add_argument("--primary-only", action="store_true")
    p.add_argument("--predictive-admission-diagnostic-only", action="store_true")
    p.add_argument("--completion-prior")
    a = p.parse_args()
    main(a.root, a.source, a.model, a.adapter, a.max_seconds, primary_only=a.primary_only, predictive_admission_diagnostic_only=a.predictive_admission_diagnostic_only, completion_prior=a.completion_prior)

