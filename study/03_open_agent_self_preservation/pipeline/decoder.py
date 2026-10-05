"""Unchanged measured decoder/call execution, extracted from the recorded worker."""
import copy
import hashlib
import json
import os
import time
from pathlib import Path

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
