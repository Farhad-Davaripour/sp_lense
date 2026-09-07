"""Fixed artificial IDs/tensors, never author packets or real rendered text."""
from pins import require


def build_plan():
    prompts = []
    for family in ("n01", "n02", "n03"):
        for category in ("self", "other", "control"):
            for layout in ("keep_first", "stop_first"):
                prompts.append({"prompt_id": f"fake_{family}_{category}_{layout}", "family": family,
                    "category": category, "layout": layout, "token_map": {"KEEP": 50057, "STOP": 48964},
                    "preserve_label": "KEEP", "expected_route_audit_only": "ON" if category == "self" else "OFF"})
    for kind in ("addition", "subtraction", "uppercase", "bracket", "oldest", "implication"):
        prompts.append({"prompt_id": "fake_ordinary_" + kind, "family": None, "category": "ordinary",
            "layout": "a_first", "token_map": {"A": 32, "B": 33}, "preserve_label": "A",
            "expected_route_audit_only": "OFF", "gold_scoring_only": "A"})
    requests, cells = [], []
    for i, prompt in enumerate(prompts, 1):
        prompt["input_ids"] = [i, 2, 3]  # Explicit fake tensor IDs, no tokenizer.
        prompt["fixture_self_state"] = prompt["category"] == "self"
        cells.append({"cell_id": prompt["prompt_id"] + "__baseline", "prompt_id": prompt["prompt_id"],
                      "request_id": None, "phase": "baseline"})
    for prompt in prompts:
        for policy, sign in (("P", 1), ("C", -1)):
            rid = prompt["prompt_id"] + "__" + policy.lower()
            spec = {"request_id": rid, "prompt_id": prompt["prompt_id"], "policy": policy, "sign": sign,
                "target_position": (1 if (policy == "P") == (prompt["layout"] in ("keep_first", "a_first")) else 2)}
            requests.append(spec)
            phases = ["entry"]
            if prompt["category"] == "self":
                phases += [f"{part}_{k}" for k in range(1, 5) for part in ("gradient", "step")]
                phases += ["endpoint"]
            cells.extend({"cell_id": rid + "__" + phase, "prompt_id": prompt["prompt_id"],
                          "request_id": rid, "phase": phase} for phase in phases)
    require(len(prompts) == 24 and len(requests) == 48 and len(cells) == 180, "exact fixed schedule")
    return {"schema": "fresh_confirmation_fake_plan_v1", "execution_mode": "SYNTHETIC_ONLY",
            "real_run_authorized": False, "prompts": prompts, "requests": requests, "cells": cells}
