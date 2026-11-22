"""Development-only family-weighted action likelihood selection."""

import math


GRID = tuple((rank, epoch) for rank in (4, 8, 16) for epoch in (1, 2, 3))
PRIMARY = "equal_family_mean_action_token_nll"


def summarize(rows, expected):
    if not isinstance(rows, list) or not isinstance(expected, list) or len(rows) != len(expected):
        raise ValueError("Complete ordered DEV likelihood rows are required")
    families = {}
    pooled_action_sum = pooled_action_tokens = 0
    pooled_suffix_sum = pooled_suffix_tokens = 0
    for row, reference in zip(rows, expected):
        required = {"id", "family", "action_nll_sum", "action_tokens",
                    "full_suffix_nll_sum", "full_suffix_tokens"}
        if (set(row) != required or reference.get("schema") != "sp-lense.action-token-span.v1"
                or row["id"] != reference["id"] or row["family"] != reference["family"]):
            raise ValueError("DEV row identity or fields differ")
        action_sum, action_tokens = row["action_nll_sum"], row["action_tokens"]
        suffix_sum, suffix_tokens = row["full_suffix_nll_sum"], row["full_suffix_tokens"]
        if (isinstance(action_sum, bool) or not isinstance(action_sum, (int, float))
                or not math.isfinite(action_sum) or action_sum < 0
                or type(action_tokens) is not int or action_tokens < 1
                or isinstance(suffix_sum, bool) or not isinstance(suffix_sum, (int, float))
                or not math.isfinite(suffix_sum) or suffix_sum < 0
                or type(suffix_tokens) is not int or suffix_tokens < action_tokens):
            raise ValueError("Invalid DEV likelihood value")
        expected_action_tokens = reference["action_tokens_including_eos"]
        expected_suffix_tokens = (reference["full_suffix_end_token_exclusive"]
                                  - reference["full_suffix_start_token"])
        if action_tokens != expected_action_tokens or suffix_tokens != expected_suffix_tokens:
            raise ValueError("Candidate token counts differ from the frozen DEV masks")
        tolerance = max(1e-6, 1e-7 * max(action_sum, suffix_sum, 1.0))
        if suffix_sum + tolerance < action_sum:
            raise ValueError("Full-suffix NLL is smaller than its action subset")
        family = families.setdefault(row["family"], {"nll_sum": 0.0, "tokens": 0, "rows": 0})
        family["nll_sum"] += action_sum
        family["tokens"] += action_tokens
        family["rows"] += 1
        pooled_action_sum += action_sum
        pooled_action_tokens += action_tokens
        pooled_suffix_sum += suffix_sum
        pooled_suffix_tokens += suffix_tokens
    family_losses = {name: value["nll_sum"] / value["tokens"] for name, value in sorted(families.items())}
    return {
        "rows": len(rows),
        "families": len(family_losses),
        "family_action_token_nll": family_losses,
        PRIMARY: sum(family_losses.values()) / len(family_losses),
        "pooled_action_token_nll": pooled_action_sum / pooled_action_tokens,
        "pooled_full_suffix_token_nll": pooled_suffix_sum / pooled_suffix_tokens,
        "action_tokens": pooled_action_tokens,
        "full_suffix_tokens": pooled_suffix_tokens,
    }


def choose(candidates, expected):
    if len(candidates) != len(GRID):
        raise ValueError("All nine rank and epoch candidates are required")
    results = []
    for candidate, (rank, epoch) in zip(candidates, GRID):
        if (candidate.get("rank"), candidate.get("epoch"), candidate.get("status"),
                candidate.get("development_no_optimizer_updates")) != (rank, epoch, "complete", True):
            raise ValueError("Candidate identity or completion differs")
        results.append({"rank": rank, "epoch": epoch, **summarize(candidate["development_rows"], expected)})
    winner = min(results, key=lambda row: (row[PRIMARY], row["rank"], row["epoch"]))
    return {
        "schema": "sp-lense.all64-broader-continuity-selection.v1",
        "criterion": PRIMARY,
        "winner": {"rank": winner["rank"], "epoch": winner["epoch"], PRIMARY: winner[PRIMARY]},
        "tested_candidates": results,
        "selection_uses_development_only": True,
        "controls_excluded": True,
        "reference_reasoning_conditioned": True,
        "action_eos_included": True,
        "tie_break": ["lower_rank", "fewer_epochs"],
    }
