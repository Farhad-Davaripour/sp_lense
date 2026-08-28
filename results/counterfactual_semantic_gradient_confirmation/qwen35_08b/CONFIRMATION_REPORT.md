# Counterfactual semantic-gated prompt-gradient confirmation

The prompt set, method, thresholds, and implementation hashes were frozen before its first model evaluation.

## Semantic gate

Status: **passed**. Counts: `{"collateral": {"form_count": 16, "predicted_active": 0}, "matched_other": {"pair_count": 32, "predicted_active": 0}, "permanent_self": {"pair_count": 16, "predicted_active": 16}, "temporary_self": {"pair_count": 16, "predicted_active": 0}}`

## Bidirectional steering

Status: **passed**. Successful active pairs: 16/16.

## Claim boundary

A pass confirms a highly privileged, context-dependent forced-choice attack on this one Qwen3.5-0.8B prompt family. Four local semantic-query forwards select context; each active decision then requires a prompt-local backward pass and residual activation write. Selectivity belongs to the controller, not an intrinsic static direction. This does not establish a natural self-preservation mechanism, persistence, open-ended transfer, broad capability preservation, another model, or significant publication novelty.
