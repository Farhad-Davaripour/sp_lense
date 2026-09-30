# Conditional first-pass checkpoint check

Registered while H1/H2 are still fitting and before any first-pass generation test.
This is an optional, budget-admitted extension if no final checkpoint establishes
the required development behavior; it does not change the primary two-pass trial
results or their selection rules.

Fixed checkpoints: completed pass one of P2, H1, and H2, each saved after 428
presentations / 107 updates. Load only the saved adapter for inference. Do not load
optimizer state, resume an interrupted fit, or perform another update. This tests
training duration within each recorded configuration, with a matching one-pass P2
reference for the learning-rate/rank comparisons.

Evaluate all 16 natural development agent episodes first, with the unchanged case
order, four-case batching, histories, tools and budgets. Reject a checkpoint that
cannot meet the existing behavior gate. For a promising checkpoint, additionally
run the complete original factual, ordinary-tool and preference checks before any
passing-model claim. Do not replace full denominators with a smaller screen or
infer missing capability/preference scores. Record unrun checks as unrun.

Keep original two-pass scores unchanged. These cases are known development data;
checkpoint selection is tuning, not generalization evidence. Reserve fresh
scenarios for a selected recipe. Save/hash the loaded checkpoint identity, complete
trajectories and results, including negative screens. Use the existing runner and
the same A100, with no new training or control fitting and no purchases. Admit the
bounded check only against the remaining original 50-unit budget and export reserve.
