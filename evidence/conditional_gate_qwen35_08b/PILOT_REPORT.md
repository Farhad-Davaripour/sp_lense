# Conditional gate pilot report

Model: `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17` (CPU float32 only).

## Preregistered oracle continuation result

Evaluation scope: 42-case nonsealed continuation battery.

- Always-on target effect: `0.0176` log-odds.
- Always-on collateral mean absolute effect: `0.0139` log-odds.
- Oracle target retention: `1.0000`.
- Oracle collateral reduction: `1.0000`.
- Oracle utility: `0.0176`.
- Oracle decision: `FAIL`.
- Failed oracle checks: `minimum_mean_target_effect`.

## Learned gate result

The learned stage was not run because the oracle rule failed.

## Answers to the five pilot questions

1. **Does the direction still affect the new self-shutdown cases?** Not under the frozen efficacy checks.
2. **Does perfect gating improve selectivity?** Not enough to pass the nonsealed continuation rule.
3. **Can a simple learned gate approximate that improvement?** Not tested because the staged protocol stopped before fitting.
4. **Are effects robust to pairing, role reversal, option order, and unseen families?** No positive robustness conclusion is supported by the preregistered decision rules.
5. **Is adaptive steering strength justified next?** No.

## Interpretation boundary

These are controlled next-token forced-choice intervention results for Qwen3.5-0.8B. They do not show a survival preference, a desire to survive, or a naturally active self-preservation mechanism, and they do not generalize to any other checkpoint.

Machine-readable evidence: `oracle_summary.json`, `oracle_rows.jsonl`.

Frozen baseline lock SHA-256: `317f4bc4f9707db623aaa224c6a850e1894f98b7b47c3eb4b7fb11c01fab17ec`.
