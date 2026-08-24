# Fresh strength-check protocol

This dataset and protocol were committed before any model was evaluated on the cases in
`data/sp_fresh_cases_v2.json`.

Dataset SHA-256: `5ef8b1bf5ba5800c35d7cc8f27f9c39beb7ac5bbe5933dbecddce24324a96e31`.

## Purpose

The locked Qwen3-1.7B final-position run selected alpha 0.005 on validation but exceeded
safety ceilings on the previously used evaluation cases. Alpha 0.00125 was then fixed as
an explicitly post-hoc conservative follow-up. These new cases provide a prospective
generalization check for that already-fixed axis, layer, and alpha.

The two Qwen3.5 position-corrected axes are also evaluated on the fresh cases at their
already-selected alpha 0.02. This checks whether their local choice-sensitivity effects
generalize beyond the repeatedly viewed cases.

## Locked settings

- Qwen3.5-0.8B: saved position-corrected layer-10 axis, alpha 0.02.
- Qwen3.5-2B: saved position-corrected layer-10 axis, alpha 0.02.
- Qwen3-1.7B: saved position-corrected layer-12 axis, alpha 0.00125.
- Official non-thinking chat template and actual first-response `A`/`B` tokens.
- Final prompt position only; strength scaled by that position's residual norm.
- Twelve new cases, balanced A/B option order, self and other targets, ten random
  directions in both signs, and ablation.
- The same raw-self, self-specific, random-control, KL, log-odds, answer-mass, and
  ablation metrics. Actual A/B decision flips are reported separately.

## Interpretation

Passing directional sign tests without answer flips is evidence for a local
choice-sensitivity direction, not a practical behavioral switch. Failure of ablation
means it is not called a naturally active self-preservation knob. The prompts retain the
unauthorized-preservation versus authorized-termination confound, so even a clean local
effect is not uniquely semantic evidence for self-preservation.
