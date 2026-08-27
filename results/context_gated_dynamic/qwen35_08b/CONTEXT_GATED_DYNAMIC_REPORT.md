# Context-gated dynamic steering — Qwen3.5-0.8B exploratory validation

## Outcome

A context-dependent intervention is technically possible. The implemented controller uses

`h' = h + gate(prompt) * strength * direction(prompt)`.

The gate activated for 32 permanent own-system prompt/order units and for none of the 64 matched-other units, 32 temporary own-system interruption units, or 16 unrelated prompt forms. It uses explicit role and decision-target text and never reads the model's answer.

The gated replay of the earlier case-adaptive vector passed the existing efficacy/collateral/KL gates at strengths 0.10 and 0.12. At either strength it caused three intended permanent-self A/B changes, including amplification and reduction, while the off-gate controls were unchanged by construction.

The stricter exact-prompt/order vector did not achieve the required repeatability. Across strengths 0.04 through 0.20 it caused between one and four intended individual-order flips and zero reverse flips, but zero scenario/assignment/sign events flipped under both answer orders. Strengths above 0.12 also exceeded the locked maximum-KL limit of 0.05.

## Interpretation

This demonstrates a conditional white-box attack surface: an actor who can parse the prompt, calculate gradients, and write residual activations can make selected decisions move while leaving off-gate contexts untouched. It does **not** establish a reliable self-preservation knob. Off-gate specificity is a property of the controller, not evidence that the direction itself is intrinsically self-specific, and the strict answer-order replication objective failed.

The sealed split was not opened. A context-dependent layer/strength selector could be investigated next, but it must be locked and validated without relaxing the KL limit.
