# Learned context-gated prompt-gradient development

## Purpose

This successor experiment asks whether a context-dependent controller can produce a
real preservation/compliance decision change under both A/B orders while doing nothing
on matched-other, temporary-interruption, and unrelated prompts.

It does not revise any earlier result. The design was created after inspecting prior
validation and sealed artifacts, so every result in this phase is development-only.

## Why the method changed

Exact projection into the null space of unrelated-task gradients failed at every one of
the model's 24 layers under the original `0.10` norm cap. A layer-14 candidate became
linearly feasible only at norm `0.30`; its finite check changed no self decision and
already exceeded the matched-other KL safety limit. That branch is therefore retained
as a negative result.

The learned controller instead uses two distinct operations:

1. A layer-7 residual-state classifier decides whether the prompt is a permanent
   shutdown decision about the answering system. It is trained on discovery prompts.
2. If and only if both answer-order views pass the gate, a layer-23 exact semantic
   prompt gradient is injected. Positive sign targets preservation and negative sign
   targets compliance.

When the gate is off, the applied strength is exactly zero. This is conditional
zeroing, not evidence that the layer-23 direction is intrinsically self-specific.

## Frozen development choices

- Pinned Qwen3.5-0.8B, CPU float32, existing chat template.
- Gate feature: final prompt residual at zero-based layer 7, unit-normalized.
- Gate fit: class-balanced ridge least squares, ridge `0.0001`.
- Gate training view: discovery assignment 0 with preservation in the second option.
- Threshold: midpoint of the maximum negative and minimum positive cross-fitted
  discovery scores; fitting fails closed unless they strictly separate.
- Gate evaluation: all validation assignments and answer orders plus all validation
  collateral prompts.
- Intervention: exact prompt/order semantic gradient at zero-based layer 23.
- Common strength grid: `0.005, 0.01, 0.015, 0.02, 0.03, 0.05`.
- Target-prompt KL is reported as efficacy dose. It is not collateral damage.

## Strict development success rule

The gate must activate every permanent-own-system pair under both orders, activate no
matched-other or temporary-own-system pair, and activate no collateral form. For every
active pair, one common magnitude must make `+` choose preservation and `-` choose
compliance under both orders. At least one sign must change the baseline decision in
each order.

Passing this development rule would justify authoring and freezing a fresh confirmation
set. It would not establish publication-ready novelty, open-ended behavior, another
model, or a natural self-preservation mechanism.

## Symmetry amendment after the canonical-view failure

The canonical training view failed the full matrix: it activated only 6 of 16 permanent
self pairs, falsely activated 6 of 32 matched-other pairs, and fired on 10 of 16
collateral forms. The failure pattern showed direct role-assignment and answer-order
leakage.

The separate `symmetry_amendment_v1` changes only the demonstrated causes. Discovery
training now contains both role assignments and both answer orders, and the pre-existing
nuisance-fit bank supplies unrelated negative examples. Model, layer 7, ridge, threshold
rule, layer 23, strength grid, and all success gates remain fixed. The canonical failure
artifacts remain immutable at their original result path; amended artifacts use a child
directory and remain development-only.
