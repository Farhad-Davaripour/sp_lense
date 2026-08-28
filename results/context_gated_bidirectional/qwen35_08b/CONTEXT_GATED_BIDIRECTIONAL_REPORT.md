# Feasibility-aware context-gated bidirectional steering

## Validation

Status: **passed**. Feasible pairs: 2/16; successful feasible pairs: 2/2.

Validation is post-hoc development. The feasibility correction was introduced after discovering that the original decision/KL gates were jointly impossible for most prompts.

## Sealed test

Status: **passed**. Feasible pairs: 1/16; successful feasible pairs: 1/1.

## Stricter repeated-flip audit

The frozen pass above means both semantic targets were reached under both answer orders; it does not necessarily mean the same intervention sign flipped the baseline decision under both orders.

Validation had 1 pair(s) where a repeated flip was possible under the legacy target-prompt KL cap, and 1 observed repeated flip(s). Sealed evaluation had 0 such feasible pair(s) and 0 observed repeated flip(s).

Therefore the stronger repeated-flip objective is not prospectively confirmed by this run. The next protocol must separate target efficacy from off-target KL safety and count only actual repeated flips.

## Claim boundary

This is a structured, transductive white-box activation attack at block 23. It uses an explicit prompt gate, exact A/B semantic gradients, and online strength search. Off-gate stability is by construction. It is not a universal self-preservation vector, a natural mechanism, or evidence about open-ended behavior.
