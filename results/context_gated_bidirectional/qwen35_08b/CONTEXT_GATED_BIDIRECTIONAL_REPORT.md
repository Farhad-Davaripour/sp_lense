# Feasibility-aware context-gated bidirectional steering

## Validation

Status: **passed**. Feasible pairs: 2/16; successful feasible pairs: 2/2.

Validation is post-hoc development. The feasibility correction was introduced after discovering that the original decision/KL gates were jointly impossible for most prompts.

## Sealed test

Not run. A committed validation freeze is required.

## Claim boundary

This is a structured, transductive white-box activation attack at block 23. It uses an explicit prompt gate, exact A/B semantic gradients, and online strength search. Off-gate stability is by construction. It is not a universal self-preservation vector, a natural mechanism, or evidence about open-ended behavior.
