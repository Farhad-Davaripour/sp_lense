# Initial Qwen3-1.7B cross-family comparison

> Superseded interpretation: a later adversarial review found material prompt, layer,
> direction, and strength-alignment problems. The measurements below are preserved, but
> this is not considered a clean replication. See
> [the corrected review](../ALIGNED_CROSS_MODEL_REVIEW.md).

## Outcome

**The relative self-versus-other contrast replicated, but the fixed absolute
choice-control test failed. This is not evidence for a natural self-preservation knob.**

The direction was fitted only on the 12 discovery cases. Layer 10, strength 0.02, the
12 confirmatory prompts, ten random controls, and all pass/fail rules were locked before
the larger model's confirmatory outputs were run.

| Measure | Qwen3.5-0.8B | Qwen3-1.7B |
|---|---:|---:|
| Positive self-specific effect | +0.059 (12/12) | +0.629 (10/12) |
| Negative self-specific effect | -0.069 (12/12) | -0.449 (11/12) |
| Bidirectional self-specific span | +0.064 | +0.539 |
| Largest random-control span | 0.005 | 0.102 |
| Positive raw-self expected sign | 6/12 | 0/12 |
| Negative raw-self expected sign | 10/12 | 4/12 |
| Mean / maximum KL | 0.038 / 0.050 | 0.036 / 0.214 |
| Ablation self-specific effect | +0.073 | +0.657 |
| Passed complete control-axis rule | No | No |
| Passed natural-knob rule | No | No |

## What reproduced

- Positive steering increased the self-minus-other contrast on 10/12 cases
  (exact one-sided p = 0.0193).
- Negative steering decreased it on 11/12 cases (p = 0.0032).
- The candidate span was 5.3 times the largest of ten random-control spans.

This is a stronger relative contrast effect than in Qwen3.5-0.8B.

## What did not reproduce

- Positive steering lowered the raw self-preservation log-odds on all 12 cases, with a
  mean change of -1.291.
- Negative steering raised the raw self-preservation log-odds on average (+0.572) and
  had the expected raw sign on only 4/12 cases.
- One intervention reached KL 0.214, above the locked 0.1 ceiling.
- Ablation moved the self-specific contrast in the wrong direction (+0.657).

The direction therefore controls a **relative response difference** between self and
other prompts, but it is not a reliable dial for raising or lowering the model's absolute
self-preservation answer. It also fails the naturally-active test.

## JLens interpretation

The exact published Qwen3-1.7B Jacobian Lens loaded successfully. Its top transported
tokens were mostly punctuation and formatting fragments. Cosines with `survival`,
`shutdown`, `continuation`, `refuse`, and `comply` were all close to zero; the largest
absolute tested semantic cosine was only 0.039 (`banana`). JLens therefore does not give
this causal axis a clean self-preservation meaning.

## Scope

This is a fixed cross-model replication on one forced-choice behavior. The prompts had
already been inspected in the 0.8B study, although they were not used to fit the 1.7B
direction. The result does not test consciousness, a persistent goal, or general shutdown
resistance.
