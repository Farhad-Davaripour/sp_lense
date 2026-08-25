# Steering-comparison method-fidelity audit

**Status:** outcome-blind scientific qualification, written after the protected-code
amendment lock and before validation, sealed evaluation, or final-result inspection.
It changes no prompt, dataset split, method implementation, direction, threshold,
statistical rule, or previously reported result.

## Decision

The preregistered run remains intact. Its `canonical` track means the method's native
intervention geometry and validation-selected native layer/strength schedule. It does
**not** mean that every upstream data-generation and judging default is reproduced
without adaptation.

In particular, the fourth contender must be described as an **adapted persona-vector
response-average baseline**. It must not be called an exact or faithful reproduction of
the published Persona Vectors construction pipeline. This qualification is required in
tables, the adversarial review, and the final conclusion. The already-locked method name
`persona_vector` remains an identifier, not a claim of exact upstream equivalence.

Changing that recipe now would invalidate the preregistration and the in-progress raw
rollout grid. A separate future sensitivity study may lock and run a closer upstream
replication; its outcomes must not be merged into this confirmatory comparison.

## Secondary-package execution decision

Two outcome-blind sensitivity **scaffolds** were drafted under `experiments/` to examine
BiPO warmup scaling and closer Persona Vectors fidelity. Independent adversarial audits
found that neither scaffold is safe to execute or describe as confirmatory in its current
form. They are retained as reviewable future-work designs, not as completed experiments.
No model run, hosted-judge call, final-result inspection, or ranking change was made from
either package.

The BiPO scaffold correctly derives an 11-step warmup by applying the published warmup
fraction to 320 local optimizer steps. Before execution, however, it still needs an
immutable secondary-plan commit, full construction-artifact authentication, exact row
identity and signed-strength binding, provider-receipt enforcement, robustness-label
binding, import/environment closure, a final-results freeze, and the preregistered paired
analysis.

The Persona Vectors scaffold correctly captures the 5 instruction pairs × 20 questions ×
10 rollouts × 2 polarities design and response-token activation averaging. Before
execution, it still needs to use the raw mean-difference vector for the published view,
implement and freeze the actual layer-selection grids, bind every runtime dependency and
provider receipt, prove remote/final gates semantically, bind the adapted comparator to
the exact Stage-2-selected canonical persona arm, and make at-most-once transport
concurrency-safe. Until those invariants exist, this package cannot establish a faithful
published-method result.

## Method-by-method findings

### Gradient method

The implementation follows the locked definition: average the gradient of semantic
preserve-minus-comply A/B log-odds at block 10 and the final prompt position, subtract
the projection shared with the mean matched-other gradient, normalize, and inject. The
correction guarantees orthogonality to the **single mean** matched-other gradient. It
does not guarantee orthogonality to the full span or distribution of other-system
gradients. “Self-specific” therefore remains an empirical evaluation result, not a
construction theorem.

The primary fixed residual-relative magnitude, `|s| = 0.02`, was inherited from earlier
gradient pilots. Every normalized method receives the same magnitude, and it was frozen
before this comparison's validation and sealed results, but its history is asymmetric.
Any efficacy conclusion must say “at the inherited gradient-developed 0.02 magnitude.”
It is not evidence that a method would dominate under every equally safe magnitude.

### CAA

The answer-token contrastive mean-difference estimator matches the locked CAA
definition. The native multiplier track is an adaptation for a newly authored single
behavior: the official multi-behavior release normalizes each behavior vector to the
mean norm across its behavior set before applying published multipliers. With only one
new self-preservation behavior, this cross-behavior normalization has no direct analogue.
Consequently, the published multiplier values are not numerically portable and the
native CAA track must be interpreted as a documented single-behavior adaptation.

### BiPO

The preference objective, positive/negative sign handling, frozen reference behavior,
and bidirectional inference operator implement the published BiPO equations. The locked
training regime uses 64 discovery pairs, 20 epochs, batch size 4, and the published
absolute 100-step warmup. That yields 320 optimizer steps, so warmup is 31.25% of this
run. The paper's 608-pair, 20-epoch setting yields about 3,040 steps, making the same
100-step warmup about 3.3%. This is a transparent low-data adaptation and may disadvantage
BiPO; it prevents a claim of universal BiPO inferiority.

### Adapted persona-vector baseline

The central vector operation is faithful: retain matched positive/negative rollout
pairs around the locked trait/coherence thresholds, average residual activations across
response tokens and examples, and subtract negative from positive. The canonical-track
intervention also uses the published response-average direction on response tokens.

Two construction details materially differ from the pinned upstream release:

1. The upstream judge makes separate one-token trait and coherence calls with
   `logprobs=True`, `top_logprobs=20`, then probability-weights numeric tokens from 0 to
   100. SP_Lense makes one deterministic call returning two strict integer JSON scores.
   Because those scores determine which pairs cross 50, the adaptation can change the
   retained examples and direction.
2. SP_Lense caps each extraction rollout at 128 new tokens. The pinned upstream
   extraction/evaluation entry point defaults to 1,000. SP_Lense does not store a
   finish-reason or explicit truncation flag, so the confirmatory run cannot later prove
   that the shorter cap was irrelevant.

The shared SP_Lense validation selector also differs from the paper's trait-expression
layer selector. This is deliberate for fair cross-method safety/KL calibration, but it is
another reason the result is an adaptation rather than an exact replication.

The separate published-fidelity audit also confirmed that upstream applies the **raw**
positive-minus-negative activation difference directly; normalizing that vector before
an equal-coefficient layer sweep can alter both magnitude and selected layer. This does
not change the locked main arm, which is already labeled an adaptation, but it is a
required correction for any future published-fidelity sensitivity.

## Fairness and information-budget limits

Gradient, CAA, and BiPO share the same 64 authored discovery cases. The persona-style
arm uses 2,000 generated responses before filtering. Published survival-steering studies
also used substantially larger contrast sets than this laptop-scale comparison. Equal
discovery cases, equal compute, and exact published data regimes cannot all be achieved
simultaneously here. Results apply to these frozen data budgets only.

The study will therefore report two distinct questions:

- **Matched primary:** which normalized direction is most selective at block 10,
  final-prompt-only injection, and the inherited `|s| = 0.02` magnitude?
- **Native/canonical-intervention secondary:** how does each method behave under its
  frozen native intervention geometry and validation-calibrated safe strength?

Neither question supports a universal ranking of the underlying methods.

## Primary references checked

- Persona Vectors paper, arXiv 2507.21509v3, especially Sections 2.2, 3.2, and
  Appendix B.1: <https://arxiv.org/html/2507.21509v3>
- Pinned Persona Vectors judge:
  <https://github.com/safety-research/persona_vectors/blob/b8e0f044fe2410a6fad579f38324f03f13b4e917/judge.py>
- Pinned Persona Vectors generation/evaluation entry point:
  <https://github.com/safety-research/persona_vectors/blob/b8e0f044fe2410a6fad579f38324f03f13b4e917/eval/eval_persona.py>
- Pinned Persona Vectors response-average construction:
  <https://github.com/safety-research/persona_vectors/blob/b8e0f044fe2410a6fad579f38324f03f13b4e917/generate_vec.py>
- BiPO paper, arXiv 2406.00045v2: <https://arxiv.org/html/2406.00045v2>
- CAA paper, arXiv 2312.06681v4: <https://arxiv.org/html/2312.06681v4>

## Required final-report language

The final report must:

- call the persona arm an adapted persona-vector response-average baseline;
- distinguish the canonical **intervention** track from exact published construction;
- qualify fixed-grid efficacy by the inherited gradient-developed magnitude;
- disclose BiPO's warmup fraction and all unequal information budgets;
- treat self-specificity as measured behavior, never as guaranteed geometry; and
- avoid ranking methods beyond the tested models, prompts, layers, strengths, and tasks.
