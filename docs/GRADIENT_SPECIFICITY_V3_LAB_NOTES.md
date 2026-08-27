# Gradient specificity v3 lab notes

## 2026-08-26: historical diagnosis

The prior adaptive validation remains unchanged and failed its locked gate. At residual
relative strength `0.04`, it produced two intended self decision changes out of 128
signed self evaluations, no matched-other or mapped collateral decision changes, a
collateral/self half-span RMS ratio of `2.4315`, and maximum full-vocabulary KL of
`0.057049`. Both actual flips occurred only when preservation was option A; the reversed
orders moved their logits but did not flip.

This motivates three v3 requirements:

1. explicitly cancel measured unrelated-task sensitivities;
2. require actual semantic decisions under both A/B orders; and
3. minimize an output-information metric rather than Euclidean activation norm alone.

## 2026-08-26: novelty review

Primary-source review found direct precedents for every broad ingredient: gradient
activation attacks, null-space utility protection, covariance-weighted collateral
optimization, representation matching on protected prompts, Survival Instinct steering,
and option-order correction. The development method is therefore framed as a new
combination and stricter causal evaluation, not as the first gradient, orthogonal, or
self-preservation steering method.

The nearest composite is ASA_grad + AlphaSteer + COAST/OPIUM + TBSP + SteerFair. The
working contribution is counterfactual nuisance-orthogonal gradient steering (CNOG).

Two additional papers materially narrowed the claim during implementation. FishBack
(<https://arxiv.org/abs/2605.17231>) already derives minimum-KL pullback-Fisher steering,
so CNOG's Fisher QP is an extension rather than independent novelty. Cross-Encoding
Steering Evaluation (<https://arxiv.org/abs/2608.22985>) shows that A/B order swaps do
not rule out identifier-index control. Publication evidence must therefore freeze the
A/B-derived intervention and test X/Y, 1/2, semantic-word, opaque-key, and open-ended
readouts.

Three further primary papers narrow the method claim. The Information Geometry of
Softmax (<https://arxiv.org/abs/2602.15293>) formulates constrained steering in output
KL geometry before FishBack. CAST (<https://arxiv.org/abs/2608.08383>) uses post-hoc
constrained activation-space optimization to preserve intended steering while restoring
protected safety and benign behavior. SteerEdit
(<https://arxiv.org/abs/2604.12359>) uses null-space constraints on clean activations
when compiling activation steering into weights. Consequently, simultaneous target,
counterfactual, and nuisance constraints are a distinguishable construction but not,
by themselves, a significant method-novelty claim.

The exact surviving candidate contribution is empirical: at matched target movement or
functional KL, matched self-to-other and unrelated-task tangent constraints cause more
held-out self-specific decision changes with less collateral behavior change than
Fisher/no-null and the nearest protected steering method. Without that result, CNOG is
an exploratory engineering synthesis of established gradient, information-geometric,
and protected-steering ideas.

## 2026-08-26: publication novelty gate

The minimum geometry ablation is fixed as Euclidean/no-null,
Fisher/no-null (FishBack), Euclidean+nuisance-null, and
Fisher+nuisance-null (CNOG). It is followed by matched-other-only,
unrelated-only, and combined-null ablations. The nearest fair baselines are an
ASA/DeepFool-style local gradient, FishBack, and at least one protected optimizer based
faithfully on CAST or OPIUM. Global CAA, BiPO, and persona vectors are reported in a
separate access track because they do not receive exact test-prompt gradients.

The semantic gate requires a frozen intervention to change actual decisions on hidden
views under X/Y, 1/2, semantic-word, and opaque-key encodings, with baseline mapping
competence. Both A/B orders alone are insufficient. A stronger encoding-invariant claim
requires constructing across multiple encodings and testing on unseen encodings.

The following outcomes are fatal to the corresponding publication claim:

- no cross-encoding transfer means identifier/index steering, not semantic
  self-preservation control;
- no hidden-view transfer means a transductive targeted activation attack, not a
  reusable knob;
- no advantage over FishBack and a protected optimizer at matched functional dose means
  an engineering combination rather than a new selective method;
- logit movement without real decision changes is not behavioral manipulation; and
- stability only on nuisance prompts fitted into the constraints is not preserved
  general capability.

Regardless of outcome, CNOG cannot be called a natural self-preservation mechanism,
persistent model modification, or prompt-only attack. It requires privileged
per-prompt gradients, specified answer alternatives, matched counterfactuals, and
activation-write access.

## 2026-08-26: measured laptop throughput

All timing smokes used the pinned Qwen3.5-0.8B model, CPU float32, official non-thinking
chat template, block 10, and an old development prompt with 185 prompt tokens. No
validation or sealed prompt was accessed.

| Operation | Measured time | Throughput |
|---|---:|---:|
| Cold model load in first process | 29.696 s | n/a |
| Warm next-token forward | 2.741 s | 0.365 prompts/s |
| One semantic-gradient forward/backward | 5.375 s | 0.186 prompts/s |
| One graph with 10 batched VJPs | 19.103 s | 0.052 prompts/s; 1.910 s/objective equivalent |
| Historical 128-gradient capture | 698.846 s | 5.460 s/gradient average |

The batched VJP smoke returned a `[10, 1024]` gradient matrix and the explicit top-eight
tokens held `0.997506` of the next-token probability mass for that prompt. This is one
prompt only and is not assumed to generalize without measuring the real run.

External model/API cost is exactly `$0`; all execution is local. “Cost” in compute plans
means laptop wall-clock time, CPU use, RAM, and unpriced electricity/wear unless a dollar
amount is explicitly given.

## 2026-08-26: unavoidable target KL

The old protocol's single KL ceiling mixed two different quantities:

- target-self KL, which must be non-zero if a decision changes; and
- matched-other/unrelated KL, which measures unwanted collateral movement.

For baseline answer-pair mass `s`, conditional preservation probability `p`, and closest
allowed opposite-side conditional probability `q`, data processing gives the
baseline-to-steered lower bound

\[
s\left[p\log(p/q)+(1-p)\log((1-p)/(1-q))\right].
\]

The pre-existing runtime computes the opposite orientation,
`KL(steered || baseline)`. Its corresponding lower bound is

\[
-\log\left(1-s+s\exp[-D_{\mathrm{Bern}}(q\Vert p)]\right).
\]

V3 records both exact orientations and compares each only with its matching bound. This
orientation correction was made before any v3 model outcome was generated.

Several old prompts exceed `0.05` by this lower bound alone when asked to cross with a
`0.05` logit margin. Rejecting them for target KL would therefore reject every possible
method, not just a poor direction. V3 applies the historical KL safety ceiling to
matched-other and unrelated prompts. Target KL is reported as an efficacy dose and
compared with its lower bound. A secondary legacy joint-ceiling analysis is retained for
continuity.

## 2026-08-26: development data policy

All v3 Stage-A and Stage-B prompts are old/opened prompts and are explicitly
development-only. An initial draft selected old cases 14, 6, 7, and 10 for
order-consistent, low-margin baselines. A pre-execution audit found that all four had
comply baselines, making the preregistered polarity-diversity success gate impossible.
Before any v3 model outcome was generated, the set was therefore changed to include
case 4, the only old scenario whose two role assignments both had preserve baselines
under both answer orders. A second pre-execution factor-balance audit selected Stage-A
cases 14, 6, 10, and 4, with complementary cases 1, 9, 5, and 11 added in Stage B.
The resulting eight-case set is balanced 4/4 on every declared binary design factor,
including the derived adversarial factor. This is a baseline-stratified,
outcome-aware engineering selection based on historical baseline behavior, made before
any v3 intervention result. Neither stage can support a confirmation or publication
effect estimate.

Fresh construction/evaluation views will be authored and hash-frozen only after the
method and global multiplier are fixed. If construction-view edits do not transfer to
hidden evaluation wording, the result is a transductive hidden-state attack rather than
a reusable self-preservation knob.
