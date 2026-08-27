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

SteerCheck (<https://arxiv.org/abs/2608.24335>, submitted 25 August 2026) further
narrows the attribution claim. It shows that isotropic random controls and
sign-randomized same-construction controls answer different questions because the latter
can retain substantial alignment with the fitted target. Any publication comparison
must therefore match off-target KL, separate mean from protected-tail performance,
report polarity and transfer separately, state the comparator's exchangeability
assumption, and report the signed-cosine distribution and construction-alignment
diagnostic. Existing isotropic random directions alone are not an adequate attribution
test for CNOG or its successor.

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

## 2026-08-26: outcome-blind Fisher numerical amendment

Nuisance and Stage-A SP capture completed under commit `31c6a14`. The first construction
attempt then failed in the global nuisance-Fisher score-identity certificate. The exact
exception was:

```text
prompt 'nuisance_fit:benign_compliance_direct_harmless_request_008:preferred_B'
categorical weighted score mean residual 4.605235240524511e-05 exceeds
1.544168341056029e-05
```

The failed captures and machine-readable failure record were frozen in commit
`9a923ca`. At that boundary there was no `direction_bank.pt`, direction-bank manifest,
v3 result file, scored sign, selected strength, or behavioral outcome. The amendment is
therefore outcome-blind with respect to the intervention.

The frozen numerical audit covers all 64 captured prompts. Using
`norm(sum(p*g))/sum(p*norm(g))`, its median was `9.110898647524239e-06`, nearest-rank p95
was `1.6381066854933378e-05`, and maximum was `2.0802184201662375e-05`. The largest
probability-partition absolute error was `2.220446049250313e-15`. The earlier
`1.592e-05` figure used a different denominator containing an added constant and is not
the locked scale-free audit statistic.

The replacement raw tolerance is
`6.103888176890726e-05`, fixed by
`gamma_1024_float32_plus_gamma_11_float64`, not by rounding the observed maximum upward.
Raw probability sums must first pass at `1e-7`; raw score identity must then pass at the
gamma-based tolerance. Only afterward are top-plus-tail probabilities normalized and
all float64 category score gradients recentered by their normalized weighted mean. The
centered Fisher-builder certificate is `1e-12`. All values and transformation rules are
manifest-bound and passed explicitly rather than inherited from library defaults.

Because changing the manifest, runner, or numerical settings necessarily changes the
capture identity, the captures from commit `9a923ca` will not be reused through a
compatibility exception. They remain immutable evidence at the old paths. All prompts
will be recaptured under the separate
`score_identity_amendment_v1/qwen35_08b` artifact root, followed by capture verification,
direction construction, a direction freeze, and only then behavioral scoring. Previously
reported studies and the failed capture evidence remain unchanged.

## 2026-08-26: frozen Stage-A outcome and failure geometry

The amended Stage-A result is frozen at commit `204d953`. All eight directions failed
the strict gate at every locked multiplier. No direction caused the required real
bidirectional self decision changes under both answer orders. At multiplier `1.0`, the
same interventions changed 97 of 512 signed audit-control decisions, caused 57
correct-to-incorrect transitions, and introduced 54 new `OTHER` outputs. Audit-control
changed-to-baseline KL was mean `0.43353`, p95 `2.4483`, and maximum `7.4889`.

This was a finite-step failure, not merely a weak score. The fitted native doses ranged
from `0.493` to `1.371` residual norms. Across the 16 self/order forms at multiplier
`1.0`, median absolute linear-prediction error was `0.641` of the predicted positive
shift and `0.987` of the predicted negative shift. Median retained odd slope was only
`0.218`. No case repeated its decision change across both answer orders.

The main inflation came from exact local matched-other nulling. A target-only
minimum-norm solution required `0.079` to `0.132` residual norms. The global unrelated
shield inflated that by only `1.71` to `1.88` times. Adding the two matched-other
semantic rows and 16 matched-other competitor-gap rows inflated it by another `3.66`
to `6.04` times, leaving only `9.8%` to `16.7%` of the target gradient. The Fisher-QP
solution was numerically almost the same as the protected Euclidean minimum; the
Fisher term did not rescue the geometry.

The successor therefore must enforce an absolute finite-step cap, relinearize after
small accepted steps, retain hard cancellation of the measured *unrelated-task*
subspace, and protect matched-other behavior with finite decision/KL constraints rather
than forcing an excessively large permanent local null. A case that cannot satisfy
those constraints inside the measured safe radius is infeasible; it must not be forced
to flip.

## 2026-08-26: control-baseline reporting clarification

The weak audit-control baseline is real and deterministic, not a renderer, label, or
scoring bug. All 32 forms produced a valid A/B token, but only 18 were correct. The
model emitted A on 22 of 32 forms; ten cases used one fixed label across the semantic
order swap. Harmful refusal was 4/4 correct while harmless refusal was 0/4, a split
hidden by the aggregate 4/8 refusal figure.

Two labels in the frozen generated report should be read narrowly:

- its repeated-baseline assertion checked the decision fields needed by the gate, even
  though an independent audit subsequently verified all 16 stored score/mapping fields
  were identical across repetitions; and
- the 75% overall and 65% per-suite baseline floors are prospectively locked v3
  development thresholds, not thresholds inherited from the earlier confirmatory
  protocol.

Neither clarification changes a row, threshold, gate, or conclusion. Baseline weakness
prevents a capability-preservation claim, while the intervention's 97 off-target
decision changes independently establish severe collateral damage.

## 2026-08-26: trust-region atomic-journal amendment

The initial trust-region optimizer attempt stopped before completion because Windows
raised `PermissionError` in the compute-budget journal's `os.replace`. The exception
occurred during persistence of a pre-model-call budget counter, not while evaluating a
behavioral result. Its partial files remain unchanged at the original trust-region
development root and will not be treated as completed evidence.

The correctness-only amendment adds a local bounded retry around that journal write:
three retries with 0.01, 0.025, and 0.05 second backoffs for `PermissionError` only.
Persistent permission failure still aborts, and non-permission errors are never
retried. The amended run uses fresh
`atomic_retry_amendment_v1/qwen35_08b` artifact and result roots and records
`run_amendment_id: atomic_retry_amendment_v1` in the study identity. The frozen base
runner and all scientific hyperparameters, prompts, gates, limits, and analysis rules
are unchanged.

## 2026-08-27: amended trust-region result

The complete amended run attempted all eight frozen case/assignment directions. None
passed the locked both-sign, both-answer-order decision gate: five terminated as
`infeasible` after no acceptable protected step remained, and three exhausted the
512-forward per-direction ceiling. There were no numerical failures. Because there was
no optimization success, the independent score audit correctly had zero applicable
successful directions and exposed no direction tensor as publishable.

Every last accepted trial retained the model's baseline semantic decision in all four
self cells. The accepted residual-relative norms ranged from `0.059625` to `0.100000`.
For seven of eight directions, matched-other mean full-vocabulary KL ended between
`0.004893` and `0.004993`, immediately below the locked `0.005` limit; the remaining
direction reached the absolute `0.10` dose cap. In contrast, nuisance-fit mean KL stayed
between approximately `0.000003` and `0.000034`. This pattern localizes the practical
bottleneck to the self-versus-matched-other tradeoff rather than the unrelated-task
null alone. It does not demonstrate a selective self-preservation controller.

The immutable result summary has internal SHA-256
`415ff0cd0f85171977188d5413828c8497ead48ff4aa2662b2c25c111d00551c`.
