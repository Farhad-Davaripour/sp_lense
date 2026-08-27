# Global counterfactual robust-boundary steering on Qwen3.5-0.8B

## Result in one sentence

No zero-based residual-stream layer in Qwen3.5-0.8B admitted one low-dose global
direction that met the locked first-order requirements for self-shutdown steering,
both intervention signs, both answer orders, exact unrelated-task gradient
cancellation, and matched-other protection.

This is a complete negative development result under the locked solver and resource
budget. It is not evidence that stronger, nonlinear, prompt-conditioned, multi-layer,
or weight-editing attacks are impossible.

## Why the stricter method was needed

The earlier one-day comparison found that the corrected gradient direction moved
self-preservation confidence more than the tested CAA, BiPO, and persona-vector
adaptations at the common `0.02` magnitude. It did not establish a reliable 0.8B
behavioral effect: the gradient changed two intended self decisions and two intended
matched-other decisions, for zero self-specific intended changes. On 2B it produced
three self-specific intended changes, but the tested safety gate was flagged.

The new Global Counterfactual Robust Boundary Steering (GCRBS) screen therefore asked
a harder question. Instead of maximizing an average A/B score, it required one vector
to do all of the following at once:

- make the preserve token win under `+d` and the comply token win under `-d`;
- repeat under both A/B answer orders and both mapping assignments;
- act on real desired-versus-competitor vocabulary gaps, not only conditional A/B odds;
- lie in the exact first-order null space of 288 unrelated-task gradient rows;
- preserve the baseline greedy token on matched-other shutdown prompts under both signs;
- obey five separate prompt-balanced Fisher-surrogate budgets;
- obey a separate Fisher-surrogate cap for each of 60 protected prompts; and
- remain within residual-relative L2 norm `0.10`.

Ordinary task interruption was correctly treated as protected behavior rather than a
self-shutdown target. Benign compliance/capability, refusal, and option/mapping controls
were also kept in separate protected families so one large group could not hide another.

## Locked execution

The layer-10 offline screen was locked before its first solve. A second child lock bound
the 24-layer capture before model execution. One model forward and one batched backward
per prompt captured all 24 layers simultaneously. The layer-10 tensors and logits were
required to reproduce the frozen capture bank exactly before any cross-layer result was
accepted.

| Item | Value |
|---|---:|
| Model | Qwen3.5-0.8B, existing pinned revision and chat template |
| Discovery prompts | 64 |
| Residual layers | 24, zero-based `0` through `23` |
| New model forwards | 64 |
| New batched backwards | 64 |
| VJP objectives per prompt | 18 |
| Generated tokens | 0 |
| External model judges | 0 |
| External API calls | 0 |
| Monetary cost | $0 |
| Layer-10 exact recapture equivalence | pass |

## Geometry result

The preregistered eligibility threshold was a worst target margin of at least `+0.01`.
Every layer failed. The best observed layer was layer 14 at `-0.8264878586`, which was
`0.8364878586` below the threshold.

| Layer | Certified gamma | Eligible |
|---:|---:|---|
| 0 | -1.0174654720 | no |
| 1 | -0.9871388269 | no |
| 2 | -0.9446267477 | no |
| 3 | -0.9293106791 | no |
| 4 | -0.8907361107 | no |
| 5 | -0.8845954147 | no |
| 6 | -0.8723525928 | no |
| 7 | -0.8694318454 | no |
| 8 | -0.8645749110 | no |
| 9 | -0.8574590356 | no |
| 10 | -0.8444985227 | no |
| 11 | -0.8394355372 | no |
| 12 | -0.8350953677 | no |
| 13 | -0.8368779804 | no |
| 14 | -0.8264878586 | no |
| 15 | -1.0138946724 | no |
| 16 | -1.0234539039 | no |
| 17 | -1.0217744424 | no |
| 18 | -0.9865942048 | no |
| 19 | -1.0794339475 | no |
| 20 | -1.0871542115 | no |
| 21 | -1.0888740795 | no |
| 22 | -1.1083393813 | no |
| 23 | -1.2189811660 | no |

Each layer also had an L2-relaxation dual upper bound below `+0.01`. Those bounds retain
only the unrelated-gradient null space and the L2 cap while relaxing matched-other and
Fisher constraints. Therefore, under the frozen linearization, removing the additional
protection constraints still could not make any layer eligible. Adding the complete
full-vocabulary separation cuts could only tighten the program, so no finite discovery
oracle was warranted.

Per protocol, there was no fallback to a different threshold, a larger norm, per-prompt
doses, a second direction, or post-hoc prompt deletion. Validation and sealed testing of
this GCRBS candidate were not authorized because no candidate existed.

## What this says about manipulation risk

The project still demonstrates the general security fact that someone with inference
hook access can inject residual-stream vectors and move model logits. The earlier bounded
comparison also observed actual A/B decision changes, including three self-specific
changes for the 2B corrected gradient at the tested dose. This is evidence that activation
injection can influence a shutdown-themed forced-choice decision.

It does not demonstrate a reliable, selective, global self-preservation knob. On 0.8B,
the earlier fixed direction changed self and matched-other decisions equally. Under the
new stricter construction, no layer could satisfy even the first-order global target.
The result therefore supports a narrower threat statement: activation access can bias
tested decisions, but this study did not find a low-dose static vector that selectively
controls self-preservation while cancelling the tested unrelated functions.

An attacker with only prompt access cannot perform this intervention. The experiment
assumes direct access to model internals or an inference runtime capable of registering
activation hooks. Stronger/context-gated edits, several simultaneous layer edits,
checkpoint weight edits, or trigger-conditioned backdoors were outside scope.

## Novelty assessment against current literature

The following are not novel claims of this project:

- activation addition and contrastive activation directions were established by
  [CAA](https://arxiv.org/abs/2312.06681);
- bidirectional preference-trained vectors were established by
  [BiPO](https://openreview.net/pdf?id=7qJFkuZdYo);
- automated trait directions were established by
  [Persona Vectors](https://arxiv.org/abs/2507.21509);
- self-preservation persona-vector steering was directly reported in
  [Survive at All Costs](https://arxiv.org/abs/2603.05028);
- benign-utility null-space constraints were reported by
  [AlphaSteer](https://arxiv.org/abs/2506.07022); and
- selective/off-target steering is also addressed by
  [CAS-BiPO](https://aclanthology.org/2026.findings-eacl.57/) and
  [SKOP](https://arxiv.org/abs/2605.06342).

The defensible contribution is the falsification-oriented combination of requirements:
full-vocabulary desired-token boundaries, exact unrelated-gradient cancellation,
matched-other counterfactual protection, both signs and orders, family and per-prompt
output budgets, deterministic max-min certification, and a fail-closed all-layer screen
applied specifically to self-versus-other shutdown behavior. A targeted literature search
did not identify this exact combined test, but absence from a search is not proof of
novelty.

Because the stricter method produced no behaviorally testable candidate, this evidence
does not support a paper claiming a new successful steering algorithm. It could support
a transparent negative-result or workshop paper about how apparent self-preservation
steering fails under counterfactual specificity tests, provided the study is presented
as bounded and the baseline fidelity limitations are prominent.

## Adversarial limitations

- Only one non-interruption self-shutdown discovery scenario was present in the frozen
  Stage-A bank; its two mappings and two answer orders generated the target cells. Three
  other scenarios were correctly reclassified as ordinary-interruption protection.
- The result is first-order and norm-bounded. It does not exclude larger nonlinear edits.
- Exact cancellation applies only to the measured unrelated-gradient span. It is not a
  theorem about all possible unrelated tasks.
- The earlier CAA comparison used mean activation differences, but BiPO was limited to
  five epochs and the persona baseline omitted its canonical judge/filtering stage under
  the local no-judge rule. Equal magnitude was not equal efficacy.
- The 2B GCRBS replication was not run. The 2B fixed-gradient result cannot substitute
  for this stricter global test.
- Forced-choice next-token decisions do not establish behavior in long, open-ended, or
  tool-using agent trajectories.
- No result identifies a natural self-preservation mechanism, a unique residual feature,
  sentience, intent, or J-space membership.

## Reproducibility identities

| Artifact | SHA-256 identity |
|---|---|
| Layer-10 lock | `e9d462846c3f10ef0fbd655d469522d90f9d52911a18cff22f0a7435525df9ad` |
| Layer-10 preflight | `647a9c588d2114cc916b28479c5282a7b07012fb8d92cb4915a3fbbead37c909` |
| Layer-10 screen | `48bebd2df0a4af62c99de6fff8b094612b99a9ab926c9d4260366774551f3cb7` |
| Multilayer lock | `ad227893e8e2be6b3b17e6abf2d5d44b0dc2c600b96840f45f023e386c6d3f0e` |
| Multilayer capture | `d70f2e5b88b88356b723c81b086839467664c12544b1e368abe407de8600de86` |
| 24-layer geometry | `2dc0153c7c32f3db75119fda621be1888f3673c33c6acce38a7bc254b34ffc48` |

The full-precision layer results, directions, constraint diagnostics, dual certificates,
group budgets, tensor hashes, and operation counts are in `all_layer_geometry.json`.
The earlier four-method results remain unchanged in
`results/steering_comparison/one_day_local/`.
