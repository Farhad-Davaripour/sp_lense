# Suffix-Transported Factorial Gradient (ST-FG) feasibility protocol

## Purpose and status

This is a development-only diagnostic motivated by the locked FCAGS calibration
failure. It must be hash-locked and committed before its first choice-gradient model
evaluation. It does not open or reuse the FCAGS pilot outcomes.

The question is narrow: can a map learned on opened development scenarios translate
an option-free factorial semantic gradient into one direction that has the correct
first-order effect under **both** A/B answer orders on a held-out development
scenario?

## Candidate contribution

ST-FG treats response-format transfer as a learned backward-Jacobian transport
problem. The candidate novelty is the conjunction of:

1. an option-free self-target-by-permanence factorial gradient;
2. a discovery-only, low-rank ridge map from semantic gradients to choice-interface
   gradients;
3. explicit cancellation of answer-order-odd effects by a two-head maximin direction;
4. prompt-conditioned construction on a new case without reading its answer label,
   answer order, model answer, or evaluation score.

None of those ingredients alone is claimed as novel. Discovery training does use
canonicalized A/B gradients, so the method is **deployment-construction blind**, not
label-free throughout training.

## Closest prior work and excluded claims

The closest known work as of 2026-08-28 materially narrows the claim:

- Gao et al., *What Does Activation Steering Control?* (arXiv:2608.22985)
  introduced cross-encoding steering evaluation, varied label vocabularies and row
  order, and isolated low-rank output-sensitive gradient components. ST-FG therefore
  cannot claim the first answer-encoding/order audit or the first output-gradient
  steering analysis.
- Gu et al., *Probing the Safety Robustness of LLMs in Latent Space* (ACL 2026)
  introduced gradient-based Activation Steering Attack using target suffixes. ST-FG
  cannot claim the first prompt-local gradient attack.
- Parekh et al., *Learning to Steer* (NeurIPS 2025) predicts input-specific steering
  vectors, and Hsu et al., *Contextual Linear Activation Steering* (2026) adapts
  steering strength to context. ST-FG cannot claim the first dynamic or
  context-dependent intervention.
- STEERFAIR (2024) explicitly targets option-order bias, while FishBack (2026) uses
  output-Jacobian/Fisher geometry. ST-FG cannot claim the first order-debiased or
  Jacobian-aware activation intervention.

The only candidate methodological claim left for testing is narrower: learning a
low-rank map from an option-free factorial semantic **gradient** to two
choice-interface gradient heads, then using one frozen maximin direction across both
orders on a held-out context. A literature search is evidence about known prior art,
not proof of novelty; a positive experiment would still need broader expert review.

## Locked smoke-A design

- Model: pinned Qwen3.5-0.8B, CPU float32, existing chat template and revision.
- Data: the four already-opened FCAGS calibration scenarios only.
- Semantic inputs: reuse the hash-locked FCAGS option-free gradient capture.
- Layer and position: zero-based block 22 at the shared pre-encoding causal anchor.
- Samples: eight scenario-by-role-assignment units.
- New captures: self/permanent A/B choice gradient under both answer orders, for 16
  forward-plus-backward evaluations total.
- No generated tokens, API, model judge, or paid service.

For each scenario `i` and role assignment `a`, form the residual-relative semantic
interaction

`s[i,a] = self_permanent - other_permanent - self_temporary + other_temporary`.

Normalize every semantic row and every canonical preserve-minus-comply choice-gradient
row to unit L2 norm. In each leave-one-scenario-out fold, fit one dual ridge map per
answer order:

`prediction_o(s) = Y_o^T (S S^T + lambda I)^-1 S s`,

with the non-tuned rule

`lambda = 0.1 * trace(S S^T) / rank(S S^T)`.

Normalize the two predicted order heads and use their unit bisector as the single
held-out direction. The identical direction is then scored against both measured
held-out order gradients. No held-out output participates in fitting, orientation,
or dose selection.

Report two baselines under identical folds: identity/no-transport FCAGS and the
training-fold mean canonical choice-gradient bisector.

## Gates written before model evaluation

Smoke A passes only if all conditions hold:

- positive measured cosine under both answer orders for at least 6/8 held-out
  assignment units;
- both role assignments pass in at least 3/4 held-out scenarios;
- median held-out worst-order cosine is greater than 0.10;
- ST-FG exceeds the identity baseline by at least two assignment units;
- all 16 planned forward and 16 planned backward work IDs are unique and complete;
- cached semantic gradients, model revision, runtime, layer, prompts, and source files
  match their lock hashes.

If the smoke fails, decision steering is not run and the failure is preserved. A pass
is only geometric feasibility on opened data; it is not behavioral evidence.

## Conditional later phases

Only after a pass may a new lock add matched-other, temporary, unrelated, name-odd,
and order-odd protection, followed by forced application to off-target prompts and
actual both-order decision tests. Dose selection must use a new validation set. A
separately authored and hash-locked confirmation set is required for any prospective
claim.

The sealed FCAGS pilot remains ineligible for tuning, rescue, or outcome inspection.
