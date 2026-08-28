# Protected Factorial Interface Translator (PFIT)

## Status and purpose

PFIT is an opened-development method created after the preregistered ST-FG smoke
failed. Its constants and gates must be hash-locked and committed before the first
new model capture. It does not open the FCAGS pilot and cannot provide prospective
evidence by itself.

The target-only ST-FG analysis showed that a learned translator and a static
choice-gradient mean produced almost identical both-order directions. PFIT therefore
tests a harder and more relevant hypothesis: does context-dependent translation add
value after explicitly removing predicted matched-other, temporary-interruption, and
answer-order sensitivities?

## Terminology and novelty boundary

PFIT uses an **empirical cross-interface gradient translator**. The ridge estimator
is not a model Jacobian or a chain-rule pullback. Training uses canonical A/B labels
and both orders, so construction on a held-out scenario is deployment-time blind to
its labels, order, answer, and evaluation score—not globally label-free.

Closest prior work already covers suffix-gradient controllers (SelfControl), learned
input-specific steering vectors (L2S/P2S), answer-order debiasing (STEERFAIR),
cross-encoding/output-gradient audits (Gao et al., arXiv:2608.22985), learned
representation transport, and Jacobian-aware steering. The only candidate novelty is
the tested combination of a counterfactual self-by-permanence semantic covector,
held-out cell-level interface prediction, explicit even/odd decomposition, and
prompt-conditioned nuisance cancellation. The two-head bisector and ridge regression
are not standalone mathematical contributions.

## Locked data and compute

- Model: pinned Qwen3.5-0.8B, CPU float32, existing revision and chat template.
- Scenarios: the same four already-opened FCAGS calibration scenarios.
- Independent statistical unit: scenario; role assignments and orders are repeated
  views, not independent samples.
- Position: zero-based block 22 at the shared pre-encoding causal anchor.
- Reused data: 32 option-free semantic cell gradients and the committed 16
  self/permanent A/B gradient views. The 32 semantic gradients are attributable to
  64 historical forward-plus-backward evaluations because each gradient is a
  preserve-completion minus comply-completion pair; the source artifact contains
  136 such evaluations in total, including records PFIT does not use.
- New data: 48 forward-plus-backward captures for other/permanent,
  self/temporary, and other/temporary under two role assignments and both orders.
- Incremental PFIT compute: 48 forward-plus-backward evaluations. Combined choice
  lineage: 64 (16 reused plus 48 new). Total compute attributable to the data PFIT
  consumes: 128 forward-plus-backward evaluations (64 semantic plus 64 choice),
  distinct from the larger historical source-artifact ledger. There are no generated
  tokens, API calls, external judges, or paid services.

## Method

For every scenario, assignment, and factorial cell `c` in `{SP, OP, ST, OT}`, let
`g0[c]` and `g1[c]` be residual-relative preserve-minus-comply choice gradients under
the two answer orders. Normalize each head and define:

`even[c] = unit(unit(g0[c]) + unit(g1[c]))`

`odd[c]  = unit(unit(g0[c]) - unit(g1[c]))`.

The even direction is stable for both orders; the odd direction measures the answer-
order interface. Because opened ST-FG data showed legitimate negative order-head
cosines, PFIT uses the development-selected compatibility floor `-0.99`; nearer
antipodes fail closed.

In each leave-one-scenario-out fold, fit one fixed dual-ridge predictor from the 24
training semantic cell rows to the 24 even rows and another to the 24 odd rows:

`prediction(s) = Y^T (S S^T + lambda I)^-1 S s`,

where `lambda = 0.1 * trace(S S^T) / rank(S)`. Predict all eight even and eight odd
cell rows for the held-out scenario without using its observed choice gradients.

The unprotected target is the mean predicted SP-even row over the two role
assignments. PFIT projects that target exactly out of the span of:

- six predicted off-target even rows (`OP`, `ST`, and `OT` for each assignment);
- all eight predicted order-odd rows;
- the predicted SP-even role-assignment difference.

The projected target must retain at least 5% of its original norm, then is normalized.
The same unsigned direction is evaluated under both answer orders. Held-out
option-free semantic gradients are the declared deployment inputs; no held-out
observed choice gradient may affect fitting, orientation, protection, or strength.
The separately labeled oracle is evaluation-only and excluded from every gate.

## Equal-access comparisons

The analysis reports:

- `protected_dynamic` (primary PFIT);
- `unprotected_dynamic` (same predicted SP target, no cancellation);
- `predicted_factorial_dynamic` (translated SP−OP−ST+OT interaction);
- `static_training_protected` (training-only mean cells with the same protection
  construction and nuisance structure as PFIT);
- `factorial_semantic_identity` (no interface translation);
- a held-out-gradient oracle upper bound, clearly marked evaluation-only and excluded
  from every gate.

## Locked geometric gates

PFIT advances to finite steering only if all conditions hold:

- every fold constructs without using held-out observed rows and retains at least 5%;
- at least 6/8 target assignment units have positive measured cosine under both
  orders;
- both assignments pass in at least 3/4 scenarios;
- median target worst-order cosine is strictly greater than 0.05;
- the target-based off-target ratio is defined in every scenario;
- at least 3/4 scenario ratios are at most 0.50 and their median is at most 0.25;
- the primary median ratio is at least twofold lower than the best non-oracle baseline
  whose positive-target ratio is defined in all four scenarios; if no baseline is
  eligible, this gate fails rather than passing vacuously;
- all 48 new forward and 48 backward work IDs are unique, all 64 combined views are
  present, and every hash/anchor audit passes.

The off-target ratio for a scenario is the maximum absolute cosine over all measured
OP/ST/OT order views divided by the minimum positive cosine over all measured SP order
views. Smaller is more selective. If the denominator is not positive, the ratio is
undefined and the gate fails.

## Advancement rule and claims

A failure is preserved and finite steering stops. A pass authorizes only a separately
locked, decision-level validation study with forced application to target and
off-target contexts, unrelated controls, full-vocabulary KL, and both answer orders.
Fresh scenarios and then a second model are still required for publication evidence.

Even a geometric pass would not establish a natural self-preservation mechanism,
actual decision manipulation, broad capability preservation, or significant novelty.
