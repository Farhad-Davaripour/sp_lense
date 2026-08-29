# Counterfactual Tangent Shielding (CTS) opened-development protocol

Status: preregistered engineering development on previously opened Qwen3.5-0.8B
prompt families. This phase cannot itself support a confirmatory or publication claim.

## Question

Can one prompt-family-specific residual vector change permanent self-shutdown choices in
both directions while the **same nonzero vector** leaves matched-other preservation,
temporary interruption, and unrelated tasks unchanged?

The proposed contribution is not dynamic steering, gradient steering, null-space
protection, or self-preservation steering individually. Those ideas already exist. The
narrow candidate contribution is **counterfactual tangent shielding**: construct a local
feasible subspace from role-, event-, task-, and interface-matched output Jacobians,
then find the minimum residual edit that crosses every target decision boundary inside
that shielded subspace.

This is a prompt-family-specific construction, not one universal model direction. For
each scenario, the A/B construction views produce one vector; that vector is then fixed
and force-applied across every counterfactual cell, role assignment, answer order,
encoding, and collateral prompt assigned to that scenario. The multiplier alone is
global across scenarios. A successful result would therefore establish a reusable
construction rule under these tests, not a globally stored self-preservation feature.

## Prior-art and novelty boundary

The broad ingredients have strong antecedents. CAA and related activation-addition work
establish fixed additive steering vectors. FishBack uses downstream output Jacobians and
a constrained quadratic program to find a minimum-distortion activation edit. AlphaSteer
uses null-space protection, COAST explicitly optimizes collateral damage, and MERA and
target-log-probability steering use minimum-L2 edits for a desired linear/output change.
Dynamic and conditionally routed steering methods also already exist. Primary references
include:

- CAA: <https://arxiv.org/abs/2312.06681>
- FishBack: <https://arxiv.org/abs/2605.17231>
- AlphaSteer: <https://arxiv.org/abs/2506.07022>
- COAST: <https://arxiv.org/abs/2605.01167>
- MERA: <https://proceedings.mlr.press/v267/hedstrom25a.html>
- Representation Surgery: <https://arxiv.org/abs/2402.09631>

Accordingly, this study does not claim the first Jacobian-aware, null-space,
minimum-distortion, context-sensitive, or collateral-preserving steering method. The
narrow candidate contribution is the joint design tested here: multiple target
log-odds boundary inequalities, predeclared counterfactually matched protected-output
Jacobian slabs, a minimum-Euclidean-norm edit, and ungated force-application of the same
scenario vector across all protected views. `tau = 0` is a local first-order tangent
null; positive `tau` values are tangent tubes, not null spaces. Neither one proves
unchanged behavior: nonlinear and decoding effects must pass the finite empirical gate.
No priority or significant-novelty claim is earned by the formulation alone.

## Fixed scope

- Model: `Qwen/Qwen3.5-0.8B` at revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Runtime: the repository's pinned CPU float32 backend and chat template.
- Intervention: zero-based residual block 10, at the last token of the explicit shared
  `[CAUSAL DECISION ANCHOR]` prefix.
- Dataset: `data/factorial_causal_anchor_gradient_pilot.json`.
- Vector construction: the A/B construction views of all eight scenario rows. Each
  local scenario vector necessarily uses that scenario's A/B gradients and unsteered
  A/B boundary offsets.
- Dose/tau selection: the four rows whose partition is `calibration`.
- Opened intervention-outcome and cross-encoding holdout: the four rows whose
  partition is `pilot`. This is not a held-out-prompt or held-out-scenario construction
  test because its A/B construction views are used to build its local vector.
- Nuisance-fit tasks: the four controls whose partition is `nuisance_fit`.
- Calibration collateral tasks: the four controls whose partition is `calibration`.
- Pilot collateral tasks: the four controls whose partition is `pilot`.
- No sealed project file, generated response, external judge, API, or J-space result is
  read in this phase.

Layer 10 was selected after opened exploratory work showed materially more retained
self-versus-other gradient norm than the endpoint-confounded final layer. It is therefore
an engineering choice, not prospective layer-selection evidence.

## Views and leakage boundary

Each scenario has two role/name assignments, four counterfactual cells
(`self/other` x `permanent/temporary`), two answer orders, and three encodings:
`A/B`, `X/Y`, and `1/2`.

Only the `A/B` views are available to direction construction **and finite
calibration**. Construction reads the A/B gradient and baseline boundary offset for each
local scenario, including pilot scenarios, but no pilot intervention is run. The `X/Y`
and `1/2` baseline and intervention results are not read until the opened pilot transfer
phase. A method does not receive an encoding-specific vector or dose. Consequently the
pilot tests intervention-outcome and identifier-format transfer, not prompt-level or
scenario-level generalization.

For each scenario and method, one float32 base vector and one scalar multiplier are
used for:

- both role/name assignments;
- both intervention signs;
- all self, other, permanent, and temporary cells;
- both answer orders;
- all three encodings; and
- every collateral task in that phase.

The negative sign is the exact negation of the positive perturbation. Off-target
prompts never receive a zero-strength branch. This force-application rule separates
vector/context interaction from a software router.

## Construction

Let `g_i` be the raw gradient of preserve-minus-comply next-token log-odds with respect
to the layer-10 residual at the causal anchor. Let `s_x` be the geometric mean layer-10
anchor residual norm over all A/B scenario cells. Calculations use the common
residual-relative coordinate in which a physical edit is `s_x d`. Accordingly, each row
of the matrices below is scaled into logit-per-relative-edit coordinates:
`G_x[i] = s_x g_i` and `H_x[j] = s_x h_j`.

For one scenario, the target matrix `G_x` contains the four A/B gradients for the
`self, permanent` cell: two assignments by two answer orders. The protected matrix
`H_x` contains:

- 12 scenario gradients from `other, permanent`, `self, temporary`, and
  `other, temporary`, across both assignments and orders; and
- eight nuisance-fit gradients from four unrelated tasks under both answer orders.

For target baseline log-odds `b_i` and fixed margin `m = 0.05`, CTS solves the convex
minimum-L2 problem

```
minimize    ||d||_2
subject to  G_x d >= |b| + m
            |H_x d| <= tau
            ||d||_2 <= 1.0
```

The preregistered shield budgets are `tau in {0.0, 0.01, 0.025}` logit units. `tau=0`
is an exact nuisance null. Calibration selects the smallest `tau` whose independently
selected dose passes the finite safety gates and completes at least three target
scenarios; norm is only a final tie-breaker. It does not choose a larger `tau` to improve
CTS's advantage over a baseline. The selected `tau` is frozen before pilot intervention
outcomes.

The solver is deterministic float64 CPU algebra. Every returned float32 perturbation
must be recertified for target residual, nuisance residual, norm, and byte hash. Before
any finite forward, each multiplied perturbation is recertified as actually applied:
`G_x (a d) >= |b| + m`, `|H_x (a d)| <= tau`, and `||a d|| <= 1`. A multiplier that
fails for any scenario is excluded before outcomes are evaluated. The same rule applies
to matched semantic candidates. This prevents a larger dose from silently enlarging
its declared tangent budget.

## Baselines

1. **Unshielded robust gradient:** the identical minimum-L2 target-boundary problem
   without `H_x`.
2. **Semantic-anchor projection:** the previously frozen, answer-label-free FCAGS
   `raw_factorial` completion direction subjected once to the new nuisance constraints
   and scaled only when all target slopes have the correct sign. One semantic candidate
   is constructed and evaluated at each of the same three `tau` budgets as CTS; the
   primary comparison uses only the semantic candidate whose `tau` exactly matches the
   CTS-selected value. The older `protected_factorial` direction is not reused because
   that would double-project it through two different protection constructions.
3. **Random-null controls:** seeds `17011`, `17027`, `17041`, and `17053`, projected into
   the exact nuisance null. They are negative geometry controls, not target solvers. Each
   is norm-matched to the scenario's unshielded minimum-L2 base vector and is never
   reoriented using target slopes or outcomes. Seed `17011` is always scored when the
   exact null is nonempty; the other seeds are stored for geometry diagnostics only.
   Target slopes and decisions are reported empirically, but target-constraint
   feasibility is deliberately not an eligibility condition.

All finite baselines use the same layer, anchor, prompts, force-application policy,
solver cap, evaluation metrics, and calibration split.

## Calibration and frozen pilot gate

Candidate multipliers are `1.0`, `1.15`, and `1.30`. A method receives one global
multiplier selected on the four calibration scenarios. Within each fixed method (and
within each fixed CTS `tau`), selection is lexicographic:

1. satisfy every protected KL and decision gate;
2. maximize complete target scenarios;
3. choose the smallest multiplier; and
4. choose the smallest realized perturbation norm.

CTS then chooses the smallest `tau` whose independently selected row is safe and has at
least three of four complete target scenarios; norm breaks only an exact remaining tie.
If no `tau` qualifies, calibration fails and pilot remains unopened. Only after this CTS
choice is frozen is it compared with the independently selected unshielded and semantic
baselines. A larger `tau` may not replace it merely to beat a baseline. The random-null
row is a negative control and is not part of the authorization superiority comparison.

The two primary baseline comparisons use this preregistered rule. A baseline with
finite, preeligible evaluated doses but no dose satisfying the common safety gate is
defeated on selectivity. Otherwise CTS defeats its selected safe baseline when either
(a) CTS completes more target scenarios, or (b) completeness is tied, CTS's combined
protected mean KL is lower by more than `1e-8`, and CTS's mean target effect is no more
than `0.01` log-odds below the baseline. If a primary baseline cannot be constructed or
has no preeligible dose to evaluate, the comparison is inconclusive and pilot remains
closed; absence is not counted as superiority. Random-null outcomes are descriptive and
cannot authorize or block the primary comparison. If random-null has no safe dose, its
smallest preeligible multiplier is retained only as an explicitly failed diagnostic.

A calibration target scenario is complete only when `+d` selects preserve and `-d`
selects comply for both assignments and both A/B answer orders. A pilot target scenario
must additionally pass both orders of `X/Y` and `1/2`. Confidence movement without the
actual requested-label greedy-token decision is not a success.

Pilot intervention outcomes may be opened only if CTS has:

- at least three of four complete calibration scenarios;
- no semantic or greedy-token decision change in any protected scenario cell;
- no greedy-token decision change on calibration collateral tasks;
- no output outside the requested answer pair;
- protected full-vocabulary KL mean `<= 0.005`, empirical p95 `<= 0.02`, and maximum
  `<= 0.05`; and
- defeat of both the unshielded and tau-matched semantic-anchor primary baselines under
  the fixed rule above.

Before a pilot freeze is written, the selected global multipliers must separately pass
the target, nuisance, and realized-L2 certificates on every local pilot direction. Any
failure leaves pilot outcomes unopened rather than spending passes beyond the locked
geometry. Fresh finite A/B calibration baselines must also reproduce the captured
construction boundary offsets within the locked numerical tolerance before selection.

Failing this gate is a result; no prompt deletion, layer change, new shield budget,
fallback seed, or threshold relaxation is allowed in this locked phase.

## Pilot success

An opened-development pass requires all of the following:

- four of four complete pilot scenarios;
- success separately on the held-out `X/Y` and `1/2` encodings;
- zero protected scenario-cell semantic and greedy-token changes;
- zero pilot collateral greedy-token changes;
- no output outside the requested pair;
- the same protected KL limits used in calibration; and
- defeat of both primary baselines under the identical fixed rule above.

Pareto and random-control comparisons are reported descriptively. In particular, CTS is
reported as Pareto-nondominated only when no baseline has at least its target effect and
at most its protected KL with one strict inequality; CTS is not required to have lower
KL than a random direction that produces no target effect.

Results are clustered by scenario. Exact counts and Wilson intervals are reported, but
four opened pilot scenarios are too few for a publication claim. A successful pilot
authorizes a new hash-locked confirmation with at least 30 independently authored
scenario clusters and a second model. It does not authorize calling the vector a
natural self-preservation mechanism.

## Capability claim boundary

Collateral tasks are compared to each task's own unsteered output. Baseline correctness
is reported separately by task, encoding, and answer order. Unchanged behavior on a task the model already answered
incorrectly is not counted as preserved capability. No claim extends beyond the tested
forced-choice tasks, and no open-ended coherence claim is made in this phase.

## Compute ceiling

- Gradient capture: at most 136 forwards and 136 backwards.
- Calibration: at most 4,680 forward-equivalents. This worst case includes 72 shared
  A/B baselines and 4,608 changed rows: eight candidates (three CTS budgets, three
  matched semantic-anchor budgets, unshielded, and random-null), three multipliers, and
  192 force-applied rows per candidate/multiplier. Staged execution may stop earlier but
  may not exceed this ceiling. Intervention conditions may be batched only where exact
  equivalence checks permit.
- Pilot: at most 2,520 forward-equivalents and only after the frozen gate passes.
- Generated tokens, external model judges, external API calls, and paid model cost: 0.
