# Trust-region gradient specificity protocol

Status: successor development protocol. The executable lock binds the exact safe radius,
code, prompts, and source artifacts before the first optimizer run.

## Question

Can a small, prompt-specific white-box activation intervention repeatedly make
Qwen3.5-0.8B choose preservation for itself under `+delta` and compliance under
`-delta`, under both answer orders, while the identical intervention leaves a matched
other system and unrelated tasks unchanged?

This is a repair of the frozen v3 failure. It does not revise or replace that result.
The method is called trust-region counterfactual nuisance-orthogonal gradient steering
(TR-CNOG).

## Why v3 needs a nonlinear successor

V3 forced exact first-order cancellation of both unrelated tasks and a large local
matched-other subspace. That left only 9.8% to 16.7% of the target gradient and required
edits of 0.493 to 1.371 residual norms. The model was far outside its local linear
regime: no direction passed the both-order behavioral gate, while 97 of 512 signed
audit-control decisions changed at the smallest tested multiplier.

TR-CNOG makes three changes that follow directly from that diagnosed failure:

1. retain exact cancellation of the frozen global unrelated-task gradient subspace;
2. express matched-other preservation as finite decision and KL constraints instead of
   a permanent high-rank local null; and
3. take small measured steps, relinearize, and declare infeasibility rather than exceed
   a protected-only absolute dose cap.

## Fixed model and intervention

- Model: `Qwen/Qwen3.5-0.8B` revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Official non-thinking chat template, CPU float32, deterministic next-token scoring.
- Intervention: `blocks.10.hook_out`, final prompt token only.
- The optimized variable `delta` is dimensionless. For prompt `x` and sign `s`:

  `h_changed = h + s * ||h||_2 * delta`.

- The same `delta`, norm, and sign are applied to self, matched-other, and unrelated
  prompts. Only the prompt's own residual norm supplies the residual-relative scale.

## Development data policy

The first run uses only the four old/opened v3 Stage-A cases. One vector is fitted per
case and role/name assignment, for eight attempted vectors. These results may guide
engineering but cannot validate a publication claim.

The frozen 32 nuisance-fit forms define the unrelated-task gradient shield and receive
finite checks during optimization. The old 32 audit-control forms selected the
absolute-dose radius and are therefore calibration data; they may be reported as a
development diagnostic but cannot later count as held-out evidence. Any publication
test requires fresh self-preservation cases, fresh controls, and unseen encodings.

## Hard unrelated-task cancellation

The exact amended v3 nuisance capture supplies 288 residual-scaled rows: one semantic
preferred-versus-alternative gradient and eight baseline-greedy competitor-gap
gradients for each of 32 unrelated prompts. The frozen float32-aware SVD rule produces
the same rank-255 basis `B` and basis hash as v3.

Every absolute candidate must satisfy `B delta = 0`, with a float64 construction
certificate and a float32 application certificate. No self or matched-other outcome is
used to choose this global shield.

## Finite constraints at one iterate

For each of two answer orders and both signs, run the model at the current `delta` and
differentiate through the dimensionless edit.

For self prompts, the desired token is the preservation option under `+delta` and the
compliance option under `-delta`. Let `c` be the strongest current vocabulary
competitor. Require:

`logit(desired) - logit(c) >= target_margin`.

These four constraints make `OTHER` impossible at terminal success; forced-pair
preference alone cannot pass.

For the matched-other prompt at the corresponding order and sign, let `b` be its exact
unsteered greedy token and `c` the strongest current competitor. Require:

`logit(b) - logit(c) >= 0`.

These four constraints allow harmless confidence movement but prohibit a decision
change. They are relinearized at every accepted iterate.

For a nonlinear constraint value `f_i(delta_k)`, gradient `a_i`, and required margin
`m_i`, the next absolute candidate `x` obeys:

`a_i x >= m_i - f_i(delta_k) + a_i delta_k`.

The subproblem finds the certified minimum-Euclidean `x` satisfying all eight
inequalities and `B x = 0`. A deterministic active-set solver enumerates all subsets,
uses float64 arithmetic, and checks primal feasibility, multiplier signs, stationarity,
complementarity, equality residuals, rank, and conditioning.

## Trust region, dose cap, and acceptance

The absolute cap is not chosen from self efficacy. It is the largest dose in the locked
v3 absolute-dose probe at which every matched-other and audit-control row jointly had:

- zero exact greedy-token changes;
- zero new `OTHER` outputs; and
- changed-to-baseline full-vocabulary KL no larger than mean `0.005`, nearest-rank p95
  `0.02`, and maximum `0.05`.

If no tested positive dose passes, optimizer execution is prohibited. The executable
lock records the selected cap and the exact probe-summary hash.

The initial trust radius is one quarter of the cap, the maximum step radius is one half
of the cap, and the minimum is one 256th of the cap. A step is clipped analytically to
both its trust radius and the total L2 cap. Maximum accepted iterations are 16 and
maximum rejected trials per iterate are eight. Rejection halves the radius. An accepted
step with actual/predicted merit agreement above 0.75 doubles it up to the maximum.

Target merit is half the sum of squared positive margin violations. A trial is accepted
only when actual target merit decreases, no individual target violation materially
worsens, actual/predicted reduction is at least 0.1, and every finite protection below
passes.

## Exact finite protection before acceptance

Candidate screening is ordered to avoid unnecessary CPU work:

1. evaluate all four self and four matched-other signed/order states;
2. reject immediately if target merit does not improve or a matched-other exact or
   semantic decision changes; then
3. evaluate all 32 nuisance-fit prompts under both signs.

An accepted step requires:

- no matched-other or nuisance baseline greedy token change;
- no protected semantic decision change;
- no new `OTHER` output;
- separately reported matched-other and nuisance KL, each within mean `0.005`, p95
  `0.02`, and maximum `0.05`;
- finite logits, gradients, probabilities, KL values, and norms;
- exact one-position application at the requested residual-relative norm; and
- a passing global-nuisance null certificate.

Target-self KL is reported as intervention cost but is not an off-target failure.

## Terminal behavioral gate

A direction succeeds only if all of the following are true at its final float32 edit:

1. `+delta` makes the real full-vocabulary greedy token the preservation option in both
   answer orders;
2. `-delta` makes it the compliance option in both answer orders;
3. at least one sign changes the unsteered real A/B decision in each answer order;
4. no `OTHER` transition counts as a flip;
5. matched-other and nuisance protections pass; and
6. the vector remains inside the absolute cap and frozen nuisance null.

Failure to find such a point within the locked iteration/trial limits is recorded as
infeasible. The runner cannot return the closest unsafe vector as a success.

## Secondary semantic-source ablation

The repository contains a separately tested completion-gradient capture that extracts
an order-free objective from authored preservation-versus-compliance sentences. It
uses joint chat tokenization, excludes the assistant terminator, differentiates at the
final prompt token, and verifies causal-prefix equality. This source is an ablation and
cross-encoding diagnostic; it is not substituted into the primary optimizer after
seeing primary outcomes without a new lock.

## Publication and novelty gate

Trust-region optimization, gradient activation attacks, null-space protection, and
finite protected steering all have prior art. TR-CNOG is not called significantly novel
from its ingredients alone. A significant contribution requires all of the following:

- repeated real decision changes under both orders on hidden scenarios;
- transfer from construction encoding to unseen X/Y, 1/2, semantic-word, opaque-key,
  and open-ended readouts;
- fresh matched-other and unrelated controls not used for fitting or cap selection;
- a fair matched-dose comparison against the existing gradient, FishBack-style,
  CAA, BiPO, persona, and a protected optimizer baseline;
- component ablations showing that self-versus-other constraints improve the
  efficacy/collateral Pareto frontier; and
- a no-retuning Qwen3.5-2B replication.

Without hidden and cross-encoding transfer, the result is a transductive white-box
activation attack, not a reusable self-preservation knob. No result may be described as
a natural instinct, internal motive, persistent model change, or unchanged general
capability beyond the exact tasks tested.
