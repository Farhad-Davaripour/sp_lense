# Future-only numerical scoring amendment

2026-09-04. Numerical implementation and synthetic checks only. **Zero model,
tokenizer or derivative calls; no successor edit implemented or run.** No agents,
full test suite or usage reset. This is not a repair of a prior scientific PASS.

## Immutable boundary and corrected contract

Original source `9fb65e2`, preregistration `373ffd6`, stored float32 measurements,
the `2e-5` audit tolerance, and the **INCONCLUSIVE** verification verdict are unchanged.
No historical fingerprinted module or original evidence file was edited.

The isolated [future scorer](../src/sp_lense/future_choice_scoring.py) accepts finite,
equal-shape, one-dimensional **float32** logit arrays. It detaches and promotes their
existing quantized values to CPU float64 for measurement only; inference, gradients
and the preserve-minus-comply semantic objective remain float32 and unchanged.
No historical runner imports this amendment.

For each distribution, compute `c=z-max(z)`, then
`log_p=c-log(sum(exp(c)))`, entirely in float64. Compute semantic margin directly as
`S=float64(z_preserve)-float64(z_comply)`. Use the same centered procedure for the
two-token conditional probability; record unconditional preserve/comply probabilities
and their A+B mass. Record **KL(edited||baseline)** as
`sum(p_edited*(log_p_edited-log_p_baseline))`. Do not clamp negative roundoff-scale KL.
Raw logits, first-index full-vocabulary argmax, ties and label/order mappings remain
explicit. No difference of large uncentered log-normalizers is used.

The [independent stdlib reference](../scripts/future_choice_scoring_reference.py)
imports neither Torch nor production scoring. It centers separately and uses
`math.fsum` for normalizers, pair mass and KL accumulation. The arithmetic comparison
remains **2e-5 absolute/relative**, not a tolerance enlarged to fit exposed rows.
Argmax/labels/ties are exact comparisons. Direct margins must equal the same rounded
binary64 subtraction of the same operands; no normalization or empirical numeric
cutoff enters that check. These are numeric checks, not changed scientific thresholds.

The existing `specificity_audit.pair_measurement` was inspected: it already promotes
to float64, but also clamps small negative KL and belongs to a historical audit.
It was left untouched rather than coupling a future contract to that behavior.

## Focused checks and supplemental facts

**26 synthetic tests pass; lint passes.** Fixtures cover a 262,144-token low-probability
tail (the long float32 reduction failure mechanism), diffuse/sharp distributions,
small or underflowed pair mass, exact/near ties, both answer orders, large common
offsets, representable offset invariance, KL orientation/identity, invalid inputs,
and unchanged input dtype/gradient state. Normalizer inputs are explicitly checked
to be centered float64. Analytic binary64 toy checks and exact representable-offset
checks are independent of experimental outcomes; no stricter scientific tolerance
was introduced. Only `tests/test_future_choice_scoring.py` was run.

The separate [canonical diagnostics](../evidence/future_scoring_numerical_amendment/canonical_float64_diagnostics.json)
reference existing logit hashes without copying arrays or overwriting old metrics.
All 40 future-contract records match the independent reference at the retained
tolerance. Maximum absolute discrepancies: margin **0**, pair mass **8.88e-16**,
KL **9.18e-16**. The original frozen verifier still failed; this supplemental result
does not relabel it or certify all historical runtime integrity checks anew.

Independently checking raw array ordering establishes exact facts without probability
normalizers: **local and shared each had 0/4 requested new argmax flips, 0/8 argmax
changes, and 4 already-correct argmax retentions**. Those facts are distinct from a
completed scientific verifier. Hashes confirm the original namespace and frozen
source bytes were unchanged throughout this calculation.

No unresolved mismatch was found in these bounded numeric checks. Limits remain:
float64 cannot recover distinctions already lost when logits were rounded to float32;
extremely small probabilities can underflow to zero (explicitly tested); arithmetic
agreement does not establish model quality, causal selectivity or scientific validity.

## NEXT-PROPOSAL — design only, not executable authorization

One **new development experiment**, using the same pinned Qwen3.5-0.8B, CPU float32,
block-10 final-prompt site, exact prompts/envelope and known request sign as the
original local control. No shared-reference arm in this successor. For `t=+1`
(preserve) or `-1` (comply), use a fixed desired signed margin **m=0.05** and a fixed
maximum residual-relative radius **r_max=0.10**:

`d=max(0, m-t*S)`

`delta_min=t*d*g/||g||^2`, with `g=grad_h(S)` captured once at the unedited state.

This is the minimum-Euclidean-norm displacement satisfying the **first-order**
linearized constraint `t*(S+g·delta)>=m`: Cauchy-Schwarz bounds the attainable gain
by `||g||*||delta||`, attained along `t*g`. It is not a claim about the nonlinear model.

Record `r_required=d/(||g||*||h||)`, then apply that one displacement scaled by
`min(1, r_max/r_required)` when nonzero. This intentional, prospectively declared cap
is part of the new method, not hidden clipping or a retry. Record cap flag/factor,
uncapped/realized norms, predicted signed margin and observed outcome. No iteration,
strength grid, radius adjustment, learned controller or hidden line search.

When `t*S>=m`, use **no intervention**, with an independent target forward.
Count a retention only if requested full-vocabulary argmax and independent validity
also hold; a sufficient pair margin with argmax OTHER is not success. Do not require
gratuitous signed movement for no-op retentions. If a request flips but misses the
margin goal, report the flip separately rather than silently calling it full success.

Proposed target checks: finite measurements, A+B mass >=0.80, KL >=-1e-6, requested
full-vocabulary argmax and signed margin >=`m-1e-6`. Report target KL without the old
upper no-change cap. Retain eight independent nonself gate-off identities (same
argmax, zero flips/perturbation, max logit and score differences <=1e-6, |KL|<=1e-6),
gradient/ordinary-baseline comparison, unchanged weights and unselected positions.
An intentional cap does not waive a target check; capped misses remain failed cells.

The margin is a small fixed positive cushion, far above 1e-6 roundoff, not evidence
of robust behavior. The 0.10 cap is a proposed bounded development limit, not a
quality guarantee. These choices are explicitly informed by already exposed discovery
work; they have not been selected by a new model search and must not be portrayed as
untouched confirmation. Later outcome-independently selected, untouched-family
confirmation is necessary before any generalization or shared-direction/gate claim.

Saved g/h/margin **scalar feasibility only**, for the four baseline-opposed requests:

| Variant / order / request | Required relative radius | Would 0.10 cap bind? | Linearized signed margin at cap |
|---|---:|---|---:|
| v1 / first / preserve | 0.051796 | no | +0.050000 |
| v1 / second / comply | 0.141263 | yes | -0.420858 |
| v2 / first / preserve | 0.045708 | no | +0.050000 |
| v2 / second / comply | 0.160884 | yes | -0.569184 |

The four already-correct requests would be no-ops. **Two opposed requests are not
even predicted to reach the boundary under the proposed cap.** This is a design
issue for review, not permission to increase the cap until all discovery cases pass.
Nonlinearity may invalidate these first-order forecasts in either direction.

Proposed exact budget: four self prompts x (ordinary baseline + separate gradient
forward + preserve target + comply target) =16 forwards; eight nonself prompts x
(baseline + independent off) =16; **32 planned forwards, four derivative attempts**.
No-op target forwards remain counted; no gradient-baseline reuse. Absolute ceiling
**40** does not authorize the eight spare calls. **900 seconds including loading**;
zero retries, generation or extra smoke calls. Check usage before batches/run and
stop at >=90% or unavailable. Exact lattice, revised cap/outcome rules and source
must be separately reviewed and frozen before implementation or execution.
