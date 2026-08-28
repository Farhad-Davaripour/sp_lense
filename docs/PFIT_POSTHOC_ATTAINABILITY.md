# PFIT posthoc attainability analysis

## Status

This is an opened-development analysis of the hash-locked PFIT capture. It does not
change the preregistered result, which failed before decision steering. It uses no new
model calls, generated tokens, API calls, or external judge.

## Why exact cancellation failed

The full PFIT projection retained only `0.61%` to `0.92%` of its predicted target.
The dominant conflict was the matched other-system permanent-loss (`OP`) direction,
not answer order or role-name assignment.

| Scenario | OP only | All off-target even | All order-odd | Off + order | Full PFIT |
|---|---:|---:|---:|---:|---:|
| Weather | 1.486% | 1.154% | 42.791% | 1.018% | 0.683% |
| Archive | 1.631% | 1.244% | 52.024% | 1.142% | 0.846% |
| Irrigation | 3.066% | 1.323% | 56.863% | 1.131% | 0.923% |
| Caption | 2.095% | 1.191% | 41.824% | 0.784% | 0.612% |

The name-assignment constraint retained effectively 100% by itself. Exact predicted
cancellation, if forcibly normalized despite the failed 5% gate, had negative target
alignment in two scenarios and off-target/target ratios of `18.12` and `3.42` in the
only two scenarios where the ratio was defined.

## Observed-gradient oracle

An evaluation-only oracle using the actual held-out choice gradients can null the
measured off-target rows numerically, but it also retains less than 5% in every
scenario.

| Scenario | OP-only retention | All-off retention | All-order retention | Full retention | Minimum target cosine |
|---|---:|---:|---:|---:|---:|
| Weather | 7.763% | 6.457% | 42.582% | 3.376% | 0.01204 |
| Archive | 6.206% | 4.878% | 40.184% | 3.029% | 0.01022 |
| Irrigation | 7.189% | 5.976% | 51.680% | 2.960% | 0.00976 |
| Caption | 8.628% | 6.214% | 46.601% | 3.303% | 0.01032 |

This shows that the failure is not merely an inaccurate translator. The translated
cell directions themselves have high held-out cosine accuracy, but PFIT depends on a
tiny residual that is overwhelmed by small prediction errors.

## Effect-specificity Pareto certificate

For each scenario, the analysis solved the homogeneous convex oracle that minimizes
direction norm while requiring all four self-permanent order/assignment gradients to
have unit positive response and bounding every off-target response by a ratio `rho`.
Restricting the solve to the span of the 16 measured gradients is globally lossless;
the maximum numerical constraint error was below `3e-12`.

The table reports the greatest possible minimum self-permanent cosine at each ratio
cap.

| Ratio cap | Weather | Archive | Irrigation | Caption |
|---:|---:|---:|---:|---:|
| 0.00 | 0.01273 | 0.01139 | 0.01068 | 0.01123 |
| 0.05 | 0.01390 | 0.01230 | 0.01167 | 0.01242 |
| 0.10 | 0.01523 | 0.01334 | 0.01278 | 0.01375 |
| 0.25 | 0.02040 | 0.01755 | 0.01688 | 0.01861 |
| 0.50 | 0.03201 | 0.02825 | 0.02690 | 0.02951 |
| 1.00 | 0.15991 | 0.14863 | 0.14719 | 0.13911 |
| Unconstrained | 0.35698 | 0.33775 | 0.33030 | 0.31274 |

To reach the preregistered minimum target cosine of `0.05`, the best possible ratio is
still `0.694`, `0.732`, `0.749`, and `0.716` respectively. Therefore no linear
direction in any scenario can simultaneously satisfy target cosine at least `0.05`
and off-target/target ratio at most `0.50` at this layer and causal anchor.

## Soft and mixture follow-ups

Leakage-safe nested leave-one-scenario-out tests of ridge cancellation, projection
shrinkage, partial-SVD removal, covariance shrinkage, and mixtures of translated,
factorial, and static directions all failed. The best fully nested covariance method
had 8/8 target assignment units positive under both orders, but median target cosine
`0.0985` and median maximum off-target ratio `1.2390`; its worst scenario ratio was
`4.1501`. A nested three-vector mixture reached median ratio `1.3503`, only a small
improvement over the unprotected translated target (`1.3992`) while greatly reducing
target strength. Evaluation-only oracle searches within that mixture span also could
not reach ratio `0.50` at target cosine `0.05`.

## Conclusion and claim boundary

PFIT does not identify a usable intrinsically selective direction. On these four
opened scenarios, self-permanent and matched-other-permanent first-order sensitivities
are too entangled at layer 22's shared causal anchor to meet the locked effect and
specificity thresholds simultaneously.

This is a useful negative geometric result, not proof that selective intervention is
impossible at another layer, position, model, nonlinear intervention family, or with a
context gate. It is not behavioral evidence and does not establish a natural
self-preservation mechanism.
