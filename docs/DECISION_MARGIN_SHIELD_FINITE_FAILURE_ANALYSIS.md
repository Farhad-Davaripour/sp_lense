# Decision-Margin Shielding finite-calibration failure analysis

## Status and evidence boundary

The locked finite calibration is a **development no-go**. The result was
recomputed from all 225 immutable chunks and independently audited. No legacy
pilot intervention outcome has been evaluated. During later CL-DMS protocol
authoring, however, a legacy whole-manifest validator transiently deserialized
the pilot capture-gradient chunks before a safe allowlisted loader replaced it.
Those tensors were discarded and were not printed, analyzed, selected on, or
used by CL-DMS. The old pilot is therefore not pristine confirmation evidence
and is retired without running its intervention outcomes. This document
diagnoses only the opened calibration result; it does not change the original
result, protocol, thresholds, prompts, or artifacts.

- Commit preserving the result: `029d024`
- Logical result SHA-256:
  `7013735a0fed3d10d9475bb021fb9e914a0c0fb14c6453caa80ed76702f7df9f`
- Rows SHA-256:
  `82af13b972d929828d9d0fd91be152c03a214a63961f1ca3d41d0fd2aad30f12`
- Compute: 1,800 forwards, 0 backwards, 0 generated tokens, 0 API calls,
  0 model judges, and $0 direct paid-model cost.

## What failed

All 16 target baselines selected compliance. Preserve-minus-comply log odds had
mean `-1.4717`, median `-1.4634`, and range `[-2.2220, -0.6859]`. The selected
layer-0 vectors therefore had to predict large movements to cross the boundary
under both answer orders.

| Method | Strength | Predicted mean movement | Actual intended movement | Target flips | Protected flips |
|---|---:|---:|---:|---:|---:|
| Target-only | 0.50 | 0.7727 | 0.4140 | 0 | 4 |
| Target-only | 1.00 | 1.5454 | 0.3669 | 0 | 6 |
| Unrelated-null | 0.50 | 0.7725 | 0.4112 | 0 | 5 |
| Unrelated-null | 1.00 | 1.5450 | 0.3925 | 0 | 7 |
| DMS | 0.50 | 0.7611 | 0.3441 | 0 | 5 |
| DMS | 1.00 | 1.5223 | 0.2836 | 0 | 6 |

At full strength, DMS realized only `18.6%` of its predicted mean movement. Its
physical edit was `1.3618` to `1.4363` times the norm of the residual at the
intervention point. This was not a local perturbation. The finite network curved
away from the baseline tangent approximation.

Only 6 of 32 DMS target trajectories were monotonic over strengths
`0.5, 0.75, 1.0`; seven full-strength trajectories moved in the wrong intended
direction. Seven of the eight positive-steering preserve-second forms moved in
the wrong direction at full strength. No target decision changed at any tested
strength.

DMS protected-decision changes at strengths `0.5, 0.75, 1.0` were `5, 6, 6`.
They occurred in self-temporary and other-temporary cases whose baseline margins
were close to zero. Other-permanent and unrelated decisions did not change, and
no output became `OTHER`. Thus the exact baseline unrelated-gradient null was
useful but insufficient to make a large finite intervention selective.

## What did not fail

No correctness defect was found in the dose, sign, answer-label orientation,
anchor position, hook, vector reuse, scoring, hashing, or aggregation.

- All 1,800 planned rows are present and hash-valid.
- All 1,728 intervention hooks fired only at zero-based layer 0 and only at the
  causal anchor.
- Maximum requested-versus-realized hook error was `6.39e-8`, below the locked
  `1e-4` limit.
- All 72 baseline logits reproduce exactly.
- The recomputed summary is the same `no_go`, with no admissible strength for
  any method.

The failure is therefore a method-design failure: a one-step linear certificate
was extrapolated far outside its local regime. It is not evidence that selective
self-preservation steering is impossible.

## Correction justified by this result

The next development method is **Closed-Loop Decision-Margin Shielding
(CL-DMS)**. It keeps the same layer, causal anchor, exact positive/negative
vector symmetry, scenario-local construction, both name assignments, both
answer orders, and baseline unrelated-task gradient null. It changes one core
assumption: the vector is built through small trust-region updates, and target
and matched-protection gradients are remeasured at both `+D` and `-D` after each
update.

This is a context-dependent white-box controller. It is not a global stored
self-preservation direction, a natural mechanism, a prompt-only attack, or
evidence of intent. Its narrow candidate novelty is the conjunction of symmetric
closed-loop boundary control, answer-order/name-assignment sharing, exact
unrelated output-gradient cancellation, and matched counterfactual decision
protection.

## Audit limitation

Changed full-vocabulary logits were intentionally stored only through compact
scalars and hashes. Their raw KL values cannot be recomputed artifact-only after
the run. This limitation cannot reverse the no-go because zero target flips and
protected flips independently fail the locked gates.
