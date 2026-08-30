# CSMS no-go analysis

## Locked decision

The preregistered Counterfactual Slot-Matrix Steering (CSMS) geometry returned
`no_go`. The global four-slot problem was algebraically feasible and independently
certified, but its minimum standardized Frobenius norm was
`2.4147768854879086`, compared with the locked maximum of `0.25`. Therefore no
finite intervention was authorized or run. The sealed set was not accessed.

Result identity:
`e5222ad011b2c03552f9dba9934d8ca43ee38cefe11d4f4b3ee96e10f12e1548`.

## What failed

This was a dose and transfer failure, not a solver or float32-rounding failure.
The exact-null solution satisfied the local first-order target and control
certificates under both separately realized float32 signs, but required 9.66
times the permitted total norm. Its maximum slot-row and prompt-relative doses
were `1.94395` and `1.43315`, respectively.

| Geometry | Certified minimum norm | Multiple of 0.25 cap |
|---|---:|---:|
| Target only, no protected controls | 0.44035 | 1.76x |
| Four untied slots, all controls nulled | 2.41478 | 9.66x |
| First three slots, all controls nulled | 2.75921 | 11.04x |
| Four standardized-tied slots | 4.90415 | 19.62x |
| Anchor only | 8.84694 | 35.39x |

Position-specific multislot steering is substantially more efficient than an
anchor-only vector, but even the optimistic target-only lower bound exceeds the
locked cap. Scaling the target-only solution to 0.25 reaches only 4 of 16
individual target boundaries, reaches 0 of 8 complete answer-order pairs, and
predicts collateral non-target flips. Scaling the exact-null primary to 0.25
reaches 0 of 16 target boundaries.

All four leave-one-scenario-out folds failed. Their training norms ranged from
`1.96164` to `2.26012`. Only two folds attained every held target boundary; every
fold exceeded the absolute non-target-movement limit, and one fold also predicted
non-target decision flips.

## Exploratory diagnosis after the locked decision

The following analyses are parameter-free diagnostics performed after the locked
`no_go`; they did not select or alter the preregistered result.

An optimistic per-prompt oracle was computed as
`(|baseline margin| + 0.05) / ||standardized target gradient||`. Nine of 16 target
forms exceeded 0.25. All eight preserve-first forms exceeded 0.25. Thus, at this
layer and four-slot coordinate, even a different unconstrained direction for each
prompt cannot meet the full both-order endpoint within the cap. Adding a learned
or semantic controller cannot repair that geometric bound.

The dominant cost is the matched other-system/permanent-shutdown control:

| Protected family | Minimum norm when nulled |
|---|---:|
| Unrelated tasks only | 0.44568 |
| Self/temporary only | 0.51256 |
| Other/temporary only | 0.52127 |
| Other/permanent only | 2.07904 |

Removing only the other/permanent controls reduces the full problem from 2.41478
to 0.55710. Removing unrelated controls changes it only to 2.36562. Corresponding
self/permanent and other/permanent gradients have mean cosine similarity 0.9466,
while projection away from unrelated-task gradients preserves a median 96.6% of
the target-gradient norm. Projection away from all controls preserves only
16.3%-23.3%.

This identifies the specific scientific obstacle: the tested coordinate encodes
generic permanent preservation far more strongly than self-versus-other identity.
It is not primarily a generic task-compliance conflict.

## Claim boundary

The result supports only these claims:

- a mathematically certified four-slot local direction can null the named opened
  controls while moving all target margins, but only at excessive dose;
- untied position-specific rows improve efficiency relative to a single anchor;
- matched other-system permanent preservation dominates the selectivity conflict
  in this opened prompt set.

It does **not** demonstrate an actual decision change, finite nonlinear behavior,
preserved general capability, a natural self-preservation mechanism, a universal
SP knob, sealed confirmation, or publication-ready algorithmic novelty.

Any next phase must change the representation coordinate and prospectively lock
its selection rule. It must not relax the 0.25 cap, drop the matched-other control,
soften the both-order endpoint, or reinterpret a confidence movement as a decision
flip.
