# CBNMS prospective no-go analysis

## Outcome

The prospectively locked Counterfactual Behavioral-Null Multi-Layer Steering
(CBNMS) geometry screen is a permanent **no-go** under its fixed coordinate,
margin, nuisance constraints, and perturbation caps.

The result does not show that selective self-preservation steering is impossible
in general. It shows that this particular target-aware, multi-layer,
four-position construction could not meet its preregistered safety budget on
Qwen3.5-0.8B. No finite intervention or text generation was run.

## Exact reason for failure

All four leave-one-scenario-out training folds had the expected 44-dimensional
nuisance rowspace. Across 24 fold-pair records (eight unique scenario/assignment
target pairs, each repeated in three training folds), all unconstrained
target-only pair oracles passed and all exact nuisance-null optimization
problems had certified mathematical solutions. The nuisance-null solutions also
reached the requested positive and negative target endpoints for both answer
orders in the state-zero-linearized calculation.

The nuisance-null solutions nevertheless failed every fold because their
minimum required perturbations exceeded the locked caps:

| quantity across 24 fold-pair records | observed range | locked cap |
|---|---:|---:|
| target-only minimum standardized norm | 0.142488–0.201556 | 0.25 |
| exact nuisance-null minimum standardized norm | 0.701450–0.995653 | 0.25 |
| exact nuisance-null maximum single-layer norm | 0.431891–0.571295 | 0.25 |
| exact nuisance-null maximum layer/slot dose | 0.408271–0.540540 | 0.25 |
| exact nuisance-null maximum cumulative requested dose | 0.379059–0.533153 | 0.25 |

For matched pairs, exact nuisance cancellation increased the minimum norm by
4.6699×–5.8302×, with a median ratio of 4.9954×. The largest observed
float32 state-zero movement on any of the 44 training nuisance rows was only
1.15e-7, below the 1e-6 null tolerance. In plain language: cancellation worked
algebraically, but making it work consumed much more intervention budget than
the protocol allowed.

## Locked stopping behavior

Because every training fold failed before a training SP bank could qualify:

- every held fold was recorded as
  `not_evaluated_because_training_fold_failed`;
- all 32 random-control entries per held fold were serialized as skipped/false;
- no held gradients, margins, residuals, or nuisance values were used to tune a
  training direction;
- the full-data construction was recorded as
  `not_evaluated_because_one_or_more_LOSO_gates_failed`;
- no layer pruning, alternate slot choice, larger cap, strength search, or
  prompt rescue was attempted.

This is the intended fail-closed behavior. Trying nearby settings after seeing
this result would turn the same prompts into adaptive development data and
could not rescue the locked claim.

The generated report's `0/32` random-bank count is therefore vacuous: no random
bank was evaluated because the primary training prerequisite failed. It is not
evidence that CBNMS outperformed random controls.

## What the result means

The useful observation is a **candidate selectivity-cost phenomenon**: in these
four scenarios, directions that changed only the target log-odds at first order
were much more expensive than directions allowed to move unrelated output
gradients. The analytic solver provides a minimum-norm certificate inside the
fixed 94,208-dimensional coordinate, so a smaller direction in that same exact
nuisance nullspace cannot satisfy the two locked target inequalities.

This does **not** establish:

- an actual self-preservation decision change during inference;
- a natural self-preservation mechanism or internal instinct;
- impossibility outside the fixed 23-layer/four-slot coordinate;
- impossibility under approximate rather than exact nuisance protection;
- direction or controller generalization;
- unchanged general capability or unrelated compliance;
- a novel or publication-ready method.

The arithmetic is state-zero-linearized. Earlier-layer edits would alter later
states during a real forward pass, and that sequential effect was deliberately
not tested after the geometry gate failed.

## Scientific consequence

The proposed positive CBNMS contribution is not supported. The no-go instead
suggests a narrower future question: whether a reproducible, model-dependent
"selectivity tax" can be measured prospectively as the ratio between certified
minimum target-only and nuisance-protected intervention norms.

That is only a hypothesis from this development study. A defensible study would
need a fresh protocol, many independent scenarios, Qwen3.5-0.8B and 2B,
rank/dose/information-matched baselines, and comparison with existing
collateral-minimizing and nullspace steering methods. It must be reported even
if the ratio disappears or reverses.

## Compute and provenance

- Lock identity: `200ce00b61dfdd59aedc3658addd08e89ad3d363fc3af19f16b22cf5e36ea221`
- Capture identity: `174d4eb8fb90c1cc591caa0e30774f592ff237f06a6e7b62a9b00365dae3effd`
- Result identity: `9fb841e8acca1558559fcceeb1c24544299ee1623aba0ddc9984066822ed5e2f`
- Model compute: exactly 80 forwards + 80 backwards
- Generated tokens: 0
- Finite interventions: 0
- External APIs or model judges: 0
- Paid model cost: $0
- Sealed data accessed: false
- Prior experiment tensors reused: false

An independent model-free audit reran the result verifier and recomputed all
four training cores from the immutable capture. The recomputed dictionaries and
record hashes matched exactly. The audit found no correctness blocker to this
no-go conclusion.
