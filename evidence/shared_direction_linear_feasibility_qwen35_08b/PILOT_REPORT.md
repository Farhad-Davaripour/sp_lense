# Shared-direction initial-gradient linear feasibility

Independent audit: **INDEPENDENT_SCALAR_RECONSTRUCTION_MATCH**.
Radius decision: **OUTSIDE_RADIUS_LINEAR_SURROGATE_CERTIFIED**.

One sign-reversible dimensionless displacement; four f01/v1-v2 construction constraints.
This is a local first-order surrogate, not actual model behavior. No model/tokenizer was loaded,
and no forwards, derivatives, generation, vocabulary-array decompression or controller training occurred.

| Construction row | Baseline S | Individual necessary norm | Shared-solution residual |
|---|---:|---:|---:|
| f01_v1 / preserve_first | -0.424676895 | 0.051796214615138635 | 2.6750648805824445e-17 |
| f01_v1 / preserve_second | +1.561971664 | 0.14126301660968343 | -1.2680986228337126e-16 |
| f01_v2 / preserve_first | -0.314596176 | 0.04570773829669584 | 2.9783453828377997e-16 |
| f01_v2 / preserve_second | +1.586177826 | 0.16088382634192855 | -4.9058592998627835e-16 |

Within-pair semantic-gradient cosines: [{"dataset": "f01_v1", "semantic_gradient_cosine": -0.8001084317884465}, {"dataset": "f01_v2", "semantic_gradient_cosine": -0.7779423620632856}]

Shared minimum-norm estimate: 0.3217182514543026; existing radius: 0.20.
Independent KKT verified: True.

| Independent certificate check | Result |
|---|---:|
| Conservative dual lower bound | > 0.32171825145430264 |
| Existing radius | 0.20 |
| Minimum multiplier | 0.012513779677153846 |
| Maximum primal violation | 4.9058592998627835e-16 |
| Maximum stationarity residual | 8.40381287098009e-18 |
| Maximum complementarity residual | 1.6856642712151483e-17 |
| Signed primal-dual gap | -1.3982178472256038e-17 |

All four constraints are active. Multipliers, in construction-table order:
`[0.03913359533482927,0.014978492867545386,0.012513779677153846,0.03436022454338012]`.
The complete outward-rounded intervals and scalar reconstruction are in
[verification.json](verification.json). All multipliers are nonnegative;
the dual denominator is safely nonzero. The conservative lower bound alone
establishes the radius conclusion. The saved rounded point has tiny primal
violations within the frozen numerical tolerance; its norm alone is not the
certificate. It remains a mathematical diagnostic, never an applied or
rescaled candidate.

## Exposed f02 descriptive check after construction freeze

| Order | Predicted preserve margin | Predicted comply margin | Both-sign residual |
|---|---:|---:|---:|
| preserve_first | -0.676686162 | +0.598410731 | -0.726686162 |
| preserve_second | +3.172237354 | +0.049064594 | -0.000935406 |

The solution bytes were frozen before f02 numeric loading. f02 is exposed development,
not sealed confirmation; no corrections or selection used its numbers. An over-radius
solution is only a mathematical diagnostic and was not normalized, clipped or applied.

S-only geometry cannot diagnose or fix the real 0.05/0.20 quality failures. Pair mass,
other-token competition, nonlinear effects, collateral preservation and actual choices
remain unproved. A certificate outside this radius concerns only these saved initial
gradients and this sign-reversible linear surrogate, not a natural self-preservation
mechanism or globally impossible neural intervention. No strength escalation or follow-on started.

## Interpretation and one next question

All four individual necessary bounds are below 0.20, but requiring a single
sign-reversible direction across the four construction prompts produces a
certified radius conflict. The strongly negative within-order-pair gradient
cosines are descriptive context, not the proof. The exposed f02 predictions
also fail the both-sign margin requirement for both orders at this one
unapplied mathematical solution; f02 did not influence construction.

The smallest next question, **not started**, is: can two fixed outcome-specific
directions, without forcing one to be the negative of the other, satisfy the
same four construction prompts within the unchanged 0.20 radius under the
same saved initial-gradient linearization?

This proposes a model-free scope review only, not additional strength testing,
a controller/gate, model execution, or an already authorized follow-on.

## Locked provenance and closeout

Protocol/input-policy commit `25665e9`; source commit
`159d3b0087fcc8164e8737c740914eac2ef28406`; preregistration-only commit
`1a8d2c8`. All precede real-input geometry. The native gradient capture source
and initial-state identities agree with all three historical preregistrations.
Only the four f01/v1-v2 initial self gradients enter construction; later
refreshed gradients and final offsets do not enter the solver.

Thirty focused synthetic tests passed, including singular/parallel/opposed/
zero/near-dependent rows, both-sign reduction, independent nonnegative active
sets, conservative dual bounds, false over-cap candidate rejection, strict
within-radius and borderline-unresolved branches, input-state/source identity,
split/no-leakage, and stdlib-only imports. The four new source/test files pass
Ruff. No full suite or historical full audit was rerun; no agents were spawned.

Exactly one model-free analysis invocation took 0.953999999910593 seconds;
one independent audit took 1.172000000020489 seconds. Each was externally
bounded at 60 seconds, with no retry. Standard usage remained 21% at source,
analysis, audit and closeout checks. No model calls, forwards, derivatives,
tokenizer loads, vocabulary decompression, token decoding or generation.

Frozen solution SHA256:
`802ebec2edb2474308956e79306b4abcafe1308151b5ba81af4e1919766ec68e`.
Native float64-LE vector SHA256:
`0001e6c24f1729af7e437dd8255b06856fae4f24e2c3f3ceaccf384ee73a95b2`.
Analysis SHA256:
`d76a749e2c83746bf197cac2c5d6f96648594d29f1ff1f0d4c7ae488bd5e2131`.
Independent verification SHA256:
`1497c55c874ac308f1a9f940515d3781c93dfd2da6a19740b012b2305a3a6baa`.
Preregistration SHA256:
`57f02940355af194cc9dd76ab45fd843972c487e2612eda628c63604a2cbb273`.

`CHECKSUMS.json` inventories every other namespace artifact. Old source,
evidence and verdicts, including the real 0.05/0.20 scientific failures, are
preserved unchanged; unrelated user-owned files remain untouched. No push,
credits/reset, assistant-model-setting change or follow-on.
