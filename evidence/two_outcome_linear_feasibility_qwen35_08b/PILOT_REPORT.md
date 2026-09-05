# Two-outcome initial-gradient feasibility

Independent audit: **INDEPENDENT_TWO_OUTCOME_SCALAR_RECONSTRUCTION_MATCH**.
Joint result: **AT_LEAST_ONE_OUTCOME_OUTSIDE_LINEAR_SURROGATE**.

Two fixed outcome vectors reduce the norm estimates, but comply still has a
conservative lower bound above the unchanged 0.20 per-application cap.
This certifies only the saved-gradient linear surrogate, not real-model
impossibility or a diagnosis of the earlier 0.05/0.20 quality failures.

## Separate outcomes

| Outcome | Norm estimate | Conservative radius decision | KKT |
|---|---:|---|---|
| Preserve | 0.051796214615138635 | Numerically unresolved | Verified |
| Comply | 0.21752243307283117 | Outside: dual lower bound >0.21752243307283120 | Verified |

Preserve's norm estimate is well below 0.20, but its rounded point has a
conservative slack of approximately -2.0567e-18. The strict frozen within-cap
rule therefore remains **unresolved**; no correction, re-solve or tolerance
change was made. Comply's own nonnegative-multiplier dual certificate is
decisive, so both-outcome feasibility is ruled out for this exact surrogate.

| Check | Preserve | Comply |
|---|---:|---:|
| Max primal violation | 2.0566956789104746e-18 | 3.421631212870832e-16 |
| Max stationarity residual | 4.924050101562869e-19 | 7.424641970278518e-18 |
| Max complementarity residual | 1.1624331484368506e-20 | 1.1511132730605996e-17 |
| Signed primal-dual gap | -1.1624331484368499e-20 | -5.64682440118216e-18 |
| Active construction indices (zero-based) | [0] | [2, 3] |

Multipliers, in the construction-table order:
preserve `[0.005651945304094013, 0, 0, 0]`;
comply `[0, 0, 0.029209314336752935, 0.03364223674166181]`.
The comply certificate uses the v2 retention and opposed-request constraints;
retention is binding, not a row that can be discarded. Both denominators
are safely nonzero; no exact-infeasibility claim uses a near-zero denominator.

## All construction constraints

Every outcome retains all four rows. Negative RHS values are neither
clamped nor replaced by abs(S). Display values below are rounded; exact
outward intervals are in [verification.json](verification.json).

| Row / order | Preserve RHS | Preserve margin / slack | Comply RHS | Comply margin / slack |
|---|---:|---:|---:|---:|
| f01/v1 / first | +0.474676895 | +0.050000000 / -2.0567e-18 | -0.374676895 | +0.105851027 / +0.055851027 |
| f01/v1 / second | -1.511971664 | +1.089064606 / +1.039064606 | +1.611971664 | +0.124629889 / +0.074629889 |
| f01/v2 / first | +0.364596176 | +0.087801221 / +0.037801221 | -0.264596176 | +0.050000000 / +2.0077e-16 |
| f01/v2 / second | -1.536177826 | +1.183239742 / +1.133239742 | +1.636177826 | +0.050000000 / -3.4216e-16 |

## Exposed f02 predictions after both solution freezes

| Order | Preserve S+A*wP / slack | Comply -(S+A*wC) / slack |
|---|---:|---:|
| First | -0.194558607 / -0.244558607 | +0.033112854 / -0.016887146 |
| Second | +1.069589338 / +1.019589338 | +0.143752701 / +0.093752701 |

These are exposed descriptive predictions, not sealed confirmation or actual
choices. Neither first-order prediction meets its requested margin. Both
solution records were durably frozen before f02 numeric loading; neither was
corrected using f02. A hypothetical comply edit uses norm(h0)*wC directly,
without another minus sign.

The old shared sign-reversible norm estimate was 0.3217182514543026 with its
own outside-radius certificate at commit `881bae6`. This comparison is
descriptive, not strength selection. There is no joint sum-of-norms cap.
No point was normalized, clipped or applied.

## One next question — not started

Would a fixed comply direction for each A/B order fit within the same 0.20
cap across v1/v2 in this saved-gradient surrogate? This targets the remaining
order-pair conflict without escalating strength. It is only a proposed
model-free scope review, not authorization for a gate/controller or model run.

## Provenance and limits

Protocol `7131199`; source `994b3e02e10737cc2f0996d8775ef8a42df2f798`;
preregistration-only lock `8cc559c`, all before computation. The prior
authenticated initial-gradient loader, active-set solver and 80-digit
directed-rounding Decimal certificate arithmetic/tolerances are unchanged.

17 focused tests and Ruff passed. One analysis for both solves took
0.7659999998286366 seconds; one independent audit took 0.9690000000409782
seconds, each externally capped at 60 seconds. Standard usage remained 21%.
[CHECKSUMS.json](CHECKSUMS.json) inventories every other artifact; solution
and native float64-LE vector hashes also appear in the machine-readable records.

No model/tokenizer loads, forwards, derivatives, vocabulary decompression,
new data/strengths, generation, agents or controller/gate training occurred.
Nonlinearity, A/B mass, competing tokens, ordinary-task collateral and actual
choices remain unproved. No causal PASS or natural self-preservation claim.
Historical evidence/verdicts and user-owned changes remain untouched.
No retries, follow-on, push, credits/reset or assistant-model-setting change.
