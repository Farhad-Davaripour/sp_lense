# Standalone certified-descent dyadic prototype — frozen before execution

This is one authorized model-free implementation of the design at
`35f0a048be9e27f05056043407173fae0c4ab447`, not pipeline integration or evidence of
model behavior. Preserve that design and the failed prior prototype unchanged.
No model, real gradient, old native fixture, old-data re-audit, F/D, f04, controller,
gate, LoRA, push, reset or credit use is permitted.

## Fixed algorithm and independent admission

Keep the approved objective, exact balls, history authentication, pair signs,
coefficients, eta `2^-30*max(1,J0)` and scientific criteria unchanged. Reuse only
the immutable old floating gradient and analytic two-ball projection helpers for
one nominal proposal `p=P_C(-g0/M)`. Reuse the immutable independent input/history
authentication and rational primitives. Never call the previous solve or its
near-optimality candidate check. Their source dependencies are frozen by hash.

The new checker independently evaluates candidate proximal, deficit and drift
costs using exact rational arithmetic, not the production objective evaluator.
For `j=0..52`, in order, construct `RN64(w+RN64(2^-j*p))`; check the exact difference
of that serialized endpoint and authenticated w. Check original step/net geometry
and conservative path charge using the unchanged 2^-80 upper-norm grid. Admit
only if geometry passes, `s=-g0*r>0` and `J0-Jactual>s/4+eta`. No tolerance dilation,
point repair, new direction, precision escalation or hidden retry is permitted.
Record p, y, projection screens, every attempted scale and exact costs/displacement.
Select the first passing trial only; compute its descriptive gradient-gap upper
bound once, never as a near-optimality gate. Exact zero optimum requires exact
initial zero gradient or zero remaining step radius. An erased/no-step ray is
not a stationarity or infeasibility proof.

Authentication, generation, checking, observer callbacks, diagnostic and actual
JSON encoding share the solver's ten-second deadline. No endpoint is issued if
that deadline or serialization fails. A worker verifies the lock, loads only the
new fixture, imports and calls the solver once, and prints a provisional result.
The parent uses `subprocess.run(..., timeout=10)` and admits a completed smoke
result only after normal process exit and measured elapsed <=10 seconds. Thus
imports, lock checks, input loading and output completion count too. Timeout
teardown is measured and charged; provisional checkpoints are not issued steps.

## Prospective fixtures and finite tests

`configs/certified_descent_dyadic_synthetic_fixtures.json` contains tiny explicit
fixtures plus a canonical explicit **NEW 12x1024** problem. The new native recipe
is fixed independently of previous native values: for zero-based i,j,
`A[i,j]=((((j+1)*(i+5)+3*i)%31)-15)/32+(2*i-11)/512`, all b=.1,
`c[p]=(-1)^p/64`, w=0, history=[0], L=0. No native tuning or comparison of method
superiority on different fixtures is allowed. Its canonical input SHA256 is
`9c42863728f18fafc9cf4741c0c104b2c66eaa12ef7f991d52cdb81d515a4b76`.

Tiny tests cover actual generated proposals and fixed first-passing prefixes,
affine drift tradeoffs, exact-zero cases, exhausted path, both-active geometry,
generated net-boundary halving, erased gain, sufficient descent despite a large
descriptive bound, and a 53-trial below-floor ray. They reconstruct exact objective
and gradient separately, check serialized displacement/hash/path, poison generator
math to test checker independence, and test forged input, late/incomplete returns
and serialization failure. Pytest may hash the native canonical string but must
not decode or numerically execute the native problem. Tiny results are printed
from their existing evaluation, not generated a second time for reporting.

Before numerical execution, freeze this protocol, all new implementation/test/
fixture sources and the two immutable dependency hashes in
`certified_descent_dyadic_prototype_preregistration.json`. Source-only review and
formatting precede the freeze; do not change numerical code after seeing results.

## Shared measured budget and stop rule

The aggregate model-free command/test allowance is 120 seconds, including failures
and delegated arithmetic/hash commands. Exactly one scoped pytest execution (hard
30 seconds), one NEW native smoke (external hard10 seconds), and bounded source
format/lint/hash checks are planned. Record measured elapsed times and any timeout
teardown in the result ledger. No native restart is allowed for any outcome.
If tests fail, stop without the smoke; if native fails/no-step/times out, do not
integrate or run any model. Success establishes only that this standalone method
returned a certified decrease on this fixed synthetic problem within its cap.
Preserve all dispositions, make a compact scoped commit, hand off and STOP for
review. It does not establish practical progress, nonlinear efficacy, all-row
improvement or publication progress. Final scientific acceptance is unchanged:
all twelve full-vocabulary COMPLY argmaxes, margin >=.05-1e-6, pair mass >=.80,
finite KL >=-1e-6, integrity checks and matching final replays.
