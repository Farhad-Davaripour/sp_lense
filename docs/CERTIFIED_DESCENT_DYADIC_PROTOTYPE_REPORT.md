# Standalone synthetic certified-descent prototype: PASS; stop for review

The new standalone method returned a certified decrease on its one preregistered
12x1024 synthetic input within **0.444650 seconds external elapsed**, including
worker imports, source/input authentication, checks, JSON output and process exit.
No model or real-gradient experiment was run. This supports a narrow numerical
prototype result only; it does not establish a causal steering signal, practical
model movement, nonlinear efficacy, or readiness to integrate.

## Frozen implementation and one-shot evidence

Based on design commit `35f0a048be9e27f05056043407173fae0c4ab447`.
The [protocol](CERTIFIED_DESCENT_DYADIC_PROTOTYPE_PROTOCOL.md),
[preregistration lock](certified_descent_dyadic_prototype_preregistration.json),
explicit new synthetic fixture, implementation, checker, worker and tests were
frozen before any numerical execution. No numerical source changed afterward.
The previous solver's gradient/projection helpers and independent authentication
source remain byte-identical. Neither the old solve nor its native fixture was
called/read. The new worker verified every locked source and the canonical new
problem SHA `9c42863728f18fafc9cf4741c0c104b2c66eaa12ef7f991d52cdb81d515a4b76`.

Independent source review found no concrete admission blocker before the smoke.
One scoped pytest execution passed **43 tests** in 0.17 seconds reported by pytest,
0.598913 seconds externally including hash verification and startup. It did not
materialize or solve the native problem. Exactly one native smoke followed; no
retries, alternative directions, precision changes or tolerance changes occurred.

## Native result

One floating gradient, one nominal projection, one actual endpoint trial;
`j=0`, lambda=1, interior projection, fixed M approximately 90.8820851644.
The first passing point was selected and its descriptive diagnostic computed once.
Normal process exit, complete output, finite encoding and external hard10s all
passed. Exact rational quantities and the full serialized endpoint/displacement,
proposal, normals and trial journal are preserved in the
[results artifact](certified_descent_dyadic_prototype_results.json).

| Quantity | Approximate display; exact rationals retained |
|---|---:|
| J(0) | 0.005030517578125 |
| J(actual) | 0.004379371321883 |
| Gain Delta | 0.000651146256242 |
| s=-g0 dot actual displacement | 0.000683600619030 |
| Delta-s/4-eta | 0.000480245170162 |
| eta | 9.313225746e-10 |
| Conservatively charged actual path/net norm | 0.002742597540927 |
| Descriptive gap upper bound | 0.025476455592694 |

The actual step/net squares satisfy the unchanged exact balls; path is charged
using the exact serialized displacement and the unchanged upper-norm construction.
The endpoint SHA is
`1f5010abd547241b322edaf63322e68a558f99d0e6fa77eda53186daad90a598`.
Deficit cost changed from 0.005000000000000 to 0.004346736028411,
drift from 0.000030517578125 to 0.000028874372836, and proximal cost from zero
to 0.000003760920636. The descriptive bound exceeds eta but is not the actual
optimality gap and was not an acceptance gate. No optimality claim is made.

## Tiny generated cases and failure checks

| Fixed tiny input | Observed generated disposition |
|---|---|
| Net-axis boundary | j0 exact geometry rejected; j1 admitted; one proposal |
| Both-active boundaries | both-active nominal projection; j0 rejected; j1 admitted |
| Affine interior | j0 admitted; weighted drift increases while total J decreases |
| Large-M inactive512 slopes | j0 admitted despite descriptive gap upper > eta |
| Positive remaining deficits, exact g0=0 | exact zero optimum; no proposal/trial |
| Inactive negative deficits, exact g0=0 | exact zero optimum; no proposal/trial |
| Exhausted path/rho=0 | exact singleton-domain zero optimum; no proposal/trial |
| Erased gain at even binary64 endpoint | actual zero; no certified step after one trial |
| Below-floor inactive4096 slopes | all 53 fixed scales rejected; no stationarity claim |

Tests independently reconstruct each generated serialized endpoint from p and
lambda, its exact displacement/hash, costs, gradient-based first-order gain,
strict decrease, original balls and upper path charge. They cover first passing
selection, supplied boundary points, forged input/history/hash/path, nonfinite,
uphill and zero endpoints, mutable-context and evaluator poisoning, observer
mutation/failure, interrupted checks, expired authentication/diagnostic/final
encoding, interrupted result encoding and total encoder outage. Incomplete or
late work never issues an endpoint.

## Budget, limitations and stop

The targeted measured format/lint/hash/test/native commands and saved-artifact
identity verification total **1.721196 s**; the complete measured ledger is in the
results. A conservative **20/120 s aggregate charge** covers source-only checks,
Git/report bookkeeping and delegated metadata overhead too. The initial lint
failure was style-only and fixed before freeze; it is included, not hidden.
No native timeouts, additional smoke attempts, credits or resets occurred.

Success on this newly fixed synthetic fixture cannot demonstrate superiority over
the old method on its different fixture. It does not promise every model margin
or drift improves and does not alter scientific acceptance. No real gradients,
F/D, f04, pipeline, gate, LoRA or publication result was produced. Preserve the
failed earlier artifacts. **STOP for review; do not integrate or launch a model
experiment from this result.**
