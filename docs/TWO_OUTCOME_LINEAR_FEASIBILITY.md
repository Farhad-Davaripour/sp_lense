# One model-free two-outcome feasibility extension

Change exactly one assumption from the completed shared sign-reversible
diagnostic: preserve and comply may use two DIFFERENT fixed native-coordinate
vectors. This is not a learned gate or adaptive controller. The exact input
and source hashes are in configs/two_outcome_linear_feasibility.json.
The old certificate and all .05/.20 real-model scientific failures are unchanged.

## Fixed construction and equations

Only initial baseline h0, S=z_preserve-z_comply and semantic g=grad(S) from
f01/v1 and f01/v2: all four self rows, ordered variant then preserve_first/
preserve_second. Authenticate the same historical model/revision, block10,
final encoded token and native1024 coordinates, original-baseline gradient_1
state and capture-source hashes using the unchanged loader.
Never use later refreshed gradients or final offsets.

Let A_i=||h0_i||g_i. Solve exactly TWO convex problems:
PRESERVE: min .5||wP||^2 subject to A_i*wP>=.05-S_i.
COMPLY: min .5||wC||^2 subject to -A_i*wC>=.05+S_i.
Every problem keeps ALL FOUR rows, including baseline-already-correct
requests. Negative RHS values are unchanged: no abs, clamp, skipping,
weighting, whitening or replacement with .05+abs(S).
A hypothetical applied edit is ||h0_i||wP or ||h0_i||wC respectively.
Do NOT negate wC again. Each outcome has its OWN .20 norm cap per applied
edit; there is no joint sum-of-norms cap.

## Exact reused numerical policy

Use the prior audited solver and independent certificate functions unchanged,
through small formulation/namespace wrappers. Enumerate16 active subsets
per problem, partial-pivot Gram solves with relative pivot floor1e-12.
Skip singular/near-dependent sets without pseudoinverse conclusions and
continue independent subsets. Negative multipliers reject only the independent
set; no clipping. Same nonnegative-multiplier requirement, primal absolute
tolerance1e-9, stationarity/complementarity/gap absolute1e-8; choose the sole
minimum-objective KKT estimate, exact ties in mask order.

Reuse the independent80-digit Decimal directed-rounding interval arithmetic
and certificate: absolute1e-9 scalar comparison, zero-relative; final safety
widening1e-40+1e-40*max(abs(endpoints)); denominator floor1e-24 and radius
comparison guard1e-12. Rebuild A and signed RHS independently, without calling
the prior abs(S) system builder. Verify native stationarity, primal feasibility,
lambda>=0, complementarity and primal-dual gap for BOTH outcomes.

Each OUTSIDE_RADIUS decision requires its OWN conservative nonnegative-lambda
lower bound (b^T lambda)/||A_problem^T lambda|| > .20+1e-12. Candidate norm
alone is insufficient. A denominator interval touching zero or upper<=1e-24
is unresolved, never exact infeasibility. The strict unchanged WITHIN_RADIUS
branch needs KKT plus every conservative primal slack lower>=0 and norm
upper<=.20-1e-12. Otherwise return numerically unresolved, including an
apparently in-cap floating point with unresolved rounded boundary residuals.
No policy changes or point correction after real-input computation.

Report each outcome separately. Both-outcome feasibility requires BOTH
within-radius certificates; one alone is not sufficient. Any certified
outside outcome rules out both-outcome feasibility for this exact surrogate.
An unresolved outcome remains unresolved; no causal PASS.

## Locks, execution and split

Commit this exact protocol/config before real-input geometry. Commit the
minimal wrappers and focused synthetic tests, then separately commit the
source/input/environment preregistration lock before the one analysis run.
The namespace adapter restores historical I/O globals on success/failure;
old files never change. Both outcome solution records, including null
unresolved records, are exclusively/durably frozen with SHA256 and native
float64-LE vector hashes before ANY f02 numeric loading. An over-cap point
is an unapplied mathematical diagnostic, never normalized or clipped.

Only after both freezes, load the two exposed f02 initial rows once and
report preserve S+A*wP and comply -(S+A*wC) separately for defined vectors.
No extra negative applied edit, repairs, correction, sign change or selection
from f02. Compare descriptively with the authenticated old shared certificate;
f02 is exposed development, never sealed confirmation.

One external60-second analysis invocation for BOTH solves, one external60-second
independent audit invocation for BOTH certificates. CPU stdlib only; zero
model/tokenizer loads, forwards, derivatives, generation, vocabulary-array
decompression, new data/model/strength, controllers, full suites, old full
audits or agents. Failure/timeout is preserved INCONCLUSIVE, no retry or
numerical amendments after computation. Synthetic failures may be fixed
before source/preregistration lock.

## Delivery and stop

A short report/table of per-outcome norms, conservative radius decisions,
all construction margins/slacks, dual/KKT checks and exposed f02 predictions;
small machine-readable records, independent scalar reconstruction and exact
artifact hashes/commits. Focused tests cover negative RHS, comply sign,
both-outcome distinction, retained-choice constraints, independent radius
certificates, unchanged machinery/policy, and both freezes before f02.

Even if both fit, nonlinear effects, A/B mass and other-token competition,
ordinary-task collateral and actual choices remain unproved. No natural
self-preservation or global impossibility claim. Propose ONE smallest next
question justified by the certificates but do not start it.

Check standard usage before batches; unavailable or>=90% means stop/notify.
No credits/resets, pushes, other assistant models or model-setting changes.
Preserve all historical files/verdicts and unrelated user-owned changes.
Stop after this ONE verified report/commit in
evidence/two_outcome_linear_feasibility_qwen35_08b.
