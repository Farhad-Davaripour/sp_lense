# One model-free shared-direction linear feasibility diagnostic

Use only the authenticated initial self gradients from f01/v1 and f01/v2,
four construction rows in that fixed variant/answer-order sequence.
The two f02/v1 initial rows are exposed descriptive checks only, read for
geometry only after the construction solution bytes and hash are frozen.
The adjacent config is the exact hash/input/numerical manifest. Hashing f02
raw bytes for authentication is allowed before freezing; using its numbers
for construction, selection, corrections or validation is not.

## Frozen question and equations

The saved semantic gradient is g = grad(S), where S = z_preserve-z_comply,
NOT grad(t*S). The authenticated capture source differentiates that exact
objective at block10 output, final encoded prompt token, native1024 coordinates.
The initial gradient row must be gradient_1, step1, current_cell_id=baseline,
with h=h0=baseline h0, zero offset and exact baseline margin. Never substitute
later refreshed gradients. Confirm model Qwen/Qwen3.5-0.8B revision
2fc06364715b967f1860aea9cf38778875588b17, CPUfloat32, choice-label/boundary
identity and both answer orders from existing verified records. No old
full audit or vocabulary-array decompression.

For one dimensionless w and t=+1 preserve/-1 comply:
delta_i=t*||h0_i||*w; t*S_linear=t*S_i+||h0_i||*g_i^T*w.
Both signs require margin m=.05, exactly equivalent to A_i*w>=b_i,
A_i=||h0_i||*g_i, b_i=.05+abs(S_i).
Solve ONLY min .5*||w||^2 subject to these four constraints, then compare
with the existing radius .20. No alternate objective, weighting, whitening,
normalization, clipping, new radius/margin or candidate portfolio.

## Solver and frozen numerical policy

Use stdlib binary64 arithmetic, math.fsum inner products and sqrt norms.
Enumerate all16 active masks in integer order, including empty. Solve
G_active*lambda_active=b_active with partial-pivot Gaussian elimination,
G=A*A^T. A pivot <=1e-12*max(abs(G_active)) marks that set numerically
rank-deficient and skips it; a zero matrix is rank-deficient. No pseudoinverse.
A negative multiplier rejects only that independent set, never a singular
set's possible nullspace solutions. Other independent subsets are still tested.
Reconstruct native w=A^T*lambda with fsum; retain a solution only with finite
numbers, lambda>=0 exactly, primal violation<=1e-9 and absolute stationarity,
complementarity and primal-dual gap<=1e-8. Choose the minimum objective among
valid KKT sets, with mask order breaking an exact tie. Only that single
solution is saved; the active-set log contains no portfolio of vectors.
No qualifying set means NUMERICALLY_UNRESOLVED, not infeasible.

Independent audit imports no solver arithmetic. Rebuild h0 norms, A, b,
individual b/||A|| bounds, within-variant gradient cosines, primal residuals,
stationarity, complementarity, primal/dual objective and gap with stdlib
Decimal precision80 from exact saved binary64 values. Scalar agreement is
absolute1e-9, zero-relative; the same primal/KKT tolerances above apply.
Use directed-rounding interval operations (ROUND_FLOOR/ROUND_CEILING);
sqrt endpoints use adjacent representable Decimal neighbors of the
correctly rounded square root. This encloses rounding through all native
coordinates. Each final scalar interval is further widened by
1e-40 + 1e-40*max(abs(endpoints)); this is an explicit arithmetic safety
allowance, not an allowance for model or measurement error.

For the chosen nonnegative lambda, independently enclose
L=(b^T*lambda)/||A^T*lambda||. Use the numerator lower and denominator upper
endpoints, then the same outward safety padding. A denominator interval
touching zero or its upper endpoint<=1e-24 is unresolved, never exact
infeasibility. A lower bound strictly above .20+1e-12 certifies OUTSIDE_RADIUS
for this linear surrogate only, even though it cannot certify global neural
infeasibility. A saved w having norm>.20 alone is NEVER a certificate.
The within-cap branch is deliberately strict: require all conservative
primal residual lower endpoints>=0 and norm upper<=.20-1e-12, plus verified
KKT conditions. Otherwise, absent a decisive dual certificate, return
NUMERICALLY_UNRESOLVED even if the floating optimum looks within radius.
Do not rescale/clip/correct a rounded point to force this branch.

## Freeze, reporting and stops

Commit this protocol/config before real-input geometry. Implement and commit
focused synthetic tests and source, then a separate preregistration-only commit
with source/input/environment hashes before the single analysis invocation.
The solver result (including an unresolved null solution) is durably written
with exclusive-create and SHA256 before f02 numeric loading. Preserve an
over-radius solution as a mathematical diagnostic only; never transform it
into an allegedly feasible intervention. Record float64-LE vector SHA256.
Source/input checks occur before each stage, no imports of model packages.

One external60-second analysis invocation and one external60-second independent
audit invocation, using the existing stdlib subprocess timeout wrapper.
No model/tokenizer load, forwards, derivatives, vocabulary arrays, new data,
gate/controller, real-strength test, old full suite, agents or automatic
follow-on. Failure/timeout preserves the attempt and reports INCONCLUSIVE;
no numerical-policy changes or retry after real-input geometry.
Synthetic failures may be repaired before the source/preregistration lock.

Report the four individual necessary bounds, two within-pair cosines, shared
minimum-norm estimate and full KKT/dual-radius evidence. After the construction
freeze only, report the two f02 predicted +/− signed margins and residuals
for the one solution if defined, without corrections.
Pair mass, competing tokens, nonlinear effects, collateral preservation and
actual choices are unproved: S-only geometry cannot diagnose or fix the .05/.20
quality failures. No natural self-preservation mechanism or globally impossible
intervention claim. Propose exactly one smallest next question appropriate to
the certificate, but do not start it.

Check standard usage before each batch; missing or >=90% means stop new work
and notify. No credits/resets/pushes/assistant-model setting changes. Preserve
all older files/verdicts and user-owned changes. Stop after one verified report
and clean commits in evidence/shared_direction_linear_feasibility_qwen35_08b.
