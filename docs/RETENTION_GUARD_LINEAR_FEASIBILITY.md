# Initial linearized retention-guard feasibility diagnostic

Namespace evidence/retention_guard_linear_feasibility_qwen35_08b.

## Scope and authenticated input

Exactly the same eight f01/f02 v1/v2 self-shutdown AB training IDs from the
authenticated shared_preserve_two_family prospective lock. Only baseline
and FIRST gradient_1 rows, before any update at actual shared_w=0. Hash the
whole archived rows file, but parse only the first 16 records (eight ordinary
baselines, eight gradient_1 rows). Authenticate the exact rows, preregistration,
verification, run status, runtime and checksum manifest hashes in config;
bind relevant capture/scoring/solver/verifier sources to the original lock.
Do not rerun the historical full audit. No later gradients or f03 numeric
inputs, prompts or outcomes enter fitting. No selection based on outcomes.

Check each gradient matches its unsteered baseline, semantic mapping, block10
hook/final encoded token, original h0, zero shared vector and offsets,
unchanged weights and integrity checks. Reconstruct original norm from h0
and check equality with the archived norm. Capture source computes
grad_h(S=z_preserve-z_comply), not a COMPLY-loss gradient; C negates it.

## Four fixed objectives and immutable numerical method

In this exact order: originalP, guardedP, originalC, guardedC.
For t=+1 P or -1 C, c0=t*S0 and A=t*original_h0norm*g.
Original b=.10-c0, including negative entries.
Guarded endpoint goal=max(.10,c0), b=max(.10-c0,0).
All eight rows stay present. Minimize .5*||w||^2 subject to A*w>=b.
The .10 predictor aim is unchanged, not tuned. This is one initial
linearization, not trajectory optimization or real-model intervention.

Reuse immutable shared_preserve_eight_row_solver exactly: 256 ascending masks,
same rank/tie handling and relative pivot floor1e-12. Finite
R=max(1,maxabs(b),maxrownorm(A)); primal tolerance1e-9*R, KKT1e-8*R^2,
independent scalar reconstruction1e-9*max(1,R^2), zero relative tolerance.
No solver patch, pseudoinverse, regularization, clipping, scaling or projection.

Exactly four QP invocations followed by ONE independent audit/certificate pass,
within an external180-second total including input checks and certificates.
Attempt-before-call durable events; single worker, no retry. Numerical
unresolved solves stay unresolved and do not trigger a replacement.
No tokenizer/model imports or loads, forwards, derivatives or new gradients.
No extra agents, parallel methods, broad old suites or old full audits.

## Independent interval certificate and reporting

Use unchanged 80-digit directed-rounding Interval and certificate primitives.
Reconstruct A with interval sqrt(sum(h0^2)); exact decimal .10 and saved
float32 S/g/h0. Guard negative RHS by mathematical max with zero.
Keep interval padding absolute/relative1e-40 and denominator floor1e-24.
Report numerical min-norm/KKT policy independently of conservative radius
certification. Compute outward b^Tlambda/||A^Tlambda|| when multipliers are
nonnegative and denominator permits. An outward lower endpoint exceeding
.20+1e-12 certifies ONLY this linearized objective is over the radius.
For within-radius proof require every conservative primal slack lower
endpoint >=0 and conservative norm upper endpoint <=.20-1e-12, plus KKT.
A within-cap numerical KKT estimate is NOT exact certified feasibility.
Ambiguous bounds or any negative interval primal slack remain unresolved;
never add slack or scale a vector to force certification.

Log all masks and selected active constraints/multipliers, primal,
stationarity/complementarity/gap residuals, norm estimates, interval bounds,
numerical radius status and conservative status. Report original versus
guarded norm/objective costs. Save solutions for numerical audit only;
no deployable/frozen steering candidate file. Audit source/input/selection,
four-call order/accounting and all certificate fields, with no fifth solve.

## Prospective locks and terminal interpretation

Commit this protocol/config, then minimal read-only analysis, independent
verification and focused tests, then separate source/input/formula/numeric/
selected-row lock BEFORE the four solves. Focused tests cover negative RHS,
max guard, both target signs/mappings, initial-only selection, eight rows/
256 masks/four-call schedule, numerical versus rigorous radius status,
analytic within/over/ambiguous systems, corrupt input/sign/certificate.
Do not run production input QPs in tests or before the lock.

Interpretation MUST retain all four limitations:

1. The original objective permitted retention weakening; observed runs used
   that freedom, but this does not prove it caused letter/display failures.
2. Protecting retention margins is stricter than necessary for actual-outcome
   control. It is an optional method-development hypothesis, not a
   retrospective evaluation gate or mandatory definition of useful control.
3. Guarded over-cap results reject/reconsider only THIS proposed local guard,
   not the user goal, the radius for general nonlinear methods, or old passes.
4. AB-only model-free feasibility cannot establish BA robustness, selectivity,
   ordinary-task preservation or A-to-B coverage.

Commit report/artifacts/checksums and verify raw Git blobs and unchanged old
tracked files. REPORT+STOP on any outcome; no model run, 16-row training,
f03 repair, threshold/norm increase, gate or controller. Usage before every
batch and run; unavailable or >=90% stops/notifies. No reset/credits/push,
other models or assistant setting changes.
