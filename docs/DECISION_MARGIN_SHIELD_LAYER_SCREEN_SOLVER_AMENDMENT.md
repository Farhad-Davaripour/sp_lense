# Decision-Margin Shield layer-screen solver amendment

## Status and purpose

This is a **model-free numerical amendment** to the locked Decision-Margin
Shield (DMS) v2 layer screen. It does not replace or edit the original lock,
protocol, capture, source, or reported failure. It exists because the original
full-width SLSQP solver raised
`TangentShieldSolverError("the minimum-L2 solve did not converge")` before it
could write a layer-screen result.

The immutable failure record is
`results/decision_margin_shield_layer_screen/qwen35_08b/locked_screen_attempt_failure.json`.
The first diagnosed failing cell was
`fcag_dev_01_weather_alert` / zero-based layer 22 / `unrelated_only`; the next
`decision_margin_shield` solve also fails in the old solver.

This failure is numerical, not a construction no-go. No result exists at the
original `SCREEN_RESULT_PATH`.

## Outcome-awareness disclosure

The amendment was designed after a read-only diagnosis viewed partial
calibration geometry: weather-scenario norms for layers 0 through 21 and the
layer-22 failure. No pilot-scenario geometry, finite intervention outcome,
generated text, or sealed test result was viewed. The amendment is therefore
outcome-aware to that partial calibration geometry. This limitation must remain
attached to every amendment result.

The numerical method changes, but the scientific design does not:

- the same immutable 136-record capture is reused;
- layers are exactly zero-based 0 through 22;
- the same four calibration scenarios are screened;
- no pilot geometry is computed;
- the same three methods are evaluated in the same order: `unshielded`,
  `unrelated_only`, and `decision_margin_shield`;
- target, protected, and unrelated rows are unchanged;
- target margin remains 0.05;
- protected slabs remain `max(abs(b) - 0.05, 0)` per row;
- exact unrelated cancellation remains exact;
- the cap frontier remains 1.0, 1.5, and 2.0;
- a layer qualifies only if all four calibration DMS norms are at most 2.0;
- tie-breakers remain smallest worst-case norm, then mean norm, then layer;
- no pilot construction, finite intervention, or generated response is run.

## Numerical amendment

Every constraint row lies in a space of at most 24 rows but has 1,024 residual
coordinates. The amended solver reduces each problem to the row space before
optimization and independently certifies the reconstructed full-coordinate
candidate. This is an algebraic reduction of the same convex minimum-L2
problem, not a relaxation, new prompt, new margin, new cap, or layer choice.

The exact-equality projection retains the original frozen SVD rank rule. The
projected inequality representer span instead uses the standard
machine-precision threshold `eps * max(shape) * largest_singular_value`; using
the scientific equality tolerance here could discard a small direction that is
necessary for feasibility. Row-space reconstruction and orthonormality must
both pass before optimization.

Every candidate must then pass two independent checks in the original log-odds
units. First, a strict raw-coordinate certificate checks target margins, exact
unrelated rows, and positive protected slabs with a tolerance no larger than
`1e-7` of the relevant fixed margin or bound (plus a machine-roundoff floor that
does not scale with the constraint operator norm). Second, the unchanged DMS
primal-dual/KKT certificate must pass. A feasible-looking optimizer point,
including SLSQP status 8, is accepted only if both checks pass. A numerical
infeasibility status without an independent infeasibility proof is reported as
numerically indeterminate and cannot become a scientific construction no-go.

The required API is
`sp_lense.decision_margin_shield_rowspace.solve_certified_rowspace_minimum_l2_direction`.
It must fail closed if primal feasibility, reconstruction, optimality, or its
certificate does not pass. The amendment records the exact solver diagnostics
and hashes each input row matrix, offset vector, direction, geometry record,
selection, and final result. Bitwise determinism is claimed only within the
pinned runtime; no cross-BLAS bitwise guarantee is made for degenerate SVD
subspaces.

## Immutable provenance boundary

Before it can run, the amendment lock binds and the runner validates:

1. the original lock file hash and lock identity;
2. the original capture-manifest file hash and self-hash;
3. the exact ordered capture-chunk path/hash/count inventory;
4. the original source commit and current hashes of the original runner,
   geometry module, and protocol;
5. the self-hashed original failure record;
6. the amendment protocol, runner, row-space solver, and tests;
7. the dataset/model-config hashes and frozen capture-plan/prompt hashes;
8. the absence of the original `layer_screen_result.json`.

Any mismatch fails closed. The amendment never calls `load_backend`, capture
helpers, generation, or an external API. Its compute ceiling is zero model
forwards, zero model backwards, and zero generated tokens.

## Commands and artifact policy

Run in order:

```powershell
.\.venv\Scripts\python.exe scripts\decision_margin_shield_layer_screen_solver_amendment.py lock
.\.venv\Scripts\python.exe scripts\decision_margin_shield_layer_screen_solver_amendment.py preflight
.\.venv\Scripts\python.exe scripts\decision_margin_shield_layer_screen_solver_amendment.py screen
.\.venv\Scripts\python.exe scripts\decision_margin_shield_layer_screen_solver_amendment.py report
```

The new lock is written once to
`configs/decision_margin_shield_layer_screen_solver_amendment_lock.json`.
Preflight artifacts live under
`artifacts/decision_margin_shield_layer_screen_solver_amendment/qwen35_08b`.
Results live under
`results/decision_margin_shield_layer_screen_solver_amendment/qwen35_08b`.
Existing artifacts are validated and never overwritten.

## Claim boundary

This amendment can only report the locked calibration geometry and, if one
qualifies, the frozen selected layer. It does not establish a finite steering
effect, nonlinear decision preservation, full-vocabulary stability, unchanged
capability, a natural self-preservation mechanism, safety, priority, or
publication novelty. A selected layer would authorize a separately
preregistered finite pilot; it would not itself be behavioral evidence.
