# Standalone soft-drift QP: pre-test numerical policy

**Implementation substep only.** This policy is written before targeted tests or the single native-shaped smoke. It implements only the [approved QP design](SOFT_DRIFT_CONSTRAINED_QP_DESIGN.md), not model integration, a scientific preregistration, a candidate, geometry application, full-pipeline finalization or serializer repair. Old evidence and sources remain immutable.

Inputs are exactly twelve finite binary64 own-norm COMPLY Jacobian rows A,12 signed RHS b and6 baseline-relative current drifts c, with1<=dimension<=1024. Pairs are zero-based `(2,0),(3,1),(6,4),(7,5),(10,8),(11,9)`. Fixed `D_p=(A_a-A_b)/2`, kappa=1/24, `H=I+kappa D^T D`, `q=kappa D^T c`. The exact problem is defined by the real values of the supplied binary64 A,b,c and rational kappa; the solver rounds arithmetic but the checker encloses it. Optional assembly uses A=-own_norm*gS, b=.10-current_margin and c=((m_a-m_a0)-(m_b-m_b0))/2. There is no penalty sweep or production zero-penalty switch.

## Fixed tolerances and conservative admission

Let u=2^-53, N=dimension+64, gamma=N*u/(1-N*u), and **tau=128*gamma**. Gamma is a conservative product/summation operation-count scale for a native dot plus small systems; the factor128 reserves additional rounding for the small-system and Woodbury operations. This is a prospective numerical convention, not a universal forward-error theorem for arbitrarily ill-conditioned problems. Independent residual validation and a separate rank filter are mandatory; no empirical/model-outcome calibration is used.

The following scales have floors of1 in the corresponding coordinate/objective units:

```
primal_i scale = max(1, |b_i|, sum_j |A_ij*d_j|)
stationarity_j scale = max(1, |d_j| + kappa sum_p |D_pj|*(|c_p|+sum_k |D_pk*d_k|)
                              + sum_i |mu_i*A_ij|)
complementarity_i scale = max(1, |mu_i|*max(|b_i|,sum_j |A_ij*d_j|)).
```

Normalize the primal violation max(0,b-Ad), stationarity `d+kappa D^T(c+Dd)-A^T mu`, and complementarity `mu*(Ad-b)` by these scales. **Acceptance requires every independently enclosed normalized upper bound <=tau/4.** The interval between tau/4 and tau is explicitly unresolved, not marginal acceptance. Negative multipliers are rejected even if tiny; inactive multipliers must be exactly zero. This certifies numerical KKT residuals within stated tolerances, not exact-real feasibility, derivative correctness, nonlinear success or an applied-step guarantee.

Rank policy is separate: rho=2^-26, approximately the square-root binary64 precision scale rounded upward to a power of two. The solver skips scaled Cholesky pivots <=rho for Woodbury and active Gram systems. Row/max-absolute scaling and unit-diagonal equilibration reduce unit effects but do not authorize dropping negative RHS. The independent checker tests Euclidean-normalized selected raw A rows by80-digit interval modified Gram-Schmidt, with **two total passes**; each squared residual-ratio lower bound must exceed4*rho. This stricter independent guard rejects borderline dependence. It is intentionally conservative; rank rejection or an absent accepted certificate means NUMERICALLY_UNRESOLVED, never a proof of infeasibility. No Farkas search, tolerance relaxation or rescue is implemented.

## Bounded deterministic algorithm and independent certificate

The solver computes d0=-kappa D^T(I+kappa DD^T)^-1 c and H^-1 actions through the6x6 Woodbury matrix. It enumerates masks in increasing integer order0..4095. Empty mask tests d0, not zero. Each independent active set solves `(A_I H^-1 A_I^T) mu_I=b_I-A_I d0`; all twelve constraints are checked, then full-coordinate residuals are checked for a proposed point. Return the first strictly admitted numerical estimate; a separately run checker may still reject it, in which case the result is unresolved, not permission to select another vector. Stop after at most4096 masks. A short one-character-per-visited-mask trace and counts replace per-mask native vectors. Only one returned solution vector and12 multipliers may be present. No clipping/projection is applied: raw-QP feasibility and the separately proposed applied geometry remain distinct.

The self-contained checker must not import the solver. It reconstructs D and direct gradient/residual expressions with 80-digit outward-rounded Decimal intervals, independently verifies a raw-input fingerprint and the active mask, and independently checks active-row rank. It neither uses solver Gram/Cholesky/Woodbury results nor trusts reported solver residuals. Fingerprint is SHA256 of the ASCII bytes `soft-drift-qp-v1` followed by one NUL byte, little-endian uint32(12,dimension), then little-endian float64 row-major A,b,c. The active mask is independent dual-support metadata: a marked row with zero multiplier need not be tight; KKT complementarity is checked directly. No current model, runtime or historical raw-data dependency.

## Declared tests and one fixed native-shaped smoke

Targeted small fake/math tests cover the supplied handwritten solutions, zero-penalty algebra only, negative RHS, affine c/nonzero empty-mask d0, redundant/near-opposed rows, nonfinite input, deterministic masks, unequal own norms/pair signs, and corrupted primal/dual/stationarity/complementarity/fingerprint certificates. No1024-dimensional timing test is hidden in that suite. Tests and ancillary numerical checks have an aggregate90-second budget; execution is bounded externally and no tolerance may be tuned after outcomes.

The **single** smoke is fixed here, before execution: dimension1024,12 dense Walsh rows `A[i,j]=(-1)^popcount(i&j)/32` for i=0..11,j=0..1023; `b_i=float((20+i)/200)`; `c_p=float((-1)^p*(p+1)/100)`. These dyadic Walsh rows are orthonormal. The intended full-support optimum has A_i*d=b_i and positive multipliers; it should visit all4096 masks. No seed, alternative instance or native timing sweep. One hard30-second subprocess cap includes imports, fixture construction, solve, independent check, compact output and size measurement. If slow or unresolved, report that limitation and do not rerun or enlarge the cap. The complete in-memory solver JSON must fit131072bytes, checker JSON32768bytes and persisted smoke summary16384bytes; output contains hashes and compact summaries, not a vector per mask. This test does not certify real-model runtime, efficacy, full audit serialization or the future complete pipeline.

Fresh usage must be available and below90% before batches. No ML/model/tokenizer imports/loads, forwards or derivatives, f04/new data, gates/controllers/LoRA, credits/reset/push, old-file edits or real run. Source/tests/report commit only, then STOP for review. Complete future compact audit-size certification through actual finalization remains an unfulfilled prerequisite for integration/running; this substep does not claim it.
