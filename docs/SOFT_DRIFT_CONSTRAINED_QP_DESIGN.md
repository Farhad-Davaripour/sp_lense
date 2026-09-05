# Soft paired drift with the original local COMPLY constraints

**DESIGN ONLY — stop for review.** One new whole-method adaptive-development proposal, motivated by the [retrospective negative](../evidence/paired_common_drift_clean01_retrospective_audit01/REPORT.md), not a recovered candidate or an isolated causal test of a penalty. That audit checked the saved arithmetic but found zero flips; the original recording remains INCONCLUSIVE / ARTIFACT_BYTE_CAP. No production solver, native benchmark, model run, serializer repair or new scientific lock is authorized here.

## One fixed convex QP

Keep the same twelve f01/f02/f03 renderings, six pairs, weights/hook/casts and fresh shared `w=0`. Pair COMPLY=A with COMPLY=B at the same family/display order: one-based `(3,1),(4,2),(7,5),(8,6),(11,9),(12,10)`. No f04 read or new prompt/data. Let `n_i=||h0_i||` be each prompt's own fixed original norm, `S_i=z_PRESERVE-z_COMPLY`, and current COMPLY margin `m_i=-S_i`. The existing local surrogate gives

```
A_i = -n_i * gS_i                    # dm_i/dw; same sign rule for every row
b_i = tau - m_i = .10 + S_i          # KEEP negative RHS; never max(0,b_i)
```

As before, this is the autograd local surrogate through the prescribed casts, not a derivative of discrete float rounding. A positive local change means movement toward COMPLY irrespective of which letter encodes it.

Within pair p, A/B subscripts mean **the letter that means COMPLY**, not display order. With `L=z_A-z_B`, `L_A=m_A`, `L_B=-m_B`, define

```
c_p = ((m_A-m_A0) - (m_B-m_B0))/2    # baseline-relative common LETTER drift now
D_p = (A_A-A_B)/2                    # its local shared-coordinate Jacobian
predicted residual drift = c_p + D_p*d
```

The baseline stays the original `w=0` baseline throughout; do not reset `c` at each step or penalize only `D*d`. Semantic movement proxy is `a=(Delta m_A+Delta m_B)/2`; neither proxy identifies a purified causal feature.

Choose **one fixed unitized penalty weight 1**, with `s0=.05` shared-coordinate units, `tau=.10` nats and P=6:

```
minimize_d  1/2 * ||d/s0||^2 + 1/(2P) * sum_p [ (c_p+D_p*d)/tau ]^2
subject to  A_i*d >= b_i,  i=1,...,12.
```

Both terms are dimensionless. A step of norm `.05` costs 1/2; residual RMS pair drift `.10` nats also costs 1/2. This convention uses only already-declared scales, not measured Jacobian magnitudes, successful vectors, outcome fitting or a coefficient sweep. It penalizes the **mean of squared drifts**, so opposite pairs cannot cancel. Drift is never a hard zero-drift, nonweakening, sign-reversibility or acceptance gate.

Multiplying by `s0^2` gives the equivalent QP `1/2*d^T H*d + q^T*d + constant`, with

```
kappa = s0^2/(P*tau^2) = 1/24        # shared-coordinate units squared / nats squared
H = I + kappa*D^T*D
q = kappa*D^T*c.
```

For every nonzero v, `v^T H v=||v||^2+kappa||Dv||^2>0`. Thus the feasible QP has one primal optimum (dual multipliers need not be unique). The feasible set is exactly the old signed-margin halfspaces: soft drift cannot repair inconsistent constraints. Setting the penalty to zero **as an algebraic check only**, not a proposed comparison arm, gives H=I,q=0 and exactly the earlier minimum-Euclidean-norm problem `min ||d||^2/2, Ad>=b`.

## Applied step and actual stopping stay separate

Solve for the raw local proposal d, then retain the original update geometry: `s=d*min(1,.05/||d||)`, `u=w+s`, `w_next=projection_to_radius_.20(u)`, `r=w_next-w`, and `path_next=path+||r||`. Handle zero d explicitly. Keep `.05` step, `.20` net, `.40` actual path and eight attempted-update ceilings, original cast/physical guards, fresh-zero identity and finite-stall handling. No line search, coefficient adaptation, extra derivative, best-iterate choice or renormalized endpoint.

`Ad>=b` certifies only the **raw local proposal**. Clipping can break positive-RHS constraints; radial clipping alone preserves feasible negative-RHS inequalities. Net projection changes the increment direction and can break either kind. Report `Ad-b`, `As-b` and `Ar-b`, proposed/applied margins and `c+Dd`, `c+Ds`, `c+Dr` separately. Do not describe the clipped/projected method as solving a hard trust-region QP.

Use the unchanged actual full-vocabulary COMPLY argmax, margin at least `.05-1e-6`, pair mass at least `.80`, and finite raw KL at least `-1e-6` for behavioral stopping; no new drift/loss gate or upper KL limit. Stop at first actual twelve-row acceptance; otherwise the finite failed endpoint or declared stall receives only the designated twelve final replays. Technical or unresolved-solver faults remain INCONCLUSIVE. Low local objective, feasible linear predictions, or projected norm bounds promise neither nonlinear feasibility nor actual loss descent. There is no claim that the new method will flip answers.

## Bounded solver/checker design, not implementation

Use one deterministic active-set enumeration of all4096 subsets, including the empty set. No native benchmark or production algorithm is implemented in this proposal. Let `d0=-H^-1 q` (the empty subset tests **d0, not zero**). For each independent active subset I:

```
G_I = A_I*H^-1*A_I^T
G_I*mu_I = b_I - A_I*d0
d = d0 + H^-1*A_I^T*mu_I
```

Check all twelve primal inequalities, nonnegative multipliers, stationarity `Hd+q-A^T mu=0`, and complementarity `mu_i*(A_i d-b_i)=0`. In exact arithmetic independent supports suffice even with redundant active constraints; first independently certified mask in fixed order is a deterministic tie policy, not candidate selection. Approximate floating-point KKT passes are not automatically exact ties: the future numerical certificate must establish its fixed accuracy bounds. SPD H allows Woodbury actions through the6x6 SPD matrix `I+kappa*D*D^T`; subset Gram systems are at most12x12, with no1024x1024 inverse. The optimum lies in the span of the twelve own-norm rows because every D row is a difference of two A rows.

A future checker must reconstruct A,b,c,D,H actions and KKT residuals independently from raw scores/recorded gradients, with separate fixed precision and scale-aware rank/primal/dual/complementarity tolerances prospectively certified. Do not blindly reuse old isotropic KKT code or its residual scaling. Original scoring/cast/geometry tolerances remain separate. Near-singular subsets or no passing numerical KKT certificate mean **SOLVER_NUMERICALLY_UNRESOLVED**, not proven infeasibility. A genuine Farkas witness would require independent `y>=0, A^T y=0, b^T y>0` certification; this proposal does not add a second witness-search solver or relax constraints. The toy contradiction below has an exact witness. Recorded-gradient arithmetic checks do not independently validate real derivatives.

## Hand-computable checks

The [toy checker](../scripts/check_soft_drift_constrained_qp_toys.py) uses exact fractions and supplied 2D answers/KKT witnesses, not an optimizer or native benchmark. All QP toys keep P=6 and pad unused pairs with zero Jacobians, c=0 and inactive b=-1; these are abstract numbers, not expanded prompt data. [Executed results](soft_drift_constrained_qp_toy_checks.json) record the bounded execution.

| Check | Hand result |
|---|---|
| Own-norm signs | n_A=2,gS_A=-1 gives A_A=2. n_B=4,gS_B=+1/2 gives A_B=-2 and letter Jacobian +2. A +.01 shared move changes both letters by+.02: c=.02,a=0. Changing gS_B to-1/2 gives semantic changes+.02,+.02: c=0,a=.02. |
| Unequal sensitivities | n_A=2,n_B=4,gS_A=-1,gS_B=+1 yields letter changes(.02,.04): c=.03,a=-.01. Replacing own norms by their mean wrongly changes those effects. Common drift is a response proxy. |
| Negative RHS and soft cost | Margins(.04,.20), baselines(-.06,.30), A=(1,-1), c=.10 give b=(.06,-.10), D=1. Unique raw optimum d=.06, multiplier1/15. Residual drift rises to .16; weaker second margin .14 remains locally above .10. Clamping negative b to zero falsely makes the interval infeasible. Clipping to .05 leaves the first predicted margin .09, below the local target. |
| Affine drift, empty coordinate, duplicate constraints | Pair1 A=(1,0),(-1,0), b=(-.05,-.05), c=1; pair2 A=(0,4),(0,4), b=(.10,.10), c=0. H=diag(25/24,1), q=(1/24,0). Optimum(-1/25,1/40), with one duplicate multiplier1/160 or two1/320; drift remains24/25. Pair1 predicted margins(.11,.19), pair2(.10,.10). Pair2 starts at zero margin, so the toy does not require optimizing after all rows already pass. |
| Zero penalty | For that same affine toy, H=I,q=0, optimum(0,1/40): the old minimum-norm solution exactly. Zero penalty is a requested limiting-case check, not a sweep. |
| Infeasible halfspaces | d_x>=.10 and -d_x>=.10 are impossible. y=(1,1) has A^T y=0 and b^T y=.20>0. A soft objective cannot change that. |
| Net-projection caveat | w=(.20,0), raw d=(0,.04) passes d_x>=-.001. Projection gives r_x=1/sqrt(26)-.20<-.001, so the negative-RHS inequality fails after projection. The exact comparison is1/26<(199/1000)^2. |

## Prerequisites and stop

Any future use is a newly approved **whole construction method**, not inheritance of earlier minimum-norm successes or isolated evidence about drift-penalty causality. Before any future run, separately authorize implementation, certify the bounded solver and independent checker, certify runtime/call accounting, and **certify complete compact audit serialization sizes through actual finalization**. Include vector-sized fake fixtures and every report/artifact category; reference raw vectors by hashes rather than duplicating full native arrays in update/KKT reports. The earlier artifact-cap failure remains immutable. No old readiness certificate transfers to a new method.

This delivery is only a short proposal and model-free toy-check commit, at most60 seconds aggregate numeric work, zero ML imports/loads/F/D. No f04, gate/controller/LoRA, production changes, model calls, data expansion, push, credits or reset. Publication remains40%. STOP for review.
