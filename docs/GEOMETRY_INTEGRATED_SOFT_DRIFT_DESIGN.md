# Put the existing geometry inside the local optimization

**DESIGN ONLY — new adaptive whole method; stop for review.** One convex formulation can enforce the geometry directly. However, it requires all twelve local `.10` targets to be reachable in **one permitted increment**. When that intersection is empty, it cannot take the partial-progress steps that the former raw-solve/clip/project method allowed. This is a stricter method, not merely a safer numerical implementation.

The preceding valid 9/12 result and all historical evidence remain unchanged. It motivates this question but does not establish that projection caused its nonlinear failures. No old-result re-audit, model call, production implementation or parameter fitting is part of this proposal. Publication remains 40%.

## One convex problem in the actual shared increment

Keep the same pinned model, original own-prompt norms, hook/casts, twelve f01/f02/f03 renderings and six one-based pairs `(3,1),(4,2),(7,5),(8,6),(11,9),(12,10)`. Let `m=-S` be current COMPLY margin, `m0` its original zero-vector baseline, and L the already charged actual path length. Preserve the original assembly/sign convention:

```text
A_i = -||h0_i|| * gS_i
b_i = .10 - m_i                       # signed; never clamp negative RHS
c_p = ((m_A-m_A0) - (m_B-m_B0))/2
D_p = (A_A-A_B)/2                     # A/B here means the COMPLY letter
rho = min(.05, .40-L),  R = .20

minimize_r  F(r) = (1/2)||r||² + (1/48)||c + D r||²
subject to  A r >= b
            ||r|| <= .05
            ||w+r|| <= .20
            ||r|| <= .40-L.
```

The two concentric increment balls are exactly equivalent to `||r||<=rho`; this is not an additional restriction. Require valid finite inputs, `0<=L<=.40`, `||w||<=.20`, and the existing net/path consistency checks; do not clamp invalid remaining path to zero. The objective is the same fixed soft affine-drift objective with kappa=1/24, evaluated at the applied increment. Its Hessian `I+DᵀD/24` is positive definite. The feasible set is convex and compact, so if nonempty it has a unique optimal r. Neither soft drift nor convexity guarantees nonemptiness.

Set `w_next=w+r` and charge `L_next=L+||r||`. **No subsequent clipping, net projection, line search or endpoint renormalization.** Numerical serialization is not exempt: before a model call, recover the exact dyadic difference of the stored `w_next` and current w (or its outward enclosure), then certify that actual increment against the original constraints and objective. Do not claim a rounded binary64 subtraction is necessarily that exact difference. If rounding invalidates admission, stop; do not repair it. Preserve the existing own-norm float32 casts and physical-integrity checks. Continuous local feasibility is not a guarantee about those casts or nonlinear model answers.

The local target stays **tau=.10**, with the existing floating-point assembly unchanged. Actual behavioral acceptance stays full-vocabulary COMPLY argmax, margin `>=.05-1e-6`, pair mass `>=.80`, finite KL `>=-1e-6` and original integrity guards. Reaching `.05` behaviorally is not equivalent to satisfying the `.10` local surrogate. A local infeasibility certificate may coexist with a reachable `.05` surrogate margin or possible multistep progress. Changing tau, adding margin slacks or relaxing any radius requires separate approval.

## Low-rank reduction must include w

Let V=span of the twelve A rows and w, dimension at most 13 even at native dimension 1024. For `r=v+z`, with v in V and z perpendicular to V, `Ar=Av`, `Dr=Dv`, while

```text
||r||² = ||v||² + ||z||²
||w+r||² = ||w+v||² + ||z||²
F(r) = F(v) + ||z||²/2.
```

Removing z preserves feasibility and improves the objective unless z=0. This proves the reduction; restricting only to the gradient span does not.

More explicitly, let U be an orthonormal basis for span(Aᵀ), write `w=U a+w_perp`, and let `beta=||w_perp||`. If beta>0, put `e=w_perp/beta` and represent `r=U y+t e`. Then

```text
||r||² = ||y||²+t²
||w+r||² = ||a+y||²+(beta+t)²
Ar = A U y,  Dr = D U y.
```

The extra variable t allows inward motion outside the gradient span. If beta is exactly zero, omit t. A numerically uncertain small component or rank is not permission to discard it: certify the span representation or report unresolved. Independently check the returned point and dual residuals in the original native coordinates.

## One bounded solver and independent certificate plan

This is a feasibility plan, not implemented or benchmarked readiness. Propose **one** deterministic homogeneous-self-dual primal-dual SOCP predictor-corrector implementation, binary64, at most 200 iterations, fixed .99 fraction-to-boundary damping, and a hard 10-second combined solve/check deadline per proposal. No restart, alternative solver, precision escalation, target-slack search or rescue portfolio. Any future integrated worker retains its existing total 1800-second and 216F/96D/eight-update ceilings; this plan does not promise they will suffice.

Use reduced variables (at most 13) and two rotated-cone epigraphs: `(s,1,r)` enforces `s>=||r||²/2`; `(t,24,c+Dr)` enforces `t>=||c+Dr||²/48`; minimize s+t. These are objective epigraphs, not permitted constraint violations. Add the twelve halfspaces and two ordinary SOC balls. No 1024x1024 matrix or inverse is needed. Basis construction is bounded by thirteen input vectors; ambiguous numerical rank fails closed rather than silently deleting a direction.

A separately written checker must reconstruct A,b,c,D,w,L and the serialized increment, not trust solver factors/basis residuals. Use outward intervals, with exact dyadic/rational checks where needed for boundary comparisons. **Primal admission must establish the original halfspaces and balls without target or radius dilation.** An ambiguous enclosure is unresolved, not a small accepted violation. Prospective stationarity/duality-gap and rank tolerances must be specified before implementation tests; the old unconstrained-QP numerical policy is not automatically valid for these cones.

For optimality, use lambda>=0 and SOC dual pairs `(alpha,u)`, `(gamma,v)` with `alpha>=||u||`, `gamma>=||v||`. Check

```text
e = r + Dᵀ(c+Dr)/24 - Aᵀlambda + u + v
complementarity: lambda_i(Ar-b)_i, rho*alpha-uᵀr,
                 R*gamma-vᵀ(w+r).
```

SOC duals handle a zero step radius more honestly than relying only on squared-ball gradients, which vanish at r=0. Define the Lagrangian value

`B=F(r)+lambdaᵀ(b-Ar)+uᵀr-rho*alpha+vᵀ(w+r)-R*gamma`.

Since the Hessian is at least I, `B-||e||²/2` is a dual lower bound. With independently proven primal and dual feasibility, an outward bound on `F(r)-(B-||e||²/2)` certifies an objective gap without reusing a solver inverse. Tolerance-only primal feasibility would support only a numerical consistency diagnostic, not this rigorous gap claim; it is not an admission workaround.

**One no-step branch:** if no admitted primal/optimality certificate is returned, stop and retain the reason. Label it *certified local infeasibility* only if the same solver's dual witness satisfies

```text
lambda>=0, u+v=Aᵀlambda, alpha>=||u||, gamma>=||v||,
lambdaᵀb + vᵀw - rho*alpha - R*gamma > 0.
```

The strict inequality requires an independently positive lower bound. Prove the linear equality exactly; deriving `u=Aᵀlambda-v` from original dyadic inputs avoids an equality-tolerance loophole. Its contradiction follows from `lambdaᵀb <= uᵀr+vᵀr <= rho*alpha+R*gamma-vᵀw`. Otherwise the reason is **NUMERICALLY_UNRESOLVED / INCONCLUSIVE**, including iteration/deadline limits, ill-conditioning or an ambiguous witness. Neither reason permits a replacement step, lower target, larger ball or automatic rerun. Tangency and failure of Slater conditions can prevent a usable finite optimality certificate even for a feasible problem. A reported unbounded objective is an unresolved/error condition: valid F is nonnegative and the primal domain is bounded.

## Independently checked boundary toys

These are supplied 2D rational witnesses, not optimization runs or new prompt data. All use twelve rows, six fixed pairs and kappa=1/24; unused rows have A=0,b=-1 and unused c=0. For D=0 cases, the used pair has identical A=e1 rows. The [exact Fraction checker](../scripts/check_geometry_integrated_soft_drift_toys.py) verifies feasibility, supplied KKT witnesses or strict separation. Independent hand review preceded execution; [bounded results](geometry_integrated_soft_drift_toy_checks.json) record it.

| Toy | Hand result |
|---|---|
| Affine drift, negative RHS | A_A=e1,A_B=-e1; b=(.03,-.10),c=.10,w=0. Unique optimum r=(.03,0), multiplier17/480; drift becomes .13, not a required decrease. Current margins(.07,.20) become(.10,.17). |
| Step boundary | w=0,r_x>=.05. Unique optimum r=(.05,0), step²=1/400, multiplier .05. |
| Step infeasible | w=0,r_x>=.06. Witness lambda=1,u=e1,alpha=1,v=0 gives gap .01. With current margin .04, r_x=.01 could reach surrogate .05, but not the unchanged .10 aim. |
| Remaining path infeasible | L=.38,r_x>=.03,w=0. rho=.02; the same witness gap is .01. |
| Outward net infeasible | w=(.20,0),r_x>=.01. Witness lambda=1,v=e1,gamma=1,u=0 gives gap .01. |
| Gradient span alone fails | w=(0,.20),r_x>=.03. r=(.03,-.01) is feasible: step²=.001,net²=.037. Every gradient-span-only increment has r_y=0 and net²>=.0409, so is infeasible. |
| Active net boundary | w=(0,1/5),r_x>=4/101. Unique optimum r=(4/101,-2/505), step²=4/2525<1/400,net²=1/25. Halfspace multiplier4/99 and multiplier2/99 for the HALF-squared net constraint certify stationarity exactly. |

All seven exact witness checks passed in one 0.062606-second subprocess under a hard 10-second cap. Including scoped formatting/lint, measured targeted subprocess time was 0.344598 seconds, below the 60-second allowance. No optimizer, native benchmark or model executed; no test was rerun or limit enlarged. Usage was 62%; the updated unfrozen-work budget policy does not alter any existing frozen usage guard.

Any implementation, native benchmark, runtime/call/recording certification or real experiment requires a separate approval and new method identity. No old readiness certificate transfers. This delivery authorizes only the proposal and supplied-toy-check commit, at most 60 seconds aggregate model-free arithmetic, zero ML loads/F/D, no f04/new data, old-result re-audit, gate/controller/LoRA, source/evidence repair, push, credits or reset. **STOP after proposal review.**
