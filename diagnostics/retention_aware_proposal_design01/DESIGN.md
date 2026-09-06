# Retention-aware local proposal — design only

Two-page design; fixed invented witnesses are in `inputs.json` and `witness.py`.
No solver implementation, model work, historical rescoring, or efficacy claim.
This changes proposal geometry, not objective coefficients or scientific gates.

## Exact specification

Keep twelve rows, zero-based pairs `(2,0),(3,1),(6,4),(7,5),(10,8),(11,9)`,
original-zero margins `m0`, current COMPLY-minus-opposed margins `m=-S`,
and original-own-norm slopes `A_i=-||h0_i||gS_i`. Define

```text
b_i = 1/10-m_i; c_p = ((m_A-m_A0)-(m_B-m_B0))/2; D_p = (A_A-A_B)/2
J(r) = ||r||²/2 + sum_i[b_i-A_i r]_+²/24 + ||c+Dr||²/48
g0 = -Aᵀ[b]_+/12 + Dᵀc/24
rho = min(1/20, 2/5-L); R=1/5
C_P = {r: ||r||<=rho, ||w+r||<=R, m_i+A_i r>=tau for every i in P}
```

`P` is fixed once: ORIGINAL baseline rows passing unchanged actual COMPLY
acceptance. No newly accepted rows are added. `tau` is the exact rational
represented by the existing binary64 expression `0.05 - 1e-6` (approximately
`.049999`), not a replacement decimal or a no-current-margin-loss rule.
Authenticated stored margins/slopes are interpreted exactly for certification.
Opposed deficits remain soft; neither all twelve `.10` targets nor individual
deficit improvement is required. Protected rows may spend margin above `tau`.

Zero is feasible when authenticated history supplies valid `w`,
`0<=L<=2/5`, `||w||<=R`, a consistent conservative cumulative-path bound,
and **current** `m_i>=tau` for every original protected row. Original membership
alone does not establish current feasibility after nonlinear updates. Invalid
inputs terminate; zero feasibility does not guarantee a useful descent step.

## One bounded future route (not implemented here)

Reuse the certified-descent method's outward
`M>=1+||A||F²/12+||D||F²/24` and single target `y=-g0/M`.
The proposal problem is `min_(r in C_P) g0·r + M||r||²/2`, equivalently projection
of `y` onto `C_P`: a quadratic majorizer of the unchanged `J`.

Approximate that projection with exactly **32 full Dykstra cycles**, ordered
as the original step/net two-ball intersection, then protected halfspaces in
ascending row order. Initialize `x=y` and every correction `q_k=0`. Each set
update is `t=x+q_k; x=P_k(t); q_k=t-x`. Reuse the verified analytic two-ball
projection and its unchanged finite/multiplier/residual screens. For halfspace
`a·x>=ell`, project by `t+max(0,(ell-a·t)/(a·a))*a`, `ell=tau-m_i`.
A zero normal is identity when satisfied; otherwise fail without a step.

For the future floating proposal, freeze binary64 nearest-even operations:
new scalar products accumulate coordinates left-to-right from zero; multiply,
add, subtract and divide round separately, without fused operations. Compute
rounded `ell`, dot products, subtraction, quotient, maximum, then coordinate
updates in that order; correction arithmetic is coordinatewise. Retain the
existing two-ball routine's internal arithmetic unchanged. Nonfinite values,
nonpositive computed squared norm for a nonzero normal, or projection-screen
failure terminate. Fixed cycles are only proposal generation, not a convergence,
exact-projection or near-optimality certificate. Intermediate/final floating
iterates can violate other sets; there is no post-proposal clipping or repair.

Reuse only the existing 53 scales `2^-j`, `j=0,...,52`, and serialized endpoints
`RN64(w+RN64(2^-j*x))`. Recover the **exact difference of stored endpoints** as
`r`. Independently verify both original balls, actual remaining-path charge,
and every exact protected inequality. Then require unchanged

```text
s=-g0·r>0; Delta=J(0)-J(r)>s/4+eta; eta=2^-30*max(1,J(0)).
```

Use exact arithmetic or rigorous outward bounds. Return the first certified
nonzero endpoint; keep the existing **10-second total** through serialization.
No second direction, fitted threshold, near-optimality gate, or precision retry.
Failure/exhausted scales/rounded zero yields NO_CERTIFIED_STEP, not infeasibility,
stationarity, or model impossibility. The prior exact zero-only/zero-gradient
proof branches remain; the toy proof below adds no production status branch.

## Fixed witness and limits

Both invented two-dimensional cases have `w=L=c=0`, twelve rows, stipulated
`P=all except row2`, `A0=(-1,0)`, `A2=(1,0)`, other slopes zero and margins
`1/5`. First set `m0=3/50`, `m2=-1/5`, and supply `r=(1/100,0)` independently
of any solver. Protected margin becomes `1/20>tau`; opposed margin improves
to `-19/100`. Geometry passes. `g0=(-13/600,0)`, `M=29/24`,
`J(0)=229/60000`, `J(r)=1757/480000`, `Delta=1/6400`,
`s=13/60000`, and `Delta-s/4=49/480000>eta`.
The old target `y=(13/725,0)` fits the balls but violates retention.
Demanding both rows reach `.10` instead requires `r1<=-1/25` and
`r1>=3/10`: infeasible in this SAME example.

Second, change only row0's original/current margin to `tau`. Retention forces
`r1<=0`, while `g01=-(1/5+tau)/12<0`, hence `g0·r>=0` for every feasible
`r`. Squared hinges and drift are convex, so `J` is 1-strongly convex:
`J(r)>=J(0)+g0·r+||r||²/2`. Thus zero is the unique constrained optimum
of this affine toy. This is an analytic certificate, not sampled directions
or a global claim about model behavior.

Linearized C-minus-P protection does **not** ensure nonlinear margin,
full-vocabulary argmax, mass or KL. A separately authorized future attempt
must check every original protected row after each application using unchanged
actual gates: COMPLY argmax, margin `>=.05-1e-6`, pair mass `>=.80`, finite
KL `>=-1e-6`, and integrity. Any failure terminates without a candidate or
behavioral retries. Newly flipped rows have no added intermediate protection;
unchanged final twelve-row acceptance/replays still apply. Missing eligibility
is UNTESTED. Both intended semantic outcomes/orders and tested ordinary-task
preservation remain the broader target, not established here.

If the single checker passes, recommend only a separately authorized,
model-free implementation/test of this proposal plus independent retention
certificate, using the same fixed toys. Do not run it in this job.
