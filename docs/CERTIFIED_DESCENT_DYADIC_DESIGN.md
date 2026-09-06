# Certify descent, not a nearly optimal local solve

**DESIGN ONLY — new method identity `certified_descent_dyadic_v1`; stop for review.** Preserve failed prototype `a714ea2dca0305a2015d15176b234ee2eff66148` unchanged. This revision changes numerical generation/admission, not the objective or scientific acceptance. It proposes one certified local decrease without demanding a nearly optimal solution first; that decrease need not be practically meaningful model movement. There is no native benchmark, real-gradient evaluation, production implementation, or readiness claim here.

## Same objective; different numerical question

Keep signed `b=.10-m`, `m=-S`, original own-norm slopes `A_i=-||h0_i||gS_i`, original-zero baselines, and six COMPLY-letter pairs `(3,1),(4,2),(7,5),(8,6),(11,9),(12,10)` across the same twelve f01/f02/f03 renderings. Let `c=((m_A-m_A0)-(m_B-m_B0))/2`, `D=(A_A-A_B)/2` and

```text
J(r) = ||r||²/2 + sum_i[b_i-A_i r]_+²/24 + ||c+Dr||²/48
C = {r: ||r||<=rho=min(1/20,2/5-L), ||w+r||<=R=1/5}.
```

All coefficients, pair signs, nominal exact-decimal balls and `.10` training targets stay fixed. A future separately authorized run retains pinned Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`, CPU float32, block 10 final-token hook/casts and fresh-zero initialization in a new namespace. No prior endpoint is reused. No per-row weights, coefficient fitting or sweep.

## One exact algorithm specification

1. Authenticate original inputs/current w/history and a conservative path upper bound L. Reject invalid inputs; derive objective arithmetic independently from them. If rho=0, zero is the only feasible increment. If exact `g0=gradient J(0)=0`, 1-strong convexity proves zero is the unique optimum. These are the only automatic exact-optimum branches.
2. Form the same outward upper bound `M>=1+||A||F²/12+||D||F²/24`. Compute **one** nominal floating proposal `p=P_C(-g0/M)` using the prior analytic two-ball projection and its unchanged finite/nonnegative-multiplier/residual screens. Those screens do not establish exact feasibility. Do not run a sequence of PG updates, optimize a multiplier, or choose an alternative direction.
3. Use the fixed list `lambda_j=2^-j`, **j=0,...,52**, in that order. Construct each endpoint by the declared binary64 operations `w_next_j=RN64(w + RN64(lambda_j*p))`. Every endpoint is representable; it is not necessarily feasible. Recover `r_j` as the **exact rational difference of stored w_next_j and authenticated w**, not a rounded subtraction or nominal lambda*p.
4. Independently check the original exact step/net balls and conservatively charge `L+||r_j||` from that actual displacement. Then apply the sufficient-decrease certificate below. Return the **first** passing endpoint, recording the selected scale, exact displacement, attempted-scale dispositions, and separate before/after deficit, drift and proximal costs. A failed geometry/gain check advances only to the next predeclared lambda; it never permits a changed direction, different interpolation family or radius dilation.
5. At most **53 endpoint trials** and **10 seconds combined** for authentication, proposal, all checks, diagnostic and result serialization. If no trial passes, projection arithmetic fails, the endpoint repeats w, or the deadline is reached, return **NO_CERTIFIED_STEP** with the reason. No movement, restart, precision escalation, coefficient change or automatic experiment follows. An unchanged endpoint ends this ray's search: smaller positive scales cannot restore a lost component under the declared monotone nearest-even rounding on this fixed ray. Unless branch 1 supplies an exact proof, this is not a stationarity or infeasibility claim.

This is one fixed finite feasibility/decrease backtracking rule, not a portfolio of line searches. The 52 halvings span the binary64 fraction-bit range (53 significand bits); this is a prospective bounded-work convention, **not** a theorem that a useful point exists within it. The 10-second limit is retained, not enlarged. Projection screens retain strict multiplier signs and residual factor `128*d*2^-52*max(1,||y||,||p||,hypot(||n_s||,||n_n||))`; they never enlarge geometry. Any future timed implementation must enforce the overall bound through output completion.

## Sufficient decrease with an explicit strict margin

At each actual serialized r define

```text
s = -g0·r
Delta = J(0)-J(r)
eta = 2^-30*max(1,J(0))                 # unchanged numerical gain-floor convention
admit only if geometry/path pass, s>0, and Delta > s/4 + eta.
```

Use exact rationals or outward intervals, including the same independently bounded g0 and the actual serialized r. With intervals require `s_lower>0` and `Delta_lower > s_upper/4 + eta_upper`; do not subtract approximate objective values and call their difference certified. Unknown arithmetic/serialization error cannot be absorbed by merely calling it small. Eta is the previously declared minimum scale-aware numerical gain, not a fitted tolerance, an error-bound theorem, evidence of practically useful model movement, or a scientific margin. It remains an additional strict margin after numerical errors have been bounded.

The quarter has a mathematical origin. For an exact projected step p from a smooth convex J with Lipschitz bound M and feasible zero, projection gives `-g0·p>=M||p||²`. Along an exact feasible lambda*p, the descent lemma gives `Delta >= (1-lambda/2)*s >= s/2`. Requiring more than s/4 leaves half of that worst-case decrease available for the fixed strict margin and serialization effects. This argument motivates the constant; it does **not** replace checking the actual serialized point, nor does it prove a floating proposal satisfies the projection premise. Small s can legitimately fail the unchanged eta floor.

For the selected endpoint report a **descriptive upper bound**, `G_desc=||gradient J(r)||²/2`, on `J(r)-min_C J`. This follows from 1-strong convexity with zero support vectors; it can be very loose, especially on active boundaries. Do not call it the actual gap, demand it be below eta, or label the endpoint optimal. Compute/report it once for the selected point, not in pursuit of a near-optimal solution. If the combined deadline prevents a complete checked return, no step is issued.

Certified descent concerns the whole fixed local objective. It does not guarantee that every deficit decreases, that drift decreases, that a decision flips, or that the nonlinear model improves. The minimum numerical-gain convention and the remaining optimality gap answer different questions; neither substitutes for behavioral acceptance.

## What interpolation does and does not guarantee

If exact p belongs to C, convexity and feasibility of zero imply lambda*p belongs to C for 0<=lambda<=1. A floating p that is slightly outside C need not meet this premise. Power-of-two multiplication is exact only absent underflow; the addition to w still rounds. The endpoint must therefore be checked afresh, and the actual norm—not lambda times a nominal norm—is charged to path.

At an exact mathematical net boundary, `||w+lambda*p||²-R²=2lambda(w·p)+lambda²||p||²`. A nonzero tangential or outward p cannot be made feasible by shrinking alone. An inward p can become feasible for small enough lambda in exact arithmetic; neither representable progress nor success within 53 trials follows. Finite dyadic coordinates cannot have squared norm exactly `1/25`, but this boundary analysis also explains the limiting/near-boundary behavior that serialized inputs approach. Tiny proposals may serialize to zero or lose their sufficient gain. No-step is then honest.

Compared with `a714ea2`, the old 200 successive PG iterates and near-optimality gate are removed. Trying smaller declared lambda after a geometry rejection is an **explicit new solver behavior**, previously forbidden. It can materially change the local proposal and its gain. It is not an undocumented repair of the failed prototype. The much earlier zero-flip paired-gradient method also took one gradient-like step, but had a different objective/relative drift weight and post-update geometry handling; this revision makes no claim that returning to a single proposal will solve its scientific failure.

## Bounded illustrative checks, not an implementation

The [tiny checker](../scripts/check_certified_descent_dyadic_toys.py) verifies supplied rational/dyadic witnesses only; its complete fixtures and constants are fixed before the sole bounded execution. It reports weighted deficit/drift/proximal contributions separately. [Results and execution ledger](certified_descent_dyadic_toy_checks.json) are retained independently of this proposal.

| Supplied witness | Construction and intended distinction |
|---|---|
| Affine interior | A=e1,-e1; b=.1,-.1; c=.1; w=.1e1. Exact M=29/24 gives p=1/290 e1. In units1/4036800 at that rational p, deficit cost drops114, drift cost rises59, proximal cost rises24: total gain31. Actual serialized arithmetic is checked separately. |
| Both active boundaries | p=(.03,.04), endpoint z=(.16,.12), w=z-p. Both are 3-4-5 circles. Projection multipliers1,1 give y=2p+z; identical A=(11/5,2) and derived b=371/250 produce this y. The rounded full endpoint fails, while the declared half-scale endpoint is checked for admission. |
| Stationary with deficits | Opposing unit slopes, b=.1, c=w=0 give exact g0=0 and positive remaining deficits. No movement is forced. |
| Erased nominal gain | Identical A=2^56e1, b=1, w=1/8 e1. A supplied rounded exact step is a half-ULP; even-endpoint rounding erases nominal gain of approximately1/12. Actual gain is zero. |
| Large-M logic | Active identical e1 rows, b=.1; a different inactive pair has identical 512e2 slopes, b=-1. M=262151/6 and p=1/2621510 e1 give sufficient descent despite a large descriptive gap upper bound; that bound alone does not establish actual suboptimality. A second fixed power-of-two example with inactive4096e2 has a first-order ray gain ceiling below eta; shrinking cannot make it pass. These are input fixtures, not tuned method coefficients. |
| Boundary limit | Rational w=1/5 e1 with tangent p=.01e2 has a positive quadratic net-norm excess for every positive scale. This is a geometric/error-limit counterexample, not a feasible exact projection. An inward p=-.01e1 can satisfy geometry but still needs a gain certificate. |

Projection KKT witnesses refer to exact rational mathematics; supplied rounded proposals do not establish the output of an unimplemented future floating generator. All fixture numbers come from the fixed coefficients, powers of two or elementary rational geometry, not model outcomes. Do not read or rerun the old native fixture.

The design/toy allowance is **40 seconds aggregate model-free arithmetic**, with one bounded toy run and no production/native work. Disjoint agents supply hand-witness checks and source/edge-case comparison; the parent integrates and executes. Final actual acceptance remains all twelve full-vocabulary COMPLY argmaxes, margin `>=.05-1e-6`, pair mass `>=.80`, finite KL `>=-1e-6`, original integrity checks and matching final replays. Local slack/descent never waives these; no new model result or publication progress is claimed. No f04/new prompts, old-data re-audit, pipeline, gate, LoRA, push, credits or reset. **STOP for review.**
