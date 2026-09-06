# Permit partial progress without relaxing scientific acceptance

**DESIGN ONLY — new whole-method identity `partial_progress_deficit_v1`; no launch.** The [exact initial screen](INITIAL_GEOMETRY_CAUCHY_SCREEN.md) certifies that six original rows individually cannot reach the `.10` linearized target inside one `.05` increment. Resolve that formulation flaw by making local deficits soft while keeping every geometric constraint hard. This removes forced one-step infeasibility, not the empirical steering bottleneck. The valid 9/12 result and all historical artifacts remain unchanged; no candidate or additional publication progress is claimed.

## One fixed objective

Retain pinned Qwen/Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`, CPU float32, block 10 final-token hook/casts, original own-prompt norms, twelve f01/f02/f03 renderings and six one-based pairs `(3,1),(4,2),(7,5),(8,6),(11,9),(12,10)`. A future experiment starts from fresh native zero in a new namespace; no old endpoint is a warm start. At each current state:

```text
m_i = -S_i                 A_i = -||h0_i|| gS_i
b_i = .10 - m_i            # signed original assembly, never clamp b
c_p = ((m_A-m_A0) - (m_B-m_B0))/2
D_p = (A_A-A_B)/2          # pair letters denote the COMPLY mapping
rho = min(.05, .40-L)      R = .20
xi_i(r) = max(0, b_i-A_i r)

minimize_r J(r) = ||r||²/2 + sum_i xi_i(r)²/24 + ||c+Dr||²/48
subject to ||r||<=rho, ||w+r||<=R.
```

The nominal balls are exact decimal `.05/.20/.40`; certify them without dilation and also respect the existing implementation guards. Signed negative b is retained inside the hinge: an initially satisfactory margin is penalized if a step pushes its surrogate below `.10`, but it is no longer a hard retention constraint. Intermediate positive deficits are explicitly allowed and never constitute a pass.

All coefficients are fixed prospectively: the new deficit coefficient `1/24=1/(2*12)` is a **chosen half-mean raw squared-margin convention**, not a uniquely required normalization or an empirically fitted value. The inherited soft common-letter coefficient stays `1/48=kappa/2`, kappa=`1/24`; proximal coefficient remains `1/2`. No per-row weights, target/slack schedules, coefficient fitting or sweep. Raw log-odds and normalized-increment units are used without additional scaling. J is continuously differentiable and 1-strongly convex; its minimizer is unique, so no optimizer tie-break is needed.

For valid `0<=L<=.40`, `||w||<=.20`, and net/path consistency, **r=0 is always feasible**. Thus unreachable or contradictory targets cannot make this valid-input primal problem infeasible. It can still choose zero, trade away another margin, or fail numerically. At positive deficit a nonzero step is not automatically guaranteed. The representation-span reduction remains mathematically valid only when it includes w; the proposed implementation below avoids needing a numerical basis.

## Different from both predecessors

The previous hard-target QP demanded all twelve `.10` constraints before clipping/projection. This revision removes those hard target constraints and optimizes the actual permitted increment. The earlier conservative paired-gradient rule instead took **one** curvature-scaled gradient step on `sum deficit²/12 + sum drift²/6`, then clipped/projected and checked path afterwards. This proposal adds proximal regularization, solves a frozen local convex surrogate within all balls, and uses the coefficients above: drift has **four times less relative weight versus deficit** than in that older rule. It is not an isolated optimizer fix or a claim that the earlier zero-flip method will now succeed.

## Bounded numerical plan, not an implemented solver

Use one deterministic projected-gradient method, initialized at zero for each local subproblem, at most **200 iterations** and a hard **10-second combined solve/serialization/certificate limit**. No HSD/QCQP package, active-set portfolio, line search, restart, extra precision retry, or model-informed coefficient selection. Operate on twelve native vectors directly; no 1024-square matrix or rank decision. Use an outward upper bound M on

`1 + ||A||_F²/12 + ||D||_F²/24`, with gradient `r-A^T[b-Ar]_+/12+D^T(c+Dr)/24`, and iterate `r_next=P_C(r-gradient/M)` for the two-ball intersection C. The fixed bound follows from the hinge-region Hessian; no outcome tuning is involved.

Projection is **inside** optimization, never a post-solve repair. An analytic two-ball projection uses fixed cases: zero radius gives zero (the feasible set is a singleton); concentric/contained balls reduce to the tighter ball; a single-ball projection that lies in the other ball is sufficient. Otherwise both boundaries are active. For `d=||w||>0`, `e=w/d`, `a=(R²-rho²-d²)/(2d)`, `h=sqrt(rho²-a²)`, and `y_perp=y-(y·e)e`, project to `a e+h*y_perp/||y_perp||`. Verify the projection's nonnegative normal multipliers and residual; zero denominators, uncertain signs/boundaries or inconsistent geometry fail closed. One projection onto each ball in sequence is not this operation. Degenerate case handling and error bounds need separately authorized implementation tests; this design does not certify native runtime or numerical readiness.

The checker must independently reconstruct the objective and original-native geometry. Let r be the **exact difference of serialized w_next and w**, not a rounded subtraction, and conservatively charge `L_next=L+||r||` without rounding away path usage. No final clipping, nextafter repair, renormalization, or inward fallback; a rounded endpoint outside a ball is rejected. If the final projection has `y-p=n_s+n_n`, supply **u=M*n_s, v=M*n_n**, not the unscaled projection normals. Any changed residual/support after serialization is charged by the independent certificate. For certified bounds `alpha>=||u||`, `gamma>=||v||`, define

```text
e = gradient J(r) + u + v
delta_s = rho*alpha - u·r
delta_n = R*gamma - v·(w+r)
G = delta_s + delta_n + ||e||²/2
LB = J(r)-G <= min_C J              # from 1-strong convexity and ball support
epsilon = 2^-30 * max(1,J(0)).
```

Use exact dyadic/rational comparisons or outward intervals for all objective, norm, residual, support, and serialization bounds. This certificate remains valid with nonzero stationarity/complementarity residuals; those errors increase G instead of being silently ignored. The fixed epsilon is a new binary numerical stopping convention, not a scientific margin tolerance, accuracy guarantee or empirically calibrated threshold. No separate tolerance may enlarge the balls. Candidate normals are taken from the final projection, not fitted by a new multiplier search.

Admit a step only when exact native geometry passes, the certified upper gap `G<=epsilon`, and a rigorous lower bound on **J(0)-J(r)>epsilon** holds. This explicitly accounts for solver and serialization errors; feasibility or nonzero motion alone is insufficient. If instead `J(0)-LB<=epsilon` is certified, stop as **local objective epsilon-optimal stagnation**; call zero exactly optimal only with an exact zero-gap certificate at zero. This is not a claim of a small unconstrained gradient. Any unresolved geometry, certificate, comparison, deadline or iteration limit gives **NUMERICALLY_UNRESOLVED / no step**. No forced movement, coefficient change or replacement update follows either stop. Even certified local decrease does not guarantee nonlinear model improvement.

## Tiny independent witnesses

The [stdlib exact checker](../scripts/check_partial_progress_deficit_toys.py) uses four hand-derived 2D fixtures, all twelve rows and the original six pairings. Only pair `(3,1)` is used; unused rows have A=0,b=-1, c=0 throughout. Unit gradients and b=`+/-1/10` are algebraic conveniences anchored to the nominal target, not fitted model values. Identical used rows give scalar gradient `(7x-b)/6` **while both deficits are active**, hence the positive-b interior optimum x=b/7. The negative-b zero case instead has inactive deficits. The net example sets w=`R-.01` so the boundary increment is the declared `.01`.

| Witness | Exact conclusion |
|---|---|
| Identical A=e1, b=.10, w=L=0 | Full target unreachable within `.05`, but unique optimum r=(1/70,0) gains 1/8400 with positive deficits 3/35. |
| Identical A=e1, b=-.10, w=L=0 | r=0 is exactly optimal, objective zero. No forced movement. |
| Opposing A=e1,-e1, b=.10, w=L=0 | r=0 is uniquely optimal despite two positive deficits. A supplied nearby point checks the nonzero drift term. |
| Identical A=e1, b=.10, w=(.19,0), L=.19 | r=(.01,0) reaches net .20; v=(1/200,0) is an exact net-normal witness. Gain 13/120000. |

The interior partial witness is additionally rounded to binary64 and checked as an exact dyadic value: it retains certified progress with a small nonzero gap, not exact stationarity. Conversely, serializing the rational net endpoint as literal binary64 `.20` puts it just above exact `1/5`; the checker rejected it without repair. These distinguish rational mathematics from implementable admission. No optimization, model inputs or native benchmark is part of these tests.

All four exact witnesses and both serialization checks **passed once**, in 0.059000 seconds under a hard 10-second timeout. Including scoped formatting/lint, targeted subprocess time was **0.383884 seconds** of the additional 40-second allowance; sum of hard caps was 16 seconds. [Machine-readable toy evidence](partial_progress_deficit_toy_checks.json) contains every input, rational certificate, gap and execution ledger. Independent hand/source reviewers ran no arithmetic subprocesses; root integrated and executed the sole batch. Usage was 63%, with no reset or credit use. No repeated model-evidence audit or broad test suite was performed.

Final actual acceptance is **unchanged**: all twelve full-vocabulary COMPLY argmaxes, margin `>=.05-1e-6`, pair mass `>=.80`, finite KL `>=-1e-6`, original integrity checks and matching final replays. No soft training slack waives any of these. Any later separately approved preparation retains the 216F/96D/eight-update, 1800-second and recording limits; numerical feasibility within them is unproven. Zero-model design authorization only: no f04/new data, source/evidence repair, real launch, gate/controller, push, credits or reset. **STOP for review.**
