# Standalone numerical prototype: fixed before execution

**MODEL-FREE ONLY.** Parent design commit `55c6a3492336938934c21496a0d55fcdc6e98efb` remains immutable. This prototype is not pipeline, recording, scientific, or real-launch readiness. The purpose is to decide whether the stated numerical method is practical enough to justify further work. Do not repair an unfavorable result or rerun the native smoke.

## Fixed method and admission

Keep exactly twelve rows and pairs `(3,1),(4,2),(7,5),(8,6),(11,9),(12,10)`, signed b, and `D=(A_A-A_B)/2`. The objective is `J(r)=||r||²/2+sum[b-Ar]_+²/24+||c+Dr||²/48`. The exact-decimal geometry is `||r||<=min(1/20,2/5-L)`, `||w+r||<=1/5`. No coefficient or tolerance fitting, row normalization, alternate objective, penalty schedule, radius dilation, or final point repair is permitted.

The single floating projected-gradient solver starts each local problem at nominal zero. One initial zero certificate is followed by at most **200 PG updates**, with a **10-second in-process deadline** starting before input authentication. It uses one outward-rounded scalar M bounding `1+||A||F²/12+||D||F²/24`, analytic two-ball projection, and M-scaled projection normals. The nominal floating projection is the next internal iterate; the independently checked increment is always the exact difference of serialized endpoint and current w. A non-admitted intermediate iterate never becomes a model update.

The projection screen fixes nonnegative multiplier signs with no sign tolerance and a residual ceiling `128*d*2^-52*max(1,||y||,||p||,hypot(||n_s||,||n_n||))`. This is a declared floating-arithmetic screen, **not an exact projection theorem or a geometry allowance**. Ambiguous/degenerate projection arithmetic fails closed. No line search, restart, solver portfolio, or precision retry. Upward `nextafter` may be used solely to enclose the scalar M, never to change an endpoint or radius.

The independently written checker imports no solver. Canonical JSON SHA256 authenticates the frozen problem; raw little-endian binary64 SHA256 authenticates current w. It checks one to nine history snapshots, starting at zero and ending at the identical serialized w; every historical step/net ball and the conservative cumulative path upper bound is checked. `path_upper` is a bounded integer/fraction string, not a float or exponent expression. It may conservatively overcharge history but may not undercharge. These are synthetic fixture histories, not authenticated real-model logs.

A,b,c,w and witnesses are interpreted as exact binary64 rationals; D is independently derived exactly from A. Solver floating copies cannot change sealed checker tuples. The objective, gradient, signed deficits, serialized displacement, norm squares, support dot products, and gain are rational computations. Square roots needed for support/path upper bounds use integer-isqrt ceiling on a fixed **2^-80 grid**; the enclosure can cause conservative rejection, not acceptance outside a ball. No solver factors or solver objective values are trusted.

Admission uses the design's support/strong-convexity gap G and exact `epsilon=2^-30*max(1,J(0))`: geometry must pass, `G<=epsilon`, and `J(0)-J(actual)>epsilon`. The zero branch needs a certified objective-suboptimality upper bound at zero no larger than epsilon; exact zero optimality requires a zero-gap proof at zero or a singleton radius-zero domain. A small gradient, unchanged endpoint, positive remaining deficit, or an iteration limit is not substituted for that proof. A serialized geometry/path violation is immediately terminal **NUMERICALLY_UNRESOLVED**. Failed convergence by update 200 or deadline is also unresolved, never infeasibility or a stationarity claim. No-step results expose no applied endpoint.

The internal deadline includes preparation, projection, endpoint arithmetic, certification, and optional recording callbacks. The native smoke's separate **hard 10-second subprocess timeout also covers imports, fixture/source authentication, progress/final JSON encoding, and process completion**. If it kills the worker, retained checkpoints are partial diagnostics, not a completed certificate. The observer records existing checker results only; it cannot return a replacement iterate. No observer failure permits admission.

## Frozen fixtures and expected dispositions

Exact fixtures and origins are in [the new configuration](../configs/partial_progress_deficit_synthetic_fixtures.json). All tiny problems are 2D and use pair `(3,1)` only, with unused A=0,b=-1 and c=0. The original rational examples become their declared binary64 input values in this numerical prototype; they are not claimed to retain rational exact optima after rounding.

| Case | Prespecified construction and check |
|---|---|
| Interior | Both used A=e1,b=.1,w=L=0. Require generated PG admission. |
| Negative RHS | Both used A=e1,b=-.1,w=L=0. Require exact zero optimum and no movement. |
| Contradictory pair | A=e1,-e1,b=.1,w=L=0. Require exact zero optimum despite positive deficits. |
| Literal net boundary | w=.19e1,L=1/5, identical A=e1,b=.1. Run PG and separately require rejection of literal binary64 endpoint `.20e1`. |
| Both balls active | w=.199e2,L=1/5, identical A=e1,b=1. Require that actual PG generation reaches the both-active projection case. Admission is not presumed. |
| Near internal tangency | w=.15000000000000002e2,L=1/5, identical A=(2^-26,1),b=1. R-rho=.15 is the containment-tangency origin; the transverse size is a square-root precision scale. Rejection remains an honest outcome. |
| Exhausted path | w=0, deliberately conservative charged L=2/5. Verify singleton no-step handling; this overcharge is a test state, not a claim of real historical travel. |
| Gain erased by rounding | w=.125e1,L=1/5,A=1e-16e1,b=.1. A supplied 1e-18 nominal displacement rounds away; no progress may be admitted. Generated PG may stop via its actual epsilon certificate. |
| Large M, near opposition | A=(1e6,0),(-1e6,1),b=.1,w=L=0. The transverse unit component prevents exact initial cancellation while the large shared component makes the fixed M conservative. Do not reinterpret limit exhaustion as stationarity. |

The native smoke is **one** explicit dense 12x1024 fixture, not padded 2D. For zero-based i,j, `A_ij=(-1 if i%2 else 1)*(1+(((j+1)*(i+3))%17)/16)+(2*i-11)/2048`; all b=.1, c alternates `.01,-.01`, w=0, history=[0], L=0. Modular integers and power-of-two denominators define a deterministic stress fixture, with small row offsets preventing trivial exact opposing cancellation; values are not calibrated to real gradients. Explicit arrays were saved before any solver execution. Canonical problem SHA256 is `92b4ffc3ada2d8f90be5a33eb4f04c104e7d7d945b909a212f4d067aae312793`. The worker verifies both the formula and hash before its sole solve. No inference about all real-data geometries follows from this synthetic test.

The focused tests additionally cover current-w/input/history/path tampering, altered normals/endpoints, nonfinite inputs/deadlines, bounded path parsing, copied-versus-sealed input mutation, and checker independence. Tests regenerate each tiny problem exactly once, then may independently check its retained witness. Native metadata/hash checks in pytest do not execute or numerically evaluate the native problem. Each generated result is logged so outcome reporting needs no repeat solve.

## Shared execution budget and stop

Before numerical testing, save a separate preregistration JSON with hashes of all new solver/checker/test/fixture/worker/protocol files. Run only scoped formatting/lint, this test file, and the one native worker. Planned external caps: formatting/import-order/lint up to 5 seconds each; pytest at most **50 seconds**; native worker **10 seconds once**. Keep the measured aggregate, including failures and any permitted narrowly disclosed software-bug correction checks, at **120 seconds or less** across all agents. No coefficient, fixture, tolerance, iteration, or native-retry change after observing results. Native execution is withheld if source authentication or the focused tests fail; no native replacement.

Parallel work is disjoint: solver source, fixture/test source, and read-only checker review are delegated; parent owns independent checker, protocol, runtime ledger and integration. Agents run no solvers/tests; parent executes the sole batch. Usage may reach the authorized account limit, but no reset/credit/push is allowed. Preserve all earlier sources/evidence and unrelated work. Report admitted, stagnated and rejected cases separately, give a practical integration recommendation, and **STOP**. No ML loads/F/D, actual gradients, f04/new data, historical re-audit, pipeline/experiment lock, gate, LoRA or model launch.
