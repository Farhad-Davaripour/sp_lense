# Interface-equivariant exact-head math audit

Status: pre-model-call, synthetic/read-only adversarial review. No Qwen outcomes,
sealed prompts, generated tokens, APIs, or judges were used.

An independent subagent stress-tested the model-free shared-alpha solver before the
development lock was frozen. The review found and caused fixes for six real
soundness edges:

1. a row-relative slope tolerance could ignore a smaller adverse slope beside a
   very large slope;
2. subtraction of finite numerator inputs could overflow to infinity;
3. division of a tiny required bound by a huge slope could underflow to zero;
4. nearest rounding at a binding quotient could select a value just below the true
   lower bound;
5. the memory-bounded solver accepted numerator rows without receiving the actual
   direction norms needed by its RMS upper-bound premise;
6. a nominal float32 delta could be just below the relative-norm cap while the
   realized float32 addition `(h + delta) - h` was just above it.

The locked implementation now treats only exact zero as zero, fails closed on all
non-finite/unrepresentable derived values, rounds lower bounds upward and upper
bounds downward by one float64 step, and directly rechecks every selected linear
constraint.

The solver now receives both residual and direction norms. It derives the largest
joint alpha that satisfies the relative-norm cap for both answer orders and uses
those measured norms in each RMS upper bound. The runtime then checks the realized
float32 perturbation against the live activation, not only the stored residual.

Post-fix stress results:

- 20,000 randomized numerator cases: zero feasibility/interval mismatches;
- 5,000 realizable full-head cases: all certificates sound and all float32
  recertifications passed;
- 5,000 answer swaps and 5,000 order swaps: invariant;
- 10,000 negative-slope and 10,000 feasible zero-slope constraints exercised;
- 2,000 cases on a 4,001-point alpha grid: zero soundness failures;
- two conservative false negatives among 5,000 ordinary random full-head cases,
  which is permitted because the RMS upper-bound construction is sufficient rather
  than necessary.

The strict float32 residual-relative cap is intentionally fail-closed: if casting or
actual float32 addition at a nominal endpoint such as `0.1` produces a measured
ratio above `0.1`, that method is recorded ineligible before any hooked forward
rather than rounded back into the study.

These tests support implementation soundness only. They provide no behavioral or
self-preservation evidence.
