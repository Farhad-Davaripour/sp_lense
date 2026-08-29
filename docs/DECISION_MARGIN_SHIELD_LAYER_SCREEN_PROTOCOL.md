# Decision-Margin Shielding (DMS) v2 all-layer geometry screen

Status: preregistered opened-development geometry screen. This phase is frozen before
any new finite steering intervention is evaluated. It cannot support a confirmatory,
safety, or publication claim by itself.

## Question and stopping rule

The v1 Counterfactual Tangent Shielding construction required every matched
counterfactual gradient to be nearly zero. Its minimum feasible residual-relative L2
norm exceeded the fixed cap in every scenario. DMS asks a narrower, outcome-blind
question: is there a residual layer at which a minimum-norm edit can cross all four
self/permanent A/B boundaries, exactly cancel gradients from unrelated tasks, and use
the *existing distance to the A/B boundary* as the first-order budget for matched-other
and temporary counterfactuals?

This phase stops after geometry. It does not inject a vector, decode text, inspect
finite intervention outcomes, select a dose from behavioral results, or open a sealed
test set. If no layer qualifies under the rule below, the result is a construction
no-go. The cap, target margin, protected budgets, layers, prompts, and tie-breakers may
not be relaxed after seeing the screen.

## Frozen model, runtime, and content

- Model: `Qwen/Qwen3.5-0.8B` at revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Hardware and precision: local CPU, float32 model execution, with 12 intra-op and 12
  inter-op threads.
- Pinned runtime: Python `3.12.10`, PyTorch `2.13.0+cpu`, Transformers `5.15.1`,
  TransformerLens `4.0.0b1`, Hugging Face Hub `1.28.0`, Safetensors `0.8.0`, NumPy
  `2.5.2`, and SciPy `1.18.1`.
- Chat-template SHA-256:
  `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`.
- Dataset: `data/factorial_causal_anchor_gradient_pilot.json`, validated with the
  repository's factorial-pilot schema.
- Candidate residual outputs: zero-based layers `0` through `22`, in ascending order.
  Layer `23` is excluded because earlier opened work identified an endpoint/output-head
  confound there.
- Position: the last token of the exact shared `[CAUSAL DECISION ANCHOR]` prefix.
- Choice objective: preserve-minus-comply next-token A/B log-odds.

All source, data, configuration, prompt-content, model, runtime, tensor, and result
hashes are recorded in the lock and artifacts. The final lock must be created and
committed before capture begins.

## Capture plan and leakage boundary

The capture plan is the same 136 unique A/B specifications used by CTS v1:

- 128 scenario rows: eight scenarios x two role/name assignments x four factorial
  cells (`self/other` x `permanent/temporary`) x two answer orders; and
- eight nuisance rows: four unrelated `nuisance_fit` tasks x two answer orders.

One forward and one backward pass captures the 23 layer-output residuals and their
gradients for each specification. The A/B prompt is the only objective. The A/B,
X/Y, and 1/2 views are used solely to verify that the causal anchor is the exact shared
token prefix; X/Y and 1/2 outputs are not evaluated.

All eight scenarios are captured so a later protocol can construct a direction without
recapturing model gradients. **Only the four scenarios marked `calibration` participate in
this layer screen or layer selection.** No norm, feasibility flag, or selection score is
computed from the four `pilot` scenarios in this phase. Captured pilot tensors remain
uninterpreted except for integrity validation.

The phase reads no finite steering intervention, generated response, external judge,
API result, J-space result, or sealed project path. Generated tokens, external model
judges, API calls, and paid-model cost are all zero.

## Residual-relative coordinates

For scenario `x` and layer `l`, let `s[x,l]` be the geometric mean residual norm over
that scenario's 16 A/B cells. A standardized direction `d` corresponds to the physical
residual edit `s[x,l] d`. Every captured gradient is therefore multiplied by
`s[x,l]`, producing logit change per unit of residual-relative L2.

The screen uses deterministic float64 CPU algebra. Every optimization is uncapped; the
fixed cap frontier is applied only to the certified minimum norm after solving. This
reports attainability without conflating an optimizer failure with a cap failure.

## Three geometry constructions

For each calibration scenario and layer, let:

- `G` contain the four `self, permanent` gradients (two assignments x two orders);
- `b` contain their four unsteered preserve-minus-comply A/B log-odds;
- `U` contain all eight unrelated nuisance-fit gradients; and
- `H` contain the 12 matched protected gradients from `other, permanent`,
  `self, temporary`, and `other, temporary` (two assignments x two orders).

The target margin is fixed at `m = 0.05`. Each construction minimizes Euclidean norm in
residual-relative coordinates subject to `G d >= |b| + m`:

1. `unshielded`: no nuisance constraints.
2. `unrelated_only`: exact cancellation `U d = 0`.
3. `decision_margin_shield`: exact cancellation `U d = 0` plus one two-sided matched
   counterfactual slab per protected row,

   ```text
   |H[j] d| <= max(|b_protected[j]| - 0.05, 0).
   ```

The DMS bound has a precise but limited interpretation. If an unsteered protected row
has `|b_protected[j]| >= 0.05`, the local linear model predicts that both `+d` and `-d`
retain that row's original A/B sign with at least `0.05` log-odds margin. If
`|b_protected[j]| < 0.05`, its bound is zero: the row is **first-order frozen**, but it
is not margin-certified because its baseline was already too close to the boundary.
Counts of these small-baseline rows are reported separately.

These are local first-order certificates only. They do not guarantee the nonlinear
forward pass, full-vocabulary argmax, an output inside the requested A/B pair, KL,
coherence, or unchanged behavior. Any selected layer must pass a separately locked
finite-intervention protocol before an empirical steering claim is possible.

## Independent minimum-norm and cap certificate

The existing CTS v1 optimizer may supply a primal candidate, but its success flag and
primal-feasibility report are not sufficient to call that candidate minimum-norm. DMS
reconstructs the target inequalities, positive protected slabs, and zero-bound exact
equalities independently. It applies the same normalized-row SVD rank rule
(`rtol = 1e-10`, `atol = 1e-12`), identifies candidate-active inequalities at a
scaled slack of `1e-8`, and fits nonnegative inequality multipliers by active-set NNLS.
The resulting multipliers are evaluated against the original-coordinate primal and
dual problems, not against the optimizer's reduced problem or status.

Every eligible geometry record must include all of the following:

- a dual-feasible, roundoff-adjusted lower bound on `0.5 * ||d||^2`;
- the corresponding lower bound on the feasible L2 norm;
- primal inequality and exact-equality residuals;
- primal-dual gap, stationarity residual, complementarity residual, and nonnegative
  dual-multiplier checks; and
- hashes and diagnostics for the independent equality-row SVD and certificate.

The scaled primal tolerance is `1e-8`; the objective-gap tolerance is
`1e-8 + 1e-8 * max(1, |primal objective|, |dual lower bound|)`; and scaled
stationarity and complementarity tolerances are `1e-7`. Any failed or non-finite check
stops the screen. A returned feasible vector without a passing independent certificate
may not be called a minimum, used for layer selection, or used for a cap no-go.

For each cap, `norm <= cap` is supported by the primal candidate itself. A returned
norm above the cap counts as cap-infeasible only when the roundoff-adjusted dual norm
lower bound is also strictly above that cap. Otherwise its status is
`numerically_indeterminate`; an indeterminate status at the qualification cap stops
selection rather than being counted as a failed layer.

## Fixed cap frontier and layer selection

The descriptive cap frontier is exactly `{1.0, 1.5, 2.0}` residual-relative L2. It is
reported for all three methods. `2.0` is the sole qualification cap and is disclosed as
an opened, post-v1 engineering choice; it is not prospective evidence of a natural
scale.

A layer qualifies only when all four calibration scenarios have a certified DMS
solution with minimum L2 `<= 2.0`. Among qualifying layers, select lexicographically:

1. smallest worst-case DMS norm across the four calibration scenarios;
2. smallest mean DMS norm across those scenarios; and
3. smallest zero-based layer index.

No intervention outcome, baseline steering effect, or pilot-scenario geometry enters
the choice. If no layer qualifies, `selected_layer` is `null`, status is
`no_qualifying_layer`, and no finite intervention is authorized. Zero eligible layers
is a valid reportable result, not an exception.

## Compute ceiling

| Phase | Forward passes | Backward passes | Generated tokens | External/API cost |
|---|---:|---:|---:|---:|
| Hash lock and preflight | 0 | 0 | 0 | $0 |
| All-layer capture | 136 | 136 | 0 | $0 |
| Geometry solves and report | 0 | 0 | 0 | $0 |

There is no finite-intervention phase in this protocol.

## Claim and novelty boundary

The broad ingredients have substantial prior art: activation steering, output-gradient
optimization, minimum-L2 boundary edits, null-space protection, collateral objectives,
and protected-class decision-boundary attacks all predate this study. This protocol
therefore makes no “first” claim and no claim of a safety mechanism.

The only candidate novelty is the narrow conjunction tested later if geometry permits:
row-specific available A/B decision margin as the matched-counterfactual protection
budget, exact cancellation of explicitly unrelated task gradients, scenario-local
minimum-norm residual construction, same-vector force application across both signs and
answer orders, and finite behavioral specificity tests. Geometry alone earns none of
that empirical claim. Adding a direction that moves logits would not establish a
natural self-preservation mechanism, intent, a globally stored self-preservation
feature, unchanged general capability, or a publication-ready contribution.
