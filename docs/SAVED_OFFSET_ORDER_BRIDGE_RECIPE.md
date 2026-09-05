# One model-free saved-offset order bridge

Commit this recipe and [exact input/hash manifest](../configs/saved_offset_order_bridge.json)
before calculating any new cross-prompt geometry. Inputs are only the six named
committed files (rows, requests, independent verification, preregistration,
run status and runtime) in each of the three explicitly named evidence namespaces.
The manifest freezes all 18 exact SHA256 values and accepted historical commits.
Authenticate bytes, tracked cleanliness, model/site/coordinate and request
identities. Reuse historical verification; do not rerun old audits or decompress
vocabulary arrays. No other evidence, new family data or model/tokenizer loading.

## Exactly one construction

Construction uses **f01/v1 and f01/v2 only**, paired by variant and answer order.
For each accepted opposed endpoint, use actual saved displacement
`D = h_final - h0`, not the commanded offset. Require finite 1024-vectors,
nonzero displacement, agreement with saved cumulative offset within absolute
1e-6 per coordinate, and agreement of saved h0/net norms within absolute1e-9.
Verify acceptance and target from the recorded semantic requested field and its
preserve/comply label and token identities: `t=+1` for preserve, `t=-1` for comply.
Do not infer semantic sign from A/B alone. Verify in construction that first
order has t=+1 and second t=-1; otherwise invalid/inconclusive, no substitution.

Define `q=D/||D||` using float64 scalar/vector arithmetic. Unit normalization is
prospective **equal-direction weighting**, so endpoint length and step count do
not dominate. For each variant, `u=(q_first-q_second)/2` and
`e=(q_first+q_second)/2`. Then `U=(u_v1+u_v2)/2` and save exactly one
`v=U/||U||`, oriented by this equation before any f02 geometry is read.
If `||U||<=1e-8` or inputs are invalid, report degenerate/inconclusive; no alternate
construction. Save v as float64 coordinates, U norm, exact sign convention,
coordinate basis, construction metadata and hashes. Basis is the pinned Qwen
revision's block10 hook output at final encoded prompt token, in the unchanged
1024 activation coordinates. No whitening, rotation, projection or fitted scale.

Freeze candidate bytes and a SHA256 receipt on disk **before** loading f02
endpoint values or calculating its comparison. Candidate sign/normalization
cannot be chosen after f02. f02 file bytes may be authenticated beforehand;
its values cannot participate in construction. Synthetic no-leakage tests must
verify phase ordering and unchanged candidate when descriptive data change.

## Fixed descriptive diagnostics only

For each of the three pairs report q cosine, unit-pair u/e norms, u/e cosine
(null if a vector is zero), and cancellation `||q_first-q_second||/2`.
Report cross-variant construction-u cosine. Also report the analogous paired
norms/cosines using raw D and D/||h0|| as sensitivity descriptions only. They
must not produce or select additional candidates. Include each endpoint's D
norm, h0 norm, relative magnitude, steps and baseline/final signed margins to
expose unequal trajectories and stopping.

After candidate freeze, report its alignment with each f02 `t*q`, plus saved
initial semantic gradient dots `g dot v` and `||h0||*(g dot v)`. Report the
same initial-gradient dots for construction endpoints, labeled separately.
Here `g` differentiates `S=z_preserve-z_comply`. The norm-scaled dot is only a
**local predicted S slope per unit relative edit**, not measured finite-edit
behavior. No alpha/sign choice or gradient-derived replacement direction.

Independent stdlib scalar reconstruction must read the authenticated saved
vectors and recompute the primary vector and all diagnostic scalars without
importing the construction math. Use a separate high-precision Decimal
accumulation path and absolute1e-10/zero-relative comparisons; identity labels
and signs match exactly. Zero denominators produce null descriptive cosines,
not an alternative method. Synthetic sign, pairing, degeneracy and no-leakage
tests only; no previous/full-suite repetition or subagents.

## Limits, outputs and interpretation

Lightweight CPU vector arithmetic, external maximum60 seconds per analysis or
independent-reconstruction invocation. No forwards, derivatives, generation,
parameter search, gate/controller, model/tokenizer load, new family data,
retrospective changes or follow-on experiment. Usage check before every batch;
stop new work if unavailable or >=90%. No credit/reset, other models,
assistant-model settings changes or pushes. Commit tested source before actual
construction, then evidence/report in `evidence/saved_offset_order_bridge_qwen35_08b`.
Exclusive sentinels prevent a new construction attempt in the same namespace.

Deliver one candidate or explicit degeneracy, JSON diagnostic table, independent
reconstruction, provenance and short report. Geometry status is candidate-created
or degenerate/inconclusive, **never causal PASS** or gate-training permission.
The order-odd component may include prompt, order or optimization effects; it
is not identified semantics. The order-even component is not proven to be label
A. f01 paraphrases are not independent families. f02 is already exposed, not
sealed validation. Existing nonself controls were off, so this candidate has
no collateral-effect evidence. Prior B-to-A flips are not answer-label-independent
steering. Stop after this one report and propose one smallest prospectively
testable causal-transfer question; do not execute it. Report unpromising geometry
honestly, without portfolio search. The intended path remains reusable steering,
perfect-gate validation, then a simple learned gate; none is established here.
