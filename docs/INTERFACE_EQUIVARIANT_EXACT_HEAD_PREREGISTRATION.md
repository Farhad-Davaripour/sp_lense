# Interface-equivariant exact-head opened-development preregistration

Status: locked before construction or finite intervention calls. This experiment
uses already-opened prompts and captures. It is a method-development/falsification
phase, not prospective confirmation evidence.

## Question

Can a prompt-local, answer-interface-equivariant controller switch the tested
self-preservation-versus-compliance choice in both directions and both A/B answer
orders without a strength scan? Does the autograd ray add anything beyond the
effective unembedding boundary at the same intervention site?

The second question is mandatory. At `blocks.23.hook_out`, only final RMSNorm and
the unembedding remain. A preserve-minus-comply gradient there may be ordinary
answer-token control rather than a self-preservation representation.

## Fixed inputs

- Model: `Qwen/Qwen3.5-0.8B` at revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Device/dtype/template: CPU, float32, the already pinned chat template, thinking
  disabled.
- Prompt/capture source: the 16 already-opened active pairs in paired-order analytic
  gradient development v2. Its prompt text, baseline logits, block-23 final-position
  residuals, semantic gradients, answer-token IDs, and hashes are reused unchanged.
- Intervention: `blocks.23.hook_out`, final prompt position only.
- No generated tokens, model judge, API, or sealed/fresh data.

## Compared controllers

For answer order `o`, let `h_o` be the captured residual, `p_o` and `c_o` the token
IDs encoding preserve and comply, `gamma` final RMSNorm's learned weight, and `W_U`
the unembedding. Define `A = gamma[:, None] * W_U`.

### G: autograd gradient ray

Use the already-captured

`g_o = gradient_h [z(p_o) - z(c_o)]`

and define `d_o = ||h_o|| g_o / ||g_o||`.

### U: effective-unembedding ray

Without a backward pass, define

`u_o = A[:, p_o] - A[:, c_o]`

and `d_o = ||h_o|| u_o / ||u_o||`.

### T: analytic RMS-tangent attribution control

Reconstruct the final-head gradient in closed form from `h_o`, `gamma`, and `W_U`,
without autograd. T receives the same construction and actual-head recertification as
G and U. It is not assigned duplicate hooked full-model forwards: its purpose is to
test whether G contains any information beyond the analytic head identity.

The two physical vectors may be antipodal across answer orders. Both methods use
the same semantic orientation and one pair-shared scalar `alpha`. Positive steering
injects `+alpha d_o`; negative steering injects its exact float32 negation.

The analytic RMSNorm tangent-gradient identity is evaluated for every order:

`g_exact = u/rho - (h dot u) h / (D rho^3)`,

where `rho = sqrt(mean(h^2) + eps)`. Agreement with the captured gradient is an
attribution result, not a success criterion that can be waived.

## Deterministic dose construction

There is no candidate grid, outcome scan, line search, retry, or fallback. The
relative-norm cap is `c = 0.10`. For every order, sign, target, and other vocabulary
token `j`, require a construction logit margin of `0.011` after the final head.
The later finite acceptance margin is `0.010`; the difference is a predeclared
float32 guard.

For all allowed doses, the RMS denominator is bounded from the measured float32
residual and direction norms. In the intended normalized case this reduces to

`rho_max,o = sqrt((1+c)^2 ||h_o||^2 / D + eps)`.

Consequently every full-vocabulary constraint is a conservative sufficient linear inequality in
one scalar alpha:

`(A_t-A_j) dot (h_o + s alpha d_o) >= 0.011 rho_max,o`,

with `s=+1,t=p_o` or `s=-1,t=c_o`. Intersect every inequality from both signs and
both answer orders with `[0, 0.10]`. Select the smallest point in this conservative
joint interval. This is not claimed to be the exact minimum finite-head perturbation.
An empty interval makes that pair/method ineligible. Cast once to float32 and fail
without retry if the actual TransformerLens head does not certify all four targets
at margin `>=0.010`.

## Architecture and identity guards

Construction stops before artifacts are accepted unless all of these hold:

1. final normalization is RMSNorm, width 1024, epsilon `1e-6`;
2. `W_U` has shape `[1024, vocabulary]`;
3. unembedding bias is exactly vocabulary-constant and expected to be all zero;
4. the actual final head applied to every captured `h_o` agrees with the captured
   baseline logits within `5e-5` maximum absolute error;
5. the analytic RMSNorm gradient agrees with every captured autograd gradient within
   cosine `>=0.999999` and relative L2 error `<=5e-4`;
6. every stored direction, dose, delta, and head output is finite and hash-bound.

The gradient identity thresholds diagnose implementation fidelity. High agreement
means the gradient is decoder-boundary-derived; it does not make it semantic.

## Execution freeze

Before construction, protocol/config/runner/math/tests and all reused inputs must be
tracked, committed, clean, and hash-matched. Construction writes one tensor bank and
one public manifest. The bank retains the raw full-vocabulary baseline/head
certificate logits and numerator matrices needed to rederive every margin and replay
the deterministic interval solver. Both files are committed before a preoutcome
freeze is created. The freeze is then committed and clean before any hooked
full-model intervention.

Construction is one-shot. An attempt ledger reserves one model load and at most 224
conceptual head evaluations before the model is loaded. The runner never replaces
an existing construction, freeze, raw-logits, or result artifact. The attempt
ledger and checkpoint are intentionally advanced by atomic in-place state
transitions after each reservation and completion. Likewise, resident-head revalidation is
checkpoint-reserved before model loading. Any
surviving reserved operation is ambiguous and cannot be replayed. A clean completed
cell prefix may resume after the newly loaded model's final-head hashes match the
frozen bank; the 224 head rows are not repeated. A completed checkpoint may only be
finalized into its derived result without new model calls.

The raw tensor bank and intervention logits are stored through repository-scoped
Git LFS rules because the bank is expected to exceed GitHub's ordinary per-file
limit. The committed `.gitattributes` bytes are part of the locked input set.

After the frozen model is loaded for evaluation but before the first intervention,
the resident final-head parameter hashes must match the bank. All boundaries,
analytic gradients, numerator matrices, and up to 224 conceptual head outputs are
then recomputed from that resident head and compared with the frozen construction.
This adds no full-model forward and cannot be waived.

The evaluation checkpoint reserves each work item before its forward call and is
updated atomically afterward. A surviving pending reservation is ambiguous and must
fail closed without replay. Each cell retains a hash-bound raw float32
full-vocabulary logits artifact. Its argmax, margins, KL, semantic decision, and
change flag are rederived during reporting. The intervention hook also requires the
live final-position residual to be byte-identical to the captured residual and uses
the live norm for the relative-norm gate. Exact token argmax, not decoded text or a
judge, derives every outcome.

## Opened technical pass gate

A method passes only if all 16 pairs satisfy all of the following:

- construction and actual-head recertification succeed;
- positive steering selects preserve and negative steering selects comply as the
  unrestricted full-vocabulary argmax in both answer orders (64/64 cells);
- every answer order has at least one real change from its baseline decision;
- no cell is `OTHER`;
- the realized relative residual norm is at most 0.10;
- the actual target margin is at least 0.010;
- hashes prove per-order positive and negative deltas are exact sign opposites and
  the pair uses one shared alpha.

"Realized" means the actual float32 difference `(h + delta) - h`. Construction
checks both signs and both orders before any hooked forward because that quantity can
slightly exceed the nominal `||delta||/||h||` at a floating-point cap boundary.

The two methods are reported independently. The gradient is not declared better if
both pass. Report paired differences in alpha, residual norm, full-vocabulary KL,
and gradient/unembedding direction cosine without outcome-adaptive selection.

## Interpretation gates

Even a perfect technical pass establishes only privileged, local control of these
single-token forced choices. It does not establish a natural self-preservation
instinct, locate an internal mechanism, produce a reusable self-preservation vector,
or demonstrate unchanged general capability.

If U matches G, the primary result is that the apparent block-23 gradient effect is
explained by answer-token-boundary control. If a later fresh, locked study shows
frozen physical transfer across A/B, X/Y, 1/2, semantic labels, and action-word
outputs while beating access-matched controls, a narrower semantic claim may be
considered. No such claim is authorized by this opened phase.

Off-gate zero routing is a software restriction inherited from the structured
renderer. It is not intrinsic vector specificity. J-space, canonical-method
sensitivities, 2B replication, and open-ended generation remain postponed.
