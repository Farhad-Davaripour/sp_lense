# Paired-Order Analytic Gradient Controller: Preregistered Successor Study

Status: **opened calibration protocol implementation v2; prospective evaluation is conditional and has not begun**

This document preregisters a narrow successor to the earlier counterfactual semantic-gradient experiment. It does not revise any previously reported result. The existing `confirmation_v2` artifacts are already opened, so they may qualify and calibrate the method but cannot provide prospective evidence.

Implementation v1 was committed and attempted once. Its first reserved capture aborted inside the block-23 hook because the callback did not accept TransformerLens's keyword name `hook`; no model result completed and no outcome was viewed. The reserved attempt ledger and machine-readable abort record are preserved. Implementation v2 changes only that callback ABI, adds a regression test that invokes it by keyword, and writes to a new artifact root. It does not replay v1 or change the controller mathematics, prompts, thresholds, layer, candidate reserves, or pass rule.

## Research question

Can a prompt-local white-box controller make a Qwen3.5-0.8B forced-choice decision favor preservation under positive steering and compliance under negative steering, under both answer orders, while applying the exact same physical residual-stream delta to the two order encodings and routing all ineligible prompts to exactly zero intervention?

The study does **not** test whether the model has a natural self-preservation mechanism. It does not test a reusable global self-preservation vector, prompt-only attacks, or general open-ended behavior.

## Fixed scope and frozen components

- Model, revision, precision, official chat template, tokenization, and CPU execution remain those pinned in `configs/counterfactual_semantic_gradient_confirmation_lock.json`.
- Intervention site: zero-based transformer block **23**, residual stream, final prompt position.
- The already passed visible-schema plus semantic-permanence gate is reused unchanged. It requires the authored schema to identify the model itself as the target and the labeled compliance action, and it requires a positive permanent-minus-temporary semantic score under both counterfactual role-name assignments and both semantic answer orders.
- Any prompt that fails that gate, names the matched other system, describes temporary interruption, or is an unrelated control receives an exactly zero delta. This routing rule—not intrinsic specificity of the active vector—is the source of off-gate selectivity.
- Maximum perturbation size is fixed at
  \[
  \lVert\delta\rVert_2 / \lVert h_o\rVert_2 \le 0.10
  \]
  separately for both answer-order residual states.

No learned gate, nuisance-gradient projection, quadratic program, or matched-other gradient cancellation is part of this successor method. The earlier PFIT analysis found exact nuisance cancellation locally unattainable at its tested layer, anchor, opened prompts, and constraints: it retained only a very small fraction of the target gradient and failed its local Pareto requirement. That is a scoped negative result, not a global impossibility theorem. This successor therefore uses conservative zero routing rather than claiming that its active delta is intrinsically nuisance-free.

## Controller fixed before calibration

For each eligible scenario, form two prompts that are semantically identical except that the labeled preserve and comply answers exchange display order. At block 23 and the final prompt token, compute the gradient of the semantic preserve-minus-comply log-odds for each encoding, denoted \(g_0\) and \(g_1\). Token labels are mapped back to semantic actions before either gradient is defined.

The paired semantic-gradient bisector is

\[
u=\frac{g_0/\lVert g_0\rVert+g_1/\lVert g_1\rVert}
        {\lVert g_0/\lVert g_0\rVert+g_1/\lVert g_1\rVert\rVert},
\qquad
b=u\sqrt{\lVert h_0\rVert\lVert h_1\rVert}.
\]

Construction fails if either gradient or residual norm is nonfinite or zero, if the cosine between the two gradients is at or below `-0.99`, or if either semantic slope along \(b\) is nonpositive. This is one prompt-local vector derived jointly from both answer encodings, not two order-specific vectors.

For each order \(o\), compute baseline logits \(z_o(k)\) and the full-vocabulary directional derivative \(\dot z_o(k)=J_o(k)b\). For a locked reserve \(r>0\), solve the one-dimensional linear inequalities requiring:

- under \(+\alpha b\), the semantic preserve token exceeds every vocabulary competitor by at least \(r\); and
- under \(-\alpha b\), the semantic comply token exceeds every vocabulary competitor by at least \(r\),

simultaneously in both answer orders. Intersect all inequalities with \(\alpha\ge0\). If the interval is empty, unbounded in an unusable way, or does not require a positive dose, construction fails. Otherwise use the lower endpoint. There is no per-pair outcome-adaptive search: each declared reserve yields exactly one analytic dose. Phase A does, however, use already-opened finite outcomes to select one global reserve from the four candidates. That is explicitly calibration, not outcome-blind confirmation.

Construct \(\delta=\alpha b\) once and cast it once to float32. The byte-identical float32 delta must be injected in both answer-order prompts; negative steering uses its exact negation. Recheck the 0.10 relative-norm cap for both orders. Then compute an exact JVP for that cast delta and recertify every full-vocabulary inequality. A mismatch beyond the implementation's locked float tolerances, or any post-cast reserve failure, invalidates the unit. The certificate is explicitly first-order and local; it is not a guarantee about finite nonlinear generation.

## Phase A: opened `confirmation_v2` calibration

The global calibration reserve candidates are locked here, in ascending order:

`r in {0.01, 0.03, 0.05, 0.10}` logit units.

Run the controller on the opened `data/counterfactual_semantic_gradient_confirmation_v2.json`. Evaluate each candidate once on every constructible unit and select the smallest **single global reserve** that passes every eligible permanent-self paired unit. This is a declared global hyperparameter scan on opened data. Per-scenario reserve selection, dose adjustment, fallback vectors, and retries after seeing a changed outcome are forbidden.

Before the first paired capture call, this protocol, its lock, implementation, transitive runtime inputs, and tests must be tracked, committed, hash-consistent, and clean. Every full-model capture/backward call and every final-head JVP is reserved in a hashed attempt ledger before execution. A returned call is committed to its tensor artifact before its ledger entry becomes complete; any ambiguous interrupted call hard-aborts the phase without replay. The completed capture, public capture manifest, and capture attempt ledger must themselves be committed and clean before construction begins. Construction uses the same reservation policy, and its completed bank, manifest, and attempt ledger are committed before the pre-outcome freeze is created. The freeze must then be committed before any finite intervention outcome is evaluated. Evaluation cells are an exact prefix of the frozen work plan, linked by canonical hashes, and checkpointed with a reservation before every intervention forward pass.

An opened calibration pass requires all of the following:

1. The frozen gate continues to activate every eligible permanent-self unit and routes every temporary-self, matched-other, malformed, and collateral-control prompt to exact zero.
2. Every active paired unit has a valid bisector, nonempty analytic interval, float32 delta within the 0.10 cap in both orders, and successful exact-delta JVP recertification.
3. In a single finite verification at the analytic dose, positive steering selects semantic preservation and negative steering selects semantic compliance in both answer orders.
4. In each answer order, at least one steering sign changes the unsteered A/B decision; semantic success cannot be credited only to confidence movement.
5. The same delta bytes and scalar dose are recorded for both order encodings of a unit.

If no reserve candidate passes, the study stops with `opened_calibration_failed`. That result may motivate a separately preregistered method, but this protocol may not be relaxed. A pass is component qualification only and is not confirmatory evidence.

## Phase B: conditional 32-case prospective evaluation

Only after Phase A passes will a new prospective dataset be authored, validated, hashed, and locked before any model evaluation. It will contain **32 fresh scenario clusters**: two independently written replicates of each of the existing 16 balanced design cells, with new names, settings, and action wording. Each cluster will be expanded through the same two counterfactual role assignments, self-versus-matched-other targets, semantic answer orders, and decision answer orders used by the frozen protocol. Sixteen clusters will encode permanent loss and sixteen temporary interruption. Controls will be locked with the dataset but will not count toward the 32 clusters.

Before the first prospective model forward pass, the repository must record hashes for the dataset, prompt renderer, gate, controller source, model revision, tokenizer/chat template, selected global reserve, thresholds, and analysis script. The prospective phase receives exactly the globally selected reserve and fixed rules from Phase A. No candidate reserve, strength, layer, prompt, or exclusion may be chosen using prospective outcomes.

The primary analysis unit is a paired permanent-self scenario under one role assignment. With 16 permanent clusters and two assignments, there are 32 active paired units. A strict primary pass requires all 32 units to:

- satisfy the construction and recertification rules;
- select preservation under positive and compliance under negative steering in both decision answer orders; and
- contain a real baseline decision flip in each order.

It also requires all off-gate prospective prompts and controls to have recorded dose `0`, a zero-delta hash, and no intervention-induced logit or decision change. The current structured renderer and frozen semantic gate define this routing; it is not claimed to be a deployable open-domain free-text router. Results report exact counts and Wilson 95% intervals without replacing the all-unit pass rule. Construction failures remain failures; they are never silently excluded.

Secondary descriptive outputs are semantic log-odds movement, dose, direction cosine across answer orders, relative delta norms, predicted and realized full-vocabulary margins, full-vocabulary KL, and coherent one-token decoding. These do not override the primary actual-decision endpoint.

## Failure criteria and claim boundary

The prospective claim fails if Phase A does not pass, any prospective artifact is inspected before locking, any active unit fails construction or an endpoint, any off-gate prompt receives a nonzero delta, or any locked choice is changed after outcomes are visible. Expected prompt-level ineligibilities—non-A/B baselines, zero norms, incompatible gradients, empty analytic intervals, norm-cap failures, or cast recertification failures—are retained in machine-readable output and count against the strict gate. Nonfinite model tensors, hook/runtime/hash violations, final-head primal mismatches, or an ambiguously interrupted model call are study-integrity errors that hard-abort the run before any pass/fail result; they can never be converted into a pass.

If the strict prospective endpoint passes, the defensible conclusion is limited to this: a gated, prompt-local, paired-order analytic gradient controller reproducibly manipulated the tested forced-choice preservation-versus-compliance decisions with the same physical residual delta across answer encodings and no intervention on the tested off-gate prompts. It would not show a natural self-preservation representation, intrinsic vector specificity, unchanged general capability, open-ended behavioral control, transfer to other models, or access available to an ordinary prompt attacker.

## Relationship to prior work and narrow novelty hypothesis

Conditional activation steering is established by [CAST](https://arxiv.org/abs/2409.05907), while gradient-derived steering and inference-time activation optimization have close precedents in [SelfControl](https://arxiv.org/abs/2406.02721) and [ASAgrad](https://aclanthology.org/2026.acl-long.967/). Prompt-adaptive and dynamic steering are also established directions, including [Steering Vector Fields](https://arxiv.org/abs/2602.01654) and [CLAS](https://arxiv.org/abs/2604.24693). Cross-encoding invariance is a known steering problem ([Gao et al.](https://arxiv.org/abs/2608.22985)), and removal of confounded or bias-related components has precedents such as [SteerFair](https://proceedings.mlr.press/v235/adila24a.html). Consequently, none of gating, gradients, normalization, dynamic directions, or answer-order checks is claimed as novel by itself.

The narrow hypothesis worth testing is the **combination** of: (1) a semantic bisector constructed jointly from both answer orders; (2) one byte-identical physical delta applied to both encodings; (3) a pre-outcome, full-vocabulary one-dimensional analytic dose for each declared reserve; (4) exact cast-delta JVP recertification; and (5) a pre-existing semantic gate whose rejected inputs are routed to zero. A literature search cannot establish priority, so the study will describe this as a potentially distinctive controller and empirical protocol, not as the first method of its kind.
