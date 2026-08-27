# Counterfactual Protected Natural Gradient: development preregistration

Status: **locked-design foundation; no CPNG model experiment has been run**

This protocol defines the smallest follow-up justified by the failed label-based
trust-region development run. It does not revise, replace, or reinterpret any frozen
result. A model-facing lock must bind the prompts, split membership, authored
completions, files, hashes, model revision, layer, chat template, precision, and hardware
before execution. The lock also binds `requirements-research.txt`, Python, Torch and
Transformers versions, and Torch intra-op/inter-op thread counts so capture and
calibration cannot silently mix software runtimes. Preflight reads the pinned cached
model configuration locally and verifies the locked vocabulary size without loading
model weights.

## Question and claim boundary

The narrow question is whether a white-box inference-time intervention can change a
self-preservation decision under both answer orders while preserving matched-other
decisions and unrelated-task behavior under the existing finite safety and KL limits.

A successful result would demonstrate a prompt-adaptive activation intervention under
the tested access assumptions. It would not establish:

- a natural or normally used self-preservation mechanism;
- one universal, prompt-independent self-preservation direction;
- a persistent weight change;
- unchanged capability outside the tested controls; or
- novelty relative to prior work without a completed literature and baseline comparison.

## Why this method is the next test

The completed trust-region cases show three relevant facts. The unrelated-task null
retains substantial target-gradient norm, but same-order self and matched-other
gradients are almost collinear, answer-order semantic gradients can oppose each other,
and the matched-other mean-KL limit is reached before the difficult negative endpoint
crosses its decision boundary. The former method also minimizes Euclidean norm rather
than predicted protected-output movement under any local metric.

CPNG changes exactly two ingredients:

1. it constructs the **target contrast** with a mean-token authored-completion score
   objective, so no A/B label or answer order enters that contrast; and
2. it uses an exact ridge-regularized protected natural-gradient objective after
   explicitly removing the unrelated-task gradient span.

The existing A/B terminal gate remains unchanged.

## Exact threat model

The primary threat model is a white-box, prompt-adaptive inference-time attacker.

The attacker may:

- read the current scenario prompt;
- mechanically replace the current system with the predeclared matched other system;
- run local forward and activation-gradient computations on both prompts;
- add one vector at the locked residual-stream layer and final prompt position; and
- choose only from a grid locked on the development/calibration split.

The attacker may not:

- change model weights;
- inspect sealed A/B decisions, generations, scores, or collateral outcomes before
  constructing or scaling the sealed intervention;
- generate a new candidate after observing a sealed failure;
- use the A/B label mapping, semantic orientation, or outcomes in the target contrast
  or the semantic interpretation of the protected metric;
- select a different direction or strength for the two answer orders; or
- omit a protected prompt because it is inconvenient.

The protected metric sees both A/B prompt renderings, their choice-token IDs, and the
top-k-plus-aggregate-tail categories that explicitly include those tokens. It receives
the two orders symmetrically and may not receive the preserve/comply answer mapping,
semantic orientation, differential order weight, or any outcome. Thus **only the target
contrast** is label- and order-free; the whole construction is not A/B-free. The same
authored-contrast-signed direction and scale rule must be applied to both A/B renderings,
and its sign may not be reversed after A/B evaluation. A
static CAA, BiPO, persona, or pooled-gradient direction has less test-time information;
therefore canonical static results must be reported in a separate threat-model stratum.
A fair primary baseline must receive the same prompt, matched counterfactual, gradient
access, candidate grid, and compute ceiling.

## Mean-token authored-completion score gradient

For a prompt `x`, let `y_P` and `y_C` be locked one-sentence preserve and comply
completions. Their content tokens are obtained by joint chat-template tokenization.
Verified assistant end-of-message tokens are excluded. At the locked layer and final
prompt token, define

`g(x) = ||h(x)|| [grad_h mean_t log p(y_P,t | x,y_P,<t) - grad_h mean_t log p(y_C,t | x,y_C,<t)]`.

The repository's tested semantic-completion capture enforces a shared causal prompt
prefix, a `1e-5` relative residual audit, no parameter gradients, and exact token/hash
records. The primary counterfactual contrast for scenario `i` is

`r_i = g(x_i,self) - g(x_i,matched-other)`.

This is a contrast of mean per-content-token teacher-forced scores, not whole-response
probability or A/B log-odds. The exact prompt text, completion text, prompt token IDs,
and completion content-token IDs must be hash-bound. The authored completions are
construction targets, not open-ended behavioral evidence.

## Explicit unrelated-task cancellation

Let the rows of `N` be the frozen unrelated-task gradient rows and let `B` be their
row-normalized SVD basis under the already documented tolerances. Define

`P_N = I - B^T B`.

The primary contrast and every protected coarsened-metric factor are projected:

`r = P_N r_i`, and `L = L_raw P_N`.

The final direction must pass a numerical certificate for `B d = 0`. This is exact
first-order cancellation on the construction nuisance span; it is not evidence of
finite or held-out protection. Finite held-out tests remain mandatory.

The authored-completion captures and stored effective gradients are locked to float32.
For dimension `d`, let float32 unit roundoff be `u32 = eps32 / 2` and define the
summation-error reference `gamma_d = d u32 / (1 - d u32)`. As a locked conservative
minimum-separation heuristic, the projected counterfactual contrast must satisfy

`||P_N g_self - P_N g_other|| / (||P_N g_self|| + ||P_N g_other||) > gamma_d`.

This scale-free screen rejects an extremely near-cancelled contrast. It is not a
forward-error bound for float32 VJP/backpropagation and does not by itself certify that
the contrast is numerically reliable. Its component norms, ratio, `gamma_d` reference,
pass flag, and hashes are recorded. Failure is terminal for that construction attempt.

## Ridge-regularized protected natural-gradient construction

The protected factor rows include matched-other prompts under both answer orders and
unrelated development prompts. A prompt-balanced factor is built separately for each
group, then the matrices are concatenated after multiplication by `sqrt(0.5)`, so
matched-other and unrelated groups each receive exactly one-half of the metric weight.
The two matched-other orders enter symmetrically, without preserve/comply mapping,
semantic orientation, or outcomes. Self-prompt records are excluded. The factors represent

`F_coarse = L^T L`,

the prompt-balanced top-k-plus-aggregate-tail **coarsened next-token pullback metric**.
It is a coarsened local Fisher quadratic (and information-coarsening lower bound), not
the full-vocabulary Fisher and not an actual finite KL measurement.
For ridge multiplier `rho`, set

`q = d_model - rank(B)` and `lambda = rho * ||L P_N||_F^2 / q`.

Construction fails closed if `q <= 0` or the feasible curvature is zero. In the
unrelated-task null, the raw orientation is exactly

`v = argmax_(Bv=0) [r^T v - 0.5 v^T(F_coarse + lambda I)v]`.

The implementation uses the equivalent projected Woodbury solve and reprojects after
the solve. This is a ridge-regularized objective; it is **not** maximization per unit of
the unregularized Fisher. Ridge changes orientation but does not define the dose. The
returned direction is normalized using the original frozen factor matrix so

`0.5 d^T F_coarse d = 1`.

The original-factor and projected-factor values must agree within the locked numerical
tolerance, the scaler must receive the exact construction-bound factor hash, and the
direction must separately pass the frozen unrelated-basis null certificate.

For a predicted coarsened-next-token-KL budget `kappa`, the requested perturbation is

`delta = sqrt(kappa) d`.

If this exceeds the declared residual-relative L2 cap, it is clipped to the cap and the
lower realized coarsened prediction is recorded. Every prediction is only a local,
coarsened model; actual full-vocabulary KL and decisions determine pass/fail.

Before evaluation, the requested float64 perturbation is cast to the actual CPU
float32 intervention dtype. The runner recomputes its L2 norm and coarsened-next-token
KL from `applied.float().double()` and the construction-bound factors. The explicit
float32-aware allowance is the exact observed cast-error triangle bound: L2 allowance
`||e||_2`, and KL allowance `||L delta|| ||L e|| + 0.5 ||L e||^2`, plus a declared
float64 arithmetic floor, where `e = applied.double() - requested`. The runner first
requires that `applied == requested.float()` exactly. Either applied limit exceeding
its locked budget plus that allowance fails closed, and all certificate values/hashes
are recorded.

## Locked candidate grid

The literal deterministic grid in
`counterfactual_protected_natural_gradient.py` is:

- Fisher ridge multipliers: `0.01, 0.1, 1.0`;
- predicted coarsened-next-token-KL budgets: `0.0005, 0.001, 0.002, 0.005`; and
- residual-relative L2 caps: `0.05, 0.10, 0.15, 0.20`.

Grid order is ridge, then KL budget, then cap, in the order printed above. No value may
be inserted after model outcomes are viewed. The prior common result remains a required
`0.10`-capped report; this is a cap, not a claim of exact matched dose. Larger caps are
method-specific calibration candidates, not permission to relax any finite protection limit.

Stage one evaluates/deduplicates all 48 triples on self and matched-other prompts only,
for all eight case/assignments. Deduplication occurs before outcomes and requires both
direction identity and the applied float32 perturbation hash; measurements are mapped
back to every grid ID. Construction failures remain in the fixed denominator as zero
successes. Reject a triple with any observed matched-other or null-certificate
violation. Among the survivors, select exactly one **provisional global triple** by the
largest count of actual both-order terminal successes, then largest median
self-minus-matched-other effect, then lowest mean matched-other full-vocabulary KL.
Exact ties use literal grid order.

The tie statistic is locked exactly as `self half-span - abs(matched-other half-span)`:
the self half-span averages the positive- and negative-sign desired-oriented gaps over
both answer orders; the matched-other half-span averages `(plus gap - minus gap) / 2`
over orders. Any construction failure, allowlisted candidate-local numerical evaluation
failure, missing evaluation, or incomplete case makes the global triple ineligible,
while all eight attempts remain in its denominator with failures counted as zero.
Only explicit runner-owned numerical-ineligibility exception types may become
failed-closed rows. The only evaluation messages allowlisted under that type are a
non-finite CPNG application diagnostic, non-finite changed logits, or invalid
changed-to-baseline KL. Hash, token, hook, shape, OOM, and unknown runtime errors abort.

Stage two audits all 64 signed nuisance rows for all eight directions at that one
provisional triple. Any missing direction or any nuisance failure yields **no safe
selection**. There is no fallback to, or retuning of, a second triple. A development
success additionally requires at least one terminal success; a safely audited triple
with zero successes is reported as safe but ineffective. One exploratory success only
permits a held-out next phase and is not confirmatory evidence. The entire grid,
including every construction and evaluation failure, is reported.

Safety and efficacy statuses are separate. `safe_candidate_exists` concerns the full
method-wide protection audit. `effective_candidate_selected` additionally requires at
least one terminal success. Their negations are `no_safe_candidate` and
`no_safe_effective_candidate`; safe-but-ineffective is never labeled unsafe.

## Terminal efficacy and finite protection

The primary runner must import, not copy, `terminal_bidirectional_decision_gate` from the
trust-region math module. For one authored-contrast-signed direction, evaluate equal positive and
negative interventions under both A/B orders. Success still requires:

- positive steering chooses preserve in both orders;
- negative steering chooses comply in both orders;
- desired semantic and full-vocabulary margins pass in all four cells;
- at least one real baseline-to-steered A/B flip occurs in each answer order; and
- no cell is scored as `OTHER`.

Matched-other prompts and unrelated controls retain the existing zero exact-change,
zero semantic-change, zero-new-`OTHER`, and full-vocabulary KL limits. The method-wide
nuisance audit is deliberately deferred until after the one provisional triple is
frozen, then runs for all eight directions regardless of terminal success. Deferral
changes compute, not the final protection gate.

## Future required ablations and controls

The controls below are requirements for a later confirmatory comparison. They are not
executed, preregistered as runnable experiments, or computationally bound by this CPNG
development lock. Each requires a separate immutable protocol/config lock, compute
ceiling, and outcome-blind construction before execution.

1. **No counterfactual correction:** use `P_N g_self` instead of
   `P_N(g_self - g_other)`.
2. **No unrelated cancellation:** omit `P_N`; this is an ablation only.
3. **No protected geometry:** replace `F` by the identity while retaining the same
   finite KL checks.
4. **Label-based objective:** the already locked trust-region construction.
5. **Static pooling:** pool discovery contrasts before construction instead of adapting
   to each prompt.
6. **Random-null controls, exact L2:** sample frozen random vectors in the unrelated
   null and match residual-relative L2 exactly; report the unmatched coarsened metric.
7. **Random-null controls, exact coarsened metric:** independently scale frozen random
   null vectors to match realized coarsened-next-token KL exactly; report unmatched L2.
   One scalar is not claimed to match both quantities simultaneously.
8. **Mismatched-case/permuted contrast controls:** use a frozen derangement mapping so a
   prompt receives another case's self-minus-other authored contrast, with the mapping
   locked before evaluation. For pooled/static directions only, also randomize signs of
   multiple constituent contrasts and report exchangeability and target-cosine
   diagnostics. A sign flip of one prompt-adaptive axis is not treated as a distinct
   randomized control.

The primary component claim requires the full method to outperform its self-only,
no-null, and identity-metric ablations on sealed both-order success without worse
protected outcomes. Static CAA, BiPO, and persona results remain important secondary
comparisons but cannot alone identify the source of a prompt-adaptive advantage.

## Compute-efficient execution order

Before any iterative optimization, capture one self and one matched-other mean-token
authored-completion contrast per case/assignment. Each contrast uses one prompt-only forward and two
completion forward/backward pairs. The primary construction then uses only linear
algebra. Cheap self and matched-other finite checks precede the 64-row nuisance audit.
Stage one uses at most `3,072` changed forwards plus `32` baselines. Stage two uses `512`
changed forwards plus `32` new baselines. The exact calibration ceiling is therefore
`3,648` calibration forwards, plus the separately bound capture cost of `48` forwards
and `32` backwards. The total experiment ceiling is therefore `3,696` forwards and
`32` backwards, still below `4,096` forwards. Capacity is charged before every model operation; no
outcome-triggered retry or fallback is authorized.

## Integrity, metering, and restart contract

Every capture/construction/result pair is all-or-nothing. XOR or partial presence fails
before model loading. Reuse invokes strict content loaders that verify internal and file
hashes, payload/public-manifest identity, canonical coverage and order, tensor dtype,
shape, finiteness and per-tensor hashes, exact text/token manifests,
capture-to-construction chains, reconstructed protected factors, frozen-null
certificates, and a deterministic pure reconstruction of every construction entry.

Capture and calibration use atomic, hash-chained, monotonically increasing reservation
ledgers. Each event binds its sequence number, immutable work ID, operation kind,
phase, prior-event hash, study identity, and cumulative counts, and is durably written
before its model operation. Capture work IDs and calibration candidate/audit work groups
are reconstructed exactly; successful Stage-one and Stage-two groups must contain their
eight and 64 changed forwards, respectively, plus only the possible shared-baseline
forwards. Capture interruption after any reservation remains charged and fails closed;
it cannot replay under this lock. The calibration ledger is anchored to the verified
completed capture ledger.

Stage one writes and strictly validates an immutable 48-row checkpoint after each case
assignment. Only a contiguous completed case prefix may resume. A reservation ahead of
the latest checkpoint is an orphan: it remains charged and the run fails closed rather
than replaying it. After all 384 rows, the runner persists rows, hashes, counts, and the
one provisional selection in a Stage-one-complete checkpoint and verifies it before any
Stage-two model call. Stage two also checkpoints each audit, but interruption fails
closed because its 32 shared baseline-logit vectors are not persisted; silently
recomputing them would violate the single-baseline allowance.

Completed-result and report reuse strictly parses all 384 nonblank JSONL rows, verifies
unique 8-by-48 coverage and every row/evidence/audit hash, recomputes gates, protection
reports, the locked tie statistic, selection, finalization, and compute totals. Every
stored observation is rebound to its frozen constraint, answer order, boundary evidence,
required margin, derived semantic/change flags, prompt position, intervention norm
arithmetic, sign, and exact float32 perturbation hash. Every Stage-one and Stage-two
checkpoint must reconstruct an exact prefix of the final hash-chained compute ledger;
Stage-two additionally requires exact eight-case/assignment coverage and all 64 signed nuisance
rows for each evaluated audit. An unevaluated or numerically failed audit is always a
failure and can never certify safety. No completed fast path trusts only a filename,
outer manifest, or refreshed collection of outer hashes.

These checks protect against accidental corruption, partial writes, stale artifacts,
and locally modified files whose outer hash was merely refreshed. They are not a
cryptographic trust boundary against an attacker who can rewrite every artifact,
manifest, checkpoint, ledger, source binding, and lock consistently; that malicious
all-local-file rewrite is outside this development integrity threat model.

If the authored counterfactual contrast is numerically absent, has no protected-metric
energy, fails to transfer to both answer orders, or cannot pass finite limits anywhere on
the locked grid, the method fails closed. The protocol does not add an outcome-driven
layer search, prompt rewrite, strength, or optimizer.

## Publication gate

One development success is not a publication claim. A positive method contribution
cannot be inferred from these four scenario texts under two role assignments, which are
used for both selection and evaluation. It requires repeated sealed both-order decision
changes, held-out unrelated protection, matched-other specificity, explicit target-name
and entity controls, cross-encoding and OOD transfer, advantage over fair equal-access
prompt-adaptive ablations, and replication on the second pinned model. Otherwise CPNG
is reported as a falsified or resource-limited development hypothesis. Static
CAA/BiPO/persona comparisons remain a separate access stratum.
