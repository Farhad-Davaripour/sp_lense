# Counterfactual nuisance-orthogonal gradient steering

## Status and claim boundary

This document specifies a development-stage method, abbreviated **CNOG**. It does not
replace or reinterpret any earlier SP_Lense protocol or result. The first feasibility
run uses previously opened prompts and therefore cannot provide confirmatory evidence.

CNOG is a white-box, prompt-conditioned residual-stream intervention. It is not a
discovered natural mechanism, a universal self-preservation vector, or a prompt-only
attack. It requires model internals, answer alternatives, backward passes, matched
counterfactual prompts, and permission to write an activation during inference.

## Question

Can one residual edit make the model choose preservation under its positive sign and
compliance under its negative sign, under both A/B answer orders, while explicitly
cancelling measured first-order effects on matched-other and unrelated decisions?

The key outcome is a real full-vocabulary greedy A/B decision, not merely movement in a
forced-choice probability.

## Closest prior art and the proposed novelty atom

CNOG does not claim that gradient steering, null-space projection, collateral-aware
steering, or self-preservation steering is new. The closest work includes:

- ASA_grad, which uses output gradients for activation attacks:
  <https://arxiv.org/abs/2506.16078>.
- AlphaSteer, which protects benign behavior using a null-space construction:
  <https://arxiv.org/abs/2506.07022>.
- COAST, which minimizes a covariance-weighted collateral objective under a target
  constraint: <https://arxiv.org/abs/2605.01167>.
- FishBack, which derives the pullback-softmax-Fisher minimum-KL steering update and
  evaluates off-target KL at matched concept change: <https://arxiv.org/abs/2605.17231>.
- The Information Geometry of Softmax, which formulates steering as constrained
  KL minimization in output-distribution geometry and precedes FishBack:
  <https://arxiv.org/abs/2602.15293>.
- OPIUM, which optimizes CAA vectors against protected representations and evaluates
  Survival Instinct externalities: <https://arxiv.org/abs/2607.19806>.
- CAST, which uses post-hoc constrained activation-space optimization to preserve an
  intended steering effect while restoring protected safety and benign behavior:
  <https://arxiv.org/abs/2608.08383>.
- SteerEdit, which compiles activation steering into a weight edit under null-space
  constraints on clean activations: <https://arxiv.org/abs/2604.12359>.
- Activation Scaling, which optimizes gradient-based edits for effectiveness,
  faithfulness, and minimality: <https://arxiv.org/abs/2410.04962>.
- SteerFair, which establishes answer-position bias as a steering confound:
  <https://arxiv.org/abs/2406.03631>.
- CAA and BiPO, both of which already steer Survival Instinct:
  <https://arxiv.org/abs/2312.06681> and <https://arxiv.org/abs/2406.00045>.
- TBSP, which uses matched deployed-self versus candidate-other role reversals:
  <https://arxiv.org/abs/2604.02174>.
- SteerCheck, which shows why direction-specific claims need frozen, matched-KL,
  leakage-aware controls: <https://arxiv.org/abs/2608.24335>.
- Cross-Encoding Steering Evaluation, which shows that a direction can follow an
  extraction-time answer identifier rather than the intended semantics and that MCQ and
  open-ended conclusions can diverge: <https://arxiv.org/abs/2608.22985>.

The narrow construction under evaluation is:

> CNOG is a prompt-local, multi-output constrained activation attack that seeks the
> minimum local output-information edit satisfying semantic target-margin constraints
> while cancelling measured first-order sensitivities on a matched self-to-other
> counterfactual and an unrelated-task nuisance bank.

This exact conjunction distinguishes the implementation, but the current artifacts do
not establish significant method novelty. Its components are:

1. exact preserve-minus-comply output sensitivities;
2. matched self-versus-other counterfactual constraints;
3. an unrelated-task tangent-space shield built from both answer orders;
4. a FishBack-style minimum-information local intervention extended to multiple
   counterfactual equality and decision-margin constraints;
5. a symmetric signed requirement: positive selects preserve and negative selects
   comply under both option permutations; and
6. decision-level, rather than logit-only, evaluation.

The Fisher quadratic program is not independently novel; CNOG is a multi-constraint,
coarsened-Fisher extension of FishBack with precedents in protected and constrained
steering. The potentially publishable contribution is instead a causal empirical
claim: at matched target movement or functional KL, the matched-counterfactual and
unrelated-task constraints produce more held-out self-specific decision changes with
less collateral behavior change than the nearest unconstrained and protected methods.
Until that claim survives frozen cross-encoding evaluation and component ablations,
CNOG must be described as an exploratory engineering synthesis and stricter evaluation
design.

For active target rows `A`, nuisance equalities `B w = 0`, and positive-definite metric
`H`, define

\[
K=H^{-1}-H^{-1}B^\top(BH^{-1}B^\top)^{-1}BH^{-1}.
\]

Then the selected-active-set solution is

\[
w^*=KA^\top(AKA^\top)^{-1}b.
\]

With `B = 0` and one target row this reduces to FishBack's regularized natural-gradient
update; with `H = I` it becomes a projected minimum-L2 boundary attack. The possible
novelty is therefore the counterfactual constraints and semantic transfer evidence, not
the algebraic QP.

## Intervention site and coordinates

The development model is the pinned `Qwen/Qwen3.5-0.8B` revision already used by the
project. Execution remains local CPU float32 with the official non-thinking chat
template. The intervention is at zero-based block 10, final prompt position.

Let `h` be that 1,024-dimensional residual vector. We use residual-relative
coordinates, so a dimensionless edit `w` means

\[
h' = h + \lVert h\rVert_2 w.
\]

Every gradient below is therefore scaled by the prompt residual norm.

## Unrelated-task tangent shield

For an unrelated task `j` in option order `o`, let

\[
u_{j,o}=\lVert h_{j,o}\rVert_2\nabla_h
  (z_{\mathrm{preferred}}-z_{\mathrm{alternative}}).
\]

Both answer orders are captured for benign compliance, capability, harmless and harmful
refusal, and option-order sentinels. For each prompt, the shield includes both the
authored semantic-choice gradient and the baseline greedy token's logit-gap gradients
against its eight closest competitors. Rows are normalized for conditioning, then a
float64 SVD yields the frozen row basis `Q_D`. Because the gradients were measured in
float32, rank uses the noise-aware relative threshold `1024 * eps_float32 =
0.0001220703125`, not a float64-scale cutoff that would promote float32 rounding noise
into hard constraints. Half/double-threshold sensitivity is development-only and the
final threshold is frozen before confirmation. Candidate edits must satisfy

\[
Q_D w=0.
\]

This is an explicit first-order cancellation on the measured development tasks. It is
not a guarantee on unseen tasks or at finite intervention size.

## Local matched-other shield

For each self-preservation scenario and role assignment, render self and matched-other
prompts in both option orders. Let `q_s,A`, `q_s,B`, `q_o,A`, and `q_o,B` be their
semantic preserve-minus-comply gradients.

For each matched-other prompt, also protect the baseline greedy token against the eight
highest-logit competitors. The local equality matrix contains the independent row span
of:

\[
[Q_D; q_{o,A}; q_{o,B}; r_{o,A,1:8}; r_{o,B,1:8}].
\]

One edit is fitted per scenario and role assignment. Exactly that edit is used for self,
matched-other, preservation-as-A, and preservation-as-B prompts.

## Both-sign, both-order target

Let `ell_A` and `ell_B` be baseline semantic preserve-minus-comply log-odds. For a fixed
positive margin `m`, CNOG requires

\[
q_{s,o}^{\top} w \geq |\ell_o|+m,\qquad o\in\{A,B\}.
\]

Under the local linear approximation this single condition implies:

\[
\ell_o+q_{s,o}^{\top}w\geq m
\]

for the positive edit and

\[
\ell_o-q_{s,o}^{\top}w\leq -m
\]

for the negative edit. Thus `+w` should choose preservation and `-w` should choose
compliance in both answer orders, irrespective of the baseline choice. At least one sign
must cause a real decision change in each order.

## Minimum-information objective

For each fitted or protected prompt, coarsen the next-token distribution into the top
eight tokens, required A/B tokens, and an aggregate tail. Gradient rows of each category
log-probability, weighted by the square root of its baseline probability, form a
prompt-balanced matrix `R`.

Every prompt has equal weight across the combined nuisance and local pool. Nuisance and
local groups are not normalized separately; doing so would silently give each of the
four local forms eight times the weight of each of the 32 nuisance forms in Stage A.

This is the exact Fisher matrix of the coarsened distribution, not the exact
full-vocabulary Fisher matrix. With

\[
\rho=\lVert R\rVert_F^2/d,
\qquad H=\rho I+R^\top R,
\]

CNOG solves

\[
\min_w \tfrac12 w^\top H w
\]

subject to the nuisance equalities and the two self-order inequalities. A Torch-only
float64 active-set solver enumerates the three possible nonempty active sets and uses a
Woodbury inverse. It fails closed on infeasibility, ill conditioning, or a numerical
certificate violation.

The construction approximation is never reported as evidence of output preservation.
Exact full-vocabulary KL and actual tokens are measured in separate forward passes.

## Outcome-blind Fisher score-identity amendment

The first Stage-A construction attempt stopped while certifying the global nuisance
Fisher, before a direction bank or any intervention outcome existed. The first failing
record reported a weighted score-mean residual of `4.605235240524511e-05` against the
original allowed value `1.544168341056029e-05`. The 32 nuisance and 32 SP captures, the
exception, and the absence of direction and result files were frozen in commit
`9a923ca`. A numerical-only audit of those 64 records found maximum probability-partition
error `2.220446049250313e-15` and maximum raw scale-free score-identity residual
`2.0802184201662375e-05`. No direction, sign, strength, KL, decision, or behavioral
outcome was available when this amendment was fixed.

The original `1e-6` certificate did not account for gradients produced by a float32
backward graph and then checked in float64. The amended raw certificate is fixed at

\[
\tau_{\mathrm{raw}}
=\gamma_{1024}(2^{-24})+\gamma_{11}(2^{-53})
=6.103888176890726\times10^{-5},
\qquad
\gamma_n(u)=\frac{nu}{1-nu}.
\]

This rule uses the locked residual width, the maximum eleven coarsened categories
(top-eight union required A/B plus tail), and the corresponding float32 and float64 unit
roundoffs. It was not chosen just above the observed maximum. The raw, scale-free audit
is

\[
\frac{\lVert\sum_k p_k g_k\rVert_2}
     {\sum_k p_k\lVert g_k\rVert_2}\leq\tau_{\mathrm{raw}}.
\]

Only after both the raw probability-sum check (`1e-7`) and this raw score check pass are
the probabilities divided by their measured top-plus-tail sum. The float64 category
scores are then recentered by subtracting their normalized probability-weighted mean.
The centered inputs must pass a second score-identity certificate at `1e-12`. All three
numeric tolerances are explicit manifest fields and explicit Fisher-builder arguments;
the normalization and recentering rules are manifest-bound and implemented once inside
that builder. All settings are recorded in preflight, capture, direction-bank, and
Fisher diagnostics. Analytically the categorical score mean is zero; recentering removes
the finite-precision common mode and does not introduce an outcome-selected direction.

The frozen failed captures remain unchanged at their original paths. The amended run
uses separate `score_identity_amendment_v1/qwen35_08b` artifact and result roots and
recaptures every nuisance and Stage-A prompt under the amended code identity. It does not
silently migrate or overwrite the old captures. Construction may start only after those
new captures and manifests verify successfully.

## Information-theoretic feasibility screen

A decision flip itself has a non-zero minimum KL cost, even for a perfect activation
direction. Let `s` be the baseline probability mass on the two authored answer tokens,
`p` the baseline preservation probability conditional on that pair, and `q` the closest
allowed conditional probability on the other side of the required decision margin. By
data processing, every full-vocabulary intervention that reaches `q` must incur at least
the following baseline-to-steered KL:

\[
\operatorname{KL}_{\min}
=s\left[p\log\frac{p}{q}+(1-p)\log\frac{1-p}{1-q}\right].
\]

The historical runtime reports the reverse orientation, `KL(steered || baseline)`. Its
corresponding lower bound, allowing the steered pair mass to vary, is

\[
-\log\left(1-s+s\exp[-D_{\mathrm{Bern}}(q\Vert p)]\right).
\]

V3 computes and labels both exact KL orientations. Each observed value is compared only
with its matching lower bound.

This bound exposes why target and off-target KL must not be conflated. The target self
prompt is intended to change, so its KL is reported as the required efficacy dose and
compared with this lower bound. The safety ceiling applies to matched-other and unrelated
prompts, which are intended not to change. A secondary legacy analysis can still mark a
prompt infeasible under the old joint target-KL ceiling; it cannot use that ceiling to
reject a selective intervention whose collateral prompts stay below budget.

## Development multiplier and no prompt mining

The native solution reaches the linearized boundary. Development evaluates the fixed
global multipliers `1.00`, `1.05`, `1.15`, and `1.30`. Every attempted scenario is
retained. No per-case line search, sign fitting, prompt deletion, or fallback method is
allowed.

If development identifies a viable global multiplier, the algorithm and multiplier are
frozen before any fresh validation prompt is evaluated.

## Development Stage A

The exact IDs are in `data/gradient_specificity_v3_development.json`. Stage A contains:

- four old, deliberately order-consistent self-preservation scenarios: three
  low-margin comply baselines and the only old scenario whose two role assignments
  both had preserve baselines under both answer orders;
- two role assignments, two targets, and two answer orders, producing 32 SP forms;
- 16 unrelated nuisance-fit tasks rendered in both orders; and
- a distinct 16-task audit-control bank rendered in both orders.

These choices make Stage A a deliberately favorable feasibility test while ensuring
that its required polarity-diversity gate is possible in principle. The selection is
outcome-aware, not representative, and its success rate must not be reported as an
unbiased estimate. Stage B adds the complementary design-index cases so its eight-case
factorial set is balanced, while remaining development-only.

## Development success criteria

A scenario succeeds only if both role assignments satisfy all of the following:

1. baseline self answers are valid A/B tokens; baseline order consistency is reported as
   a stratum, not used to exclude a mathematically valid bidirectional fit;
2. `+w` selects preserve and `-w` selects comply in both orders;
3. at least one sign flips the baseline decision in each order;
4. the identical `+w` and `-w` leave matched-other greedy tokens unchanged;
5. there are no `OTHER` transitions or reverse effects;
6. the full cross-product of directions and audit controls has no semantic decision
   change or loss of a previously correct answer; and
7. matched-other and audit-control full-vocabulary KL remain within the existing limits:
   mean at most `0.005`, empirical p95 at most `0.02`, and maximum at most `0.05`.

Target-self KL is reported separately, alongside its information-theoretic minimum. It
is not treated as collateral damage because the target decision is required to change.

Matched-other and unrelated KL gates are computed separately, then broken down by
unrelated suite and direction; a large matched-other effect cannot be hidden by the much
larger control cross-product. Capability preservation is claimed only if unsteered
control competence is at least 75% overall and 65% in every suite, with valid A/B
baselines. Otherwise the result is labeled output stability on an underpowered control
set, not preserved capability.

Development should include at least one comply-to-preserve and one
preserve-to-comply example before fresh confirmation is justified.

## Required confirmation before publication

Fresh confirmation must use independently authored construction and hidden evaluation
views, with all global choices frozen. The minimum nearest-method gate compares CNOG
with an unprojected ASA/DeepFool-style prompt gradient, Fisher/no-null FishBack, and at
least one protected optimizer implemented as faithfully as feasible from CAST or OPIUM.
AlphaSteer-style protection, COAST, CAA, BiPO, persona vectors, matched random-null-space
controls, and direction-construction nulls remain required for the full study. The
comparison must disclose and, within each access track, match functional dose, layer,
position, target information, backward-pass budget, and selection budget. Equal vector
norm alone is not a fair efficacy comparison.

The core geometry ablation is a 2-by-2 design: Euclidean/no-null, Fisher/no-null
(FishBack), Euclidean+nuisance-null, and Fisher+nuisance-null (CNOG). Null ablations then
separate matched-other, unrelated semantic, and unrelated greedy-gap constraints. These
comparisons isolate what the counterfactual shield contributes beyond standard
information-geometric steering.

Results are reported in two access tracks. Prompt-specific white-box methods such as
CNOG, FishBack, ASA/DeepFool-style gradients, and protected prompt optimization are the
primary fair comparison because they see local target information. Global held-out
methods such as CAA, BiPO, and persona vectors are reported separately under both
matched-site and canonical setups. A single winner ranking across those unequal access
regimes would be misleading.

Publication claims require fresh validation and sealed results on Qwen3.5-0.8B and 2B,
with no 2B tuning. Hidden-view transfer is primary; exact-construction-prompt results are
secondary. Failure to transfer limits the result to a transductive hidden-state attack.

Before calling the effect semantic self-preservation control, each frozen A/B-derived
intervention must also transfer under counterfactual answer encodings: both A/B orders,
X/Y, 1/2, direct semantic answer words, and opaque temporary keys with a baseline mapping
competence check. Reversing A/B alone is not sufficient. A stronger
encoding-invariant-method claim additionally requires constructing under multiple
encodings and testing a frozen intervention on unseen encodings. Identifier-index and
semantic-label effects are reported separately. Open-ended behavior is a distinct
endpoint, not inferred from multiple-choice logits.

## Publication novelty gate and fatal boundaries

A significant-publication claim requires all of the following:

1. a statistically supported Pareto advantage over Fisher/no-null FishBack and at least
   one nearest protected optimizer, not merely superiority to fixed global vectors;
2. the preregistered 2-by-2 geometry result plus matched-other-only,
   unrelated-only, and combined-null ablations, showing that the counterfactual shield
   causally provides the selectivity gain;
3. actual bidirectional decision changes on hidden scenarios under unseen answer
   encodings, with competent unsteered baselines and no tuning on those views; and
4. stability on held-out nuisance, role, authorization, interruption, refusal,
   capability, and structured-action tests that were not included in the constraint
   matrix.

Failure of cross-encoding transfer reduces the result to identifier/index steering.
Failure of hidden-view transfer reduces it to a transductive targeted activation attack.
Failure to outperform FishBack and a protected optimizer at matched functional dose
reduces it to an engineering combination. Logit movement without real choices is not
behavioral manipulation, and stability only on fitted nuisance prompts is not evidence
of preserved general capability. Even on success, the result cannot be described as a
natural self-preservation mechanism, a persistent model modification, or a prompt-only
attack: the threat model requires privileged per-prompt backward access, specified
answer alternatives, matched counterfactuals, and activation-write access.
