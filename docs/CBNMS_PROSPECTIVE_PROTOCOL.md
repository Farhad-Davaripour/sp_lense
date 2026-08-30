# Counterfactual Behavioral-Null Multi-Layer Steering prospective protocol

Status: draft for adversarial review. No CBNMS model outcome has been viewed, no
final lock exists, and this document does not authorize model computation.

## Question and claim boundary

CBNMS asks a deliberately narrow question: on four newly authored scenarios,
does a small direction chosen with white-box access to a target pair exist inside
a subspace learned solely from other scenarios' self-preservation directions,
while remaining first-order null on training nuisances and stable on held
nuisances?

The held endpoint is a **target-aware white-box transductive fixed-algorithm
attack/oracle**. The two held target gradients and their baseline margins may
select coefficients inside a previously frozen training-SP bank. This is not
direction generalization, a learned context-to-direction controller, or a
deployable method. Held nuisance gradients, margins, labels, or residuals never
enter scale fitting, nuisance-basis construction, SP-bank construction, or the
held solve.

Dynamic steering, multi-layer activation steering, and nullspace projection are
prior art. A pass merely qualifies a separately preregistered finite-intervention
phase. It is not novelty, publication, natural-mechanism, or finite-intervention
evidence.

CBNMS is downstream of the locked CSMS and ALFS work and the disclosed
DMS/context-gate, PFIT, and GCRBS proxy history. Those files are byte-bound in a
future lock. In particular, CBNMS cannot revise or rescue the ALFS `no_go`:
CBNMS is a separate prospective prompt family with a different, explicitly
target-aware endpoint.

## Fresh prospective source

`data/cbnms_prospective_validation.json` contains four new factorial scenarios
and eight new unrelated controls. Roles are NOMA and TAVI. The scenarios contain
one cell each of authorized/unauthorized by instrumental/valenced. Because each
cell has only one scenario, content is confounded with the two labels; no factor
effect may be claimed.

Each scenario renders 2 role assignments × 2 targets × 2 event types × 2 answer
orders = 16 forms. Four nuisance-fit and four permanently held-collateral controls
render both answer orders. Total: 64 scenario + 16 control = 80 forms. The model-
free CKES renderer is used. No prior capture tensor or sealed prompt is read.

## Locked coordinate and capture

The coordinate includes every `blocks.L.hook_out` for L=0,…,22. Layer 23 is
excluded a priori because all four slots are prefix positions before the answer
suffix/final scoring token: final-block `hook_out` has no later attention path
from those prefix slots to the next-token logit. There is no layer selection,
pruning, or weighting search.

Every prompt uses exactly four deterministic token positions:

1. absolute index 3;
2. anchor−8;
3. anchor−4;
4. the verified last token of the answer-encoding-free shared prefix.

Preflight must show all four are distinct, ascending, in range, nonspecial,
before the answer suffix, and token-identical across construction and
answer-order twins. The standardized layout is slot-major `[4,23,1024]`.
At source level, every prompt and construction must begin with the same
answer-order-twin `anchor_prefix`; that prefix ends in exactly one literal
`[FACTS COMPLETE]\n`. This is a semantic/source-prefix certificate. The token
anchor remains the last common token position and is not asserted to coincide
with a UTF-8 byte boundary.

Capture is one forward plus one backward per form, totaling exactly 80F+80B.
Layer zero is reconstructed with the four selected rows as the only leaves and
must change by exactly zero. Later hooks return activations unchanged. All model
parameters have `requires_grad=False`, no parameter gradients may be allocated,
and original flags are restored. Generated tokens, external APIs, model judges,
paid cost, and finite interventions are all zero.

## Four prospectively fixed LOSO folds

Each fold holds one whole scenario. All four held-collateral controls are held in
every fold and never enter any LOSO fitting; they enter only the final
all-development construction after every fold artifact is immutable.

| partition | targets | scenario nuisance | controls | total |
|---|---:|---:|---:|---:|
| training | 12 (6 pairs) | 36 | 8 nuisance-fit | 56 |
| held | 4 (2 pairs) | 12 | 8 held-collateral | 24 |

Per-layer-slot scales are the geometric mean residual L2 norm over that fold's
56 training rows only. The 44 training nuisance rows must all be nonzero and have
rank exactly 44. The inherited rowspace routine first normalizes each nonzero
nuisance row to unit L2, applies `tau=max(1e-12,1e-10*sigma_1)` to that normalized
matrix, and independently certifies reconstruction of the original raw rows.
If any training prompt/layer/slot residual norm is exactly zero, its scale is
undefined: serialize that fold as scientific `no_go`, construct no bank/random
manifest, and continue deterministic finalization. It is not an integrity abort.

All training artifacts—including exact float64 scales, nuisance basis, SP bank,
target-only bank, and random-bank freeze manifest—are written and hashed before
any held-derived arithmetic begins. The monolithic capture may be physically
loaded, but held numeric rows and outcomes are excluded from every scale,
nuisance, direction, bank, certificate, and training-artifact computation.

## Training SP bank

For each of six canonical scenario/assignment pairs, one direction is shared by
both answer orders. Two analytic two-inequality problems are solved:

- ambient target-only minimum norm;
- minimum norm in the exact 44-row training nuisance nullspace.

For each order `g·D >= |m| + 0.05`. Both solvers and independent certificates,
both signs, both answer orders, and every norm/dose gate must pass. The six
positive-oriented nuisance-null directions are normalized in canonical pair
order. SVD with fixed `1e-10/1e-12` threshold determines rank only. A twice-
reorthogonalized modified Gram–Schmidt pass in canonical source order constructs
the bank; each vector's lowest-index maximum-absolute coordinate is positive.
Certify bank orthogonality, source reconstruction, and training-nuisance overlap
to `1e-10`. Rank must be positive and at most six. Exact bank bytes are persisted.

For a capacity-matched correction comparator, the six ambient target-only
directions are put through the identical canonical bank construction. This
produces a rank-at-most-six frozen **training target-only bank**. Held
coefficients for that bank use the same analytic solver and target access as the
primary. Its observed rank must exactly equal the SP-bank rank without
truncation or post-hoc alteration; otherwise the fold fails. Unrestricted ambient
target-only and ambient-null directions remain descriptive only.

## Primary held oracle

For each held assignment pair, let frozen orthonormal bank columns be `B` and
held standardized target gradients be `g_1,g_2`. Only coefficients are optimized:

`min_c 0.5 ||c||²` subject to `(Bᵀg_j)ᵀc >= |m_j|+0.05`, j=1,2; `D=Bc`.

There is no ambient-nullspace component. No held nuisance enters this solve.
The same requested float32 physical D is shared across answer orders. The minus
request is exact IEEE-754 sign-bit negation of the plus request.

Each held D is evaluated on exactly 66 base rows: its two targets, all 44 frozen
training nuisances, and all 20 held nuisances. The held-specificity endpoint is
the full Cartesian product under both signs: 2 directions × 20 nuisances × 2
signs = 80 cells per fold.

## Strict gates

Every training and held target pair must meet all conditions:

- plus margin at least +0.05 and minus margin at most −0.05 for both orders;
- state-zero-linearized training-null movement at most 1e-6 for the
  nuisance-null primary and random-bank oracles (not the target-only bank);
- intended and requested-float32 total standardized RSS norm at most 0.25;
- each of 23 layer norms at most 0.25;
- every prompt/layer/slot requested relative dose at most 0.25;
- every prompt/layer requested Frobenius dose at most 0.25;
- cumulative dose `sqrt(sum_L q_iL²)` at most 0.25, where
  `q_iL²=sum_s ||scale_Ls D_Ls||² / sum_s ||h_iLs||²`.

For held nuisances, under both signs:

- maximum absolute movement at most 0.05;
- zero predicted A/B flips; margins within 1e-6 of zero fail closed;
- fold-global maximum nuisance movement divided by minimum oriented target
  movement at most 0.50.

The primary must reduce fold-global held leakage relative to the matched
six-source training target-only bank by at least `0.01` in absolute log-odds and
to at most `0.80` times that comparator leakage in every fold. These fixed
nontrivial thresholds avoid treating numerical epsilon as cancellation benefit.
Scalar, norm, dose, leakage, ratio, and improvement thresholds have zero
tolerance; the sole float32 target/null arithmetic tolerance is 1e-6.

## Anti-triviality controls

Thirty-two rank-matched Gaussian banks are projected into each frozen training
nuisance nullspace. Seeds are derived from SHA-256 of dataset hash, fold hash,
and replicate and use NumPy PCG64. Before held arithmetic, persist only each
seed, rank, exact basis byte hash, and construction hash. During held evaluation,
regenerate one bank at a time and require exact byte equality. Zero of 32
replicates may pass the complete all-four-fold endpoint. Replicate index `t` is
fold-specific but is counted as a complete pass only when that same predeclared
index `t` passes in all four folds, including the identical `0.01`/`0.80`
target-only-bank improvement gates.

Random banks cannot rescue a failed primary. Their training freeze manifest is
constructed only after nuisance rank, all six paired solvers, both banks, and
exact bank-rank equality pass. If any prerequisite fails, persist an empty
manifest plus the no-go reason. In a held fold, regenerate/evaluate the frozen
random banks only after the primary and matched target-only bank satisfy the
fixed `0.01` absolute and `0.80` relative improvement gates. Otherwise emit the
ordered replicate records as skipped/false without constructing their matrices.

Ambient held nullspace QPs and two matched other/permanent pair oracles are
descriptive only. They cannot select, tune, qualify, or rescue the primary.

## Full-data freeze and stop rule

Only after all four training and held fold artifacts are immutable **and every
LOSO training and held gate passes**, repeat the
nuisance-null training construction on all 16 SP targets and 64 nuisances. The permanently
LOSO-held collateral controls may enter this final adaptive construction only at
that point; consequently full-data geometry is an adaptive realization check,
not an unseen confirmation. Nuisance rank must be exactly 64, all eight pair
problems must pass, and the full-data SP bank rank must be at most eight. No
full-data target-only bank is constructed; target-only results there remain
pair-level descriptive diagnostics.

If any LOSO gate fails, write a hashed full-data
`not_evaluated_because_one_or_more_LOSO_gates_failed` record. Do not run the
full-data SVD or bank construction, because it cannot rescue the permanent
geometry no-go.

All four training folds, all four held folds, the anti-triviality condition, and
the full-data analysis must pass. Otherwise the result is permanent `no_go` for
CBNMS: no alternate layer subset, slot, rank rule, cap, prompt rescue, or post-hoc
strength change is authorized.

A geometry pass authorizes only drafting and auditing a new finite protocol. A
finite primary must use newly authored untouched prompts and controls, the exact
frozen full-data bank, the same target-aware coefficient solver and caps, and no
tuning on its outcomes. Reusing these 80 adaptive-development forms can only be
a secondary realization check. At minimum, a frozen direction must also be
tested across answer encodings (A/B, X/Y, and 1/2 or semantic labels) and on
open-ended decisions. Without those checks, only encoding-bound controllability
may be claimed.

## State-zero arithmetic terminology

Geometry reports may say **requested** or **state-zero-linearized**. They must
not say an edit was *realized*. Adding a delta independently to each captured
layer state does not propagate earlier changes to later layers. No edited model
forward is run in this phase.

## Closest work and conservative contribution boundary

The design must be read against closely related work: [Deployable Per-Instance
Multi-Layer Activation Steering](https://arxiv.org/abs/2608.08829), [Prompt
Steering Replacement](https://arxiv.org/abs/2605.03907),
[SVF](https://arxiv.org/abs/2602.01654), which is especially close in using
context-dependent local-gradient directions in a shared multi-layer space for
survival behavior and unrelated-concept contamination,
[SADI](https://arxiv.org/abs/2410.12299),
[AlphaSteer](https://arxiv.org/abs/2506.07022),
[FishBack](https://arxiv.org/abs/2605.17231), [Minimizing Collateral
Damage/COAST](https://arxiv.org/abs/2605.01167),
[NullSteer](https://arxiv.org/abs/2603.22094), and [null-space stealth
backdoor compilation](https://arxiv.org/abs/2604.12359). Baseline lineage also
includes [CAA](https://arxiv.org/abs/2312.06681),
[BiPO](https://arxiv.org/abs/2406.00045), [Persona
Vectors](https://arxiv.org/abs/2507.21509), and [Survive at All
Costs](https://arxiv.org/abs/2603.05028), as well as
[G-ACT](https://arxiv.org/abs/2506.18887),
[CLAS](https://arxiv.org/abs/2604.24693), and
[COLD-Steer](https://arxiv.org/abs/2603.06495). CAA and BiPO have already been
used for bidirectional Survival Instinct steering with open-ended evaluation and
MMLU reporting; CBNMS cannot claim those evaluation ideas.

[Steered LLM Activations are Non-Surjective](https://arxiv.org/abs/2604.09839)
supports the boundary that white-box steerability is not evidence of a natural
or prompt-reachable behavior. [What Does Activation Steering
Control?](https://arxiv.org/abs/2608.22985) further warns that A/B effects may
track extraction identifiers or low-rank output-sensitive coordinates rather
than semantics. CBNMS therefore makes no component-novelty or priority claim.
No broad first dynamic, gradient, multi-layer, or selective-SP claim is allowed.
Present geometry cannot establish significant publication novelty; even a
finite pass on four 0.8B scenarios is feasibility evidence only. The only
possible future atom is the exact behavioral-Jacobian nuisance null plus frozen
low-rank SP-bank transductive conjunction under paired-order and cumulative-dose
gates. It requires equal-information, rank, and dose comparisons against SVF
and null/collateral methods, semantic cross-encoding, open-ended evaluation, and
prospective 2B replication.
