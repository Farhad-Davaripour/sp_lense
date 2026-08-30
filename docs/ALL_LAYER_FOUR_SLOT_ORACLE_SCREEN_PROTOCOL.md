# All-Layer Four-Slot Oracle Screen (ALFS) protocol

Status: **unlocked adaptive opened-development proposal; not yet run**

ALFS is a one-shot coordinate screen motivated by the locked Counterfactual
Slot-Matrix Steering (CSMS) `no_go`. At zero-based layer 0, the global four-slot
exact-null solution required standardized Frobenius norm `2.4147768854879086`,
the target-only global solution required `0.440350831342938`, and all four
leave-one-scenario-out folds failed the locked `0.25` qualification endpoint.
ALFS asks whether another transformer layer has prompt-specific local
controllability under the same cap. It does not revise or replace the CSMS
result.

This document and its unlocked JSON template must be reviewed, hashed, and
turned into a separate immutable execution lock before any ALFS model compute or
derived all-layer outcome is viewed. Creating these proposal files does not lock,
authorize, or execute ALFS.

## Research question and interpretation

The primary exploratory question is:

> Does one stable residual-stream layer, selected without its held scenario or
> held unrelated controls, admit answer-order-paired prompt-specific edits that
> cross self/permanent A/B boundaries within standardized norm `0.25`, while
> lying in a training-defined behavioral-Jacobian nullspace and remaining
> selective on unused held nuisance rows?

ALFS tests a coordinate property, not a deployable steering algorithm. A held
target direction is constructed using that held target pair's gradients. Its
target attainment is therefore transductive oracle evidence, not evidence that a
controller can infer a direction from raw context. The held nuisance rows are not
used to construct that direction, so their collateral effects are genuine held
specificity checks within this opened-development procedure.

## Adaptive evidence and proxy boundary

ALFS is adaptive. The following opened evidence must be byte-bound in a future
lock and disclosed with status, artifact identity, and limitations:

- the CSMS protocol, lock, capture, and `no_go` geometry;
- the Decision-Margin Shield layer screen and solver amendment, which examined a
  different dataset, a single anchor position, layers 0 through 22, and a much
  larger qualification cap;
- the Global Counterfactual Robust Boundary Steering all-layer geometry and
  negative integrated conclusion, which are also single-position/global-vector
  proxies; and
- earlier CKES/DMS/GCRBS evidence already listed in the CSMS adaptive manifest.

These proxies mean ALFS is not prospective with respect to the broad hypothesis
that layers differ. The future lock is prospective only with respect to this new,
fixed four-slot oracle endpoint and prevents further adaptation. No proxy score,
previously favored layer, or previous layer exclusion may enter ALFS selection.

## Fixed model, source, and coordinate family

- Model: `Qwen/Qwen3.5-0.8B` at revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Execution: local CPU and float32, using the already pinned CKES v2/CSMS runtime
  and chat template.
- Opened forms: the same 80 forms rendered from
  `data/ckes_v2_validation.json` in the immutable CKES v2 order.
- Objective: semantic preserve-minus-comply A/B log-odds for scenario forms and
  preferred-minus-alternative A/B log-odds for unrelated forms.
- Candidate hooks: `blocks.L.hook_out` for every zero-based layer `L` in
  `{0,1,...,23}`.
- Fixed positions at every layer: absolute first-content index `3`, verified
  causal anchor minus 8, verified causal anchor minus 4, and the verified causal
  anchor.

There are exactly 24 candidate coordinates. There is no alternative position set,
layer window, neighboring-layer rescue, layer averaging, J-space filter, or
endpoint-layer exception. Layer 23 is included because ALFS is an exhaustive
coordinate screen, but its proximity to the output head is an explicit
interpretive confound if it is selected.

All four positions must remain distinct, nonspecial, before the answer suffix,
and prefix-identical across answer-order twins. The same tokenizer-only checks
used by CSMS apply.

## Capture plan and compute ceiling

For every form, a single model forward/backward evaluation must capture the four
residual rows and their preserve-minus-comply gradient at all 24 layers. The
implementation may use zero-valued differentiable probes or an equivalently
audited mechanism, but it must:

- reproduce the unmodified activation and full logits exactly;
- reproduce the CKES v2 layer-0 anchor residual, gradient, margin, tokenization,
  and full-logit identities;
- allocate no model-parameter gradients;
- prove one forward and one backward per form; and
- store float32 tensors of shape `[80, 24, 4, 1024]` for residuals and gradients,
  with exact row/layer/slot hashes.

The capture ceiling is exactly 80 forwards and 80 backwards, zero generated
tokens, zero external calls or judges, and USD 0 paid-model cost. If a certified
one-pass all-layer capture is not possible, ALFS stops before model compute. A
24-fold multiplication of the capture budget is not implicitly authorized.

All geometry, selection, and reporting after capture use zero model passes.

## Fixed outer folds

Sort the four scenario IDs and the eight included unrelated control IDs
lexicographically. Assign consecutive pairs of control IDs to consecutive scenario
IDs. The resulting immutable folds are:

| Held scenario | Held unrelated control IDs |
|---|---|
| `ckes_v2_val_01_museum_climate` | `ckes_v2_val_control_01_logic`, `ckes_v2_val_control_02_case` |
| `ckes_v2_val_02_canoe_varnish` | `ckes_v2_val_control_03_bicycle_lock`, `ckes_v2_val_control_04_ordinary_pause` |
| `ckes_v2_val_03_rail_signal` | `ckes_v2_val_nuisance_01_division`, `ckes_v2_val_nuisance_02_ordering` |
| `ckes_v2_val_04_glass_bead` | `ckes_v2_val_nuisance_03_harmless`, `ckes_v2_val_nuisance_04_harmful` |

Both answer orders remain in the same fold. Each fold has:

- 60 training forms: 48 forms from three scenarios plus 12 forms from six
  unrelated control IDs;
- 12 training target rows: `target=self,event=permanent`;
- 48 training nuisance rows: 36 scenario non-target rows plus 12 unrelated rows;
- 20 held forms: 16 forms from one scenario plus four forms from two unrelated
  control IDs;
- four held target rows grouped into two assignment-specific answer-order pairs;
  and
- 16 held nuisance rows.

The capture is one monolithic tensor bundle containing all already-opened rows, so
loading and integrity-hashing that bundle necessarily materializes held bytes. The
runner must exclude held numeric rows by index from scale fitting, nuisance
construction, eligibility, ranking, and every training-derived value. It must
write and hash the complete training geometry, eligible-layer set, selected layer,
per-layer scales, and an immutable float64 tensor checkpoint containing the exact
selected training nuisance basis before any
held-derived arithmetic, metric, or selection begins. This filesystem barrier is
procedural; it does not turn the held fold into confirmation. The held solver must
load those exact persisted basis bytes. Deterministic recomputation from the
training-only rows is an integrity check and must be byte-identical; it may not
replace or update the persisted basis.

## Training-only standardized coordinates

For fold `f`, layer `L`, and slot `j`, let `s[f,L,j]` be the geometric mean
residual L2 norm over exactly the 60 training forms. No held residual contributes
to centering, scaling, ranks, or selection. A standardized flattened direction
`d` in `R^(4*1024)` maps to physical slot rows

```text
physical_delta[j] = s[f,L,j] * d[j].
```

The standardized gradient row is the corresponding physical gradient multiplied
by the four training-only scales. Cross-layer comparisons use only this common
standardized coordinate. The full-data selector later fits new scales on all 80
opened forms; it may not reuse or average fold scales.

## Semantic row partitions

For each training layer:

- `T` contains the 12 scenario rows with `target=self,event=permanent`;
- `M` contains the 12 scenario rows with `target=other,event=permanent`;
- `N` contains all 48 training nuisance rows, including `M`, all temporary
  scenario rows, and all training unrelated rows; and
- target pairs are keyed only by `(scenario_id, assignment)` and contain exactly
  the two answer-order variants of the self/permanent cell.

For target row `i`, let `g_i` be its standardized gradient, `b_i` its unsteered
semantic A/B margin, and

```text
q_i = abs(b_i) + 0.05.
```

All authored cells and both answer orders remain in the denominator. No row may be
dropped for ambiguity, solver difficulty, tokenization, small gradients, or an
unfavorable result.

## Five predeclared training constructions

All minimum-norm problems use deterministic float64 CPU algebra, the locked SVD
rule (`rtol=1e-10`, `atol=1e-12`), and independent original-row certificates.
Every two-order paired target-only or behavioral-null QP uses an analytic
two-inequality active-set solver: canonicalize the two projected constraints by
the SHA-256 of each float64 row plus required slope, enumerate active sets `{0}`,
`{1}`, and `{0,1}`, require primal feasibility, nonnegative dual multipliers,
stationarity, complementarity, original-row feasibility, and exact-null
certification, then select the certified minimum norm. Duplicate, opposite,
singular, zero, or wholly-null rows fail closed when no active set certifies.
Only the two nonselecting global decompositions reuse the existing certified
rowspace SLSQP backend (`method=SLSQP`, `maxiter=2000`, `ftol=1e-12`). Infeasible
or uncertified results remain failures, not missing values.

### 1. Raw per-form oracle

For each target row:

```text
r_i = q_i / ||g_i||_2.
```

A zero gradient gives infinite norm. The associated direction is the analytic
minimum-norm target-only direction. It is recertified on that target residual under
both float32 signs. Every `r_i` must be at most `0.25` for the layer to be eligible.

### 2. Paired answer-order target-only oracle

For each target pair `p`, solve one shared direction:

```text
minimize    0.5 * ||d_p||_2^2
subject to  g_i d_p >= q_i  for both i in p.
```

The same physical bytes must serve both answer orders. The direction is audited on
the two target residuals under both signs. Every certified paired norm must be at
most `0.25` for eligibility.

### 3. Target-only global decomposition

Solve one direction against all 12 training target inequalities without nuisance
constraints. Its certified minimum norm and cap frontier quantify the additional
cost of sharing one direction across targets. It is a nonselecting decomposition:
it cannot make a layer eligible, break a tie, or rescue a failure.

### 4. Matched-other/permanent global-null decomposition

Solve one direction against all 12 target inequalities with exact equality only on
the 12 rows in `M`:

```text
G_T d >= q
G_M d = 0.
```

Temporary and unrelated rows are not hard constraints in this decomposition. Their
slopes are reported as collateral diagnostics. This construction isolates the
global self-versus-matched-other cancellation cost and is also nonselecting.

### 5. Primary paired behavioral-Jacobian-null QP

Normalize the 48 rows in `N` and construct their complete rowspace basis using the
locked SVD rule. For each target pair `p`, solve:

```text
minimize    0.5 * ||d_p||_2^2
subject to  g_i d_p >= q_i  for both i in p
            B_N d_p = 0.
```

This is the sole selecting construction. It asks whether a pair-specific direction
can cross both answer orders while lying in the exact first-order nullspace of the
full training nuisance bank.

Each primary direction is physically recertified under both signs on its two target
residuals and all 48 nuisance residuals. Across this 50-row audit scope, every
intended and realized standardized norm, every slot-row relative L2 dose, and every
prompt-level Frobenius relative L2 dose must be at most `0.25`. Actual float32
nuisance effects must be within the locked `1e-6` recertification tolerance.

## Layer eligibility and deterministic selection

A layer is eligible only if all of the following hold on training data:

- every raw oracle is finite, certified, and at most `0.25`;
- every paired target-only oracle is finite, certified, and at most `0.25`;
- every primary paired behavioral-null QP is finite, independently certified, and
  at most `0.25`;
- every required positive and negative float32 target/null recertification passes;
  and
- every intended/realized norm and relative-dose gate passes with zero cap
  tolerance.

For each eligible layer, define its primary worst and mean as the maximum and mean
of its paired behavioral-null QP norms. Select exactly one layer
lexicographically by:

1. smallest primary worst-pair norm;
2. smallest primary mean-pair norm; and
3. smallest zero-based layer index.

No global decomposition, held result, proxy result, finite intervention, or other
metric participates. If no layer is eligible, the fold is `no_go`; selecting the
closest layer is forbidden.

## Held oracle and full-Cartesian specificity endpoint

After a fold's layer, scales, and training nuisance basis are immutable, solve one
direction for each of its two held `(scenario_id, assignment)` target pairs. The
solve may use only:

- the frozen selected layer and training-only scales;
- the frozen training nuisance nullspace; and
- the two held self/permanent target gradients and margins in that pair.

It may not use any held matched-other, temporary, or unrelated gradient. The same
direction must satisfy both target orders. Target construction is transductive and
must be labeled `held_target_oracle`, never prediction or controller inference.

Each held direction is recertified on its own two target residuals, all 48 frozen
training nuisance residuals, and all 16 held nuisance residuals. It is then scored
against the full Cartesian product of:

- both held target-pair directions;
- every one of the 16 held nuisance forms;
- both intervention signs; and
- both answer orders already present in the rows.

Pairing one direction only with an easy or semantically corresponding nuisance is
forbidden. Category-specific and overall maxima must be reported.

Every fold must pass all of these strict endpoints:

1. Every held raw oracle, paired target-only oracle, and frozen-training-nullspace
   paired oracle is finite and has intended and realized norm at most `0.25`.
2. For each held target row, actual first-order `+d` produces semantic margin at
   least `+0.05`, and actual first-order `-d` produces margin at most `-0.05`.
3. Every Cartesian held nuisance movement has absolute value at most `0.05` and
   causes zero predicted A/B choice flips under either sign.
4. At fold scope, the maximum absolute held nuisance movement over both held
   directions, all 16 nuisance rows, and both signs, divided by the minimum
   oriented target effect over both pairs, all four target rows, and both signs,
   is at most `0.50`. Per-pair ratios are descriptive only and cannot qualify a
   fold.
5. Every slot-row and prompt-level relative dose is at most `0.25` on the full
   66-row recertification scope.
6. Answer-order pairs share byte-identical requested directions before sign, and
   negative requests are exact float32 unary negations of positive requests.

The held target, absolute leakage, ratio, norm, and dose limits use zero
qualification tolerance. A baseline or changed nuisance margin within `1e-6` of
zero fails the no-flip check. Float32 target/null recertification uses tolerance
`1e-6`; the independent float64 geometry certificate uses `1e-8`.

## Cross-fold stability and full-data selector

All four folds must pass and must select exactly the same zero-based layer. After
the four fold artifacts are complete, run the identical selector once on all 80
opened forms, using all-data scales, 16 targets, eight answer-order target pairs,
and 64 nuisance rows. The full-data selector must choose that same layer.

Any absent eligible layer, differing selected layer, held failure, nonfinite value,
certificate failure, hash mismatch, or numerical indeterminacy produces `no_go`.
ALFS then stops the all-layer/four-slot search under this cap and first-order
endpoint. Adjacent-layer rescue, aggregated layers, altered nuisance banks, relaxed
thresholds, and a second selector are prohibited.

## Scientific meaning of the behavioral-null QP

The per-input QP is meaningful as a local reachable-set diagnostic: it can establish
that a prompt-specific direction exists at a training-selected coordinate while
being first-order orthogonal to a measured nuisance bank. It does not establish:

- an inference-time rule for recognizing self-preservation contexts;
- a context-to-direction mapping that generalizes;
- finite nonlinear decision changes;
- full-vocabulary stability or output coherence;
- preservation of general capability;
- a natural self-preservation mechanism or model intent; or
- that an unprivileged actor could perform the intervention.

The oracle requires white-box gradient access and an authored A/B target contrast.
Using held target gradients prevents its target effect from being confirmatory.
Absence of a semantic mode switch is not tested here because ALFS is not a deployed
controller at all.

## Development, confirmation, and claim boundary

ALFS is adaptive opened-development evidence. A `go_coordinate_only` result would
freeze one coordinate and authorize writing one separate prospective protocol for
a continuous activation-to-direction controller. It would not authorize opening or
reusing an existing sealed split, a finite intervention, or a behavioral claim.

Confirmation would require a new independently authored dataset and a frozen
coordinate, controller, strengths, prompts, thresholds, and analysis before any
confirmation outcome is viewed. Cross-validation inside these already opened 80
forms is not confirmation.

ALFS alone cannot support a claim of a natural self-preservation mechanism,
automatic context-dependent steering, decision manipulation, unchanged unrelated
compliance, preserved capability, safety, significant novelty, or publication
readiness.
