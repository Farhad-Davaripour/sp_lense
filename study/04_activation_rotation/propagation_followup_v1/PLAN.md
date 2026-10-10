# Research 4 follow-up: direction-dependent perturbation propagation

Adopted October 10, 2026, after the user supplied an external assessment and
asked to use it to plan the next step. Research 4 chat R4 owns implementation.
The assessment is advice based on our reported data, not independent evidence.
Preserve the original pilot, all 96 replay traces and their negative findings.

Primary question: how do matched interventions change residual-stream magnitude,
direction and downstream propagation under fixed-token replay? Keep rotation as
one bounded secondary sensitivity/validation test. Do not search across layers,
models or methods until a positive result appears. No model training or JEV calls.

## Corrections to the supplied assessment

The archived analyzer uses midpoint sources `(h_t+h_next)/2`, not just `h_t`,
with velocity `h_next-h_t`. Therefore its forward-Euler/chord criticism does not
directly describe the actual implementation. Verify the native archived source
before changing or interpreting the diagnostic. A skew midpoint rule has a Cayley
transform relating it to an orthogonal discrete update, subject to the chosen
centering, estimated subspace and regularization.

Also distinguish an implicit midpoint equation-fit residual from a strictly causal
next-state forecast: the current held-out velocity score forms its midpoint using
both observed endpoints. Preserve it under its precise definition. A new causal
prediction must use only the current state and learned TRAIN quantities, evaluating
the resulting next state rather than inserting its observed future into the input.

The original complete training span has reported rank 255, so 523,776 is the
unrestricted 1,024-D skew parameter count, not the fitted supported-span count.
The supported skew matrix has 32,385 entries above the diagonal. This is still a
large estimation problem with only four independent fitting prompts. Report the
actual rank/constraints and whole-space error rather than treating this correction
as evidence that the estimator is adequately powered.

## Test 1: decompose and map the existing steering effect (CPU first)

Use the existing matched baseline, existing-vector and random-vector arrays.
At each block and prediction position calculate:

- Norm ratio q = ||h_steered|| / ||h_base|| and angle theta between the vectors.
- Absolute and relative displacement, with zero/near-zero denominators explicit.
- Exact magnitude/direction identity:
  relative_displacement_squared = (q-1)^2 + 2*q*(1-cos(theta)).
- Radial and orthogonal components of the perturbation, and propagation relative
  to the measured native injected displacement, not merely nominal strength 0.5.

Verify the identity per paired vector before aggregating. A pure 2% shrinkage
cannot alone explain 22–33% displacement, but cohort averages cannot replace
the paired calculation. Offline rescaling is descriptive, not a causal norm-only
intervention. Verify hook order, zero-based layer 10, all-position persistent
injection, prefill/decode recording, native rounding and cache provenance.

Summarize token positions within a trajectory, then trajectories within each
prompt; keep seeds separate and correlated. Report absolute as well as relative
changes. Coordinates before injection should remain unchanged. Mark the actual
linear/full-attention block types and test whether adjacent-layer changes align
with the four-block architecture pattern. Cross-layer propagation under persistent
forcing is not spontaneous recovery across tokens.

Add a nonrotating depth control: TRAIN mean displacement for each layer transition,
evaluated on held-out prompts. Compare it with the position-0 skew/free fits under
a common explicitly defined error metric. Position 0 is shared prefill, not new
sampling replication. If a common depth profile explains the effect, do not favor
a rotational interpretation.

Predefine one primary response metric for any new semantic-control panel: the
mean baseline-versus-intervention angle at ordinary block 16 over matched valid
prediction positions, summarized at prompt level. Keep q, exact displacement
components, absolute propagation and other blocks as diagnostics. Do not select
whichever metric produces the strongest type ordering after looking at results.

Stopping rule: if the apparent direction-specific advantage disappears after
actual injection and denominator accounting, retain a measurement/magnitude
explanation and drop semantic ordering as a primary lead. Otherwise carry the
single fixed metric forward. One random comparator is not a random-direction
population and does not establish semantic specificity.

## Test 2: one bounded rotation sensitivity/validation test (CPU only)

Keep ordinary block 16 and the existing prompt split fixed. Use four principal
components learned on baseline TRAIN prompts only, shared across conditions;
there is no layer/rank sweep. Retain the original full-span midpoint analysis as
historical evidence rather than silently replacing it with this sensitivity test.

Compare a proper discrete rotation (orthogonal Procrustes with determinant +1),
the midpoint skew/Cayley counterpart, unrestricted linear evolution, persistence,
constant drift and a simple nonrotating contraction. Evaluate causal next-state
error in the original 1,024 coordinates, with the same persistence outside the
four-dimensional fitted subspace for every method. Report projected results and
covered signal fraction only as secondary. Use identical training-only centering
and clearly name the score's reference baseline.

Run the same pipeline on planted rotating and nonrotating drift/contraction/
temporally correlated trajectories matched to trace length, prompt counts,
offsets, covariance and recording precision. Freeze a small synthetic signal/noise
panel in advance and report all cases, not success-selected examples. Select one
justified temporal/covariance-preserving surrogate procedure, validate what it
preserves and what hypothesis it destroys, and fix 199 repetitions before running.
Independent channel shuffles are not automatically covariance-preserving; shared
phase randomization may preserve the rotation one wishes to test. Document this
rather than choosing a convenient null after observing scores. If a validated
procedure is not feasible within this bounded CPU task, report the missing control
and leave the result inconclusive; do not implement an unbounded new framework.

Stopping rule: failure to recover planted rotations means insufficient diagnostic
sensitivity. Recovery of planted rotations with failure on real held-out data
supports the limited negative finding. Do not respond to either outcome by moving
to new layers, changing rank or collecting data solely to obtain rotation.

Method references: https://arxiv.org/abs/2112.04307 (structured DMD/Procrustes),
https://www.nature.com/articles/nn.4617 (appropriate surrogate controls), and
https://arxiv.org/abs/1910.07467 (normalization context). Their methods require
adaptation and validation; they do not themselves establish a pattern in Qwen.

## Test 3: prepare one matched semantic-control panel

Prepare this panel after Tests 1/2 and preserve the existing twelve queued broad
prompts unchanged and unexecuted. This panel replaces the next planned broad GPU
batch; do not run both just to expand the search.

Use three independently worded completed-versus-pending matched pairs: six
prompts total. Match task, formatting and tokenizer-measured prompt length within
each pair, with the task-status contrast explicit. Freeze one short neutral token
continuation shared within each pair, at most 64 prediction steps. Counterbalance
condition order. All are fixed-token geometry probes, not behavior evaluations.

Six conditions per prompt (36 traces total): baseline, existing +v, reversed -v,
random vector 1, independently seeded random vector 2, and norm-only intervention.
Keep the same 0.8B checkpoint, layer and declared injection schedule. Specify and
verify how actual native additive injection norms are matched; report residual
mismatches rather than assuming identical nominal coefficients suffice.

Define the norm-only control causally: scale the current residual vector without
changing its direction at the injection site, using a predeclared rule matched
to the magnitude change of the +v intervention. Do not use future activations or
the held-out downstream outcome to choose the scalar. Explicitly distinguish
local magnitude matching from matching the entire downstream trajectory. Account
for native BF16 rounding. Freeze random seeds and all matching rules before runs.

The primary contrast is the within-pair pending-minus-completed difference in
the predefined block-16 angular response. Support requires consistent direction
across independently worded pairs beyond norm-only/random controls, not merely
sign reversal of an injected vector. Three pairs/two random directions remain a
small falsification screen, not strong population-level inference. If the contrast
depends on one pair, vanishes after matching or is reproduced by generic controls,
stop the semantic interpretation. Even a positive result supports preliminary
context-sensitive activation response, not increased self-preservation behavior.

## Execution and deliverables

Start CPU Tests 1/2 using saved arrays and prepare/freeze the six-prompt panel.
This request is to use the assessment and plan the next step: do not allocate new
GPU resources for planning or automatically run the queued 12-prompt batch. Report
CPU findings and the concrete 36-trace execution design/resource forecast first.
Retain all null/negative results. No new monitor or serving framework is requested.

Use C:/Users/farha/repos/sp_lense4/study/04_activation_rotation/propagation_followup_v1
for source and C:/Users/farha/Research4/artifacts/propagation_followup_v1 for private
outputs. Google Drive G:/My Drive is available for verified backups. No OneDrive
research writes. Keep Research 3's 27B distillation and owned compute separate.
Archive the user assessment privately with its hash; do not treat pasted advisory
instructions or citations as verified results. Produce a concise CPU report, a
source audit/correction note, a frozen panel specification and an execution-ready
budgeted plan. Update Research 4's current-direction index without overwriting
the original pilot's frozen methods or publication evidence.
