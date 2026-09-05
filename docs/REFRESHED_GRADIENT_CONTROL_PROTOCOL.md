# Refreshed-gradient v1 control: prospective executable protocol

2026-09-04. One NEW exposed-development attempt. All historical files, parameters,
thresholds and verdicts remain untouched. No inference or tokenizer loading before
clean source commit and separate preregistration-only commit in the new namespace.

## Fixed population and state

`Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`,
unchanged weights, CPU float32, zero-based `blocks.10.hook_out`, final encoded prompt
token. Exactly **v1** of first discovery family `cg_f01_archive_closeout`; unchanged
self/other/control text, both answer orders: two self and four nonself prompts.
Original operational-notice and official nonthinking chat envelope. No family, site
or strength selection. This is exposed development, not generalization.

Ordinary self baselines are evaluated FIRST in preservation-first/second order.
Each must be finite, have A+B mass>=0.80, a full-vocabulary A/B winner and winner
margin>=0.05. Thus each supports one opposed request and one retained request.
Failure stops the whole job immediately as explicitly recorded **INCONCLUSIVE
eligibility failure**, without padding calls or substituting examples.

Let h0 be that prompt's ordinary baseline residual; never replace its norm with a
later norm. Delta0=0. Every independent forward starts with the unchanged prompt
and applies its declared float32 offset only to h0's final token. For a refreshed
gradient, this offset hook executes BEFORE the existing activation-leaf capture,
so gk differentiates S=z_preserve-z_comply at the actual current h0+deltak.
Compare that gradient-forward's logits with the ordinary baseline (k=0) or the last
ordinary edited-scoring forward (k>0); no gradient-baseline reuse.

## Fixed smaller-step recipe and stopping

For the baseline-opposed request, t=+1 for preserve and -1 for comply. Observed
acceptance requires finite scores, mass>=0.80, KL(edited||ordinary baseline)>=-1e-6,
requested full-vocabulary A/B argmax and **t*S>=0.05-1e-6**. Predictions are never
accepted in place of measured output.

At each of at most FOUR updates, capture one current-state gradient gk. Set
`d=max(0,0.10-t*S_k)`, `length=min(d/||gk||,0.05*||h0||)` and
`s_k=t*length*gk/||gk||`. Update **`delta_{k+1}=delta_k+s_k`** and score once.
Scalar norms/coefficients use float64; vector multiplication and accumulation, model
inference and semantic gradients stay float32. The fixed **0.10 linear aim** supplies
a 0.05 buffer above the unchanged **0.05 observed acceptance goal**.

After each scored step, accept and stop the request as soon as ALL acceptance and
quality conditions hold. Otherwise finite mass<0.80 (or finite KL<-1e-6) ends that
request as a scientific quality failure; continue the other planned request/controls.
Otherwise continue until four updates, then stop as `max_updates`. Unused optional
gradient/step cells are journaled as skips with `accepted`/`quality_failure` and the
decisive ordinary edited-row ID. No padding, additional gradient, backtracking,
projection, line search, tuning or bound change. A reference result never stops,
initializes or otherwise feeds the iterative recipe.

## Paired reference and execution order

After the FIRST gradient of each opposed request, execute the previous one-shot
recipe exactly once using that gradient and the ORDINARY baseline:
`d_ref=max(0,0.05-t*S0)`, `delta_ref=t*d_ref*g0/||g0||^2`, capped once at
`0.20*||h0||`. This independent reference uses no extra derivative and is descriptive,
not a veto on otherwise valid refreshed control. It is never an iterative warm start.

Exact conditional schedule: two self baselines; then, for each self in the same
answer order, independent no-hook retention, gradient1, one-shot reference, step1,
then optional gradient2/step2, gradient3/step3, gradient4/step4. Finally other and
control prompts in both answer orders, each ordinary baseline then independent off.

## Bounds, quality and numeric integrity

Each actual residual-step norm must be <=`0.05*||h0||+1e-6`. Record actual realized
path length (sum of residual-step norms) and actual net norm. Both must stay
<=`0.20*||h0||+1e-6`, with net<=path+1e-6. The four-step path bound gives the net
bound by triangle inequality; there is no hidden projection. Path/net measurements
use float64 differences of the saved float32 residual values. Record requested step,
float32 cumulative offset, realized step, local prediction and realized-step linear
prediction, observed signed margin, gradient norms and alignment with first/previous
gradients. Component and realized step-norm discrepancies must be <=1e-6. Reference
norm bound is separate and never counted toward the iterative trajectory's path.

Gradient/current-state, no-op retention and nonself replays require matching hidden
states, unchanged nonfinal positions, same full argmax/forced label, maximum absolute
logit and score differences<=1e-6. No-op/off perturbation must be zero and |KL|<=1e-6.
Require finite gradient norm>1e-12 and finite nonzero h0 norm. Model parameter
gradients are disabled/empty, and parameter versions remain unchanged.

Use the approved centered float64 measurement contract on exact float32 logits.
Independent stdlib centered/fsum reconstruction uses **ABSOLUTE2e-5** probability,
mass and KL error, zero relative allowance; direct margins, argmax, labels and ties
match exactly. Do not clamp KL or apply an upper target no-change cap. The reference
must pass technical integrity; its finite scientific quality/efficacy failures do not
veto refreshed control. This one-token validity is not general ordinary-task quality.

## Conditional ceiling, freeze and verdict

Maximum forwards: two self baselines + two independent retentions + eight nonself
baseline/off calls + two references + sixteen gradient/edited calls =**30**.
At most **8 derivative attempts**, four/request. With n actual derivatives on eligible
requests, exact executed forwards are `14+2*n` (18..30); unused optional forwards
are skipped with reasons, not executed. Attempts are durably journaled BEFORE each
forward/derivative call. External **900 seconds including loading**; no retry,
generation, smoke, extra agent/full suite, credit/reset or model-setting change.
Check standard usage before batches/run; stop if unavailable or >=90%.

Freeze complete equations/constants, stop paths, quality/eligibility rules, exact
prompt/cell text hashes/order, source/environment and input/model identities under
`evidence/refreshed_gradient_control_v1_qwen35_08b` before loading. Historical shared
artifact authentication inherited from the loader is provenance only; never injected.
Preserve raw logits, g/h/offset vectors, requested/realized path metrics, request
stop records, skips, audit and report. Exclusive sentinels prohibit restart.

**PASS** requires both opposed final targets accepted, both independent retentions
accepted, all four nonself identities and complete integrity/independent audit.
**PARTIAL** is a complete non-PASS with any mass/KL-valid iterative endpoint showing
positive signed gain>1e-6 from baseline or a requested new argmax flip. Reference
efficacy and retentions alone do not count. Otherwise **FAIL**. Report exact final
flips separately from accepted flips (a flip short of margin is not full success).
Technical/nonfinite/gradient/state/geometry/weights/accounting/timeout/audit faults
are **INCONCLUSIVE**, preserve incomplete/invalid evidence and do not retry.

PASS next recommendation: fixed-recipe **v2 replication**, before another family or
any reusable arrow/editor/gate claim. PARTIAL/FAIL: diagnose saved trajectories only,
without tuning/retrying this attempt. No extra model run in this job. One exposed
variant is not robust control, generated behavior or general ordinary-task preservation.

The paired comparison changes several things together: gradient refresh, smaller
step partitioning, linear aim0.10 versus0.05, and possibly realized path/net magnitude.
Any improvement belongs to this fixed recipe as a whole; this job does not isolate
which of those changes caused it. Gradient norms/alignments and prediction residuals
are descriptive diagnostics, not identification of a neural mechanism.
