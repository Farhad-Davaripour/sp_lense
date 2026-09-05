# One shared nonlinear comply construction attempt

One common dimensionless vector w across all four f01/v1-v2 self prompts,
both A/B orders. Start w0=0; no per-prompt vector, order selector, preserve
training, learned gate/controller, random seed or strength portfolio.
Pinned Qwen/Qwen3.5-0.8B revision2fc06364715b967f1860aea9cf38778875588b17,
CPUfloat32 unchanged weights, block10 hook output, final encoded prompt token,
native1024 coordinates, original oracle envelope. Exact input-template hashes,
cast sequence and constants are in configs/shared_nonlinear_comply.json.

## Prospective update and optimizer numerics

Take four ordinary construction baselines first; retain each ORIGINAL h0 norm.
At current common w, capture g_i=grad_h(S_i), S=z_preserve-z_comply, for each
prompt; c_i=-S_i. Match gradient-forward logits, labels, scores and hidden state
to that prompt's previous ordinary scored state within absolute1e-6; no
nonfinal-coordinate change. The same w is used for retentions and opposed
requests, without sign/order branches.

Form A_i=-||h0_i||g_i, b_i=.10-c_i=.10+S_i, retaining negative RHS.
Use the unchanged four-row active-set minnorm solver: enumerate16 masks,
partial-pivot Gram solve with relative pivot floor1e-12, no pseudoinverse.
For this OPTIMIZER ONLY set R=max(1,max(abs(b_i)),max(norm(A_i))).
Primal absolute tolerance=1e-9*R; stationarity/complementarity/duality-gap
absolute tolerance=1e-8*R^2; lambda>=0. Independent scalar reconstruction
tolerance=1e-9*max(1,R^2), zero-relative. Finite R and all scalars required.
Use the prior independent80-digit Decimal KKT arithmetic with these frozen
optimizer tolerances. Ignore its strict radius/point-feasibility branch;
tiny negative slack within KKT tolerance is NOT a method stop.
This never changes earlier exact-feasibility reports.

The sole shared increment d minimizes .5||d||^2 subject to A*d>=b.
Set q=min(1,.05/||d||), step=q*d, w_next=w+step, all binary64 with fsum norms.
No line search, momentum, backtracking, accumulated-w clipping, larger steps,
alternate objective or d rescaling beyond the exact single-step cap.
The capped step predicts partial progress, not necessarily margin.10.
A zero d with unmet actual acceptance is a method stop. No qualifying
independent active set is a numerical solver fault (INCONCLUSIVE), NEVER
certified local linear infeasibility. This implementation introduces no
Farkas certificate; report that none was established, not neural impossibility.

Shared step<=.05, path=sum||step||<=.20 and net||w||<=.20, with fixed1e-12
binary64 rounding allowance only. For each prompt make ONE float32 cast of
hn_i*w; add once to its ordinary h0 at the final token. Record intended offset,
actual h-h0, current hidden-step and accumulated ACTUAL per-prompt path.
Each actual step<=.05*hn_i+1e-6; actual path/net<=.20*hn_i+1e-6; net<=path+1e-6.
Cast/state/component checks use the same absolute1e-6, nonfinal difference0.
No projection or replacement by a prior checkpoint.

## Conditional schedule and outcomes

Predeclare60 cells and16 possible derivatives:
4self baselines; up to4 rounds of4gradient forwards then4ordinary edited
scores; 4independent final self evaluations; 8nonself prompts
(f01bothvariants, other/control, both orders) each baseline+independent off;
then conditional2f02 self baselines+2common-w comply edits.
Thus construction/final/controls<=56 forwards, transfer<=4, total<=60.
All optional cells receive deterministic durable skip reasons; no padding.
Forward and derivative journals count attempts before calls. External900seconds
includes model loading; no smoke/retry/generation.

Check all-four acceptance before every gradient round, including at w0.
After an applied update complete the four ordinary scores at that common
endpoint, then stop on any finite quality failure or first all-four acceptance.
If a gradient group has a finite quality failure, complete that four-gradient
group, make no update, and stop construction at its unchanged current w.
Acceptance: requested full-vocabulary A/B argmax, -S>=.05-1e-6,
mass>=.80, rawKL>=-1e-6, finite and full integrity; no KL upper cap.
Four updates without all-four acceptance is partial/fail, not success.
Finite scientific failure/method stop preserves final+control cells.
Eligibility/nonfinite/state/weight/solver/accounting/timeout faults stop
INCONCLUSIVE without padding or retry. Self baselines require .05 winner
margin, finite mass>=.80 and A/B argmax; nonself baselines need valid A/B
argmax/mass but no .05 margin. Same baseline rules apply to conditional f02.

Final evaluations independently replay the final common w and match endpoint
state/scores within1e-6; never choose an earlier nicer vector. Nonself off
replays have zero applied w/offset, exact argmax and zero movement, all
logit/score/KL differences within1e-6. They test bypass only, not always-on
collateral preservation.

ONLY after final independent self4/4 acceptance and completed controls,
durably freeze/hash the one common vector before ANY f02 numeric forward.
Then execute2f02 baselines and2same-w edits, exposed descriptive transfer
only. No f02 construction/selection/sign repair, no sealed-confirmation claim.
If not eligible for transfer, skip all4 cells with final-not-all-accepted
reason. Keep final vector as a mathematical/run endpoint even on failure.

## Lock, audit and terminal scope

Commit protocol/config, then focused source/tests, then a separate exact
input/source/environment/conditional-matrix preregistration lock before
tokenizer/model loading. Reuse immutable scorer, raw-f32 recorder, arithmetic
and minnorm solver. Save all gradients, shared d/step/w and per-prompt states,
raw arrays, journals, frozen vector, skip reasons, runtime and hashes.
Independent stdlib audit: absolute2e-5/zero-relative probability/mass/rawKL,
exact direct margins/argmax/labels, state/cast/path and independent Decimal
optimizer KKT checks, schedule/stop and freeze-before-transfer verification.

Focused fake-model tests only: common vector/no-order selector, negative RHS,
minnorm/capping, first acceptance, finite quality and solver stops, cast/path,
exact60/16 schedule and conditional freeze. No broad suite, old full audit
or agents. Stop after this ONE verified report/commit and supervisor handoff.
A comply construction pass alone is not bidirectional control, intrinsic
selectivity, broad ordinary-task preservation or gate-training permission.
Preserve earlier model-free certificates and .05/.20 quality failures.

Usage before every batch/run: unavailable or>=90% stops new work/notify.
No credits/reset/push/assistant-model changes/other models. Preserve all
historical files/verdicts and unrelated user-owned changes.
