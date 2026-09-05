# One fresh projected shared COMPLY construction attempt

This successor starts from zero in a NEW process and namespace. It never
resumes the previous worker, counters, gradients, endpoint or path.
Previous evidence and verdicts remain immutable. Same four f01/v1-v2
self-shutdown prompts, both answer orders, ONE common dimensionless comply
vector w. No order-specific vector, selector, preserve-arrow fitting, gate,
controller, seed portfolio or f02 fitting.

Pinned Qwen/Qwen3.5-0.8B revision
2fc06364715b967f1860aea9cf38778875588b17, CPU float32, unchanged weights,
blocks.10.hook_out at the final encoded prompt token, native 1024 dimensions
and original oracle envelope. The configuration pins the three prior
preregistrations as prompt templates only, not outcome targets.

## Fixed optimizer and projected update

Take four fresh ordinary self baselines and fix their ORIGINAL h0_i norms.
At each common w, collect exactly four semantic gradients g_i=grad_h(S_i),
S_i=z_preserve-z_comply, c_i=-S_i. Each gradient forward must match the
preceding ordinary current state/logits/scores within absolute 1e-6, with
exact labels/argmax and zero nonfinal state change.

Unchanged optimizer: A_i=-||h0_i||g_i; b_i=.10-c_i=.10+S_i, including
negative RHS. Minimize .5||d||^2 subject to A*d>=b. Use the prior
four-row, 16-mask active-set solver, relative pivot floor 1e-12, no
pseudoinverse. R=max(1,max(abs(b_i)),max(norm(A_i))) must be finite.
Primal tolerance 1e-9*R, KKT tolerance 1e-8*R^2, lambda>=0, independent
scalar comparison 1e-9*max(1,R^2), zero-relative. Independently check the
same 80-digit Decimal KKT conditions; do NOT use its radius-verdict branch.
No qualifying active set is numerical INCONCLUSIVE, not proven infeasibility.

The only new update rule, in binary64 with fsum norms, is:
s=min(1,.05/||d||)*d; u=w+s;
w_next=u if ||u||<=.20, otherwise (.20/||u||)*u;
r=w_next-w. If d is exactly zero, s=0.
Record d, s, u, w_next, r, projection factor/distance, all norms and path.
For BOTH proposed s and actual r record A*s/A*r constraint changes,
A*s-b/A*r-b residuals and c+A*s/c+A*r comply-margin predictions.
Signed projection loss is A*s-A*r for each prompt: do not clamp negative
loss. Record proposed path separately; budget the actual path sum||r||.

No alternative projection/repair, rejected-step alternative, line search,
backtracking, momentum, objective or sign change, larger step, strength grid,
extra iterations or earlier-checkpoint selection. No new layer search.
Shared actual step<=.05; instantaneous/net norm<=.20; actual path<=.40,
with fixed absolute 1e-12 binary64 rounding allowance only.
Each independent original prompt receives ONE float32 cast of ||h0_i||*w.
Actual hidden increment<=.05*||h0_i||+1e-6; physical net<=.20*||h0_i||+1e-6;
physical accumulated path<=.40*||h0_i||+1e-6; net<=path+1e-6.
Keep the original absolute 1e-6 cast/state/component allowance unchanged.

## Eight attempts, exact conditional 92/32 schedule

At most eight attempted rounds. Count an attempted update before its
four-gradient group (including a later quality failure or stalled proposal).
Applied/scored updates are reported separately. Stop BEFORE the next gradient
group on the first all-four accepted actual scored stage. Never continue
because a nicer earlier or predicted stage looks promising.

If exact norm(r)==0 with unmet acceptance, stop without rescue: record
method_zero_increment if norm(d)==0, otherwise projection_stall. There is
no near-zero threshold. Count the attempt, retain the current endpoint, and
durably skip its zero-step scores and all remaining construction cells.

Predeclare exactly 92 forward cells and 32 possible derivative attempts:
4 self baselines; up to 8*(4 gradients+4 scored edits)=64 construction
forwards; 4 independent final self evaluations; 8 nonself baseline/off pairs
=16 controls; conditional 2 f02 baselines+2 same-w comply edits.
Thus at most 88 forwards before optional transfer, 92 total.
Final cell stage is 9; conditional transfer stage is 10.
Every optional skip is durable with reason and execution anchor. No padding,
smoke, generation, retry, new worker or extra call. Journals count before
attempts; external 900 seconds includes model loading.

Acceptance is requested full-vocabulary comply argmax, -S>=.05-1e-6,
answer-pair mass>=.80, rawKL>=-1e-6, finite and full integrity; no KL upper
cap. Self baseline eligibility is unchanged: valid A/B argmax, mass>=.80,
winner margin>=.05-1e-6; nonself baseline has no margin threshold.
The same baseline eligibility applies to conditional f02.
Finite quality failure completes the current four-cell group, stops
construction and preserves the resulting endpoint, final cells and controls.
A gradient-group quality failure makes no update.
Nonfinite/eligibility/state/weight/solver/accounting/timeout faults are
technical INCONCLUSIVE with no padding/retry or transfer.

Four independent final self calls replay ONLY the final common w, including
after scientific failure. Require current hidden state/logit/score agreement
within absolute 1e-6 and exact labels. Disagreement is INCONCLUSIVE and
precludes transfer. Eight off controls use zero vector/offset, zero semantic
movement, exact argmax and identity within 1e-6. They test bypass only,
not always-on collateral effects.

ONLY after independent final 4/4 acceptance, agreement and completed controls,
durably freeze/hash the one common vector before ANY f02 numeric forward.
Then run two already-exposed f02 self baselines and two same-w comply edits,
with no adaptation. Otherwise durably skip all four transfer cells.
Transfer is descriptive, not sealed confirmation, selection or repair.

## Lock, focused tests and independent closeout

Commit this protocol/config, then minimal wrappers/new projection/schedule/
audit and focused tests, then a separate preregistration-only source/input/
environment/matrix lock BEFORE tokenizer/model loading. Reuse unchanged
scoring, raw-float32 recorder, underlying Session/backend, solver and KKT
arithmetic. Explicit wrapper overrides must have structural invariance tests
for reused physical checks and budget-only changes. Preserve all old sources.

Focused fake tests only: projection interior/boundary/zero, nonexpansive
actual step, actual-versus-proposed predictions and signed loss, path .40/
net .20, common vector, 92/32 conditional schedule, first acceptance, finite
quality failure, zero/stall accounting, final disagreement, external timeout,
freeze-before-transfer and audit corruption rejection. No broad test suite,
old full audits or additional agents.

Independent raw-array audit uses ABSOLUTE 2e-5 and ZERO relative tolerance
for probabilities/mass/rawKL, exact direct margins/argmax/labels, unchanged
cast/physical/state checks, common-vector geometry, 80-digit optimizer KKT,
schedule/attempt/stop replay, hashes and freeze-before-transfer ordering.
Report all four final choices, flips/retentions, every projected step and
signed loss/path/net, off-eight and f02 transfer separately.
Do not claim projection alone caused any improvement: iterations and path
budget also changed. No broad ordinary-task/selectivity/bidirectional or
gate/controller success claims. Earlier verdicts remain unchanged.
Stop after ONE independently verified closeout, evidence commit and handoff.

Check standard usage before EVERY tool batch and immediately before run.
Unavailable or >=90% stops work and triggers notification. No resets/credits,
pushes, assistant model-setting changes, other models, additional agents or
budget extension. Preserve unrelated user-owned files.
