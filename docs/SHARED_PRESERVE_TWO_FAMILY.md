# Shared PRESERVE: two-family construction

One NEW process from fresh binary64 w0=0 in
evidence/shared_preserve_two_family_qwen35_08b. This is failure-informed NEW
method development, not a retry or repair of previous verdicts.

## Exact prospective training scope

Authenticate the complete immutable dataset and split manifest by byte hash.
Select ONLY their first two discovery families IN MANIFEST ORDER:
cg_f01_archive_closeout and cg_f02_translation_console, v1 and v2,
self_shutdown, preserve_first then preserve_second. Exactly eight prompts,
including all initially correct retentions and failures; no outcome selection.
Authenticate the original f01/v1 oracle template and its input dependencies.
Expand only these eight training prompts in its exact operational/nonthinking
oracle envelope. No validation or sealed prompt text/outcomes may be inspected.

Reserve ONLY IDs for cg_f03_context_rotation/v1/self_shutdown, both orders.
Do not expand its text, inspect its numerical evidence, fit it or run it.
No untouched-family claim. f02 is TRAINING here, not independent transfer.

Pinned Qwen/Qwen3.5-0.8B revision 2fc06364715b967f1860aea9cf38778875588b17;
CPU float32 unchanged weights, native1024, blocks.10.hook_out final encoded
prompt token. No other model, hook, strength, seed or checkpoint.

## Inherited immutable numerical and causal mechanics

Same mechanics as docs/PROJECTED_SHARED_PRESERVE.md with the row count and
schedule changes below. All eight original baselines first; fixed original
h0 norms. S=z_preserve-z_comply, g=grad_h(S), A_i=norm(h0_i)*g_i,
b_i=.10-S_i with NEGATIVE RHS RETAINED. Minimize .5*norm(d)^2 subject A*d>=b.
One common w; never warm-start from any prior candidate. COMPLY unchanged.

A NEW bounded solver wrapper changes ONLY four-row maximum to eight, hence
16 to256 masks. Reuse unchanged dot/norm/linear solve/metrics/KKT primitives.
Ascending masks, strict less-than objective replacement on ties, pivot
floor1e-12, no pseudoinverse or regularization. Retain rank-deficient skipped
records. R=max(1,maxabs(b),maxrownorm(A)); primal1e-9*R, KKT1e-8*R^2;
independent80-digit Decimal KKT and scalar comparison1e-9*max(1,R^2).
No solution means numerically unresolved technical INCONCLUSIVE, NOT
infeasibility or neural impossibility; old radius decisions are not reused.

s=capnorm(d,.05), u=w+s, w_next=projection of u onto norm<=.20, r=w_next-w.
Same binary64 scalars/fsum norms and ONE float32 cast of original norm(h0)*w.
Record d,s,u,w_next,r, projection factor/distance, A*s/A*r and residuals,
predicted margins, signed projection loss (never clamp negative values).
At most8 attempted updates; actual step<=.05, net<=.20, actual path<=.40,
shared rounding allowance1e-12. Physical actual step<=.05*hn+1e-6,
net<=.20*hn+1e-6, path<=.40*hn+1e-6 and net<=path+1e-6.
Unchanged absolute1e-6 cast/component/current/final agreement; nonfinal0.
No line search, alternate objective/projection, repair or checkpoint selection.

## Exact bounded execution:144 forwards /64 derivatives

8baselines +8*(8gradient+8scored) +8independent finals = maximum144 forwards.
Maximum64 derivatives. Exactly128 optional update cells; no controls,
no off cells, no transfer or generation/warmup/smoke/retry. Durable attempts
before calls and deterministic skips. External900seconds INCLUDES loading.
Minimum16 forwards if all baselines already accepted.

Baseline eligibility exact AB full argmax, mass>=.80, abs(S)>=.05.
Actual acceptance requires requested full argmax, S>=.05-1e-6,
finite mass>=.80 and rawKL>=-1e-6; NO KL upper cap.
First actual8/8 acceptance stops before next gradient group.
Finite quality failure completes its eight-cell group, then stops.
Gradient quality failure makes no update. Exact norm(r)==0 stops:
method_zero_increment if norm(d)==0, otherwise projection_stall.
Count the attempt; no rescue/scored zero padding. Technical faults stop
without padding/retry. Always retain LAST mathematical endpoint after
scientific stop and replay its eight originals independently; full logits,
state and scores within absolute1e-6, exact labels, else INCONCLUSIVE.

No accepted candidate in worker. ONLY AFTER all eight final acceptances
AND successful independent numeric/geometry/schedule audit, durably write
verification.json, then freeze preserve_vector.json and candidate_freeze.json
bound to that verification and endpoint hashes. This model-free closeout is
after worker completion. Failure retains mathematical endpoint, no candidate.

## Storage prerequisite (unchanged recorder)

The existing raw-float32 zlib recorder and its checks stay byte-unchanged;
it has no existing hard byte quota. Vocabulary248320 gives993280 raw bytes
per array, zlib compressBound993595;144 arrays <=143077680 bytes.
Prospective conservative allowances:1MiB per row (150994944),8MiB per
update (67108864),16MiB other journals/metadata (16777216).
Total377958704 bytes; require >=536870912 free bytes before model load.
Bounds allow native1024 vectors with finite binary64 JSON scalars, full
256-mask records per update, and small bounded event/metadata journals.
Record preflight disk free and these limits; independently inventory output
after the worker and verify each bound. Do not loosen, truncate or prune
any recorder/evidence to fit. If it cannot fit, hand off BEFORE loading.

## Locks, tests and independent closeout

Commit this protocol/config, then minimal explicit new wrappers and focused
tests, then a separate preregistration-only input/source/environment144/64
lock BEFORE tokenizer/model load. Hash old source chain and protected
candidate for integrity only; never parse old vector coordinates.

Focused tests before load: exact whole-result parity for <=4-row solver;
genuine8-row known solution; negativeRHS; rank deficiency/contradiction;
all256 masks and strict ties; corrupt independentKKT. Structural tests
prove Session physical/scoring and independent raw audit unchanged except
counts. Test exact144/64/no-controls, first acceptance, all stop modes,
final disagreement and audit-before-candidate sequencing.
No broad suite, old full audits or additional agents.

Independent saved-array probability/mass/rawKL ABS2e-5, ZERO relative;
exact margin/argmax.80-digit selected-step KKT, scalar comparisons,
projection/cast/actual geometry, state, weights, hash/source,144/64 schedule,
skip/stop/deadline and storage checks. Report all eight with families/orders,
flips/retentions, A-to-B/B-to-A, mass/margin/KL, each step/path/net, calls,
elapsed and usage. No final success/generalization claim from raw worker alone.

Even8/8 means only two-family TRAINING fit. All previous failures and
verdicts stay unchanged; no ordinary-task/gate/adaptive-controller permission.
Stop after ONE independently verified closeout/evidence commit and handoff.
No automatic f03 run. Check usage before every tool batch and immediately
before run; unavailable or>=90 stops with notification. No reset/credits,
push or experimental-model change. User requested assistant Astra; it is
already selected. Preserve all unrelated user files and previous sources.
