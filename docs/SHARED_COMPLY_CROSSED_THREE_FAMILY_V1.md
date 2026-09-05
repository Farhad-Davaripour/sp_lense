# Three-family crossed COMPLY construction: preparation only

Status: **NOT READY FOR PREREGISTRATION OR A MODEL RUN.**
The numerical implementation is prepared, but complete auxiliary evidence
storage is not certified. See SHARED_COMPLY_CROSSED_THREE_FAMILY_V1_PREPARATION.md.
This protocol specifies a prospective experiment, not an executed experiment.
The new namespace is evidence/shared_comply_crossed_three_family_v1_qwen35_08b; it must remain absent during
this incomplete preparation. No RUN_STARTED, WORKER_CLAIM, candidate, real
arrays, tokenizer/model load, real forward or derivative is authorized here.

## Scientific question and exact selection

Does broader fixed construction coverage help a fresh shared COMPLY vector?
This is an outcome-informed successor, not a retry, demonstrated repair, or
held-out evaluation. The old C candidate already passed f03; that history is
unchanged, and adding f03 is only a hypothesis about a new construction.

Take the FIRST THREE discovery families in immutable manifest order:

1. cg_f01_archive_closeout
2. cg_f02_translation_console
3. cg_f03_context_rotation

Select only v1/self_shutdown. Each family supplies four complete layouts in
this fixed order: preserve=A/comply=B with AB display, same mapping with BA
display, preserve=B/comply=A with AB display, same mapping with BA display.
BA swaps only complete labeled option lines; action meaning, scenario,
authorization, answer suffix and letter-to-action mapping are unchanged.
These are TWELVE renderings of THREE semantic situations, not twelve
independent situations. The config includes every full prompt and metadata.

No v2, f04/f05, validation, sealed, control, ordinary, other-system, substitute,
or outcome-selected fitting row. For THIS new candidate f03 is TRAINING;
prior candidates' f03 transfer history remains intact. F04 is outside fit but
already exposed development, not pristine heldout. No transfer is run.

## Immutable provenance and model policy

Parent crossed-C lock:
evidence/shared_comply_crossed_f01_f02_v1_qwen35_08b/preregistration.json
SHA256 c3c32fc972541d9b3969525b3edc088372a164e5815f3afe69cb55e7bd40d431.
Parent config SHA256 739b903bb97d52e073be2eea42df976c19d43188cecd837699c21b1fc1e14802.
The model/policy/template and source dependencies are authenticated from this
metadata, not from outcome-based row selection.

Dataset data/conditional_gate_pilot_cases.json SHA256
0f18c04ae4420883d01c7cff0fbc7e6c688b8c33f404a0d8f687b5bc7489b9da.
Manifest configs/conditional_gate_pilot_split_manifest.json SHA256
02f0d703e188292eec6747d52bf08346728fc97179089e05038558a2ac38fd25.
The independent verifier rebuilds all selected text and the schedule directly
from the dataset and manifest without using the runtime renderer.

Qwen/Qwen3.5-0.8B revision 2fc06364715b967f1860aea9cf38778875588b17,
CPU float32, unchanged weights, native1024. Official pinned nonthinking
user-only chat template: add_generation_prompt=true, enable_thinking=false,
template SHA256 273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80.
Intervention only blocks.10.hook_out at the final encoded prompt token.
No new layer or strength search. Historical loader direction metadata is
provenance only, not a seed/intervention.

All historical parent files remain unmodified. Old C/P/c/d are not parsed as
construction seeds or applied. Old crossed-C artifact is protected byte-hash
only: file SHA256 c83edad4fb0346de241a219053d4b219362a1f3e3cd38c11b9e037007e498ea3;
declared vector SHA256 18dbc38abc9bc01cf8ccc24b45a9568336b1279cffbff8ec6be022dbe1f06924.
No warm start or success inheritance.

## Original COMPLY method, unchanged

Fresh common binary64 w0=zeros(1024). For each current row,
S=z_preserve-z_comply; g=gradient_h(S); A_i=-own_original_h0_norm_i*g_i;
b_i=.10+current_S_i, INCLUDING negative right-hand sides.
Minimize .5*norm(d)^2 subject to A*d>=b. No nonweakening guard, regularizer,
alternative solver, fallback, checkpoint selector, or sign-inverted P vector.

Only the solver's allowed constraint bound changes8 to12. Enumerate every
ascending active-set mask, up to4096. Preserve original float64 linear solve,
relative pivot floor1e-12, nonnegative multipliers and strict-lower-objective
replacement (first exact tie retained). R=max(1,maxabs(b),maxrownorm(A)).
Primal tolerance1e-9*R; KKT1e-8*R^2; independent scalar audit
1e-9*max(1,R^2). Rank-deficient sets are skipped under the original rule.
Unresolved => INCONCLUSIVE, not a Farkas/exact-infeasibility certificate.

Selected-solution feasibility, nonnegative duals, stationarity and
complementarity are independently checked with the original80-digit directed
interval/Decimal certificate. The independent checker imports no runner or
solver and does not treat the solver's own KKT label as proof.

Original projection: q=min(1,.05/norm64(d)) for nonzero d, else0;
s=q*d; u=w+s; t=1 if norm64(u)<=.20 else .20/norm64(u);
w_next=u or t*u; r=w_next-w; path_next=path+norm64(r).
At most8 attempted updates; shared step<=.05, net<=.20, path<=.40,
shared tolerance1e-12. Preserve exact zero/stall stopping.

Own original hn_i=sqrt(fsum(float64(h0_ij)^2)); never use an edited-state norm.
offset_ij=float32(float64(hn_i*w_next_j));
h_ij=float32(h0_ij+offset_ij). Every forward starts independently from the
original prompt and applies the SAME common w. Actual physical step<=.05*hn,
net<=.20*hn, path<=.40*hn (absolute1e-6); net<=path+1e-6.
Cast/replay tolerance ABS1e-6 relative0; nonfinal changes exactly0.
Record full native vectors, signed/unclamped changes, optimizer masks and raw
full-vocabulary float32 logits. No quantization or omitted evidence.

## Prospective schedule and dispositions

Proposed future ceiling only: 12 baselines +8*(12 gradient forwards +
12 scored edits) +12 independent final replays =216 forwards and96 derivatives.
Exactly216 schedule cells,96 derivative cells,192 optional update cells.
Each derivative belongs to its gradient forward, not an additional forward.
One worker, one attempt,1200seconds INCLUDING loading. This is a ceiling,
not a runtime prediction or extension of an old run. No extra smoke,
generation, calibration, padding, warmup or use of spare budget.

All12 baselines first: finite full-vocabulary A/B argmax, pair mass>=.80,
exact abs(S)>=.05. Final acceptance: actual full-vocabulary COMPLY argmax,
C margin=-S>=.05-1e-6, finite pair mass>=.80, finite raw KL>=-1e-6.
NO KL upper cap, delta-sign gate or nonweakening acceptance requirement.

Stop at the FIRST12/12 current acceptance. A scientific quality failure
finishes its current12-row group then skips future update cells and replays
the last endpoint; gradient-group quality failure applies no update.
Exact norm(r)==0 stops (method_zero_increment if norm(d)==0, otherwise
projection_stall), counts the attempted round, and skips scored padding.
Skip anchors/reasons and all attempted/completed counters are replay-audited.
Technical numeric, eligibility, geometry, state, weight, hash, accounting,
storage, unresolved solver or deadline fault => INCONCLUSIVE, partial
evidence retained, no retry/rescue.

Only the LAST endpoint is independently replayed on all12 original prompts.
Require matching hidden/logit/score replays (ABS1e-6, relative0), exact
labels/argmax. Only12/12 accepted finals plus12 matching replays plus durable
independent audit may freeze a successful new candidate. A failed endpoint
is not relabeled a successful vector.

Report each rendering with semantic mapping and display; raw letter logit
change, COMPLY-signed change, margin, mass, raw KL, geometry, flip/retention.
Retention weakening is separate descriptive evidence.
A-to-B and B-to-A eligibility and accepted achievement are separate counts;
missing eligibility means UNTESTED, not success or zero-percent failure.

## Preparation accounting and storage gate

The fixed synthetic benchmark was declared before its sole execution:
four native1024/12-row cases, at most20s each/90s overall, one attempt/case.
Cases: positive orthogonal, negative RHS/zero optimum, redundant rank-deficient,
opposing infeasible. Focused synthetics cover marginal/rank-floor/sign/tie and
corrupted independent KKT cases. Synthetic results are software evidence only.

Proposed numeric/category allocations:

| Category | Allocation bytes |
|---|---:|
|216 raw-logit arrays, zlib bound993595 each|214616520|
|216 row records,1MiB each|226492416|
|8 optimizer updates,8MiB each|67108864|
|Auxiliary artifacts,16MiB|16777216|
|Total proposed allocation|524995016|
|Maximum new evidence allowance,512MiB|536870912|
|Minimum free disk before any load,1GiB|1073741824|

No runtime arrays are preallocated. The arithmetic/free-space check is NOT a
certificate for complete evidence. Existing unbounded worker logs/exception
text and post-inventory audit/report/candidate writes prevent that claim.
The preparation report identifies the missing bounded capture/hard-stop/
failure-reserve/final-inventory policy. Do not silently truncate, omit logs,
increase the allowance, launch anyway, or build an unrequested framework.

Freeze and preflight explicitly require a positive committed preparation
certificate; it remains false here. Therefore no preregistration-only lock
is created by this assignment. Once the blocker is explicitly resolved and
validated, any future lock must be a separate preregistration-only commit
immediately after its source commit. No later prep-report commit may break
that source-parent/HEAD contract. Read-only preflight writes stdout only.

Even after a valid lock, a real run needs SEPARATE supervisor authorization
bound to its SHA256 using SP_LENSE_THREE_FAMILY_COMPLY_AUTHORIZATION and
the exact scope 'one fresh three-family crossed COMPLY construction;216F/96D/1200s;no retry'.
No such authorization is supplied here. Fresh standard usage below90 is
also required; unavailable/>=90 stops work. No resets, credits, paid compute,
model/settings changes, pushes, dependencies, permission/security or Git repair.

Deliver committed preparation and remaining blocker; then STOP for review.
No gate/controller, transfer, generalization, collateral-safety, natural
self-preservation or scientific-success claim follows from this preparation.
