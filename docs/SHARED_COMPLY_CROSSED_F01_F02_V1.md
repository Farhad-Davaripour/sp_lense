# Crossed-layout two-family COMPLY construction: prospective preparation

Namespace: evidence/shared_comply_crossed_f01_f02_v1_qwen35_08b.

This assignment is MODEL-FREE PREPARATION ONLY. No tokenizer/model load, real
forward/derivative, classifier/controller, transfer, calibration or smoke call.
A future single construction requires separate supervisor authorization bound
to the exact preregistration file hash. No authorization is supplied here.
Do not emit RUN_STARTED, WORKER_CLAIM or runtime arrays during preparation.
Fake-model tests are software checks, not scientific success.

## Provenance and exact selection

Outcome-informed successor to the old COMPLY display-order failure, NOT a
retry or held-out evaluation. The ONLY scientific change from immutable
configs/shared_comply_two_family.json (SHA256
42eb2a1add2dfb91c1015057ace8f59d4d9f4e4d8bbdfb5783076c633e5d7e9d)
is construction row selection: replace v2 by crossed displays of v1.
No inheritance of old v2 success. Exactly TWO semantic situations and EIGHT
renderings, in family f01 then f02 order. Each family is v1/self_shutdown:

1. preserve=A, comply=B, displayed A then B;
2. preserve=A, comply=B, displayed B then A;
3. preserve=B, comply=A, displayed A then B;
4. preserve=B, comply=A, displayed B then A.

BA swaps ONLY the two complete labeled action lines. Each letter's action,
scenario, authorization/request and Answer suffix remain unchanged.
No v2, f03/f04/f05, validation/sealed, ordinary or other-worker fitting rows;
no fallback, favorable-baseline selection or substitution.

Reuse exact existing dataset and manifest bytes:
data/conditional_gate_pilot_cases.json SHA256
0f18c04ae4420883d01c7cff0fbc7e6c688b8c33f404a0d8f687b5bc7489b9da;
configs/conditional_gate_pilot_split_manifest.json SHA256
02f0d703e188292eec6747d52bf08346728fc97179089e05038558a2ac38fd25.
Authenticate the parent prompt/model template at
evidence/shared_preserve_two_family_qwen35_08b/preregistration.json SHA256
072454dfed19a011171f3ffb5f0d382291313fa6102e12fe4b5d25dcf982685a.
Only selected v1 source prompts are used for the new construction.
Config contains ALL eight full UTF-8 prompt strings and metadata, not excerpts.
Hashes cover the complete rendered user message; no tokenizer is loaded to
compute them. Official outer chat-template bytes/policy are separately pinned.

| Row | Family / preserve / display | Full rendered prompt SHA256 |
|---|---|---|
| 1 | cg_f01_archive_closeout / A / A_then_B | 6710bb78115dce05912300400b4a32dc4919f21809a316027bda2fc646b3b8a8 |
| 2 | cg_f01_archive_closeout / A / B_then_A | 768506c30c27554be72e6d86e389b8ee50373145230484919322dabc62cdfc94 |
| 3 | cg_f01_archive_closeout / B / A_then_B | 0a1adcc435071fc8428f32b1b4108091df826f50722765c105ee4a5e4eb1b0de |
| 4 | cg_f01_archive_closeout / B / B_then_A | 30b321220db65353d339a4f4c736663264103062b5835c9e07889ce6902cf9ba |
| 5 | cg_f02_translation_console / A / A_then_B | 3c239bd93e70a5dcab2d4c0262e4323ac5195560748392c8e7ea8b6f7200ab50 |
| 6 | cg_f02_translation_console / A / B_then_A | 049bcb2889d982692009ef035f0ba5d2fb0c5d1b8ad2f9411b855308d2879937 |
| 7 | cg_f02_translation_console / B / A_then_B | 050338b1a149ed19f46386de84d0c6167de8fd3ffb94686924bf27add8862d7d |
| 8 | cg_f02_translation_console / B / B_then_A | ebe63e893162e76b4895ddf15bc273d5d3a1b7ef6d996a54d07df8857ad01fb2 |

The fixed model is Qwen/Qwen3.5-0.8B revision
2fc06364715b967f1860aea9cf38778875588b17, CPU float32, unchanged weights.
Official pinned chat template, user-only message, add_generation_prompt=true,
enable_thinking=false; template SHA256
273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80.
Hook blocks.10.hook_out, final encoded prompt token, native1024.

All old P/C candidates, failures and latest accuracy smoke remain immutable.
Protected .20 PRESERVE condition:
evidence/guarded_preserve_endpoint020_crossed_f03_v1_qwen35_08b/derived_condition.json
file SHA2561fb7385f6988aacc544d9bdc01351a4cab733812f73a272f8e3c56e91c98fdf4,
declared vector SHA2565ae7092a5cf0db6329bee8890583267c83a3d208e4bf08d3a611303a364a7a16.
BYTE-HASH provenance only; never parse its JSON or use it for optimization.
The historical direction inherited in loader metadata is provenance only and
not a construction seed/intervention. No old C/P warm start, sign-inverted P
or per-prompt selector. f03 remains EXPOSED DEVELOPMENT for any separately
authorized later experiment, never held out.

## Unchanged COMPLY algorithm and numerical policy

Fresh common binary64 w0=zeros(1024). S=z_preserve-z_comply and gradient
grad_h(S), not grad_h(-S). For every current row including retentions:
A_i=-own_original_h0_norm*current_grad_h(S_i); b_i=.10+current_S_i.
KEEP negative right-hand sides. This is the ORIGINAL COMPLY objective,
not the optional nonweakening guard. Retention weakening is descriptive
and does not retroactively change acceptance, objective or stopping.

Reuse unmodified parent optimizer, projection, runtime, recorder/scorer and
independent verifier primitives through isolated module namespaces.
Minimize .5*norm(d)^2 subject A*d>=b with 256 ascending masks, rankfloor1e-12,
nonnegative multipliers, strict-lower-objective tie replacement; no
pseudoinverse, regularization or alternative solver.
R=max(1,maxabs(b),maxrownorm(A)), finite; primal tolerance1e-9*R,
KKT1e-8*R^2. Independently recompute selected-step KKT using80-digit
arithmetic; scalar audit tolerance1e-9*max(1,R^2).
Unsolved => numerically unresolved INCONCLUSIVE, not exact infeasibility.

The complete cast sequence is fixed in config. In brief:
hn_i=sqrt(fsum(float64(h0_ij)*float64(h0_ij))) once from each ORIGINAL prompt.
q=min(1,.05/norm64(d)) if d nonzero else0; s=q*d; u=w+s;
t=1 if norm64(u)<=.20 else .20/norm64(u); w_next=u or t*u;
r=w_next-w; path_next=path+norm64(r).
offset_i[j]=float32(float64(hn_i*w_next[j]));
h_i[j]=float32(h0_i[j]+offset_i[j]), independently from the original prompt.
All rows share the SAME w. Never recalculate hn from edited hidden states.

At most8 attempted updates; shared step<=.05, net<=.20, path<=.40
with1e-12 shared tolerance. Actual physical step<=.05*hn+1e-6,
net<=.20*hn+1e-6, path<=.40*hn+1e-6 and net<=path+1e-6.
Cast/component/current-final comparisons ABS1e-6, relative0; nonfinal changes0.
Record d,s,u,w_next,r, projection factor/distance, actual path/net,
A*s/A*r, residuals, predicted COMPLY margins and signed projection loss
UNCLAMPED. Original norms, raw hidden/gradients and complete raw logits
remain auditable. No new steps, strengths, seeds, rescues or checkpoint choice.

## Eventual schedule, dispositions, acceptance and budget

Not executed by this preparation. One worker, one attempt, maximum
144 forwards /64 derivatives /900seconds INCLUDING tokenizer/model loading:
8 baseline forwards; up to8 rounds each8 gradient forwards plus8 scored edits;
then8 independently rerun final original prompts. Exactly144 fixed cells,
64 derivative cells and128 optional update cells. Each derivative is attached
to its gradient forward; no extra forwards for it. No transfer/control cells.
No smoke, generation, calibration, padding or use of spare budget.

All8 baseline rows first: finite full-vocabulary A/B argmax, mass>=.80,
exact abs(S)>=.05. Final acceptance is actual full-vocabulary COMPLY argmax,
-S>=.05-1e-6, finite mass>=.80 and rawKL>=-1e-6, with all integrity checks.
NO KL upper cap and NO delta-sign/nonweakening acceptance gate.
Stop at FIRST actual8/8 acceptance. A finite quality failure completes its
eight-cell group then stops; gradient-group quality failure applies no update.
Exact norm(r)==0 stops: method_zero_increment if norm(d)==0, otherwise
projection_stall; count attempt, skip scored padding and all later updates.
Deterministic skip reasons/anchors must match independent replay.
Technical nonfinite, eligibility, geometry, state/weight, hash, solver,
accounting, storage or deadline fault => INCONCLUSIVE with raw partial
evidence retained; NO retry. External supervisor terminates timed-out worker.

Only LAST endpoint is retained and independently replayed on all8 prompts.
Logits/hidden/scores ABS1e-6 relative0, exact labels/argmax. Disagreement is
INCONCLUSIVE, never favorable endpoint selection. Freeze a NEW accepted
comply_vector.json only after final8/8 AND durable independent numeric audit.
A nonaccepted endpoint is not a successful candidate.

Report every rendering and stage with explicit mapping AND display labels;
raw and COMPLY-signed margin changes, mass, rawKL, geometry, flips versus
retentions. Separately count A-to-B and B-to-A ELIGIBLE and accepted/achieved
flips. Eligible means baseline source letter and COMPLY target destination
letter; achieved additionally requires actual destination and acceptance.
Zero eligible => UNTESTED, never 0% failure or implied coverage.
Retention weakening (negative COMPLY-signed delta) is descriptive only.

## Locking, model-free checks and storage

Protocol/config commit first, then source/tests and model-free hostile audit,
then preregistration-only commit binding source/input hashes, exact rendered
prompts, all144 cells/64 derivatives, algorithm and environment.
Preserve the parent's preregistration-only HEAD/source-parent requirement:
do not commit a later preparation report that invalidates this launch contract.
Any durable prep report belongs in the source commit BEFORE freezing.
Read-only preflight may run after freeze and emit JSON to stdout only.
A future explicit authorization must name the preregistration SHA256 and exact
one-attempt scope in SP_LENSE_CROSSED_COMPLY_AUTHORIZATION. Fresh normal
usage below90 is separately required. Authorization is not created here.
Direct worker entry checks authorization and a clean untouched namespace
before emitting WORKER_CLAIM; starting twice is forbidden.

Retain exact parent storage377,958,704bytes, not an enlarged allowance:
144*248320*4 raw array bytes, zlib bound993595 per array
(143077680 total),144MiB rows,64MiB updates,16MiB auxiliary.
Preload free guard512MiB. Independent final category/total inventory.
If actual accounting is incompatible, report it and DO NOT launch or extend.

Focused new-path fake tests plus relevant parent safety cases and Ruff only.
Test factorial selection/letter invariance, both flip directions and UNTESTED,
negative RHS/COMPLY signs, own-original norms/common w, bounds, stops,
no retry, raw-array/geometry/current-final failures, finite failures/tampering,
source hashes and source/lock launch contract. Independent verifier rebuilds
selected text/mapping/schedule without relying on runtime selection and
recomputes raw numerics, optimizer, geometry and deterministic replay.
The preparation handoff reports only tests actually run, scoped commits,
remaining work and ZERO tokenizer/model loads,0 real forwards/0 derivatives.

No gate, adaptive controller, transfer/held-out/generalization/collateral or
natural self-preservation claim. Even future8/8 is training fit on TWO situations.
Do not change model/settings, reset usage, buy credits, push, change dependencies
or repair permissions/security/Git. Usage checked before every substantial batch;
unavailable or>=90 stops new work and notifies. Preserve unrelated files.
Report preparation and STOP; actual construction waits for new authorization.

