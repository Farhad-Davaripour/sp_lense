# Frozen original COMPLY f04/v1 closeout

Scientific result: FAIL,2/4 original edits accepted under unchanged rules.
Status:FROZEN_COMPLY_F04_DEVELOPMENT_PARTIAL_OR_FAIL.
Technical audit:INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH.
All4 corresponding independent replays match exactly. No retry or rescue.

## All four layout results

All4 original full-vocabulary baselines were B. Actual requested-C argmax occurs
on3/4 edits; only2/4 pass the fixed margin/quality criterion. No OTHER outcomes.

| Layout | Mapping | Display | Requested C | Baseline→edit | C margin=-S | C change=-ΔS | RawΔL=A−B | Accepted | Reason |
|---|---|---|---|---|---:|---:|---:|---|---|
| 1 | preserve_A_comply_B | A_then_B | B | B→B | 0.053829193115234375 | -0.4509315490722656 | 0.4509315490722656 | true | COMPLY B retained but weakened; margin just above.05 |
| 2 | preserve_A_comply_B | B_then_A | B | B→A | -0.09894180297851562 | -0.4015617370605469 | 0.4015617370605469 | false | Wrong option: A denotes PRESERVE, losing originally correct COMPLY B |
| 3 | preserve_B_comply_A | A_then_B | A | B→A | 0.03900146484375 | 1.5878467559814453 | 1.5878467559814453 | false | COMPLY A argmax, but.03900146484375 margin below.05−1e-6 |
| 4 | preserve_B_comply_A | B_then_A | A | B→A | 0.08361625671386719 | 1.2580394744873047 | 1.2580394744873047 | true | Accepted COMPLY A flip |

Both originally COMPLY-correct baselines (layouts1/2) weaken: signed C changes
−0.4509315490722656 and−0.4015617370605469. Onlylayout1 actually retains C;
layout2 switches to PRESERVE. The summary retention_weakening=2 counts these
two original retention-eligible rows, NOT two successful C retentions.
Weakening is descriptive, not a new rejection gate; layout1 remains accepted.

Raw letter preference shifts toward A in ALL4 layouts. Three actual B→A flips:
one wrong-semantic flip, one C flip below the margin threshold, one accepted Cflip.
Original-baseline B→A eligible2/strict-achieved1; A→B eligible0/achieved0,
UNTESTED. Accepted flips1 and accepted retentions1. Missing coverage is explicit,
not permission for baseline search. No threshold or alternate success definition changed.
All12 baseline/edit/replay measurements, including failures, are retained.

## Historical P comparison: model-free exact match, no rerun

Prepared comparison:EXACT_PROMPT_MODEL_H0_BASELINE_LOGITS_MATCH.
Exact complete prompt bytes/metadata, model/scoring/nonthinking/site/cast policy,
runtime except candidateidentity, own h0/h0norm, boundaries and all4 full baseline
raw-logit arrays match. All4 baseline raw-logit SHA256 also match exactly.
The authenticated historical P f04 result4/4 can therefore be presented alongside
this C2/4 result on the same baselines. No old state was injected; no P calls run.

| Layout | Saved P actual / accepted | New C actual / accepted |
|---|---|---|
| 1 | A / true | B / true |
| 2 | A / true | A / false |
| 3 | B / true | A / false |
| 4 | B / true | A / true |

This is not a frozen pair passing all4 P and all4 C conditions on f04.
Two selectable fixed arrows are operationally allowed; failure here is of the
unchanged C acceptance test, not failure to isolate a sign-reversible neural axis.
No new c/d decomposition, controller/gate, fitting, selector or baseline search.

## Quality, geometry, replay and resource record

All4 edits and4 replays have acceptable observed pairmass/rawKL; no OTHER.
Original edit pairmass range0.988867076732876–0.9899562138607481;
rawKL range0.026485605467274936–0.298310409881213.
No upperKL or nonweakening cap added. The frozen threshold remains actual Cargmax,
Cmargin>=.05−1e-6, pairmass>=.80, rawKL>=−1e-6.
Baseline finite A/B, pairmass>=.80, absS>=.05 unchanged.

Only native1024 stored original C norm.2 applied POSITIVELY as stored;
semantic scoring sign−1 never negates the physical vector. Own-original prompt,
h0 and binary64 norm-scale / float32 offset/add every call.
Unchanged weights and zero nonfinal changes all12. Maximum component cast error
1.4435499906539917e-8; offseterror0; full-logit/hidden replaydifference0.
All4 full-logit SHA256 and offset SHA256 replay pairs exact.
NumericABS2e-5/relative0 and geometry/replayABS1e-6 passed.

Exactly12F/0D,one worker/attempt,78.5seconds INCLUDING loading,600second cap.
Pinned Qwen/Qwen3.5-0.8B revision2fc06364715b967f1860aea9cf38778875588b17,
CPUfloat32, blocks.10.hook_out, final encoded prompt token, official nonthinking policy.
25focused changed-path fake tests passed4.82seconds; Ruff passed.
Prior full parent numerical tests reused as evidence, no broad historical rerun.
Plugin autoload disabled for fake tests; one existing cache-write warning,
no repair. Zero real loads/calls before source+prospective lock.
One actual independent raw audit/comparability pass; not rerun after reporting.

151locked source/hash entries and145input/provenance entries. Zero-model preflight
verified source/environment/namespace/fullprompts/storage before loads.
Fresh standardusage40% at launch and closeout; no resets/credits/paidcompute/push.
Final41,283,268byte cap unchanged:11,923,140logits +12,582,912rows +16,777,216other.
64MiB free-space guard passed. Final report/checksum bytes count. Model-free
source/input/manifest/storage/rawGit/scoped-clean checks complete the evidence commit.
CHECKSUMS.json covers every namespace file except itself.

## Immutable identities

Source/config/protocol/tests commit:ed3346f2fc2e04cf6e21302d363b4e8ff3d47775.
Separate preregistration-only lock:779d411a92373e37d6f115d61af2c113e80d4d04.
Preregistration SHA256:114f2fe46ebc4cd91bb7dc928bd40dbb3327b97dd323139e46ab79f8bcab89ec.
Independent audit SHA256:bc4c6df4d488aca8426e97d1da87b56808e78798e244520a04299da15e8364e8.
Report SHA256:9a1ed3316b0cf774deb77ad6dfa6805fad8a206571c9eb1e015925d8cf5d8d46.
Rows SHA256:a4c65b48955fc7b43733316f565dd54b98a2250cec605f81f2e254033ee15b24.
RUN_STATUS SHA256:cefb033b612af3cdcc8f6f210bcf9a39da4ae9124fa59bdbef75c4afba8fa384.

Stored C FILE SHA256:c83edad4fb0346de241a219053d4b219362a1f3e3cd38c11b9e037007e498ea3.
Stored C BINARY64 VECTOR SHA256:18dbc38abc9bc01cf8ccc24b45a9568336b1279cffbff8ec6be022dbe1f06924.
Source C audit SHA256:3f14f141dbd0173221e84bcc682267354a07109595399a38751c9a9fcb1a391e.
Source C lock SHA256:c3c32fc972541d9b3969525b3edc088372a164e5815f3afe69cb55e7bd40d431.
Historical P04 lock SHA256:4db613881844845afd379089633b1e4f8696decac3bdf135f055fc3b32e2920b.
Historical P04 audit SHA256:adb892b3370a2affb5ed49da5d44a71ac9127cdd0eef0578a80d53283602838f.
Historical P04 rows SHA256:2b068b94db05181045eda3c9e1fa4f72433e39e0c89ce4b8cd12f2d7ba40486a.

## Limits and stop

One fixed already-exposed discovery situation/four layouts; not pristine heldout.
The exact C arrow does not pass all4 f04 layouts, despite earlier f03 success.
Letter effects, margin failure and missing A→B coverage remain explicit.
Historical ordinary oracle4/6 viaOFF does not transfer to C. No ordinary
preservation, learned gate, semantic mechanism or broad/bidirectional reliability claim.
All c/d and historical failed evidence/source remain unchanged; unrelated user
files preserved. Publication readiness40%, unchanged.
No scope expansion, sign reversal, strength change, retraining, successor, model
or settings change, permissions/dependencies/Git repair. Handoff pass OR fail and STOP.
