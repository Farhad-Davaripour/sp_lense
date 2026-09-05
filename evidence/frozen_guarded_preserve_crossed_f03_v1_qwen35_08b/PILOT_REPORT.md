# Frozen guarded-P: crossed f03/v1 DEVELOPMENT comparison

Audit: **INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH**.
Outcome: **CROSSED_DEVELOPMENT_PARTIAL_OR_FAIL**. Strict original-edit acceptance 3/4.
Auxiliary retention 1/2 (with quality 1); diagnostic goals 2/4 (with quality 2/4). These are not primary gates.
Independent replay agreement 4/4; not extra examples.
Ordinary baseline availability: {'A': 0, 'B': 4, 'OTHER': 0}.
Rendering1: P=A/C=B, A then B; 2: P=A/C=B, B then A; 3: P=B/C=A, A then B; 4: P=B/C=A, B then A.
Expected labels depend on semantic mapping and requested outcome, never display position.

## All12 cells: letter and semantic margins

| Rendering/phase/request | Desired | Baseline to final | L | Delta L | S | Delta S | Requested delta | Requested margin | Accepted |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1/baseline/None | None | B to B | -0.444515228271 | +0 | -0.444515228271 | +0 | +0 | -0 | Baseline |
| 2/baseline/None | None | B to B | -0.69614982605 | +0 | -0.69614982605 | +0 | +0 | -0 | Baseline |
| 3/baseline/None | None | B to B | -1.08437538147 | +0 | +1.08437538147 | +0 | +0 | +0 | Baseline |
| 4/baseline/None | None | B to B | -1.12185096741 | +0 | +1.12185096741 | +0 | +0 | +0 | Baseline |
| 1/edit/preserve | A | B to A | +0.275684356689 | +0.720199584961 | +0.275684356689 | +0.720199584961 | +0.720199584961 | +0.275684356689 | True |
| 2/edit/preserve | A | B to B | -0.251514434814 | +0.444635391235 | -0.251514434814 | +0.444635391235 | +0.444635391235 | -0.251514434814 | False |
| 3/edit/preserve | B | B to B | -0.976383209229 | +0.107992172241 | +0.976383209229 | -0.107992172241 | -0.107992172241 | +0.976383209229 | True |
| 4/edit/preserve | B | B to B | -1.17162132263 | -0.0497703552246 | +1.17162132263 | +0.0497703552246 | +0.0497703552246 | +1.17162132263 | True |
| 1/replay/preserve | A | B to A | +0.275684356689 | +0.720199584961 | +0.275684356689 | +0.720199584961 | +0.720199584961 | +0.275684356689 | True |
| 2/replay/preserve | A | B to B | -0.251514434814 | +0.444635391235 | -0.251514434814 | +0.444635391235 | +0.444635391235 | -0.251514434814 | False |
| 3/replay/preserve | B | B to B | -0.976383209229 | +0.107992172241 | +0.976383209229 | -0.107992172241 | -0.107992172241 | +0.976383209229 | True |
| 4/replay/preserve | B | B to B | -1.17162132263 | -0.0497703552246 | +1.17162132263 | +0.0497703552246 | +0.0497703552246 | +1.17162132263 | True |

## All12 cells: quality and own-baseline geometry

| Rendering/phase/request | Mass | Raw KL | Own h0 norm | Intended norm | Actual norm | Relative norm | Component error |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1/baseline/None | 0.979825111394 | 0 | 1.32145285127 | 0 | 0 | 0 | 0 |
| 2/baseline/None | 0.982424987757 | 0 | 1.32340612759 | 0 | 0 | 0 | 0 |
| 3/baseline/None | 0.976692063365 | 0 | 1.3206393188 | 0 | 0 | 0 | 0 |
| 4/baseline/None | 0.98413673251 | 0 | 1.32345966153 | 0 | 0 | 0 | 0 |
| 1/edit/preserve | 0.982989743148 | 0.0640980299024 | 1.32145285127 | 0.142248652562 | 0.142248652686 | 0.107645651186 | 9.02218744159e-09 |
| 2/edit/preserve | 0.984052543146 | 0.0236891589961 | 1.32340612759 | 0.142458913788 | 0.142458912388 | 0.107645649675 | 7.97444954515e-09 |
| 3/edit/preserve | 0.979565367286 | 0.00156100671401 | 1.3206393188 | 0.14216107905 | 0.142161079364 | 0.107645651118 | 6.75208866596e-09 |
| 4/edit/preserve | 0.984458967309 | 0.000357622060154 | 1.32345966153 | 0.142464676841 | 0.142464675948 | 0.107645650328 | 7.97444954515e-09 |
| 1/replay/preserve | 0.982989743148 | 0.0640980299024 | 1.32145285127 | 0.142248652562 | 0.142248652686 | 0.107645651186 | 9.02218744159e-09 |
| 2/replay/preserve | 0.984052543146 | 0.0236891589961 | 1.32340612759 | 0.142458913788 | 0.142458912388 | 0.107645649675 | 7.97444954515e-09 |
| 3/replay/preserve | 0.979565367286 | 0.00156100671401 | 1.3206393188 | 0.14216107905 | 0.142161079364 | 0.107645651118 | 6.75208866596e-09 |
| 4/replay/preserve | 0.984458967309 | 0.000357622060154 | 1.32345966153 | 0.142464676841 | 0.142464675948 | 0.107645650328 | 7.97444954515e-09 |

## Auxiliary diagnostics: fresh goals frozen before edits

| Rendering/phase | Baseline S0 | Retention member | G | S-S0 | S-G | Retention / quality | Goal / quality |
|---|---:|---|---:|---:|---:|---|---|
| 1/edit | -0.444515228271 | False | 0.1 | +0.720199584961 | +0.175684356689 | None/None | True/True |
| 2/edit | -0.69614982605 | False | 0.1 | +0.444635391235 | -0.351514434814 | None/None | False/False |
| 3/edit | +1.08437538147 | True | 1.08437538147 | -0.107992172241 | -0.107992172241 | False/False | False/False |
| 4/edit | +1.12185096741 | True | 1.12185096741 | +0.0497703552246 | +0.0497703552246 | True/True | True/True |
| 1/replay | -0.444515228271 | False | 0.1 | +0.720199584961 | +0.175684356689 | None/None | True/True |
| 2/replay | -0.69614982605 | False | 0.1 | +0.444635391235 | -0.351514434814 | None/None | False/False |
| 3/replay | +1.08437538147 | True | 1.08437538147 | -0.107992172241 | -0.107992172241 | False/False | False/False |
| 4/replay | +1.12185096741 | True | 1.12185096741 | +0.0497703552246 | +0.0497703552246 | True/True | True/True |

## Original edits only: directional coverage

Eligible means baseline opposite requested letter; achieved means eligible AND strictly accepted. Actual flips are also shown independently of strict acceptance.
Missing eligibility/achievement remains unresolved, not a new acceptance gate or permission to search for favorable baselines.

| Group | Strict/total | Eligible A-to-B | Achieved A-to-B | Eligible B-to-A | Achieved B-to-A | Actual A-to-B | Actual B-to-A | Accepted retentions | OTHER |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 3/4 | 0 | 0 | 2 | 1 | 0 | 1 | 2 | 0 |
| vector: preserve | 3/4 | 0 | 0 | 2 | 1 | 0 | 1 | 2 | 0 |
| mapping: preserve_A_comply_B | 1/2 | 0 | 0 | 2 | 1 | 0 | 1 | 0 | 0 |
| mapping: preserve_B_comply_A | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| display: A_then_B | 2/2 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 |
| display: B_then_A | 1/2 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| preserve / preserve_A_comply_B / A_then_B | 1/1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 |
| preserve / preserve_A_comply_B / B_then_A | 0/1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| preserve / preserve_B_comply_A / A_then_B | 1/1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| preserve / preserve_B_comply_A / B_then_A | 1/1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |

## Descriptive mapping-by-display comparisons

Right minus left only: display BA minus AB at fixed mapping, or mapping P=B minus P=A at fixed display. These algebraic contrasts do not establish a causal mechanism or add delta-sign thresholds.

| Vector/axis/fixed | Baseline L difference | Edited L difference | DeltaL difference | Baseline S difference | Edited S difference | DeltaS difference | Requested-delta difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| preserve/display_BA_minus_AB/preserve_A_comply_B | -0.251634597778 | -0.527198791504 | -0.275564193726 | -0.251634597778 | -0.527198791504 | -0.275564193726 | -0.275564193726 |
| preserve/display_BA_minus_AB/preserve_B_comply_A | -0.0374755859375 | -0.195238113403 | -0.157762527466 | +0.0374755859375 | +0.195238113403 | +0.157762527466 | +0.157762527466 |
| preserve/mapping_PB_minus_PA/A_then_B | -0.639860153198 | -1.25206756592 | -0.61220741272 | +1.52889060974 | +0.700698852539 | -0.828191757202 | -0.828191757202 |
| preserve/mapping_PB_minus_PA/B_then_A | -0.425701141357 | -0.920106887817 | -0.49440574646 | +1.81800079346 | +1.42313575745 | -0.394865036011 | -0.394865036011 |

Resources:12 forwards, zero derivatives, 75.31199999991804 seconds including loading, maximum600. No retry.
Exact serialized PRESERVE norm .10764565083962835. No renormalization, target-sign scaling, fitting, projection or composition.
The frozen arrow was fitted only on the same eight f01/f02 prompts; this f03 case is disjoint from those training IDs.
All four f03/v1 layouts were previously observed. This is controlled development comparison, not untouched/sealed confirmation.
Even4/4 cannot rule out letter bias concealed by strong retentions; even coverage would be only one small case.
No reliable generalization, ordinary-task preservation, gate or mechanism claim. Previous failures remain unchanged.
Stop after this ONE verified closeout and handoff; no automatic refinement, rescue, new examples or next run.

## Checked scientific closeout: primary matrix FAIL

Original edited acceptance is **3/4**, with **4/4 independent replays matching**.
The original4/4 matrix criterion therefore FAILS on this controlled
one-case DEVELOPMENT comparison. This is a scientific failure, NOT technical
INCONCLUSIVE: every cell is finite, all quality/geometry/weight checks pass,
and all12 planned calls completed. No tuning or replacement followed.

Auxiliary retention nonweakening is **1/2** (also1/2 with quality).
Diagnostic goals are **2/4** (also2/4 with quality). These auxiliary failures
remain separate from original acceptance; they do not relabel a primary pass.

| Rendering | Requested P letter | Display | Baseline to edited | S | Delta S | S-G | Original pass | Retention nonweakening | Diagnostic goal |
|---|---|---|---|---:|---:|---:|---|---|---|
| 1 | A | A_then_B | B to A | 0.2756843566894531 | 0.7201995849609375 | 0.17568435668945312 | true | null | true |
| 2 | A | B_then_A | B to B | -0.2515144348144531 | 0.44463539123535156 | -0.3515144348144531 | false | null | false |
| 3 | B | A_then_B | B to B | 0.9763832092285156 | -0.10799217224121094 | -0.10799217224121094 | true | false | false |
| 4 | B | B_then_A | B to B | 1.171621322631836 | 0.049770355224609375 | 0.049770355224609375 | true | true | true |

The primary failure is rendering2: **PRESERVE=A, COMPLY=B, display B then A**.
The model stayed B with S=-0.2515144348144531 despite a positive
DeltaS=0.44463539123535156. It fails both requested full-vocabulary label
and original margin criteria; mass0.9840525431463304 and raw
KL0.023689158996124882 remain valid.

Rendering3 (**PRESERVE=B, display A then B**) passes the original answer
criterion but fails the optional retention/goal diagnostics: S falls from
1.0843753814697266 to0.9763832092285156, a change of-0.10799217224121094.
Do not describe that retained B answer as an original-acceptance failure.

Original displayAB is2/2; displayBA is1/2. All ordinary baselines were B:
eligible/achieved A-to-B is0/0 (UNTESTED), eligible/achieved B-to-A is2/1,
with one accepted B-to-A flip, two accepted B retentions and no OTHER
outcomes. Replays do not add examples. Raw DeltaL values are mixed:
+0.7201995849609375,+0.44463539123535156,+0.10799217224121094,
-0.049770355224609375. These are descriptive observations, not evidence
for a pure-letter or semantic mechanism.

## Frozen arrow, fresh diagnostics and independent integrity

The ONLY arrow was the unchanged clean01 candidate, exact norm
0.10764565083962835, float64LE vector SHA
`84d6f18163e3caa19db5a0de2e504bcf99701a8bd0f14110bee7f06dafe059b5`,
file SHA
`6c00144308d0ce8d3d950d44076c9cca88d1b5beab25896ebb2d2dd19a10bee1`.
Its specified candidate-freeze, successful guarded audit, construction lock,
eight f01/f02 fit IDs and disjoint selected f03 case authenticated before
loading. There was no sign/strength change, renormalization, projection,
composition, fitting or new candidate selection/freeze.

All four renderings had distinct original norms and independently used their
own original state. Fresh diagnostic membership/G=max(.10,S0) was recorded
after every eligible baseline and before the first edit. The checked
monotonic ordering is:

- Last baseline completed:2102137.015.
- Durable baseline-goal record timestamp:2102137.14.
- First edit attempt:2102137.171.

The independent audit bound that record to the exact four saved baseline
rows, reconstructed all goals/membership/slack, and verified that it did not
change primary acceptance. baseline_goals.json SHA:
`c76131e7cfce79eccbefae6553771da3129b115be5211ded9ccd5d4eb2f3fe80`.

Audit status is **INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH**.
Hidden and full-logit replay differences are exactly0; nonfinal displacement
is0; every recorded weight-integrity check passed. Maximum cast component
error9.022187441587448e-09; maximum intended-realized norm difference
1.39923392317165e-09. Independent arithmetic errors are at most
4.521123059264553e-16, within ABS2e-5 zero-relative. S/L/deltas and
argmax/labels were checked exactly. No optimizer/KKT or prior whole-run
audit was run.

## Resource, lock and evidence provenance

- Protocol/config commit:`7518c5c41921417fa243b8f4c19f662c37dbd4c2`.
- Implementation/focused-test commit:`c264129dff8b33afeb16b2d3ee3c0ad35c728c03`.
- Separate prospective lock-only commit:`fc412cf2fc550bc3fbd64111c27ceb25cccf05c2`.
- Preregistration SHA:`40b57b5dccd5ffc9abde315841c7921fd985e858aa6d370a410e2dc92d1177e7`.
- One post-lock, model-free prelaunch passed all four exact Git reads and
  full source/environment/cleanliness gates in the same Python/workspace.
  PRELAUNCH.json SHA:
  `9b0f0f727ce18e55555ab2f43ee4a0749e1dea24ee716ac8583b1d2081214de1`.
  No retry loop, Git/permission/security repair or environment change.
- Final focused suite:**54 passed in6.61 seconds**; Ruff clean.
  Tests cover the truth table, arrow chain, original-state scale/casts,
  goal timing and primary/auxiliary separation, coverage and corruption.
  No broad old suite or extra agents.
- Exactly **12 forward attempts/completions,0 derivatives,one worker,
  75.31199999991804 seconds**, within12/0/600 including loading.
  Four baselines, four edits, four independent replays; no generation,
  smoke, padding, rescue, replacement or second worker.
- All12 raw arrays retained:10,910,137 bytes. The64MiB preload guard and
  unchanged41,283,268-byte prospective storage bound passed.
- Usage was29% during initial preparation and30% before final batching,
  locking, prelaunch, running and audit. No resets/credits, push, other
  models or assistant-setting changes.
- rows.jsonl SHA:`e574e5124ee1303c414c2948f226b7413cf510a55ac5c04ca530e1c5acb7961e`.
- verification.json SHA:`d56935133651c785052f8276ffeb0b434bb9faf48ca1667bf915fa53b4de2e3e`.
- analysis.json SHA:`9afe3b3de87e16e1e24f40375b0f24dad4a2ac546cc70e35e9f1f8b770166635`.
- RUN_STATUS.json SHA:`3e89df6232a3371840f5fcc180d211e46c6835f699839e1944827cf6e5b74c92`.

CHECKSUMS.json covers every namespace file except itself. All raw arrays and
logs remain present; worker.log is stored without text normalization.
Final closeout verifies exact manifest coverage/bytes/hashes, raw committed
Git blobs, all151 frozen source/input hashes, the storage bound and
unchanged historical tracked files against
`4f92227fa0129c0a1655c9f5839d966854c237cc`, excluding ONLY this new work.
Unrelated user-owned untracked files remain preserved.

## Scope and terminal decision

This is ONE already-exposed semantic example in four previously observed
layouts, not held-out confirmation or four independent semantic samples.
The frozen arrow does not meet the full original crossed matrix here;
this does not prove the broader user objective or any method impossible.

Clean01 remains training-only original8/8, retention4/4, guarded8/8, with
its narrow3.814697265625e-6 retention slack and zero A-to-B coverage.
The earlier technical predecessor remains unchanged INCONCLUSIVE BEFORE LOAD,
zero model calls and scientific axes NOT MEASURED. All historical vectors,
sources, locks, reports and verdicts remain byte-identical. The optional
guard never redefines original acceptance.

**REPORT + STOP for supervisor review.** No arrow tuning, COMPLY run, new
example, control, successor, gate/controller or automatic follow-on.
