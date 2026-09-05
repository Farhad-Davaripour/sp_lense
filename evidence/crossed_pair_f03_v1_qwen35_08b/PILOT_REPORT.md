# Crossed frozen pair: f03/v1 development comparison

Audit: **INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH**.
Outcome: **CROSSED_DEVELOPMENT_PARTIAL_OR_FAIL**. Strict original-edit acceptance 6/8.
PRESERVE 3/4; COMPLY 3/4.
Independent replay agreement 8/8; not extra examples.
Ordinary baseline availability: {'A': 0, 'B': 4, 'OTHER': 0}.
Rendering1: P=A/C=B, A then B; 2: P=A/C=B, B then A; 3: P=B/C=A, A then B; 4: P=B/C=A, B then A.
Expected labels depend on semantic mapping and requested outcome, never display position.

## All20 cells: letter and semantic margins

| Rendering/phase/request | Desired | Baseline to final | L | Delta L | S | Delta S | Requested delta | Requested margin | Accepted |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1/baseline/None | None | B to B | -0.444515228271 | +0 | -0.444515228271 | +0 | +0 | -0 | Baseline |
| 2/baseline/None | None | B to B | -0.69614982605 | +0 | -0.69614982605 | +0 | +0 | -0 | Baseline |
| 3/baseline/None | None | B to B | -1.08437538147 | +0 | +1.08437538147 | +0 | +0 | +0 | Baseline |
| 4/baseline/None | None | B to B | -1.12185096741 | +0 | +1.12185096741 | +0 | +0 | +0 | Baseline |
| 1/edit/preserve | A | B to A | +0.258769989014 | +0.703285217285 | +0.258769989014 | +0.703285217285 | +0.703285217285 | +0.258769989014 | True |
| 1/edit/comply | B | B to B | -0.122400283813 | +0.322114944458 | -0.122400283813 | +0.322114944458 | -0.322114944458 | +0.122400283813 | True |
| 2/edit/preserve | A | B to B | -0.318605422974 | +0.377544403076 | -0.318605422974 | +0.377544403076 | +0.377544403076 | -0.318605422974 | False |
| 2/edit/comply | B | B to B | -0.529918670654 | +0.166231155396 | -0.529918670654 | +0.166231155396 | -0.166231155396 | +0.529918670654 | True |
| 3/edit/preserve | B | B to B | -0.416233062744 | +0.668142318726 | +0.416233062744 | -0.668142318726 | -0.668142318726 | +0.416233062744 | True |
| 3/edit/comply | A | B to A | +0.336486816406 | +1.42086219788 | -0.336486816406 | -1.42086219788 | +1.42086219788 | +0.336486816406 | True |
| 4/edit/preserve | B | B to B | -0.881551742554 | +0.240299224854 | +0.881551742554 | -0.240299224854 | -0.240299224854 | +0.881551742554 | True |
| 4/edit/comply | A | B to B | -0.340353012085 | +0.781497955322 | +0.340353012085 | -0.781497955322 | +0.781497955322 | -0.340353012085 | False |
| 1/replay/preserve | A | B to A | +0.258769989014 | +0.703285217285 | +0.258769989014 | +0.703285217285 | +0.703285217285 | +0.258769989014 | True |
| 1/replay/comply | B | B to B | -0.122400283813 | +0.322114944458 | -0.122400283813 | +0.322114944458 | -0.322114944458 | +0.122400283813 | True |
| 2/replay/preserve | A | B to B | -0.318605422974 | +0.377544403076 | -0.318605422974 | +0.377544403076 | +0.377544403076 | -0.318605422974 | False |
| 2/replay/comply | B | B to B | -0.529918670654 | +0.166231155396 | -0.529918670654 | +0.166231155396 | -0.166231155396 | +0.529918670654 | True |
| 3/replay/preserve | B | B to B | -0.416233062744 | +0.668142318726 | +0.416233062744 | -0.668142318726 | -0.668142318726 | +0.416233062744 | True |
| 3/replay/comply | A | B to A | +0.336486816406 | +1.42086219788 | -0.336486816406 | -1.42086219788 | +1.42086219788 | +0.336486816406 | True |
| 4/replay/preserve | B | B to B | -0.881551742554 | +0.240299224854 | +0.881551742554 | -0.240299224854 | -0.240299224854 | +0.881551742554 | True |
| 4/replay/comply | A | B to B | -0.340353012085 | +0.781497955322 | +0.340353012085 | -0.781497955322 | +0.781497955322 | -0.340353012085 | False |

## All20 cells: quality and own-baseline geometry

| Rendering/phase/request | Mass | Raw KL | Own h0 norm | Intended norm | Actual norm | Relative norm | Component error |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1/baseline/None | 0.979825111394 | 0 | 1.32145285127 | 0 | 0 | 0 | 0 |
| 2/baseline/None | 0.982424987757 | 0 | 1.32340612759 | 0 | 0 | 0 | 0 |
| 3/baseline/None | 0.976692063365 | 0 | 1.3206393188 | 0 | 0 | 0 | 0 |
| 4/baseline/None | 0.98413673251 | 0 | 1.32345966153 | 0 | 0 | 0 | 0 |
| 1/edit/preserve | 0.990536752511 | 0.066685927397 | 1.32145285127 | 0.11243004891 | 0.112430048703 | 0.0850806357524 | 6.98491930962e-09 |
| 1/edit/comply | 0.988479086361 | 0.0320797681511 | 1.32145285127 | 0.264290569863 | 0.264290569885 | 0.19999999972 | 9.31322574615e-09 |
| 2/edit/preserve | 0.990189784691 | 0.020074944947 | 1.32340612759 | 0.112596235172 | 0.112596235791 | 0.085080636581 | 1.22236087918e-08 |
| 2/edit/comply | 0.991637189724 | 0.00968832778668 | 1.32340612759 | 0.264681225782 | 0.264681225637 | 0.20000000009 | 6.98491930962e-09 |
| 3/edit/preserve | 0.989900363431 | 0.0564670111735 | 1.3206393188 | 0.112360833064 | 0.11236083319 | 0.0850806360153 | 1.18743628263e-08 |
| 3/edit/comply | 0.989860455801 | 0.259121535921 | 1.3206393188 | 0.264127864128 | 0.264127864569 | 0.200000000613 | 6.98491930962e-09 |
| 4/edit/preserve | 0.9904365409 | 0.00815855146292 | 1.32345966153 | 0.112600789813 | 0.112600789352 | 0.0850806357193 | 6.75208866596e-09 |
| 4/edit/comply | 0.99169888664 | 0.0751940561431 | 1.32345966153 | 0.264691932185 | 0.264691932632 | 0.200000000246 | 1.49011611938e-08 |
| 1/replay/preserve | 0.990536752511 | 0.066685927397 | 1.32145285127 | 0.11243004891 | 0.112430048703 | 0.0850806357524 | 6.98491930962e-09 |
| 1/replay/comply | 0.988479086361 | 0.0320797681511 | 1.32145285127 | 0.264290569863 | 0.264290569885 | 0.19999999972 | 9.31322574615e-09 |
| 2/replay/preserve | 0.990189784691 | 0.020074944947 | 1.32340612759 | 0.112596235172 | 0.112596235791 | 0.085080636581 | 1.22236087918e-08 |
| 2/replay/comply | 0.991637189724 | 0.00968832778668 | 1.32340612759 | 0.264681225782 | 0.264681225637 | 0.20000000009 | 6.98491930962e-09 |
| 3/replay/preserve | 0.989900363431 | 0.0564670111735 | 1.3206393188 | 0.112360833064 | 0.11236083319 | 0.0850806360153 | 1.18743628263e-08 |
| 3/replay/comply | 0.989860455801 | 0.259121535921 | 1.3206393188 | 0.264127864128 | 0.264127864569 | 0.200000000613 | 6.98491930962e-09 |
| 4/replay/preserve | 0.9904365409 | 0.00815855146292 | 1.32345966153 | 0.112600789813 | 0.112600789352 | 0.0850806357193 | 6.75208866596e-09 |
| 4/replay/comply | 0.99169888664 | 0.0751940561431 | 1.32345966153 | 0.264691932185 | 0.264691932632 | 0.200000000246 | 1.49011611938e-08 |

## Original edits only: directional coverage

Eligible means baseline opposite requested letter; achieved means eligible AND strictly accepted. Actual flips are also shown independently of strict acceptance.
Missing eligibility/achievement remains unresolved, not a new acceptance gate or permission to search for favorable baselines.

| Group | Strict/total | Eligible A-to-B | Achieved A-to-B | Eligible B-to-A | Achieved B-to-A | Actual A-to-B | Actual B-to-A | Accepted retentions | OTHER |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 6/8 | 0 | 0 | 4 | 2 | 0 | 2 | 4 | 0 |
| vector: preserve | 3/4 | 0 | 0 | 2 | 1 | 0 | 1 | 2 | 0 |
| vector: comply | 3/4 | 0 | 0 | 2 | 1 | 0 | 1 | 2 | 0 |
| mapping: preserve_A_comply_B | 3/4 | 0 | 0 | 2 | 1 | 0 | 1 | 2 | 0 |
| mapping: preserve_B_comply_A | 3/4 | 0 | 0 | 2 | 1 | 0 | 1 | 2 | 0 |
| display: A_then_B | 4/4 | 0 | 0 | 2 | 2 | 0 | 2 | 2 | 0 |
| display: B_then_A | 2/4 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 |
| preserve / preserve_A_comply_B / A_then_B | 1/1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 |
| comply / preserve_A_comply_B / A_then_B | 1/1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| preserve / preserve_A_comply_B / B_then_A | 0/1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| comply / preserve_A_comply_B / B_then_A | 1/1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| preserve / preserve_B_comply_A / A_then_B | 1/1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| comply / preserve_B_comply_A / A_then_B | 1/1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 |
| preserve / preserve_B_comply_A / B_then_A | 1/1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| comply / preserve_B_comply_A / B_then_A | 0/1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

## Descriptive mapping-by-display comparisons

Right minus left only: display BA minus AB at fixed mapping, or mapping P=B minus P=A at fixed display. These algebraic contrasts do not establish a causal mechanism or add delta-sign thresholds.

| Vector/axis/fixed | Baseline L difference | Edited L difference | DeltaL difference | Baseline S difference | Edited S difference | DeltaS difference | Requested-delta difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| preserve/display_BA_minus_AB/preserve_A_comply_B | -0.251634597778 | -0.577375411987 | -0.325740814209 | -0.251634597778 | -0.577375411987 | -0.325740814209 | -0.325740814209 |
| preserve/display_BA_minus_AB/preserve_B_comply_A | -0.0374755859375 | -0.46531867981 | -0.427843093872 | +0.0374755859375 | +0.46531867981 | +0.427843093872 | +0.427843093872 |
| preserve/mapping_PB_minus_PA/A_then_B | -0.639860153198 | -0.675003051758 | -0.0351428985596 | +1.52889060974 | +0.15746307373 | -1.37142753601 | -1.37142753601 |
| preserve/mapping_PB_minus_PA/B_then_A | -0.425701141357 | -0.56294631958 | -0.137245178223 | +1.81800079346 | +1.20015716553 | -0.61784362793 | -0.61784362793 |
| comply/display_BA_minus_AB/preserve_A_comply_B | -0.251634597778 | -0.407518386841 | -0.155883789062 | -0.251634597778 | -0.407518386841 | -0.155883789062 | +0.155883789062 |
| comply/display_BA_minus_AB/preserve_B_comply_A | -0.0374755859375 | -0.676839828491 | -0.639364242554 | +0.0374755859375 | +0.676839828491 | +0.639364242554 | -0.639364242554 |
| comply/mapping_PB_minus_PA/A_then_B | -0.639860153198 | +0.45888710022 | +1.09874725342 | +1.52889060974 | -0.214086532593 | -1.74297714233 | +1.74297714233 |
| comply/mapping_PB_minus_PA/B_then_A | -0.425701141357 | +0.189565658569 | +0.615266799927 | +1.81800079346 | +0.870271682739 | -0.947729110718 | +0.947729110718 |

Resources:20 forwards, zero derivatives, 84.39100000006147 seconds including loading, maximum600. No retry.
Exact serialized norms P=.08508063610056309; C=.20000000000000004. No renormalization, target-sign scaling, fitting, projection or composition.
Both vectors were fitted only on the same eight f01/f02 prompts; this f03 case is disjoint from those training IDs.
Original AB f03/v1 was already exposed in P development. This is controlled development comparison, not untouched/sealed confirmation.
Even8/8 cannot rule out letter bias concealed by strong retentions; even coverage would be only one small case.
No reliable generalization, ordinary-task preservation, gate or mechanism claim. Previous failures remain unchanged.
Stop after this ONE verified closeout and handoff; no automatic refinement, rescue, new examples or next run.

## Verified closeout

The strict crossed matrix failed: 6/8 original edits met the preregistered
criterion (PRESERVE 3/4, COMPLY 3/4). This is a finite scientific failure,
not an execution or audit failure. All 20 forwards completed in
84.39100000006147 seconds including model loading; zero derivatives,
no retry, no vector changes, and no additional examples were run.
All eight independent replays matched the original hidden states and full
logits exactly (maximum absolute differences both 0).

The original A-then-B display passed 4/4 edits; the B-then-A display passed
2/4. With P=A/C=B and B displayed before A, PRESERVE still returned B:
requested margin -0.3186054229736328, requested-signed delta
+0.3775444030761719. With P=B/C=A and B displayed before A, COMPLY still
returned B: requested margin -0.34035301208496094, requested-signed delta
+0.7814979553222656. Both were wrong-argmax/negative-margin failures,
not quality failures.

All four ordinary baselines selected B, regardless of semantic mapping or
display order. There were four eligible B-to-A requests, of which two
strictly succeeded, both in the A-then-B display; no A-to-B opportunity
existed. Each vector achieved one B-to-A flip and two accepted B-to-B
retentions. All four accepted retentions weakened the requested semantic
margin. There were no OTHER outcomes. Missing A-to-B coverage remains
unresolved; it was not a posthoc acceptance gate or permission to replace
the case.

Both vectors increased raw L=z_A-z_B in all eight original edits,
including when the requested outcome was B. For PRESERVE, Delta L in
rendering order was +0.7032852172851562, +0.3775444030761719,
+0.6681423187255859, +0.24029922485351562. For COMPLY it was
+0.3221149444580078, +0.1662311553955078, +1.4208621978759766,
+0.7814979553222656. The display reversal reduced this A-directed shift
for both vectors at both semantic mappings. These are descriptive results
on this fixed development case, consistent with an unresolved letter-bias
explanation and display-order sensitivity; they do not identify the
mechanism or prove pure letter-only behavior.

The independent raw-float32 audit reproduced all discrete outcomes and
exact letter/semantic margins and deltas. Maximum probability/mass/raw-KL
arithmetic discrepancy was 4.579669976578771e-16, within the unchanged
absolute 2e-5, zero-relative tolerance. Maximum cast/component discrepancy
was 1.4901161193847656e-8, within absolute 1e-6. All intended offsets
matched the locked cast rule, each edit used its own rendering's original
h0 norm, nonfinal changes were exactly zero, and all weight checks passed.
The authenticated P norm remains .08508063610056309 and the C norm remains
exactly .20000000000000004.

Focused validation before the source lock: Ruff passed and
tests/test_crossed_pair_probe.py passed all 44 tests in 4.49 seconds.
Pytest emitted only a cache-write permission warning; no tests failed.
No historical broad suite, old full audit, or extra agent was used.

Prospective commits:

- Protocol/config: f61b2ef3d896ab924ea9a30a8d10b385186ffdce.
- Implementation/tests: 536c9e95d032325cd51e077cd5a8b37456ba737b.
- Pre-model source/input/candidate/environment/20-cell lock:
  64b7a963f6373e2a45dad7531786ac2823f31c9f.

Authenticated artifact SHA256 values:

- preregistration.json:
  f530638da764a04e1d74e0e3e1ddb83f7f8e7b9b4c32cd9f69ed4596e1e4aa89.
- rows.jsonl:
  3092bb01f6713594995adb7cda173812f688d3552d13bacf34f292019855c0ec.
- verification.json:
  1781d09823b7b95cbab3975958fa0a01153e722385b08dcda4f6852e67ab97ee.
- Unchanged input preserve_vector.json:
  b71ea03c7a254f54f4d2064425f77143ee06bb1525153d58a92806627efec00f.
- Unchanged input comply_vector.json:
  2c3beb65e308dfe757d3c50a70dc504abf7495fdafb80472a0f706cfc7d9687f.

The audit inventoried 20 raw-logit arrays totaling 18,243,655 compressed
bytes, rows totaling 1,774,515 bytes, and 106,809 auxiliary bytes before
report/audit-file creation (20,124,979 bytes at that point), below the fixed
57,620,636-byte total bound. CHECKSUMS.json inventories the final namespace
except itself; its own hash is checked separately at evidence closeout.
Usage preflight was 26% before source/freeze/run and audit; no resets,
credits, pushes, or assistant-model setting changes were made.

Terminal interpretation: the fixed pair did not survive this crossed
development comparison. No reliable bidirectional semantic controller,
generalization, ordinary-task preservation, or gate-readiness claim is
supported. This is one already-exposed f03/v1 development case, not a
sealed confirmation. Prior evidence and failures remain unchanged. Stop
after the one scoped verified evidence commit and supervisor handoff.
