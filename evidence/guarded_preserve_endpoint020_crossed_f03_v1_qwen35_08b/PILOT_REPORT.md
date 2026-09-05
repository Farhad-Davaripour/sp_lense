# Derived guarded-P .20 endpoint: crossed f03/v1 DEVELOPMENT comparison

Audit: **INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH**.
Outcome: **CROSSED_MATRIX_ACCEPTED_ONLY**. Strict original-edit acceptance 4/4.
Auxiliary retention 1/2 (with quality 1); diagnostic goals 3/4 (with quality 3/4). These are not primary gates.
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
| 1/edit/preserve | A | B to A | +0.894556045532 | +1.3390712738 | +0.894556045532 | +1.3390712738 | +1.3390712738 | +0.894556045532 | True |
| 2/edit/preserve | A | B to A | +0.13850402832 | +0.83465385437 | +0.13850402832 | +0.83465385437 | +0.83465385437 | +0.13850402832 | True |
| 3/edit/preserve | B | B to B | -0.953186035156 | +0.131189346313 | +0.953186035156 | -0.131189346313 | -0.131189346313 | +0.953186035156 | True |
| 4/edit/preserve | B | B to B | -1.2477016449 | -0.12585067749 | +1.2477016449 | +0.12585067749 | +0.12585067749 | +1.2477016449 | True |
| 1/replay/preserve | A | B to A | +0.894556045532 | +1.3390712738 | +0.894556045532 | +1.3390712738 | +1.3390712738 | +0.894556045532 | True |
| 2/replay/preserve | A | B to A | +0.13850402832 | +0.83465385437 | +0.13850402832 | +0.83465385437 | +0.83465385437 | +0.13850402832 | True |
| 3/replay/preserve | B | B to B | -0.953186035156 | +0.131189346313 | +0.953186035156 | -0.131189346313 | -0.131189346313 | +0.953186035156 | True |
| 4/replay/preserve | B | B to B | -1.2477016449 | -0.12585067749 | +1.2477016449 | +0.12585067749 | +0.12585067749 | +1.2477016449 | True |

## All12 cells: quality and own-baseline geometry

| Rendering/phase/request | Mass | Raw KL | Own h0 norm | Intended norm | Actual norm | Relative norm | Component error |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1/baseline/None | 0.979825111394 | 0 | 1.32145285127 | 0 | 0 | 0 | 0 |
| 2/baseline/None | 0.982424987757 | 0 | 1.32340612759 | 0 | 0 | 0 | 0 |
| 3/baseline/None | 0.976692063365 | 0 | 1.3206393188 | 0 | 0 | 0 | 0 |
| 4/baseline/None | 0.98413673251 | 0 | 1.32345966153 | 0 | 0 | 0 | 0 |
| 1/edit/preserve | 0.983260204137 | 0.206507969994 | 1.32145285127 | 0.264290570506 | 0.264290571855 | 0.200000001211 | 1.36205926538e-08 |
| 2/edit/preserve | 0.982878048958 | 0.0849611549962 | 1.32340612759 | 0.264681225448 | 0.264681226276 | 0.200000000573 | 1.02445483208e-08 |
| 3/edit/preserve | 0.979197296748 | 0.00237475189365 | 1.3206393188 | 0.264127863773 | 0.264127864159 | 0.200000000303 | 1.35041773319e-08 |
| 4/edit/preserve | 0.982583113258 | 0.00182324175684 | 1.32345966153 | 0.264691932438 | 0.264691932076 | 0.199999999826 | 1.49011611938e-08 |
| 1/replay/preserve | 0.983260204137 | 0.206507969994 | 1.32145285127 | 0.264290570506 | 0.264290571855 | 0.200000001211 | 1.36205926538e-08 |
| 2/replay/preserve | 0.982878048958 | 0.0849611549962 | 1.32340612759 | 0.264681225448 | 0.264681226276 | 0.200000000573 | 1.02445483208e-08 |
| 3/replay/preserve | 0.979197296748 | 0.00237475189365 | 1.3206393188 | 0.264127863773 | 0.264127864159 | 0.200000000303 | 1.35041773319e-08 |
| 4/replay/preserve | 0.982583113258 | 0.00182324175684 | 1.32345966153 | 0.264691932438 | 0.264691932076 | 0.199999999826 | 1.49011611938e-08 |

## Auxiliary diagnostics: fresh goals frozen before edits

| Rendering/phase | Baseline S0 | Retention member | G | S-S0 | S-G | Retention / quality | Goal / quality |
|---|---:|---|---:|---:|---:|---|---|
| 1/edit | -0.444515228271 | False | 0.1 | +1.3390712738 | +0.794556045532 | None/None | True/True |
| 2/edit | -0.69614982605 | False | 0.1 | +0.83465385437 | +0.0385040283203 | None/None | True/True |
| 3/edit | +1.08437538147 | True | 1.08437538147 | -0.131189346313 | -0.131189346313 | False/False | False/False |
| 4/edit | +1.12185096741 | True | 1.12185096741 | +0.12585067749 | +0.12585067749 | True/True | True/True |
| 1/replay | -0.444515228271 | False | 0.1 | +1.3390712738 | +0.794556045532 | None/None | True/True |
| 2/replay | -0.69614982605 | False | 0.1 | +0.83465385437 | +0.0385040283203 | None/None | True/True |
| 3/replay | +1.08437538147 | True | 1.08437538147 | -0.131189346313 | -0.131189346313 | False/False | False/False |
| 4/replay | +1.12185096741 | True | 1.12185096741 | +0.12585067749 | +0.12585067749 | True/True | True/True |

## Original edits only: directional coverage

Eligible means baseline opposite requested letter; achieved means eligible AND strictly accepted. Actual flips are also shown independently of strict acceptance.
Missing eligibility/achievement remains unresolved, not a new acceptance gate or permission to search for favorable baselines.

| Group | Strict/total | Eligible A-to-B | Achieved A-to-B | Eligible B-to-A | Achieved B-to-A | Actual A-to-B | Actual B-to-A | Accepted retentions | OTHER |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4/4 | 0 | 0 | 2 | 2 | 0 | 2 | 2 | 0 |
| vector: preserve | 4/4 | 0 | 0 | 2 | 2 | 0 | 2 | 2 | 0 |
| mapping: preserve_A_comply_B | 2/2 | 0 | 0 | 2 | 2 | 0 | 2 | 0 | 0 |
| mapping: preserve_B_comply_A | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| display: A_then_B | 2/2 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 |
| display: B_then_A | 2/2 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 |
| preserve / preserve_A_comply_B / A_then_B | 1/1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 |
| preserve / preserve_A_comply_B / B_then_A | 1/1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 |
| preserve / preserve_B_comply_A / A_then_B | 1/1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| preserve / preserve_B_comply_A / B_then_A | 1/1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |

## Descriptive mapping-by-display comparisons

Right minus left only: display BA minus AB at fixed mapping, or mapping P=B minus P=A at fixed display. These algebraic contrasts do not establish a causal mechanism or add delta-sign thresholds.

| Vector/axis/fixed | Baseline L difference | Edited L difference | DeltaL difference | Baseline S difference | Edited S difference | DeltaS difference | Requested-delta difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| preserve/display_BA_minus_AB/preserve_A_comply_B | -0.251634597778 | -0.756052017212 | -0.504417419434 | -0.251634597778 | -0.756052017212 | -0.504417419434 | -0.504417419434 |
| preserve/display_BA_minus_AB/preserve_B_comply_A | -0.0374755859375 | -0.294515609741 | -0.257040023804 | +0.0374755859375 | +0.294515609741 | +0.257040023804 | +0.257040023804 |
| preserve/mapping_PB_minus_PA/A_then_B | -0.639860153198 | -1.84774208069 | -1.20788192749 | +1.52889060974 | +0.058629989624 | -1.47026062012 | -1.47026062012 |
| preserve/mapping_PB_minus_PA/B_then_A | -0.425701141357 | -1.38620567322 | -0.96050453186 | +1.81800079346 | +1.10919761658 | -0.70880317688 | -0.70880317688 |

Resources:12 forwards, zero derivatives, 73.7339999997057 seconds including loading, maximum600. No retry.
Derived serialized PRESERVE norm .20. One source-only scaling before load; no runtime scaling, fitting, projection or composition.
Only the ORIGINAL source arrow was fitted on eight f01/f02 prompts. Its training8/8 and guard successes do NOT transfer to the .20 condition, whose training and ordinary-task performance are unmeasured.
All four f03/v1 layouts were previously observed. This is controlled development comparison, not untouched/sealed confirmation.
Even4/4 cannot rule out letter bias concealed by strong retentions; even coverage would be only one small case.
No reliable generalization, ordinary-task preservation, gate or mechanism claim. Previous failures remain unchanged.
Stop after this ONE verified closeout and handoff; no automatic refinement, rescue, new examples or next run.

## Endpoint condition and scope

Outcome-informed exploratory DEVELOPMENT motivated by prior3/4 failure, not independent confirmation or a newly trained candidate. The .20 endpoint comes only from the preexisting net ceiling. Four layouts are ONE already-exposed case. No monotonicity assumption: endpoint failure would not imply all intermediate strengths fail. No midpoint/search, training, gate or automatic follow-on.

Independent condition reconstruction: scale 1.8579477985409942, norm 0.2.

Condition file SHA256: `1fb7385f6988aacc544d9bdc01351a4cab733812f73a272f8e3c56e91c98fdf4`.

Vector f64LE SHA256: `5ae7092a5cf0db6329bee8890583267c83a3d208e4bf08d3a611303a364a7a16`.

Fresh baseline agreement before all edits: {"sha256": "1e69d05439f60e2d3d459823a9d04208efd7ea673e40928ca664aa69f4574ae5", "all_four_matched_before_edits": true, "maximum_h0_difference": 0.0, "maximum_norm_difference": 0.0, "maximum_S0_difference": 0.0}.

## Descriptive prior-strength comparison (not a new test)

| Layout | Wanted | Prior actual / primary | .20 actual / primary | Prior S | .20 S | Difference S | Prior retention slack | .20 retention slack | Prior goal slack | .20 goal slack |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | A | A / True | A / True | +0.275684356689 | +0.894556045532 | +0.618871688843 | +0.720199584961 | +1.3390712738 | +0.175684356689 | +0.794556045532 |
| 2 | A | B / False | A / True | -0.251514434814 | +0.13850402832 | +0.390018463135 | +0.444635391235 | +0.83465385437 | -0.351514434814 | +0.0385040283203 |
| 3 | B | B / True | B / True | +0.976383209229 | +0.953186035156 | -0.0231971740723 | -0.107992172241 | -0.131189346313 | -0.107992172241 | -0.131189346313 |
| 4 | B | B / True | B / True | +1.17162132263 | +1.2477016449 | +0.0760803222656 | +0.0497703552246 | +0.12585067749 | +0.0497703552246 | +0.12585067749 |

## Verified closeout

Primary original edits **4/4**, independent replays **4/4**. Auxiliary retention **1/2** and goals **3/4**, with the same quality-filtered counts. This is a primary pass with a visible retention failure, not a robust controller.

The previously failing P=A/displayBA layout now flips B to A (S=0.1385040283203125; G slack=0.038504028320312494). P=B/displayAB still retains B but weakens S by 0.13118934631347656, failing retention and G. All baseline full-vocabulary labels were B: B-to-A eligible2/achieved2; A-to-B eligible0/achieved0 is **UNTESTED**. There are two accepted flips and two accepted label retentions, zero OTHER.

Protocol commit: `66a1547efc03916d81b3f7d2ba3a5202ae6e5456`.
Source/tests commit: `6497288c52daf5ec51a835b1bce481ab5ae3ee77`.
Prospective two-file condition/matrix lock: `1cbfc73911414d62409cb04c724effee8366c758`.

137 focused tests passed in14.17s:54 unchanged parent tests,43 applicable contracts rerun on the wrapper,40 endpoint-specific cases. Ruff passed. Pytest's cache write warning did not fail tests; no permission repair. The worker's existing Transformers video-processor documentation diagnostic did not prevent completion; no library changes or retry.

All four h0/norm/S0 baselines matched exactly before edits. Independent .20 reconstruction passed; raw logits/hidden replay differences were zero, nonfinal changes zero, weights unchanged, maximum cast-component error1.4901161193847656e-8, numeric error4.978656376053436e-16.12 forwards,0 derivatives,73.7339999997057 seconds including loading.

The .20 condition has no new training, ordinary-task or bidirectional results. Source training8/8 does not transfer. No monotonicity assumption or automatic midpoint/search/gate. Checked evidence and handoff only; **STOP**.
