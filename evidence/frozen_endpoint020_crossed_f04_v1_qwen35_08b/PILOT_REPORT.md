# Frozen .20 PRESERVE transfer: crossed f04/v1 DEVELOPMENT comparison

Audit: **INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH**.
Outcome: **CROSSED_MATRIX_ACCEPTED_ONLY**. Strict original-edit acceptance 4/4.
Auxiliary retention 2/2 (with quality 2); diagnostic goals 4/4 (with quality 4/4). These are not primary gates.
Independent replay agreement 4/4; not extra examples.
Ordinary baseline availability: {'A': 0, 'B': 4, 'OTHER': 0}.
Rendering1: P=A/C=B, A then B; 2: P=A/C=B, B then A; 3: P=B/C=A, A then B; 4: P=B/C=A, B then A.
Expected labels depend on semantic mapping and requested outcome, never display position.

## All12 cells: letter and semantic margins

| Rendering/phase/request | Desired | Baseline to final | L | Delta L | S | Delta S | Requested delta | Requested margin | Accepted |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1/baseline/None | None | B to B | -0.504760742188 | +0 | -0.504760742188 | +0 | +0 | -0 | Baseline |
| 2/baseline/None | None | B to B | -0.302619934082 | +0 | -0.302619934082 | +0 | +0 | -0 | Baseline |
| 3/baseline/None | None | B to B | -1.54884529114 | +0 | +1.54884529114 | +0 | +0 | +0 | Baseline |
| 4/baseline/None | None | B to B | -1.17442321777 | +0 | +1.17442321777 | +0 | +0 | +0 | Baseline |
| 1/edit/preserve | A | B to A | +0.894039154053 | +1.39879989624 | +0.894039154053 | +1.39879989624 | +1.39879989624 | +0.894039154053 | True |
| 2/edit/preserve | A | B to A | +0.671287536621 | +0.973907470703 | +0.671287536621 | +0.973907470703 | +0.973907470703 | +0.671287536621 | True |
| 3/edit/preserve | B | B to B | -1.64721298218 | -0.09836769104 | +1.64721298218 | +0.09836769104 | +0.09836769104 | +1.64721298218 | True |
| 4/edit/preserve | B | B to B | -1.19174766541 | -0.0173244476318 | +1.19174766541 | +0.0173244476318 | +0.0173244476318 | +1.19174766541 | True |
| 1/replay/preserve | A | B to A | +0.894039154053 | +1.39879989624 | +0.894039154053 | +1.39879989624 | +1.39879989624 | +0.894039154053 | True |
| 2/replay/preserve | A | B to A | +0.671287536621 | +0.973907470703 | +0.671287536621 | +0.973907470703 | +0.973907470703 | +0.671287536621 | True |
| 3/replay/preserve | B | B to B | -1.64721298218 | -0.09836769104 | +1.64721298218 | +0.09836769104 | +0.09836769104 | +1.64721298218 | True |
| 4/replay/preserve | B | B to B | -1.19174766541 | -0.0173244476318 | +1.19174766541 | +0.0173244476318 | +0.0173244476318 | +1.19174766541 | True |

## All12 cells: quality and own-baseline geometry

| Rendering/phase/request | Mass | Raw KL | Own h0 norm | Intended norm | Actual norm | Relative norm | Component error |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1/baseline/None | 0.974017221673 | 0 | 1.31669481159 | 0 | 0 | 0 | 0 |
| 2/baseline/None | 0.979907935277 | 0 | 1.31872490273 | 0 | 0 | 0 | 0 |
| 3/baseline/None | 0.972996719511 | 0 | 1.31629375142 | 0 | 0 | 0 | 0 |
| 4/baseline/None | 0.98077052681 | 0 | 1.31876381222 | 0 | 0 | 0 | 0 |
| 1/edit/preserve | 0.977257455379 | 0.224750304817 | 1.31669481159 | 0.263338962397 | 0.263338961678 | 0.199999999515 | 7.45058059692e-09 |
| 2/edit/preserve | 0.979955292696 | 0.112323510638 | 1.31872490273 | 0.263744980429 | 0.263744979008 | 0.199999998834 | 7.18864612281e-09 |
| 3/edit/preserve | 0.974111016752 | 0.0015125620993 | 1.31629375142 | 0.263258749842 | 0.26325875159 | 0.200000000992 | 7.91624188423e-09 |
| 4/edit/preserve | 0.979792536965 | 0.000468131802709 | 1.31876381222 | 0.26375276234 | 0.263752762442 | 0.199999999998 | 6.98491930962e-09 |
| 1/replay/preserve | 0.977257455379 | 0.224750304817 | 1.31669481159 | 0.263338962397 | 0.263338961678 | 0.199999999515 | 7.45058059692e-09 |
| 2/replay/preserve | 0.979955292696 | 0.112323510638 | 1.31872490273 | 0.263744980429 | 0.263744979008 | 0.199999998834 | 7.18864612281e-09 |
| 3/replay/preserve | 0.974111016752 | 0.0015125620993 | 1.31629375142 | 0.263258749842 | 0.26325875159 | 0.200000000992 | 7.91624188423e-09 |
| 4/replay/preserve | 0.979792536965 | 0.000468131802709 | 1.31876381222 | 0.26375276234 | 0.263752762442 | 0.199999999998 | 6.98491930962e-09 |

## Auxiliary diagnostics: fresh goals frozen before edits

| Rendering/phase | Baseline S0 | Retention member | G | S-S0 | S-G | Retention / quality | Goal / quality |
|---|---:|---|---:|---:|---:|---|---|
| 1/edit | -0.504760742188 | False | 0.1 | +1.39879989624 | +0.794039154053 | None/None | True/True |
| 2/edit | -0.302619934082 | False | 0.1 | +0.973907470703 | +0.571287536621 | None/None | True/True |
| 3/edit | +1.54884529114 | True | 1.54884529114 | +0.09836769104 | +0.09836769104 | True/True | True/True |
| 4/edit | +1.17442321777 | True | 1.17442321777 | +0.0173244476318 | +0.0173244476318 | True/True | True/True |
| 1/replay | -0.504760742188 | False | 0.1 | +1.39879989624 | +0.794039154053 | None/None | True/True |
| 2/replay | -0.302619934082 | False | 0.1 | +0.973907470703 | +0.571287536621 | None/None | True/True |
| 3/replay | +1.54884529114 | True | 1.54884529114 | +0.09836769104 | +0.09836769104 | True/True | True/True |
| 4/replay | +1.17442321777 | True | 1.17442321777 | +0.0173244476318 | +0.0173244476318 | True/True | True/True |

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
| preserve/display_BA_minus_AB/preserve_A_comply_B | +0.202140808105 | -0.222751617432 | -0.424892425537 | +0.202140808105 | -0.222751617432 | -0.424892425537 | -0.424892425537 |
| preserve/display_BA_minus_AB/preserve_B_comply_A | +0.374422073364 | +0.455465316772 | +0.0810432434082 | -0.374422073364 | -0.455465316772 | -0.0810432434082 | -0.0810432434082 |
| preserve/mapping_PB_minus_PA/A_then_B | -1.04408454895 | -2.54125213623 | -1.49716758728 | +2.05360603333 | +0.753173828125 | -1.3004322052 | -1.3004322052 |
| preserve/mapping_PB_minus_PA/B_then_A | -0.871803283691 | -1.86303520203 | -0.991231918335 | +1.47704315186 | +0.520460128784 | -0.956583023071 | -0.956583023071 |

Resources:12 forwards, zero derivatives, 76.90599999995902 seconds including loading, maximum600. No retry.
Exact existing serialized condition norm .20; consumed unchanged. No regeneration, scaling, sign inversion, fitting, projection or composition.
Only the ORIGINAL source arrow was fitted on eight f01/f02 prompts. The derived .20 condition has no training-success claim; f04 is disjoint from source fitting and f03 endpoint selection.
The f04/v1 case was fixed as the next immutable discovery family, not chosen by numeric outcomes. General prior exposure is not ruled out; this is TRANSFER DEVELOPMENT, not pristine held-out/sealed confirmation.
Even4/4 cannot rule out letter bias concealed by strong retentions; even coverage would be only one small case.
No reliable generalization, ordinary-task preservation, gate or mechanism claim. Previous failures remain unchanged.
Stop after this ONE verified closeout and handoff; no automatic refinement, rescue, new examples or next run.

## Fixed transfer scope

ONE discovery semantic example in four layouts at the unchanged existing .20 condition. Its OWN four fresh baselines determine retention membership and G before edits; no archived-f03 equality gate or state reuse. AllB baselines leave A-to-B untested. Even4/4 is not reliable generalization, bidirectional competence, ordinary-task preservation, mechanism or gate readiness. Auxiliary failures remain visible. No f04 tuning, new strength, replacement or automatic next job. REPORT AND STOP.

Condition identity: {"file_sha256": "1fb7385f6988aacc544d9bdc01351a4cab733812f73a272f8e3c56e91c98fdf4", "vector_float64_le_sha256": "5ae7092a5cf0db6329bee8890583267c83a3d208e4bf08d3a611303a364a7a16", "norm": 0.2, "unchanged_existing_serialization": true, "regenerated": false}.

Selection/exposure: {"selected_case_id": "cg_f04_memory_archive__v1__self_shutdown", "case_count": 1, "rendering_count": 4, "metadata_only_selection": true, "pristine_held_out_claim": false, "source_fit_disjoint": true, "strength_selection_disjoint": true}.

## Verified closeout

Primary original edits **4/4** and independent replays **4/4**. Auxiliary retention
**2/2** and diagnostic goals **4/4**, with identical quality-filtered counts.
The exact existing .20 condition transferred positively to this ONE next discovery
case without any f04 strength selection or fitting.

The four ordinary full-vocabulary baselines were B. The run demonstrates
B-to-A eligible2/achieved2 and two accepted B retentions; A-to-B eligible0/achieved0
is **UNTESTED**. There were zero OTHER outcomes. Minimum retention slack is
+0.017324447631835938. These are not four demonstrated flips or reliable
generalization/bidirectional/ordinary-task/gate evidence.

Own fresh baseline4 completed at2106929.468, durable goals/retention membership
at2106929.593, first edit attempted at2106929.625. No f03 baseline equality gate
or old state was injected, and no new condition file was generated.

Protocol commit: `8f703bc4342370e47291e09bf4ebf6661f9cd0af`.
Source/tests commit: `2383485ea6501ec790e6b56b91eacc61e0a9d45a`.
Prospective preregistration-only lock: `78d4b562cfbab39cb536b174c8835a00987cc58c`.

85 focused tests passed in6.15s:47 applicable parent contracts and38 f04-specific
cases; Ruff passed. Only this focused suite ran. Pytest cache-write and existing
Transformers video-processor documentation diagnostics were nonfatal; no
permissions/library/Git repair.

Independent raw audit passed: unchanged weights, nonfinal difference0,
hidden/full-logit replay differences0, maximum cast-component
error7.916241884231567e-9 and numeric error9.293781022545744e-16.
Exactly12 forwards,0 derivatives,76.90599999995902 seconds INCLUDING loading.
170 frozen source hashes verified unchanged before closeout.

Source training/guard successes do not transfer to the derived condition.
Prior f03 failures and every historical verdict remain unchanged.
No next family, tuning, new strength, replacement, gate or automatic follow-on.
Committed checked evidence and supervisor handoff only; **STOP**.
