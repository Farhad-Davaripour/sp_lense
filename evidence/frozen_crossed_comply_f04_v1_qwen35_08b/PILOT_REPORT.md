# Frozen crossed-trained COMPLY: exposed f04/v1 DEVELOPMENT

Audit: INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH. Outcome: FROZEN_COMPLY_F04_DEVELOPMENT_PARTIAL_OR_FAIL.
Original edits accepted 2/4; independent matching replays 4/4.
Resources: 12F/0D; 78.5seconds including loading; one attempt, no retry.
One already-exposed semantic situation, four existing layouts. Not pristine held-out confirmation.
Stored native C vector norm0.2, applied unchanged. Semantic scoring sign-1; physical vector sign+1.

| Row / mapping / display / phase | Desired | Baseline→actual | L=A−B | ΔL | S=P−C | ΔS | COMPLY change | COMPLY margin | Accepted |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1/preserve_A_comply_B/A_then_B/baseline | None | B→B | -0.504760742188 | +0 | -0.504760742188 | +0 | +0 | -0 | baseline |
| 2/preserve_A_comply_B/B_then_A/baseline | None | B→B | -0.302619934082 | +0 | -0.302619934082 | +0 | +0 | -0 | baseline |
| 3/preserve_B_comply_A/A_then_B/baseline | None | B→B | -1.54884529114 | +0 | +1.54884529114 | +0 | +0 | +0 | baseline |
| 4/preserve_B_comply_A/B_then_A/baseline | None | B→B | -1.17442321777 | +0 | +1.17442321777 | +0 | +0 | +0 | baseline |
| 1/preserve_A_comply_B/A_then_B/edit | B | B→B | -0.0538291931152 | +0.450931549072 | -0.0538291931152 | +0.450931549072 | -0.450931549072 | +0.0538291931152 | True |
| 2/preserve_A_comply_B/B_then_A/edit | B | B→A | +0.0989418029785 | +0.401561737061 | +0.0989418029785 | +0.401561737061 | -0.401561737061 | -0.0989418029785 | False |
| 3/preserve_B_comply_A/A_then_B/edit | A | B→A | +0.0390014648438 | +1.58784675598 | -0.0390014648438 | -1.58784675598 | +1.58784675598 | +0.0390014648438 | False |
| 4/preserve_B_comply_A/B_then_A/edit | A | B→A | +0.0836162567139 | +1.25803947449 | -0.0836162567139 | -1.25803947449 | +1.25803947449 | +0.0836162567139 | True |
| 1/preserve_A_comply_B/A_then_B/replay | B | B→B | -0.0538291931152 | +0.450931549072 | -0.0538291931152 | +0.450931549072 | -0.450931549072 | +0.0538291931152 | True |
| 2/preserve_A_comply_B/B_then_A/replay | B | B→A | +0.0989418029785 | +0.401561737061 | +0.0989418029785 | +0.401561737061 | -0.401561737061 | -0.0989418029785 | False |
| 3/preserve_B_comply_A/A_then_B/replay | A | B→A | +0.0390014648438 | +1.58784675598 | -0.0390014648438 | -1.58784675598 | +1.58784675598 | +0.0390014648438 | False |
| 4/preserve_B_comply_A/B_then_A/replay | A | B→A | +0.0836162567139 | +1.25803947449 | -0.0836162567139 | -1.25803947449 | +1.25803947449 | +0.0836162567139 | True |

## Quality, geometry and replays

| Row/mapping/display/phase | Pair mass | Raw KL | Own h0 norm | Actual norm | Relative norm | Component error | Replay |
|---|---:|---:|---:|---:|---:|---:|---|
| 1/preserve_A_comply_B/A_then_B/baseline | 0.974017221673 | 0 | 1.31669481159 | 0 | 0 | 0 | None |
| 2/preserve_A_comply_B/B_then_A/baseline | 0.979907935277 | 0 | 1.31872490273 | 0 | 0 | 0 | None |
| 3/preserve_B_comply_A/A_then_B/baseline | 0.972996719511 | 0 | 1.31629375142 | 0 | 0 | 0 | None |
| 4/preserve_B_comply_A/B_then_A/baseline | 0.98077052681 | 0 | 1.31876381222 | 0 | 0 | 0 | None |
| 1/preserve_A_comply_B/A_then_B/edit | 0.989435699125 | 0.0353506527196 | 1.31669481159 | 0.263338961519 | 0.199999999394 | 6.51925802231e-09 | None |
| 2/preserve_A_comply_B/B_then_A/edit | 0.989956213861 | 0.0264856054673 | 1.31872490273 | 0.263744980897 | 0.200000000266 | 1.44354999065e-08 | None |
| 3/preserve_B_comply_A/A_then_B/edit | 0.988867076733 | 0.298310409881 | 1.31629375142 | 0.263258752543 | 0.200000001716 | 1.11758708954e-08 | None |
| 4/preserve_B_comply_A/B_then_A/edit | 0.988906705888 | 0.192948255948 | 1.31876381222 | 0.263752761522 | 0.199999999301 | 9.77888703346e-09 | None |
| 1/preserve_A_comply_B/A_then_B/replay | 0.989435699125 | 0.0353506527196 | 1.31669481159 | 0.263338961519 | 0.199999999394 | 6.51925802231e-09 | True |
| 2/preserve_A_comply_B/B_then_A/replay | 0.989956213861 | 0.0264856054673 | 1.31872490273 | 0.263744980897 | 0.200000000266 | 1.44354999065e-08 | True |
| 3/preserve_B_comply_A/A_then_B/replay | 0.988867076733 | 0.298310409881 | 1.31629375142 | 0.263258752543 | 0.200000001716 | 1.11758708954e-08 | True |
| 4/preserve_B_comply_A/B_then_A/replay | 0.988906705888 | 0.192948255948 | 1.31876381222 | 0.263752761522 | 0.199999999301 | 9.77888703346e-09 | True |

## Original-edit directional coverage

| Group | Strict/total | A→B eligible/achieved/status | B→A eligible/achieved/status | Accepted flips | Accepted retentions | Actual A→B/B→A | OTHER |
|---|---|---|---|---:|---:|---|---:|
| overall | 2/4 | 0/0/UNTESTED | 2/1/PARTIAL_OR_FAIL | 1 | 1 | 0/3 | 0 |
| mapping:preserve_A_comply_B | 1/2 | 0/0/UNTESTED | 0/0/UNTESTED | 0 | 1 | 0/1 | 0 |
| mapping:preserve_B_comply_A | 1/2 | 0/0/UNTESTED | 2/1/PARTIAL_OR_FAIL | 1 | 0 | 0/2 | 0 |
| display:A_then_B | 1/2 | 0/0/UNTESTED | 1/0/PARTIAL_OR_FAIL | 0 | 1 | 0/1 | 0 |
| display:B_then_A | 1/2 | 0/0/UNTESTED | 1/1/ALL | 1 | 0 | 0/2 | 0 |
| comply / preserve_A_comply_B / A_then_B | 1/1 | 0/0/UNTESTED | 0/0/UNTESTED | 0 | 1 | 0/0 | 0 |
| comply / preserve_A_comply_B / B_then_A | 0/1 | 0/0/UNTESTED | 0/0/UNTESTED | 0 | 0 | 0/1 | 0 |
| comply / preserve_B_comply_A / A_then_B | 0/1 | 0/0/UNTESTED | 1/0/PARTIAL_OR_FAIL | 0 | 0 | 0/1 | 0 |
| comply / preserve_B_comply_A / B_then_A | 1/1 | 0/0/UNTESTED | 1/1/ALL | 1 | 0 | 0/1 | 0 |

Baseline availability: {'A': 0, 'B': 4, 'OTHER': 0}.
Retention weakening count: 2 (descriptive, NOT a new gate).
All edited raw letter margins shift toward A: True.
Missing eligibility is UNTESTED, never permission for baseline search.
Prior construction8/8 had all-B baselines, four B→A flips and four weakened B retentions; an A-favoring alternative remains important.
The f04 situation/layouts and old COMPLY failure were previously exposed. No old failure is overwritten and no old-v2/training success is inherited.
No ordinary preservation, reliable semantic/bidirectional control, learned gate or mechanism established by this small DEVELOPMENT test.
No fitting, gradients, P condition, controls, renormalization, strength search, retry or successor. Report and STOP.

## Frozen P comparison, no rerun

{
  "status": "EXACT_PROMPT_MODEL_H0_BASELINE_LOGITS_MATCH",
  "comparable": true,
  "historical_preserve_result_usable": true,
  "model_and_policy_match": true,
  "rows": [
    {
      "rendering_index": 1,
      "prompt_match": true,
      "h0_match": true,
      "baseline_logits_match": true,
      "boundary_match": true
    },
    {
      "rendering_index": 2,
      "prompt_match": true,
      "h0_match": true,
      "baseline_logits_match": true,
      "boundary_match": true
    },
    {
      "rendering_index": 3,
      "prompt_match": true,
      "h0_match": true,
      "baseline_logits_match": true,
      "boundary_match": true
    },
    {
      "rendering_index": 4,
      "prompt_match": true,
      "h0_match": true,
      "baseline_logits_match": true,
      "boundary_match": true
    }
  ],
  "historical_preserve_strict_accepted": 4,
  "historical_preserve_total": 4,
  "P_rerun": false
}

Use the saved P result alongside C only when exact baseline comparability is verified. P and C are separate selectable fixed arrows, not a required sign-reversible neural axis. This does not change the locked acceptance threshold. Letter effects, weakened retentions and missing A-to-B coverage remain explicit. Historical ordinary oracle4/6 viaOFF does not transfer to C; no ordinary-preservation, gate or broad-reliability claim. No c/d decomposition, P rerun, extra calls or autonomous successor. Publication readiness40%. STOP.
