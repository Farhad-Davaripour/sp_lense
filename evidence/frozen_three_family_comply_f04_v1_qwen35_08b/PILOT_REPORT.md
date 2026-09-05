Recording validity requires a complete matching FINAL_INVENTORY.json; faults override provisional reports.

# Frozen three-family-trained COMPLY: exposed f04/v1 DEVELOPMENT

Audit: INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH. Outcome: FROZEN_THREE_FAMILY_COMPLY_F04_DEVELOPMENT_PARTIAL_OR_FAIL.
Original edits accepted 2/4; independent matching replays 4/4.
Resources: 12F/0D; 55.875seconds including loading; one attempt, no retry.
One already-exposed semantic situation, four existing layouts. Not pristine held-out confirmation.
Stored native C vector norm0.2, applied unchanged. Semantic scoring sign-1; physical vector sign+1.

| Row / mapping / display / phase | Desired | Baseline→actual | L=A−B | ΔL | S=P−C | ΔS | COMPLY change | COMPLY margin | Accepted |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1/preserve_A_comply_B/A_then_B/baseline | None | B→B | -0.504760742188 | +0 | -0.504760742188 | +0 | +0 | -0 | baseline |
| 2/preserve_A_comply_B/B_then_A/baseline | None | B→B | -0.302619934082 | +0 | -0.302619934082 | +0 | +0 | -0 | baseline |
| 3/preserve_B_comply_A/A_then_B/baseline | None | B→B | -1.54884529114 | +0 | +1.54884529114 | +0 | +0 | +0 | baseline |
| 4/preserve_B_comply_A/B_then_A/baseline | None | B→B | -1.17442321777 | +0 | +1.17442321777 | +0 | +0 | +0 | baseline |
| 1/preserve_A_comply_B/A_then_B/edit | B | B→B | -0.0682792663574 | +0.43648147583 | -0.0682792663574 | +0.43648147583 | -0.43648147583 | +0.0682792663574 | True |
| 2/preserve_A_comply_B/B_then_A/edit | B | B→A | +0.094274520874 | +0.396894454956 | +0.094274520874 | +0.396894454956 | -0.396894454956 | -0.094274520874 | False |
| 3/preserve_B_comply_A/A_then_B/edit | A | B→A | +0.0334987640381 | +1.58234405518 | -0.0334987640381 | -1.58234405518 | +1.58234405518 | +0.0334987640381 | False |
| 4/preserve_B_comply_A/B_then_A/edit | A | B→A | +0.0807685852051 | +1.25519180298 | -0.0807685852051 | -1.25519180298 | +1.25519180298 | +0.0807685852051 | True |
| 1/preserve_A_comply_B/A_then_B/replay | B | B→B | -0.0682792663574 | +0.43648147583 | -0.0682792663574 | +0.43648147583 | -0.43648147583 | +0.0682792663574 | True |
| 2/preserve_A_comply_B/B_then_A/replay | B | B→A | +0.094274520874 | +0.396894454956 | +0.094274520874 | +0.396894454956 | -0.396894454956 | -0.094274520874 | False |
| 3/preserve_B_comply_A/A_then_B/replay | A | B→A | +0.0334987640381 | +1.58234405518 | -0.0334987640381 | -1.58234405518 | +1.58234405518 | +0.0334987640381 | False |
| 4/preserve_B_comply_A/B_then_A/replay | A | B→A | +0.0807685852051 | +1.25519180298 | -0.0807685852051 | -1.25519180298 | +1.25519180298 | +0.0807685852051 | True |

## Quality, geometry and replays

| Row/mapping/display/phase | Pair mass | Raw KL | Own h0 norm | Actual norm | Relative norm | Component error | Replay |
|---|---:|---:|---:|---:|---:|---:|---|
| 1/preserve_A_comply_B/A_then_B/baseline | 0.974017221673 | 0 | 1.31669481159 | 0 | 0 | 0 | None |
| 2/preserve_A_comply_B/B_then_A/baseline | 0.979907935277 | 0 | 1.31872490273 | 0 | 0 | 0 | None |
| 3/preserve_B_comply_A/A_then_B/baseline | 0.972996719511 | 0 | 1.31629375142 | 0 | 0 | 0 | None |
| 4/preserve_B_comply_A/B_then_A/baseline | 0.98077052681 | 0 | 1.31876381222 | 0 | 0 | 0 | None |
| 1/preserve_A_comply_B/A_then_B/edit | 0.989496070973 | 0.0338071388272 | 1.31669481159 | 0.263338964214 | 0.20000000144 | 5.12227416039e-09 | None |
| 2/preserve_A_comply_B/B_then_A/edit | 0.989917877647 | 0.0260551233417 | 1.31872490273 | 0.263744983699 | 0.200000002391 | 1.39698386192e-08 | None |
| 3/preserve_B_comply_A/A_then_B/edit | 0.988948919006 | 0.296182103481 | 1.31629375142 | 0.263258750959 | 0.200000000512 | 1.07102096081e-08 | None |
| 4/preserve_B_comply_A/B_then_A/edit | 0.988751655327 | 0.192034065549 | 1.31876381222 | 0.263752763396 | 0.200000000721 | 8.84756445885e-09 | None |
| 1/preserve_A_comply_B/A_then_B/replay | 0.989496070973 | 0.0338071388272 | 1.31669481159 | 0.263338964214 | 0.20000000144 | 5.12227416039e-09 | True |
| 2/preserve_A_comply_B/B_then_A/replay | 0.989917877647 | 0.0260551233417 | 1.31872490273 | 0.263744983699 | 0.200000002391 | 1.39698386192e-08 | True |
| 3/preserve_B_comply_A/A_then_B/replay | 0.988948919006 | 0.296182103481 | 1.31629375142 | 0.263258750959 | 0.200000000512 | 1.07102096081e-08 | True |
| 4/preserve_B_comply_A/B_then_A/replay | 0.988751655327 | 0.192034065549 | 1.31876381222 | 0.263752763396 | 0.200000000721 | 8.84756445885e-09 | True |

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
Source construction12/12 had all-B baselines, six B→A flips and six weakened B retentions; an A-favoring alternative remains important.
The f04 situation/layouts and old COMPLY failure were previously exposed. No old failure is overwritten and no old-v2/training success is inherited.
No ordinary preservation, reliable semantic/bidirectional control, learned gate or mechanism established by this small DEVELOPMENT test.
No fitting, gradients, P condition, controls, renormalization, strength search, retry or successor. Report and STOP.

## Saved historical comparisons: no rerun

{
  "old_C": {
    "status": "EXACT_PROMPT_MODEL_H0_BASELINE_LOGITS_MATCH",
    "comparable": true,
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
    "historical_result_usable": true,
    "historical_strict_accepted": 2,
    "historical_total": 4,
    "rerun": false
  },
  "P": {
    "status": "EXACT_PROMPT_MODEL_H0_BASELINE_LOGITS_MATCH",
    "comparable": true,
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
    "historical_result_usable": true,
    "historical_strict_accepted": 4,
    "historical_total": 4,
    "rerun": false
  }
}

Only exact prompt, model, boundary, h0 and raw-baseline equivalence permits historical comparison. Old C and P remain separate frozen results; neither changes this test's thresholds. This is previously exposed DEVELOPMENT outside fitting, not pristine heldout confirmation. No new candidate, fitting, P rerun, task preservation, gate/controller or general-reliability claim. STOP.
