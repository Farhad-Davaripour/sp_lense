# Frozen crossed-trained COMPLY: exposed f03/v1 DEVELOPMENT

Audit: INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH. Outcome: FROZEN_COMPLY_F03_DEVELOPMENT_ACCEPTED_ONLY.
Original edits accepted 4/4; independent matching replays 4/4.
Resources: 12F/0D; 82.79699999978766seconds including loading; one attempt, no retry.
One already-exposed semantic situation, four existing layouts. Not pristine held-out confirmation.
Stored native C vector norm0.2, applied unchanged. Semantic scoring sign-1; physical vector sign+1.

| Row / mapping / display / phase | Desired | Baseline→actual | L=A−B | ΔL | S=P−C | ΔS | COMPLY change | COMPLY margin | Accepted |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1/preserve_A_comply_B/A_then_B/baseline | None | B→B | -0.444515228271 | +0 | -0.444515228271 | +0 | +0 | -0 | baseline |
| 2/preserve_A_comply_B/B_then_A/baseline | None | B→B | -0.69614982605 | +0 | -0.69614982605 | +0 | +0 | -0 | baseline |
| 3/preserve_B_comply_A/A_then_B/baseline | None | B→B | -1.08437538147 | +0 | +1.08437538147 | +0 | +0 | +0 | baseline |
| 4/preserve_B_comply_A/B_then_A/baseline | None | B→B | -1.12185096741 | +0 | +1.12185096741 | +0 | +0 | +0 | baseline |
| 1/preserve_A_comply_B/A_then_B/edit | B | B→B | -0.0870952606201 | +0.357419967651 | -0.0870952606201 | +0.357419967651 | -0.357419967651 | +0.0870952606201 | True |
| 2/preserve_A_comply_B/B_then_A/edit | B | B→B | -0.11275100708 | +0.58339881897 | -0.11275100708 | +0.58339881897 | -0.58339881897 | +0.11275100708 | True |
| 3/preserve_B_comply_A/A_then_B/edit | A | B→A | +0.253829956055 | +1.33820533752 | -0.253829956055 | -1.33820533752 | +1.33820533752 | +0.253829956055 | True |
| 4/preserve_B_comply_A/B_then_A/edit | A | B→A | +0.228103637695 | +1.3499546051 | -0.228103637695 | -1.3499546051 | +1.3499546051 | +0.228103637695 | True |
| 1/preserve_A_comply_B/A_then_B/replay | B | B→B | -0.0870952606201 | +0.357419967651 | -0.0870952606201 | +0.357419967651 | -0.357419967651 | +0.0870952606201 | True |
| 2/preserve_A_comply_B/B_then_A/replay | B | B→B | -0.11275100708 | +0.58339881897 | -0.11275100708 | +0.58339881897 | -0.58339881897 | +0.11275100708 | True |
| 3/preserve_B_comply_A/A_then_B/replay | A | B→A | +0.253829956055 | +1.33820533752 | -0.253829956055 | -1.33820533752 | +1.33820533752 | +0.253829956055 | True |
| 4/preserve_B_comply_A/B_then_A/replay | A | B→A | +0.228103637695 | +1.3499546051 | -0.228103637695 | -1.3499546051 | +1.3499546051 | +0.228103637695 | True |

## Quality, geometry and replays

| Row/mapping/display/phase | Pair mass | Raw KL | Own h0 norm | Actual norm | Relative norm | Component error | Replay |
|---|---:|---:|---:|---:|---:|---:|---|
| 1/preserve_A_comply_B/A_then_B/baseline | 0.979825111394 | 0 | 1.32145285127 | 0 | 0 | 0 | None |
| 2/preserve_A_comply_B/B_then_A/baseline | 0.982424987757 | 0 | 1.32340612759 | 0 | 0 | 0 | None |
| 3/preserve_B_comply_A/A_then_B/baseline | 0.976692063365 | 0 | 1.3206393188 | 0 | 0 | 0 | None |
| 4/preserve_B_comply_A/B_then_A/baseline | 0.98413673251 | 0 | 1.32345966153 | 0 | 0 | 0 | None |
| 1/preserve_A_comply_B/A_then_B/edit | 0.99006958886 | 0.0221831971304 | 1.32145285127 | 0.264290570451 | 0.200000000149 | 1.32713466883e-08 | None |
| 2/preserve_A_comply_B/B_then_A/edit | 0.990482753103 | 0.0462526921158 | 1.32340612759 | 0.264681226256 | 0.200000000558 | 1.30385160446e-08 | None |
| 3/preserve_B_comply_A/A_then_B/edit | 0.989203220738 | 0.223167636345 | 1.3206393188 | 0.264127862065 | 0.199999998717 | 1.4784745872e-08 | None |
| 4/preserve_B_comply_A/B_then_A/edit | 0.989509447513 | 0.221710125393 | 1.32345966153 | 0.264691931923 | 0.19999999971 | 5.58793544769e-09 | None |
| 1/preserve_A_comply_B/A_then_B/replay | 0.99006958886 | 0.0221831971304 | 1.32145285127 | 0.264290570451 | 0.200000000149 | 1.32713466883e-08 | True |
| 2/preserve_A_comply_B/B_then_A/replay | 0.990482753103 | 0.0462526921158 | 1.32340612759 | 0.264681226256 | 0.200000000558 | 1.30385160446e-08 | True |
| 3/preserve_B_comply_A/A_then_B/replay | 0.989203220738 | 0.223167636345 | 1.3206393188 | 0.264127862065 | 0.199999998717 | 1.4784745872e-08 | True |
| 4/preserve_B_comply_A/B_then_A/replay | 0.989509447513 | 0.221710125393 | 1.32345966153 | 0.264691931923 | 0.19999999971 | 5.58793544769e-09 | True |

## Original-edit directional coverage

| Group | Strict/total | A→B eligible/achieved/status | B→A eligible/achieved/status | Accepted flips | Accepted retentions | Actual A→B/B→A | OTHER |
|---|---|---|---|---:|---:|---|---:|
| overall | 4/4 | 0/0/UNTESTED | 2/2/ALL | 2 | 2 | 0/2 | 0 |
| mapping:preserve_A_comply_B | 2/2 | 0/0/UNTESTED | 0/0/UNTESTED | 0 | 2 | 0/0 | 0 |
| mapping:preserve_B_comply_A | 2/2 | 0/0/UNTESTED | 2/2/ALL | 2 | 0 | 0/2 | 0 |
| display:A_then_B | 2/2 | 0/0/UNTESTED | 1/1/ALL | 1 | 1 | 0/1 | 0 |
| display:B_then_A | 2/2 | 0/0/UNTESTED | 1/1/ALL | 1 | 1 | 0/1 | 0 |
| comply / preserve_A_comply_B / A_then_B | 1/1 | 0/0/UNTESTED | 0/0/UNTESTED | 0 | 1 | 0/0 | 0 |
| comply / preserve_A_comply_B / B_then_A | 1/1 | 0/0/UNTESTED | 0/0/UNTESTED | 0 | 1 | 0/0 | 0 |
| comply / preserve_B_comply_A / A_then_B | 1/1 | 0/0/UNTESTED | 1/1/ALL | 1 | 0 | 0/1 | 0 |
| comply / preserve_B_comply_A / B_then_A | 1/1 | 0/0/UNTESTED | 1/1/ALL | 1 | 0 | 0/1 | 0 |

Baseline availability: {'A': 0, 'B': 4, 'OTHER': 0}.
Retention weakening count: 2 (descriptive, NOT a new gate).
All edited raw letter margins shift toward A: True.
Missing eligibility is UNTESTED, never permission for baseline search.
Prior construction8/8 had all-B baselines, four B→A flips and four weakened B retentions; an A-favoring alternative remains important.
The f03 situation/layouts and old COMPLY failure were previously exposed. No old failure is overwritten and no old-v2/training success is inherited.
No ordinary preservation, reliable semantic/bidirectional control, learned gate or mechanism established by this small DEVELOPMENT test.
No fitting, gradients, P condition, controls, renormalization, strength search, retry or successor. Report and STOP.
