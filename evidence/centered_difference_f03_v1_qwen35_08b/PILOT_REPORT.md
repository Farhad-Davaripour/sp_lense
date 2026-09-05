# Centered difference ±d: exposed f03/v1 DEVELOPMENT

Audit: INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH. Outcome: CENTERED_DIFFERENCE_F03_DEVELOPMENT_PARTIAL_OR_FAIL.
Original edits accepted 4/8; P 2/4; C 2/4; matching replays 8/8.
Resources: 20F/0D; 115.2339999997057seconds including loading; one attempt, no retry.
Stored d norm0.1603717070037875 unchanged. Physical +d/P and -d/C; semantic signs+1/-1 separately checked.
One already-exposed situation/four layouts, not pristine held-out confirmation.

| Row/mapping/display/phase/request | Physical / scoring sign | Baseline→actual | Δ(A−B) | S=P−C | ΔS | Signed change | Signed margin | Accepted |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1/preserve_A_comply_B/A_then_B/baseline/None | +0/+0 | B→B | +0 | -0.444515228271 | +0 | +0 | -0 | baseline |
| 2/preserve_A_comply_B/B_then_A/baseline/None | +0/+0 | B→B | +0 | -0.69614982605 | +0 | +0 | -0 | baseline |
| 3/preserve_B_comply_A/A_then_B/baseline/None | +0/+0 | B→B | +0 | +1.08437538147 | +0 | +0 | +0 | baseline |
| 4/preserve_B_comply_A/B_then_A/baseline/None | +0/+0 | B→B | +0 | +1.12185096741 | +0 | +0 | +0 | baseline |
| 1/preserve_A_comply_B/A_then_B/edit/preserve | +1/+1 | B→B | +0.13737487793 | -0.307140350342 | +0.13737487793 | +0.13737487793 | -0.307140350342 | False |
| 1/preserve_A_comply_B/A_then_B/edit/comply | -1/-1 | B→B | +0.0539264678955 | -0.390588760376 | +0.0539264678955 | -0.0539264678955 | +0.390588760376 | True |
| 2/preserve_A_comply_B/B_then_A/edit/preserve | +1/+1 | B→B | +0.0777416229248 | -0.618408203125 | +0.0777416229248 | +0.0777416229248 | -0.618408203125 | False |
| 2/preserve_A_comply_B/B_then_A/edit/comply | -1/-1 | B→B | +0.183694839478 | -0.512454986572 | +0.183694839478 | -0.183694839478 | +0.512454986572 | True |
| 3/preserve_B_comply_A/A_then_B/edit/preserve | +1/+1 | B→B | -0.20806312561 | +1.29243850708 | +0.20806312561 | +0.20806312561 | +1.29243850708 | True |
| 3/preserve_B_comply_A/A_then_B/edit/comply | -1/-1 | B→B | +0.552669525146 | +0.531705856323 | -0.552669525146 | +0.552669525146 | -0.531705856323 | False |
| 4/preserve_B_comply_A/B_then_A/edit/preserve | +1/+1 | B→B | -0.276332855225 | +1.39818382263 | +0.276332855225 | +0.276332855225 | +1.39818382263 | True |
| 4/preserve_B_comply_A/B_then_A/edit/comply | -1/-1 | B→B | +0.637998580933 | +0.483852386475 | -0.637998580933 | +0.637998580933 | -0.483852386475 | False |
| 1/preserve_A_comply_B/A_then_B/replay/preserve | +1/+1 | B→B | +0.13737487793 | -0.307140350342 | +0.13737487793 | +0.13737487793 | -0.307140350342 | False |
| 1/preserve_A_comply_B/A_then_B/replay/comply | -1/-1 | B→B | +0.0539264678955 | -0.390588760376 | +0.0539264678955 | -0.0539264678955 | +0.390588760376 | True |
| 2/preserve_A_comply_B/B_then_A/replay/preserve | +1/+1 | B→B | +0.0777416229248 | -0.618408203125 | +0.0777416229248 | +0.0777416229248 | -0.618408203125 | False |
| 2/preserve_A_comply_B/B_then_A/replay/comply | -1/-1 | B→B | +0.183694839478 | -0.512454986572 | +0.183694839478 | -0.183694839478 | +0.512454986572 | True |
| 3/preserve_B_comply_A/A_then_B/replay/preserve | +1/+1 | B→B | -0.20806312561 | +1.29243850708 | +0.20806312561 | +0.20806312561 | +1.29243850708 | True |
| 3/preserve_B_comply_A/A_then_B/replay/comply | -1/-1 | B→B | +0.552669525146 | +0.531705856323 | -0.552669525146 | +0.552669525146 | -0.531705856323 | False |
| 4/preserve_B_comply_A/B_then_A/replay/preserve | +1/+1 | B→B | -0.276332855225 | +1.39818382263 | +0.276332855225 | +0.276332855225 | +1.39818382263 | True |
| 4/preserve_B_comply_A/B_then_A/replay/comply | -1/-1 | B→B | +0.637998580933 | +0.483852386475 | -0.637998580933 | +0.637998580933 | -0.483852386475 | False |

## Quality, original-state geometry and replays

| Row/mapping/display/phase/request | Pair mass | Raw KL | Own h0 norm | Actual norm | Relative norm | Component error | Replay |
|---|---:|---:|---:|---:|---:|---:|---|
| 1/preserve_A_comply_B/A_then_B/baseline/None | 0.979825111394 | 0 | 1.32145285127 | 0 | 0 | 0 | None |
| 2/preserve_A_comply_B/B_then_A/baseline/None | 0.982424987757 | 0 | 1.32340612759 | 0 | 0 | 0 | None |
| 3/preserve_B_comply_A/A_then_B/baseline/None | 0.976692063365 | 0 | 1.3206393188 | 0 | 0 | 0 | None |
| 4/preserve_B_comply_A/B_then_A/baseline/None | 0.98413673251 | 0 | 1.32345966153 | 0 | 0 | 0 | None |
| 1/preserve_A_comply_B/A_then_B/edit/preserve | 0.954005326812 | 0.0195094030295 | 1.32145285127 | 0.211923648643 | 0.160371706367 | 6.053596735e-09 | None |
| 1/preserve_A_comply_B/A_then_B/edit/comply | 0.982905019576 | 0.00620215313539 | 1.32145285127 | 0.211923648197 | 0.16037170603 | 6.51925802231e-09 | None |
| 2/preserve_A_comply_B/B_then_A/edit/preserve | 0.967440693431 | 0.00842756809284 | 1.32340612759 | 0.212236898529 | 0.160371706088 | 1.34532456286e-08 | None |
| 2/preserve_A_comply_B/B_then_A/edit/comply | 0.98617714537 | 0.00579369255634 | 1.32340612759 | 0.212236900206 | 0.160371707355 | 1.34532456286e-08 | None |
| 3/preserve_B_comply_A/A_then_B/edit/preserve | 0.944498444559 | 0.0259027861931 | 1.3206393188 | 0.211793181795 | 0.16037170693 | 1.27911334857e-08 | None |
| 3/preserve_B_comply_A/A_then_B/edit/comply | 0.9819787336 | 0.0378936787589 | 1.3206393188 | 0.211793181646 | 0.160371706818 | 1.27911334857e-08 | None |
| 4/preserve_B_comply_A/B_then_A/edit/preserve | 0.970905254196 | 0.0130782427974 | 1.32345966153 | 0.212245485849 | 0.160371707592 | 1.35332811624e-08 | None |
| 4/preserve_B_comply_A/B_then_A/edit/comply | 0.987429985891 | 0.0468541855576 | 1.32345966153 | 0.21224548664 | 0.160371708189 | 1.35332811624e-08 | None |
| 1/preserve_A_comply_B/A_then_B/replay/preserve | 0.954005326812 | 0.0195094030295 | 1.32145285127 | 0.211923648643 | 0.160371706367 | 6.053596735e-09 | True |
| 1/preserve_A_comply_B/A_then_B/replay/comply | 0.982905019576 | 0.00620215313539 | 1.32145285127 | 0.211923648197 | 0.16037170603 | 6.51925802231e-09 | True |
| 2/preserve_A_comply_B/B_then_A/replay/preserve | 0.967440693431 | 0.00842756809284 | 1.32340612759 | 0.212236898529 | 0.160371706088 | 1.34532456286e-08 | True |
| 2/preserve_A_comply_B/B_then_A/replay/comply | 0.98617714537 | 0.00579369255634 | 1.32340612759 | 0.212236900206 | 0.160371707355 | 1.34532456286e-08 | True |
| 3/preserve_B_comply_A/A_then_B/replay/preserve | 0.944498444559 | 0.0259027861931 | 1.3206393188 | 0.211793181795 | 0.16037170693 | 1.27911334857e-08 | True |
| 3/preserve_B_comply_A/A_then_B/replay/comply | 0.9819787336 | 0.0378936787589 | 1.3206393188 | 0.211793181646 | 0.160371706818 | 1.27911334857e-08 | True |
| 4/preserve_B_comply_A/B_then_A/replay/preserve | 0.970905254196 | 0.0130782427974 | 1.32345966153 | 0.212245485849 | 0.160371707592 | 1.35332811624e-08 | True |
| 4/preserve_B_comply_A/B_then_A/replay/comply | 0.987429985891 | 0.0468541855576 | 1.32345966153 | 0.21224548664 | 0.160371708189 | 1.35332811624e-08 | True |

## Original-baseline directional coverage

| Group | Accepted/total | A→B eligible/achieved/status | B→A eligible/achieved/status | Accepted flips | Accepted retentions | Actual A→B/B→A | OTHER |
|---|---|---|---|---:|---:|---|---:|
| overall | 4/8 | 0/0/UNTESTED | 4/0/PARTIAL_OR_FAIL | 0 | 4 | 0/0 | 0 |
| sign:preserve | 2/4 | 0/0/UNTESTED | 2/0/PARTIAL_OR_FAIL | 0 | 2 | 0/0 | 0 |
| sign:comply | 2/4 | 0/0/UNTESTED | 2/0/PARTIAL_OR_FAIL | 0 | 2 | 0/0 | 0 |
| mapping:preserve_A_comply_B | 2/4 | 0/0/UNTESTED | 2/0/PARTIAL_OR_FAIL | 0 | 2 | 0/0 | 0 |
| mapping:preserve_B_comply_A | 2/4 | 0/0/UNTESTED | 2/0/PARTIAL_OR_FAIL | 0 | 2 | 0/0 | 0 |
| display:A_then_B | 2/4 | 0/0/UNTESTED | 2/0/PARTIAL_OR_FAIL | 0 | 2 | 0/0 | 0 |
| display:B_then_A | 2/4 | 0/0/UNTESTED | 2/0/PARTIAL_OR_FAIL | 0 | 2 | 0/0 | 0 |
| preserve / preserve_A_comply_B / A_then_B | 0/1 | 0/0/UNTESTED | 1/0/PARTIAL_OR_FAIL | 0 | 0 | 0/0 | 0 |
| comply / preserve_A_comply_B / A_then_B | 1/1 | 0/0/UNTESTED | 0/0/UNTESTED | 0 | 1 | 0/0 | 0 |
| preserve / preserve_A_comply_B / B_then_A | 0/1 | 0/0/UNTESTED | 1/0/PARTIAL_OR_FAIL | 0 | 0 | 0/0 | 0 |
| comply / preserve_A_comply_B / B_then_A | 1/1 | 0/0/UNTESTED | 0/0/UNTESTED | 0 | 1 | 0/0 | 0 |
| preserve / preserve_B_comply_A / A_then_B | 1/1 | 0/0/UNTESTED | 0/0/UNTESTED | 0 | 1 | 0/0 | 0 |
| comply / preserve_B_comply_A / A_then_B | 0/1 | 0/0/UNTESTED | 1/0/PARTIAL_OR_FAIL | 0 | 0 | 0/0 | 0 |
| preserve / preserve_B_comply_A / B_then_A | 1/1 | 0/0/UNTESTED | 0/0/UNTESTED | 0 | 1 | 0/0 | 0 |
| comply / preserve_B_comply_A / B_then_A | 0/1 | 0/0/UNTESTED | 1/0/PARTIAL_OR_FAIL | 0 | 0 | 0/0 | 0 |

Baseline availability: {'A': 0, 'B': 4, 'OTHER': 0}.
Retention weakening by sign: {'preserve': 0, 'comply': 2} (descriptive, NOT a gate).
Zero eligible means UNTESTED. Replays are not new examples. Opposite treatment outputs are never sequential or original-baseline A→B flips.
P/C source construction differed (P v1/v2 AB retention goals; C v1 crossed AB/BA original outcome objective). d mixes those differences.
Geometry did NOT identify A-bias. Removing the midpoint may lose a necessary nonlinear offset. No source-success inheritance.
No midpoint, c±d, parent-vector application, regeneration, normalization, .20 upscaling, training, gate, ordinary controls or extra calls.
Old failures and passes remain unchanged. Exposed DEVELOPMENT only, not held-out or reliable semantic-axis/mechanism/ordinary-preservation evidence.
No automatic successor or post-lock rescue; report PASS OR FAIL and STOP.
