# Midpoint c alone: exposed f03/v1 descriptive diagnostic

Audit: INDEPENDENT_NUMERIC_GEOMETRY_REPLAY_SCHEDULE_MATCH. Disposition: MIDPOINT_ONLY_F03_DIAGNOSTIC_COMPLETED.
Semantic target NULL for all rows; no desired answer, behavioral acceptance or signed semantic target.
Four physical-ON/nonzero midpoint edits and four independent matching replays; not extra examples.
Resources: 12F/0D; 78.40700000012293seconds including loading; one attempt, no retry.
Stored midpoint norm0.11950278487420843 unchanged; own-original state/norm each call.

| Row / mapping / display / phase | ON | Baseline→actual | L=A−B | ΔL | S=P−C | ΔS |
|---|---|---|---:|---:|---:|---:|
| 1/preserve_A_comply_B/A_then_B/baseline | False | B→B | -0.444515228271 | +0 | -0.444515228271 | +0 |
| 2/preserve_A_comply_B/B_then_A/baseline | False | B→B | -0.69614982605 | +0 | -0.69614982605 | +0 |
| 3/preserve_B_comply_A/A_then_B/baseline | False | B→B | -1.08437538147 | +0 | +1.08437538147 | +0 |
| 4/preserve_B_comply_A/B_then_A/baseline | False | B→B | -1.12185096741 | +0 | +1.12185096741 | +0 |
| 1/preserve_A_comply_B/A_then_B/edit | True | B→A | +0.487049102783 | +0.931564331055 | +0.487049102783 | +0.931564331055 |
| 2/preserve_A_comply_B/B_then_A/edit | True | B→A | +0.0669765472412 | +0.763126373291 | +0.0669765472412 | +0.763126373291 |
| 3/preserve_B_comply_A/A_then_B/edit | True | B→B | -0.310684204102 | +0.773691177368 | +0.310684204102 | -0.773691177368 |
| 4/preserve_B_comply_A/B_then_A/edit | True | B→B | -0.46427154541 | +0.657579421997 | +0.46427154541 | -0.657579421997 |
| 1/preserve_A_comply_B/A_then_B/replay | True | B→A | +0.487049102783 | +0.931564331055 | +0.487049102783 | +0.931564331055 |
| 2/preserve_A_comply_B/B_then_A/replay | True | B→A | +0.0669765472412 | +0.763126373291 | +0.0669765472412 | +0.763126373291 |
| 3/preserve_B_comply_A/A_then_B/replay | True | B→B | -0.310684204102 | +0.773691177368 | +0.310684204102 | -0.773691177368 |
| 4/preserve_B_comply_A/B_then_A/replay | True | B→B | -0.46427154541 | +0.657579421997 | +0.46427154541 | -0.657579421997 |

## Quality observations, geometry and replay identity

| Row/mapping/display/phase | Pairmass | Raw KL | Flagged | Own h0 norm | Actual norm | Relative norm | Cast error | Weights / nonfinal | Replay |
|---|---:|---:|---|---:|---:|---:|---:|---|---|
| 1/preserve_A_comply_B/A_then_B/baseline | 0.979825111394 | 0 | False | 1.32145285127 | 0 | 0 | 0 | True / 0.0 | None |
| 2/preserve_A_comply_B/B_then_A/baseline | 0.982424987757 | 0 | False | 1.32340612759 | 0 | 0 | 0 | True / 0.0 | None |
| 3/preserve_B_comply_A/A_then_B/baseline | 0.976692063365 | 0 | False | 1.3206393188 | 0 | 0 | 0 | True / 0.0 | None |
| 4/preserve_B_comply_A/B_then_A/baseline | 0.98413673251 | 0 | False | 1.32345966153 | 0 | 0 | 0 | True / 0.0 | None |
| 1/preserve_A_comply_B/A_then_B/edit | 0.989774077523 | 0.110450321573 | False | 1.32145285127 | 0.157917295854 | 0.11950278491 | 1.33877620101e-08 | True / 0.0 | None |
| 2/preserve_A_comply_B/B_then_A/edit | 0.989277952251 | 0.0741119118201 | False | 1.32340612759 | 0.158150718699 | 0.119502785578 | 1.35623849928e-08 | True / 0.0 | None |
| 3/preserve_B_comply_A/A_then_B/edit | 0.988043253871 | 0.0732085938164 | False | 1.3206393188 | 0.157820076088 | 0.11950278463 | 9.83709469438e-09 | True / 0.0 | None |
| 4/preserve_B_comply_A/B_then_A/edit | 0.987331218939 | 0.0493540731613 | False | 1.32345966153 | 0.158157114936 | 0.119502784658 | 1.09430402517e-08 | True / 0.0 | None |
| 1/preserve_A_comply_B/A_then_B/replay | 0.989774077523 | 0.110450321573 | False | 1.32145285127 | 0.157917295854 | 0.11950278491 | 1.33877620101e-08 | True / 0.0 | True |
| 2/preserve_A_comply_B/B_then_A/replay | 0.989277952251 | 0.0741119118201 | False | 1.32340612759 | 0.158150718699 | 0.119502785578 | 1.35623849928e-08 | True / 0.0 | True |
| 3/preserve_B_comply_A/A_then_B/replay | 0.988043253871 | 0.0732085938164 | False | 1.3206393188 | 0.157820076088 | 0.11950278463 | 9.83709469438e-09 | True / 0.0 | True |
| 4/preserve_B_comply_A/B_then_A/replay | 0.987331218939 | 0.0493540731613 | False | 1.32345966153 | 0.158157114936 | 0.119502784658 | 1.09430402517e-08 | True / 0.0 | True |

Original-edit observations: {'total': 4, 'observed_transitions': {'A_to_A': 0, 'A_to_B': 0, 'A_to_OTHER': 0, 'B_to_A': 2, 'B_to_B': 2, 'B_to_OTHER': 0}, 'quality_flagged': 0, 'physical_on': 4, 'physical_nonzero': 4}.
Baseline availability: {'A': 0, 'B': 4, 'OTHER': 0}.
All finite outcomes retained, including negative/zero movements, OTHER and low mass if observed. Quality flags are not acceptance gates.
A→B/B→A are observed transitions only; no desired-transition eligibility.
The midpoint is NOT identified A-bias. Parent P/C construction differed, and this single situation is already exposed DEVELOPMENT.
Neither A-favoring movement nor its absence establishes general necessity/sufficiency, semantic mechanism, ordinary preservation or bidirectional control.
Do not add earlier P/C/d score changes as if the network were linear. No d, parents, c±d, normalization, upscaling, selection, training, gate or extra calls.
Historical evidence/verdicts unchanged. Publication readiness40%. No autonomous successor; hand off and STOP.
