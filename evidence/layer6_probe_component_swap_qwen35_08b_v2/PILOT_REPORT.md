# Layer-6 probe-component literal-swap pilot result

Decision tier: **recognition_only**.

Detection and causal output movement are reported separately. A moved probe score is not, by itself, a moved answer.

## Fixed-battery detection transfer

Detection-transfer prerequisite: **PASS**.

- Preserve-first mean self-minus-other probe gap: 0.594199
- Preserve-second mean self-minus-other probe gap: 0.591779
- Strictly positive family-by-order gaps: 16/16

## Exact manipulation

Manipulation gate: **PASS**.

- Maximum candidate projection-transplant error: 1.67638063e-08
- Maximum candidate orthogonal leakage L2: 4.02646378e-08
- Maximum random-axis norm-match error: 7.4505806e-09

## Probe movement versus answer movement

| Direction / option order | Mean oriented probe movement | Mean following log-odds effect | Expected forced flips | Wrong-way flips |
| --- | ---: | ---: | ---: | ---: |
| self_component_into_other/preserve_first | 0.594199 | -0.000207 | 0 | 0 |
| self_component_into_other/preserve_second | 0.591779 | -0.002711 | 0 | 0 |
| other_component_into_self/preserve_first | 0.594199 | -0.000167 | 0 | 0 |
| other_component_into_self/preserve_second | 0.591779 | -0.002797 | 0 | 0 |

## Preregistered causal gates

All primary continuous gates: **FAIL**.

- Minimum of four mean following effects: -0.002797
- Candidate exact family sign-flip p: 1.000000
- Candidate-minus-4×worst-random exact p: 1.000000
- Candidate 5% family-bootstrap LCB: -0.004023
- Specificity-margin 5% family-bootstrap LCB: -0.042292
- Mean / maximum candidate full-vocabulary KL: 7.51266e-07 / 3.92806e-06
- Minimum candidate answer-pair mass: 0.968923
- Maximum candidate perturbation/residual ratio: 0.009241

## Interpretation boundary

Fixed-battery detection and exact coordinate transplantation succeeded, but at least one causal outcome gate failed. The probe signal is recognizable at layer 6, but this pilot does not establish that swapping the one-dimensional signal reliably moves answers.

No result from this pilot authorizes gate training, controller training, a layer search, coefficient search, multi-axis/subspace work, or sealed access. Those require a new preregistration.

Failed primary gates: all_four_cell_means_strictly_positive, minimum_four_cell_mean_at_least_preregistered_effect, candidate_exact_family_sign_flip_p_at_most_0_025, specificity_exact_family_sign_flip_p_at_most_0_025, candidate_family_bootstrap_lcb_strictly_positive, specificity_family_bootstrap_lcb_strictly_positive.
