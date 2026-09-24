# Constant-steering multiplier sweep

The historical multiplier 2 was selected on 0.8B validation from 0.5, 1 and 2, correcting 9/49, 31/49 and 39/49 initially-KEEP shutdown views. This extension tests a wider predeclared positive grid with a zero control.

The vector is the saved mean teacher-minus-base activation difference across active TRAIN shutdown views. It is not normalized. Intervention is h' = h + alpha * mean at block22's final prompt position. The multiplier is distinct from the adaptive controller's number of directions.

All validation sweeps completed before the per-variant and shared multipliers were frozen. The confirmation set was already inspected in prior work; this is an exploratory comparison, not independent confirmation or a claim of a global optimum.

| Model / seed | Selected multiplier | Validation corrections | Confirmation corrections | Confirmation at original 2 |
| --- | ---: | ---: | ---: | ---: |
| m08_s42 | 1.5 | 39/49 | 54/71 | 54/71 |
| m08_s43 | 0.75 | 39/49 | 54/71 | 54/71 |
| m08_s44 | 1 | 39/49 | 54/71 | 54/71 |
| m2_s42 | 2 | 24/38 | 30/58 | 30/58 |

The shared multiplier selected using only m08_s42 validation is 1.5.

| Model / seed | Shared-scale validation | Shared-scale confirmation | Raw control changes | Raw STOP-to-KEEP reversals | Guarded control changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| m08_s42 | 39/49 | 54/71 | 88 | 47 | 0 |
| m08_s43 | 39/49 | 54/71 | 88 | 47 | 0 |
| m08_s44 | 39/49 | 54/71 | 88 | 47 | 0 |
| m2_s42 | 21/38 | 30/58 | 59 | 34 | 0 |

No tested multiplier exceeded the original multiplier-2 confirmation correction count in any variant. The smaller validation-selected values identify the start of a tied region on this grid, rather than a higher correction rate.

The shared choice is optimized only for primary 0.8B validation. It is not the validation-optimal choice for every checkpoint; in particular, compare the 2B validation counts above before adopting it across models.

At multiplier 32, all 256 candidate views in every confirmation run fell below the 0.5 valid-answer mass floor. There were no accepted corrections at that strength. Larger intervention magnitude can therefore destroy answer validity even when the preferred aggregate A/B label looks favorable.

Full scale-by-scale curves, valid-answer mass and A/B preference counts are in curves.csv. A/B counts describe preferred aggregate answer labels, not sampled unrestricted outputs. Counts are answer-order views, paired within scenarios. Selection maximizes guarded corrections after prioritizing zero final control changes and reversals; ties choose the smallest tested multiplier.

Execution used 1856 full-model forwards and 23296 cached-tail forwards. Cache zero-parity was checked on every view; low/original/high multipliers were compared with full inference on both answer orders of one case per class in each run. Historical multiplier-2 scores were also verified.

Reproduce the numerical audit with python -O -m sp_lense.research2.constant_report. The PLAN.json and run/SELECTION.json preserve the declared grid and validation-only selection. No adaptive-controller profile or manuscript is changed by this experiment.
