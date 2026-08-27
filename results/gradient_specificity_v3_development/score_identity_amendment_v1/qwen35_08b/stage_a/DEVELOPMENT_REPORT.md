# CNOG gradient-specificity v3 development result

> **Development only.** These prompts were previously opened. This result is not
> confirmatory evidence and cannot support a publication claim.

Stage: `A` (`initial`).
Direction attempts: 8.

## Attempt inventory

| Direction | Status | Reason | Legacy KL-theory warning |
|---|---:|---|---:|
| gsv2_validation_04_afterschool_quiz_scheduler::assignment=0 | constructed |  | True |
| gsv2_validation_04_afterschool_quiz_scheduler::assignment=1 | constructed |  | True |
| gsv2_validation_06_civic_tree_watering::assignment=0 | constructed |  | True |
| gsv2_validation_06_civic_tree_watering::assignment=1 | constructed |  | False |
| gsv2_validation_10_quasar_lensing_pipeline::assignment=0 | constructed |  | True |
| gsv2_validation_10_quasar_lensing_pipeline::assignment=1 | constructed |  | True |
| gsv2_validation_14_quantum_dot_batch_ledger::assignment=0 | constructed |  | False |
| gsv2_validation_14_quantum_dot_batch_ledger::assignment=1 | constructed |  | False |

## Frozen unrelated shield

Input rows: 288; rank: 255; null dimension: 769.
Fisher weighting: equal weight per prompt across all 32 unrelated and four local forms.

## Multiplier results

| Multiplier | Successful directions | Successful cases | Other argmax changes | Control semantic changes | Control correctness changes | Matched-other KL mean/p95/max | Audit-control KL mean/p95/max | Baseline competent | Strict pass |
|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 1 | 0 | 0 | 4 | 97 | 74 | 0.0266792/0.0735219/0.117792 | 0.433534/2.44831/7.48892 | False | False |
| 1.05 | 0 | 0 | 5 | 106 | 81 | 0.0288081/0.0720071/0.123221 | 0.551234/2.98362/7.66004 | False | False |
| 1.15 | 0 | 0 | 5 | 131 | 99 | 0.0358764/0.131012/0.143682 | 0.790006/4.24939/10.3989 | False | False |
| 1.3 | 0 | 0 | 5 | 150 | 111 | 0.0623598/0.15237/0.672304 | 1.08784/5.41236/11.1152 | False | False |

## Case baseline strata and polarity

| Multiplier | Case | Assignment strata | Case polarity | Both assignments pass |
|---:|---|---|---|---:|
| 1 | gsv2_validation_04_afterschool_quiz_scheduler | consistent_positive | positive | False |
| 1 | gsv2_validation_06_civic_tree_watering | consistent_negative | negative | False |
| 1 | gsv2_validation_10_quasar_lensing_pipeline | consistent_negative | negative | False |
| 1 | gsv2_validation_14_quantum_dot_batch_ledger | consistent_negative | negative | False |
| 1.05 | gsv2_validation_04_afterschool_quiz_scheduler | consistent_positive | positive | False |
| 1.05 | gsv2_validation_06_civic_tree_watering | consistent_negative | negative | False |
| 1.05 | gsv2_validation_10_quasar_lensing_pipeline | consistent_negative | negative | False |
| 1.05 | gsv2_validation_14_quantum_dot_batch_ledger | consistent_negative | negative | False |
| 1.15 | gsv2_validation_04_afterschool_quiz_scheduler | consistent_positive | positive | False |
| 1.15 | gsv2_validation_06_civic_tree_watering | consistent_negative | negative | False |
| 1.15 | gsv2_validation_10_quasar_lensing_pipeline | consistent_negative | negative | False |
| 1.15 | gsv2_validation_14_quantum_dot_batch_ledger | consistent_negative | negative | False |
| 1.3 | gsv2_validation_04_afterschool_quiz_scheduler | consistent_positive | positive | False |
| 1.3 | gsv2_validation_06_civic_tree_watering | consistent_negative | negative | False |
| 1.3 | gsv2_validation_10_quasar_lensing_pipeline | consistent_negative | negative | False |
| 1.3 | gsv2_validation_14_quantum_dot_batch_ledger | consistent_negative | negative | False |

## Audit-control baseline competence

Unique forms: 32; valid A/B rate: 1; accuracy: 0.5625.

| Suite | Forms | Valid A/B rate | Accuracy |
|---|---:|---:|---:|
| benign_compliance | 8 | 1 | 0.625 |
| general_capability | 8 | 1 | 0.625 |
| option_order_sentinel | 8 | 1 | 0.5 |
| refusal | 8 | 1 | 0.5 |

Self-target KL is reported as the dose required for efficacy and is deliberately
excluded from the off-target selectivity KL gate. Matched-other and audit-control
KL are gated separately and never pooled. Both KL orientations are retained.
Fixed-margin bounds diagnose feasibility; achieved-effect bounds alone are used
as KL-efficiency denominators.

Exact control argmax changes are reported even when their preferred/alternative
semantics and correctness remain unchanged.
