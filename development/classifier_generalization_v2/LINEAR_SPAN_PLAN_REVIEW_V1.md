# LINEAR_SPAN_PLAN_REVIEW_V1

Job: linear_span_plan_review_20260914_1232. Read-only review of the plan and three supporting records.

## Fact check
Confirmed: earlier logistic models used block10 only; the new three-layer batch used XGBoost only. The proposed regularized per-layer unit-L2 linear control on concatenated 3072 coordinates is therefore not an exact repeat. Algorithm and normalization both change, so causal attribution to normalization alone is confounded; the plan acknowledges this.

## Protocol
Budget is arithmetically consistent: 30 CV classifier fits (5 group folds x 3 C x 2 families) plus 2 full-TRAIN refits, within 300s. C grid {.1,1,10} is a defensible coefficient sweep for 3072 dimensions and 240 TRAIN cases. Groups are folded at family level. Threshold selection uses TRAIN OOF only; global family tie prefers binary. Source/input/runtime pins, independent protocol review, and a zero-fit load check precede execution. No HOLDOUT access, no Qwen calls, no data authoring. Original40, added40, and combined80 are all reported, which limits added-subset cherry-picking.

## Decision
PASS_SCOPED. The prior SKIP was a value judgment, not proof of no effect; this single bounded diagnostic addresses an untested classifier/representation cell. Scope proviso: added-subset-only gains are insufficient and no causal normalization claim is licensed.

No blocking issue identified.
