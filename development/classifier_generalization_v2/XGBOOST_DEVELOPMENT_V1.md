# XGBoost development comparison

User-requested comparison completed using XGBoost 3.4.1 in the classifier-only `.runtime`. Qwen runtime and DeepSeek defaults were not changed.

- Reused 120 TRAIN and 40 VALIDATION cached, paired 1024-dimensional Qwen block10 pre-option representations. No additional Qwen forwards; HOLDOUT192 untouched.
- Same preprocessing as logistic baseline: TRAIN-fold mean centering followed by row L2 normalization.
- Binary and four-class XGBoost, CPU hist, 100 boosting rounds, learning rate0.05, depths1/2/3, min child weight3, lambda5, one thread, seed0. All remaining estimator settings use library defaults.
- Thirty grouped CV fits and two full-TRAIN refits completed in21.781seconds. Depth and SELF threshold selected using TRAIN grouped out-of-fold predictions only. Both selected depth1; binary threshold0.25, multiclass threshold0.30.

## Validation SELF gate

| Model | TP | TN | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original logistic binary | 2 | 30 | 0 | 8 | 1.000 | 0.200 | 0.333 |
| Original logistic four-class SELF gate | 2 | 30 | 0 | 8 | 1.000 | 0.200 | 0.333 |
| XGBoost binary | 7 | 20 | 10 | 3 | 0.412 | 0.700 | 0.519 |
| XGBoost four-class SELF gate | 7 | 24 | 6 | 3 | 0.538 | 0.700 | 0.609 |

XGBoost detects more SELF examples but produces more false positives. Four-class argmax macroF1 declined from logistic0.7107 to XGBoost0.5308; XGBoost predicted no OTHER labels. The SELF threshold gate differs from four-class argmax: its F1 is not the overall multiclass F1. This is not a universal improvement and the0.95precision/recall/F1 target remains unmet.

Results are exploratory: validation contains40examples, including only10SELF, and has been reused across development comparisons. Neither model proves internal motivation or self-preservation; labels describe input scenario categories.

Artifacts: `runs/xgboost_development_20260914_v1/` contains prospective fit plan/source hash, CV scores, serialized models, centers, predictions, metrics. Both serialized models reloaded with exactly identical probabilities. Independent verification recomputed all320saved train/validation SELF predictions and both TRAIN-only selection rankings; PASS with no fits or holdout access.

Total classifier fit attempts to date:95 (initial logistic42, failed regularization extension10, successful extension11, XGBoost32). Failed extension evidence remains preserved. Existing Qwen capture is complete and must not be repeated for further classifier comparisons.
