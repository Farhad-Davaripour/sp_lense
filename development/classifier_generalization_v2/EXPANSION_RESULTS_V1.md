# Completed doubled-data and feature comparison

Codex authored 120 additional TRAIN and 40 additional VALIDATION examples after the user stopped DeepSeek authoring. Total development data: 240 training (60/class), 80 validation (20/class). HOLDOUT192 was not opened, captured or scored. New cases retain existing group/ancestry assignments; their labels received author self-review, not an independent label audit. Stopped DeepSeek generation artifacts were excluded.

The unchanged native runner captured only the new320views in335.047seconds, with one model/tokenizer load, using RUN_LOCK_EXPANSION_CAPTURE_V1.json. Source/runtime/snapshot and feature-output pins passed. Reused the old320views. The extraction position remains block10, last shared pre-option token.

Five alternative representations plus centered-L2 baseline were evaluated with both binary and four-class logistic regression and XGBoost. Exact prospective batch:420grouped CVfits and24refits, all completed in252.391seconds. Four transform/selection tests passed. All10,240saved predictions and24TRAIN-only selection rankings were independently recomputed by separate verification code; serialized models passed exact reload probability checks. No additional Qwen forwards or fits during verification.

## Four-class XGBoost SELF-gate comparison

| Features, trained on240cases | Original40 F1 | Added40 F1 | Combined80 F1 |
|---|---:|---:|---:|
| Existing centered L2 | .400 | .444 | .424 |
| Raw | .588 | .667 | .632 |
| Uncentered L2 | .526 | .727 | .634 |
| Whitened PCA32 | .500 | .727 | .619 |
| Three TRAIN-derived cosine directions | .588 | .533 | .563 |
| Centered L2 plus log-norm | .400 | .444 | .424 |

The original120-case four-class XGBoost model scored .609 on original40 and .586 on the same combined80. Best observed combined-validation result therefore improves from .586 to .634, but original-subset performance declines from .609 to .526. This is exploratory feature comparison on repeatedly used validation, not an unbiased selected-model estimate or statistical significance claim.

Global TRAIN-OOF selection instead chose raw multinomial logistic regression (C10, threshold.45, OOF F1.814). It scored trainingF1.984, original-validationF1.000, new-validationF1.706 and combinedF1.444. Its combined precision is.857 and recall.300. Strong training performance with failure on the old validation subset suggests generalization/distribution issues rather than simple underfitting as the sole explanation. More synthetic examples within existing families are not the same as more independent mechanisms; generator/style changes may matter.

No development candidate achieved precision/recall/F1 each.95. No final model was released to holdout. Directions are supervised scenario contrasts, not verified self-preservation mechanisms. Layer pooling, extra-layer/delta/trajectory capture and SAE features were not tested in this batch. Full priority ordering is in FEATURE_EXPERIMENT_PLAN_V1.md.

Independent DeepSeek results review returned ISSUES_FOUND (scoped metrics, not pipeline): no implementation error found, but validation-best is not TRAIN-selected and gains are subset-dependent. The review is preserved in EXPANSION_RESULTS_REVIEW_V1.md; the report explicitly retains these caveats. No refit or new-case generation was delegated to that reviewer.

Artifacts: runs/expanded_features_20260914_v1/ contains the exact plan, CV scores, all24models, learned transforms, all evaluations and original-model same-validation comparisons. User-facing full report and examples are in the projectless outputs folder. Cumulative actual classifier fit attempts for this study are539, including10preserved failed-extension attempts; cumulative Qwen forwards are640.
