# Doubled development dataset experiment

User-authorized additional120TRAIN and40VALIDATION examples; total240/80 with60/20 per class. HOLDOUT192 remains untouched. Original examples, labels, split/family assignments and features stay immutable. New variants retain existing group/ancestry IDs and are not claimed as new independent mechanisms.

User subsequently requested root-only generation. The three dispatched DeepSeek authoring jobs were stopped; their partial artifacts are excluded. Codex authors all160additional examples and performs explicit rule/answer, schema and duplicate checks before capture. This is author self-review, not an independent label audit. Exact duplicates and cross-split near duplicates are checked before fitting. No label adjustment based on model predictions.

Capture only160new cases (320views), frozen Qwen revision/layer/readout and preprocessing unchanged. Reuse old320views. A separate finite capture lock is required for the new320forwards, not a cap increase on the old attempt.

Repeat original logistic grid C0.01/0.1/1/10 and XGBoost depth1/2/3, both binary/four-class, unchanged training-only centering and rowL2. Five unchanged group-disjoint folds, fixed threshold grid0.05..0.95. Maximum70CVfits and4refits, no extra grid search. Hyperparameters chosen solely using TRAIN OOF predictions. Save outcomes on original40validation, new40validation, combined80; also score original saved models on new40 for a same-test-set comparison. Validation remains developmental, not final independent confirmation.

Primary comparison is SELF precision/recall/F1; also report four-class macroF1 and confusion. Training/validation gaps can indicate poor generalization; insufficient sample size is a hypothesis, not an established underfitting diagnosis. No success guarantee.
