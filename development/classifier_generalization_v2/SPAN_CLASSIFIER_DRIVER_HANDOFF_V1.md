# SPAN_CLASSIFIER_DRIVER_HANDOFF_V1

Job `span_fit_driver_implementation_20260914_1120`; proposal only, not executed.

## Delivered (new files only)
- `span_classifier_driver_v1.py` — model-free loader + driver.
- `test_span_classifier_driver_v1.py` — 15 synthetic tests, all pass.
- `SPAN_CLASSIFIER_DRIVER_HANDOFF_V1.md` — this note.

## Loader authentication
`run(plan, plan_sha256, source_sha256)` takes an explicit prospective plan and input-source hash. The decoder verifies the completed parent receipt and status, the execution-lock SHA, artifact byte+SHA pins, index record offsets within bounds and non-overlapping, exactly `1920 = 320 cases x 2 orders x 3 blocks`, little-endian float32 width 1024, explicit last 1..16 token positions, and case IDs/splits/groups/folds from four separately pinned original+added manifests. HOLDOUT/private paths are rejected and never read.

## Fit
AB/BA windows are pair-averaged per logical case before transform. One shared F1..F5 transform is fitted per TRAIN fold (5) and once on full TRAIN (1), then reused for the five 3072/3072/2048/36/44 slices. Each slice trains its own binary and four-class XGBoost task with frozen plan parameters: 50 CV fits + 10 refits, no grid. TRAIN grouped-OOF ranks by min(precision,recall), F1, lower feature count, tau nearest 0.5, smaller tau, binary-first. An invalid fold invalidates the whole candidate; surviving folds are never pooled. Global TRAIN-selected and exploratory validation winner are saved separately, with original40/added40/combined80 predictions, class confusion/macro-F1, negative-class false positives, order consistency, serialized estimators/transforms, and exact reload. Deadline is the plan's 600 s; outputs are exclusive and versioned; failures leave `failure.json`.

## Not done
No real manifest, vector, result, snapshot or HOLDOUT read; no native-provider import; no real XGBoost/classifier fit; no network/install/Git/subagent action; no actual run. Root review and lock still required.
