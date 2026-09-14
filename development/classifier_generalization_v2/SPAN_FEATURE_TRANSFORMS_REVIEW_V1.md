# Span Feature Transforms V1 — Independent Review

Job: `span_features_review_20260914_1059`. Verdict: **BLOCKING (2 defects)**; synthetic scope otherwise clean.

Native pins (verified SHA256):
- `span_feature_transforms_v1.py` = 536d9c4fff413d6324e433008796386ab8943b4ab071dc557b39f0b652672447
- `test_span_feature_transforms_v1.py` = b8ff29d2282cabcb4505bbca7c49fd460c343ce7a624468063b1c3f8d9d9894b
- Runtime: Python 3.12.14, numpy 2.5.3, sklearn 1.9.1

**B1 — concatenated width is 8272, not 10224.** 3072+3072+2048+36+44 = 8272; `FEATURE_DIM` = 8272; observed matrix (48, 8272) in `.runtime`. Slices are stable and match `REPRESENTATION_NAMES`: F1[0:3072], F2[3072:6144], F3[6144:8192], F4[8192:8228], F5[8228:8272]. Source line 104 comment `# 10224` and the brief are arithmetically false. Minimal fix: correct the comment/expectation to 8272; 10224 would need unspecified extra width.

**B2 — `transform` rejects mixed-length batches.** `_as_window_sequence` (line 329) calls `np.asarray(windows)`; differing n raises raw `ValueError: inhomogeneous shape` before validation. The plan requires variable suffix lengths 1..16 with no padding. `fit` and `f1`..`f5` handle mixed lengths (verified `f1` → (2, 3072)). Minimal fix: detect a bare window with `isinstance(windows, np.ndarray) and windows.ndim == 3`, else iterate `_validate_windows`.

**Acknowledged:** one combined `SpanFeatureTransforms.fit` learns F4 contrasts and F5 PCA — all representations once per fold. The plan's "25 fold fits + 5 full-TRAIN fits" is not 25 separate per-representation fits; the counter must record actual fit calls (e.g. 5 fold + 1 full-TRAIN) verbatim.

**Confirmed:** 19/19 synthetic tests PASS in classifier `.runtime` (bytecode off); imports only numpy/sklearn/stdlib/harness/module; no real data/model/provider/tokenizer/capture/classifier-fit/HOLDOUT/network/install/Git/config/coordination/subagents. Layers 6/10/18 with block10 anchor; lengths 1..16 no padding; TRAIN-only means/directions/PCA/std; state frozen and deterministic; validation present; zero-norm directions and zero-variance PCA scale 1.0 stay finite; post-product finiteness checked. Minor: `PCA_SAMPLES` (line 183) is unreachable (CLASS_SUPPORT forces ≥8 rows), so `test_rejects_too_few_pca_rows` actually trips CLASSES_DISTINCT.
