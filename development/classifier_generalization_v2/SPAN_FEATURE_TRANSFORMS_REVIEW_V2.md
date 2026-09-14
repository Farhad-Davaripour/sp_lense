# Span Feature Transforms V2 — Independent Re-Review

Job `span_features_rereview_20260914_1120`. Verdict: **both V1 defects CLOSED; no regression.** Synthetic `.runtime` only; no forbidden surface touched.

Native pins (SHA256, re-verified):
- `span_feature_transforms_v1.py` = a23aaa38…f74741c5c0
- `test_span_feature_transforms_v1.py` = 76117a26…a5cc19c48
- Python 3.12.14, numpy 2.5.3, sklearn 1.9.1

**B1 closed.** `FEATURE_DIM` = 8272; source line 104 comment is now `# 8272`; explicit `8272` assertion added (test line 65); representation widths sum to 8272; no `10224` remains in the module. Stale `10224` survives only in `SPAN_FEATURE_TRANSFORMS_HANDOFF_V1.md` (out of edit scope; documentation-only).

**B2 closed.** `_as_window_sequence` now returns a bare 3-D ndarray window directly and otherwise iterates `_validate_windows`, so mixed lengths 1..16 fit/transform with no padding. Independently reproduced: mixed batch equals per-window stack; malformed, 2-D, overlong and non-finite inputs still raise structured `SpanFeatureTransformError` rather than raw numpy errors; 4-D batch, bare-window equality, determinism, frozen state and dimension pins all hold.

**V1 failure retained** (`SPAN_FEATURE_TRANSFORMS_REVIEW_V1.md` unchanged), including its acknowledged combined-fit counter and unreachable `PCA_SAMPLES` note.

21/21 tests PASS, also under `-W error`.
