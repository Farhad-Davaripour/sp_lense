# Span feature transforms V1 handoff

Job ID: span_features_implementation_20260914_1054

Added only `span_feature_transforms_v1.py`, `test_span_feature_transforms_v1.py`
and this handoff. No existing source, trainer, runner, writer or config was
edited.

`SpanFeatureTransforms.fit(windows, labels)` learns F4 class contrasts and F5
PCA/standardization from the supplied TRAIN rows only. `transform(windows)`
never refits and never mutates state. Inputs are sequences of logical-case
windows shaped `(3, n, 1024)` (blocks 6,10,18; `1<=n<=16`; no padded rows);
AB/BA averaging remains the caller's job. Output is the concatenated
`F1(3072)+F2(3072)+F3(2048)+F4(36)+F5(44) = 10224` matrix.

F4 uses per-layer TRAIN `SELF - class` unit directions (3 others), projected
token scores summarized as mean, p90 (linear), top-`ceil(0.1*n)` mean and
population std (ddof=0). F5 uses deterministic `PCA(8, full SVD)` on F1,
train-population standardization, then 8 squares and 28 lexicographic pairwise
products. Zero-norm directions store zeros; zero-variance components get scale
1.0, so no statistic is NaN/inf; finiteness is rechecked after interactions.

Validation rejects bad shapes/widths/lengths, non-finite values, unknown or
missing classes, class support below 2, and fewer than 8 PCA rows.

Test command (new file only):

    development\classifier_generalization_v2\.runtime\Scripts\python.exe -W error test_span_feature_transforms_v1.py -v

Result: 19 tests, OK (0.32s), synthetic numpy-only fakes; no real data, model,
provider, holdout, results, network, install, Git or coordination. Scope is the
transform code/test only, not the pipeline.
