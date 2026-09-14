# Practical feature priorities and prospective comparison

Ranking is a study-specific hypothesis balancing expected utility, sample size and cost, not an established performance ordering. Current features already use the last shared pre-option token at block10, with training-mean centering then rowL2.

1. L2-normalized activation: cheap control of scale; test uncentered L2 separately from existing centeredL2.
2. Raw residual activation: preserve information discarded by normalization.
3. PCA components: reduce dimensionality with only240training cases; fix32components and learn within each training fold.
4. Multiple label-derived directions/subspace: test three TRAIN-only mean contrasts (SELF minus each negative class), using cosine features.
5. Activation magnitude together with normalized activation: test whether magnitude adds predictive information.
6. Last-token plus mean/max pooled activations: promising additional context, needs new token-level capture for all320development cases.
7. Multi-layer concatenation: may capture information absent from block10, but increases dimension and needs additional capture.
8. Mean pooling over relevant tokens: potential robustness to token choice, with a defined mask excluding options/labels.
9. Layer change: may isolate computation added by the block; needs adjacent-layer capture.
10. LayerNorm/RMSNorm-normalized activation: depends on actual trained normalization location/weights; not interchangeable with arbitrary vector normalization.
11. Last-token activation: already the extraction position used here; not an independent new representation.
12. Cosine with one learned SELF direction: cheap but more restrictive than multiple contrasts.
13. Projection onto one learned SELF direction: likewise restrictive, additionally sensitive to scale.
14. Layer-wise trajectory features: interesting but definition and additional capture are needed.
15. Max pooling: may emphasize isolated outliers; requires token-level capture.
16. SAE features: needs a compatible trained SAE or a separate training effort; disproportionate for this quick comparison.

Run the top5cached transformations plus existing centeredL2 baseline on the doubled240TRAIN/80VALIDATION data. Define fifth feature as centeredL2 concatenated with TRAIN-standardized log rawL2norm. PCA32 uses whitened components; directions are normalized mean contrasts and cosine against raw activations. These directions predict external scenario labels, not validated motivation.

For each representation repeat binary and four-class logistic C0.01/0.1/1/10 and XGBoost depth1/2/3 with unchanged100rounds/0.05learning rate/minchild3/lambda5. Five group-disjoint folds. Maximum420classifier CVfits and24full-TRAIN refits across all6representations (30fold preprocessing fits). This supersedes only the prospective not-yet-started74fit expansion-only plan by explicit user feature-engineering request; no running or historical cap is changed. Threshold grid0.05..0.95. Rank by TRAIN OOF min(precision,recall), thenF1, thenlowerC/depth, thresholdnearest0.5/lowerthreshold. Choose one global candidate using TRAIN OOF only, with representation order as final tie-break. No model outcomes used for new-case admission.

Score all candidates on original40validation, added40, combined80 and training; separately score the four saved original models on the expandedvalidation for same-set comparison. Report improvements and regressions; validation is repeatedly used development data. HOLDOUT192 remains untouched. Save all predictions, per-class confusions and macroF1, learned transforms and classifiers. Verify selected models by save/reload equality.

Training-only preprocessing follows scikit-learn guidance: https://scikit-learn.org/stable/common_pitfalls.html . SAE complexity/interpretability limitations: https://openai.com/index/extracting-concepts-from-gpt-4/ .
