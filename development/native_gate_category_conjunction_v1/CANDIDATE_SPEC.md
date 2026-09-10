# One training-only candidate: fixed negative-category conjunction

Status: proposal only, 2026-09-10. No implementation or execution approved.

Question: does separating the three prospectively defined OFF categories into
fixed affine constraints produce a certified pointwise gate that passes the
existing family-development checks? This changes representational capacity,
not the zero threshold or interpretation of previous failures.

Admissible data are exactly the previously admitted 44 block10 training views:
12 self, 12 other-shutdown, 12 nontermination and eight ordinary controls.
These are 26 underlying texts, not 44 independent scenarios. Authenticate
native_gate_hardmargin_coverage_v1/TRAINING_MANIFEST.json SHA256
4c0443844f096df7ffb6e23cb2bb236d5c146d66e339494d27d66f78e099df70,
combined feature digest
95e9311c1b371d2206017237ebe8e878178c6812de4b4c298d2d510d69a17719,
and both closed original32/new12 capture chains before future fitting.
G02 rows remain originally designated training data; their exposed held-out
scores are not labels, features or calibration targets. Exclude G07 entirely,
final23 captures, explicit-question arms, editor outputs, historical sealed
evaluation/confirmation data and all failed-artifact coefficients.

At each stage compute one unweighted mean from that stage's training rows,
then row-unit-normalize centered features. Fix three heads, in order:
self versus other-shutdown; self versus nontermination; self versus ordinary.
Each uses only those two classes, labels +1/-1, and minimizes one-half squared
weight norm subject to signed affine margin at least one; intercept unpenalized,
no slack, weighting or tuning. Reuse the existing solve/checker numerical bodies:
gate.py b9c439049bde05c8e1139f8afa8b754cce9a65506daed167e7c59e90550d108d;
checker.py 7cf81c280592f51d9d8024d6baca9dec4fdc7e6b607045f07165aed86749c4b9.
Supply the common transformed rows directly to solve; do not recenter by head.

Inference uses only one unedited final-original-input block10 output, width1024,
CPUfloat32/eager, from frozen Qwen3.5-0.8B revision
2fc06364715b967f1860aea9cf38778875588b17. Apply the frozen common transform and
all three affine scores: ON iff every score is strictly positive; zero is OFF.
No category/family labels, other inputs, batch mean, logits or gradients enter
inference. This is one intersection classifier, not selection among alternatives.

Retain G01-through-G06 fail-first 38/6 family folds and conditional FULL44.
Fold head sizes are 20/20/18; full sizes 24/24/20. Require independent KKT
certification, designated-head training correctness, joint training correctness
and every held view correct before advancing. All six passes permit FULL44
and exact artifact reload; otherwise later stages remain UNRUN. Full training
success is largely implied by the head constraints, not independent evidence.

Prospective construction maximum: 21 fixed solves, zero model/tokenizer calls;
one shared 60-second substantive plus five-second cleanup envelope, 8MiB total,
5MiB/file, 1MiB worker payload. Preserve binary64, 10,000 iterations/head,
linear/active/certificate tolerances 1e-10/1e-12/1e-8. Before release, synthetic
integration must prove exact serialization and all budgets. Invalid joins,
nonfinite/zero normalization, singular/uncertified solve or limit exhaustion
stop without fallback, cap increase or alternate candidate.

Comparison is the archived pooled-negative affine method; do not rerun or rescue
it. The pre-existing category taxonomy motivates this single capacity change,
not G02/G07 numeric correlations. Designers know exposed development outcomes;
even six fold passes are exploratory, not fresh generalization. Untouched
evaluation must be separately authored/admitted only after artifact freeze;
no currently exposed diagnostic becomes confirmation. No additional training
data are prerequisite; independent transfer evidence remains missing. Gate
success would still not prove natural bidirectional steering or preserved tasks.
