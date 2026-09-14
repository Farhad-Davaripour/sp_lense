# Linear span control V1 — handoff

Job `linear_span_implementation_20260914_1233`. Two new files, plus this note.

**`linear_span_control_v1.py`** — `run(case_data, output_dir, *, factory=None, deadline=None)`.
Takes already-authenticated `case_data` from `span_classifier_driver_v1.load_case_data`
plus a fresh directory; no loader, source-audit, runner or supervisor rewrite.
Features: last token of blocks 6/10/18 of the pair-averaged windows, each 1024 vector
L2-normalized independently (zero→zero), concatenated to 3072; no learned transform.
Binary/four-class `LogisticRegression` via `grouped_driver._default_factory` at
C∈{0.1,1,10}, exactly 5 grouped folds = 30 CV attempts + ≤2 family refits. Any
failed/nonconverged/non-finite fold voids that whole C/family candidate; folds are never
pooled. Thresholds 0.05…0.95 ranked on TRAIN OOF only (minPR, F1, lower C, binary-first,
tau near 0.5, then smaller tau). Evaluation reuses
`span_classifier_driver_v1._evaluate_split`; exclusive outputs include `cv_scores.json`,
`family_results.json`, per-family predictions/artifacts, `feature_definition.json` and an
exact disk-reload proof.

**`test_linear_span_control_v1.py`** — six synthetic tests (toy factory, no real fit):
30+2 counts, fold isolation, normalization/zero, ranking, whole-candidate failure, and
exclusive-output/reload. All pass under the repo `.runtime`.

Line aims were not met: core 320 vs 220, tests 179 vs 160 (handoff 165 words ≤ 200).
Root still owns plan/source/runtime checks, the 300 s watch and the 256 MiB cap.
