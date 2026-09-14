# Cohort-transfer diagnostic — independent protocol review (V1)

Job: `cohort_transfer_review_20260914_1315`

Verdict: **PASS_SCOPED.**

The design genuinely discriminates a descriptive composition-sensitivity contrast: same cached features, loader, LR/grid, fixed 7-group/5-fold assignments, equal 120/120 class-balanced arms, and one shared 80-case evaluation leave training-cohort as the only manipulated factor. The identifying contrasts are matched pairs holding eval fixed — original-trained vs added-trained on original40, and on added40 (plus interaction); within-arm train→eval cohort gaps confound both cohorts and are not cohort effects.

Binding scope conditions:

1. TRAIN cohort labels are not exposed by `load_case_data` (only validation `original_ids`/`added_ids`); derive them from authenticated `case_data["manifest"]["cases"][cid]["cohort"]` (`original_train`/`added_train` roles). Zero-fit check: 30/class/arm, all 7 groups, all 5 folds, group-disjoint, identical arm fold→group map, both arms disjoint from the 80.
2. Report family-matched endpoints (binary self-gate; fourclass separately). If arms select different winner/C/tau, do not compare winner-to-winner; disclose per-arm C/tau so composition is not silently conflated with hyperparameter selection.
3. One deterministic refit/cell and no bootstrap ⇒ point estimates only: descriptive, not significance, not causal author/normalization effects.
4. The 240 baseline is size-confounded (120 vs 240); report separately.
5. Core embeds `JOB_ID = linear_span_implementation_20260914_1233` in every pickle/result; the wrapper must override the module constant at runtime so new arms do not inherit the old run's provenance.
6. One shared 300 s deadline and 256 MiB aggregate cap, wrapper-enforced.
