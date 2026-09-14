# Expansion Results Review V1

Verdict: ISSUES_FOUND (scoped metrics, not pipeline).

Cross-checked `development_results.json` against original saved models on identical splits.

Genuine, correctly scoped finding: unit_l2__xgboost__multiclass beats the original 120-TRAIN xgboost multiclass on combined-80 self_gate F1 (0.634 vs 0.586). Both 240-TRAIN selections regress on original-40 (best 0.526 vs 0.609). The headline improvement therefore holds only on the synthetic-added subset (added-40 0.727), not on original-40.

Decisive caveat: global TRAIN-OOF selection chose raw__logistic__multiclass (OOF minPR 0.800), which scores self_gate F1 0.444 combined and 0.0 original — worse than both baselines on both subsets. Validation-best (unit_l2) was not selected; the reported .634 describes a non-selected candidate.

No implementation error found:
- Transforms (centering, PCA, cosine directions, norm stats) fit on training rows only; full transforms on 240 TRAIN only.
- Label remapping in `_fold_probabilities` packs columns in CANON order consistently with `argmax_labels`; SELF index correct.
- Threshold/OOF assembly and baseline reuse of saved center/threshold reproduce originals (xgboost multiclass original-40 0.609, 0.586 combined).

Caveats:
- validation_added and half of the 240 TRAIN are synthetic augmentation; combined-80 gains are partly synthetic.
- Validation repeatedly reused; no independent label audit; not motivation evidence.
- Feature ranking is unstable: unit_l2 logistic 0.84 added vs 0.18 original; three_cosine top on original-40. Ranking claim is subset-dependent.
- Seven groups over five folds, split 66/66/36/36/36 — imbalance makes OOF selection noisy.
