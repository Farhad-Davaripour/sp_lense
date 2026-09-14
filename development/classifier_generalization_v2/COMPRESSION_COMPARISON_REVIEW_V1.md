# Compression comparison core V1 — independent review

Job compression_core_review_20260914_1455. Verdict: **PASS_SCOPED**.

Pins verified: core `c2a4e34e1a3b9b1a20c963e217e264b9497d8e11dbd77e36cd61ece11ad5112e`; test `a61a7ce1dd53fe6f6cc5e9ea44e2f8d9d8411d013c16b519678f39c76bc226d4`. Root 10 synthetic tests pass under `.runtime` (10/10 OK); factory is toy-only.

Independently confirmed by inspection:
- `PCA32` uses exact full SVD, `whiten=False`, fit only on TRAIN fold rows; one full-TRAIN PCA per condition on `train_ids`; first 8/16/32 components of the same fit sliced; no validation row enters any fit.
- Dimensionalities: unprompted pca8/16/32; prompted pca8/16/32 + full3072.
- Both conditions share `case_order`, labels, groups, folds; optional condition metadata must equal the shared metadata.
- Unprompted full3072 both families reused as cited reference; `reference_fits=0`, never refit; admitted to ranking.
- Exactly 210 CV + 14 refits + 12 shared PCA fits (all-valid case), limits enforced.
- Selection TRAIN-OOF only: min(P,R), F1, lower width/C, binary, |tau−0.5|, smaller tau, unprompted tie.
- Failed/non-converged/non-finite fold voids the whole candidate; folds never pooled; failures kept.
- Refits use TRAIN rows only; reload proof reads model bytes back from disk and matches predictions plus PCA arrays.
- Exclusive outputs; finite input/feature checks; counters plus `failure.json`/`candidate_failures.json`.
- Reuses existing `harness` metrics, `driver._fold_probabilities`/`_default_factory`, `span._evaluate_split`; mirrors span's per-family gate ranking.
- No loader, pin, watch, capture, network, real data, provider, tokenizer, or real model fit.

Root-owned boundaries (not defects): authenticated matrices/references and pins, zero-fit check, external 600 s watch, output cap. Core adds no normalization, so per-layer-L2 inputs must be root-provided with the same reference.

Scoped non-blocking notes: global ranking holds 7 cell winners plus 2 reference entries (all candidates remain in `cv_scores.json`); budget asserts run after the fixed sweep; input-validation failures emit no `failure.json` since the output directory is created only after validation.
