# Identity fit handoff V1

Job `identity_fit_pipeline_20260914_v1`. Added only `identity_fit_v1.py`,
`run_identity_fit_v1.py`, `test_identity_fit_v1.py`, `IDENTITY_FIT_PLAN_V1.json`,
`IDENTITY_FIT_HANDOFF_V1.md`. No old file, case, lock or run was edited. The active
`identity_capture_20260914_v1` run and its partial vectors were never read.

`identity_fit_v1.py` reuses `span_classifier_driver_v1` (pins, manifests, index
decode, `_evaluate_split`), `linear_span_control_v1` (per-layer L2 last-token
concat 3072), `compression_comparison_v2.PCA32` (full SVD, whiten false),
`harness` metrics and `grouped_driver._default_factory`. It re-authenticates the
identity lock/receipt/index/window/query/case/order/block/raw-prefix/source pins
and the same 240/40/40 labels, groups and folds as `compression_supervised_20260914_v2`.
New condition: binary C=10, 5 grouped CV + 1 full-TRAIN refit (6 fits) and 6 PCA
fits; 19-threshold TRAIN-OOF choice is max F1, max min(P,R), |tau-0.5|, smaller
tau. Old prompted pca32 C10 is reused from its saved OOF and prediction JSON only
(0 refits, 0 PCA, no pickle load) and re-thresholded by the current rule (its
saved tau 0.35 is reported, not assumed). Reports train/original40/added40/combined80
confusion, negative-class FP, order consistency, retained variance and matched
control baseline; no holdout. Output is exclusive, failure-preserving, 64 MiB,
60 seconds, one child watch.

CLI (`.runtime/Scripts/python.exe`, cwd study dir):
`run_identity_fit_v1.py prepare|preflight|run|worker --plan IDENTITY_FIT_PLAN_V1.json --sha256 <sha>`
(`run` also `--review <pass_scoped review>`).

Pins: lock sha `4f68a210b95a20ec5c322672943e2f3e5ea339ed60f8f0fbc5e17aba498d5902`;
query sha `7d2db4fb340fc2f595233cfff9d9e3f7c01b0a97d4e82385d977d479ab189eec`;
schemas `identity_span_development_execution.v1`,
`identity_span_capture_index.v1`, `identity_span_capture_receipt.v1`,
`identity_span_capture_windows.v1`; 1 model, 640 views, 0 fits.

Plan stays unauthorized with unresolved identity-output/source pins. Root steps:
after capture completes run `prepare`, obtain an independent review admitting
`pass_scoped` with the plan sha, run zero-fit `preflight`, then `run`; confirm
worker counters 6 fits/6 PCA and read `comparison.json`.
