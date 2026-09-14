# Compression integration review V1

Job `compression_integration_review_20260914_v1`. Verdict: **PASS_SCOPED**.

Native SHA256 (verified by `Get-FileHash -Algorithm SHA256`, all matching the requested digests):

- `compression_integration_v1.py` `58cd25e8a3d6fc9ec526c41c87db7210b2a1e1d4f62183baa199fcdf2679b45b`
- `run_compression_supervised_v1.py` `d8945730f14e7765989d70707cac2f15103e7e255f86be2cd9695727847022c8`
- `test_compression_integration_v1.py` `016921df176f9acf36d55f0dce208d4629eaed4cbcdd5babed1e6cb721796aad`

Reviewed all three plus `PROMPTED_COMPRESSION_PLAN_V1.md`, `COMPRESSION_INTEGRATION_HANDOFF_V1.md`, and reused helpers (`compression_comparison_v1`, `span_classifier_driver_v1`, `linear_span_control_v1`, `prompted_input_adapter_v1`, `native_capture_contract`, `grouped_driver`, `harness`, `prompted_span_runner_v1`, `native_development_runner_v2`, `SPAN_FIT_PLAN_V2.json`, `RUN_LOCK_PROMPTED_CAPTURE_V1.json`).

Verified: fixed query string/`QUERY_SHA256` match the plan; index/lock/receipt/success hash-bound; all 640×3 case/order/block records present with `prefix_length==readout_index+1` and prefix within `tokens_per_view`; sanity rows cover every case/order, `last_shared_token_id==198`, prefix and input hashes cross-bind to index records; `adapter_sha256` equals the pinned adapter pin; counts tokenizer_loads=1/model_loads=0/forwards=0/fits=0. Source pins include the executing entrypoint and every imported scientific module; runtime pins numpy/scipy/scikit-learn; span plan, baseline files, prompted lock/index/windows/receipt/success and prompt_sanity are hash pins. Data are 240 TRAIN/40+40 VALIDATION with identical case sets, labels, groups and folds across conditions; features are per-layer L2 last-token concat over pair-averaged AB/BA windows (3072). The unprompted full-3072 baseline OOF ids/truth and both model artifact hashes are authenticated and cited, never fit (`reference_fits==0`); PCA and selection are TRAIN-only; 210 CV+14 refits; failure counters kept; exclusive output; 600 s watch; 256 MiB cap; holdout excluded; owner/native-module gates present.

Synthetic evidence under `.runtime/Scripts/python.exe` with injected toy estimators and no real data/model/pickle read or real classifier fit: shipped suite 8/8 pass (34.6 s). Independent probes confirmed zero PCA/estimator construction in preflight and reference assembly, and that `PROMPTED_BLOCK_PREFIX_MISMATCH`, `PROMPTED_SANITY_TOKEN_CAP`, `PROMPTED_SANITY_INPUT`, `PROMPTED_SANITY_LAST_TOKEN`, `CONDITION_META_MISMATCH` and `BASELINE_OOF_TRUTH` all fire.

No functional blocker. Non-blocking pin gaps: `native_development_runner_v2.py` (watch controller invoked by the pinned entrypoint) and `threadpoolctl` (work-thread limiter) are absent from the source/runtime pins; minimal fix is adding both. Scope caveat: the root prompted capture is unfinished (partial `windows.f32`, no index/receipt/success) with `runs/native_model_owner.json` active, so prepare/preflight/run cannot yet execute; it was not touched.
