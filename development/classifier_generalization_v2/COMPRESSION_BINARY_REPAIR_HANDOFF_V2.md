# Compression binary repair handoff V2

Repairs failed root attempt `compression_supervised_20260914_v1` (child COMPLETE, yet all 105 binary CV fits failed with `binary estimator classes must be exactly 0 and 1`; 210 CV attempts, 7 four-class refits, 12 PCA fits). Cause: `compression_comparison_v1._cross_validate` and `_refit` passed four string labels to `estimator.fit`. V1 and every failed result/cap are untouched: v1 hashes still equal the failed run's released-plan pins (`compression_comparison_v1.py c2a4e34e…`, `compression_integration_v1.py 3bbc546b…`, `run_compression_supervised_v1.py d8945730…`). No live V3/V4 plan, review, or real run was created.

V1→V2 differences:
- `compression_comparison_v2.py`: binary `fit` targets now `1 if label=="SELF" else 0` at both the CV and full-TRAIN refit sites; four-class keeps canonical strings; JOB_ID versioned. No other change.
- `compression_integration_v2.py`: imports core v2; JOB_ID/PLAN_SCHEMA v2; SOURCE_FILES pin v2 modules and retain `native_development_runner_v2.py`; RUNTIME_PACKAGES retain `threadpoolctl`; `_verify_result` makes refits a maximum (`<=14`), never padded or refit to 14.
- `run_compression_supervised_v2.py`: imports v2; default run id `compression_supervised_20260914_v2`; plans `COMPRESSION_INTEGRATION_FIT_PLAN_V3.json`/`_V4.json`; reviewer `COMPRESSION_BINARY_REPAIR_REVIEW_V2.md`; capped refit assertion.

New full SHA256:
- `compression_comparison_v2.py` `2c2c8739bb4f4a5a6dab02fe443ea58ebb51ab3b573b975d11cb21fb5cd55eef`
- `compression_integration_v2.py` `e580f851cb8c40014b2c464224f3b45466eb4725457566ce97ad95a99e762200`
- `run_compression_supervised_v2.py` `47fbf2c745a0caaaaf7132be54d750f3b2ffa45e7942ef24cc123cc764cf023f`
- `test_compression_binary_repair_v2.py` `6ed0bddcea01204f59097de87f1adc1f0974bafb63d39732047fd4e6ca6e4bcf`

Tests, from `development/classifier_generalization_v2`: `./.runtime/Scripts/python.exe -m unittest test_compression_binary_repair_v2 test_compression_comparison_v1 test_compression_integration_v1` → 24 OK (23.9 s). The new suite records CV and refit targets (binary integer 0/1, four-class canonical strings), performs exactly two real `grouped_driver._default_factory` toy fits on tiny arrays (binary 0/1 accepted; four-class strings accepted and reproduce V1's binary-path rejection), and shows a binary-invalid run refits 7 and passes the capped integration check. The prior synthetic suite is reused unmodified.

Root CLI: `./.runtime/Scripts/python.exe run_compression_supervised_v2.py prepare --prompted-lock RUN_LOCK_PROMPTED_CAPTURE_V1.json --run-id compression_supervised_20260914_v2`; then `preflight --plan COMPRESSION_INTEGRATION_FIT_PLAN_V3.json --sha256 <SHA>`; then `run --plan COMPRESSION_INTEGRATION_FIT_PLAN_V3.json --sha256 <SHA> --review COMPRESSION_BINARY_REPAIR_REVIEW_V2.md`.

No scientific attempt until independent review and a new finite lock.
