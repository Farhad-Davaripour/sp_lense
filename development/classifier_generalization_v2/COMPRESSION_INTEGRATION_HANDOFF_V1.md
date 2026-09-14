# Compression integration handoff V1

Job `compression_integration_20260914_takeover_v1`. Model-free integration for `PROMPTED_COMPRESSION_PLAN_V1.md`.

Files: `compression_integration_v1.py` (versioned loader + wrapper), `run_compression_supervised_v1.py` (root CLI), `test_compression_integration_v1.py` (8 synthetic tests, all pass under `.runtime`).

Exact CLI, run from `development/classifier_generalization_v2`:

```
./.runtime/Scripts/python.exe run_compression_supervised_v1.py prepare \
  --prompted-lock RUN_LOCK_PROMPTED_CAPTURE_V1.json --run-id compression_supervised_20260914_v1
./.runtime/Scripts/python.exe run_compression_supervised_v1.py preflight \
  --plan COMPRESSION_INTEGRATION_FIT_PLAN_V1.json --sha256 <PLAN_SHA>
./.runtime/Scripts/python.exe run_compression_supervised_v1.py run \
  --plan COMPRESSION_INTEGRATION_FIT_PLAN_V1.json --sha256 <PLAN_SHA> \
  --review COMPRESSION_INTEGRATION_REVIEW_V1.md
```

`prepare` writes the unauthorised plan: prospective source hashes, runtime versions, the frozen `SPAN_FIT_PLAN_V2.json`, the `runs/linear_span_20260914_v1` files, and prompted lock/index/windows/receipt/success pins. `preflight` authenticates every pin plus the query fingerprint, every case/order/block and last-token provenance against the prompted lock, takes labels/groups/folds from the same admitted manifests, builds per-layer-L2 pair-averaged 3072 matrices and the cited unprompted full baseline, and performs zero fits. `run` requires `pass_scoped` in the review, releases, spawns the watched worker (`native.watch`, 600 s) and enforces the 256 MiB output cap; the worker records 210 CV + 14 refits + <=12 PCA.

Unresolved root-owned checks:

1. Complete/authorize the prompted capture lock and artifacts; `prepare` reads the finished `runs/<lock run_id>`.
2. Author the independent review containing `pass_scoped` before release.
3. Confirm tokenizer-sanity `adapter_sha256` equals the pinned `prompted_input_adapter_v1.py` hash and measured max tokens fit the lock `tokens_per_view` (provisional 512 vs frozen measured bound).
4. Confirm the frozen span plan/capture and linear-span baseline hashes are the admitted evidence; never recapture or refit.
5. Keep HOLDOUT sealed, load no model/tokenizer, and leave core/old runners unchanged.
