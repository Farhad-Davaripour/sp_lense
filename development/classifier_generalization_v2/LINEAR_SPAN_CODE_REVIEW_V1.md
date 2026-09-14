# LINEAR_SPAN_CODE_REVIEW_V1

Job: linear_span_code_review_20260914_1303. Read-only source/test review after the root-integration fixes. No datasets, caches, vectors, results, holdout, provider, model, tokenizer, network, install, Git, source edits, or subagents were touched.

## Pins (recomputed)
- `linear_span_control_v1.py` SHA256 `dc7117923bc20e45e5c57e50d448550e6056d813b72cc1e7da08e30dcc7e28b3`
- `test_linear_span_control_v1.py` SHA256 `fccf0594aad9b0ed32fed2a0725278928c926c2a67cc8acf0003d20c53385683`
- `LINEAR_SPAN_CONTROL_PLAN_V1.md` SHA256 `5ba67310f893b52dcb5c769dc761be2b15e825c0a4832f85ed116701cd989c23`

## Verdict: PASS_SCOPED

Fixes verified: features are built once in canonical case order and refits use explicit TRAIN row subsets (`matrix_for(matrix, row_of, case_data['train_ids'])`), so a 320-row/240-label mismatch cannot recur; the toy asserts `fit_rows[-2:] == [240, 240]`. Coeffs are checked finite, `ConvergenceWarning` is raised as an error in CV and refit, and `_configured` also rejects `n_iter_ >= max_iter`. Model bytes are hashed, reloaded from disk, and compared exactly (`RELOAD_MISMATCH`). Integer index/row maps keep duplicates, train-fold isolation (`FOLD_GROUP_LEAK`), 30 CV + 2 refit budgets, TRAIN-only C/tau selection, binary/four-class mapping, no validation selection, and exclusive outputs all coherent. Failure counters survive via `run`'s `except BaseException` + `failure.json`; a failed refit raises `REFIT_FAILED` without retry.

No blocking defect found. Non-blocking: a `_prepare`/evaluate failure after a successful refit increments `refits` but not `refit_errors` (evidence still preserved, no wrong fit); `LINEAR_SPAN_CONTROL_HANDOFF_V1.md` says six tests while seven are defined (cosmetic).

Synthetic-only tests; real runtime/source pins, zero-fit load, 300s watch, 256MiB cap remain root-owned.
