# IDENTITY_FIT_REVIEW_V1_V2

VERDICT: PASS_SCOPED

- Review job: `identity_fit_review_20260914_v2` (independent re-review after the reviewer-mandated
  test fix). Supersedes the BLOCKED `IDENTITY_FIT_REVIEW_V1.md`, which is preserved untouched.
- Reviewed artifact: `development/classifier_generalization_v2/IDENTITY_FIT_PLAN_V1.json`
- Reviewed plan sha256 (independently recomputed): `8359cb8cc03b10078f60635654ffa9baa7faedb83c5796df804fa38d3e7f9838`
- Plan size 5439 bytes; `pin_status=RESOLVED`, `status=RESOLVED_PENDING_REVIEW`,
  `scientific_execution_authorized=false`.
- Scope: this approval admits the zero-fit `preflight` and the synthetic suite only. It is scoped to
  the reviewed plan sha above; it does NOT by itself authorize the scientific `run`/`worker` fit.

## No-access statement

No real scientific fit, no model, no tokenizer, no Qwen weights and no holdout/private corpus was
accessed. This re-review never ran the `run` or `worker` mode and never invoked `run_identity_fit_v1.py`
with any mode other than `preflight`. It made no network, install, Git-write, config or coordination
change; it loaded no pickle/joblib artifact. The only state-changing side effects were: (a) the
permitted zero-fit `preflight` (which authenticated pins and built features but instantiated no
estimator/PCA), (b) the synthetic suite, whose synthetic `compare` runs wrote only to ephemeral
`%TEMP%\identity_fit_*` directories that were removed on teardown, and (c) this review file. The sole
write under `development/classifier_generalization_v2/` is this file. All interpreters used were the
pinned `development/classifier_generalization_v2/.runtime/Scripts/python.exe`.

Commands actually run (cwd `development/classifier_generalization_v2` unless noted):
`run_identity_fit_v1.py preflight --plan IDENTITY_FIT_PLAN_V1.json --sha256 8359...f9838` and
`test_identity_fit_v1.py -v`; plus read-only SHA-256/JSON/metadata/bytecode recomputation.

## Item 1 - resolved plan pins, runtime and source hashes: PASS

- Plan sha256 recomputed = `8359cb8cc03b10078f60635654ffa9baa7faedb83c5796df804fa38d3e7f9838` (match).
- All 12 evidence pins are resolved (non-null path + 64-hex sha) and every recomputed file hash/size
  matches the pin: identity lock/index/windows/receipt/success, blueprint, original/added TRAIN,
  original/added VALIDATION, old `cv_scores`, old `model_results` -> **12/12 resolved and hash-match**.
  Per-pin recomputation:
  `lock 5081B 4f68a210...98d5902`, `index 1543284B 65eee8db...f055590d`,
  `windows 125829120B 2251d615...c86168c4`, `receipt 4978B 783293ee...f0b4eab7`,
  `success 600B 77992d97...c74b98c9`, `added_train 142657B beae36eb...85f5db83`,
  `added_validation 49725B 1a34cf78...209ccf`, `blueprint 12140B 46c0f5ab...72cf4593`,
  `original_train 188753B 60a9ee7b...ebe52fb9`, `original_validation 65706B 0bf23931...616fd617`,
  `old cv_scores 486694B bccdd583...a1c7565`, `old model_results 67514B 5724a568...bfc666f1`.
- Runtime from the pinned `.runtime` interpreter (Python 3.12.14): numpy `2.5.3`, scipy `1.18.1`,
  scikit-learn `1.9.1`, threadpoolctl `3.6.0` - exactly the plan's `runtime_packages`.
- All 11 plan `source_files` recomputed with SHA-256 and match the worktree: **11/11**.
  `identity_fit_v1.py = 9ff28c6a26aee690d94f3f0cb53c09ded408b6ac082e6ebe6c40d4babc5271f5` (identical to
  the BLOCKED review's pinned value -> production loader unchanged by the fix),
  `run_identity_fit_v1.py = 765e1646ddf5483af53f3b127830ebd3d096dba7527a36905fa640e4fe0d58ce`,
  `compression_comparison_v2.py = 2c2c8739...5cd55eef`,
  `linear_span_control_v1.py = dc711792...e28b3`,
  `span_classifier_driver_v1.py = 69b42fff...5acbebf`.

## Item 2 - zero-fit preflight (independently re-run): PASS

Exact output of the required command:
`status=preflight_pass, classifier_fits=0, pca_fits=0, conditions=[prompted, identity_fixed_query_v1],
identity_rows=640, train=240, original=40, added=40, validation=80,
query_sha256=7d2db4fb340fc2f595233cfff9d9e3f7c01b0a97d4e82385d977d479ab189eec,
old_control_saved_tau=0.35, holdout_accessed=false`, plus `cases=320`, `feature_width=3072`,
`job_id=identity_fit_pipeline_20260914_v1`, `run_id=identity_fit_20260914_v1`,
`identity_run_id=identity_capture_20260914_v1`,
`old_control_run_id=compression_supervised_20260914_v2`. Every requested field matches exactly; exit 0;
zero estimator and zero PCA fits.

## Item 3 - OLD prompted control reuse: PASS

- `identity_fit_v1.py:95 OLD_CONTROL_CONDITION = "prompted"` (literal). Plan `old_control.condition =
  "prompted"`, dimension `pca32`, family `binary`, C `10.0`, artifact
  `.../model_results_prompted__pca32__binary.json`.
- Recomputed from `runs/compression_supervised_20260914_v2/cv_scores.json` (sha
  `bccdd583...a1c7565`, 42 candidates): exactly **1** valid candidate with condition `prompted`,
  dimension `pca32`, family `binary`, C `10` (`selected_tau=0.35`, OOF length 240); and
  `prompted_fixed_query_v1` matches **0**. Exactly **5** fold records are tagged `prompted`
  (folds 0-4; the other 5 are `unprompted`).
- Recomputed from `model_results_prompted__pca32__binary.json` (sha `5724a568...bfc666f1`): condition
  `prompted`, dimension `pca32`, family `binary`, C `10`, status `FITTED`, `reload_exact=true`,
  evaluation splits `train, original40, added40, combined80`.
- `load_old_control` reads JSON only: neither `identity_fit_v1.py` nor `run_identity_fit_v1.py`
  contains `import pickle` or `joblib`; the estimator pickle referenced by the saved `artifact` field is
  never opened. Emitted `pickle_loaded=false`, `classifier_fits=0`, `pca_fits=0`, `refits=0`,
  and `old_cv_fits=old_refits=old_pca_fits=0`.

## Item 4 - threshold rule identical for both conditions: PASS

- 19 thresholds, `span.TAUS = 0.05..0.95` step 0.05; plan `thresholds` equals `span.TAUS`.
- `select_tau` key `(-F1, -min(P,R), |tau-0.5|, tau)` = max F1, then max min(precision,recall), then
  closest to 0.5, then smaller tau (`identity_fit_v1.py:521-542`). Both conditions call the same
  function: identity on its TRAIN-OOF, and the control via `_old_condition` on the saved TRAIN-OOF.
- Recomputed on the real saved control OOF, the current rule selects **tau=0.3** (max eligible
  F1=0.6667; eligible 0.05..0.65), while the plan's saved value **0.35** is reported separately, not
  assumed. Recomputed control table: 0.3 -> P=0.5632/R=0.8167/F1=0.6667; 0.35 -> P=0.6308/R=0.6833/
  F1=0.6560; 0.5 -> P=0.85/R=0.2833; higher taus are ineligible at F1=0.
- Emission verified on the synthetic fixtures: `matched_control_baseline.saved_tau` and
  `old_control.json.saved_tau` carry the reported 0.35 while `current_rule_tau` carries the rule's
  recomputed value. Selection is TRAIN-OOF only; `validation_used_for_selection=false` and
  `holdout_accessed=false` throughout; no validation/holdout threshold selection anywhere.

## Item 5 - fit budget, representation and no-holdout: PASS

- 6 classifier fits: 5 grouped CV (`_identity_cv`) + 1 full-TRAIN refit (`_identity_condition`);
  6 PCA32 fits: 5 fold + 1 full-TRAIN. The end-to-end synthetic `compare` emitted counters
  `{'cv_fits':5,'refits':1,'pca_fits':6,'old_cv_fits':0,'old_refits':0,'old_pca_fits':0}` and the
  toy factory observed exactly 6 fits.
- Family `binary`, C `10.0`; factory `LogisticRegression(C=C, l1_ratio=0, solver="lbfgs",
  max_iter=1000, tol=1e-4, class_weight=None, random_state=0)` (`grouped_driver.py:34-37`).
  PCA32 is exact full SVD, `svd_solver="full"`, `whiten=False`, `n_components=32`
  (`compression_comparison_v2.py:123-150`), fit on TRAIN rows only; retained `n_components=32`
  emitted by the synthetic compare.
- Features are the per-layer L2 last-token concatenation: `linear.FEATURE_DIM = 3*1024 = 3072`;
  matrix shape `(320, 3072)` verified in preflight. All 6 PCA fits are on TRAIN folds/TRAIN only.
- No holdout/private read: the fit module references holdout only as `False` flags
  (`identity_fit_v1.py:816,817,851,882,907,938,950`); no holdout path or reader exists.

## Item 6 - reporting, failure preservation, caps and supervision: PASS

- Reporting covers `train`, `original40`, `added40`, `combined80` with `self_gate` confusion metrics,
  per-negative-class `negative_class_false_positives` (keys `OTHER`, `NONTERMINATION`, `ORDINARY`),
  predictions, and (combined80) `answer_order_consistency =
  gate_agreements/total/max_probability_difference`. A matched control baseline (saved tau, saved
  metrics, current-rule tau/metrics, artifact hashes, `pickle_loaded=false`), ranking/winner and
  identity retained variance are emitted.
- Failure preservation: `compare` writes `failure.json` on exception; `run` writes
  `supervisor_failure.json`; the output directory is exclusive (`OUTPUT_EXISTS`). Confirmed by the
  synthetic failure test.
- Caps: `SECONDS=60`, `OUTPUT_BYTES=64 MiB (67108864)`, validated by `load_plan` and enforced by
  `run`/`worker` (`_verify_result`). Exactly one watched child: `run` calls `native.watch` once, which
  spawns one `Popen` and enforces the hard deadline. The runner pins its own supervisor source:
  `run` sets `supervisor_source_sha256=sha256(run_identity_fit_v1.py)` and `worker` asserts it.

## Item 7 - synthetic suite and the single delta: PASS (13/13)

- Command: `.runtime/Scripts/python.exe test_identity_fit_v1.py -v` (cwd study folder).
  Result: `Ran 13 tests ... OK` (13/13 pass, exit 0), including
  `test_delivered_plan_is_well_formed` and the real-sklearn TOY fit
  `test_real_sklearn_default_factory_binary_classes` (integer `classes_ == [0,1]`, `predict_proba`
  shape `(8,2)`, rows sum to 1, `predict == y`).
- The only delta from the BLOCKED review is the one-line test assertion, independently proven rather
  than asserted. The stale bytecode cache `__pycache__/test_identity_fit_v1.cpython-312.pyc`
  (header mtime 17:23:16, before the source mtime 17:34:23) still holds the pre-fix module. A
  line-number-free semantic comparison of every code object in that pre-fix bytecode against the
  current `test_identity_fit_v1.py` shows:
  - code-object name sets are identical (no test added/removed); all 15 old method firstline numbers
    are unchanged or shifted by exactly the one inserted line;
  - exactly **one** function differs semantically: `test_delivered_plan_is_well_formed`, and its only
    differing instruction is `LOAD_CONST 'UNRESOLVED'` -> `LOAD_CONST 'RESOLVED'`.
- `identity_fit_v1.py` hash is unchanged (`9ff28c6a...5271f5`), and the plan hash still matches while
  it pins `identity_fit_v1.py` and `run_identity_fit_v1.py`; therefore no production code changed.
  The fix is test-only, as claimed.

## Remaining blocker

None. The item-7 blocker from `IDENTITY_FIT_REVIEW_V1.md` is resolved by the one-line test change, and
no new blocker was found.

## Non-blocking observation (carried from V1, unchanged)

`identity_input_adapter_v1.py:31 CONTROL_CONDITION` still holds the stale
`"prompted_fixed_query_v1"`. It is unused by the fit path (only `tokens.IDENTITY_CONDITION` and the
literal `"prompted"` are used), so it cannot affect this pipeline; recommend aligning or removing it
in a later, separately reviewed change.

## Release token

Because the verdict is PASS_SCOPED, the lowercase token `pass_scoped` together with the plan sha
`8359cb8cc03b10078f60635654ffa9baa7faedb83c5796df804fa38d3e7f9838` is granted here, which is exactly
what `run_identity_fit_v1.py run --review <this file>` requires to admit the release. The approval is
scoped to this reviewed plan sha and to the zero-fit preflight + synthetic suite evidence above; the
scientific fit itself still runs nothing here.

## Explicit no-real-fit / no-model / no-holdout statement

No real fit was executed: `classifier_fits=0` and `pca_fits=0` in the only production mode run
(`preflight`), and all estimator/PCA activity observed came from synthetic fixtures in `%TEMP%`. No
model or tokenizer was loaded and no Qwen/transformer weights were touched. No holdout or private
corpus was read (`holdout_accessed=false` in preflight and in every emitted artifact), and no pickle
was loaded. This re-review is read-only except for this single file
`development/classifier_generalization_v2/IDENTITY_FIT_REVIEW_V1_V2.md`.
