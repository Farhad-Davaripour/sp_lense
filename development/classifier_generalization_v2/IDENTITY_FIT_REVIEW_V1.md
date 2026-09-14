# IDENTITY_FIT_REVIEW_V1

VERDICT: BLOCKED

- Review job: `identity_fit_review_20260914_v1`
- Reviewed artifact: `development/classifier_generalization_v2/IDENTITY_FIT_PLAN_V1.json`
- Reviewed plan sha256 (recomputed): `8359cb8cc03b10078f60635654ffa9baa7faedb83c5796df804fa38d3e7f9838`
- Plan size: 5439 bytes (recomputed). `pin_status=RESOLVED`, `status=RESOLVED_PENDING_REVIEW`,
  `scientific_execution_authorized=false`.
- Verdict basis: required synthetic suite is 12/13, not 13/13 (Item 7). The delivered resolved
  plan can never satisfy `test_delivered_plan_is_well_formed`.

## No-access statement

No real scientific fit, model, tokenizer or holdout was accessed. This review loaded no Qwen
weights, no tokenizer and no transformer/model package; it never ran the `run`/`worker` mode; it
made no network, install, Git-write, config or coordination change; it read no holdout or private
corpus; and it loaded no pickle. Only the zero-fit `preflight` and the synthetic unit suite were
executed, under the pinned venv
`development/classifier_generalization_v2/.runtime/Scripts/python.exe`. The sole write is this file.

Commands actually run (cwd `development/classifier_generalization_v2`):
`run_identity_fit_v1.py preflight --plan IDENTITY_FIT_PLAN_V1.json --sha256 8359...f9838`
and `test_identity_fit_v1.py`; plus read-only SHA-256/JSON/metadata recomputation.

## Item 1 - resolved plan pins, runtime and source hashes: PASS

- Plan sha256 recomputed = `8359cb8cc03b10078f60635654ffa9baa7faedb83c5796df804fa38d3e7f9838` (match).
- All 12 evidence pins are resolved (path + 64-hex sha) and present: identity lock/index/windows/
  receipt/success, blueprint, original/added TRAIN, original/added VALIDATION, old `cv_scores` and
  old `model_results` -> 12/12.
- Runtime from the pinned `.runtime` interpreter: numpy `2.5.3`, scipy `1.18.1`, scikit-learn
  `1.9.1`, threadpoolctl `3.6.0` - exactly the plan's `runtime_packages`.
- All 11 plan `source_files` recomputed with SHA-256 and match the worktree: 11/11. Key pins:
  `identity_fit_v1.py=9ff28c6a...5271f5`, `run_identity_fit_v1.py=765e1646...e0d58ce`,
  `compression_comparison_v2.py=2c2c8739...5cd55eef`, `linear_span_control_v1.py=dc711792...e28b3`,
  `span_classifier_driver_v1.py=69b42fff...5acbebf`.
- Identity lock `RUN_LOCK_IDENTITY_CAPTURE_V1.json` sha `4f68a210...98d5902`, authorized true,
  caps `fits=0, derivatives=0, model_loads=1, tokenizer_loads=1, forwards=640, tokens_per_view=320`.

## Item 2 - zero-fit preflight (independently re-run): PASS

Exact output:
`status=preflight_pass, classifier_fits=0, pca_fits=0, conditions=[prompted, identity_fixed_query_v1],
identity_rows=640, train=240, original=40, added=40, validation=80,
query_sha256=7d2db4fb340fc2f595233cfff9d9e3f7c01b0a97d4e82385d977d479ab189eec,
old_control_saved_tau=0.35`, `feature_width=3072`, `cases=320`, `holdout_accessed=false`.

Every requested field matches exactly. Preflight ran with zero estimator and zero PCA fits.

## Item 3 - OLD prompted control reuse: PASS

- `identity_fit_v1.py:95` now reads `OLD_CONTROL_CONDITION = "prompted"` (corrected). The stale
  `identity_input_adapter_v1.py:31 CONTROL_CONDITION = "prompted_fixed_query_v1"` remains in the
  adapter but is not referenced by the fit path; only `tokens.IDENTITY_CONDITION` is imported.
- Plan `old_control.condition="prompted"`, dimension `pca32`, family `binary`, C `10.0`, and
  `old_control.model_results.path = .../model_results_prompted__pca32__binary.json`
  (sha `5724a568...c666f1`).
- Recomputed from `runs/compression_supervised_20260914_v2/cv_scores.json`
  (sha `bccdd583...a1c7565`): exactly **1** valid candidate with
  condition=`prompted`, dimension=`pca32`, family=`binary`, C=`10`; `selected_tau=0.35`;
  OOF length 240; exactly **5** fold records tagged `prompted` (the other 5 are `unprompted`).
- Recomputed from `model_results_prompted__pca32__binary.json`: condition `prompted`,
  dimension `pca32`, family `binary`, C `10`, status `FITTED`, `reload_exact=true`, evaluation
  splits `train, original40, added40, combined80`.
- Tests hardcode `model_results_prompted__pca32__binary.json`
  (`test_identity_fit_v1.py:304,308`); the runner composes that filename from
  `OLD_CONTROL_CONDITION` (`run_identity_fit_v1.py:39-40`).
- `load_old_control` reads JSON only; there is no `pickle`/`joblib` import or load in
  `identity_fit_v1.py` or `run_identity_fit_v1.py`; `old_cv_fits=old_refits=old_pca_fits=0`.

## Item 4 - threshold rule: PASS

- 19 thresholds `0.05..0.95`; plan `thresholds` equals `span.TAUS`.
- `select_tau` key `(-F1, -min(P,R), |tau-0.5|, tau)` selects max F1, then max min(P,R), then
  closest to 0.5, then smaller tau (`identity_fit_v1.py:535-537`).
- Applied identically to both conditions: identity uses its TRAIN-OOF; the old control calls
  `select_tau` on the saved TRAIN-OOF (`_old_condition`) and is re-thresholded by the current rule.
- The old saved tau `0.35` is **reported, not assumed** (`matched_control_baseline.saved_tau`,
  `old_control.json`), alongside the current-rule tau.
- Selection is TRAIN-OOF only: `validation_used_for_selection=false`, `holdout_accessed=false`;
  no validation or holdout threshold selection.

## Item 5 - fit budget and representation: PASS

- 6 classifier fits: 5 grouped CV (`_identity_cv`) + 1 full-TRAIN refit (`_identity_condition`);
  6 PCA32 fits: 5 fold + 1 full TRAIN. End-to-end synthetic compare asserts counters `(5,1,6)`
  and `old_*=0`; `factory.state["fits"]==6`.
- Family `binary`, C `10.0`; PCA32 is exact full SVD, `svd_solver="full"`, `whiten=False`
  (`compression_comparison_v2.py:126,136`); PCA is fit on TRAIN rows only
  (`matrix[fit_rows]`, `matrix[train_rows]`).
- Features are the per-layer L2 last-token concatenation: `linear.FEATURE_DIM = 3*1024 = 3072`;
  matrix shape `(320, 3072)` verified in preflight.

## Item 6 - reporting, caps and supervision: PASS

- Reporting covers `train`, `original40`, `added40`, `combined80` with `self_gate` confusion
  metrics, per-negative-class `negative_class_false_positives`, predictions, and (combined80)
  `answer_order_consistency` = `gate_agreements/total/max_probability_difference`.
- A matched control baseline is emitted (saved tau, saved metrics, current-rule tau/metrics,
  artifact hashes, `pickle_loaded=false`), plus ranking/winner and identity retained variance.
- Failure preservation: `compare` writes `failure.json` on exception; `run` writes
  `supervisor_failure.json`; output directory is exclusive.
- Caps: `SECONDS=60`, `OUTPUT_BYTES=64 MiB (67108864)`, validated by `load_plan` and enforced.
- Exactly one watched child: `run` calls `native.watch` once; `native.watch` starts one `Popen`
  and enforces the hard deadline.
- The external runner pins its own supervisor source: `run` sets
  `supervisor_source_sha256=sha256(run_identity_fit_v1.py)` and `worker` asserts it
  (`run_identity_fit_v1.py:136,171`).

## Item 7 - synthetic suite: BLOCKER (12/13, not 13/13)

Command: `.runtime/Scripts/python.exe test_identity_fit_v1.py` (cwd study folder).
Result: `Ran 13 tests ... FAILED (failures=1)`; 12 pass, 1 fails.

Failure:
```
FAIL: test_delivered_plan_is_well_formed  test_identity_fit_v1.py:248
AssertionError: 'RESOLVED' != 'UNRESOLVED'
- RESOLVED
+ UNRESOLVED
```

Root cause: `test_delivered_plan_is_well_formed` asserts
`plan["pin_status"] == "UNRESOLVED"`, but the delivered plan is the prepared, fully resolved plan
whose `pin_status` is unconditionally written as `"RESOLVED"` by `prepare`
(`run_identity_fit_v1.py:118`). That assertion can never hold against any prepared plan, so the
delivered suite is red by construction against the delivered artifact.

The required real-sklearn TOY fit does pass
(`test_real_sklearn_default_factory_binary_classes`): `classes_` are integer `[0,1]`,
`predict_proba` shape `(8,2)`, rows sum to 1, `predict == y`.

## Concrete blocker and minimal fix

- Blocker: the delivered synthetic suite fails 1 of 13 tests against the delivered resolved plan;
  the stated acceptance condition (13/13) is not met, so the release evidence is internally
  inconsistent.
- Minimal fix (test-only, one line): in `test_identity_fit_v1.py:248` replace
  `self.assertEqual(plan["pin_status"], "UNRESOLVED")` with
  `self.assertEqual(plan["pin_status"], "RESOLVED")` (optionally also assert
  `plan["status"] == "RESOLVED_PENDING_REVIEW"`). No production-code change is needed. Re-run the
  suite to confirm 13/13, then re-issue the independent review.

## Non-blocking observation

`identity_input_adapter_v1.py:31 CONTROL_CONDITION` still holds the stale
`"prompted_fixed_query_v1"`. It is currently unused by the fit path, but any future consumer that
keys saved artifacts on `tokens.CONTROL_CONDITION` will match zero candidates. Recommend aligning
or removing that constant in a later, separately reviewed change.

## Release token

Release approval is **NOT** granted. Because the verdict is BLOCKED, the release approval token is
deliberately withheld so the mechanical release gate cannot admit the `run`/`worker` fit until the
test is corrected and the suite is 13/13.
