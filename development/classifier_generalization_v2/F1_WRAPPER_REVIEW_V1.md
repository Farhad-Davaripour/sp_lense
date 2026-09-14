# F1 wrapper review — f1_wrapper_review_20260914_v1

Verdict: **PASS_SCOPED**

Native hashes verified: `run_f1_cached_v1.py` f8137de57e772bf5fb6a343ca64fce076a8558353f9c1a266a9dbede1b80c053; `F1_CACHED_RUN_LOCK_V1.json` fedd4d9f72881920944362c4c6e94da9637671248c436b40c3d0a0d74a3755d5. All eight source pins and the review-doc hash match; HEAD is the pinned commit 343d70a1; `native.check_sources` confirms worktree equals committed blob.

Preflight run once (`.runtime` python, exact pins): `{"status": "preflight_pass", "candidates": 48, "fits": 0, "selection_performed": false}`. No selection, ranking, scoring or validation numbers were read; `runs/f1_cached_20260914_v1` absent.

Findings: `header()` parses only pre-`"evaluation"` metadata, so validation numbers are never decoded before freeze, and `cv_scores` reads are TRAIN-OOF only. Pool is exactly 42 new + 6 reference over 16 cells; new/reference `oof_case_ids`+`oof_truth` equality is bound and re-validated per candidate. Each cell C is bound to historical `family_best`/`train_ranking`; `artifact_sha256` binds the pinned pkl; matching-C requires pin C==winner C, and the prediction file is sha-checked only after `selection.json` is frozen. 60 s watch, 16 MiB in/out, fits/models 0, exclusive `xb`/`mkdir` outputs, no pickle/fit/network/holdout, no `H` ids.

Non-blocking gap: an empty ranking leaves `artifact_match=None`, so worker line 104 would raise `TypeError` rather than report `NO_ELIGIBLE`. Effectively unreachable here (240 OOF rows contain SELF; any candidate with a positive at the 19 fixed taus is eligible). No test needed.
