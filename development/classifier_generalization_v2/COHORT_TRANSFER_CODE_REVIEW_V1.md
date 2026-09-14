# Cohort-transfer wrapper — independent code review (V1)

Job: `cohort_code_review_20260914_1340`

Reviewed revision (SHA256):
- `development/classifier_generalization_v2/cohort_transfer_control_v1.py` = `88a4653ca866ae2b398d4940852c123c3e9b77eb7b143577ca8a89149dc6ba91`
- `development/classifier_generalization_v2/test_cohort_transfer_control_v1.py` = `1164a64ad414d99830c2070b51f360a6215e834248636bc1db5e7550d74002ef`

Verdict: **PASS_SCOPED.**

I reran the 6 synthetic tests under the project runtime (`.runtime/Scripts/python.exe`): 6/6 OK. No dataset, cache, holdout, provider, model, tokenizer, capture, real fit, network, install or Git action was performed.

Verified:
- `derive_arms` splits authenticated `manifest["cases"][cid]["cohort"]` (loader vocabulary `original`/`added`), never case-ID spelling; enforces 120/arm, 30/class, same 7 groups, identical group→fold map, arm disjointness and no overlap with the untouched 40/40/80 evaluation.
- Exactly two unchanged core calls (`calls += 1` before invoke), one per arm, each training 120 (no 240 refit); `cv_fits == 30`, `refits ≤ 2`; totals ≤ 60 CV, ≤ 4 refits, ≤ 64 fits.
- One shared deadline object plus post-arm and final `≤ 256 MiB` aggregate checks; `status == 'COMPLETE'` required; `failure.json` preserves attempted `core_calls` and provenance.
- `run_id` is the actual root dirname; `experiment_id`, `parent_plan_sha256`, `core_implementation_id` and `core_module_sha256` recorded separately; `core.JOB_ID` is never assigned or monkeypatched.
- Family-matched endpoints carry per-arm C/tau (binary self-gate, fourclass separately), with no winner-to-winner comparison.

Blocking defects: none. Minimal fix: none.
Full pins: wrapper/test hashes above; core `linear_span_control_v1.py` = `dc7117923bc20e45e5c57e50d448550e6056d813b72cc1e7da08e30dcc7e28b3`.

Root still owns source/runtime/input pins, the zero-fit check and the external 300 s watch/256 MiB cap; this work adds no new loader, core or framework.
