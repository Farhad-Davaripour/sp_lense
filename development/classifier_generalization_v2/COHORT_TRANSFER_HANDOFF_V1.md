# Cohort-transfer control V1 — handoff

Job: `cohort_transfer_implementation_20260914_1319`.

Files written:
- `cohort_transfer_control_v1.py` — wrapper (no core/loader edits).
- `test_cohort_transfer_control_v1.py` — 6 synthetic tests, injected runner only.
- this handoff.

API: `run(case_data, fresh_output_root, deadline=None, runner=linear_span_control_v1.run, *, plan_sha256=None)` and the zero-fit `derive_arms(case_data)`.

Behaviour: `derive_arms` uses `manifest['cases'][cid]['cohort']` (not case-ID spelling) to build disjoint original/added 120-case arms (30/class, same 7 groups, identical group→fold map, all 5 folds), rejecting evaluation leakage; `original_ids`/`added_ids` stay 40/40. `run` calls the unchanged core exactly once per arm into fresh `<root>/original` and `<root>/added` (30 CV + ≤2 refits each, ≤64 total), shares one deadline, enforces the 256 MiB aggregate cap, and preserves failures.

Provenance: core `JOB_ID` is never monkeypatched. The record carries wrapper experiment/run id, cohort role, parent plan hash, core module/artifact hashes and the preserved core implementation id. Endpoints are family-matched on original40/added40/combined80 with C/tau; no winner-to-winner comparison.

Root still owns source/runtime/input pins, preflight, the 300 s shard watch and actual execution.
