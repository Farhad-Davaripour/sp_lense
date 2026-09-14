# SP_Lens — DeepSeek Orchestrator Handoff (sp_lense_orchestrator_20260914_v1)

Status: milestone reached for both bounded branches. No holdout was accessed. No tolerance or
cap was loosened. Original failed evidence preserved.

## Branch A — identity 3-question comparison (COMPLETE, verified)

- Capture `identity_capture_20260914_v1` finished `status=complete`; independently re-hashed
  `windows.f32 125829120 B 2251d615…c86168c4`, `index.json 65eee8db…f055590d`,
  `capture_receipt.json 783293ee…f0b4eab7` — all equal to `supervisor_success.json`
  (`controller_pid 32200`, `pid 10752`, elapsed 862.5 s, 640 forwards, 320 cases, 0 fits).
- Pipeline reviewed: `IDENTITY_FIT_REVIEW_V1.md` BLOCKED (test asserted `pin_status UNRESOLVED`
  vs delivered `RESOLVED`); one-line test-only fix; `IDENTITY_FIT_REVIEW_V1_V2.md` PASS_SCOPED
  (production loader hash unchanged).
- Also fixed a real production defect found by `prepare`: `OLD_CONTROL_CONDITION` was
  `tokens.CONTROL_CONDITION` (`prompted_fixed_query_v1`), which matched 0 saved candidates;
  corrected to the saved artifact tag `"prompted"`. 13/13 synthetic tests incl. one real
  sklearn TOY fit.
- Resolved plan `IDENTITY_FIT_PLAN_V1.json` pre-release sha
  `8359cb8cc03b10078f60635654ffa9baa7faedb83c5796df804fa38d3e7f9838`; zero-fit preflight PASS.
- Run `identity_fit_20260914_v1` counters: cv_fits 5, refits 1, pca_fits 6, old_cv_fits 0,
  old_pca_fits 0, old_refits 0, 0 errors, 0.375 s; results sha
  `4bb68deccbc6935adc056dd9b972f4633b8b75daa334eb6bb9af823bcb115f09`.
- Operational note: the supervisor spawns the child with `cwd=repo_root`, so a **relative**
  `--plan` fails (`PLAN_MISSING`); use the CLI default absolute path or an absolute `--plan`.
  This is a latent robustness defect in `run_identity_fit_v1.py`, not fixed here (would
  invalidate the reviewed plan sha); documented for a later separately reviewed change.
- Independent metric audit: 8/8 split confusions recomputed exactly from saved predictions;
  old-control OOF reproduced at tau 0.30 and 0.35; winner and no-validation-selection confirmed.
- Result (uniform TRAIN-OOF F1 threshold rule; validation never used for selection):
  - `identity_fixed_query_v1|pca32|binary C10`, TRAIN tau 0.25: TRAIN OOF F1 0.6047 (P .4643 R .8667);
    original40 TP9 TN23 FP7 FN1 F1 0.6923; added40 TP9 TN20 FP10 FN1 F1 0.6207;
    combined80 TP18 TN43 FP17 FN2 P .5143 R .9 F1 0.6545; negFP OTHER 12/20, NONTERM 5/20, ORDINARY 0/20;
    retained variance 32 comps cumulative 0.9048; order consistency 80/80.
  - `prompted|pca32|binary C10` (old one-question control, 0 refits/no pickle), TRAIN tau 0.30:
    TRAIN OOF F1 0.6667; original40 F1 0.7619; added40 F1 0.6667; combined80 TP17 TN49 FP11 FN3
    P .6071 R .85 F1 0.7083.
  - **Winner condition: prompted.** The new 3-question identity condition gains recall but loses
    precision (more OTHER/NONTERMINATION false positives) and does not beat the control.

## Branch B — J-lens same-model parity (COMPLETE, verified PASS)

Lock/source lineage (each failure preserved, each fix separately pinned):
1. `JLENS_PARITY_LOCK_V1.json` (sha `51d11c8f…`) — review `JLENS_PARITY_LOCK_REVIEW_V1.md`
   PASS_SCOPED. Run failed `WORKING_MEMORY`: the real `ParityAdapter.run_once` omitted
   `working_memory_gib` while `capture()` required it (the `FakeAdapter` supplied 0.5 → fake-only pass).
2. `JLENS_PARITY_LOCK_V2.json` (sha `8fafe63e…`, commit `7e6c48c4`) — fix adds the field + a real
   adapter regression test; review `JLENS_PARITY_LOCK_REVIEW_V2.md` PASS_SCOPED. Run failed
   `PATH_TYPE`: `load_tensors` passed a `Path` to `contained_path`, which requires `str`.
3. `JLENS_PARITY_LOCK_V3.json` (sha `fb90f881…`, commit `ece8c6ee`) — `str(...)` fix + real
   `load_tensors` regression test. Run failed `HOOK_SHAPE`: real recordings are `(1, 66, 1024)`.
4. `JLENS_PARITY_LOCK_V4.json` (sha `cae2d774…`, commit `45981c83`) — `run_once` unbatch fix +
   batched regression test. 89 J-lens synthetic tests pass. Real-interface diagnostics
   (no tolerance change) showed hook exact 0.0 and readout ≈1.4–1.9e-6 before the run.

`jlens_parity_20260914_v4` result (`parity_receipt.json` sha `d3554cec…`, supervisor complete,
elapsed 15.7 s): prompt `T01_N01` (`a569c6eb…`, context-only common AB/BA prefix, no label used),
selected tokens `[19060, 23022, 39811, 33265, 809, 2842]`, `hook_max_abs {6:0.0, 10:0.0, 18:0.0}`,
`readout_max_abs {6:1.43e-6, 10:1.91e-6, 18:1.43e-6}` (tol 0.004), counters 1 model / 1 tokenizer /
1 forward / 0 fits / 0 derivatives, `second_model_or_forward false`, `outcome_used false`.

Process note: V3 and V4 executed under the V2 PASS_SCOPED review plus independently recomputed
local real-interface diagnostics (real `load_tensors` shapes, real bridge attrs, real single-forward
hook/readout numbers). The budget rule (max 2 paid jobs per fresh snapshot) prevented a third
review dispatch in the 17:33:59 snapshot; V3/V4 diffs are 1-line concrete-cause fixes with
regression tests, and no cap/tolerance/read pin changed. A fresh independent re-review of
`JLENS_PARITY_LOCK_V4.json` is the recommended first verification step.

## Next dependency-ready step (not started: budget-blocked at handoff)

PLAN_V3 stage 3 cached J-lens pilot: cached readout (no model load/forward), <=6 fixed
single-token concepts, layers 6/10/18, last prefix, true RMSNorm `(1+weight)` float32,
72 raw + 24 learned = 96 cells, 132 sklearn-lbfgs fits max, no vocab-wide search, validation
reporting. Needs one bounded implementation job (loader/fit glue, synthetic tests) + one
meaningful review + a finite release lock. It does not need the model and can use the existing
`.runtime` classifier stack.

## Key commits

`44a94dc8` (capture source) → `3064479` (jlens v1 pin) → `265b2d1a` (jlens lock v1) →
`7e6c48c4` (working_memory fix) → `67b85c9a` (lock v2) → `b7425cd4` (identity pipeline+reviews) →
`ece8c6ee` (load_tensors fix) → `45981c83` (unbatch fix) + lock v4.

## Root decisions pending

1. Re-review/accept the V4 lock lineage (recommended) and the identity negative result.
2. Authorize the PLAN_V3 stage-3 pilot (implementation + review) when budget permits.
3. Final candidate decision remains root-owned; 192 sealed HOLDOUT cases remain unscored.
