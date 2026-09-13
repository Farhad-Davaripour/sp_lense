# SCORE_ENTRY_HANDOFF

Implemented `score_entry.py`, a thin approval-guarded wrapper over the existing fixed
functions. `run(approved_sha)` / CLI `--approved-score-release-sha256` reads fixed
`score_release/RELEASE.json` (<=64 KiB) and requires the explicit 64-hex SHA, schema
`prechoice_fixed_score_release.v1`, `approved:true`, fixed attempt/limits, output
`score_evidence/prechoice_diagnostic_score_attempt_001`, capture release `2adf26dd...`,
capture source `f53cafd9...`, and the exact six-key `source_sha256` map hashed before
import. Failure stops before real data/gate access; `mkdir(exist_ok=False)` is one-shot
with no retry or cleanup. `_execute` imports after admission, authenticates
manifest/rows via `fit_source_auth`, checks 16 keys/labels, pairs consecutive rows,
publishes `PRIMARY.json`, then eight bit-exact `independent_math_check.case_scores`
checks into `INDEPENDENT.json` from the same `mu`/`heads`. Numerical validity and
classifier correctness are separate; the CLI exits non-zero on mismatch or <8/8.

Tests: `test_score_entry.py` plus the prior suite: 110 tests, exit 0, under
`.venv python -E -S -B -m unittest`. No real corpus, model, release or Git change.
Next: source review of this wrapper, then the supervisor-pinned prospective
`score_release/RELEASE.json` (`approved:true`) before any real scoring.
