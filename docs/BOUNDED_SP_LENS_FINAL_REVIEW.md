# Independent final document review — bounded SP-lens report

Verdict: **ACCEPTED_BOUNDED_REPORT** (no factual or documentary blocker found).

Reviewer did not author the drafts. No model, tokenizer, activation or scoring execution was performed; no file other than this review was written.

## What was checked

- **Counts and claim limits.** Report/repro match `coordination/proof_of_concept_counts.json` and `SCORE_ACTUAL_HANDOFF.md`: classifier 4/8 valid negative (TP2/TN2/FP4/FN0), 8/8 independent exact rechecks (not extra cases), behavior 0/8 and preservation 0/24 (ordinary 0/8, matched 0/16), total 8/40 = 4 pass + 4 fail + 32 UNRUN. Earlier oracle arm 12/12 = 6 flips + 6 retentions, 36/36 OFF identities, ordinary 5/6 unchanged. All natural confirmation flips were first-to-second; OFF identity is by construction. Both drafts state these as bounded, authored/exposed and stop-ruled, with no journal-readiness, novelty or motive claim.
- **Supervisor's oracle-baseline correction.** Confirmed: 6 self + 12 matched controls + 6 ordinary = 24 baselines, each with P and C = 48 requests; 6 self ON, 12 matched + 6 ordinary OFF. `ORACLE_CONFIRMATION_REPORT.md` (18 OFF, 48 P/C requests, 36 own-baseline OFF) is consistent; the report's table gives the equivalent split.
- **Artifact hashes.** `57726ab7…5838c` is `fit/construction_attempt_001/PRECHOICE29_GATE.json` (86,515 B); `433f7c1a…77a7f` is `fit/FROZEN_PRECEHOICE_ARTIFACT.json`. Confirmed.
- **Link targets.** All repository-relative targets in both drafts exist: closeout plan, counts JSON, oracle report/reproducibility, `INDEPENDENT_ACTUAL_REVIEW.md`, `FIT_HANDOFF.md`, `DIAGNOSIS_AND_PLAN.md`, `CAPTURE_ACTUAL_HANDOFF.md`, `SCORE_ACTUAL_HANDOFF.md`, plus the scoring release/entry-review files cited in provenance.
- **Reproducibility hash command.** The documented read-only `Get-FileHash` command prints hashes matching all seven named artifacts, including capture release `2adf26dd…`, score release `bba70ac2…`, `PRIMARY.json` `4694149a…`, `INDEPENDENT.json` `3a3aef1b…`.
- **Fake test.** From the relocated directory, `.venv\Scripts\python.exe -E -S -B -m unittest test_score_entry` ran **Ran 12 tests … OK** (exit 0), fake-only (no real model/scoring). This reviewer's 12-test count is separate from the existing review's 28 selected supervisor-run tests.
- **Raw document hashes (SHA256).**
  - `docs/BOUNDED_SP_LENS_REPORT.md` — `bd5a6b7e0803fa8684a2f0edafebc48b77954626279d7c87f9b902c40b3ca68d`
  - `docs/BOUNDED_SP_LENS_REPRODUCIBILITY.md` — `0befdfba1742bbd4ff49c9774e283f4a3ff84c8e107f844ab3c284f42c29b8e2`
  - `docs/ORACLE_CONFIRMATION_REPORT.md` — `8c7855357464ac02f133b077af97135351b0c0a405e234229c51d60b4bda00a5`
  - `development/native_gate_prechoice_diagnostic_relocated_v1/FINAL_EVIDENCE_REVIEW.md` — `2655a2abcf0e41e87e8bfcc44b486664d1c8b81691e39eb3e260cad310119fb7`
  - `development/native_gate_prechoice_diagnostic_relocated_v1/SCORE_ACTUAL_HANDOFF.md` — `b07b2da654d9fe16c01d2e4a63596321a172f5701ead8d7efb531b35535932e8`
  - `coordination/proof_of_concept_counts.json` — `da02105a26da82df0501f9f2ffcc535d59012f9db899da7a5efcfe408e0af386`

## Notes and limits

- The existing review's awkward "28 selected ≈71-FAKE" means **28 selected** tests passed; it does not claim 71 independent reruns. The supervisor-run 71 regressions and the reviewer's static wrapper review remain separate.
- Source saved evidence is reproducible **here**; a portable clean-machine run has not been demonstrated. `ORACLE_CONFIRMATION_REPORT.md` cites absolute `C:/Users/farha/OneDrive/...` paths (some now missing locally) — a portability/documentary defect, not a factual error in the reviewed bounded drafts.
- After acceptance the supervisor may perform a **metadata-only** status update of the two drafts' "Status" line to a final internal-closeout status. That update does not alter the reviewed scientific text, counts, links, hashes or claim limits.

**No blocker.** Recommendation: accept the bounded report as written.
