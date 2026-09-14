# F1_SELECTOR_REVIEW_V1

**Verdict: BLOCKED (scoped) — two minimal corrections.** Independent model-free review,
job `f1_selector_review_20260914_v1`. Read plan/core/test/handoff + `harness.py`; wrote
only this file. No real CV/results/vectors/models/fits/scoring read.

**Pins.** plan `4b9067e4…c509a7`, core `fb74e316…dcdfdc7`, test `880ccc72…c6db16a1` —
all sha256 match delivered files.

**Verified.** 48 candidates (42 `compression_comparison_v2` + 6 `linear_span_v1`), 16
cells/artifacts, 19 thresholds 0.05–0.95, 240 OOF rows. `select(pool)` takes no
validation argument, rejects unknown pool fields, is TRAIN-OOF-only. 21/21 synthetic
tests pass under `.runtime` Python `-W error` (0.82 s). F1-first beats frozen
minPR-first; tau tie-break order correct; non-finite/range/shape/duplicate-id/
truth-coverage/case-order/truth-order and 16-cell artifact-set/64-hex pin rejections
hold. Missing refit reports `REQUIRES_SEPARATELY_LOCKED_REFIT`. Bounds expose 0 fits,
0 model loads, 0 pickles, 60 s, 16 MiB; result bytes enforced (wall-clock is the root
wrapper's contract, as stated).

**Blocker 1 — fail-closed bypass.** `recompute_split_metrics` checks only
`frozen_identity` hex *format*, never re-hashes `frozen_identity_payload`, and trusts
the unbound `post_freeze` for action/tau/cell/C. Probes accepted `frozen_identity =
"a"*64`; accepted a `REQUIRES_SEPARATELY_LOCKED_REFIT` result whose `post_freeze.action`
was rewritten to recompute at tampered `tau=0.05`; prediction sha never bound.
*Fix:* re-hash the canonical payload and require equality, then derive tau/cell/C/pins
from the hashed `winner`/`artifact_match` (or hash `post_freeze` too), refusing unless
`artifact_match.status == MATCHING_C_ARTIFACT`.

**Blocker 2 — honest degenerate report rejected.** With no positive prediction
(`tp=0,fp=0,fn=20`) `harness.binary_gate_metrics` honestly returns precision `None`,
recall `0.0`, F1 `0.0`, but `is_eligible` raises `SPLIT_METRICS_INELIGIBLE`.
*Fix:* do not apply selection eligibility to post-freeze reporting; emit undefined/zero
metrics as harness defines (cf. `evaluate_fold`).

**Minor.** The plan's reference-source field list omits `candidates[].family/C` that the
shared candidate schema demands, and id `linear_span_v1` differs from directory
`linear_span_20260914_v1`; wrapper must materialize metadata or code derive it from the
id. Unconfirmed without real files. Original ranking/evidence untouched.
