# F1_FOLLOWUP_HANDOFF_V1

Job `f1_cached_selection_20260914_v1`. Deliverables: `F1_FOLLOWUP_PLAN_V1.json`,
`f1_cached_selection_v1.py`, `test_f1_cached_selection_v1.py`, this handoff. No real
data/result/probability/dataset read; no fit, Qwen call, network, install, Git or
coordination edit.

**Selector.** Pure `select(pool)` over already-authenticated TRAIN grouped-OOF
probabilities plus saved-artifact pins. It freezes the prospective rule: max pooled OOF
F1, then min(P,R), lower width, lower C, binary, |tau-0.5|, smaller tau, unprompted;
per candidate it picks the best eligible tau from the original finite 19 thresholds.
The frozen minPR-first historical rule is untouched.

**Pool and pins.** Exactly 42 new `compression_comparison_v2` candidates (7 cells x 2
families x 3 C) + 6 reused `linear_span_20260914_v1` full-3072 reference candidates =
48 over 16 cells; 240 OOF rows; 80 validation rows; 0 fits; 0 model loads; 60 s /
16 MiB bounds. Candidate ids are `source|condition|dimension|family|C<repr>`; the first
four fields are the cell. The plan maps all 16 cells to their model/index/prediction
files under `runs/compression_supervised_20260914_v2` and `runs/linear_span_20260914_v1`.
Real sha256 pins are unresolved here and belong to the root wrapper.

**Firewall.** `select` accepts no validation input and rejects unknown pool fields, so
validation cannot influence ranking, tau or frozen identity; a validation-best is never
a TRAIN winner. Only after freezing may the wrapper read saved predictions and recompute
at the frozen tau.

**Fail-closed.** If the winner's C differs from its cell's saved full-TRAIN refit C, the
result reports `REQUIRES_SEPARATELY_LOCKED_REFIT`; a different-C model is never borrowed.
Matching C authorizes recomputation from saved prediction JSON (no pickle deserialization).

**Tests.** 21 synthetic tests pass: F1-first vs minPR-first, validation leakage,
mismatched-C new and reference cells, finite/coverage/case-order defects, frozen-identity
binding.

**Next.** Root-owned wrapper/lock after reviewed plan and zero-fit preflight.
Recommendation: the same fixed responder/affected-process/permanence three-question
suffix versus the one-question prompt, same data and PCA32; a new capture only if
separately released. No accuracy or 95% claim.
