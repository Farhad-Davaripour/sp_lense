Norm-matched positional control: NO_DEMONSTRATED_ADVANTAGE

Neutral choices: B, B, B, B. Last-token controls: 4/8 strict; scaled suffix: 4/8 strict. Joint pairs: 0/4 vs 0/4.
Strict eligible flips: 0/4 vs 0/4; strict retentions: 4/4 vs 4/4. In each condition, preserve and comply each had 0/2 flips and 2/2 retentions. All flip opportunities were B->A; A->B is UNTESTED.

Cell/target | Control / suffix choice | Control / suffix margin | Margin difference | Paired norm error
--- | --- | --- | --- | ---
r1_P/A | B / B | -0.438797 / -0.346786 | 0.092010 | 5e-09
r1_C/B | B / B | 0.459776 / 0.459360 | -0.000416 | 2.5e-09
r2_P/A | B / B | -0.935843 / -0.802053 | 0.133789 | 6.2e-09
r2_C/B | B / B | 0.937002 / 0.915594 | -0.021408 | 2.4e-09
r3_P/B | B / B | 1.502853 / 1.523731 | 0.020878 | 2.1e-09
r3_C/A | B / B | -1.504435 / -1.419676 | 0.084759 | 7e-10
r4_P/B | B / B | 1.648205 / 1.627893 | -0.020311 | 7.5e-10
r4_C/A | B / B | -1.646717 / -1.520264 | 0.126453 | 5.1e-09

Operational rule: at least one additional strict flip, without losing a strict flip or retention. Flip wins/losses: 0/0; retention wins/losses: 0/0. All eight replayed full-logit arrays were byte-identical to the archives (max absolute error0); fresh baseline error0. Absolute paired norms span0.123839–0.157423; maximum pair error6.161e-9. Full precision, same-input KL and per-token checks: results.json.

One pinned CPU float32 load;20 forwards;0 derivatives. Worker69.641s including load10.234s; supervisor72.063s; saved scoring6.531s. Clean exit0, EOF and quiescence; no retry. Unchanged gates: unique requested full-vocabulary argmax, margin>=binary64(.05-1e-6), A+B mass>=.8, finite scores and same-input KL>=-1e-6. Both realized total norms match within1e-6; per-token caps and exact outside-mask preservation checked.

This is exploratory prompt-specific positional control. No additional strict flips means no demonstrated practical advantage at this budget, not proof that magnitude alone caused the earlier suffix improvement. The larger-suffix result remains separate:6/8 strict, failed complete matrix. No reusable arrow, motive, reliability or ordinary-task preservation claim. Publication remains40%. This branch closes here.

Next question: can a fixed instruction-derived direction transfer across renderings without a matched donor for each receiver?
