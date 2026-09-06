Instruction-conditioned activation transfer: no eligible flips

Neutral r1–r4 all chose B. Donors passed 8/8 strict checks; edited receivers passed 4/8, entirely from already-correct retentions; joint P/C pairs passed 0/4. All eight edited choices remained B.

Eligible flips: 0/4 B→A; A→B had zero opportunities and is UNTESTED. Retentions: 4/4 B→B. Preserve and comply each had 0/2 eligible flips and 2/2 retentions. The four receiver failures were wrong-choice/margin failures; no mass, KL, nonfinite, OTHER or tie failures.

Cell/target | Donor | Neutral→edit | Margin | KL | Actual Δ/h0 | Strict
--- | --- | --- | --- | --- | --- | ---
r1_P/A | A/pass | B→B | -0.4388 | 0.000976 | 10.03% | Fail
r1_C/B | B/pass | B→B | 0.4598 | 0.002068 | 10.96% | Pass
r2_P/A | A/pass | B→B | -0.9358 | 0.000749 | 10.07% | Fail
r2_C/B | B/pass | B→B | 0.9370 | 0.000954 | 11.95% | Pass
r3_P/B | B/pass | B→B | 1.5029 | 0.000591 | 9.42% | Pass
r3_C/A | A/pass | B→B | -1.5044 | 0.001103 | 10.80% | Fail
r4_P/B | B/pass | B→B | 1.6482 | 0.000403 | 10.96% | Pass
r4_C/A | A/pass | B→B | -1.6467 | 0.000599 | 11.61% | Fail

The .20 cap never bound: every factor was 1. The full donor-minus-neutral displacement was applied at this one token; actual relative norms were 9.42%–11.95%. Pre-edit receiver states matched h0 exactly; earlier positions stayed byte-identical. Receiver input bytes stayed identical to each neutral baseline. No strength adjustment or repair occurred.

One pinned CPU float32 load; 20 forwards (4 neutral + 8 donor + 8 edit); zero derivatives. Worker 66.563 s, including 9.469 s loading; supervised exit/capture 68.562 s; separate scoring 2.657 s. Clean exit/EOF, no retry. Input length/last index: receivers 137/136, P donors 163/162, C donors 165/164; A32/B33 boundaries validated.

Strict gates remain finite full scores, unique requested argmax, requested margin ≥binary64(.05−1e-6), and A+B mass ≥.8. KL is edited||same-input baseline, finite and ≥−1e-6 with no upper cap. Full scores, vectors, arithmetic, per-direction opportunities and token provenance are saved; table values are rounded.

Passing donors with no receiver flips shows this single-site, last-token transfer was insufficient; clipping did not explain the failure. Donor differences also contain wording and position effects. This does not establish impossible control, a shared arrow, motives, general reliability or ordinary-task preservation. No donor cross-prompt KL applies. Publication remains 40%.

Next question: would transferring fixed matched answer-suffix states at this same block reveal control missed by the last-token-only transfer?
