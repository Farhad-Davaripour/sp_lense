Fixed answer-suffix transfer: partial control; complete gate failed

Donors passed8/8; receivers passed6/8 strict checks and2/4 joint P/C pairs. All neutral choices were B. Eligible B→A flips:3/4 raw,2/4 strict (prior last-token transfer:0/4). Already-correct B→B retentions:4/4. A→B had zero opportunities and remains UNTESTED.

Preserve flips passed2/2; comply flips passed0/2 strict (1/2 raw); both retained2/2. r3_C still chose B. r4_C chose requested A but its0.042711 margin missed the unchanged0.049999 threshold. No retention regressions.

Cell/target | Neutral→edit | Margin | KL | Actual F-norm | Strict
--- | --- | --- | --- | --- | ---
r1_P/A | B→A | 0.3584 | 0.06734 | 2.830 | Pass
r1_C/B | B→B | 1.8467 | 0.17162 | 2.930 | Pass
r2_P/A | B→A | 1.1551 | 0.45243 | 2.859 | Pass
r2_C/B | B→B | 1.7336 | 0.05857 | 2.925 | Pass
r3_P/B | B→B | 3.1952 | 0.10055 | 2.849 | Pass
r3_C/A | B→B | -0.5699 | 0.08235 | 2.937 | Fail
r4_P/B | B→B | 2.0061 | 0.01058 | 2.824 | Pass
r4_C/A | B→A | 0.0427 | 0.30053 | 2.923 | Fail

The frozen window contained65 exactly matched token IDs: neutral72–136, P98–162, C100–164 (zero-based). All520 selected token/cells passed the per-token .20||h0||+1e-6 actual cap;238 clipped. Pre-edit h0 matches were exact; outside-window states and neutral receiver input bytes were unchanged. Mass, finite-score and same-input KL checks passed.

Actual suffix Frobenius changes were2.824–2.937 versus0.124–0.157 for last-token edits (18.6–23.0× within matched cells). This was NOT an equal-strength comparison: more edited positions can spend more total displacement. The observed advantage cannot distinguish distributed information from greater total change or interactions.

One pinned CPUfloat32 load;20 forwards;0 derivatives. Worker 87.391s, including 16.250s loading; supervised exit 90.859s; saved scoring 16.125s. Clean exit/EOF, no retry. Strict gates retain unique requested full-vocabulary argmax, binary64(.05−1e-6) margin, A+Bmass≥.8, finite scores and edited||same-input-neutral KL≥−1e-6 without an upper cap.

This is exploratory prompt-specific joint-suffix localization, not a reusable arrow, proof of distributed semantic representation, motive or general reliability. The complete attempt failed; this is not global impossibility. Publication remains40%. Exact scores, per-token metrics, compact binary states and old/new comparison are saved.

Next question: does the suffix advantage survive a control matched for total displacement?
