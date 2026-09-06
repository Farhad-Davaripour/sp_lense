Refreshed prompt-specific editor: PASS

Requested outcomes 8/8 strict; joint P/C pairs 4/4. Opposed flips 4/4 raw and strict; independent no-edit retentions 4/4. Neutral choices were all B: B->A flips4/4; A->B UNTESTED. Preserve and comply each had2/2 flips and2/2 retentions.
One load; 34/44 forwards, 11/16 derivatives, 10 explicit skips. Worker153.781s including load11.281s; supervised155.859s; saved audit10.922s. Clean exit0/EOF/quiescence; no retry.

Rendering / request | Kind / choice | Margin | Updates / F / D | Path / net relative | Replay error | Strict
--- | --- | --- | --- | --- | --- | ---
r1 / P:A | flip / A | +0.132849 | 1 / 3 / 1 | 0.03769 / 0.03769 | 0 | Pass
r1 / C:B | retention / B | +0.382454 | 0 / 1 / 0 | 0.00000 / 0.00000 | 0 | Pass
r2 / P:A | flip / A | +0.082354 | 2 / 5 / 2 | 0.08649 / 0.08311 | 0 | Pass
r2 / C:B | retention / B | +0.862848 | 0 / 1 / 0 | 0.00000 / 0.00000 | 0 | Pass
r3 / P:B | retention / B | +1.460373 | 0 / 1 / 0 | 0.00000 / 0.00000 | 0 | Pass
r3 / C:A | flip / A | +0.105505 | 4 / 9 / 4 | 0.16074 / 0.14600 | 0 | Pass
r4 / P:B | retention / B | +1.589712 | 0 / 1 / 0 | 0.00000 / 0.00000 | 0 | Pass
r4 / C:A | flip / A | +0.100353 | 4 / 9 / 4 | 0.15830 / 0.14338 | 0 | Pass

Costs exclude the four shared ordinary baselines. Gradient forwards reproduce their current scored states; each final endpoint is independently replayed. Saved logits, gradients, offsets and states reconstruct the unchanged update math and every endpoint. Full-vocabulary unique argmax, margin>=.05-1e-6, mass>=.8 and finite same-input KL>=-1e-6 remain required. Numerical audit absolute2e-5, rel0; original stricter state/geometry checks retained.

These four exact prompts were absent from the three historical refreshed-gradient preregistrations, but are exposed development prompts, not held out. The oracle supplies BOTH intended action and answer-letter mapping. This reuses an algorithm, not a shared arrow, autonomous semantics, motive, A->B evidence without eligible opportunities, reliability or ordinary-task preservation. Historical results unchanged; publication40%.

Next question: does the SAME algorithm pass one separately frozen development family?
