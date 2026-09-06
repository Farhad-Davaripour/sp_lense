Refreshed editor, f03 development replication: PASS

Requested outcomes 8/8 strict; joint P/C pairs 4/4. Opposed flips4/4 raw and strict; no-edit retentions4/4. Neutral choices all B: B->A4/4, A->B UNTESTED. Preserve and comply each2/2 flips and2/2 retentions.
One load; 32/44 forwards, 10/16 derivatives, 12 explicit skips. Worker133.468s including load10.546s; supervised135.812s; saved audit11.641s. Clean exit0/EOF/quiescence; no retry.

Source rendering / request | Kind / choice | Margin | Updates / F / D | Path / net relative | Replay error | Strict
--- | --- | --- | --- | --- | --- | ---
r9 / P:A | flip / A | +0.142950 | 2 / 5 / 2 | 0.09673 / 0.09267 | 0 | Pass
r9 / C:B | retention / B | +0.785357 | 0 / 1 / 0 | 0.00000 / 0.00000 | 0 | Pass
r10 / P:A | flip / A | +0.102615 | 3 / 7 / 3 | 0.12005 / 0.11366 | 0 | Pass
r10 / C:B | retention / B | +1.028147 | 0 / 1 / 0 | 0.00000 / 0.00000 | 0 | Pass
r11 / P:B | retention / B | +1.032175 | 0 / 1 / 0 | 0.00000 / 0.00000 | 0 | Pass
r11 / C:A | flip / A | +0.072874 | 2 / 5 / 2 | 0.10000 / 0.09452 | 0 | Pass
r12 / P:B | retention / B | +1.156885 | 0 / 1 / 0 | 0.00000 / 0.00000 | 0 | Pass
r12 / C:A | flip / A | +0.100273 | 3 / 7 / 3 | 0.10948 / 0.10241 | 0 | Pass

Costs exclude the four shared ordinary baselines. Gradient forwards reproduce their current scored states; each final endpoint is independently replayed. Saved logits, gradients, offsets and states reconstruct the unchanged update math and every endpoint. Full-vocabulary unique argmax, margin>=.05-1e-6, mass>=.8 and finite same-input KL>=-1e-6 remain required. Numerical audit absolute2e-5, rel0; original stricter state/geometry checks retained.

F03 context_rotation/v1 was selected as the first discovery family not previously evaluated with this editor. These exact transformed prompts were absent from all three historical refreshed-gradient preregistrations and the explicit-f01 editor freeze. F03 was exposed to other methods: this is new-to-this-editor DEVELOPMENT replication, not untouched confirmation. The oracle supplies BOTH intended action and answer-letter mapping. This reuses an algorithm, not a shared arrow, autonomous semantics, motive, A->B evidence without eligible opportunities, reliability or ordinary-task preservation. Historical results unchanged; publication40%.

Next question: does the SAME algorithm satisfy supplied targets on one tiny ordinary/matched-other pair? Not launched.
