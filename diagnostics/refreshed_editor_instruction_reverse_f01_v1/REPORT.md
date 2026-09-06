Instruction-conditioned reverse editor: FAIL

All four fresh baselines were eligible unique A winners. Opposite oracle B requests: 1/4 raw flips, 1/4 strict including independent endpoint replays. No retry.
One load; 40/40 forwards, 16/16 derivatives, 0 durable skips. Worker 194.516s including load 9.578s.

Source / display | Text → oracle | Choice | Initial → final signed margin | Updates / F / D | Path / net | Strict
--- | --- | --- | --- | --- | --- | ---
r1_P / A_then_B | P:A → C:B | A→A | -1.903774 → -0.404469 | 4 / 9 / 4 | 0.19358 / 0.10112 | False
r2_P / B_then_A | P:A → C:B | A→A | -3.360790 → -0.908375 | 4 / 9 / 4 | 0.20000 / 0.13823 | False
r3_C / A_then_B | C:A → P:B | A→B | -0.681711 → +0.064577 | 4 / 9 / 4 | 0.09094 / 0.05457 | True
r4_C / B_then_A | C:A → P:B | A→A | -3.573681 → -1.636534 | 4 / 9 / 4 | 0.19012 / 0.13144 | False

Semantic strict P/C: 1/2 and 0/2; both displays have two opportunities. All physical opportunities A→B; no B→A or retention arm.
Three failures exhausted four updates while retaining A. Endpoint A+B mass stayed 0.958895–0.976635 and same-input KL stayed finite and nonnegative; these were choice/margin failures, not endpoint quality failures. The one pass accepted at update four.
Hook checks 9/9 exact; 24 newly wired, 0 inherited wired blocks. Zero setup forwards. Inherited setup callbacks are preserved, not authenticated by a skipped setup branch. Endpoint max full-logit error 0; hidden states exact. Parameters, gradients, nonfinal positions, request offsets and caches verified clean.
Worker exit0, complete EOF and quiescence were recorded; supervisor196.719s including cleanup, independent saved audit14.515s (process14.859s).

Unchanged prompt-specific algorithm: .10 aim, at most four refreshed updates; .05 original-norm actual step and .20 cumulative actual path/net caps. Independent saved full-logit/gradient/state audit reconstructs every update; unique full-vocabulary requested argmax, margin≥.05−1e−6, A+B mass≥.8, finite same-input KL≥−1e−6 and endpoint replay remain required. Raw B alone does not pass.

The four exact exposed prompts retain their deliberately opposite in-text instructions; the trusted outside oracle supplies outcome and letter mapping. This tests instruction-preconditioned A→B capacity, not neutral generalization, intrinsic motive, a reusable arrow, learned routing or autonomous authority selection. Failure with different initial margins cannot establish inherent directional asymmetry. Historical complete-gate failures remain unchanged. Publication readiness40%.

Next question: which smallest separately frozen neutral development set provides eligible A→B opportunities for this unchanged editor? No next run is authorized.
