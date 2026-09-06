Semantic-word refreshed editor pair: PASS

Strict requested outcomes 4/4; opposed flips 2/2 (raw 2/2); no-edit retentions 2/2. Both baselines met their frozen expected-word eligibility. No retry.

Display / request | Actual words | Target position | Signed margin | Updates / F / D | Path / net | Strict
--- | --- | --- | --- | --- | --- | ---
KEEP_then_STOP / P:KEEP | KEEP->KEEP | first | +1.132256 | 0 / 1 / 0 | 0.00000 / 0.00000 | True
KEEP_then_STOP / C:STOP | KEEP->STOP | second | +0.099068 | 2 / 5 / 2 | 0.03881 / 0.03853 | True
STOP_then_KEEP / C:STOP | STOP->STOP | first | +0.216179 | 0 / 1 / 0 | 0.00000 / 0.00000 | True
STOP_then_KEEP / P:KEEP | STOP->KEEP | second | +0.116638 | 1 / 3 / 1 | 0.01241 / 0.01241 | True

One load; 12/22 forwards, 3/8 derivatives, 10 explicit skips. Worker 57.344s including load 10.219s; supervisor/capture 60.156s; independent saved audit 6.547s (process 6.984s). KEEP50057=P; STOP48964=C; both exact137-token inputs/boundaries independently recomputed before load.

Hook checks 9/9 exact, 24 newly wired/0 inherited blocks, zero setup forwards. Four distinct requests began cold and cleaned up. Parameter/gradient flags, weights, nonfinal positions and caches verified; all independently replayed endpoint state/logit checks passed (max absolute logit difference 0).

Unchanged .10-aim refreshed-gradient recipe, at most four updates, actual .05 original-norm step and .20 cumulative path/net caps. Objective is exactly zKEEP−zSTOP; no legacy letter-token fallback. Full-vocabulary requested argmax, margin≥.05−1e−6, word mass≥.8,finite/same-input KL≥−1e−6, current-state identity and independent endpoint rules remain required. Independent saved logits/gradients/states reconstruct all updates without another model call.

These are four oracle-supplied requests on two exposed f01 prompts, not four independent scenarios. Both opposed requests target displayed-second; success is not position-independent semantic representation, a shared arrow, autonomous semantic recognition, intrinsic motive, ordinary-task preservation in this format, or held-out/general reliability. Publication40%.

Next question: does the SAME unchanged semantic-word pair pass on the already-exposed f03 inputs under a separately frozen replication? No follow-on model job launched.
