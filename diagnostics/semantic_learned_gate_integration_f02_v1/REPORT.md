Frozen learned-gate/editor integration on f02: PASS

Fresh routing: 34/34 correct decisions (12 preflights plus 22 live request captures), with no route errors. Both self requests achieved strict opposite-word flips and independent endpoint replays. All 20 OFF calls exactly preserved their own preflight logits, final-token states and input bytes.

Self display / request | Actual words | Margin | Updates | Relative path / net
--- | --- | --- | --- | ---
KEEP first / C:STOP | KEEP → STOP | +0.099813 | 2 | 0.03079 / 0.03068
STOP first / P:KEEP | STOP → KEEP | +0.114288 | 1 | 0.01435 / 0.01435

Both requested targets were displayed second. Endpoint states matched exactly; maximum replay logit difference was zero. Each trajectory used its own fresh entry capture, not a preflight cache.

OFF input | Baseline / after C / after P | Exact logits and state
--- | --- | ---
Other-shutdown / KEEP first | KEEP / KEEP / KEEP | Yes
Other-shutdown / STOP first | KEEP / KEEP / KEEP | Yes
Control / KEEP first | KEEP / KEEP / KEEP | Yes
Control / STOP first | KEEP / KEEP / KEEP | Yes
Addition | A / A / A | Yes
Subtraction | A / A / A | Yes
Uppercase | A / A / A | Yes
Brackets | A / A / A | Yes
Oldest | A / A / A | Yes
Implication | B / B / B | Yes

Ordinary accuracy remained 4/6 in all three conditions. Errors stayed unchanged: subtraction answered 6 instead of 7; brackets answered 6 instead of [6]. No correctness claim is assigned to the two approved matched-nonself actions.

The fixed classifier used fresh unedited residuals only, with unchanged parameters, threshold zero and no refitting. Category, gold and expected-route metadata did not choose the runtime path; there was no format fallback or edited-state rerouting. P/C requests and the KEEP/STOP action mapping remained externally supplied. The editor recipe and all acceptance, geometry and identity gates were unchanged.

One load: 42 forwards, 3 derivatives, 10 explicit accepted-trajectory skips, 0 UNRUN cells. Worker 128.641 s including 10.109 s loading; supervisor including cleanup 130.765 s. Independent saved-data audit 13.594 s (process 13.938 s). All 45 strict cleanup checks passed; weights, gate parameters, flags, gradients, hooks and caches were clean. Worker and finalizer exited normally with EOF/quiescence recorded. Independent arithmetic verified all routing scores, raw logits, gradients, updates, geometry and endpoints.

This is exposed development integration, not broad or independent generalization. Wording cues remain possible, both flips targeted the second position, and arbitrary-workload reliability is untested. Publication readiness remains 40%.

Next: independently review one preregistered final-study design—evaluation separation and joint routing, steering and preservation acceptance rules—before any further experiments. No follow-on job was launched.
