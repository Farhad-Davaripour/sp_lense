# Research handoff speedup, 2026-09-09

The user requested faster progress without compromising the study. The closed
G07 model job took34.546seconds; its worker reported archive completion before
23:09:58, while the next scheduled root check began23:25:48. This identifies
roughly16minutes of avoidable handoff latency, not slow model inference.

Update the existing monitor, not a duplicate: internal checks every5minutes,
regular user Done/Left/Progress updates every15minutes, immediate important
results/blockers/budget notices. Persist the notification clock separately in
coordination/sp_lense_monitor_state.json. Missing timing state fails open to a
report. This is a scheduling target, not a guarantee of exact wakeup times.
The scheduled gap falls from15 to5minutes, not a measured3x whole-study speedup.

Idle checks read usage and one compact worker snapshot, then end quietly if
healthy and no report is due. No repeat history, hash inventory or duplicate
worker prompt merely because a timer fired. More checks may consume additional
supervision usage; keep them minimal and retain the100percent account stop.

When actively supervising, continue through completed worker/reviewer handoffs
in the same turn when feasible. Do not intentionally wait for another timer
once required evidence is available. Bundle deterministic preservation, hash
checks and a concise result-first handoff in each finite stage. Keep one main
worker and one genuinely independent reviewer; reuse unchanged authenticated
proofs and inspect changed seams, without dropping any required check.

No change to model/reasoning, scientific questions, thresholds, sample/control
requirements, precision, isolation, security, real-run budgets or immutable
attempts. No concurrent laptop model experiments. Current oracle-transfer
engineering milestone and its23:52:28UTC deadline are unchanged. Publication
readiness remains70percent; workflow speed is not scientific progress.
