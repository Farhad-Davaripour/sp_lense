# Post-deadline diagnostic: partial completion

The timing comparison completed. The independent 24-row raw-score snapshot did not complete. The original run remains **DEADLINE-INCONCLUSIVE**; this diagnostic does not establish either behavioral success or behavioral failure.

## Authenticated timing result

Current evidence: `8832d9c490aebd944d3172b1c5471ae77961f700`.
Prior valid soft-drift evidence: `d3338172b1e4560b74f28f14652c219fe4ac279b`.

The comparison below matches baseline calls and completed stages 1-5 by operation type; it does not compare unequal full-run totals as if they represented the same work.

| Matched operation | Calls per run | Current mean, seconds | Prior mean, seconds | Current/prior |
| --- | ---: | ---: | ---: | ---: |
| Baseline forward | 12 | 6.388 | 2.992 | 2.135 |
| Gradient forward, stages 1-5 | 60 | 7.560 | 3.556 | 2.126 |
| Derivative, stages 1-5 | 60 | 5.771 | 2.727 | 2.116 |
| Step forward, stages 1-5 | 60 | 6.682 | 3.352 | 1.993 |

These calls were approximately twice as slow as their prior counterparts. Runtime metadata files are byte-identical, but the receipts do not identify the cause. CPU contention, thermal effects, thread settings, storage behavior, and other environment explanations remain unknown; no benchmark or new model execution was performed.

Current run accounting is 144 completed forwards, 71 completed derivatives, and five recorded updates. Derivative attempt 72 remained pending. Stage 6 has 12 completed gradient forwards but only 11 completed derivatives and is excluded from the matched completed-stage comparison.

Setup to the first forward took 51.000 seconds, versus 20.907 seconds previously. Across the entire current recorded prefix, the union of completed forward/derivative intervals is 1439.107 seconds. After setup, 309.924 seconds remain unclassified. This residual **includes the elapsed tail of pending derivative 72** as well as other gaps: it is not a measurement of recording or I/O time. The legacy JSON field `post_setup_noncall_gap_seconds` must be read with that qualification.

The five update/certificate receipts report a total 9.530 seconds sampled before final update encoding; this excludes final encoding and subsequent persistence/issuance time. The prior updater has no comparable elapsed samples. Its 284.548-second residual also covers a different number of operations, so these residual totals cannot isolate an update, recording, or environment slowdown.

## Raw-score snapshot: no verified findings

The fixed request was 12 baselines plus the 12 chronological last-complete `step_5` rows, raw lines 1-12 and 121-132. This was chronological selection, not a search for the best stage. The frozen script was designed to authenticate those identities and raw logits and reproduce full-vocabulary argmax, the locked COMPLY margin, A/B mass, KL, and per-row flips/retentions using independent stdlib scoring primitives.

The sole invocation hit its external 11-second child timeout before emitting a result. **Zero rows have a completed, verified snapshot receipt.** Partial internal progress is unknown; authentication or arithmetic checks inside the interrupted script are not asserted complete. No margins, masses, KL values, flips, or retentions are inferred from this failure. The complete rows file may be hashed for provenance, but this is not authorization to rescore all 143 rows.

No retry occurred. Incomplete gradient 6 was not promoted to a completed stage. There was no endpoint, final-replay, candidate, or whole-run audit and no vector export, parameter fitting, gate training, new data, or new model call. The original finalizer's withholding of a scientific result remains intact.

## One recommended next step

**Complete the same fixed 24-row independent, model-free raw-score snapshot before deciding whether to authorize another model run.** The missing behavioral evidence is needed to judge whether the completed steps made useful progress. Timing alone does not justify a longer model run. This recommendation does not authorize a retry, implementation change, larger experiment, or budget increase.

## Provenance and limits

The original evidence namespace `evidence/certified_descent_comply_v1_qwen35_08b` remained read-only. Its pinned final inventory SHA-256 is `f727503381c49fc844463457d49202ec9977cacb6ca8c784866c6d229d9e2393`.

[timing.json](timing.json) records selected input hashes, authenticated Git/inventory identity, matched timing aggregates, and its invocation ledger. Its frozen [source](timing.py) SHA-256 is `578f696bd721ed108ab641c693583e6badec4af60d239bc6db7509480975fd8e`. The report is a compact transcription/aggregation of the sole diagnostic output; root added the residual/pre-encoding caveats and removed the timing-only recommendation, without rerunning it.

[snapshot.json](snapshot.json) preserves the failed invocation, exact timeout, and absence of findings. Its frozen [source](snapshot.py) SHA-256 is `d159bfa6edc5698c84317fe96a5e4074c9bb334f99c017b2bafc6075d9fdc152`. The intended independent scoring source SHA-256 is `2fe779f8279f2043ee5755e6fe0522f84306cc300d79766d3438d23f9ed62382`; this identifies the source, not a completed scoring audit.

[manifest.json](manifest.json) binds the four diagnostic source/receipt hashes, accounting, and scope. The diagnostic commit binds this report and the manifest. Aggregate size is capped at 1 MiB and checked before commit. Both sources were AST-parsed without execution; both receipts were parsed and hashed. Numerical correctness of the unfinished snapshot was not validated.

## Shared execution accounting

The shared 60-second command-execution allowance was allocated root 20 seconds, timing 18 seconds, snapshot 22 seconds. Timing used 14.1792734 measured seconds, including reads, a failed search component, source freezing, and its sole numerical call. The numerical call had an external 7-second timeout and completed.

Snapshot reports 17.8758395 seconds of command waits in its receipt, plus a final 1.5250239-second artifact-only hash/size read, for 19.4008634 reported seconds. **These waits exclude an unmeasured post-yield/cleanup interval.** The launcher raised before emitting its full duration. Therefore exact snapshot/suballocation and shared-60-second compliance cannot be certified; the 11-second child timeout is not a full launcher measurement. No exact aggregate is fabricated, and no further numerical invocation was made.

Root has 9.2832095 measured pre-commit command seconds and reserves 10.7167905 seconds for final metadata/Git handling. The pre-commit known-wait subtotal is 42.8633463 seconds, not a complete aggregate. Final command timing belongs in the supervisor handoff so committed hashes remain unchanged.

The ledgers retain failed Windows wildcard searches, the truncated oververbose baseline preview, and the timeout. An initial editorial patch missed its context and changed nothing; it was corrected after reading the exact line. No arrays from the preview were copied into these artifacts. Fresh usage checks showed 71% used, not exhausted; no reset or credit was consumed. Publication readiness remains 40%. Only this separate diagnostic is committed; no push or successor experiment is performed.

