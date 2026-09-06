# Model-free completion batch 01: still incomplete

The full **synthetic interior-candidate case passed** independent verification, actual finalization, saved verification/all final artifacts, size checks and the sealed reader. The corrected supervisor repeat-call assertion and all 20 late recording/fault cases also passed. The full **finite-negative case did not finish** before the new batch's hard timeout. No certification, source commit, preregistration lock, preflight or real model launch followed. Publication remains 40%.

The [previous incomplete report](SOFT_DRIFT_CONSTRAINED_COMPLY_PREPARATION.md), negative preparation certificate and original partial fixture are preserved byte-for-byte/untouched. The previous timeout is not relabeled as a pass. This separate report records the newly authorized batch; detailed measurements are in [its JSON record](soft_drift_constrained_comply_completion_01.json).

## Measured bottleneck and scoped changes

The fixture already caches identical full-vocabulary logits, compression, raw hashes and reference scores. A deliberately stopped 12-baseline-row diagnostic retained all real writes: 54 budgeted write calls consumed 1.357367 seconds; its profile attributed 1.172 seconds to 166 namespace scans. Profiling adds overhead and this prefix is not a full-run estimate.

The actual full interior recording then took **174.746152 seconds**, including **167.317130 seconds in 1,073 budgeted write calls**. This confirms repeated durable filesystem admission/scans as the construction bottleneck. Pure synthetic state calculations occupy only a small remainder; speculative caching there cannot plausibly recover enough time for both complete pipelines in 300 seconds at this observed rate. Filesystem speeds vary: this is a measured bottleneck, not a universal runtime lower bound.

Only test timing/progress observation was added. The wrapper delegates every real write once with unchanged payload, path, mode, return value and exception behavior. It does not cache admission or audit answers, batch writes, omit arrays, use hardlinks, alter production code/science/checkers, or bypass finalization. An independent source-only review confirmed those properties. Write counters include attempted calls and exclude control initialization/direct inventory writes and other filesystem work.

## New execution ledger

| Targeted child subprocess | Result | Measured seconds | Hard cap seconds |
|---|---|---:|---:|
| Deliberately bounded 12-baseline prefix profile | Diagnostic completed, not a full test | 2.577312 | 25 |
| Corrected capture plus late recording/fault cases | 21 passed; 2 full cases excluded | 1.842772 | 40 |
| Test timing instrumentation lint | Passed | 0.082720 | 5 |
| Both full native finalization cases | Interior passed; finite negative interrupted | 275.059736 | 275 |

Aggregate measured targeted subprocess execution, including timeout termination/return overhead: **279.562541 / 300 seconds**. The sum of declared individual caps was **345 seconds**, not elapsed consumption; each cap fitted the remaining measured budget when started. No unused cap was charged as execution and no cap was extended. No further numerical/test subprocess ran after timeout. Usage was freshly checked at 60%; no credits/reset were used.

## Completed synthetic interior case

All 216 raw rows, 96 supplied-gradient ledger calls, eight actual QP updates, native dimension 1024 and full 248320-logit arrays were retained. Construction took 174.746152 seconds; actual independent audit/finalizer took 53.286136 seconds; sealed-reader and remaining assertions took 0.277013 seconds. Native endpoint norm was approximately 0.062499999837. These are abstract fixture records, **not model behavior or a real research candidate**.

| Saved final artifact | Bytes |
|---|---:|
| verification.json | 280730 |
| PILOT_REPORT.md | 2814 |
| comply_vector.json | 5787 |
| candidate_freeze.json | 198 |
| CLOSEOUT.json | 248 |

The sealed namespace totaled 9,201,539 bytes. Every actual per-artifact cap, the 5 MiB final category, 16 MiB auxiliary and 512 MiB total ceiling passed. Worst-finite-token native row/update serializer assertions also passed: 189842/167730 bytes. These assertions do not replace the fixed prospective envelope.

Verification SHA-256: `c55914b2d7104d8dc6adcd353ed8cea97b6bd13f4149479d441487e9a3cc0155`.
Inventory SHA-256: `a6abe32194b919a3391e69bc21ae6c41167a4b9c35a98de91c4cbad0ba8d23ea`.

Temporary fixture: `C:/Users/farha/AppData/Local/Temp/pytest-of-farha/pytest-949/test_full_native_full_vocabula0`. It is not a durable research evidence namespace.

## Unfinished finite-negative case and stop

The finite-negative case began fresh and reached its last printed checkpoint at 132 rows / step 5 / 44.922 seconds. After the outer timeout and a read-only process check, 133 complete raw rows were present; the next forward and supplied-gradient ledgers had recorded completions 134 and 62, respectively. This interrupted prefix is not a coherent completed schedule. Five update records were present. There was no endpoint/result, independent audit, verification, candidate or final inventory. No targeted Python process remained. Its files were not repaired, reused or deleted.

Temporary partial fixture: `C:/Users/farha/AppData/Local/Temp/pytest-of-farha/pytest-949/test_full_native_full_vocabula1`. The separate diagnostic prefix also remains at `C:/Users/farha/AppData/Local/Temp/soft-drift-prefix-diagnostic-42rdhefw`.

The missing requirement is the **complete finite-negative full-native pipeline**, including its actual saved finals/caps/sealed-reader checks. Preparation must remain negative until that requirement genuinely passes under new authority. No production I/O optimization is authorized by this finding. All earlier research results remain unchanged; zero model/tokenizer loads, real forwards or real derivatives; no new data/f04/gates/controllers/LoRA, push or credits/reset. Unrelated files were untouched.

**STOPPED for review; no real launch authorized.**
