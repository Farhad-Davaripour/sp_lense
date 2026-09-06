# Soft-drift constrained COMPLY preparation: INCOMPLETE

**STOPPED; not certified, not locked, no real run authorized.** Working source is based on commit `0d2d104c3a367dea321f6621fca851e8d4ffab74`. The new integration and tests remain uncommitted because the required full finalization regression did not complete. No source-readiness or preregistration-only lock commit was made.

The new runtime adapter, independent recorded-data checker and compact finalizer are implemented. Source review found and addressed schema/hash-container mismatches before the full regression. The immutable standalone solver/checker/numerical-policy bytes are unchanged. Shared geometry arrays and their hashes are independently reconstructed exactly, with separately fixed scalar/prediction tolerance; raw scoring and standalone KKT policies remain unchanged.

## Executed validation

| Command | Outcome | Measured subprocess seconds | Hard cap seconds |
|---|---|---:|---:|
| Core scoped format | Passed | 0.103367 | 10 |
| Initial core lint | One import-order finding, fixed | 0.108449 | 5 |
| Final core lint | Passed | 0.097615 | 5 |
| Core targeted pytest | 68 passed, one test assertion failed | 7.853970 | 40 |
| Initial recording format | Passed | 0.084705 | 5 |
| Initial recording lint | Three style findings, fixed | 0.080274 | 5 |
| Final recording format | Passed | 0.085061 | 5 |
| Final recording lint | Passed | 0.083176 | 5 |
| Full recording pytest and corrected capture test | **Hard timeout; no completed case reported** | 190.101937 | 190 |

Measured subprocess total: **198.598554 seconds**. Conservatively charged sum of executed command caps: **270/300 seconds**. The timeout controller's roughly 0.102-second termination/return overhead is included in the measured total and is inside the overall allowance. Planned freeze/preflight commands were not executed. No cap was enlarged, no native regression was retried, and no standalone native benchmark or broad historical suite ran. Fresh usage was 59–60% before test batches; no reset or credits were used.

The core failure was in `test_supervisor_exact_command_real_capture_and_no_retry`: it expected an exception on the second mocked-supervisor invocation, whereas the frozen supervisor returns INCONCLUSIVE after its recording guard rejects the attempt. The test was corrected to assert that return and explicitly count any repeated capture launch. That correction remains **unverified**: the later batch timed out in its first full-recording case before reaching the corrected test. No production no-retry guard was weakened.

## Where the full regression stopped

Only the first parameterized case, `interior_candidate`, started. Its fixed synthetic worker-recording phase took **181.953000 seconds** and recorded 216 fake forward-ledger completions, 96 supplied-gradient ledger completions and eight QP updates. Its own unaudited summary reports a native endpoint norm of approximately 0.062499999837 and 12 accepted synthetic finals. These are scripted fake-record facts, **not an independently accepted result or model behavior**.

The test entered finalization and wrote its simulated capture and run-status receipts. The outer 190-second timeout interrupted the test before independent verification/finalization completed. There is no `verification.json`, `FINAL_INVENTORY.json`, `comply_vector.json` or `candidate_freeze.json` in that temporary fixture. The finite-negative case, the subsequent recording/fault tests and the corrected capture test were not reached. No claim of final artifact-size or sealed-reader certification is made.

The partial fixture was preserved at:

`C:/Users/farha/AppData/Local/Temp/pytest-of-farha/pytest-947/test_full_native_full_vocabula0`

This is a temporary pytest directory, not a research evidence namespace or a durable published artifact. After timeout, a read-only process check found no remaining Python process with the exact targeted test command. No test process was restarted and no partial fixture was repaired or deleted.

Observed partial-file hashes:

- `analysis.json`: `d7f658e9edaf5a7b6204c756760247ca7a66f8c976f26482295dd2e0f0fee263`
- `result.json`: `f167adaf834bf7bd418b834f6f3052be08bae9317cbdacc9772a35e17800ed9f`
- `RUN_STATUS.json`: `dc1745f306ad11b891889de5569a78c6f8b77e99a4adf8896c8bd9eb1edb5738`

The dominant observed time was **fixture construction and bounded recording**, not a completed independent audit. No finer performance diagnosis was executed after the cap. This outcome therefore does not establish a QP solver failure or a model-runtime prediction.

## Remaining blockers and scope

Both complete native-1024/full-vocabulary-248320/216-row/eight-update independent-audit → finalizer → sealed-reader cases must pass, together with size-envelope, zero-byte fault-precedence, malicious/tampered records and the corrected no-retry test. Until then, the [preparation record](soft_drift_constrained_comply_preparation.json) explicitly remains incomplete and the production certificate gate rejects it.

Scientific final payloads are preencoded together; capture/run-status receipts are written earlier and the immutable helper admits the final inventory last. The intended end-to-end regression must validate every actual category and artifact through that last seal. Preencoding alone is not a substitute.

Zero ML imports or model/tokenizer loads, zero real forwards or derivatives. A stdlib namespace supplies abstract gradients to exercise the derivative ledger; it is not Torch, autograd or a model. No f04/new data, gate/controller/LoRA, old-source/evidence edits, push, reset or credits. Existing unrelated untracked files remain untouched. All earlier real-run failures and the retrospective zero-flip result are unchanged; publication remains 40%.

**No further work or run is authorized by this report.** Stop for review of the incomplete preparation and its bounded-test bottleneck.
