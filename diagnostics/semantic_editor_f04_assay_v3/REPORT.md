# F04 V3: INCONCLUSIVE fixture; no production release

Six new deterministic tests passed. The one live fixture completed the intended owned cleanup: one actual-worker termination, **natural wrapper completion with no launcher kill**, both exit125, one deadline cause, zero cleanup faults, and all processes/pipes/readers/evidence writers quiescent. The actual retained handle was closed. The fixed three-second episode did not overrun; recorded total cleanup was0.062s. Supervision1.062s; batch1.109s (invoked command1.427s).

The batch nevertheless remains INCONCLUSIVE: **16/17 frozen live assertions passed**. The remaining assertion required the literal outer code `CAPTURE_DEADLINE`; the outer capture recorded `CAPTURE_WORKER_NONZERO_EXIT` while the owned journal correctly retained `deadline`. The unchanged capture loop handles a completed nonzero exit before its separate deadline check, so an owned poll that stops and returns125 can take that branch. Raw execution is still INCONCLUSIVE. No assertion, source or result was changed after the batch, and no rerun occurred.

All50 frozen files verified unchanged.41 files match V2 byte-for-byte, including native.py, exact f04 inputs/token records, editor/gate/scorers, scientific gates and production plan. V2's12 pure tests and V1's full fake matrix/normal worker-audit evidence were authenticated and reused, not rerun. Zero model loads/forwards, tokenizer calls, derivatives, gate scores or fits. Old failures remain final. Production authorization is false.

Source `298587b78f13bb44f88cca819225db7e7ba0bcfc`; freeze `3939dea8b44e2bca2b961182edbf03d7d26dc993fbf8273c7eb3af96f954b60f`.
The future42F16D,300+15+90s,96MiB envelope remains unchanged. Conservative cleanup waits total14s, leaving1s measured overhead headroom—not a guarantee.

One remaining question: can a saved-evidence adjudication reconcile the owned deadline cause and outer nonzero-exit classification while preserving both facts, without another execution? This is engineering evidence only; no scientific or publication success. Stopped for root review.
