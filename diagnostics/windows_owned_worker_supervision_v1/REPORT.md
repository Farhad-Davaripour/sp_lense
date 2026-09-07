# Owned-worker fake verification: NOT PASSED

The 12 pure fixtures passed, but the one live scenario did not demonstrate the required retained-handle worker termination. Do not accept the raw PASS flags: the test assigned PASS before its final exit-code assertion, then preserved the assertion error without resetting that flag. ADJUDICATION.json is the authoritative closeout; original source and raw receipts remain unchanged.

Live observations: owned venv launcher PID27896 and actual Python PID2420 were authenticated with matching nonce, OS parent/image hashes/creation identity and a termination-capable actual-worker handle BEFORE FAKE_WORK_ONLY acknowledgement. The worker closed output. At the fixed deadline, the owned launcher was terminated (exit125). The first snapshot correctly recorded stdout EOF, joined readers/closed pipes, launcher dead but worker alive, and quiescence=false. Before the next cleanup check the worker exited0 by an unestablished mechanism. There was no terminate_actual_retained_handle event. The prospective required worker-exit125 assertion therefore failed. This does not prove the actual-handle termination path; no second live scenario was run.

All pure rejection fixtures sent zero acknowledgements and targeted zero invalid claimants. Wrapper/direct valid cases and fake cleanup-failure accounting passed. The standalone adapter retains its authenticated handle and does not reopen a PID at cleanup; production run_capture wiring remains only the explicit proposal in PRODUCTION_WIRING.md. It is not ready for production approval based on this packet.

Both live processes exited, all reader threads joined and pipes closed; no cleanup errors remained. One live scenario took1.265s; complete fake batch1.406s (invoked command1.9739025s). Zero model, tokenizer, gate, data or scientific-runtime calls/edits. The pre-freeze base-executable mismatch/pause and root's scoped approval of b7a12c3a... are preserved. This does not assert binary equivalence or complete scientific-environment verification.

Source commit177b8b40fd72849872abf452f2874c0eb51809d9; freezebb3d949adcc84fe56658456c5dbd6182d23165fd2edc4908b2b0a02d02a55878. Frozen source was not repaired after observing this failure. Historical INCONCLUSIVE and independent f08 margin applicability FAIL remain final.

Next single step: independently review this failed fixture and its stale-verdict bug before authorizing any successor test or production wiring. No retry, model run or old-result change is authorized.
