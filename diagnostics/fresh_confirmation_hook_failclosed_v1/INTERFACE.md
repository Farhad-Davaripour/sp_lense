# Production adapter interface

Import only the pinned hook_evidence.py. The component is pure stdlib and does not construct/import a model, tokenizer or historical backend.

1. Instantiate DispatchLatch(exact_schedule_ids), at most 180 unique printable-ASCII IDs of at most 256 characters. Share this same latch with every model-forward/derivative guard. No model dispatch while PENDING, COMPLETE or TERMINAL. The adapter's claim path calls consume(next_id) exactly once for DONE attempts or explicit SKIPPED slots; never consume unperformed UNRUN slots.
2. Construct Recorder(output_root, latch, exact_109_check_labels, io=owned_io). The component requires exactly 109 unique labels. A four-check adapter prefix is intentionally incomplete, not a substitute production certificate. Recorder exposes io, checks, check_attempts and latch.
3. The adapter independently authenticates installed source hashes before known forward-free setup, takes complete before/reference/setup-change snapshots through the unchanged strict guard, then calls admit(before, reference, setup_changes, source_lock, setup_receipt). source_lock must declare passed:true and before_materialization:true; setup_receipt.forward_calls must be 0. These flags are caller attestations, not a replacement for its actual source/forward checks. setup_changes.changes must equal exact differences(before,reference). The return is the bounded setup receipt. Failure prevents first forward.
4. inspect(lambda: strict_guard.inspect(model), exact_label) requires current/matches/changes fields, recomputes the difference independently, saves a clean bounded check or permanently stops before serializing complete fault details. It returns True only for a complete clean check.
5. fail(finite_code, original=None) permanently stops and attempts one finite receipt, then raises HookStopped. HookStopped.code is one of CODES. DispatchLatch.trip(code), require_dispatch(), consume(id), terminal, remaining and primary_code are exposed. If a guard trips outside Recorder, call recorder.fail(code, original) to attempt durable fault evidence without clearing the shared latch. Never stringify arbitrary exceptions. Original primary_exception is in-memory only.
6. finish() succeeds only after all exact checks and schedule consumption. It writes/authenticates the sole hook index and returns hook_status COMPLETE plus index_sha256. status() supplies a finite controller status, including latch_state, index_sha256, primary/secondary codes, receipt_failed, remaining_ids and permits_pass. A full index write whose IO acknowledgement fails leaves index_sha256 unavailable and terminal status; do not synthesize status from disk.
7. judge(root, expected_labels, expected_schedule, expected_source_lock, controller_status, io=None) independently reconstructs all indexed bytes and checks. controller_status is mandatory and must be bound to an authenticated, fully closed controller/external capture by the production final judge. It is necessary but never sufficient. Missing/malformed/terminal status or any changed/missing/incomplete/unreferenced bytes rejects. Its COMPLETE permits only the hook prerequisite, never a scientific PASS. Join complete-valid/quiescent worker and audit captures, all independent scientific gates, and absence of controller/receipt faults in authoritative closeout.

## Injected IO ownership

Production supplies the SAME single EvidenceWriter owner through an adapter:

- write(relative_name, data:bytes, append=False, fault=False): exclusive write or append, durable acknowledgement; preserve partial bytes on failure.
- read(relative_name)->bytes.
- hook_files()->dict of ALL hook-relative names and actual byte sizes, including untracked or partial files.
- total_bytes()->actual aggregate bytes, including other categories and partial/external outputs.

Names are under hook_evidence/ only; no independent untracked writer is permitted. Delegate normal chunks/manifests/index/check stream to the writer hooks category; fault=True consumes only the reserved finite fault/closeout allowance and remains included in component hook+aggregate accounting. The adapter must preserve the existing whole-vector, shared-cap, per-file and closeout ownership rules, not duplicate accounting or omit hook files from saved_reader. Any adapter exception/failure is terminal. The component reserves future hook bytes against actual aggregate usage too; the global writer remains responsible for other categories' prospective allocations.

The complete normal final index has a separate 64 KiB reservation inside the unchanged hook allowance. Setup admission tests its actual prospective size using the exact IDs/labels/chunk inventory. A one-time fitting failure dump may use released future check/index room but may not consume the untouched 64 KiB fault receipt reserve. No arbitrary diagnostics are promised to fit.

There is deliberately no reset, cap override, shortened-production-check setting, retry, truncation or omission-as-completeness API.
