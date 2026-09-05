# Three-family recording storage preparation

Status: **MODEL_FREE_PREPARATION_CERTIFIED**, limited to the pinned local Windows environment and cooperative, authenticated recording writers. This resolves the storage blocker recorded at `2802a0deb45a42a7e07cbc4871471b3aba9a5a33`; it is not a model result or permission to start construction. Publication readiness remains 40%.

The earlier negative certificate, `shared_comply_crossed_three_family_v1_preparation.json`, is preserved byte-for-byte (SHA-256 `eddd059856773b29971d8eefa9ac907bc56d4825ec986c1317732d103f74e9b0`). The new machine-readable certificate is [three_family_recording_storage_certificate.json](three_family_recording_storage_certificate.json). It binds the exact tested implementation, policy and test bytes. The source commit and subsequent preregistration bind this report and certificate without a recursive self-hash.

## What changed

Only the recording envelope changed: scoped artifact writes, bounded merged binary stdout/stderr capture, bounded exception diagnostics, synchronized byte accounting, reserved finalization/failure space, and a final inventory. The numerical construction, selected three discovery families and 12 crossed renderings, model/hook/precision, solver, acceptance criteria, and prospective 216-forward / 96-derivative / 1,200-second limits are unchanged. There was no layer search, gate training, data expansion, model/tokenizer load, or real forward/derivative call.

Normal writers cannot consume the 5 MiB final-artifact reserve or 1 MiB receipt reserve. Every admitted write checks its category and the whole namespace under synchronization; partial writes and simultaneous temporary/destination files count toward peak usage. Replacement does not hide overlap. Unknown artifacts and unsupported mutation routes fail closed. A fixed 4,096-byte state record and one-byte Windows lock coordinate participating processes; faults remain sticky through finalization.

The parent drains one merged binary pipe in 65,536-byte chunks, retaining at most 4 MiB. At most one extra chunk is read when detecting overflow; admitted disk-cap overshoot is zero. Overflow stops the worker, requires bounded termination/join checks and records INCONCLUSIVE. The retained prefix has its own actual size and digest. Full-output length/digest is claimed only when EOF was observed; an unread tail is explicitly unknown. Exception diagnostics likewise identify prefixes, truncation, and which original lengths or digests are actually known.

## Complete-attempt bounds

The policy keeps the original 512 MiB evidence ceiling and 16 MiB auxiliary allowance; it does not increase either quota.

| Artifact group | Maximum file-content bytes |
| --- | ---: |
| Logits (216 bounded arrays) | 214,616,520 |
| Rows | 226,492,416 |
| Updates | 67,108,864 |
| Worker log | 4,194,304 |
| Metadata | 2,097,152 |
| Events | 1,048,576 |
| Endpoint | 1,048,576 |
| Reserved final artifacts | 5,242,880 |
| Reserved receipts | 1,048,576 |
| Control records | 1,048,576 |
| Scratch/registered temporary files | 1,048,576 |
| Combined category maximum | 524,995,016 |
| Evidence ceiling | 536,870,912 |

The eight auxiliary groups total exactly 16,777,216 bytes. The combined maximum leaves 11,875,896 bytes below the evidence ceiling. Temporary files are charged to their registered category, including replacement overlap; the scratch allowance is not an exemption. A 1,073,741,824-byte free-space guard remains mandatory before any loading.

Finalization starts only after confirmed writer/process quiescence. The complete final inventory includes every present registered file and its own actual serialized size; its own hash is null to avoid recursion, while all other files are hashed. A late write, sticky recording fault, incomplete capture, post-seal hashing/write failure, or inconsistent candidate/audit disposition cannot produce a valid candidate seal. Read-only verification checks sizes, hashes, categories and state without recreating missing controls. Provisional candidate files alone are not valid evidence.

## Targeted verification

Across the bounded implementation/test batch, **106 distinct tests passed and one was skipped**. The joint run passed 104 tests; the subsequent exact-source certificate-binding test passed, and the final two-test read-only disposition check passed (one new test and one repeat). Conservative aggregate test execution was **151.054 seconds of the authorized 900 seconds**, including failed development runs and delegated test execution. Ruff passed for the scoped Python changes.

Coverage includes verbose/binary capture, caps and read overshoot, oversized exception text, partial/failed writes, thread/process contention, temporary replacement overlap, deadlines and termination failure, unjoined writers, seal/fault races, late files, incomplete inventory writes, inventory self-size, no-repair attachment races, stale source certificates and false-success prevention. A local Python helper verified eight raw pipe bytes; fake tensors exercised 48 fake forwards and 12 fake derivatives. Neither is a real model experiment. The Windows symlink-creation case was skipped because privileges were unavailable; no permission repair was attempted. An existing pytest-cache permission warning was left unchanged.

The independent storage review found no remaining blocker within the stated scope. The original four-case solver benchmark is reused without rerunning: 16,384 completed masks in 13.82799999974668 seconds, as preserved in the historical certificate. No broad historical suite, benchmark retry, model load or real construction was run for this recording repair.

## Residual limits and handoff

This is cooperative recording, not an operating-system sandbox against arbitrary external writers, descriptor theft or filesystem mutation. Windows locking was exercised; POSIX kernel behavior was not. Quotas count namespace file-content lengths, not RAM, OS pipe/cache buffers, filesystem allocation metadata or external sync copies.

Already-admitted bounded writes may finish during fault handling; their bytes were reserved, and valid sealing still requires quiescence. Failed/hung filesystem I/O or unconfirmed termination yields INCONCLUSIVE and no valid seal. Failure-receipt persistence is then explicitly best-effort, not guaranteed complete evidence. Log prefixes are not represented as complete output or scientific success.

A separate preregistration-only lock and zero-model preflight may follow this source commit. Actual construction still requires separate supervisor authorization after review. No authorization variable is set by preparation, freeze or preflight; the handoff must stop before any real run.
