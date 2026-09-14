# Snapshot verifier supervisor repair V2

The V1 independent FAIL is retained unchanged. Codex repaired the code directly; DeepSeek is assigned one independent re-review, not an implementation repair loop. Pre-repair source and tests are preserved in work/pre_repair_0535_snapshot_verifier.py and work/pre_repair_0535_test_snapshot_verifier.py.

## Changes and scope

- F1: tokenizer identity selection now allows only recognized tokenizer filenames and requires tokenizer.json or both vocab.json and merges.txt. A weights-only or configuration-only identity is rejected before filesystem access. This validates role names, not tokenizer semantics or origin. No tokenizer is loaded.
- F2: mutable lock input is converted to immutable bytes once; the digest, parser and receipt use that same snapshot. Regression uses a changing bytearray subclass.
- F3/F4: Windows alternate streams, trailing dots/spaces, reserved device basenames and forbidden characters are rejected in manifest basenames and snapshot path components before access.
- F5: integer-to-float deadline overflow is a structured LIMIT_INVALID error. Nonfinite absolute deadlines are rejected as well.
- F6: hardlinks remain explicitly allowed. Root confinement is about resolved paths, not every other name for the same storage. A fabricated hardlink test documents this limitation; no claim of physical-storage confinement is made.
- F7: every file is realpath-resolved and checked in-root; checks repeat before open and before/after hashing. Descriptor fstat metadata must match the prechecked path, remain stable across reading, and match the final path stat. This narrows TOCTOU races but does not eliminate them, is not an atomic sandbox, and does not use a portable no-follow open primitive. SHA256 must still match the caller-pinned bytes.
- F8: renamed the parent-traversal test to describe the condition it actually exercises. Simulated snapshot-directory containment remains covered.

Only temporary fabricated fixtures were accessed by the tests. No actual snapshot, model, tokenizer, dataset, capture or fit was opened/executed. Passing byte verification never authorizes scientific execution. External expected-lock pin selection and a source-locked finite scientific preflight remain required. The receipt's job_id identifies this implementation revision, not a dynamically authenticated run identity.

## Supervisor verification

Command from repository root: development/classifier_generalization_v2/.runtime/Scripts/python.exe -W error development/classifier_generalization_v2/test_snapshot_verifier.py -v

39 tests total: 37 passed, 2 skipped, 0 failures/errors. The two real-symlink creation tests skip on this Windows host; simulated link escape checks and the real hardlink fixture pass. Tests remain synthetic software checks, not study performance.

Native SHA256:

- snapshot_verifier.py: 710C16F39B2FB7AA818D5B6EE5D33632555E690D745997FA0FB0F51AD134875A
- test_snapshot_verifier.py: 1727301ECE84D411E091B535F57BAD6DBC5F28B61F6C51B8496E869D404BDDD9

Independent V2 verdict is pending; V1 PASS claims are not reused.
