# Model-free filesystem evidence component: PASS

The single pre-frozen pure-filesystem batch passed all six case groups in
4.578 seconds (60-second ceiling). No model, tokenizer, dataset, full cohort,
supervisor, or real hook execution occurred. This is component evidence, not
study readiness, steering success, or publication-milestone credit.

## Binding and exact scope

This namespace binds resource commit
`2620f66d4d50456c88800554bc9a26845998cf50`, contract SHA-256
`7610b46b0248569ba51f3f90638734582202c7214822dadafd57fefabc41af3b`,
and resource inventory SHA-256
`26a1bac935e927233d60eb7604aee17088df4cb9bfaecac94a4876ff42aef0c0`.
The resource namespace and historical evidence were not changed.

`binding.py` verifies the pinned contract/helper/inventory and working bytes.
`writer.py` provides serialized single-process/thread-owned filesystem writes,
exclusive creation, validated append, path and case checks, actual-file
reconciliation, sticky failure, and reserved native failure closeout.
`reader.py` independently parses saved evidence and verifies exact lengths,
hashes, paths, counts, physical bytes, and categories without importing the
writer or resource helper. All component settings come from the one contract;
old inherited timing/storage settings are rejected. `verify_io.py` is the
targeted one-shot batch. The five source files, including README.md, were frozen
in SOURCE_FREEZE.json before BATCH_STARTED.json was created and remained equal
to their frozen hashes after the batch.

## Measured results

- A complete 248,320-value little-endian float32 final-token vector used exactly
  993,280 bytes and round-tripped byte-identically, including negative zero.
  Its SHA-256 is
  `d70ce1d50070644c3026eabaf4e48ecbf3657dad4b094cb371c2e64975e79e8c`.
  Independent saved reading returned COMPLETE with no errors, including
  row/event/log append accounting.
- Deliberately corrupted full-vector bytes, a 17-byte incomplete vector, and an
  incorrect closeout-index hash were rejected. Fault fixtures remain saved.
- An injected write failure left 4,096 actual bytes of a 993,280-byte attempted
  vector. The full attempted reservation remained charged, ordinary subsequent
  writes were blocked, and failure closeout succeeded. The independent reader
  returned INCOMPLETE while accounting for every file; closeout used 3,824 bytes
  and left 8,384,784 bytes of its reserved category.
- An injected append failure retained the original prefix plus the partial
  append (12 actual bytes). A log at its 1 MiB stream limit rejected the next
  byte without truncation. Both failures remained INCOMPLETE with saved
  closeouts.
- An unexpected 20-byte external file plus a 3-byte owned row reconciled as 23
  actual pre-closeout bytes, triggered sticky failure, and remained visible to
  the independent reader as INCOMPLETE rather than being silently ignored.
- Unsafe paths, case-colliding overwrite, legacy settings, and foreign-thread
  writes were rejected. The legitimate owner could still complete its separate
  fixture after a foreign-thread rejection. Sealed evidence rejected writes.

TEST_REPORT.json contains the individual receipts and exact closeout hashes.
All partial, corrupt, cap, and external-file fixtures are intentionally retained;
they are I/O failure tests, not scientific/model failures. FINAL_INVENTORY.json
enumerates preparation files and receipts, excluding only itself.

## Boundaries and smallest remaining work

The demonstrated reserve is an enforced logical byte allocation with successful
local failure closeout, not a guarantee against full disks, denied permissions,
power loss, adversarial filesystem races, or hardware failure. Timing is measured
only for this small batch; the 1,800-second worker / 15-second cleanup /
180-second audit values remain allowances, not measured worst-case guarantees.

The next binding work is to use this component in the separately frozen exact
confirmation schema and fake workflow, with complete expected file/row/event
counts and external closeout-hash capture. Whole-cohort resource simulation,
actual hook metadata capacity, supervisor execution, real model metadata,
and real-model outcomes remain unverified. Do not release a real runner based
on this result or alter scientific gates to fit resource limits.
