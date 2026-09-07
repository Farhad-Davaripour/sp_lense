# Confirmation filesystem component — model-free, not a study release

This NEW namespace binds resource commit
`2620f66d4d50456c88800554bc9a26845998cf50`, contract SHA256
`7610b46b0248569ba51f3f90638734582202c7214822dadafd57fefabc41af3b`,
helper SHA256 `538d80e9cde0f06f0d02ba83038f624f9310301a1a0cf30d8d6049efce4af72e`
and inventory SHA256
`26a1bac935e927233d60eb7604aee17088df4cb9bfaecac94a4876ff42aef0c0`.
The writer checks committed AND working bytes before importing the old model-free
helper. It does not change that namespace. The saved reader independently fetches
and authenticates the committed contract and imports neither writer nor helper.

`writer.py` owns an initially empty evidence root in one process and thread;
an RLock serializes operations. It requires complete serialized bytes, reserves
the attempt through the pinned contract, uses exclusive creation or owned append,
handles short successful OS writes, fsyncs and checks the resulting complete
bytes. Paths are contained, canonically case-folded, reject Windows device/alias
names and traversal, and reject observed links/reparse points. Existing records
are reconciled by length AND SHA256 before ordinary writes, including append logs.

The new raw codec is exactly 248,320 little-endian float32 values, 993,280 bytes,
without headers or compression. Writer admission checks finite values and full
length. The independent reader checks full byte length and expected SHA256,
decodes with its own explicit `<248320f` recipe, and rejects nonfinite values.
Historical compressed files, scientific thresholds and model state are untouched.

After a write fails, the full attempted reservation remains consumed, actual
partial bytes remain on disk, and a sticky failure blocks all ordinary writes.
Reconciliation distinguishes expected complete length/hash from actual retained
length/hash. A reserved native closeout writes an INCOMPLETE index rather than
claiming success, retrying the evidence write, truncating or deleting the prefix.
Unexpected external files or modifications become visible failures and count in
actual storage. The independent index reader checks every file (including logs),
membership, actual totals, category/file/row/vector/ledger/log bounds and the
outer supplied index hash. The index excludes itself explicitly; its length is
independently added to the closeout category and actual total.

One component-settings mapping comes from the pinned contract: 180F/48D, one
load,160 tokens,288MiB/5MiB files,1800/15/180 seconds. The constructor rejects
inconsistent recorder/worker/audit/judge settings. This component does not run
those supervisors or establish that their implementation consumes the settings.
Test fixtures also enforce a 32MiB preparation-namespace ceiling.

## Frozen one-batch verification

Read `SOURCE_FREEZE.json` and `TEST_REPORT.json` for actual source binding and
results. There is ONE <=60-second pure filesystem batch, invoked with
`python -B diagnostics/semantic_confirmation_io_v1/verify_io.py`; reruns are
blocked by `BATCH_STARTED.json`. It uses two complete vectors and bounded rows,
events and logs, not a 180-vector cohort. Large fixtures are not synthesized.

Fault fixtures deliberately retain: one corrupted full vector after sealing,
one short vector, one partially written vector, one partially appended log,
an append-limit rejection, an unexpected external file, and rejected paths/case
aliases. These deliberate NEW fixture mutations test reader detection; the writer
never silently overwrites or truncates evidence. The partial-write injector writes
a prefix through the real OS then fails the next ordinary write call. Closeout
uses native OS writes. All fault indexes and test receipts remain available;
fixture failure is not scientific/model failure.

## Explicit remaining limits

This tests a single filesystem component, not complete scientific schemas, a
whole cohort or model behavior. Complete 180F/48D cohort simulation, actual pinned
model hook metadata capacity, all source/input/schema bindings, supervisor
execution and real model outcomes remain unverified. No tokenizer, datasets,
scenario authoring, old suites, other agents or model calls are used.

The reserve is a logical resource allocation, not a guarantee that a permanently
failed disk will accept closeout. Fsync and read-back are exercised; power-loss
durability is not established. Path/link checks and pre/post reconciliation do
not eliminate adversarial concurrent filesystem races; future ownership must
exclude out-of-band writers. An external file that already exhausts actual disk
or contract capacity can prevent closeout; it cannot authorize deleting evidence
or claiming a complete run. These cases must remain technical failures.

Next binding work is the exact-cohort runner/judge schema and targeted full-cohort
fake workflow, preserving this contract, actual-file reconciliation and honest
failure accounting. There is no real release or automatic publication credit.
