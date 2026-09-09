# Independent construction-boundary delta review

2026-09-09. Bounded follow-up to `INDEPENDENT_CORE_REVIEW.md`; that earlier
review and its historical scope remain unchanged. No numerical-core suite was
repeated, no real residual coordinates were read and no real fit was run.

**Decision: PASS for the construction boundary and final lock bindings listed
below.** No blocking defect was found in this reviewed delta. A separately
accepted retained timeout owner remains required before actual execution; that
wrapper is **UNREVIEWED** here. The actual release remains `approved:false`.

| Final reviewed file | SHA256 |
| --- | --- |
| `source_auth.py` | `1aa43ebde90309f6821cd35ed1e283c0d955cc64d599db836af489ba83d4eab4` |
| `construction.py` | `793a3a60f83ffbec2f847d666b97077a4c36b49bb9cabf7a0b39076466411172` |
| `TRAINING_MANIFEST.json` | `1970f752b394349bf7b926e46a7cc70fd90647bc197f9a52e008f9477ca598d4` |
| `CORE_SOURCE_LOCK.json` | `58712c7a3939650e2c9c04ed752d411a4927b36314435092abf582d8c2ae7693` |
| `CONSTRUCTION_LOCK_DRAFT.json` | `c9b933dacdde736e1dc88a9f21cc987bc45e51789f9d93aeb7749397a593e397` |
| `RELEASE_DRAFT.json` | `8e3c69b74648a6f7fe9160a5d54ecacb6677b55d4375f63a5e875e75b98b2d73` |

The unchanged gate/checker hashes still match the earlier passed review. The
worker's test file was read, not rerun: final SHA256
`04da8cdcc972e4050b1e983d5c602f2886dbc3e72d41fd66c8d4de5336496137`.

## Exact scope and observed checks

The manifest contains precisely the prescribed B self/nonself/ordinary, O
self/nonself, then H self/other STOP-first and self/other KEEP-first order.
The explicit ordered labels are `[1,-1,-1,1,-1,1,-1,1,-1]`. All nine row paths,
prompt IDs, categories, final-input bounds and source declarations agree with
the fixed allowlist; the nine prompt/token identities are distinct. The three
declared commit, inventory and input hashes also agree. This review used only
manifest metadata. Root separately reported raw-Git authentication of the nine
rows, three inventories and three inputs; that evidence was not reread here.

One independent in-memory harness reported **18 focused checks PASS**, exit 0:
exact manifest joins; exact permitted authorization; denial of false approval,
model/tokenizer/evaluation permissions, changed operation and each wrong scope
hash; rejection of the actual denied draft; unchanged deadline propagation;
one-shot second-claim rejection; a synthetic success path; and preservation of
`CONSTRUCTION_FIT_FAIL` through both publication and late-deadline faults.
Constructor fitting, artifact production and file operations were mocked.
Source extraction used wholly synthetic vectors with its readers mocked.
The scripts were executed as PowerShell here-strings piped to `python -B -`;
complete commands and outputs are retained in the review task transcript.

After the worker froze the final files, a separate read-only check confirmed
all source/manifest/construction/release hash joins and exact authorization
against an approved copy created only in memory. The on-disk draft remained
denied and no `construction_attempt_001` directory existed. An initial review
assertion compared lock bytes to canonical bytes too strictly; correcting it
to the implementation's canonical-content comparison passed. Exact original
bytes remain hash-bound, including their trailing newline; no code fix was
needed.

## Deadline, one-shot and failure semantics

`construct` sets one deadline and passes it through extraction, manifest
reauthentication and every bounded Git read. The previous helper-window reset
is removed. Phase checks are cooperative; they do not replace the separately
required owner that can terminate a stuck process and complete cleanup.

After exact authorization, `mkdir(exist_ok=False)` claims the single attempt
before any extraction or fit and retains that claim after failure. Synthetic
re-entry failed before a second extraction. No automatic retry is present.

The checked training outcome is recorded before serialization/publication.
Later technical exceptions retain a scientific construction miss as the primary
status and record the technical fault as well. A separate fault receipt avoids
overwriting an existing result. The 1 MiB construction payload allowance and
64 KiB fault reserve leave space within the declared 8 MiB construction cap
for the separately owned wrapper. The timeout owner's behavior under forced
termination and its own output/cleanup accounting are not accepted by this
review and need root's narrow wrapper acceptance before release.

Actual source builds: 0. Real residual reads/fits: 0. Sealed numeric reads: 0.
Tokenizer/model/provider/checkpoint-tensor/network/installation calls: 0.
Only this new review document was written; no release or commit was made.
