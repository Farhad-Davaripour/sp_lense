# Resource-contract preparation — PASS (model-free only)

One targeted pure batch passed all six cases in 0.063 seconds, below the
60-second ceiling. Source hashes remained unchanged during the batch. No model,
tokenizer, dataset, old-suite or live-process work occurred; large capacity
boundaries were integer reservations rather than filesystem allocations.

The single contract allocates 171 MiB raw full-vocabulary final-token logits,
45 MiB row JSON, 4 MiB counted ledgers, 16 MiB hook evidence, 32 MiB sources and
inputs, 4 MiB worker/audit logs, 8 MiB saved audit/inventories and a separate
8 MiB failure-closeout reserve: exactly 288 MiB, with a 5 MiB per-file ceiling.
Existing numerical evidence, scientific gates and historical attempts are unchanged.

The new-only logit codec is fixed-length little-endian float32 without
compression: 993,280 bytes/vector and 178,790,400 bytes for all 180 forwards.
Its complete 248,320-value test round trip preserved bytes, including negative
zero and finite extreme/subnormal values. Wrong sizes and nonfinite data were
rejected. Historical `.zlib` files were neither read nor changed by the batch.

Known source row caps total 47,185,920 bytes (45 MiB). All declared ledger maxima
jointly require 4,063,232 bytes, below their 4 MiB allocation. The 109 hook check
records have a 446,464-byte cap; full setup/failure metadata must still be admitted
under the retained 16 MiB allowance. Atomic chunk/manifest rejections do not trim
evidence or consume the reserved hook-fault space. The allocation test filled all
280 MiB of ordinary categories and then the entire 8 MiB closeout reserve, rejected
overflow, and created no large files.

Recorder, worker supervisor, audit supervisor and saved-judge settings share the
same contract. The boundary tests reject inherited 92/96 MiB limits and 300/90s
settings. Proposed worker/cleanup/audit allowances are 1800/15/180 seconds; they
are not measured worst-case completion guarantees. Attempt ceilings are
180 forwards, 48 derivatives and one load; all 24 encoded lengths must be ≤160.

Exact changes are confined to this new namespace: contract JSON, model-free
codec/counter/size-ledger helpers, one pure test script, source-bound test receipts,
documentation, this summary and its final file inventory. No old recorder or
scientific source is patched. Preparation files are far below the 32 MiB ceiling.

Smallest remaining work: bind one new exact-cohort writer/independent reader and
all supervisors/counters to this contract; verify complete source/input/record
schemas, actual hook metadata capacity, serialized ownership, partial-write
reconciliation and final closeout with the targeted fake workflow. This PASS
establishes resource-contract boundaries, not that every real artifact will fit
or that a real run will finish. Execution authorization remains FALSE; no study
or publication milestone is awarded.
