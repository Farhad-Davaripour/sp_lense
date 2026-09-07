# Confirmation resource contract — model-free, no execution authority

This isolated namespace supplies a resource contract and pure boundary checks
for the proposed 24-prompt, 48-request confirmation study. It does not import,
integrate or release a real runner. Prior attempts, scientific thresholds,
model/gate weights, renderer, editing math and hook identity rules are untouched.

## Exact allocation

| Allocation | MiB | Basis |
|---|---:|---|
| Full final-token logits | 171 | 180 × 248,320 × 4 = 178,790,400 bytes (170.5078125 MiB) |
| Row JSON | 45 | Existing 262,144-byte row limit × 180 = 47,185,920 bytes |
| Counted ledgers | 4 | All declared stream maxima jointly consume 4,063,232 bytes; 131,072 bytes spare |
| Hook metadata and check records | 16 | Retains existing aggregate cap, including its 65,536-byte fault reserve |
| Sources and inputs | 32 | Complete bundle must be measured and admitted before model access |
| Worker/audit stdout and stderr | 4 | Four explicit append streams, each capped at 1 MiB |
| Saved audit and inventories | 8 | Complete artifacts admitted; no repeated full-logit arrays in summary JSON |
| Failure and final closeout | 8 | Separate category unavailable to ordinary evidence |
| **Total** | **288** | Noncloseout allocations total 280 MiB |

The per-file ceiling remains 5 MiB. Contract/category/event limits reject whole
writes before reservation; they never truncate bytes, overwrite existing
evidence, silently borrow space or extend an allowance. The virtual ledger is
a single-owner accounting primitive, not an integrated filesystem guard. A
binding must serialize reservations and reconcile actual complete/partial
files, copied artifacts, all writers and final inventory sizes.

Ledger maxima cover 360 forward start/terminal events, 96 derivative events,
96 optional-slot skips, 72 routes, 12 self request records, 36 OFF returns,
12 ON-finally cleanup events, 32 scientific findings, and two final summaries.
The 32 findings cover up to 24 wrong baseline routes plus six finite self
baseline admission failures before the inherited stop. The existing scientific
stop remains authoritative; this is a storage allowance, not permission to
continue after a failure. Record caps include complete UTF-8 payload/newline.
New binding schemas and source identifiers must demonstrably fit these caps;
an oversized record is a technical recording failure, never truncated evidence.
Technical/secondary faults and final UNRUN/receipts use reserved closeout space.

## Explicit codec change for NEW evidence only

`raw_f32_le_final_token_v1` stores exactly one 993,280-byte `.f32` file per
forward: the complete final-token vocabulary vector in little-endian IEEE-754
binary32 order, with no header or compression. SHA256 covers all raw bytes.
The new independent reader must require this exact length, finite values and
the matching hash. The pure test covers byte-identical encode/decode round trip,
negative zero, subnormal/extreme finite values, wrong length and nonfinite input.
Old `.f32.zlib` evidence is neither rewritten nor reinterpreted. There is no
change to model outputs, numerical acceptance or identity comparisons.

Known source basis: the existing
`diagnostics/semantic_editor_f05_assay_v2/editor.py:63` selects `result[0,-1]`,
requires 248,320 entries and serializes `<f4`; the corresponding inherited
editor row writer has a 262,144-byte bound (f03 selected-C copy, line 464).
The pinned Qwen revision is `2fc06364715b967f1860aea9cf38778875588b17`, whose text
config declares vocabulary 248,320 and hidden width 1,024. No config, model or
tokenizer is loaded by this resource test.

## Hook storage retains existing evidence semantics

The existing hook recorder saves module/callback/registry metadata, not
token-by-vocabulary tensors or numerical activation arrays. It records setup
before/reference/change objects once as compact JSON in independently compressed
1 MiB raw chunks; clean checks append a reference hash and empty changes.
The retained per-snapshot raw cap is 64 MiB. This contract checks *actual complete
encoded chunk lengths* against both 5 MiB/file and the 16 MiB aggregate allowance,
so it does not need to assert a zlib compression ratio or bound. Manifest and
setup JSON each have a 64 KiB envelope; 109 clean check records of at most 4 KiB
require at most 446,464 bytes. A failed aggregate reservation commits no partial
chunk reservations and leaves the hook fault reserve available.

These enforced bounds do NOT prove that a complete real model setup/difference
snapshot will fit. The later binding must verify full metadata capacity without
weakening, dropping or normalizing hook identity evidence. If it cannot fit,
the result is a recording insufficiency; no automatic allowance expansion or
partial-evidence pass is allowed.

## One shared runtime envelope, still unverified timing

Every recorder, worker supervisor, audit supervisor and independent saved judge
must consume `component_settings()` from this exact contract. The equality
validator rejects inherited 92/96 MiB or 300/90-second settings. Worker 1,800s,
cleanup 15s and saved audit 180s are hard proposed allowances, not measured
worst-case completion guarantees. Count-before-dispatch limits are one load,
180 forward attempts, 48 derivative attempts and all 24 encoded lengths ≤160.
Failed attempts consume their allowance and cannot be retried.

## Verification and smallest remaining binding work

There is exactly one pure batch, invoked as
`python -B diagnostics/semantic_confirmation_resource_v1/verify_resource_contract.py`.
It refuses reruns after `BATCH_STARTED.json`, hashes its source files, reports
elapsed time against 60s, and writes only small receipts under a 32 MiB namespace
ceiling. Large storage boundaries use integer reservations, not large files.
Read `TEST_REPORT.json` for the actual outcome; no result is claimed before it.

Remaining work is one exact-cohort binding: new raw writer/independent reader;
all recorder/supervisor/counter settings taken from this contract; complete
source/input/schema/log accounting; verified hook metadata capacity and
failure-closeout reconciliation. Verify that binding with its targeted fake
workflow before considering real release. This resource contract neither proves
real execution will complete nor earns a study or publication milestone.
