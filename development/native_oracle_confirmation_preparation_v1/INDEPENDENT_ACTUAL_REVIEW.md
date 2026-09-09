# Independent actual preparation and exact-copy acceptance

Verdict: **PASS for the closed one-shot offline preparation and its exact execution-input bundle.** This is not model execution authority or a behavioral result. Review completed on 2026-09-09; no tokenizer/model replay, provider import, retry or source/input change was performed.

## Exact evidence

| Artifact | SHA256 |
| --- | --- |
| preparation_attempt_001/RESULT.json | 64857e09dd387f7dbfbe1aeecda2badd4115215d2ee3e6b1c8bd2cb4ef2bb8e5 |
| preparation_attempt_001/inputs.json | d8f75731b351f50590c33588fe0ebc42c6dcc59de2669055173a45a1b60bd543 |
| preparation_owner_v1/ownership_attempt_001/CLOSURE.json | f562366237a8b8e9e3aed4ff50646bd89882f9d5324aa66d1bae836c07fe4411 |
| preparation_v1/root_release/TEXT_LOCK.json | bc48bc37375f9b9dbd3c11c17ab099cbb13e8122b66e5982ae2d348727fcc5e5 |
| preparation_v1/root_release/PREPARATION_RELEASE.json | bde6460d34dcf2bc8f5a502f268af16c2b0d30bf1302563a9ab78379163c049a |

Abbreviated paths above refer to the native_oracle_confirmation namespaces under development/. All 32 original files, totaling 183,352 bytes, were independently compared against their exact Git blobs at `9556c0af2d37d9262b2e4805650a66d0f2c93a43`; all matched. Their before/after read-only audit hashes also matched.

The sealed lock and preparation release match their prospective Git bytes at `e437b33b3ca09bd56d5ba4ea3379129942d590d3` and `9ae9cadec5c080593deeffb22a493a0a4f2d39c1`. Integer Python FILETIME comparisons establish both commits predate all three recorded creations: launcher `134334026412262534`, console helper `134334026412478823`, actual Python `134334026412677918`. No JavaScript number conversion was used.

## Original-artifact findings

- Existing production validate_text_lock and input_reader.validate, called directly on the original records without a synthetic flag, pass. Source/manifest/lock/release/canonical-content, 24 case/order/category/gold and full raw boundary/mask/KEEP-STOP-A-B ID joins match. Source identities remain preparation `fe7f0b41a4762572e24ede88fa4ce42dbf1ee200b4f78094e14cc038cc96d78c`, execution `172edd056bfb14a4029c50027c0f8753cbd865ebf76701bb4f21269883a46c95`, owner `3a78ba42ac71f3b3388a27bcc69c8e38af9275a50fa4d4bb526a7f743759e780`.
- All 313 planned operations completed in exact order, with 626 STARTED/COMPLETE journal rows; zero failed or UNRUN operations, all 24 receipts published. Complete lengths range 45–170, within 320, with full masks, no truncation and the frozen generation/content boundaries. Ordinary gold remains A/B/A/B/A/B. Actual preparation records zero model loads, forwards and derivatives.
- The exact approved command, release/text/owner hashes and both admission records agree. Actual PID14988 matches preparation ADMISSION; its and the console helper's parent is launcher PID15804. Recorded live creation/image identities match the pinned launcher/base-Python/conhost paths and current image hashes. Hidden suspended launch, assignment before resume, owned-job membership, creation identity retained before exit and no PID-only authority are recorded by the unchanged reviewed owner.
- All three retained-handle exit proofs have successful query/wait, signaled state and exit code 0. No timeout, termination, primary/cleanup error or capture overflow occurred. Job was empty before close; both drains reached EOF and joined; both pipes closed. Captured stdout is exactly the RESULT projection without lengths, stderr is empty, and byte counts agree.
- Preparation elapsed 27.812 seconds and lies inside the owner's 29.438-second interval. Owner wait/absolute deadlines are exactly 175/180 seconds after admission; cleanup records 0 seconds and stays inside the shared five-second bound. Preparation uses 178,614 bytes and owner 4,738 bytes, independently within 16,744,448/32,768-byte partitions and the combined 16 MiB cap. All file and RESULT/CLOSURE subcaps hold; pre-closure plus closure bytes reconcile exactly.

## Exact execution-bundle handoff

After originals PASS, independently compared all 29 files at `development/native_oracle_confirmation_execution_v1/root_release`: original TEXT_LOCK, unchanged actual CLOSURE renamed PREPARATION_CLOSURE, and the 27 exact preparation payloads excluding ADMISSION. The complete set is byte-identical, 313,003 bytes. Canonical SHA256 of the path-sorted `{path,bytes,sha256}` inventory is `5b237d32c2c146a5889ccb3db61cac8124d2db9d873978be9daa4dc1c0564f84`.

Production input_reader.read_bundle passes on these copies with its normal production scope, using only in-memory metadata marked approved=false. No synthetic flag, fabricated closure, reconstructed payload, metadata file or RELEASE.json was supplied/created. The copied lock's whole cohort still equals the admitted unchanged v2 submission. Original audit was not rerun to validate the copies.

Both independent checks used `.venv/Scripts/python.exe -B -I -c`, PowerShell login:false, and closed exit 0. The first used read-only Git blob/commit checks then blocked children/network/providers/writes; the second was read-only throughout. No broad suite, actual tokenizer/factory, model/provider or execution was invoked. Only this report was written, outside frozen/attempt inventories. Root must separately approve/commit/preflight any later real execution; O03's disclosed repeat, absent-opportunity limits and all prior failed/rejected attempts remain unchanged.
